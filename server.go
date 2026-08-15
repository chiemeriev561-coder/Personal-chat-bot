package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/sashabaranov/go-openai"

	"personalchatbot/provider"
)

// Minimal OpenAI-compatible request types (subset)
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatCompletionRequest struct {
	Model            string        `json:"model"`
	Messages         []ChatMessage `json:"messages"`
	Temperature      float32       `json:"temperature,omitempty"`
	TopP             float32       `json:"top_p,omitempty"`
	MaxTokens        int           `json:"max_tokens,omitempty"`
	N                int           `json:"n,omitempty"`
	Stop             []string      `json:"stop,omitempty"`
	PresencePenalty  float32       `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32       `json:"frequency_penalty,omitempty"`
	Stream           bool          `json:"stream,omitempty"`
}

func writeSSEData(w http.ResponseWriter, value string) {
	encoded, _ := json.Marshal(value)
	fmt.Fprintf(w, "data: %s\n\n", encoded)
}

// default model
var defaultModel = "nvidia/nemotron-3.5-lightning-30b-a3b"

// simple in-memory (and on-disk) history store
type HistoryStore struct {
	mu   sync.Mutex
	data map[string][]ChatMessage
	file string
}

func NewHistoryStore(file string) *HistoryStore {
	hs := &HistoryStore{data: make(map[string][]ChatMessage), file: file}
	if b, err := os.ReadFile(file); err == nil {
		_ = json.Unmarshal(b, &hs.data)
	}
	return hs
}

func (hs *HistoryStore) Save() error {
	hs.mu.Lock()
	defer hs.mu.Unlock()
	b, err := json.MarshalIndent(hs.data, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(hs.file, b, 0644)
}

func (hs *HistoryStore) Get(session string) []ChatMessage {
	hs.mu.Lock()
	defer hs.mu.Unlock()
	return append([]ChatMessage(nil), hs.data[session]...)
}

func (hs *HistoryStore) Append(session string, msg ChatMessage) error {
	hs.mu.Lock()
	hs.data[session] = append(hs.data[session], msg)
	hs.mu.Unlock()
	return hs.Save()
}

var store = NewHistoryStore("history.json")

func withCORS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		// Allow browser clients, including Lovable and local development.
		if origin != "" {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Add("Vary", "Origin")
		} else {
			w.Header().Set("Access-Control-Allow-Origin", "*")
		}
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Allow-Credentials", "true")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next(w, r)
	}
}

func requireAuth(w http.ResponseWriter, r *http.Request) bool {
	// /health is intentionally public
	if r.URL.Path == "/health" {
		return true
	}
	token := os.Getenv("API_AUTH_TOKEN")
	if token == "" {
		return true
	}
	auth := r.Header.Get("Authorization")
	if auth != "Bearer "+token {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return false
	}
	return true
}

// Start a simple HTTP API exposing the requested endpoints.
func startServer(addr string) {
	if addr == "" {
		addr = ":8080"
	}

	// Provider selection priority: NVIDIA -> Groq -> Gemini
	var prov provider.Provider
	if os.Getenv("NVIDIA_API_KEY") != "" {
		p, err := provider.NewNvidiaProviderFromEnv()
		if err != nil {
			log.Printf("failed to init NVIDIA provider: %v", err)
		} else {
			prov = p
			log.Printf("NVIDIA provider initialized")
		}
	}
	if prov == nil && os.Getenv("GROQ_API_KEY") != "" {
		p, err := provider.NewGroqProviderFromEnv()
		if err != nil {
			log.Printf("failed to init Groq provider: %v", err)
		} else {
			prov = p
			log.Printf("Groq provider initialized")
		}
	}
	if prov == nil && (os.Getenv("GEMINI_API_KEY") != "" || os.Getenv("GOOGLE_API_KEY") != "") {
		p, err := provider.NewGeminiProviderFromEnv()
		if err != nil {
			log.Printf("failed to init Gemini provider: %v", err)
		} else {
			prov = p
			log.Printf("Gemini provider initialized")
		}
	}

	mux := http.NewServeMux()

	// Health is intentionally public. API endpoints require API_AUTH_TOKEN if set.
	mux.HandleFunc("/health", withCORS(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))

	// History endpoints (basic)
	mux.HandleFunc("/v1/history", withCORS(func(w http.ResponseWriter, r *http.Request) {
		if !requireAuth(w, r) {
			return
		}
		switch r.Method {
		case http.MethodGet:
			session := r.URL.Query().Get("session")
			msgs := store.Get(session)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(msgs)
		case http.MethodPost:
			var p struct {
				Session string      `json:"session"`
				Message ChatMessage `json:"message"`
			}
			if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
				http.Error(w, "invalid request body", http.StatusBadRequest)
				return
			}
			_ = store.Append(p.Session, p.Message)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}))

	// OpenAI-compatible (basic) completions endpoint
	mux.HandleFunc("/v1/chat/completions", withCORS(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// auth (except /health)
		if !requireAuth(w, r) {
			return
		}

		var req ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid request body", http.StatusBadRequest)
			return
		}

		// default model
		if req.Model == "" {
			req.Model = defaultModel
		}

		// If stream=true, perform streaming on this endpoint (OpenAI-compatible).
		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")

			flusher, ok := w.(http.Flusher)
			if !ok {
				http.Error(w, "streaming unsupported", http.StatusInternalServerError)
				return
			}

			openReq := openai.ChatCompletionRequest{
				Model:            req.Model,
				Temperature:      req.Temperature,
				TopP:             req.TopP,
				MaxTokens:        req.MaxTokens,
				N:                req.N,
				Stop:             req.Stop,
				PresencePenalty:  req.PresencePenalty,
				FrequencyPenalty: req.FrequencyPenalty,
				Stream:           true,
			}
			for _, m := range req.Messages {
				openReq.Messages = append(openReq.Messages, openai.ChatCompletionMessage{Role: m.Role, Content: m.Content})
			}

			if prov != nil {
				stream, err := prov.CreateChatCompletionStream(context.Background(), openReq)
				if err != nil {
					if err == provider.ErrNotSupported {
						http.Error(w, "provider does not support streaming", http.StatusNotImplemented)
						return
					}
					http.Error(w, "provider stream error: "+err.Error(), http.StatusInternalServerError)
					return
				}
				defer stream.Close()

				for {
					chunk, err := stream.Recv()
					if err != nil {
						if err == io.EOF {
							break
						}
						fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
						flusher.Flush()
						return
					}
					if chunk.Content != "" {
						writeSSEData(w, chunk.Content)
						flusher.Flush()
					}
				}

				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				return
			}

			// fallback demo stream
			chunks := []string{"Starting stream...", "(no provider)"}
			for _, c := range chunks {
				writeSSEData(w, c)
				flusher.Flush()
				time.Sleep(200 * time.Millisecond)
			}
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}

		// Non-streaming path follows: call provider if available, else echo
		if prov != nil {
			openReq := openai.ChatCompletionRequest{
				Model:            req.Model,
				Temperature:      req.Temperature,
				TopP:             req.TopP,
				MaxTokens:        req.MaxTokens,
				N:                req.N,
				Stop:             req.Stop,
				PresencePenalty:  req.PresencePenalty,
				FrequencyPenalty: req.FrequencyPenalty,
			}
			for _, m := range req.Messages {
				openReq.Messages = append(openReq.Messages, openai.ChatCompletionMessage{Role: m.Role, Content: m.Content})
			}

			resp, err := prov.CreateChatCompletion(context.Background(), openReq)
			if err != nil {
				http.Error(w, "provider error: "+err.Error(), http.StatusInternalServerError)
				return
			}

			// Map provider response to OpenAI-compatible response body
			out := map[string]interface{}{
				"id":      resp.ID,
				"object":  resp.Object,
				"created": resp.Created,
				"choices": []map[string]interface{}{},
				"usage":   resp.Usage,
			}
			for i, ch := range resp.Choices {
				choice := map[string]interface{}{
					"index": i,
					"message": map[string]string{
						"role":    ch.Role,
						"content": ch.Content,
					},
					"finish_reason": ch.FinishReason,
				}
				out["choices"] = append(out["choices"].([]map[string]interface{}), choice)
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(out)
			return
		}

		// fallback echo
		userContent := ""
		if len(req.Messages) > 0 {
			userContent = req.Messages[len(req.Messages)-1].Content
		}
		reply := fmt.Sprintf("Echo: %s", userContent)
		resp := map[string]interface{}{
			"id":      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"choices": []map[string]interface{}{{"index": 0, "message": map[string]string{"role": "assistant", "content": reply}, "finish_reason": "stop"}},
			"usage":   map[string]int{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(resp)
	}))

	// SSE streaming endpoint — supports GET (simple) and POST (OpenAI-compatible request body).
	mux.HandleFunc("/v1/chat/stream", withCORS(func(w http.ResponseWriter, r *http.Request) {
		// Allow GET for simple demo; POST to stream with full ChatCompletionRequest in body.
		if !requireAuth(w, r) {
			return
		}
		var reqBody ChatCompletionRequest
		var msg string
		if r.Method == http.MethodGet {
			msg = r.URL.Query().Get("message")
			if msg == "" {
				msg = "This is a demo stream from the Personal Chat Bot API."
			}
		} else if r.Method == http.MethodPost {
			if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
				http.Error(w, "invalid request body", http.StatusBadRequest)
				return
			}
			// prefer last user message as default
			if len(reqBody.Messages) > 0 {
				for i := len(reqBody.Messages) - 1; i >= 0; i-- {
					if reqBody.Messages[i].Role == "user" {
						msg = reqBody.Messages[i].Content
						break
					}
				}
			}
		} else {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}

		// If provider supports streaming, forward a streaming chat completion as SSE.
		if prov != nil {
			// build openai request from either GET-derived msg or POST body
			if reqBody.Model == "" {
				reqBody.Model = defaultModel
			}
			openReq := openai.ChatCompletionRequest{
				Model:  reqBody.Model,
				Stream: true,
			}
			if len(reqBody.Messages) > 0 {
				for _, m := range reqBody.Messages {
					openReq.Messages = append(openReq.Messages, openai.ChatCompletionMessage{Role: m.Role, Content: m.Content})
				}
			} else {
				openReq.Messages = []openai.ChatCompletionMessage{{Role: "user", Content: msg}}
			}

			stream, err := prov.CreateChatCompletionStream(context.Background(), openReq)
			if err != nil {
				if err == provider.ErrNotSupported {
					http.Error(w, "provider does not support streaming", http.StatusNotImplemented)
					return
				}
				http.Error(w, "provider stream error: "+err.Error(), http.StatusInternalServerError)
				return
			}
			defer stream.Close()

			for {
				chunk, err := stream.Recv()
				if err != nil {
					if err == io.EOF {
						break
					}
					fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
					flusher.Flush()
					return
				}
				if chunk.Content != "" {
					writeSSEData(w, chunk.Content)
					flusher.Flush()
				}
			}

			// close marker
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}

		// Fallback demo stream
		chunks := []string{
			"Starting stream...",
			msg,
			"[END]",
		}

		for _, c := range chunks {
			writeSSEData(w, c)
			flusher.Flush()
			// small delay to simulate streaming
			time.Sleep(250 * time.Millisecond)
		}
	}))

	addrToUse := addr
	if env := os.Getenv("API_ADDR"); env != "" {
		addrToUse = env
	}

	log.Printf("Starting HTTP API server on %s", addrToUse)
	if err := http.ListenAndServe(addrToUse, mux); err != nil {
		log.Fatalf("API server failed: %v", err)
	}
}
