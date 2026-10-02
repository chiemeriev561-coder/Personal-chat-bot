package main

import (
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

// Minimal OpenAI-compatible request types
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

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	log.Printf("API error status=%d msg=%s", status, msg)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]interface{}{
			"message": msg,
			"type":    "api_error",
			"code":    status,
		},
	})
}

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
	if r.URL.Path == "/health" {
		return true
	}
	token := os.Getenv("API_AUTH_TOKEN")
	if token == "" {
		return true
	}
	auth := r.Header.Get("Authorization")
	if auth != "Bearer "+token {
		writeJSONError(w, http.StatusUnauthorized, "unauthorized")
		return false
	}
	return true
}

func startServer(addr string, prov provider.Provider) {
	if addr == "" {
		addr = ":8080"
	}

	mux := http.NewServeMux()

	mux.HandleFunc("/health", withCORS(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "ok",
			"model":  prov.ModelName(),
		})
	}))

	mux.HandleFunc("/v1/models", withCORS(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		if !requireAuth(w, r) {
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"object": "list",
			"data": []map[string]interface{}{
				{
					"id":       prov.ModelName(),
					"object":   "model",
					"created":  time.Now().Unix(),
					"owned_by": "llm",
				},
			},
		})
	}))

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
				writeJSONError(w, http.StatusBadRequest, "invalid request body")
				return
			}
			_ = store.Append(p.Session, p.Message)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
		default:
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		}
	}))

	mux.HandleFunc("/v1/chat/completions", withCORS(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		if !requireAuth(w, r) {
			return
		}

		var req ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
			return
		}

		if len(req.Messages) == 0 {
			writeJSONError(w, http.StatusBadRequest, "messages array is required and must not be empty")
			return
		}

		targetModel := req.Model
		if targetModel == "" {
			targetModel = prov.ModelName()
		}

		openReq := openai.ChatCompletionRequest{
			Model:            targetModel,
			Temperature:      req.Temperature,
			TopP:             req.TopP,
			MaxTokens:        req.MaxTokens,
			N:                req.N,
			Stop:             req.Stop,
			PresencePenalty:  req.PresencePenalty,
			FrequencyPenalty: req.FrequencyPenalty,
			Stream:           req.Stream,
		}
		for _, m := range req.Messages {
			openReq.Messages = append(openReq.Messages, openai.ChatCompletionMessage{Role: m.Role, Content: m.Content})
		}

		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")

			flusher, ok := w.(http.Flusher)
			if !ok {
				writeJSONError(w, http.StatusInternalServerError, "streaming unsupported by server")
				return
			}

			stream, err := prov.CreateChatCompletionStream(r.Context(), openReq)
			if err != nil {
				log.Printf("stream error: %v", err)
				fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
				flusher.Flush()
				return
			}
			defer stream.Close()

			chunkID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
			created := time.Now().Unix()

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
					chunkData := map[string]interface{}{
						"id":      chunkID,
						"object":  "chat.completion.chunk",
						"created": created,
						"model":   targetModel,
						"choices": []map[string]interface{}{
							{
								"index": 0,
								"delta": map[string]string{
									"content": chunk.Content,
								},
								"finish_reason": nil,
							},
						},
					}
					encoded, _ := json.Marshal(chunkData)
					fmt.Fprintf(w, "data: %s\n\n", encoded)
					flusher.Flush()
				}
			}

			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}

		resp, err := prov.CreateChatCompletion(r.Context(), openReq)
		if err != nil {
			log.Printf("provider CreateChatCompletion failed model=%q: %v", targetModel, err)
			writeJSONError(w, http.StatusInternalServerError, err.Error())
			return
		}

		out := map[string]interface{}{
			"id":      resp.ID,
			"object":  resp.Object,
			"created": resp.Created,
			"model":   targetModel,
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
	}))

	mux.HandleFunc("/v1/chat/stream", withCORS(func(w http.ResponseWriter, r *http.Request) {
		if !requireAuth(w, r) {
			return
		}
		var reqBody ChatCompletionRequest
		var msg string
		if r.Method == http.MethodGet {
			msg = r.URL.Query().Get("message")
			if msg == "" {
				msg = "Hello from Personal Chat Bot API."
			}
		} else if r.Method == http.MethodPost {
			if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
				writeJSONError(w, http.StatusBadRequest, "invalid request body")
				return
			}
			if len(reqBody.Messages) > 0 {
				for i := len(reqBody.Messages) - 1; i >= 0; i-- {
					if reqBody.Messages[i].Role == "user" {
						msg = reqBody.Messages[i].Content
						break
					}
				}
			}
		} else {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		targetModel := reqBody.Model
		if targetModel == "" {
			targetModel = prov.ModelName()
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			writeJSONError(w, http.StatusInternalServerError, "streaming unsupported")
			return
		}

		openReq := openai.ChatCompletionRequest{
			Model:  targetModel,
			Stream: true,
		}
		if len(reqBody.Messages) > 0 {
			for _, m := range reqBody.Messages {
				openReq.Messages = append(openReq.Messages, openai.ChatCompletionMessage{Role: m.Role, Content: m.Content})
			}
		} else {
			openReq.Messages = []openai.ChatCompletionMessage{{Role: "user", Content: msg}}
		}

		stream, err := prov.CreateChatCompletionStream(r.Context(), openReq)
		if err != nil {
			fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
			flusher.Flush()
			return
		}
		defer stream.Close()

		chunkID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
		created := time.Now().Unix()

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
				chunkData := map[string]interface{}{
					"id":      chunkID,
					"object":  "chat.completion.chunk",
					"created": created,
					"model":   targetModel,
					"choices": []map[string]interface{}{
						{
							"index": 0,
							"delta": map[string]string{
								"content": chunk.Content,
							},
							"finish_reason": nil,
						},
					},
				}
				encoded, _ := json.Marshal(chunkData)
				fmt.Fprintf(w, "data: %s\n\n", encoded)
				flusher.Flush()
			}
		}

		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))

	addrToUse := ":8080"
	if addr != "" {
		addrToUse = addr
	}
	if port := os.Getenv("PORT"); port != "" {
		addrToUse = ":" + port
	}
	if env := os.Getenv("API_ADDR"); env != "" {
		addrToUse = env
	}

	log.Printf("Starting HTTP API server on %s with model %s", addrToUse, prov.ModelName())
	if err := http.ListenAndServe(addrToUse, mux); err != nil {
		log.Fatalf("API server failed: %v", err)
	}
}
