package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
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

func getDefaultModel() string {
	if m := os.Getenv("CHAT_MODEL"); m != "" {
		return m
	}
	if m := os.Getenv("DEFAULT_MODEL"); m != "" {
		return m
	}
	if os.Getenv("NVIDIA_API_KEY") != "" {
		return provider.NvidiaDeepSeekV4Flash
	}
	return "deepseek-v4-flash"
}

type ProviderRegistry struct {
	mu        sync.RWMutex
	providers map[string]provider.Provider
}

func NewProviderRegistry() *ProviderRegistry {
	reg := &ProviderRegistry{
		providers: make(map[string]provider.Provider),
	}

	if os.Getenv("NVIDIA_API_KEY") != "" {
		p, err := provider.NewNvidiaProviderFromEnv()
		if err != nil {
			log.Printf("failed to init NVIDIA provider: %v", err)
		} else {
			reg.providers["nvidia"] = p
			log.Printf("NVIDIA provider initialized")
		}
	}

	if os.Getenv("DEEPSEEK_API_KEY") != "" {
		p, err := provider.NewDeepSeekProviderFromEnv()
		if err != nil {
			log.Printf("failed to init DeepSeek provider: %v", err)
		} else {
			reg.providers["deepseek"] = p
			log.Printf("DeepSeek provider initialized")
		}
	}

	if os.Getenv("GEMINI_API_KEY") != "" || os.Getenv("GOOGLE_API_KEY") != "" {
		p, err := provider.NewGeminiProviderFromEnv()
		if err != nil {
			log.Printf("failed to init Gemini provider: %v", err)
		} else {
			reg.providers["gemini"] = p
			log.Printf("Gemini provider initialized")
		}
	}

	if os.Getenv("GROQ_API_KEY") != "" {
		p, err := provider.NewGroqProviderFromEnv()
		if err != nil {
			log.Printf("failed to init Groq provider: %v", err)
		} else {
			reg.providers["groq"] = p
			log.Printf("Groq provider initialized")
		}
	}

	return reg
}

func (reg *ProviderRegistry) ProviderNames() []string {
	reg.mu.RLock()
	defer reg.mu.RUnlock()
	var names []string
	for k := range reg.providers {
		names = append(names, k)
	}
	return names
}

func (reg *ProviderRegistry) ResolveProvider(requestedModel string) (provider.Provider, string, error) {
	reg.mu.RLock()
	defer reg.mu.RUnlock()

	if len(reg.providers) == 0 {
		return nil, "", fmt.Errorf("no AI providers configured. Please set DEEPSEEK_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, or NVIDIA_API_KEY")
	}

	if requestedModel == "" {
		requestedModel = getDefaultModel()
	}

	modelLower := strings.ToLower(requestedModel)

	parts := strings.SplitN(requestedModel, "/", 2)
	if len(parts) == 2 {
		prefix := strings.ToLower(parts[0])
		switch prefix {
		case "nvidia", "deepseek", "gemini", "groq":
			if p, ok := reg.providers[prefix]; ok {
				target := parts[1]
				if prefix == "nvidia" {
					target = provider.NormalizeDeepSeekModelForNVIDIA(target)
				}
				return p, target, nil
			}
		}
	}

	if strings.Contains(modelLower, "deepseek") || strings.Contains(modelLower, "v4-flash") {
		if p, ok := reg.providers["nvidia"]; ok {
			return p, provider.NormalizeDeepSeekModelForNVIDIA(requestedModel), nil
		}
		if p, ok := reg.providers["deepseek"]; ok {
			return p, requestedModel, nil
		}
		if p, ok := reg.providers["groq"]; ok {
			return p, requestedModel, nil
		}
	}

	if strings.Contains(modelLower, "gemini") || strings.Contains(modelLower, "google") {
		if p, ok := reg.providers["gemini"]; ok {
			return p, requestedModel, nil
		}
	}

	if strings.Contains(modelLower, "llama") || strings.Contains(modelLower, "mixtral") || strings.Contains(modelLower, "gemma") || strings.Contains(modelLower, "groq") {
		if p, ok := reg.providers["groq"]; ok {
			return p, requestedModel, nil
		}
		if p, ok := reg.providers["nvidia"]; ok {
			return p, requestedModel, nil
		}
	}

	if strings.Contains(modelLower, "nvidia") || strings.Contains(modelLower, "nemotron") || strings.HasPrefix(modelLower, "nvdev") {
		if p, ok := reg.providers["nvidia"]; ok {
			return p, requestedModel, nil
		}
	}

	for _, key := range []string{"nvidia", "deepseek", "gemini", "groq"} {
		if p, ok := reg.providers[key]; ok {
			target := requestedModel
			if key == "nvidia" {
				target = provider.NormalizeDeepSeekModelForNVIDIA(requestedModel)
			}
			return p, target, nil
		}
	}

	for _, p := range reg.providers {
		return p, requestedModel, nil
	}

	return nil, "", fmt.Errorf("no provider available for model: %s", requestedModel)
}

func (reg *ProviderRegistry) ListModels() []map[string]interface{} {
	reg.mu.RLock()
	defer reg.mu.RUnlock()

	var models []map[string]interface{}
	now := time.Now().Unix()

	if _, ok := reg.providers["nvidia"]; ok {
		nvidiaModels := []string{
			provider.NvidiaDeepSeekV4Flash, // deepseek-ai/deepseek-v4-flash-0731
			"deepseek-ai/deepseek-r1",
			"deepseek-ai/deepseek-v3",
			"nvidia/nemotron-3.5-lightning-30b-a3b",
		}
		for _, m := range nvidiaModels {
			models = append(models, map[string]interface{}{
				"id":       m,
				"object":   "model",
				"created":  now,
				"owned_by": "nvidia",
			})
		}
	}

	if _, ok := reg.providers["deepseek"]; ok {
		if _, hasNvidia := reg.providers["nvidia"]; !hasNvidia {
			deepseekModels := []string{"deepseek-v4-flash", "deepseek-chat", "deepseek-coder", "deepseek-reasoner"}
			for _, m := range deepseekModels {
				models = append(models, map[string]interface{}{
					"id":       m,
					"object":   "model",
					"created":  now,
					"owned_by": "deepseek",
				})
			}
		}
	}

	if _, ok := reg.providers["gemini"]; ok {
		geminiModels := []string{"gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-pro"}
		for _, m := range geminiModels {
			models = append(models, map[string]interface{}{
				"id":       m,
				"object":   "model",
				"created":  now,
				"owned_by": "google",
			})
		}
	}

	if _, ok := reg.providers["groq"]; ok {
		groqModels := []string{"llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "mixtral-8x7b-32768"}
		for _, m := range groqModels {
			models = append(models, map[string]interface{}{
				"id":       m,
				"object":   "model",
				"created":  now,
				"owned_by": "groq",
			})
		}
	}

	return models
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

func writeProviderError(w http.ResponseWriter, err error) {
	if err == nil {
		return
	}
	msg := err.Error()
	lower := strings.ToLower(msg)
	status := http.StatusInternalServerError
	if strings.Contains(lower, "model_not_found") ||
		strings.Contains(lower, "404") ||
		strings.Contains(lower, "410") ||
		strings.Contains(lower, "gone") ||
		strings.Contains(lower, "end of life") ||
		strings.Contains(lower, "not found") ||
		strings.Contains(lower, "invalid model") {
		status = http.StatusBadRequest
	}
	if strings.Contains(lower, "401") || strings.Contains(lower, "unauthorized") || strings.Contains(lower, "invalid api key") || strings.Contains(lower, "authentication") {
		status = http.StatusUnauthorized
	}
	writeJSONError(w, status, msg)
}

func startServer(addr string) {
	if addr == "" {
		addr = ":8080"
	}

	reg := NewProviderRegistry()

	mux := http.NewServeMux()

	mux.HandleFunc("/health", withCORS(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status":        "ok",
			"providers":     reg.ProviderNames(),
			"default_model": getDefaultModel(),
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
			"data":   reg.ListModels(),
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

		if req.Model == "" {
			req.Model = getDefaultModel()
		}

		if provider.IsDeepSeekV4Flash(req.Model) {
			req.Stream = false
		}

		prov, targetModel, err := reg.ResolveProvider(req.Model)
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, "model resolution failed: "+err.Error())
			return
		}

		log.Printf("chat/completions model=%q target=%q stream=%v messages=%d", req.Model, targetModel, req.Stream, len(req.Messages))

		if len(req.Messages) == 0 {
			writeJSONError(w, http.StatusBadRequest, "messages array is required and must not be empty")
			return
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

			openReq := openai.ChatCompletionRequest{
				Model:            targetModel,
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
						openReq.Stream = false
						resp, err2 := prov.CreateChatCompletion(context.Background(), openReq)
						if err2 != nil {
							log.Printf("non-stream fallback failed: %v", err2)
							fmt.Fprintf(w, "event: error\ndata: %s\n\n", err2.Error())
							flusher.Flush()
							return
						}
						if len(resp.Choices) > 0 {
							writeSSEData(w, resp.Choices[0].Content)
							flusher.Flush()
						}
						fmt.Fprintf(w, "data: [DONE]\n\n")
						flusher.Flush()
						return
					}
					log.Printf("stream error: %v", err)
					fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
					flusher.Flush()
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

			writeSSEData(w, "(no provider)")
			flusher.Flush()
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}

		if prov != nil {
			openReq := openai.ChatCompletionRequest{
				Model:            targetModel,
				Temperature:      req.Temperature,
				TopP:             req.TopP,
				MaxTokens:        req.MaxTokens,
				N:                req.N,
				Stop:             req.Stop,
				PresencePenalty:  req.PresencePenalty,
				FrequencyPenalty: req.FrequencyPenalty,
				Stream:           false,
			}
			for _, m := range req.Messages {
				openReq.Messages = append(openReq.Messages, openai.ChatCompletionMessage{Role: m.Role, Content: m.Content})
			}

			resp, err := prov.CreateChatCompletion(context.Background(), openReq)
			if err != nil {
				log.Printf("provider CreateChatCompletion failed model=%q: %v", targetModel, err)
				writeProviderError(w, err)
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
			return
		}

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

	mux.HandleFunc("/v1/chat/stream", withCORS(func(w http.ResponseWriter, r *http.Request) {
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

		if reqBody.Model == "" {
			reqBody.Model = getDefaultModel()
		}

		if provider.IsDeepSeekV4Flash(reqBody.Model) {
			reqBody.Stream = false
		}

		prov, targetModel, err := reg.ResolveProvider(reqBody.Model)
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, "model resolution failed: "+err.Error())
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			writeJSONError(w, http.StatusInternalServerError, "streaming unsupported")
			return
		}

		if prov != nil {
			openReq := openai.ChatCompletionRequest{
				Model:  targetModel,
				Stream: !provider.IsDeepSeekV4Flash(targetModel),
			}
			if len(reqBody.Messages) > 0 {
				for _, m := range reqBody.Messages {
					openReq.Messages = append(openReq.Messages, openai.ChatCompletionMessage{Role: m.Role, Content: m.Content})
				}
			} else {
				openReq.Messages = []openai.ChatCompletionMessage{{Role: "user", Content: msg}}
			}

			if provider.IsDeepSeekV4Flash(targetModel) || !openReq.Stream {
				openReq.Stream = false
				resp, err2 := prov.CreateChatCompletion(context.Background(), openReq)
				if err2 != nil {
					log.Printf("stream-endpoint non-stream failed: %v", err2)
					fmt.Fprintf(w, "event: error\ndata: %s\n\n", err2.Error())
					flusher.Flush()
					return
				}
				if len(resp.Choices) > 0 {
					writeSSEData(w, resp.Choices[0].Content)
					flusher.Flush()
				}
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				return
			}

			stream, err := prov.CreateChatCompletionStream(context.Background(), openReq)
			if err != nil {
				if err == provider.ErrNotSupported {
					openReq.Stream = false
					resp, err2 := prov.CreateChatCompletion(context.Background(), openReq)
					if err2 != nil {
						fmt.Fprintf(w, "event: error\ndata: %s\n\n", err2.Error())
						flusher.Flush()
						return
					}
					if len(resp.Choices) > 0 {
						writeSSEData(w, resp.Choices[0].Content)
						flusher.Flush()
					}
					fmt.Fprintf(w, "data: [DONE]\n\n")
						flusher.Flush()
					return
				}
				fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
				flusher.Flush()
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

		writeSSEData(w, msg)
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
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
