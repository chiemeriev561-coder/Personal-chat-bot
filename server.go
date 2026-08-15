package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
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

// Start a simple HTTP API exposing the requested endpoints.
func startServer(addr string) {
	if addr == "" {
		addr = ":8080"
	}

	// Optionally initialize an NVIDIA provider if environment vars are set.
	var prov provider.Provider
	if os.Getenv("NVIDIA_API_KEY") != "" && os.Getenv("NVIDIA_API_BASE") != "" {
		p, err := provider.NewNvidiaProviderFromEnv()
		if err != nil {
			log.Printf("failed to init NVIDIA provider: %v", err)
		} else {
			prov = p
			log.Printf("NVIDIA provider initialized")
		}
	}

	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	// OpenAI-compatible (basic) completions endpoint
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid request body", http.StatusBadRequest)
			return
		}

		// If an NVIDIA provider is configured, call it (non-streaming). Otherwise fall back to echo.
		if prov != nil {
			// translate to go-openai request
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

		// Simple echo-style reply when no provider is configured.
		userContent := ""
		if len(req.Messages) > 0 {
			// prefer the last user message
			for i := len(req.Messages) - 1; i >= 0; i-- {
				if req.Messages[i].Role == "user" || req.Messages[i].Role == "assistant" || req.Messages[i].Role == "system" {
					userContent = req.Messages[i].Content
					break
				}
			}
		}
		if userContent == "" && len(req.Messages) > 0 {
			userContent = req.Messages[len(req.Messages)-1].Content
		}

		reply := fmt.Sprintf("Echo: %s", userContent)

		resp := map[string]interface{}{
			"id":      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]string{
						"role":    "assistant",
						"content": reply,
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]int{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(resp)
	})

	// SSE streaming endpoint — uses provider streaming if available, otherwise demo stream.
	mux.HandleFunc("/v1/chat/stream", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Allow a simple ?message= override to echo; otherwise use default text
		msg := r.URL.Query().Get("message")
		if msg == "" {
			msg = "This is a demo stream from the Personal Chat Bot API."
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
			// For GET-based demo, convert query message into a single user message and call the provider stream.
			openReq := openai.ChatCompletionRequest{
				Model: "gpt-4o-mini", // default model name; provider may ignore or require a specific one
				Messages: []openai.ChatCompletionMessage{
					{Role: "user", Content: msg},
				},
				Stream: true,
			}

			stream, err := prov.CreateChatCompletionStream(context.Background(), openReq)
			if err != nil {
				http.Error(w, "provider stream error: "+err.Error(), http.StatusInternalServerError)
				return
			}
			defer stream.Close()

			for {
				resp, err := stream.Recv()
				if err != nil {
					if err == io.EOF {
						break
					}
					fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
					flusher.Flush()
					return
				}
				for _, choice := range resp.Choices {
					if choice.Delta.Content != "" {
						fmt.Fprintf(w, "data: %s\n\n", choice.Delta.Content)
						flusher.Flush()
					}
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
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
			// small delay to simulate streaming
			time.Sleep(250 * time.Millisecond)
		}
	})

	addrToUse := addr
	if env := os.Getenv("API_ADDR"); env != "" {
		addrToUse = env
	}

	log.Printf("Starting HTTP API server on %s", addrToUse)
	if err := http.ListenAndServe(addrToUse, mux); err != nil {
		log.Fatalf("API server failed: %v", err)
	}
}
