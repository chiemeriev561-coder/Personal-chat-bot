package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"
)

// Minimal OpenAI-compatible request types (subset)
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatCompletionRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream,omitempty"`
}

// Start a simple HTTP API exposing the requested endpoints.
func startServer(addr string) {
	if addr == "" {
		addr = ":8080"
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

		// Simple echo-style reply for now. Integration with provider layer will replace this.
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

	// SSE streaming endpoint — simple demo stream that emits a few chunks and ends.
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
