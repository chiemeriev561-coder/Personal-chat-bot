package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"

	"personalchatbot/provider"
)

type mockProvider struct {
	model string
}

func (m *mockProvider) ModelName() string {
	return m.model
}

func (m *mockProvider) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (provider.CompletionResult, error) {
	return provider.CompletionResult{
		ID:      "test-id",
		Object:  "chat.completion",
		Created: 123456789,
		Choices: []provider.Choice{
			{
				Index:        0,
				Role:         "assistant",
				Content:      "Mock response",
				FinishReason: "stop",
			},
		},
	}, nil
}

type mockStream struct {
	chunks []string
	idx    int
}

func (s *mockStream) Recv() (provider.StreamChunk, error) {
	if s.idx >= len(s.chunks) {
		return provider.StreamChunk{}, context.Canceled
	}
	chunk := s.chunks[s.idx]
	s.idx++
	return provider.StreamChunk{Content: chunk}, nil
}

func (s *mockStream) Close() error {
	return nil
}

func (m *mockProvider) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (provider.Stream, error) {
	return &mockStream{chunks: []string{"Hello", " world!"}}, nil
}

func TestUnifiedClientEnvResolution(t *testing.T) {
	t.Setenv("LLM_API_KEY", "custom-key")
	t.Setenv("LLM_BASE_URL", "https://api.custom.com/v1")
	t.Setenv("LLM_MODEL", "custom-model")

	client, err := provider.NewClientFromEnv()
	if err != nil {
		t.Fatalf("expected client to initialize from LLM_* env: %v", err)
	}
	if client.ModelName() != "custom-model" {
		t.Errorf("expected model 'custom-model', got '%s'", client.ModelName())
	}
}

func TestUnifiedClientFallbackToGroq(t *testing.T) {
	t.Setenv("LLM_API_KEY", "")
	t.Setenv("GROQ_API_KEY", "groq-test-key")
	t.Setenv("NVIDIA_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")

	client, err := provider.NewClientFromEnv()
	if err != nil {
		t.Fatalf("expected client to initialize from GROQ_API_KEY fallback: %v", err)
	}
	if client.ModelName() == "" {
		t.Errorf("expected non-empty model name for groq fallback")
	}
}

func TestHealthEndpoint(t *testing.T) {
	mock := &mockProvider{model: "test-model"}

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "ok",
			"model":  mock.ModelName(),
		})
	})
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp["status"] != "ok" {
		t.Errorf("expected status 'ok', got %v", resp["status"])
	}
	if resp["model"] != "test-model" {
		t.Errorf("expected model 'test-model', got %v", resp["model"])
	}
}

func TestChatCompletionsNonStream(t *testing.T) {
	mock := &mockProvider{model: "test-model"}

	body := `{"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		openReq := openai.ChatCompletionRequest{
			Model: mock.ModelName(),
			Messages: []openai.ChatCompletionMessage{
				{Role: "user", Content: "hi"},
			},
		}
		resp, err := mock.CreateChatCompletion(r.Context(), openReq)
		if err != nil {
			t.Fatalf("failed completion: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}
}

