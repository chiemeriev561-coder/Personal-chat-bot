package provider

import (
	"context"
	"errors"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// Choice is a normalized completion choice.
type Choice struct {
	Index        int
	Role         string
	Content      string
	FinishReason string
}

// CompletionResult is a normalized non-streaming completion.
type CompletionResult struct {
	ID      string
	Object  string
	Created int64
	Choices []Choice
	Usage   map[string]int
}

// StreamChunk represents a single streaming delta from the provider.
type StreamChunk struct {
	Content string
}

// Stream is a minimal streaming interface used by the server to pull chunks.
type Stream interface {
	Recv() (StreamChunk, error)
	Close() error
}

var ErrNotSupported = errors.New("streaming not supported by provider")

// IsDeepSeekV4Flash checks if a requested model name refers to deepseek v4 flash.
func IsDeepSeekV4Flash(model string) bool {
	m := strings.ToLower(model)
	return strings.Contains(m, "deepseek-v4-flash") ||
		strings.Contains(m, "deepseek-v4") ||
		strings.Contains(m, "v4-flash")
}

// NormalizeDeepSeekModelForNVIDIA returns the canonical NVIDIA Build model ID.
// NVIDIA expects: deepseek-ai/deepseek-v4-flash (not bare deepseek-v4-flash).
func NormalizeDeepSeekModelForNVIDIA(model string) string {
	m := strings.TrimSpace(model)
	if m == "" {
		return "deepseek-ai/deepseek-v4-flash"
	}
	lower := strings.ToLower(m)

	// Already correct
	if strings.HasPrefix(lower, "deepseek-ai/") {
		return m
	}

	// Common aliases users / frontend will send
	switch lower {
	case "deepseek-v4-flash", "deepseek-v4-flash-0731", "v4-flash", "deepseek-v4":
		return "deepseek-ai/deepseek-v4-flash"
	case "deepseek-r1", "deepseek-ai/deepseek-r1":
		return "deepseek-ai/deepseek-r1"
	case "deepseek-v3", "deepseek-ai/deepseek-v3":
		return "deepseek-ai/deepseek-v3"
	}

	if strings.HasPrefix(lower, "deepseek") {
		return "deepseek-ai/" + strings.TrimPrefix(m, "deepseek-ai/")
	}
	return m
}

// Provider is a minimal interface for chat completions.
type Provider interface {
	CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (CompletionResult, error)
	CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (Stream, error)
}
