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

// Canonical NVIDIA model for DeepSeek V4 Flash (post-EOL of deepseek-ai/deepseek-v4-flash).
const NvidiaDeepSeekV4Flash = "deepseek-ai/deepseek-v4-flash-0731"

// IsDeepSeekV4Flash checks if a requested model name refers to deepseek v4 flash.
func IsDeepSeekV4Flash(model string) bool {
	m := strings.ToLower(model)
	return strings.Contains(m, "deepseek-v4-flash") ||
		strings.Contains(m, "deepseek-v4") ||
		strings.Contains(m, "v4-flash")
}

// NormalizeDeepSeekModelForNVIDIA returns the canonical NVIDIA Build model ID.
// deepseek-ai/deepseek-v4-flash was EOL 2026-08-07; use deepseek-ai/deepseek-v4-flash-0731.
func NormalizeDeepSeekModelForNVIDIA(model string) string {
	m := strings.TrimSpace(model)
	if m == "" {
		return NvidiaDeepSeekV4Flash
	}
	lower := strings.ToLower(m)

	// Already the current canonical id
	if lower == NvidiaDeepSeekV4Flash || lower == "deepseek-ai/deepseek-v4-flash-0731" {
		return NvidiaDeepSeekV4Flash
	}

	// EOL / alias → current flash
	switch lower {
	case "deepseek-ai/deepseek-v4-flash",
		"deepseek-v4-flash",
		"deepseek-v4-flash-0731",
		"v4-flash",
		"deepseek-v4",
		"deepseek-ai/deepseek-v4":
		return NvidiaDeepSeekV4Flash
	case "deepseek-r1", "deepseek-ai/deepseek-r1":
		return "deepseek-ai/deepseek-r1"
	case "deepseek-v3", "deepseek-ai/deepseek-v3":
		return "deepseek-ai/deepseek-v3"
	}

	if strings.HasPrefix(lower, "deepseek-ai/") {
		return m
	}
	if strings.HasPrefix(lower, "deepseek") {
		return "deepseek-ai/" + strings.TrimPrefix(strings.TrimPrefix(m, "deepseek-ai/"), "deepseek/")
	}
	return m
}

// Provider is a minimal interface for chat completions.
type Provider interface {
	CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (CompletionResult, error)
	CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (Stream, error)
}
