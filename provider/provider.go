package provider

import (
	"context"

	"github.com/sashabaranov/go-openai"
)

// Provider is a minimal interface for OpenAI-compatible chat completions.
// Implementations may wrap different backends (Groq, Gemini, NVIDIA).
type Provider interface {
	CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error)
	CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionStream, error)
}
