package provider

import (
	"context"
	"errors"

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

// Provider is a minimal interface for chat completions.
type Provider interface {
	CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (CompletionResult, error)
	CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (Stream, error)
}
