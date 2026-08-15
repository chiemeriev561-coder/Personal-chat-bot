package provider

import (
	"context"
	"fmt"
	"os"

	"github.com/sashabaranov/go-openai"
)

// GeminiProvider is a lightweight marker provider: it advertises Gemini availability
// but does not implement streaming in this wrapper. If a real Gemini integration
// is required, extend this with genai SDK usage.
type GeminiProvider struct {
	available bool
}

func NewGeminiProviderFromEnv() (*GeminiProvider, error) {
	if os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_API_KEY") == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY or GOOGLE_API_KEY not set")
	}
	return &GeminiProvider{available: true}, nil
}

func (g *GeminiProvider) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (CompletionResult, error) {
	return CompletionResult{}, ErrNotSupported
}

func (g *GeminiProvider) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (Stream, error) {
	return nil, ErrNotSupported
}
