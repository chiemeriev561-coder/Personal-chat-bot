package provider

import (
	"context"
	"fmt"
	"os"

	"github.com/sashabaranov/go-openai"
)

// GroqProvider uses the OpenAI-compatible Groq API.
type GroqProvider struct {
	client *openai.Client
}

func NewGroqProviderFromEnv() (*GroqProvider, error) {
	apiKey := os.Getenv("GROQ_API_KEY")
	base := os.Getenv("GROQ_API_BASE")
	if apiKey == "" {
		return nil, fmt.Errorf("GROQ_API_KEY not set")
	}
	cfg := openai.DefaultConfig(apiKey)
	// groq default base URL used in main.go was https://api.groq.com/openai/v1
	if base != "" {
		cfg.BaseURL = base
	} else {
		cfg.BaseURL = "https://api.groq.com/openai/v1"
	}
	client := openai.NewClientWithConfig(cfg)
	return &GroqProvider{client: client}, nil
}

func (g *GroqProvider) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (CompletionResult, error) {
	resp, err := g.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return CompletionResult{}, err
	}
	out := CompletionResult{
		ID:      resp.ID,
		Object:  resp.Object,
		Created: resp.Created,
		Usage:   map[string]int{"prompt_tokens": resp.Usage.PromptTokens, "completion_tokens": resp.Usage.CompletionTokens, "total_tokens": resp.Usage.TotalTokens},
	}
	for i, ch := range resp.Choices {
		out.Choices = append(out.Choices, Choice{Index: i, Role: ch.Message.Role, Content: ch.Message.Content, FinishReason: string(ch.FinishReason)})
	}
	return out, nil
}

// streaming
func (g *GroqProvider) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (Stream, error) {
	stream, err := g.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, err
	}
	return &openAIStreamWrapper{stream: stream}, nil
}
