package provider

import (
	"context"
	"fmt"
	"os"

	"github.com/sashabaranov/go-openai"
)

// NvidiaProvider uses an OpenAI-compatible client configured for an NVIDIA
// inference endpoint (NVIDIA may provide OpenAI-compatible REST endpoints).
type NvidiaProvider struct {
	client *openai.Client
}

func NewNvidiaProviderFromEnv() (*NvidiaProvider, error) {
	apiKey := os.Getenv("NVIDIA_API_KEY")
	base := os.Getenv("NVIDIA_API_BASE")
	if apiKey == "" || base == "" {
		return nil, fmt.Errorf("NVIDIA_API_KEY or NVIDIA_API_BASE not set")
	}
	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = base
	client := openai.NewClientWithConfig(cfg)
	return &NvidiaProvider{client: client}, nil
}

func (n *NvidiaProvider) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	return n.client.CreateChatCompletion(ctx, req)
}

func (n *NvidiaProvider) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionStream, error) {
	return n.client.CreateChatCompletionStream(ctx, req)
}
