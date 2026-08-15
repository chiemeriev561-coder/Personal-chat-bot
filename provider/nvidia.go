package provider

import (
	"context"
	"fmt"
	"os"

	"github.com/sashabaranov/go-openai"
)

// NvidiaProvider uses an OpenAI-compatible client configured for an NVIDIA
// inference endpoint.
type NvidiaProvider struct {
	client *openai.Client
}

func responseContent(message openai.ChatCompletionMessage) string {
	if message.Content != "" {
		return message.Content
	}
	// Some NVIDIA reasoning models return their only text in this field.
	return message.ReasoningContent
}

func NewNvidiaProviderFromEnv() (*NvidiaProvider, error) {
	apiKey := os.Getenv("NVIDIA_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("NVIDIA_API_KEY not set")
	}
	base := os.Getenv("NVIDIA_API_BASE")
	if base == "" {
		// sensible default for NVIDIA's OpenAI-compatible endpoint
		base = "https://integrate.api.nvidia.com/v1"
	}
	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = base
	client := openai.NewClientWithConfig(cfg)
	return &NvidiaProvider{client: client}, nil
}

func (n *NvidiaProvider) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (CompletionResult, error) {
	resp, err := n.client.CreateChatCompletion(ctx, req)
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
		out.Choices = append(out.Choices, Choice{Index: i, Role: ch.Message.Role, Content: responseContent(ch.Message), FinishReason: string(ch.FinishReason)})
	}
	return out, nil
}

// openai.ChatCompletionStream implements Recv() that returns choices with Delta.
// Wrap it to our Stream interface.
type openAIStreamWrapper struct {
	stream *openai.ChatCompletionStream
}

func (w *openAIStreamWrapper) Recv() (StreamChunk, error) {
	resp, err := w.stream.Recv()
	if err != nil {
		return StreamChunk{}, err
	}
	// aggregate all delta pieces into a single chunk
	var combined string
	for _, ch := range resp.Choices {
		combined += ch.Delta.Content
		if ch.Delta.Content == "" {
			combined += ch.Delta.ReasoningContent
		}
	}
	return StreamChunk{Content: combined}, nil
}

func (w *openAIStreamWrapper) Close() error {
	return w.stream.Close()
}

func (n *NvidiaProvider) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (Stream, error) {
	stream, err := n.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, err
	}
	return &openAIStreamWrapper{stream: stream}, nil
}
