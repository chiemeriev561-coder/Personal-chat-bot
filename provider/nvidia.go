package provider

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// NvidiaProvider uses an OpenAI-compatible client configured for an NVIDIA
// inference endpoint (integrate.api.nvidia.com).
type NvidiaProvider struct {
	client  *openai.Client
	baseURL string
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
		base = "https://integrate.api.nvidia.com/v1"
	}
	base = strings.TrimRight(base, "/")
	base = strings.TrimSuffix(base, "/chat/completions")
	if base == "https://integrate.api.nvidia.com" {
		base += "/v1"
	}
	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = base
	client := openai.NewClientWithConfig(cfg)
	return &NvidiaProvider{client: client, baseURL: base}, nil
}

func (n *NvidiaProvider) prepareRequest(req openai.ChatCompletionRequest) openai.ChatCompletionRequest {
	// Always normalize DeepSeek model IDs for NVIDIA Build.
	req.Model = NormalizeDeepSeekModelForNVIDIA(req.Model)
	return req
}

func formatEndpointError(err error, baseURL string, model string) error {
	if err == nil {
		return nil
	}
	errStr := err.Error()
	lower := strings.ToLower(errStr)

	// Non-JSON / HTML responses almost always mean 404 model-not-found on NVIDIA.
	if strings.Contains(errStr, "invalid character") ||
		strings.Contains(errStr, "looking for beginning of value") ||
		strings.Contains(lower, "404") ||
		strings.Contains(lower, "not found") ||
		strings.Contains(lower, "model_not_found") {
		if strings.Contains(baseURL, "nvidia.com") {
			return fmt.Errorf("model_not_found: NVIDIA Build rejected model %q (endpoint %s). Use deepseek-ai/deepseek-v4-flash (or deepseek-ai/deepseek-r1 / deepseek-ai/deepseek-v3). Original: %v", model, baseURL, err)
		}
		return fmt.Errorf("model_not_found: endpoint %s rejected model %q. Original: %v", baseURL, model, err)
	}
	return err
}

func (n *NvidiaProvider) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (CompletionResult, error) {
	req = n.prepareRequest(req)
	resp, err := n.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return CompletionResult{}, formatEndpointError(err, n.baseURL, req.Model)
	}
	out := CompletionResult{
		ID:      resp.ID,
		Object:  resp.Object,
		Created: resp.Created,
		Usage:   map[string]int{"prompt_tokens": resp.Usage.PromptTokens, "completion_tokens": resp.Usage.CompletionTokens, "total_tokens": resp.Usage.TotalTokens},
	}
	for i, ch := range resp.Choices {
		out.Choices = append(out.Choices, Choice{
			Index:        i,
			Role:         ch.Message.Role,
			Content:      responseContent(ch.Message),
			FinishReason: string(ch.FinishReason),
		})
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
	req = n.prepareRequest(req)
	// DeepSeek v4 Flash on NVIDIA Build does NOT support streaming.
	// Return immediately so the server falls back to non-streaming.
	if IsDeepSeekV4Flash(req.Model) {
		return nil, ErrNotSupported
	}
	stream, err := n.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, formatEndpointError(err, n.baseURL, req.Model)
	}
	return &openAIStreamWrapper{stream: stream}, nil
}
