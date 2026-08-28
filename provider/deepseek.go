package provider

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// DeepSeekProvider uses an OpenAI-compatible client configured for DeepSeek API
// or, when only an NVIDIA key is present, the NVIDIA Build endpoint.
type DeepSeekProvider struct {
	client  *openai.Client
	baseURL string
}

func NewDeepSeekProviderFromEnv() (*DeepSeekProvider, error) {
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("NVIDIA_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("DEEPSEEK_API_KEY or NVIDIA_API_KEY not set")
	}

	base := os.Getenv("DEEPSEEK_API_BASE")
	if base == "" {
		// Prefer NVIDIA endpoint when the key is an NVIDIA key or NVIDIA_API_KEY is set.
		if strings.HasPrefix(apiKey, "nvapi-") || os.Getenv("NVIDIA_API_KEY") != "" {
			base = os.Getenv("NVIDIA_API_BASE")
			if base == "" {
				base = "https://integrate.api.nvidia.com/v1"
			}
		} else {
			base = "https://api.deepseek.com/v1"
		}
	}
	base = strings.TrimRight(base, "/")
	base = strings.TrimSuffix(base, "/chat/completions")
	if base == "https://integrate.api.nvidia.com" {
		base += "/v1"
	}

	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = base
	client := openai.NewClientWithConfig(cfg)
	return &DeepSeekProvider{client: client, baseURL: base}, nil
}

func (d *DeepSeekProvider) prepareRequest(req openai.ChatCompletionRequest) openai.ChatCompletionRequest {
	if strings.Contains(d.baseURL, "nvidia.com") {
		req.Model = NormalizeDeepSeekModelForNVIDIA(req.Model)
	}
	return req
}

func (d *DeepSeekProvider) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (CompletionResult, error) {
	req = d.prepareRequest(req)
	resp, err := d.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return CompletionResult{}, formatEndpointError(err, d.baseURL, req.Model)
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

func (d *DeepSeekProvider) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (Stream, error) {
	req = d.prepareRequest(req)
	stream, err := d.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		if IsDeepSeekV4Flash(req.Model) {
			return nil, ErrNotSupported
		}
		return nil, formatEndpointError(err, d.baseURL, req.Model)
	}
	return &openAIStreamWrapper{stream: stream}, nil
}
