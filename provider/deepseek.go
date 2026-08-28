package provider

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// DeepSeekProvider uses an OpenAI-compatible client configured for DeepSeek API or NVIDIA Build endpoint.
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
		// If using an NVIDIA Build API key (nvapi-...) or NVIDIA key, default base URL to NVIDIA endpoint
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
	// If connecting to NVIDIA Build endpoint, ensure deepseek models carry the "deepseek-ai/" prefix required by NVIDIA
	if strings.Contains(d.baseURL, "nvidia.com") {
		if strings.HasPrefix(req.Model, "deepseek") && !strings.HasPrefix(req.Model, "deepseek-ai/") {
			req.Model = "deepseek-ai/" + req.Model
		}
	}
	return req
}

func formatEndpointError(err error, baseURL string, model string) error {
	if err == nil {
		return nil
	}
	errStr := err.Error()
	if strings.Contains(errStr, "invalid character") || strings.Contains(errStr, "looking for beginning of value") {
		if strings.Contains(baseURL, "nvidia.com") {
			return fmt.Errorf("HTTP 404 from NVIDIA Build (%s) for model '%s'. Available DeepSeek models on NVIDIA Build are 'deepseek-ai/deepseek-r1' and 'deepseek-ai/deepseek-v3'", baseURL, model)
		}
		return fmt.Errorf("HTTP 404 / Invalid response from endpoint (%s) for model '%s'. Please check your model name and API key", baseURL, model)
	}
	return err
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
		out.Choices = append(out.Choices, Choice{Index: i, Role: ch.Message.Role, Content: responseContent(ch.Message), FinishReason: string(ch.FinishReason)})
	}
	return out, nil
}

func (d *DeepSeekProvider) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (Stream, error) {
	req = d.prepareRequest(req)
	if IsDeepSeekV4Flash(req.Model) {
		return nil, ErrNotSupported
	}
	stream, err := d.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, formatEndpointError(err, d.baseURL, req.Model)
	}
	return &openAIStreamWrapper{stream: stream}, nil
}
