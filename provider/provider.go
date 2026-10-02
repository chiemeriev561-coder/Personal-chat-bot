package provider

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

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

// Stream is a minimal streaming interface used by the app to pull chunks.
type Stream interface {
	Recv() (StreamChunk, error)
	Close() error
}

var ErrNotSupported = errors.New("streaming not supported by provider")

// Provider is the common interface for chat completions.
type Provider interface {
	CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (CompletionResult, error)
	CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (Stream, error)
	ModelName() string
}

// Config defines the configuration for the unified LLM client.
type Config struct {
	APIKey  string
	BaseURL string
	Model   string
	Timeout time.Duration
}

// Client is a clean, robust OpenAI-compatible LLM provider.
type Client struct {
	client  *openai.Client
	model   string
	baseURL string
}

// NewClient creates a new unified LLM client from the given config.
func NewClient(cfg Config) (*Client, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("API key is required")
	}

	clientCfg := openai.DefaultConfig(cfg.APIKey)
	if cfg.BaseURL != "" {
		base := strings.TrimRight(cfg.BaseURL, "/")
		base = strings.TrimSuffix(base, "/chat/completions")
		clientCfg.BaseURL = base
	}

	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	clientCfg.HTTPClient = &http.Client{
		Timeout: timeout,
	}

	return &Client{
		client:  openai.NewClientWithConfig(clientCfg),
		model:   cfg.Model,
		baseURL: clientCfg.BaseURL,
	}, nil
}

// NewClientFromEnv initializes the client from environment variables.
// It checks standard LLM configuration keys with graceful fallbacks.
func NewClientFromEnv() (*Client, error) {
	apiKey := os.Getenv("LLM_API_KEY")
	baseURL := os.Getenv("LLM_BASE_URL")
	model := os.Getenv("LLM_MODEL")

	// Fallback to legacy env variables if LLM_* is not set
	if apiKey == "" {
		if k := os.Getenv("GROQ_API_KEY"); k != "" {
			apiKey = k
			if baseURL == "" {
				baseURL = "https://api.groq.com/openai/v1"
			}
			if model == "" {
				model = "llama-3.3-70b-versatile"
			}
		} else if k := os.Getenv("NVIDIA_API_KEY"); k != "" {
			apiKey = k
			if baseURL == "" {
				baseURL = "https://integrate.api.nvidia.com/v1"
			}
			if model == "" {
				model = "deepseek-ai/deepseek-v4.1-flash"
			}
		} else if k := os.Getenv("OPENAI_API_KEY"); k != "" {
			apiKey = k
			if model == "" {
				model = "gpt-4o-mini"
			}
		}
	}

	if chatModel := os.Getenv("CHAT_MODEL"); chatModel != "" {
		model = chatModel
	}

	if apiKey == "" {
		return nil, fmt.Errorf("no LLM API key configured (set LLM_API_KEY, GROQ_API_KEY, NVIDIA_API_KEY, or OPENAI_API_KEY)")
	}

	return NewClient(Config{
		APIKey:  apiKey,
		BaseURL: baseURL,
		Model:   model,
		Timeout: 90 * time.Second,
	})
}

func (c *Client) ModelName() string {
	return c.model
}

func (c *Client) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (CompletionResult, error) {
	if req.Model == "" {
		req.Model = c.model
	}
	resp, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return CompletionResult{}, err
	}

	out := CompletionResult{
		ID:      resp.ID,
		Object:  resp.Object,
		Created: resp.Created,
		Usage: map[string]int{
			"prompt_tokens":     resp.Usage.PromptTokens,
			"completion_tokens": resp.Usage.CompletionTokens,
			"total_tokens":      resp.Usage.TotalTokens,
		},
	}

	for i, ch := range resp.Choices {
		content := ch.Message.Content
		if content == "" && ch.Message.ReasoningContent != "" {
			content = ch.Message.ReasoningContent
		}
		out.Choices = append(out.Choices, Choice{
			Index:        i,
			Role:         ch.Message.Role,
			Content:      content,
			FinishReason: string(ch.FinishReason),
		})
	}

	return out, nil
}

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
		if ch.Delta.Content != "" {
			combined += ch.Delta.Content
		} else if ch.Delta.ReasoningContent != "" {
			combined += ch.Delta.ReasoningContent
		}
	}
	return StreamChunk{Content: combined}, nil
}

func (w *openAIStreamWrapper) Close() error {
	if w.stream != nil {
		return w.stream.Close()
	}
	return nil
}

func (c *Client) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (Stream, error) {
	if req.Model == "" {
		req.Model = c.model
	}
	stream, err := c.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, err
	}
	return &openAIStreamWrapper{stream: stream}, nil
}
