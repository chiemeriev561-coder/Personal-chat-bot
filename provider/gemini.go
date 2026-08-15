package provider

import (
	"context"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/sashabaranov/go-openai"
	"google.golang.org/genai"
)

// GeminiProvider integrates with Google's genai SDK (Gemini).
// It implements both streaming and non-streaming completions.
type GeminiProvider struct {
	client *genai.Client
	chat   *genai.Chat
	model  string
}

func NewGeminiProviderFromEnv() (*GeminiProvider, error) {
	if os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_API_KEY") == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY or GOOGLE_API_KEY not set")
	}
	// create client
	client, err := genai.NewClient(context.Background(), nil)
	if err != nil {
		return nil, err
	}
	model := os.Getenv("GEMINI_MODEL")
	if model == "" {
		model = "gemini-3.6-flash"
	}
	config := &genai.GenerateContentConfig{
		SystemInstruction: &genai.Content{Parts: []*genai.Part{{Text: "You are an expert developer assistant. Provide precise, idiomatic code examples and direct technical answers."}}},
	}
	chat, err := client.Chats.Create(context.Background(), model, config, nil)
	if err != nil {
		return nil, err
	}
	return &GeminiProvider{client: client, chat: chat, model: model}, nil
}

func partsFromMessages(messages []openai.ChatCompletionMessage) []genai.Part {
	parts := make([]genai.Part, 0, len(messages))
	for _, m := range messages {
		// Prefix role for context clarity
		text := fmt.Sprintf("%s: %s", m.Role, m.Content)
		parts = append(parts, genai.Part{Text: text})
	}
	return parts
}

func (g *GeminiProvider) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (CompletionResult, error) {
	// Use streaming send and accumulate results
	parts := partsFromMessages(req.Messages)
	iter := g.chat.SendMessageStream(ctx, parts...)
	var combined string
	for chunk, err := range iter {
		if err != nil {
			if err == io.EOF {
				break
			}
			return CompletionResult{}, err
		}
		for _, cand := range chunk.Candidates {
			if cand.Content != nil {
				for _, p := range cand.Content.Parts {
					if p.Text != "" {
						combined += p.Text
					}
				}
			}
		}
	}
	out := CompletionResult{
		ID:      "gemini-" + fmt.Sprintf("%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Choices: []Choice{{Index: 0, Role: "assistant", Content: combined, FinishReason: "stop"}},
		Usage:   map[string]int{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
	}
	return out, nil
}

// genai stream wrapper
type genaiStreamWrapper struct {
	ch chan struct {
		chunk StreamChunk
		err   error
	}
}

func (w *genaiStreamWrapper) Recv() (StreamChunk, error) {
	it, ok := <-w.ch
	if !ok {
		return StreamChunk{}, io.EOF
	}
	if it.err != nil {
		return StreamChunk{}, it.err
	}
	return it.chunk, nil
}

func (w *genaiStreamWrapper) Close() error { close(w.ch); return nil }

func (g *GeminiProvider) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (Stream, error) {
	parts := partsFromMessages(req.Messages)
	iter := g.chat.SendMessageStream(ctx, parts...)
	ch := make(chan struct {
		chunk StreamChunk
		err   error
	}, 8)
	go func() {
		defer close(ch)
		for chunk, err := range iter {
			if err != nil {
				ch <- struct {
					chunk StreamChunk
					err   error
				}{err: err}
				return
			}
			var combined string
			for _, cand := range chunk.Candidates {
				if cand.Content != nil {
					for _, p := range cand.Content.Parts {
						combined += p.Text
					}
				}
			}
			ch <- struct {
				chunk StreamChunk
				err   error
			}{chunk: StreamChunk{Content: combined}}
		}
	}()
	return &genaiStreamWrapper{ch: ch}, nil
}
