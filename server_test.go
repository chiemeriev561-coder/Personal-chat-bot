package main

import (
	"context"
	"testing"

	"github.com/sashabaranov/go-openai"

	"personalchatbot/provider"
)

func TestProviderRegistryModelResolution(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
	t.Setenv("GROQ_API_KEY", "test-groq-key")

	reg := NewProviderRegistry()

	// Test resolving deepseek model
	prov, targetModel, err := reg.ResolveProvider("deepseek-v4-flash")
	if err != nil {
		t.Fatalf("expected resolution for deepseek-v4-flash, got err: %v", err)
	}
	if prov == nil {
		t.Fatalf("expected non-nil provider for deepseek-v4-flash")
	}
	if targetModel != "deepseek-v4-flash" {
		t.Errorf("expected target model 'deepseek-v4-flash', got '%s'", targetModel)
	}

	// Test resolving with explicit prefix
	prov2, targetModel2, err := reg.ResolveProvider("deepseek/deepseek-v4-flash")
	if err != nil {
		t.Fatalf("expected resolution for deepseek/deepseek-v4-flash, got err: %v", err)
	}
	if prov2 == nil {
		t.Fatalf("expected non-nil provider")
	}
	if targetModel2 != "deepseek-v4-flash" {
		t.Errorf("expected target model 'deepseek-v4-flash', got '%s'", targetModel2)
	}

	// Test default model does not require NVIDIA
	t.Setenv("CHAT_MODEL", "")
	defModel := getDefaultModel()
	if defModel == "nvidia/nemotron-3.5-lightning-30b-a3b" {
		t.Errorf("default model should not be NVIDIA hardcoded")
	}
}

func TestDeepSeekNvidiaBuildKeyAutoDetection(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "nvapi-test123456789")
	t.Setenv("DEEPSEEK_API_BASE", "")

	p, err := provider.NewDeepSeekProviderFromEnv()
	if err != nil {
		t.Fatalf("failed to create provider with nvapi key: %v", err)
	}
	if p == nil {
		t.Fatalf("expected non-nil DeepSeekProvider for nvapi key")
	}
}

func TestDeepSeekV4FlashStreamingDisabled(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "test-key")
	p, err := provider.NewDeepSeekProviderFromEnv()
	if err != nil {
		t.Fatalf("failed to create DeepSeekProvider: %v", err)
	}

	req := openai.ChatCompletionRequest{Model: "deepseek-v4-flash"}
	_, streamErr := p.CreateChatCompletionStream(context.Background(), req)
	if streamErr != provider.ErrNotSupported {
		t.Errorf("expected ErrNotSupported when streaming deepseek-v4-flash, got: %v", streamErr)
	}
}

func TestModelsEndpoint(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

	reg := NewProviderRegistry()
	models := reg.ListModels()

	if len(models) == 0 {
		t.Fatalf("expected models list to be non-empty when DEEPSEEK_API_KEY is set")
	}

	foundDeepSeek := false
	for _, m := range models {
		if id, ok := m["id"].(string); ok && id == "deepseek-v4-flash" {
			foundDeepSeek = true
			break
		}
	}

	if !foundDeepSeek {
		t.Errorf("expected to find deepseek-v4-flash in listed models")
	}
}
