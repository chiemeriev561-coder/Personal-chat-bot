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
	t.Setenv("NVIDIA_API_KEY", "test-nvidia-key")

	reg := NewProviderRegistry()

	// Test resolving deepseek model
	prov, targetModel, err := reg.ResolveProvider("deepseek-v4-flash")
	if err != nil {
		t.Fatalf("expected resolution for deepseek-v4-flash, got err: %v", err)
	}
	if prov == nil {
		t.Fatalf("expected non-nil provider for deepseek-v4-flash")
	}
	if targetModel != provider.NvidiaDeepSeekV4Flash {
		t.Errorf("expected target model '%s', got '%s'", provider.NvidiaDeepSeekV4Flash, targetModel)
	}

	// DeepSeek aliases are routed to NVIDIA's canonical model.
	prov2, targetModel2, err := reg.ResolveProvider("deepseek/deepseek-v4-flash")
	if err != nil {
		t.Fatalf("expected resolution for deepseek/deepseek-v4-flash, got err: %v", err)
	}
	if prov2 == nil {
		t.Fatalf("expected non-nil provider")
	}
	if targetModel2 != provider.NvidiaDeepSeekV4Flash {
		t.Errorf("expected target model '%s', got '%s'", provider.NvidiaDeepSeekV4Flash, targetModel2)
	}

	// The current NVIDIA Build V4 Flash model must not fall through to Groq.
	prov3, targetModel3, err := reg.ResolveProvider("deepseek-v4-flash-0731")
	if err != nil {
		t.Fatalf("expected resolution for deepseek-v4-flash-0731, got err: %v", err)
	}
	if _, ok := prov3.(*provider.NvidiaProvider); !ok {
		t.Fatalf("expected deepseek-v4-flash-0731 to route to NVIDIA")
	}
	if targetModel3 != provider.NvidiaDeepSeekV4Flash {
		t.Errorf("expected target model '%s', got '%s'", provider.NvidiaDeepSeekV4Flash, targetModel3)
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
	t.Setenv("NVIDIA_API_KEY", "test-nvidia-key")

	reg := NewProviderRegistry()
	models := reg.ListModels()

	if len(models) != 2 {
		t.Fatalf("expected exactly two enabled models, got %d", len(models))
	}

	foundDeepSeek, foundNemotron := false, false
	for _, m := range models {
		if id, ok := m["id"].(string); ok && id == provider.NvidiaDeepSeekV4Flash {
			foundDeepSeek = true
		}
		if id, ok := m["id"].(string); ok && id == "nvidia/nemotron-3.5-lightning-30b-a3b" {
			foundNemotron = true
		}
	}

	if !foundDeepSeek || !foundNemotron {
		t.Errorf("expected DeepSeek V4 Flash and NVIDIA Nemotron in listed models")
	}
}
