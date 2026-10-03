package provider

import (
	"testing"
)

func TestCleanEnvRemovesURLQuoting(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   string
		want string
	}{
		{name: "plain quotes", in: `"https://integrate.api.nvidia.com/v1"`, want: "https://integrate.api.nvidia.com/v1"},
		{name: "escaped quotes", in: `\"https://integrate.api.nvidia.com/v1\"`, want: "https://integrate.api.nvidia.com/v1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := cleanEnv(tc.in); got != tc.want {
				t.Fatalf("cleanEnv(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestNewClientRemovesQuotedBaseURL(t *testing.T) {
	client, err := NewClient(Config{
		APIKey:  "test-key",
		BaseURL: `"https://integrate.api.nvidia.com/v1"`,
		Model:   "test-model",
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if client.baseURL != "https://integrate.api.nvidia.com/v1" {
		t.Fatalf("client base URL = %q, want unquoted NVIDIA URL", client.baseURL)
	}
}
