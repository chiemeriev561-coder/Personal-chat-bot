# Victor AI Go CLI Chatbot

A terminal-based personal chatbot built in Go using [Bubble Tea](https://github.com/charmbracelet/bubbletea) for a rich, interactive text user interface (TUI). It supports both **Google Gemini** and **Groq** backends with real-time markdown-rendered streaming output using [Glamour](https://github.com/charmbracelet/glamour).

---

## Features

- **Interactive Terminal UI**: Scrollable chat viewport, responsive resizing, and multiline-ready textarea built with Bubble Tea.
- **Multiple AI Backends**:
  - **Google Gemini** (using the new `google.golang.org/genai` client).
  - **Groq** (using an OpenAI-compatible stream client with `github.com/sashabaranov/go-openai`).
- **Live Markdown Streaming**: View answers typed out in real-time with technical syntax highlighting.
- **Automatic Environment Configuration**: Loads your keys from a `.env` file on startup.

---

## Installation & Setup

### Prerequisites

- [Go](https://go.dev/doc/install) (version 1.26.5 or later recommended)

### 1. Clone & Set Up Configuration

Create a `.env` file in the root of the project (one is generated automatically for you as `.env`):

```env
GEMINI_API_KEY="your-gemini-api-key-here"
GROQ_API_KEY="your-groq-api-key-here"
CHAT_MODEL="gemini" # Set to "groq" or any GroqCloud model ID for the CLI.
```

> **Note**: `.env` is already configured in `.gitignore` to prevent you from accidentally committing your API secrets.

### 2. Build the Application

Build the executable:

```bash
go build -o personalchatbot .
```

---

## Usage

### Run with Default Settings
Run the application directly. It will read settings from your `.env` file:

```bash
./personalchatbot
```

### Overriding the Model at Runtime
You can override the configured default model by using the `--model` CLI flag:

```bash
# Run using Gemini
./personalchatbot --model gemini

# Run using the default Groq model (openai/gpt-oss-20b)
./personalchatbot --model groq

# Run using a specific Groq model
./personalchatbot --model openai/gpt-oss-20b
```

### Controls in TUI
- **Type messages**: Simply start typing in the lower text area.
- **Send Message**: Press `Ctrl + S` to send your message to the assistant.
- **Switch Models**:
  - Press **`Ctrl + M`** to cycle through all available models across active providers in real-time.
  - Type **`/model <name>`** (e.g. `/model gemini-3.6-flash`, `/model deepseek-v4-flash`, `/model groq`, `/model nvidia`) and press `Ctrl + S` to switch to a specific model.
  - Type **`/models`** or **`/list`** and press `Ctrl + S` to list all available models and active providers.
- **Normal Mode**: Press `Esc` to toggle normal mode (`c` to copy last code block, `y` to copy last AI response, `i` to resume typing).
- **Scroll Viewport**: Use your mouse scroll wheel or click-drag to scroll through the conversation history.
- **Quit**: Press `Ctrl + C` or type `exit` / `quit` and press `Ctrl + S`.

### HTTP API

The app exposes an OpenAI-compatible HTTP API when started with `--api`. Endpoints:
- **POST `/v1/chat/completions`** — OpenAI-compatible completion (JSON request/response). Supports dynamic model switching per request (`"model": "deepseek-v4-flash-0731"`, `"model": "gemini-3.6-flash"`, etc. or prefixed like `"model": "nvidia/..."`).
- **POST or GET `/v1/chat/stream`** — SSE streaming endpoint. POST accepts a ChatCompletion request body (`stream=true`) to stream model deltas as SSE; GET supports a simple `?message=` demo.
- **GET `/v1/models`** — OpenAI-compatible endpoint listing available models for active providers.
- **GET `/health`** — Health check (returns `{"status":"ok", "providers":[...], "default_model":"..."}`).

#### Model & Provider Switching
The API dynamically routes requests to the correct provider based on the `"model"` field in the JSON request body. Supported environment variables:
- **DeepSeek/NVIDIA Build**: `NVIDIA_API_KEY` (model `deepseek-ai/deepseek-v4-flash-0731` or alias `deepseek-v4-flash-0731`)
- **Google Gemini**: `GEMINI_API_KEY` or `GOOGLE_API_KEY` (models like `gemini-3.6-flash`, `gemini-2.5-flash`)
- **Groq**: `GROQ_API_KEY` (model `mixtral-8x7b-32768`)
- **NVIDIA**: `NVIDIA_API_KEY` (models like `nvidia/nemotron-3.5-lightning-30b-a3b`)

NVIDIA is not used as a hardcoded default. You can set `CHAT_MODEL` in your `.env` to specify your preferred default model.

Example:

```bash
# Start API server (default :8080)
DEEPSEEK_API_KEY=... GEMINI_API_KEY=... go run main.go --api

# List available models
curl http://localhost:8080/v1/models

# Request completion with DeepSeek V4 Flash
curl -X POST -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello"}]}' \
  http://localhost:8080/v1/chat/completions

# Switch models dynamically for a different task (e.g. Gemini)
curl -X POST -H "Content-Type: application/json" \
  -d '{"model":"gemini-3.6-flash","messages":[{"role":"user","content":"Compare Go and Rust"}]}' \
  http://localhost:8080/v1/chat/completions
```
