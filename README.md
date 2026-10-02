# Victor AI Go CLI Chatbot

A fast, interactive terminal-based personal chatbot and HTTP API server built in Go using [Bubble Tea](https://github.com/charmbracelet/bubbletea) and [Glamour](https://github.com/charmbracelet/glamour).

---

## Features

- **Fluid Terminal UI**: Scrollable chat viewport, responsive resizing, and textarea built with Bubble Tea.
- **Ultra-Fast Zero-Latency Streaming**: Chunks stream in real-time with zero CPU lag, followed by Glamour syntax-highlighted Markdown rendering when complete.
- **Multi-Turn Chat Memory**: Remembers context across conversation turns within the session.
- **Unified Standard LLM Architecture**: Connects to any standard OpenAI-compatible provider (Groq, NVIDIA, DeepSeek, OpenAI, Ollama, etc.).
- **OpenAI-Compatible HTTP API**: Serves `/v1/chat/completions` (JSON and SSE streaming), `/v1/models`, and `/health`.
- **Cancellation & Safety**: In-flight requests can be cancelled cleanly with `Esc` or `Ctrl+C`.

---

## Installation & Setup

### 1. Configuration (.env)

Set your chosen provider's credentials in `.env`:

```env
LLM_API_KEY="your-api-key"
LLM_BASE_URL="https://api.groq.com/openai/v1"   # Optional: defaults to provider's standard base URL
LLM_MODEL="llama-3.3-70b-versatile"            # The model to use
```

*(Legacy variables like `GROQ_API_KEY`, `NVIDIA_API_KEY`, and `OPENAI_API_KEY` are also automatically detected as fallbacks.)*

### 2. Build the Application

```bash
go build -o personalchatbot .
```

---

## Usage

### Run TUI Chatbot

```bash
./personalchatbot
```

Override model at startup:
```bash
./personalchatbot --model your-model-name
```

### Controls in TUI
- **Send Message**: `Ctrl + S`
- **Multi-line Input**: Press `Enter` for new lines
- **Cancel In-Flight Request**: `Esc` or `Ctrl + C` while waiting/streaming
- **Reset Chat Memory**: Type `/clear` or `/reset` and press `Ctrl + S`
- **Copy Response / Code**: Press `Esc` for normal mode (`c` to copy last code block, `y` to copy response, `i` to resume typing)
- **Quit**: `Ctrl + C` (when not waiting) or type `exit`

### Start HTTP API Server

```bash
./personalchatbot --api --api-addr :8080
```

#### API Endpoints
- **POST `/v1/chat/completions`** — OpenAI-compatible JSON or SSE streaming (`"stream": true`).
- **GET `/v1/models`** — Lists active model.
- **GET `/health`** — Health check.

