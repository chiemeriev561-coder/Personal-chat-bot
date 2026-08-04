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
CHAT_MODEL="gemini" # Default model. Set to "groq" to use Groq by default.
```

> **Note**: `.env` is already configured in `.gitignore` to prevent you from accidentally committing your API secrets.

### 2. Build the Application

Build the executable:

```bash
go build -o personalchatbot main.go
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

# Run using the default Groq model (llama-3.3-70b-versatile)
./personalchatbot --model groq

# Run using a specific Groq model
./personalchatbot --model deepseek-r1-distill-llama-70b
```

### Controls in TUI
- **Type messages**: Simply start typing in the lower text area.
- **New Line**: Press `Enter` to create a new line in the text area.
- **Send Message**: Press `Ctrl + S` to send your message to the assistant.
- **Scroll Viewport**: Use your mouse scroll wheel or click-drag to scroll through the conversation history.
- **Quit**: Press `Ctrl + C` or type `exit` / `quit` and press `Ctrl + S`.
