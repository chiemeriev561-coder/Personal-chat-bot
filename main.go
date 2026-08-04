package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
	"google.golang.org/genai"
)

// Messages for async Bubble Tea updates
type streamChunkMsg string
type streamDoneMsg struct{}
type streamErrMsg struct{ err error }

type model struct {
	client       *genai.Client
	chat         *genai.Chat
	groqClient   *openai.Client
	groqModel    string
	viewport     viewport.Model
	textarea     textarea.Model
	spinner      spinner.Model
	glamour      *glamour.TermRenderer
	history      string
	currentAi    string
	isWaiting    bool
	width        int
	height       int
	providerName string
}

func initialModel(client *genai.Client, chat *genai.Chat, groqClient *openai.Client, groqModel string) model {
	ta := textarea.New()
	ta.Placeholder = "Type a message... (Enter for new line · Ctrl+S to send)"
	ta.Focus()
	ta.CharLimit = 1000000
	ta.SetWidth(80)
	ta.SetHeight(5)

	vp := viewport.New(80, 20)

	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))

	renderer, _ := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(80),
	)

	var providerName string
	var systemHeader string
	if groqClient != nil {
		providerName = fmt.Sprintf("Groq (%s)", groqModel)
		systemHeader = fmt.Sprintf("# Victor AI GO CLI Chatbot [%s]\n*Type your message below. Press Ctrl+C to quit.*\n\n---\n", providerName)
	} else {
		providerName = "Gemini"
		systemHeader = "# Victor AI GO CLI Chatbot [Gemini]\n*Type your message below. Press Ctrl+C to quit.*\n\n---\n"
	}

	m := model{
		client:       client,
		chat:         chat,
		groqClient:   groqClient,
		groqModel:    groqModel,
		textarea:     ta,
		viewport:     vp,
		spinner:      s,
		glamour:      renderer,
		history:      systemHeader,
		providerName: providerName,
	}

	m.updateViewport()
	return m
}

func (m model) Init() tea.Cmd {
	return textarea.Blink
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var (
		tiCmd tea.Cmd
		vpCmd tea.Cmd
		spCmd tea.Cmd
	)

	m.textarea, tiCmd = m.textarea.Update(msg)
	m.viewport, vpCmd = m.viewport.Update(msg)

	if m.isWaiting {
		m.spinner, spCmd = m.spinner.Update(msg)
	}

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		headerHeight := 1
		footerHeight := 7 // textarea (5 lines) + borders + padding
		m.viewport.Width = msg.Width
		m.viewport.Height = msg.Height - headerHeight - footerHeight
		m.textarea.SetWidth(msg.Width)

		renderer, _ := glamour.NewTermRenderer(
			glamour.WithAutoStyle(),
			glamour.WithWordWrap(msg.Width-4),
		)
		m.glamour = renderer
		m.updateViewport()

	case tea.KeyMsg:
		// Bracketed paste arrives as a KeyMsg with .Paste == true in bubbletea v1.3.10.
		// Handle it before the inner switch so regular key bindings aren't triggered.
		if msg.Paste {
			if !m.isWaiting {
				m.textarea.InsertString(msg.String())
			}
			break
		}
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit

		case tea.KeyCtrlS:
			// Ctrl+S sends; Enter inserts a newline (handled natively by textarea)
			if m.isWaiting {
				return m, nil
			}

			input := strings.TrimSpace(m.textarea.Value())
			if input == "" {
				return m, nil
			}
			if input == "exit" || input == "quit" {
				return m, tea.Quit
			}

			m.textarea.Reset()
			m.history += fmt.Sprintf("**You:** %s\n\n", input)
			m.currentAi = ""
			m.isWaiting = true
			m.updateViewport()

			return m, tea.Batch(
				tiCmd,
				m.spinner.Tick,
				m.sendStreamCmd(input),
			)
		}

	case streamChunkMsg:
		// Keep isWaiting true while chunks are arriving — cleared only on done/err.
		m.currentAi += string(msg)
		m.updateViewport()

	case streamDoneMsg:
		m.isWaiting = false
		m.history += fmt.Sprintf("**Assistant:**\n%s\n\n---\n", m.currentAi)
		m.currentAi = ""
		m.updateViewport()

	case streamErrMsg:
		m.isWaiting = false
		m.history += fmt.Sprintf("\n\n*Assistant: the request failed (%v); please try again.*\n\n---\n", msg.err)
		m.currentAi = ""
		m.updateViewport()
	}

	return m, tea.Batch(tiCmd, vpCmd, spCmd)
}

func (m model) View() string {
	var footerView string
	if m.isWaiting {
		footerView = fmt.Sprintf("%s Assistant (%s) is thinking...", m.spinner.View(), m.providerName)
	} else {
		footerView = m.textarea.View()
	}

	return fmt.Sprintf(
		"%s\n\n%s",
		m.viewport.View(),
		footerView,
	)
}

func (m *model) updateViewport() {
	fullText := m.history
	if m.currentAi != "" {
		fullText += fmt.Sprintf("**Assistant:**\n%s", m.currentAi)
	}

	rendered, err := m.glamour.Render(fullText)
	if err != nil {
		m.viewport.SetContent(fullText)
	} else {
		m.viewport.SetContent(rendered)
	}
	m.viewport.GotoBottom()
}

// sendStreamCmd collects all stream chunks from the active API (Gemini or Groq)
// and returns them as a tea.Sequence so Bubble Tea dispatches each streamChunkMsg
// individually, re-rendering after every chunk for a live typing effect.
func (m model) sendStreamCmd(input string) tea.Cmd {
	return func() tea.Msg {
		ctx := context.Background()

		if m.groqClient != nil {
			req := openai.ChatCompletionRequest{
				Model: m.groqModel,
				Messages: []openai.ChatCompletionMessage{
					{
						Role:    openai.ChatMessageRoleSystem,
						Content: "You are an expert developer assistant. Provide precise, idiomatic code examples and direct technical answers.",
					},
					{
						Role:    openai.ChatMessageRoleUser,
						Content: input,
					},
				},
				Stream: true,
			}
			stream, err := m.groqClient.CreateChatCompletionStream(ctx, req)
			if err != nil {
				return streamErrMsg{err: err}
			}
			defer stream.Close()

			var cmds []tea.Cmd
			for {
				response, err := stream.Recv()
				if errors.Is(err, io.EOF) {
					break
				}
				if err != nil {
					return streamErrMsg{err: err}
				}
				if len(response.Choices) > 0 {
					text := response.Choices[0].Delta.Content
					if text != "" {
						cmds = append(cmds, func() tea.Msg {
							return streamChunkMsg(text)
						})
					}
				}
			}
			cmds = append(cmds, func() tea.Msg { return streamDoneMsg{} })
			return tea.Sequence(cmds...)()
		}

		// Otherwise, use Gemini
		iter := m.chat.SendMessageStream(ctx, genai.Part{Text: input})

		var cmds []tea.Cmd
		for chunk, err := range iter {
			if err != nil {
				return streamErrMsg{err: err}
			}

			for _, cand := range chunk.Candidates {
				if cand.Content != nil {
					for _, part := range cand.Content.Parts {
						if part.Text != "" {
							text := part.Text // capture loop variable
							cmds = append(cmds, func() tea.Msg {
								return streamChunkMsg(text)
							})
						}
					}
				}
			}
		}

		// Append the done signal after all chunks.
		cmds = append(cmds, func() tea.Msg { return streamDoneMsg{} })

		// Return the sequence as a Cmd (not invoked here — Bubble Tea runs it).
		return tea.Sequence(cmds...)()
	}
}

func main() {
	// Load environment variables from .env file if it exists
	_ = godotenv.Load()

	modelFlag := flag.String("model", "", "Model to use ('gemini' or specific Groq models like 'llama-3.3-70b-versatile', or 'groq' to use default Groq model)")
	flag.Parse()

	ctx := context.Background()

	// Check if we should use Groq
	useGroq := false
	selectedModel := *modelFlag

	// Fallback to environment variable if no flag was provided
	if selectedModel == "" {
		selectedModel = os.Getenv("CHAT_MODEL")
	}

	selectedModel = strings.ToLower(selectedModel)

	var groqModel string
	if selectedModel == "groq" || strings.HasPrefix(selectedModel, "llama") || strings.HasPrefix(selectedModel, "mixtral") || strings.HasPrefix(selectedModel, "deepseek") {
		useGroq = true
		if selectedModel == "groq" {
			groqModel = "llama-3.3-70b-versatile"
		} else {
			groqModel = *modelFlag
			if groqModel == "" {
				groqModel = os.Getenv("CHAT_MODEL")
			}
		}
	}

	var client *genai.Client
	var chat *genai.Chat
	var groqClient *openai.Client

	if useGroq {
		apiKey := os.Getenv("GROQ_API_KEY")
		if apiKey == "" {
			log.Fatal("missing Groq API key: set GROQ_API_KEY before starting")
		}

		config := openai.DefaultConfig(apiKey)
		config.BaseURL = "https://api.groq.com/openai/v1"
		groqClient = openai.NewClientWithConfig(config)
	} else {
		if os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_API_KEY") == "" {
			log.Fatal("missing API key: set GEMINI_API_KEY or GOOGLE_API_KEY before starting")
		}

		var err error
		client, err = genai.NewClient(ctx, nil)
		if err != nil {
			log.Fatalf("Failed to create client: %v", err)
		}

		config := &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{Parts: []*genai.Part{{
				Text: "You are an expert developer assistant. Provide precise, idiomatic code examples and direct technical answers.",
			}}},
		}

		geminiModel := "gemini-3.6-flash"
		if selectedModel == "gemini-3.5-flash" || selectedModel == "gemini-3.6-flash" {
			geminiModel = selectedModel
		}

		chat, err = client.Chats.Create(ctx, geminiModel, config, nil)
		if err != nil {
			log.Fatalf("Failed to initialize chat: %v", err)
		}
	}

	p := tea.NewProgram(
		initialModel(client, chat, groqClient, groqModel),
		tea.WithAltScreen(),       // Use full terminal buffer
		tea.WithMouseCellMotion(), // Allow mouse scroll
		// Bracketed paste is enabled by default in bubbletea v1.3.10
	)

	if _, err := p.Run(); err != nil {
		log.Fatalf("Error running program: %v", err)
	}
}

