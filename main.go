package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/atotto/clipboard"
	osc52 "github.com/aymanbagabas/go-osc52/v2"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"

	"personalchatbot/provider"
)

// Messages for async Bubble Tea updates
type streamChunkMsg string
type streamDoneMsg struct{}
type streamErrMsg struct{ err error }

type model struct {
	prov           provider.Provider
	modelName      string
	viewport       viewport.Model
	textarea       textarea.Model
	spinner        spinner.Model
	glamour        *glamour.TermRenderer
	history        string
	currentAi      string
	lastAiResponse string
	copyStatus     string
	isWaiting      bool
	width          int
	height         int
	providerName   string
}

func parseLastCodeBlock(markdown string) string {
	lines := strings.Split(markdown, "\n")
	var codeBlocks []string
	var currentBlock []string
	inBlock := false

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			if inBlock {
				codeBlocks = append(codeBlocks, strings.Join(currentBlock, "\n"))
				currentBlock = nil
				inBlock = false
			} else {
				inBlock = true
			}
		} else if inBlock {
			currentBlock = append(currentBlock, line)
		}
	}

	if inBlock && len(currentBlock) > 0 {
		codeBlocks = append(codeBlocks, strings.Join(currentBlock, "\n"))
	}

	if len(codeBlocks) == 0 {
		return ""
	}
	return codeBlocks[len(codeBlocks)-1]
}

func (m model) copyToClipboard(text string, successMsg string) model {
	if text == "" {
		m.copyStatus = "Nothing to copy."
		return m
	}

	err := clipboard.WriteAll(text)
	seq := osc52.New(text)
	fmt.Print(seq.String())

	if err != nil {
		fileErr := os.WriteFile("last_response.txt", []byte(text), 0644)
		if fileErr != nil {
			m.copyStatus = "Failed to copy: " + err.Error() + " (failed to write last_response.txt)"
		} else {
			m.copyStatus = "Clipboard util missing. Copied via OSC52 & saved to last_response.txt!"
		}
	} else {
		m.copyStatus = successMsg
	}
	return m
}

func initialModel(prov provider.Provider, providerName string, modelName string) model {
	ta := textarea.New()
	ta.Placeholder = "Type a message... (Ctrl+S: send · Ctrl+Y/K: copy AI/code · Esc: normal mode)"
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

	var systemHeader string
	if providerName != "" {
		systemHeader = fmt.Sprintf("# Victor AI GO CLI Chatbot [%s]\n*Type your message below. Press Ctrl+C to quit.*\n\n---\n", providerName)
	} else {
		systemHeader = "# Victor AI GO CLI Chatbot\n*Type your message below. Press Ctrl+C to quit.*\n\n---\n"
	}

	m := model{
		prov:         prov,
		modelName:    modelName,
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

	switch msg := msg.(type) {
	case tea.KeyMsg:
		if msg.Paste {
			if !m.isWaiting {
				m.textarea.InsertString(msg.String())
			}
			break
		}

		if !m.textarea.Focused() {
			switch msg.String() {
			case "c":
				code := parseLastCodeBlock(m.lastAiResponse)
				if code != "" {
					m = m.copyToClipboard(code, "Copied last code block to clipboard!")
				} else {
					m.copyStatus = "No code blocks found in the last response."
				}
				return m, nil
			case "y":
				m = m.copyToClipboard(m.lastAiResponse, "Copied last AI response to clipboard!")
				return m, nil
			case "i", "a":
				m.textarea.Focus()
				m.copyStatus = ""
				return m, nil
			}
		}

		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit

		case tea.KeyEsc:
			if m.textarea.Focused() {
				m.textarea.Blur()
				m.copyStatus = "Normal mode: 'c' to copy code, 'y' to copy response, 'i' to resume typing"
			} else {
				m.textarea.Focus()
				m.copyStatus = ""
			}
			return m, nil

		case tea.KeyCtrlY:
			m = m.copyToClipboard(m.lastAiResponse, "Copied last AI response to clipboard!")
			return m, nil

		case tea.KeyCtrlK:
			code := parseLastCodeBlock(m.lastAiResponse)
			if code != "" {
				m = m.copyToClipboard(code, "Copied last code block to clipboard!")
			} else {
				m.copyStatus = "No code blocks found in the last response."
			}
			return m, nil

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
			m.copyStatus = ""
			m.updateViewport()

			return m, tea.Batch(
				tiCmd,
				m.spinner.Tick,
				m.sendStreamCmd(input),
			)
		}
	}

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
		footerHeight := 8 // textarea (5 lines) + borders + padding + status line
		m.viewport.Width = msg.Width
		m.viewport.Height = msg.Height - headerHeight - footerHeight
		m.textarea.SetWidth(msg.Width)

		renderer, _ := glamour.NewTermRenderer(
			glamour.WithAutoStyle(),
			glamour.WithWordWrap(msg.Width-4),
		)
		m.glamour = renderer
		m.updateViewport()

	case streamChunkMsg:
		// Keep isWaiting true while chunks are arriving — cleared only on done/err.
		m.currentAi += string(msg)
		m.updateViewport()

	case streamDoneMsg:
		m.isWaiting = false
		m.history += fmt.Sprintf("**Assistant:**\n%s\n\n---\n", m.currentAi)
		m.lastAiResponse = m.currentAi
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

	var statusLine string
	if m.copyStatus != "" {
		statusLine = "\n" + lipgloss.NewStyle().Foreground(lipgloss.Color("42")).Render(m.copyStatus)
	}

	return fmt.Sprintf(
		"%s\n\n%s%s",
		m.viewport.View(),
		footerView,
		statusLine,
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

		// Build openai-compatible request
		openReq := openai.ChatCompletionRequest{
			Model: m.modelName,
			Messages: []openai.ChatCompletionMessage{
				{Role: "system", Content: "You are an expert developer assistant. Provide precise, idiomatic code examples and direct technical answers."},
				{Role: "user", Content: input},
			},
			Stream: true,
		}

		if m.prov != nil {
			stream, err := m.prov.CreateChatCompletionStream(ctx, openReq)
			if err != nil {
				// If provider doesn't support streaming, fall back to non-streaming call and emit full text
				if err == provider.ErrNotSupported {
					_, err2 := m.prov.CreateChatCompletion(ctx, openReq)
					if err2 != nil {
						return streamErrMsg{err: err2}
					}
					return streamDoneMsg{}
				}
				return streamErrMsg{err: err}
			}
			defer stream.Close()

			var cmds []tea.Cmd
			for {
				chunk, err := stream.Recv()
				if err != nil {
					if err == io.EOF {
						break
					}
					return streamErrMsg{err: err}
				}
				if chunk.Content != "" {
					text := chunk.Content
					cmds = append(cmds, func() tea.Msg { return streamChunkMsg(text) })
				}
			}
			cmds = append(cmds, func() tea.Msg { return streamDoneMsg{} })
			return tea.Sequence(cmds...)()
		}

		// No provider: simple local echo
		cmds := []tea.Cmd{
			func() tea.Msg { return streamChunkMsg("(no provider) " + input) },
			func() tea.Msg { return streamDoneMsg{} },
		}
		return tea.Sequence(cmds...)()
	}
}

func main() {
	// Load environment variables from .env file if it exists
	_ = godotenv.Load()

	modelFlag := flag.String("model", "", "Model to use ('gemini' or specific Groq models like 'llama-3.3-70b-versatile', or 'groq' to use default Groq model)")
	serverFlag := flag.Bool("api", false, "Start HTTP API server (don't run TUI)")
	apiAddr := flag.String("api-addr", ":8080", "Address for the HTTP API server (when --api is set)")
	flag.Parse()

	if *serverFlag {
		// Start HTTP API server and exit (keeps CLI unchanged)
		startServer(*apiAddr)
		return
	}

	// Determine selected model and groqModel default if needed.
	selectedModel := *modelFlag
	if selectedModel == "" {
		selectedModel = os.Getenv("CHAT_MODEL")
	}
	selectedModel = strings.ToLower(selectedModel)

	var groqModel string
	if selectedModel == "groq" || strings.HasPrefix(selectedModel, "llama") || strings.HasPrefix(selectedModel, "mixtral") || strings.HasPrefix(selectedModel, "deepseek") {
		if selectedModel == "groq" {
			groqModel = "llama-3.3-70b-versatile"
		} else {
			groqModel = *modelFlag
			if groqModel == "" {
				groqModel = os.Getenv("CHAT_MODEL")
			}
		}
	}

	// Provider selection: NVIDIA -> Groq -> Gemini
	var prov provider.Provider
	var providerName string
	if os.Getenv("NVIDIA_API_KEY") != "" {
		p, err := provider.NewNvidiaProviderFromEnv()
		if err != nil {
			log.Printf("NVIDIA init failed: %v", err)
		} else {
			prov = p
			providerName = "NVIDIA"
		}
	}
	if prov == nil && os.Getenv("GROQ_API_KEY") != "" {
		p, err := provider.NewGroqProviderFromEnv()
		if err != nil {
			log.Printf("Groq init failed: %v", err)
		} else {
			prov = p
			providerName = fmt.Sprintf("Groq (%s)", groqModel)
		}
	}
	if prov == nil && (os.Getenv("GEMINI_API_KEY") != "" || os.Getenv("GOOGLE_API_KEY") != "") {
		p, err := provider.NewGeminiProviderFromEnv()
		if err != nil {
			log.Printf("Gemini init failed: %v", err)
		} else {
			prov = p
			providerName = "Gemini"
		}
	}

	if prov == nil {
		log.Fatal("no provider available: set NVIDIA_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY/GOOGLE_API_KEY")
	}

	p := tea.NewProgram(
		initialModel(prov, providerName, selectedModel),
		tea.WithAltScreen(),       // Use full terminal buffer
		tea.WithMouseCellMotion(), // Allow mouse scroll
	)

	if _, err := p.Run(); err != nil {
		log.Fatalf("Error running program: %v", err)
	}
}
