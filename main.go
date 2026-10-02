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
type streamStartMsg struct{ stream provider.Stream }
type streamChunkWithStreamMsg struct {
	content string
	stream  provider.Stream
}
type streamDoneMsg struct{}
type streamErrMsg struct{ err error }

type model struct {
	prov            provider.Provider
	modelName       string
	viewport        viewport.Model
	textarea        textarea.Model
	spinner         spinner.Model
	glamour         *glamour.TermRenderer
	history         string
	renderedHistory string
	currentAi       string
	lastAiResponse  string
	copyStatus      string
	isWaiting       bool
	width           int
	height          int
	messages        []openai.ChatCompletionMessage
	cancelFunc      context.CancelFunc
}

func newRenderer(width int) *glamour.TermRenderer {
	if width < 1 {
		width = 1
	}
	renderer, err := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(width),
	)
	if err != nil {
		log.Printf("markdown renderer unavailable: %v", err)
		return nil
	}
	return renderer
}

func (m *model) renderMarkdown(md string) string {
	if m.glamour == nil {
		return md
	}
	rendered, err := m.glamour.Render(md)
	if err != nil {
		return md
	}
	return rendered
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

func initialModel(prov provider.Provider, selectedModel string) model {
	if selectedModel == "" {
		selectedModel = prov.ModelName()
	}

	ta := textarea.New()
	ta.Placeholder = "Type a message... (Ctrl+S: send · Esc: cancel / normal mode · /clear: reset chat)"
	ta.Focus()
	ta.CharLimit = 1000000
	ta.SetWidth(80)
	ta.SetHeight(5)

	vp := viewport.New(80, 20)

	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))

	systemHeader := fmt.Sprintf("# Victor AI Chatbot [%s]\n*Type your message below. Press Ctrl+S to send. Type /clear to reset history. Press Ctrl+C to quit.*\n\n---\n", selectedModel)

	renderer := newRenderer(80)
	var renderedHeader string
	if renderer != nil {
		if r, err := renderer.Render(systemHeader); err == nil {
			renderedHeader = r
		} else {
			renderedHeader = systemHeader
		}
	} else {
		renderedHeader = systemHeader
	}

	m := model{
		prov:            prov,
		modelName:       selectedModel,
		textarea:        ta,
		viewport:        vp,
		spinner:         s,
		glamour:         renderer,
		history:         systemHeader,
		renderedHistory: renderedHeader,
		messages: []openai.ChatCompletionMessage{
			{
				Role:    "system",
				Content: "You are an expert developer assistant. Provide precise, idiomatic code examples and direct technical answers.",
			},
		},
	}

	m.viewport.SetContent(m.renderedHistory)
	m.viewport.GotoBottom()
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

		if !m.textarea.Focused() && !m.isWaiting {
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
			if m.isWaiting && m.cancelFunc != nil {
				m.cancelFunc()
				m.cancelFunc = nil
				m.isWaiting = false
				m.history += "\n*[Cancelled by user]*\n\n---\n"
				m.renderedHistory = m.renderMarkdown(m.history)
				m.currentAi = ""
				m.updateViewport(false)
				return m, nil
			}
			return m, tea.Quit

		case tea.KeyEsc:
			if m.isWaiting && m.cancelFunc != nil {
				m.cancelFunc()
				m.cancelFunc = nil
				m.isWaiting = false
				m.history += "\n*[Cancelled by user]*\n\n---\n"
				m.renderedHistory = m.renderMarkdown(m.history)
				m.currentAi = ""
				m.updateViewport(false)
				return m, nil
			}
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

			if input == "/clear" || input == "/reset" {
				m.textarea.Reset()
				systemHeader := fmt.Sprintf("# Victor AI Chatbot [%s]\n*Chat history reset. Press Ctrl+S to send. Press Ctrl+C to quit.*\n\n---\n", m.modelName)
				m.history = systemHeader
				m.renderedHistory = m.renderMarkdown(systemHeader)
				m.messages = []openai.ChatCompletionMessage{
					{
						Role:    "system",
						Content: "You are an expert developer assistant. Provide precise, idiomatic code examples and direct technical answers.",
					},
				}
				m.currentAi = ""
				m.lastAiResponse = ""
				m.copyStatus = "Conversation history cleared."
				m.updateViewport(false)
				return m, nil
			}

			m.textarea.Reset()
			userTurnMd := fmt.Sprintf("**You:** %s\n\n", input)
			m.history += userTurnMd
			m.renderedHistory += m.renderMarkdown(userTurnMd)
			m.currentAi = ""
			m.isWaiting = true
			m.copyStatus = ""

			// Append user message to conversation memory
			m.messages = append(m.messages, openai.ChatCompletionMessage{
				Role:    "user",
				Content: input,
			})

			m.updateViewport(false)

			ctx, cancel := context.WithCancel(context.Background())
			m.cancelFunc = cancel

			return m, tea.Batch(
				tiCmd,
				m.spinner.Tick,
				m.sendStreamCmd(ctx),
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

		m.glamour = newRenderer(msg.Width - 4)
		m.renderedHistory = m.renderMarkdown(m.history)
		m.updateViewport(false)

	case streamStartMsg:
		return m, receiveStreamCmd(msg.stream)

	case streamChunkWithStreamMsg:
		m.currentAi += msg.content
		// During streaming, update viewport fast without full markdown re-parsing
		m.updateViewport(true)
		return m, receiveStreamCmd(msg.stream)

	case streamDoneMsg:
		m.isWaiting = false
		m.cancelFunc = nil
		aiTurnMd := fmt.Sprintf("**Assistant:**\n%s\n\n---\n", m.currentAi)
		m.history += aiTurnMd
		m.renderedHistory += m.renderMarkdown(aiTurnMd)

		// Append assistant response to conversation memory
		m.messages = append(m.messages, openai.ChatCompletionMessage{
			Role:    "assistant",
			Content: m.currentAi,
		})

		m.lastAiResponse = m.currentAi
		m.currentAi = ""
		m.updateViewport(false)

	case streamErrMsg:
		m.isWaiting = false
		m.cancelFunc = nil
		errMd := fmt.Sprintf("\n\n*Assistant error: %v*\n\n---\n", msg.err)
		m.history += errMd
		m.renderedHistory += m.renderMarkdown(errMd)
		m.currentAi = ""
		m.updateViewport(false)
	}

	return m, tea.Batch(tiCmd, vpCmd, spCmd)
}

func (m model) View() string {
	var footerView string
	if m.isWaiting {
		footerView = fmt.Sprintf("%s Assistant (%s) is thinking...", m.spinner.View(), m.modelName)
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

// updateViewport updates viewport content. During streaming, it avoids re-parsing
// the full document through Glamour on every token, guaranteeing ultra-fast streaming.
func (m *model) updateViewport(isStreaming bool) {
	if isStreaming {
		content := m.renderedHistory + fmt.Sprintf("\n**Assistant:**\n%s", m.currentAi)
		m.viewport.SetContent(content)
	} else {
		m.viewport.SetContent(m.renderedHistory)
	}
	m.viewport.GotoBottom()
}

func (m model) sendStreamCmd(ctx context.Context) tea.Cmd {
	return func() tea.Msg {
		openReq := openai.ChatCompletionRequest{
			Model:    m.modelName,
			Messages: m.messages,
			Stream:   true,
		}

		if m.prov != nil {
			stream, err := m.prov.CreateChatCompletionStream(ctx, openReq)
			if err != nil {
				// Fall back to non-streaming if provider does not support it
				if err == provider.ErrNotSupported {
					openReq.Stream = false
					resp, err2 := m.prov.CreateChatCompletion(ctx, openReq)
					if err2 != nil {
						return streamErrMsg{err: err2}
					}
					var text string
					if len(resp.Choices) > 0 {
						text = resp.Choices[0].Content
					}
					return tea.Sequence(
						func() tea.Msg { return streamChunkWithStreamMsg{content: text, stream: nil} },
						func() tea.Msg { return streamDoneMsg{} },
					)()
				}
				return streamErrMsg{err: err}
			}
			return streamStartMsg{stream: stream}
		}

		return streamErrMsg{err: fmt.Errorf("no provider initialized")}
	}
}

func receiveStreamCmd(stream provider.Stream) tea.Cmd {
	if stream == nil {
		return func() tea.Msg { return streamDoneMsg{} }
	}
	return func() tea.Msg {
		chunk, err := stream.Recv()
		if err != nil {
			_ = stream.Close()
			if err == io.EOF {
				return streamDoneMsg{}
			}
			return streamErrMsg{err: err}
		}
		return streamChunkWithStreamMsg{content: chunk.Content, stream: stream}
	}
}

func main() {
	_ = godotenv.Load()

	modelFlag := flag.String("model", "", "Model name override")
	serverFlag := flag.Bool("api", false, "Start HTTP API server (don't run TUI)")
	apiAddr := flag.String("api-addr", "", "Address for the HTTP API server (defaults to $PORT or :8080)")
	flag.Parse()

	prov, err := provider.NewClientFromEnv()
	if err != nil {
		log.Fatalf("failed to initialize LLM provider: %v", err)
	}

	selectedModel := *modelFlag
	if selectedModel == "" {
		selectedModel = prov.ModelName()
	}

	if *serverFlag {
		startServer(*apiAddr, prov)
		return
	}

	p := tea.NewProgram(
		initialModel(prov, selectedModel),
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)

	if _, err := p.Run(); err != nil {
		log.Fatalf("Error running program: %v", err)
	}
}

