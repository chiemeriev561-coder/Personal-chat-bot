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

const defaultGroqModel = "openai/gpt-oss-20b"

// Messages for async Bubble Tea updates
type streamChunkMsg string
type streamDoneMsg struct{}
type streamErrMsg struct{ err error }

type model struct {
	registry       *ProviderRegistry
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

func (m model) cycleModel() model {
	if m.registry == nil {
		return m
	}
	modelsList := m.registry.ListModels()
	if len(modelsList) == 0 {
		return m
	}

	currIndex := -1
	for i, item := range modelsList {
		if id, ok := item["id"].(string); ok && id == m.modelName {
			currIndex = i
			break
		}
	}

	nextIndex := (currIndex + 1) % len(modelsList)
	nextModel, ok := modelsList[nextIndex]["id"].(string)
	if !ok || nextModel == "" {
		return m
	}

	prov, targetModel, err := m.registry.ResolveProvider(nextModel)
	if err != nil {
		m.copyStatus = "Failed to switch model: " + err.Error()
		return m
	}

	m.prov = prov
	m.modelName = targetModel
	m.providerName = "Active"
	if ownedBy, ok := modelsList[nextIndex]["owned_by"].(string); ok {
		m.providerName = strings.Title(ownedBy)
	}
	m.copyStatus = fmt.Sprintf("Switched active model to: %s (%s)", m.modelName, m.providerName)
	return m
}

func initialModel(reg *ProviderRegistry, selectedModel string) model {
	if selectedModel == "" {
		selectedModel = getDefaultModel()
	}

	prov, targetModel, err := reg.ResolveProvider(selectedModel)
	if err != nil {
		log.Fatalf("failed to resolve initial model: %v", err)
	}

	ta := textarea.New()
	ta.Placeholder = "Type a message... (Ctrl+S: send · Ctrl+M: switch model · /model <name> · Esc: normal mode)"
	ta.Focus()
	ta.CharLimit = 1000000
	ta.SetWidth(80)
	ta.SetHeight(5)

	vp := viewport.New(80, 20)

	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))

	systemHeader := fmt.Sprintf("# Victor AI GO CLI Chatbot [%s]\n*Type your message below. Press Ctrl+M or use /model <name> to switch models. Press Ctrl+C to quit.*\n\n---\n", targetModel)

	m := model{
		registry:     reg,
		prov:         prov,
		modelName:    targetModel,
		textarea:     ta,
		viewport:     vp,
		spinner:      s,
		glamour:      newRenderer(80),
		history:      systemHeader,
		providerName: "Active",
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

		case tea.KeyCtrlM:
			m = m.cycleModel()
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

			// CLI Commands for model switching and model listing
			if input == "/models" || input == "/list" {
				m.textarea.Reset()
				modelsList := m.registry.ListModels()
				var sb strings.Builder
				sb.WriteString("**System:** Available models and active providers:\n")
				for _, item := range modelsList {
					id := item["id"].(string)
					owner := item["owned_by"].(string)
					sb.WriteString(fmt.Sprintf("- `%s` (%s)\n", id, owner))
				}
				sb.WriteString("\n*Type `/model <name>` or press `Ctrl+M` to switch models.*\n\n---\n")
				m.history += sb.String()
				m.updateViewport()
				return m, nil
			}

			if strings.HasPrefix(input, "/model ") || strings.HasPrefix(input, "/use ") {
				m.textarea.Reset()
				target := strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(input, "/model "), "/use "))
				prov, targetModel, err := m.registry.ResolveProvider(target)
				if err != nil {
					m.copyStatus = "Failed to switch model: " + err.Error()
					return m, nil
				}
				m.prov = prov
				m.modelName = targetModel
				m.copyStatus = fmt.Sprintf("Switched active model to: %s", targetModel)
				m.history += fmt.Sprintf("*Switched active model to **%s***\n\n---\n", targetModel)
				m.updateViewport()
				return m, nil
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

		m.glamour = newRenderer(msg.Width - 4)
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

func (m *model) updateViewport() {
	fullText := m.history
	if m.currentAi != "" {
		fullText += fmt.Sprintf("**Assistant:**\n%s", m.currentAi)
	}

	if m.glamour == nil {
		m.viewport.SetContent(fullText)
		m.viewport.GotoBottom()
		return
	}

	rendered, err := m.glamour.Render(fullText)
	if err != nil {
		m.viewport.SetContent(fullText)
	} else {
		m.viewport.SetContent(rendered)
	}
	m.viewport.GotoBottom()
}

// sendStreamCmd collects all stream chunks from the active provider
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

	modelFlag := flag.String("model", "", "Model to use ('deepseek-v4-flash', 'gemini', 'groq', 'nvidia', or specific model name)")
	serverFlag := flag.Bool("api", false, "Start HTTP API server (don't run TUI)")
	apiAddr := flag.String("api-addr", ":8080", "Address for the HTTP API server (when --api is set)")
	flag.Parse()

	if *serverFlag {
		// Start HTTP API server and exit
		startServer(*apiAddr)
		return
	}

	reg := NewProviderRegistry()
	if len(reg.providers) == 0 {
		log.Fatal("no AI providers configured: set DEEPSEEK_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY, GROQ_API_KEY, or NVIDIA_API_KEY in .env")
	}

	selectedModel := *modelFlag
	if selectedModel == "" {
		selectedModel = os.Getenv("CHAT_MODEL")
	}

	p := tea.NewProgram(
		initialModel(reg, selectedModel),
		tea.WithAltScreen(),       // Use full terminal buffer
		tea.WithMouseCellMotion(), // Allow mouse scroll
	)

	if _, err := p.Run(); err != nil {
		log.Fatalf("Error running program: %v", err)
	}
}
