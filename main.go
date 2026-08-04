package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
	"google.golang.org/genai"
)

// Messages for async Bubble Tea updates
type streamChunkMsg string
type streamDoneMsg struct{}
type streamErrMsg struct{ err error }

type model struct {
	client    *genai.Client
	chat      *genai.Chat
	viewport  viewport.Model
	textarea  textarea.Model
	spinner   spinner.Model
	glamour   *glamour.TermRenderer
	history   string
	currentAi string
	isWaiting bool
	width     int
	height    int
}

func initialModel(client *genai.Client, chat *genai.Chat) model {
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

	m := model{
		client:   client,
		chat:     chat,
		textarea: ta,
		viewport: vp,
		spinner:  s,
		glamour:  renderer,
		history:  "# Victor AI GO CLI Chatbot\n*Type your message below. Press Ctrl+C to quit.*\n\n---\n",
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
		footerView = fmt.Sprintf("%s Assistant is thinking...", m.spinner.View())
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

// sendStreamCmd collects all stream chunks from the Gemini API and returns
// them as a tea.Sequence so Bubble Tea dispatches each streamChunkMsg
// individually, re-rendering after every chunk for a live typing effect.
func (m model) sendStreamCmd(input string) tea.Cmd {
	return func() tea.Msg {
		ctx := context.Background()
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
	ctx := context.Background()

	if os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_API_KEY") == "" {
		log.Fatal("missing API key: set GEMINI_API_KEY or GOOGLE_API_KEY before starting")
	}

	client, err := genai.NewClient(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	config := &genai.GenerateContentConfig{
		SystemInstruction: &genai.Content{Parts: []*genai.Part{{
			Text: "You are an expert developer assistant. Provide precise, idiomatic code examples and direct technical answers.",
		}}},
	}

	chat, err := client.Chats.Create(ctx, "gemini-3.6-flash", config, nil)
	if err != nil {
		log.Fatalf("Failed to initialize chat: %v", err)
	}

	p := tea.NewProgram(
		initialModel(client, chat),
		tea.WithAltScreen(),       // Use full terminal buffer
		tea.WithMouseCellMotion(), // Allow mouse scroll
		// Bracketed paste is enabled by default in bubbletea v1.3.10
	)

	if _, err := p.Run(); err != nil {
		log.Fatalf("Error running program: %v", err)
	}
}