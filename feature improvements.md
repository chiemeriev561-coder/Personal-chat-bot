## 🚀 Future Improvements
  ### 🔴 High Priority — Core UX gaps

  1. True real-time streaming
  Currently sendStreamCmd collects all chunks first, then replays them. The user sees nothing until the full response is received. True streaming requires
  running the API iterator in a goroutine and using p.Send() to push each streamChunkMsg to the program as it arrives — which needs the *tea.Program reference
  passed in at startup.

  2. Multi-line input
  Enter sends immediately, making it impossible to write multi-line prompts (e.g. pasting code). The fix is: use Alt+Enter or Ctrl+Enter for newlines, and Enter
  alone to send.
  3. Conversation persistence
  The chat history lives only in memory. Closing the app loses everything. Add JSON/SQLite save/load so sessions survive restarts, with a --resume flag.
  ──────
  ### 🟡 Medium Priority — Quality of life

  4. Configurable system prompt
  The system prompt is hardcoded as "expert developer assistant". Let the user pass --persona <file.txt> or pick from built-in personas (coding, writing,
  general).

  5. Copy-to-clipboard on code blocks
  When the assistant returns code, add a keybinding (e.g. c) to copy the last code block directly to clipboard.

  6. /clear command
  A slash command to wipe the current conversation and start fresh without restarting the binary — resets history, currentAi, and creates a new genai.Chat.

  7. Token / cost display
  Show a small status bar with token usage from GenerateContentResponse.UsageMetadata (input tokens, output tokens, cumulative).

  8. Abort in-flight request
  Add Ctrl+X to cancel a streaming response mid-flight using a context.CancelFunc stored on the model.
  ──────
  ### 🟢 Nice to have — Polish

  9. Message timestamps
  Show [HH:MM] next to each turn in the history.

  10. Syntax-highlighted code export
  Ctrl+E to export the full conversation as a .md file to disk.

  11. Model switcher
  Ctrl+M to cycle between models (gemini-3.6-flash, gemini-3.5-flash, etc.) without restarting.

  12. Search in history
  / key to open an inline search bar that highlights matching messages in the viewport.
  ──────