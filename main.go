package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	if os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_API_KEY") == "" {
		log.Fatal("missing API key: set GEMINI_API_KEY or GOOGLE_API_KEY before starting")
	}

	// The SDK reads GEMINI_API_KEY or GOOGLE_API_KEY from the environment.
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

	fmt.Println("--- Victor AI GO CLI Chatbot (Type 'exit' to quit) ---")
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024), 1024*1024)

	for {
		fmt.Print("\nYou: ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "exit" || input == "quit" {
			break
		}

		fmt.Print("\nAssistant: ")

		iter := chat.SendMessageStream(ctx, genai.Part{Text: input})
		streamFailed := false

		for chunk, err := range iter {
			if err != nil {
				log.Printf("\nError during streaming: %v", err)
				streamFailed = true
				break
			}

			// Render streamed text immediately
			for _, cand := range chunk.Candidates {
				if cand.Content != nil {
					for _, part := range cand.Content.Parts {
						if part.Text != "" {
							fmt.Print(part.Text)
						}
					}
				}
			}
		}
		fmt.Println()
		if streamFailed {
			fmt.Println("Assistant: the request failed; please try again.")
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("Error reading input: %v", err)
	}
}
