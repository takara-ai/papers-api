package main

import (
	"bytes"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// RSS structure definitions
type RSS struct {
	XMLName xml.Name `xml:"rss"`
	Channel Channel  `xml:"channel"`
}

type Channel struct {
	Title         string `xml:"title"`
	Description   string `xml:"description"`
	LastBuildDate string `xml:"lastBuildDate"`
	Items         []Item `xml:"item"`
}

type Item struct {
	Title       string `xml:"title"`
	Link        string `xml:"link"`
	Description string `xml:"description"`
}

// Groq API structures
type GroqRequest struct {
	Model               string    `json:"model"`
	Messages            []Message `json:"messages"`
	Temperature         float64   `json:"temperature"`
	MaxCompletionTokens int       `json:"max_completion_tokens"`
	TopP                float64   `json:"top_p"`
	Stream              bool      `json:"stream"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type GroqResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		LogProbs     interface{} `json:"logprobs"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		QueueTime       float64 `json:"queue_time"`
		PromptTokens    int     `json:"prompt_tokens"`
		PromptTime      float64 `json:"prompt_time"`
		CompletionTokens int    `json:"completion_tokens"`
		CompletionTime   float64 `json:"completion_time"`
		TotalTokens      int    `json:"total_tokens"`
		TotalTime        float64 `json:"total_time"`
	} `json:"usage"`
	SystemFingerprint string `json:"system_fingerprint"`
	XGroq             struct {
		ID string `json:"id"`
	} `json:"x_groq"`
}

// fetchRSS fetches RSS feed content from a URL
func fetchRSS(url string) (string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP error: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	return string(body), nil
}

// parseRSSToMarkdown parses RSS XML content and converts to markdown format
func parseRSSToMarkdown(xmlContent string) (string, error) {
	var rss RSS
	err := xml.Unmarshal([]byte(xmlContent), &rss)
	if err != nil {
		return "", err
	}

	// Format date
	var formattedDate string
	parsedDate, err := time.Parse("Mon, 02 Jan 2006 15:04:05 -0700", rss.Channel.LastBuildDate)
	if err != nil {
		formattedDate = rss.Channel.LastBuildDate // Fallback to original format
	} else {
		formattedDate = parsedDate.Format("2006-01-02")
	}

	// Create markdown
	var markdown strings.Builder

	markdown.WriteString(fmt.Sprintf("# %s\n\n", rss.Channel.Title))
	markdown.WriteString(fmt.Sprintf("*%s*\n\n", rss.Channel.Description))
	markdown.WriteString(fmt.Sprintf("*Last updated: %s*\n\n", formattedDate))
	markdown.WriteString("---\n\n")

	// Process each item
	for _, item := range rss.Channel.Items {
		title := strings.ReplaceAll(item.Title, "\n", " ")
		title = strings.TrimSpace(title)
		
		markdown.WriteString(fmt.Sprintf("## [%s](%s)\n\n", title, item.Link))
		markdown.WriteString(fmt.Sprintf("%s\n\n", item.Description))
		markdown.WriteString("---\n\n")
	}

	return markdown.String(), nil
}

// summarizeWithGroq summarizes the markdown content using Groq API
func summarizeWithGroq(markdownContent, apiKey string) (string, error) {
	groqURL := "https://api.groq.com/openai/v1/chat/completions"

	prompt := `Create a brief morning briefing on these AI research papers, written in a conversational style for busy professionals. Focus on what's new and what it means for businesses and society.
Format:
1. Morning Headline (1 sentence)
2. What's New (2-3 sentences, written like you're explaining it to a friend over coffee, with citations to papers as [Paper Name](link))
 • Cover all papers in a natural, flowing narrative
 • Group related papers together
 • Include key metrics and outcomes
 • Keep the tone light and engaging
Keep it under 200 words. Focus on outcomes and implications, not technical details. Write like you're explaining it to a friend over coffee.
Below are the paper abstracts and information in markdown format:
` + markdownContent

	request := GroqRequest{
		Model: "deepseek-r1-distill-llama-70b",
		Messages: []Message{
			{
				Role:    "user",
				Content: prompt,
			},
		},
		Temperature:         0.6,
		MaxCompletionTokens: 4096,
		TopP:                0.95,
		Stream:              false,
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", groqURL, bytes.NewBuffer(requestBody))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP error from Groq API: %d", resp.StatusCode)
	}

	var groqResp GroqResponse
	if err := json.NewDecoder(resp.Body).Decode(&groqResp); err != nil {
		return "", err
	}

	if len(groqResp.Choices) == 0 {
		return "", fmt.Errorf("no response choices returned from Groq API")
	}

	response := groqResp.Choices[0].Message.Content

	// Extract only the content after <think> tags if present
	if strings.Contains(response, "<think>") {
		parts := strings.Split(response, "</think>")
		if len(parts) > 1 {
			response = strings.TrimSpace(parts[len(parts)-1])
		}
	}

	return response, nil
}

func main() {
	rssURL := "https://papers.takara.ai/api/feed"
	outputFile := "papers_summary.md"
	apiKey := "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"

	if apiKey == "" {
		fmt.Println("Error: GROQ_API_KEY environment variable is not set")
		os.Exit(1)
	}

	// Fetch RSS
	xmlContent, err := fetchRSS(rssURL)
	if err != nil {
		fmt.Printf("Error fetching RSS: %v\n", err)
		os.Exit(1)
	}

	// Parse to markdown
	markdown, err := parseRSSToMarkdown(xmlContent)
	if err != nil {
		fmt.Printf("Error parsing RSS to markdown: %v\n", err)
		os.Exit(1)
	}

	// Summarize with Groq
	summary, err := summarizeWithGroq(markdown, apiKey)
	if err != nil {
		fmt.Printf("Error summarizing with Groq: %v\n", err)
		os.Exit(1)
	}

	// Write to file
	err = os.WriteFile(outputFile, []byte(summary), 0644)
	if err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Summarization successful! Output saved to %s\n", outputFile)
}