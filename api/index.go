package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"golang.org/x/net/html"
)

func init() {
	// Load environment variables from .env file
	if err := godotenv.Load(); err != nil {
		slog.Warn("No .env file found, using environment variables")
	}
}

const (
	baseURL              = "https://huggingface.co/papers"
	liveURL              = "https://tldr.takara.ai"
	scrapeTimeout        = 30 * time.Second
	llmTimeout           = 90 * time.Second
	maxPapers            = 50
	cacheKey             = "hf_papers_cache"
	summaryCacheKey      = "hf_papers_summary_cache"
	conversationCacheKey = "hf_papers_conversation_cache"
	podcastCacheKey      = "hf_papers_podcast_cache"
	cacheDuration        = 24 * time.Hour
)

type Paper struct {
	Title    string
	URL      string
	Abstract string
	PubDate  time.Time
}

type RSS struct {
	XMLName xml.Name `xml:"rss"`
	Version string   `xml:"version,attr"`
	Channel Channel  `xml:"channel"`
	XMLNS   string   `xml:"xmlns:atom,attr"`
}

type Channel struct {
	Title         string   `xml:"title"`
	Link          string   `xml:"link"`
	Description   string   `xml:"description"`
	LastBuildDate string   `xml:"lastBuildDate"`
	AtomLink      AtomLink `xml:"atom:link"`
	Items         []Item   `xml:"item"`
}

type AtomLink struct {
	Href string `xml:"href,attr"`
	Rel  string `xml:"rel,attr"`
	Type string `xml:"type,attr"`
}

type Item struct {
	Title       string `xml:"title"`
	Link        string `xml:"link"`
	Description CDATA  `xml:"description"`
	PubDate     string `xml:"pubDate"`
	GUID        GUID   `xml:"guid"`
}

type GUID struct {
	IsPermaLink bool   `xml:"isPermaLink,attr"`
	Text        string `xml:",chardata"`
}

// CDATA represents CDATA-wrapped content in XML
type CDATA struct {
	Text string `xml:",cdata"`
}

// LLM API structures
type LLMRequest struct {
	Model         string    `json:"model"`
	Messages      []Message `json:"messages"`
	MaxTokens     int       `json:"max_tokens"`
	Stream        bool      `json:"stream"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	Temperature       float64 `json:"temperature"`
	TopP              float64 `json:"top_p"`
	SeparateReasoning bool    `json:"separate_reasoning"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

type LLMResponse struct {
	ID      string  `json:"id"`
	Object  string  `json:"object"`
	Created float64 `json:"created"`
	Model   string  `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role             string `json:"role"`
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		CompletionTokens int `json:"completion_tokens"`
		PromptTokens     int `json:"prompt_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

var (
	ctx    = context.Background()
	logger = slog.New(slog.NewJSONHandler(os.Stderr, nil))
)

func scrapeAbstract(ctx context.Context, url string) (string, error) {
	client := &http.Client{
		Timeout: scrapeTimeout,
	}

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create request for %s: %w", url, err)
	}

	resp, err := client.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return "", fmt.Errorf("timeout fetching abstract from %s: %w", url, err)
		}
		return "", fmt.Errorf("failed to fetch abstract from %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to fetch abstract from %s: status code %d", url, resp.StatusCode)
	}

	doc, err := html.Parse(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to parse HTML from %s: %w", url, err)
	}

	var abstract string
	var found bool
	var crawler func(*html.Node)
	crawler = func(node *html.Node) {
		if found { // Optimization: stop crawling once found
			return
		}
		if node.Type == html.ElementNode && node.Data == "div" {
			for _, attr := range node.Attr {
				if attr.Key == "class" && strings.Contains(attr.Val, "pb-8 pr-4 md:pr-16") {
					abstract = extractText(node)
					found = true
					return
				}
			}
		}
		for c := node.FirstChild; c != nil; c = c.NextSibling {
			crawler(c)
		}
	}
	crawler(doc)

	if !found {
		logger.Warn("Abstract div not found", "class", "pb-8 pr-4 md:pr-16", "url", url)
	}

	abstract = strings.TrimPrefix(abstract, "Abstract")
	abstract = strings.ReplaceAll(abstract, "\n", " ")
	return strings.TrimSpace(abstract), nil
}

func extractText(n *html.Node) string {
	var text string
	if n.Type == html.TextNode {
		return n.Data
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		text += extractText(c)
	}
	return text
}

func scrapePapers(ctx context.Context) ([]Paper, error) {
	client := &http.Client{
		Timeout: scrapeTimeout,
	}

	req, err := http.NewRequestWithContext(ctx, "GET", baseURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request for %s: %w", baseURL, err)
	}

	resp, err := client.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, fmt.Errorf("timeout fetching papers from %s: %w", baseURL, err)
		}
		return nil, fmt.Errorf("failed to fetch papers from %s: %w", baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to fetch papers from %s: status code %d", baseURL, resp.StatusCode)
	}

	doc, err := html.Parse(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to parse HTML from %s: %w", baseURL, err)
	}

	var papers []Paper

	var crawler func(*html.Node)
	crawler = func(node *html.Node) {
		if node.Type == html.ElementNode && node.Data == "h3" {
			var title, href string
			for c := node.FirstChild; c != nil; c = c.NextSibling {
				if c.Type == html.ElementNode && c.Data == "a" {
					for _, attr := range c.Attr {
						if attr.Key == "href" {
							href = attr.Val
						}
					}
					title = extractText(c)
					break
				}
			}

			if href != "" {
				url := fmt.Sprintf("https://huggingface.co%s", href)
				abstract, err := scrapeAbstract(ctx, url)
				if err != nil {
					logger.Error("Failed to extract abstract", "url", url, "error", err)
					abstract = "[Abstract not available]" // Placeholder
				}

				papers = append(papers, Paper{
					Title:    strings.TrimSpace(title),
					URL:      url,
					Abstract: abstract,
					PubDate:  time.Now().UTC(),
				})
			}
		}
		for c := node.FirstChild; c != nil; c = c.NextSibling {
			crawler(c)
		}
	}
	crawler(doc)

	// Limit number of papers
	if len(papers) > maxPapers {
		papers = papers[:maxPapers]
	}

	return papers, nil
}

func generateRSS(papers []Paper, requestURL string) ([]byte, error) {
	items := make([]Item, len(papers))
	for i, paper := range papers {
		items[i] = Item{
			Title:       paper.Title,
			Link:        paper.URL,
			Description: CDATA{Text: paper.Abstract},
			PubDate:     paper.PubDate.Format(time.RFC1123Z),
			GUID: GUID{
				IsPermaLink: true,
				Text:        paper.URL,
			},
		}
	}

	rss := RSS{
		Version: "2.0",
		XMLNS:   "http://www.w3.org/2005/Atom",
		Channel: Channel{
			Title:         "宝の知識: Hugging Face 論文フィード",
			Link:          baseURL,
			Description:   "最先端のAI論文をお届けする、Takara.aiの厳選フィード",
			LastBuildDate: time.Now().UTC().Format(time.RFC1123Z),
			AtomLink: AtomLink{
				Href: requestURL,
				Rel:  "self",
				Type: "application/rss+xml",
			},
			Items: items,
		},
	}

	// Add XML header and proper encoding
	output, err := xml.MarshalIndent(rss, "", "  ")
	if err != nil {
		return nil, err
	}

	// Prepend the XML header
	return append([]byte(xml.Header), output...), nil
}

// Simple CORS middleware
func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		next(w, r)
	}
}

func generateFeed(ctx context.Context, requestURL string) ([]byte, error) {
	// Pass context to scrapePapers
	papers, err := scrapePapers(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed scraping papers: %w", err)
	}
	return generateRSS(papers, requestURL)
}

func parseRSSToMarkdown(xmlContent string) (string, error) {
	var rss RSS
	err := xml.Unmarshal([]byte(xmlContent), &rss)
	if err != nil {
		return "", fmt.Errorf("failed to unmarshal RSS XML: %w", err)
	}

	// Format date
	var formattedDate string
	parsedDate, err := time.Parse(time.RFC1123Z, rss.Channel.LastBuildDate)
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
		markdown.WriteString(fmt.Sprintf("%s\n\n", item.Description.Text))
		markdown.WriteString("---\n\n")
	}

	// logger.Info("Markdown Generated", "markdown", markdown.String())
	return markdown.String(), nil
}

// summarizeWithLLM summarizes the markdown content using Hugging Face Router API
// It now accepts a context for cancellation and timeout, and uses an HTTP client with a timeout.
func summarizeWithLLM(ctx context.Context, markdownContent string) (string, error) {
	apiURL := "https://router.huggingface.co/hf-inference/models/Qwen/Qwen2.5-72B-Instruct/v1/chat/completions"
	apiKey := os.Getenv("HF_API_KEY")

	if apiKey == "" {
		return "", fmt.Errorf("HF_API_KEY environment variable is not set")
	}

	prompt := `Create a brief morning briefing on these AI research papers, written in a conversational style for busy professionals. Focus on what's new and what it means for businesses and society.
Format the output in HTML:
<h2>Morning Headline</h2>
<p>(1 sentence)</p>

<h2>What's New</h2>
<p>(2-3 sentences, written like you're explaining it to a friend over coffee, with citations to papers as <a href="link">Paper Name</a>)</p>
<ul>
  <li>Cover all papers in a natural, flowing narrative</li>
  <li>Group related papers together</li>
  <li>Include key metrics and outcomes</li>
  <li>Keep the tone light and engaging</li>
</ul>

Keep it under 200 words. Start with the most impressive or important paper. Focus on outcomes and implications, not technical details. Write like you're explaining it to a friend over coffee. Do not write a word count.

Do not enclose the HTML in a markdown code block, just return the HTML.

Below are the paper abstracts and information in markdown format:
` + markdownContent

	request := LLMRequest{
		Model: "Qwen/Qwen2.5-72B-Instruct",
		Messages: []Message{
			{
				Role:    "user",
				Content: prompt,
			},
		},
		MaxTokens: 4096,
		Stream:    false,
		StreamOptions: struct {
			IncludeUsage bool `json:"include_usage"`
		}{
			IncludeUsage: true,
		},
		Temperature:       0.6,
		TopP:              0.95,
		SeparateReasoning: true,
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("failed to marshal LLM request: %w", err)
	}

	// Create request with context
	req, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(requestBody))
	if err != nil {
		return "", fmt.Errorf("failed to create LLM request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	// Create an HTTP client with the LLM timeout
	client := &http.Client{
		Timeout: llmTimeout,
	}
	resp, err := client.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return "", fmt.Errorf("timeout calling Hugging Face Router API: %w", err)
		}
		return "", fmt.Errorf("failed to send request to Hugging Face Router API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("HTTP error %d from Hugging Face Router API: %s", resp.StatusCode, string(bodyBytes))
	}

	var llmResp LLMResponse
	if err := json.NewDecoder(resp.Body).Decode(&llmResp); err != nil {
		return "", fmt.Errorf("failed to decode LLM response: %w", err)
	}

	if len(llmResp.Choices) == 0 || llmResp.Choices[0].Message.Content == "" {
		logger.Warn("LLM response contained no choices or empty content", "response", llmResp)
		return "", fmt.Errorf("no valid response content returned from Hugging Face Router API")
	}

	response := llmResp.Choices[0].Message.Content

	// Extract only the content after <think> tags if present
	if strings.Contains(response, "<think>") {
		parts := strings.Split(response, "</think>")
		if len(parts) > 1 {
			response = strings.TrimSpace(parts[len(parts)-1])
		}
	}

	return response, nil
}

func generateSummaryRSS(summary string, requestURL string) ([]byte, error) {
	now := time.Now().UTC()

	// Ensure the summary is properly wrapped in a div for better HTML structure
	summary = fmt.Sprintf("<div>%s</div>", summary)

	item := Item{
		Title:       "AI Research Papers Summary for " + now.Format("January 2, 2006"),
		Link:        liveURL,
		Description: CDATA{Text: summary},
		PubDate:     now.Format(time.RFC1123Z),
		GUID: GUID{
			IsPermaLink: false,
			Text:        fmt.Sprintf("summary-%s", now.Format("2006-01-02")),
		},
	}

	rss := RSS{
		Version: "2.0",
		XMLNS:   "http://www.w3.org/2005/Atom",
		Channel: Channel{
			Title:         "Takara TLDR",
			Link:          liveURL,
			Description:   "Daily summaries of AI research papers from takara.ai",
			LastBuildDate: now.Format(time.RFC1123Z),
			AtomLink: AtomLink{
				Href: requestURL,
				Rel:  "self",
				Type: "application/rss+xml",
			},
			Items: []Item{item},
		},
	}

	// Add XML header and proper encoding
	output, err := xml.MarshalIndent(rss, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to marshal summary RSS: %w", err)
	}

	// Prepend the XML header
	return append([]byte(xml.Header), output...), nil
}

func generateSummary(ctx context.Context, requestURL string) ([]byte, error) {
	// Get the feed content, passing context
	// Directly generate the feed now instead of checking cache
	feedBytes, err := generateFeed(ctx, requestURL)
	if err != nil {
		return nil, fmt.Errorf("failed to get feed for summary generation: %w", err)
	}

	// Convert feed to markdown (no context needed for this part)
	markdown, err := parseRSSToMarkdown(string(feedBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to parse RSS to markdown for summary generation: %w", err)
	}

	// Summarize with LLM, passing context
	summaryContent, err := summarizeWithLLM(ctx, markdown)
	if err != nil {
		return nil, fmt.Errorf("failed to summarize markdown with LLM: %w", err)
	}

	// Use the original requestURL for the summary RSS self-link
	return generateSummaryRSS(summaryContent, requestURL)
}

// Conversation represents the structure of a podcast conversation
type ConversationData struct {
	Conversation []DialogueEntry `json:"conversation"`
}

type DialogueEntry struct {
	Speaker string `json:"speaker"`
	Text    string `json:"text"`
}

func extractConversation(ctx context.Context, text string, maxRetries int) (*ConversationData, error) {
	var lastErr error

	for attempt := 1; attempt <= maxRetries; attempt++ {
		logger.Info("Attempting to generate conversation", "attempt", attempt, "maxRetries", maxRetries)

		// Create a context with timeout for this attempt
		attemptCtx, cancel := context.WithTimeout(ctx, llmTimeout)
		defer cancel()

		conversation, err := tryGenerateConversation(attemptCtx, text)
		if err == nil {
			return conversation, nil
		}

		lastErr = err
		logger.Warn("Conversation generation attempt failed",
			"attempt", attempt,
			"error", err,
			"remainingRetries", maxRetries-attempt)

		if attempt < maxRetries {
			// Exponential backoff with jitter
			backoff := time.Duration(attempt*2) * time.Second
			jitter := time.Duration(rand.Int63n(1000)) * time.Millisecond
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("context cancelled during retry wait: %w", ctx.Err())
			case <-time.After(backoff + jitter):
				continue
			}
		}
	}

	return nil, fmt.Errorf("failed to generate conversation after %d attempts: %w", maxRetries, lastErr)
}

func tryGenerateConversation(ctx context.Context, text string) (*ConversationData, error) {

	apiURL := "https://router.huggingface.co/sambanova/v1/chat/completions"
	apiKey := os.Getenv("HF_API_KEY")

	if apiKey == "" {
		return nil, fmt.Errorf("HF_API_KEY environment variable is not set")
	}

	prompt := fmt.Sprintf(`Welcome to Daily Papers! Today, we're diving into the latest AI research in an engaging and 
        informative discussion. The goal is to make it a **bite-sized podcast** that's **engaging, natural, and insightful** while covering 
        the key points of each paper.

        Here are today's research papers:
        %s

        Convert this into a **conversational podcast-style discussion** between two experts, Brian and Jenny. 
        Ensure the conversation:
        1. Flows naturally with realistic back-and-forth dialogue
        2. Uses casual phrasing and occasional filler words (like "um", "you know")
        3. Maintains professional insights while being engaging
        4. Covers each paper meaningfully but concisely
        5. Focuses on practical implications and key findings
        6. Keeps a dynamic pace with natural transitions
		7. Avoid's Host calling each other by name, just "you" and "I".

        Return the conversation in this exact JSON format:
        {
            "conversation": [
                {"speaker": "Brian", "text": ""},
                {"speaker": "Jenny", "text": ""}
            ]
        }`, text)

	request := LLMRequest{
		Model: "Qwen2.5-72B-Instruct",
		Messages: []Message{
			{
				Role:    "user",
				Content: prompt,
			},
		},
		MaxTokens:   4096,
		Temperature: 0.7,
		TopP:        0.95,
		Stream:      false,
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{Timeout: llmTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var llmResp LLMResponse
	if err := json.NewDecoder(resp.Body).Decode(&llmResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(llmResp.Choices) == 0 || llmResp.Choices[0].Message.Content == "" {
		return nil, fmt.Errorf("no valid content in response")
	}

	content := llmResp.Choices[0].Message.Content

	// Extract JSON using regex if needed
	re := regexp.MustCompile(`\{(?:[^{}]|(?:\{[^{}]*\}))*\}`)
	match := re.FindString(content)
	if match == "" {
		return nil, fmt.Errorf("no valid JSON found in response")
	}

	var conversation ConversationData
	if err := json.Unmarshal([]byte(match), &conversation); err != nil {
		return nil, fmt.Errorf("failed to parse conversation JSON: %w", err)
	}

	// Validate conversation structure
	if len(conversation.Conversation) == 0 {
		return nil, fmt.Errorf("parsed JSON contains no conversation entries")
	}

	return &conversation, nil
}

func generatePodcastConversation(ctx context.Context, text string) (string, error) {
	conversation, err := extractConversation(ctx, text, 3)
	if err != nil {
		return "", fmt.Errorf("failed to extract conversation: %w", err)
	}

	// Convert back to JSON string
	result, err := json.MarshalIndent(conversation, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal conversation: %w", err)
	}

	// Removed Redis caching logic for conversation
	/*
		if redisConnected {
			err = rdb.Set(ctx, conversationCacheKey, result, cacheDuration).Err()
			if err != nil {
				logger.Warn("Failed to cache conversation", "key", conversationCacheKey, "error", err)
			} else {
				logger.Info("Successfully cached new conversation")
			}
		}
	*/

	return string(result), nil
}

func generateaudiopodcast(ctx context.Context, text string) ([]byte, error) {
	// Parse the conversation JSON
	var conversation ConversationData
	if err := json.Unmarshal([]byte(text), &conversation); err != nil {
		return nil, fmt.Errorf("failed to parse conversation: %w", err)
	}

	apiKey := os.Getenv("DEEPINFRA_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("DEEPINFRA_API_KEY environment variable is not set")
	}

	url := "https://api.deepinfra.com/v1/openai/audio/speech"

	// Create a buffer to store the audio data
	var audioBuffer bytes.Buffer

	// Process each dialogue entry
	for _, entry := range conversation.Conversation {
		voice := "af_bella"
		if entry.Speaker == "Jenny" {
			voice = "af_bella"
		} else if entry.Speaker == "Brian" {
			voice = "am_michael"
		}

		// Prepare request body
		requestBody := map[string]interface{}{
			"model":           "hexgrad/Kokoro-82M",
			"input":           entry.Text,
			"voice":           voice,
			"response_format": "mp3",
		}

		jsonBody, err := json.Marshal(requestBody)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}

		// Create request
		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonBody))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		// Set headers
		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("Content-Type", "application/json")

		// Make request
		client := &http.Client{}
		resp, err := client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to make request: %w", err)
		}
		defer resp.Body.Close()

		// Check response status
		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(bodyBytes))
		}

		// Write the audio data to buffer
		_, err = io.Copy(&audioBuffer, resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to write audio data: %w", err)
		}
	}

	return audioBuffer.Bytes(), nil
}

// Helper function to calculate seconds until next 6 AM UTC
func calculateSecondsUntilNext6AMUTC() int64 {
	nowUTC := time.Now().UTC()
	next6AMUTC := time.Date(nowUTC.Year(), nowUTC.Month(), nowUTC.Day(), 6, 0, 0, 0, time.UTC)

	if nowUTC.After(next6AMUTC) {
		// If current time is past 6 AM today, the next 6 AM is tomorrow
		next6AMUTC = next6AMUTC.Add(24 * time.Hour)
	}

	durationUntilNext6AM := next6AMUTC.Sub(nowUTC)
	seconds := int64(durationUntilNext6AM.Seconds())

	// Ensure minimum cache duration (e.g., 1 minute) if calculation is very close to 6 AM
	if seconds < 60 {
		return 60
	}
	return seconds
}

// Handler handles all requests
func Handler(w http.ResponseWriter, r *http.Request) {
	// Get the request context
	reqCtx := r.Context()

	// Remove trailing slash and normalize path
	path := strings.TrimSuffix(r.URL.Path, "/")
	if path == "" {
		path = "/api" // Normalize empty path to /api
	}

	// Get the full request URL for self-referential links
	requestURL := "https://" + r.Host + r.URL.Path

	// Apply CORS middleware
	corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
		switch path {
		case "/api":
			// Health check endpoint
			w.Header().Set("Content-Type", "application/json")
			healthStatus := map[string]interface{}{
				"status":       "ok",
				"endpoints":    []string{"/api/feed", "/api/summary", "/api/conversation", "/api/podcast"},
				"timestamp":    time.Now().UTC().Format(time.RFC3339),
				"version":      "1.1.0",
			}

			if err := json.NewEncoder(w).Encode(healthStatus); err != nil {
				http.Error(w, "Error encoding response", http.StatusInternalServerError)
			}
			return

		case "/api/feed":
			// Directly generate feed
			feed, err := generateFeed(reqCtx, requestURL)
			if err != nil {
				logger.Error("Failed to generate feed", "error", err)
				http.Error(w, "Error generating feed", http.StatusInternalServerError)
				return
			}

			// Set Vercel Edge Cache headers
			secondsToCache := calculateSecondsUntilNext6AMUTC()
			w.Header().Set("Cache-Control", fmt.Sprintf("public, max-age=0, s-maxage=%d", secondsToCache))
			w.Header().Set("Content-Type", "application/rss+xml")
			w.Write(feed)
			return

		case "/api/summary":
			// Directly generate summary
			summary, err := generateSummary(reqCtx, requestURL)
			if err != nil {
				logger.Error("Failed to generate summary", "error", err)
				http.Error(w, fmt.Sprintf("Error generating summary: %v", err), http.StatusInternalServerError)
				return
			}

			// Set Vercel Edge Cache headers
			secondsToCache := calculateSecondsUntilNext6AMUTC()
			w.Header().Set("Cache-Control", fmt.Sprintf("public, max-age=0, s-maxage=%d", secondsToCache))
			w.Header().Set("Content-Type", "application/rss+xml")
			w.Write(summary)
			return

		case "/api/conversation":
			// Get summary first (which will now be generated directly if not edge-cached)
			summaryBytes, err := generateSummary(reqCtx, requestURL)
			if err != nil {
				logger.Error("Failed to generate summary for conversation", "error", err)
				http.Error(w, fmt.Sprintf("Error generating summary for conversation: %v", err), http.StatusInternalServerError)
				return
			}

			// Generate podcast conversation directly
			conversation, err := generatePodcastConversation(reqCtx, string(summaryBytes))
			if err != nil {
				logger.Error("Failed to generate podcast conversation", "error", err)
				http.Error(w, fmt.Sprintf("Error generating podcast conversation: %v", err), http.StatusInternalServerError)
				return
			}

			// Set Vercel Edge Cache headers
			secondsToCache := calculateSecondsUntilNext6AMUTC()
			w.Header().Set("Cache-Control", fmt.Sprintf("public, max-age=0, s-maxage=%d", secondsToCache))
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(conversation))
			return

		case "/api/podcast":
			// Get summary first
			summaryBytes, err := generateSummary(reqCtx, requestURL)
			if err != nil {
				logger.Error("Failed to generate summary for podcast", "error", err)
				http.Error(w, fmt.Sprintf("Error generating summary for podcast: %v", err), http.StatusInternalServerError)
				return
			}

			// Generate conversation directly (needed for audio generation)
			conversation, err := generatePodcastConversation(reqCtx, string(summaryBytes))
			if err != nil {
				logger.Error("Failed to generate conversation for podcast", "error", err)
				http.Error(w, fmt.Sprintf("Error generating conversation for podcast: %v", err), http.StatusInternalServerError)
				return
			}

			// Generate audio podcast directly
			audioData, err := generateaudiopodcast(reqCtx, conversation)
			if err != nil {
				logger.Error("Failed to generate podcast audio", "error", err)
				http.Error(w, fmt.Sprintf("Error generating podcast audio: %v", err), http.StatusInternalServerError)
				return
			}

			// Set Vercel Edge Cache headers
			secondsToCache := calculateSecondsUntilNext6AMUTC()
			w.Header().Set("Cache-Control", fmt.Sprintf("public, max-age=0, s-maxage=%d", secondsToCache))
			w.Header().Set("Content-Type", "audio/mpeg")
			w.Header().Set("Content-Disposition", "inline; filename=\"daily-papers-podcast.mp3\"")
			w.Header().Set("Content-Length", fmt.Sprintf("%d", len(audioData)))
			w.Header().Set("Accept-Ranges", "bytes")

			if _, err := w.Write(audioData); err != nil {
				logger.Error("Failed to write audio response", "error", err)
			}
			return

		default:
			http.NotFound(w, r)
		}
	})(w, r)
}
