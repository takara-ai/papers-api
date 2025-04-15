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
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
	"golang.org/x/net/html"
)

const (
	baseURL       = "https://huggingface.co/papers"
	liveURL       = "https://tldr.takara.ai"
	scrapeTimeout = 30 * time.Second
	llmTimeout    = 90 * time.Second
	maxPapers     = 50
	cacheKey      = "hf_papers_cache"
	summaryCacheKey = "hf_papers_summary_cache"
	cacheDuration = 24 * time.Hour
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
	Title         string    `xml:"title"`
	Link          string    `xml:"link"`
	Description   string    `xml:"description"`
	LastBuildDate string    `xml:"lastBuildDate"`
	AtomLink      AtomLink  `xml:"atom:link"`
	Items         []Item    `xml:"item"`
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
	Model               string    `json:"model"`
	Messages            []Message `json:"messages"`
	MaxTokens           int       `json:"max_tokens"`
	Stream              bool      `json:"stream"`
	StreamOptions       struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	Temperature         float64 `json:"temperature"`
	TopP               float64 `json:"top_p"`
	SeparateReasoning  bool    `json:"separate_reasoning"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

type LLMResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created float64 `json:"created"`
	Model   string `json:"model"`
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
	rdb *redis.Client
	ctx = context.Background()
	redisConnected bool
	initOnce sync.Once
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
				Text:       paper.URL,
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

func initRedis() {
	redisURL := os.Getenv("KV_URL")
	if redisURL == "" {
		return
	}

	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		logger.Error("Error parsing Redis URL", "error", err)
		return
	}

	rdb = redis.NewClient(opt)
	pingErr := rdb.Ping(ctx).Err()
	if pingErr != nil {
		logger.Error("Error connecting to Redis", "error", pingErr)
		return
	}

	redisConnected = true
	logger.Info("Successfully connected to Redis")
}

func getCachedFeed(ctx context.Context, requestURL string) ([]byte, error) {
	if !redisConnected {
		return generateFeedDirect(ctx, requestURL)
	}

	// Try to get from cache first
	cachedData, err := rdb.Get(ctx, cacheKey).Bytes()
	if err == nil {
		return cachedData, nil
	} else if !errors.Is(err, redis.Nil) {
		logger.Warn("Redis Get failed, generating feed directly", "key", cacheKey, "error", err)
	}

	// Cache miss or Redis error, generate new feed
	feed, err := generateFeedDirect(ctx, requestURL)
	if err != nil {
		return nil, fmt.Errorf("failed to generate direct feed: %w", err)
	}

	// Cache the new feed if Redis is connected
	if redisConnected {
		err = rdb.Set(ctx, cacheKey, feed, cacheDuration).Err()
		if err != nil {
			logger.Warn("Failed to cache feed", "key", cacheKey, "error", err)
		}
	}

	return feed, nil
}

func generateFeedDirect(ctx context.Context, requestURL string) ([]byte, error) {
	// Pass context to scrapePapers
	papers, err := scrapePapers(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed scraping papers: %w", err)
	}
	return generateRSS(papers, requestURL)
}

// updateAllCaches generates fresh feed and summary data and updates both caches.
func updateAllCaches(ctx context.Context) error {
	if !redisConnected {
		return fmt.Errorf("redis not connected, cannot update caches")
	}

	logger.Info("Starting cache update for feed and summary")

	// 1. Generate fresh feed data
	// Use baseURL for the canonical cache content's requestURL in generateRSS
	freshFeedBytes, err := generateFeedDirect(ctx, baseURL)
	if err != nil {
		return fmt.Errorf("failed to generate direct feed for cache update: %w", err)
	}

	// 2. Update feed cache
	// Use a separate context for Redis operations if needed, but reqCtx is usually fine
	// Adding a small timeout specifically for Redis Set might be wise.
	err = rdb.Set(ctx, cacheKey, freshFeedBytes, cacheDuration).Err()
	if err != nil {
		// Log the error but continue to attempt summary update if possible
		logger.Error("Failed to update feed cache", "key", cacheKey, "error", err)
		// Decide if this error should prevent summary update (e.g., return err here)
		// For now, we log and continue.
	} else {
		logger.Info("Successfully updated feed cache", "key", cacheKey)
	}


	// --- Summary Update ---

	// 3. Parse the *fresh* feed bytes to markdown
	markdown, err := parseRSSToMarkdown(string(freshFeedBytes))
	if err != nil {
		// If parsing fails, we cannot generate a summary. Log and return error.
		logger.Error("Failed to parse fresh feed to markdown for summary update", "error", err)
		return fmt.Errorf("failed to parse fresh feed to markdown: %w", err)
	}

	// 4. Summarize markdown with LLM
	// Use a context with appropriate timeout for the LLM call
	summaryCtx, cancel := context.WithTimeout(ctx, llmTimeout)
	defer cancel()
	summaryContent, err := summarizeWithLLM(summaryCtx, markdown)
	if err != nil {
		// If LLM fails, log and return error.
		logger.Error("Failed to summarize markdown with LLM for cache update", "error", err)
		return fmt.Errorf("failed to summarize markdown with LLM: %w", err)
	}

	// 5. Generate summary RSS
	// Use baseURL for the canonical requestURL
	summaryRSSBytes, err := generateSummaryRSS(summaryContent, baseURL)
	if err != nil {
		// If summary RSS generation fails, log and return error.
		logger.Error("Failed to generate summary RSS for cache update", "error", err)
		return fmt.Errorf("failed to generate summary RSS: %w", err)
	}

	// 6. Update summary cache
	err = rdb.Set(ctx, summaryCacheKey, summaryRSSBytes, cacheDuration).Err()
	if err != nil {
		// Log the error, but the feed cache might have updated successfully.
		logger.Error("Failed to update summary cache", "key", summaryCacheKey, "error", err)
		// Decide if this should return an overall error.
		// Returning error indicates the full update wasn't successful.
		return fmt.Errorf("failed to update summary cache: %w", err)
	} else {
		logger.Info("Successfully updated summary cache", "key", summaryCacheKey)
	}

	logger.Info("Successfully updated both feed and summary caches")
	return nil
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
		MaxTokens:          4096,
		Stream:             false,
		StreamOptions: struct {
			IncludeUsage bool `json:"include_usage"`
		}{
			IncludeUsage: true,
		},
		Temperature:        0.6,
		TopP:              0.95,
		SeparateReasoning:  true,
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
			Text:       fmt.Sprintf("summary-%s", now.Format("2006-01-02")),
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

// getCachedSummary retrieves the summary from cache or generates it if missed.
// It now accepts a context for Redis operations and summary generation.
func getCachedSummary(ctx context.Context, requestURL string) ([]byte, error) {
	if !redisConnected {
		logger.Warn("Redis not connected, generating summary directly")
		return generateSummaryDirect(ctx, requestURL)
	}

	// Try to get from cache first
	cachedData, err := rdb.Get(ctx, summaryCacheKey).Bytes()
	if err == nil {
		logger.Info("Summary cache hit", "key", summaryCacheKey)
		return cachedData, nil
	} else if !errors.Is(err, redis.Nil) {
		logger.Warn("Redis Get failed for summary, generating summary directly", "key", summaryCacheKey, "error", err)
	}

	// Cache miss or Redis error, generate new summary
	logger.Info("Summary cache miss, generating new summary")
	summary, err := generateSummaryDirect(ctx, requestURL)
	if err != nil {
		return nil, fmt.Errorf("failed to generate summary directly after cache miss: %w", err)
	}

	// Cache the new summary if Redis is connected
	if redisConnected {
		err = rdb.Set(ctx, summaryCacheKey, summary, cacheDuration).Err()
		if err != nil {
			logger.Warn("Failed to cache summary", "key", summaryCacheKey, "error", err)
		} else {
			logger.Info("Successfully cached new summary")
		}
	}

	return summary, nil
}

// generateSummaryDirect generates the summary by getting feed, parsing, and calling LLM.
// It now accepts a context to pass down the call chain.
func generateSummaryDirect(ctx context.Context, requestURL string) ([]byte, error) {
	// Get the feed content, passing context
	// This now correctly uses the feed cache if available, or generates directly.
	feedBytes, err := getCachedFeed(ctx, requestURL)
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

// Handler handles all requests
func Handler(w http.ResponseWriter, r *http.Request) {
	// Initialize Redis on first request (using background context for initialization)
	initOnce.Do(func() {
		initRedis()
	})

	// Get the request context
	reqCtx := r.Context()

	// Remove trailing slash and normalize path
	path := strings.TrimSuffix(r.URL.Path, "/")
	if path == "" {
		path = "/api"  // Normalize empty path to /api
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
				"endpoints":    []string{"/api/feed", "/api/summary"},
				"cache_status": redisConnected,
				"timestamp":    time.Now().UTC().Format(time.RFC3339),
				"version":      "1.0.0",
			}
			
			if err := json.NewEncoder(w).Encode(healthStatus); err != nil {
				http.Error(w, "Error encoding response", http.StatusInternalServerError)
			}
			return

		case "/api/feed":
			// Pass request context to feed retrieval/generation
			feed, err := getCachedFeed(reqCtx, requestURL)
			if err != nil {
				logger.Error("Failed to get cached feed", "error", err)
				http.Error(w, "Error generating feed", http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/rss+xml")
			w.Write(feed)
			return
			
		case "/api/summary":
			// Pass request context to summary retrieval/generation
			summary, err := getCachedSummary(reqCtx, requestURL)
			if err != nil {
				logger.Error("Failed to get cached summary", "error", err)
				http.Error(w, fmt.Sprintf("Error generating summary: %v", err), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/rss+xml")
			w.Write(summary)
			return

		case "/api/update-cache":
			// Check for secret key to prevent unauthorized updates
			secretKey := r.Header.Get("X-Update-Key")
			expectedKey := os.Getenv("UPDATE_KEY")
			
			if expectedKey == "" || secretKey != expectedKey {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}

			// Use request context for the update process.
			// Consider background context if updates are long-running & shouldn't be tied to client connection.
			err := updateAllCaches(reqCtx)
			if err != nil {
				// Use a more specific error message if possible
				logger.Error("Failed to update caches via API", "error", err)
				http.Error(w, fmt.Sprintf("Error updating caches: %v", err), http.StatusInternalServerError)
				return
			}
			
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]string{
				"status":    "Cache updated successfully",
				"timestamp": time.Now().UTC().Format(time.RFC3339),
			})
			return

		default:
			http.NotFound(w, r)
		}
	})(w, r)
}