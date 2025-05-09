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
	"regexp"
	"strings"
	"sync"
	"time"

	md "github.com/gomarkdown/markdown" // Import markdown library
	"github.com/joho/godotenv"
	"github.com/redis/go-redis/v9"
	"golang.org/x/net/html"
	"github.com/gomarkdown/markdown/parser"
	"github.com/gomarkdown/markdown/ast"
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

// OpenAI API structures
type OpenAIRequest struct {
	Model          string         `json:"model"`
	Input          []OpenAIMessage `json:"input"`
	Text           OpenAIText     `json:"text"`
	Reasoning      map[string]any `json:"reasoning"` // Empty object for now
	Tools          []any          `json:"tools"`     // Empty array for now
	Temperature    float64        `json:"temperature"`
	MaxOutputTokens int            `json:"max_output_tokens"`
	TopP           float64        `json:"top_p"`
	Store          bool           `json:"store"`
}

type OpenAIMessage struct {
	Role    string            `json:"role"`
	Content []OpenAIContentBlock `json:"content"`
}

type OpenAIContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type OpenAIText struct {
	Format OpenAIFormat `json:"format"`
}

type OpenAIFormat struct {
	Type string `json:"type"`
}

// OpenAIResponse represents the top-level response object from /v1/responses
type OpenAIResponse struct {
	ID     string                `json:"id"`
	Object string                `json:"object"`
	Model  string                `json:"model"`
	Output []OpenAIOutputMessage `json:"output"` // Added Output field
	// Other top-level fields like status, usage, etc., can be added if needed
}

// OpenAIOutputMessage represents the message object within the 'output' array
type OpenAIOutputMessage struct {
	ID      string              `json:"id"`
	Type    string              `json:"type"`
	Role    string              `json:"role"`    // Role is here
	Content []OpenAIResponseContent `json:"content"` // Content is here
}

type OpenAIResponseContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
	// Annotations field omitted as it's not needed for extraction
}

var (
	rdb *redis.Client
	ctx = context.Background()
	redisConnected bool
	initOnce sync.Once
	logger = slog.New(slog.NewJSONHandler(os.Stderr, nil))
)

func init() {
	// Load .env file on package initialization
	err := godotenv.Load()
	if err != nil {
		// Log if .env is not found, but don't treat as fatal error
		// Environment variables might be set directly
		logger.Info("Error loading .env file (this is expected if using system env vars)", "error", err)
	}
	initRedis() // Keep Redis init here as well
}

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
	var findPaperDetails func(*html.Node) (title string, href string, found bool)

	// Helper function to find the h3 > a within an article
	findPaperDetails = func(node *html.Node) (string, string, bool) {
		if node.Type == html.ElementNode && node.Data == "h3" {
			for c := node.FirstChild; c != nil; c = c.NextSibling {
				if c.Type == html.ElementNode && c.Data == "a" {
					var href string
					for _, attr := range c.Attr {
						if attr.Key == "href" {
							href = attr.Val
						}
					}
					title := extractText(c)
					if href != "" && title != "" {
						return title, href, true
					}
				}
			}
		}
		// Recursively search children
		for c := node.FirstChild; c != nil; c = c.NextSibling {
			if title, href, found := findPaperDetails(c); found {
				return title, href, true
			}
		}
		return "", "", false
	}


	var crawler func(*html.Node)
	crawler = func(node *html.Node) {
		// Target the <article> tag for each paper
		if node.Type == html.ElementNode && node.Data == "article" {
			
			// Find the title and href within the article
			title, href, found := findPaperDetails(node)

			if found {
				url := fmt.Sprintf("https://huggingface.co%s", href)
				// Scrape abstract using the existing function
				// Note: The abstract page structure might also have changed.
				// This might need adjustment if abstract scraping fails.
				abstract, err := scrapeAbstract(ctx, url)
				if err != nil {
					logger.Error("Failed to extract abstract", "url", url, "error", err)
					abstract = "[Abstract not available]" // Placeholder
				}

				papers = append(papers, Paper{
					Title:    strings.TrimSpace(title),
					URL:      url,
					Abstract: abstract,
					PubDate:  time.Now().UTC(), // Consider parsing date if available in new structure
				})
			} else {
				// Log if details couldn't be found within an article element
				logger.Warn("Could not find paper details (h3 > a) within article element")
			}
		}
		
		// Continue crawling siblings and children
		// Check NextSibling first before FirstChild to avoid redundant crawls within found articles
		// (though findPaperDetails already handles recursion within the article)
		for c := node.FirstChild; c != nil; c = c.NextSibling {
			// Optimization: Limit recursion depth or check node type if performance becomes an issue
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
		logger.Warn("KV_URL not set, Redis connection skipped")
		return // Skip Redis connection if URL is not set
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
	summaryContent, err := summarizeWithLLM(summaryCtx, markdown, extractLinksFromMarkdown(markdown))
	if err != nil {
		// If LLM fails, log and return error.
		logger.Error("Failed to summarize markdown with LLM for cache update", "error", err)
		return fmt.Errorf("failed to summarize markdown with LLM: %w", err)
	}

	// 5. Generate summary RSS
	// Use baseURL for the canonical requestURL
	summaryRSSBytes, err := generateSummaryRSS(summaryContent, baseURL, markdown)
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

// ValidationSeverity represents the severity level of a validation issue
type ValidationSeverity int

const (
	SeverityWarning ValidationSeverity = iota
	SeverityError
)

// ValidationError represents a validation error with details and severity
type ValidationError struct {
	Field    string
	Message  string
	Details  string
	Severity ValidationSeverity
}

func (e ValidationError) Error() string {
	return fmt.Sprintf("validation %s in %s: %s", e.severityString(), e.Field, e.Message)
}

func (e ValidationError) severityString() string {
	if e.Severity == SeverityWarning {
		return "warning"
	}
	return "error"
}

// validateMarkdownStructure checks if the markdown has the required sections and structure
func validateMarkdownStructure(markdown string) error {
	// Create markdown parser with extensions
	extensions := parser.CommonExtensions | parser.AutoHeadingIDs
	p := parser.NewWithExtensions(extensions)
	
	// Parse the markdown into an AST
	doc := p.Parse([]byte(markdown))
	
	foundSections := make(map[string]bool)
	var headlineText string
	var collectingHeadlineContent bool
	var currentSectionTextBuilder strings.Builder
	
	ast.WalkFunc(doc, func(node ast.Node, entering bool) ast.WalkStatus {
		if h, ok := node.(*ast.Heading); ok && h.Level == 2 {
			if entering {
				// If we were collecting for Morning Headline and hit a new H2, finalize.
				if collectingHeadlineContent {
					headlineText = strings.TrimSpace(currentSectionTextBuilder.String())
					collectingHeadlineContent = false // Stop collecting
				}

				var currentHeadingTitle string
				for _, child := range h.Children {
					if t, ok := child.(*ast.Text); ok {
						currentHeadingTitle += string(t.Literal)
					}
				}
				currentHeadingTitle = strings.TrimSpace(currentHeadingTitle)
				foundSections[currentHeadingTitle] = true // Mark this H2 section as found

				if currentHeadingTitle == "Morning Headline" {
					collectingHeadlineContent = true
					currentSectionTextBuilder.Reset() // Reset builder for Morning Headline content
				}
			}
			return ast.GoToNext // Continue to process children of heading if any, then move on
		}

		// If we are collecting content for the "Morning Headline"
		if collectingHeadlineContent && entering {
			if para, ok := node.(*ast.Paragraph); ok {
				var paragraphContent strings.Builder
				for _, child := range para.Children {
					if t, ok := child.(*ast.Text); ok {
						paragraphContent.WriteString(string(t.Literal))
					} else if l, ok := child.(*ast.Link); ok {
						// If we want link text in the headline, extract it
						for _, linkChild := range l.Children {
							if lt, ok := linkChild.(*ast.Text); ok {
								paragraphContent.WriteString(string(lt.Literal))
							}
						}
					}
					// Can add more inline types like Emphasis, Strong if needed
				}
				// Append paragraph content to the headline builder
				if currentSectionTextBuilder.Len() > 0 {
					currentSectionTextBuilder.WriteString(" ") // Add space between paragraphs
				}
				currentSectionTextBuilder.WriteString(paragraphContent.String())
			}
		}
		return ast.GoToNext
	})

	// After the walk, if still collecting (Morning Headline was the last section)
	if collectingHeadlineContent {
		headlineText = strings.TrimSpace(currentSectionTextBuilder.String())
	}
	
	// Check for required sections
	requiredSections := []string{"Morning Headline", "What's New"}
	for _, section := range requiredSections {
		if !foundSections[section] {
			logger.Error("Missing required section",
				"section", section,
				"found_sections", foundSections)
			return ValidationError{
				Field:    "structure",
				Message:  fmt.Sprintf("missing required section: %s", section),
				Details:  markdown,
				Severity: SeverityError,
			}
		}
	}
	
	// Validate headline content
	if headlineText == "" {
		logger.Error("Empty headline content", "details", "Extracted headline string was empty after AST parsing and trimming.")
		return ValidationError{
			Field:    "headline",
			Message:  "headline content is empty",
			Details:  markdown, // Full markdown for context
			Severity: SeverityError,
		}
	}
	
	// Clean up the headline text (collapse multiple spaces to one)
	headlineText = regexp.MustCompile(`\s+`).ReplaceAllString(headlineText, " ")
	
	// Check headline length
	if len(headlineText) > 200 {
		logger.Error("Headline too long",
			"headline", headlineText,
			"length", len(headlineText))
		return ValidationError{
			Field:    "headline",
			Message:  fmt.Sprintf("headline too long: %d characters (limit 200)", len(headlineText)),
			Details:  headlineText,
			Severity: SeverityError,
		}
	}
	
	logger.Debug("Markdown structure validation successful",
		"headline", headlineText,
		"headline_length", len(headlineText))
	
	return nil
}

// Helper function to get minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// normalizeURL standardizes URL format for comparison
func normalizeURL(url string) string {
	original := url
	// Remove trailing slashes
	url = strings.TrimSuffix(url, "/")
	// Convert to lowercase
	url = strings.ToLower(url)
	// Remove any query parameters
	if idx := strings.Index(url, "?"); idx != -1 {
		url = url[:idx]
	}
	// Remove any hash fragments
	if idx := strings.Index(url, "#"); idx != -1 {
		url = url[:idx]
	}
	// Remove any double slashes (except after protocol)
	url = regexp.MustCompile(`([^:])//+`).ReplaceAllString(url, "$1/")
	
	logger.Debug("URL normalization",
		"original", original,
		"normalized", url)
	return url
}

// validateMarkdownLinks checks if all markdown links are properly formatted and contain URLs
func validateMarkdownLinks(markdown string, feedURLs map[string]string) error {
	// Create normalized feed URLs map
	normalizedFeedURLs := make(map[string]string)
	for url := range feedURLs {
		normalized := normalizeURL(url)
		normalizedFeedURLs[normalized] = url
	}

	// Regex to find markdown links: [text](url)
	linkRegex := regexp.MustCompile(`\[([^\]]+)\]\(([^)]+)\)`)
	matches := linkRegex.FindAllStringSubmatch(markdown, -1)

	if len(matches) == 0 {
		return ValidationError{
			Field:    "links",
			Message:  "no markdown links found in summary",
			Details:  markdown,
			Severity: SeverityError,
		}
	}

	var warnings []string
	var errors []string

	for _, match := range matches {
		if len(match) != 3 {
			continue
		}
		text := match[1]
		url := match[2]

		// Validate URL format
		if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
			errors = append(errors, fmt.Sprintf("%s: %s (invalid protocol)", text, url))
			continue
		}

		// Normalize URL for comparison
		normalizedURL := normalizeURL(url)
		
		// Check if normalized URL exists in feed
		if _, exists := normalizedFeedURLs[normalizedURL]; !exists {
			warnings = append(warnings, fmt.Sprintf("%s: %s (not in feed)", text, url))
			continue
		}
	}

	// Log warnings but don't fail validation
	if len(warnings) > 0 {
		logger.Warn("Link validation warnings",
			"warnings", warnings,
			"markdown", markdown)
	}

	// Return error if there are any critical issues
	if len(errors) > 0 {
		return ValidationError{
			Field:    "links",
			Message:  fmt.Sprintf("found %d invalid URLs", len(errors)),
			Details:  fmt.Sprintf("invalid URLs: %v", errors),
			Severity: SeverityError,
		}
	}

	return nil
}

// validateSummaryLength checks if the summary is within reasonable bounds
func validateSummaryLength(markdown string) error {
	// Remove markdown links for word count
	plainText := regexp.MustCompile(`\[([^\]]+)\]\([^)]+\)`).ReplaceAllString(markdown, "$1")
	words := strings.Fields(plainText)
	
	if len(words) > 1000 {
		return ValidationError{
			Field:    "length",
			Message:  fmt.Sprintf("summary too long: %d words", len(words)),
			Details:  fmt.Sprintf("max allowed: 1000, current: %d", len(words)),
			Severity: SeverityError,
		}
	}

	if len(words) < 50 {
		return ValidationError{
			Field:    "length",
			Message:  fmt.Sprintf("summary too short: %d words", len(words)),
			Details:  fmt.Sprintf("min expected: 50, current: %d", len(words)),
			Severity: SeverityError,
		}
	}

	// Log warning if summary is getting close to the limit
	if len(words) > 800 {
		logger.Warn("Summary approaching length limit",
			"word_count", len(words),
			"max_allowed", 1000)
	}

	return nil
}

// validateSummaryContent performs all validations on the summary in parallel
func validateSummaryContent(markdown string, feedURLs map[string]string) error {
	// Create channels for validation results
	errChan := make(chan error, 3)
	var wg sync.WaitGroup

	// Run validations in parallel
	wg.Add(3)
	go func() {
		defer wg.Done()
		if err := validateMarkdownStructure(markdown); err != nil {
			errChan <- err
		}
	}()

	go func() {
		defer wg.Done()
		if err := validateMarkdownLinks(markdown, feedURLs); err != nil {
			errChan <- err
		}
	}()

	go func() {
		defer wg.Done()
		if err := validateSummaryLength(markdown); err != nil {
			errChan <- err
		}
	}()

	// Wait for all validations to complete
	go func() {
		wg.Wait()
		close(errChan)
	}()

	// Collect errors
	var errors []error
	var warnings []error
	for err := range errChan {
		if err != nil {
			if validationErr, ok := err.(ValidationError); ok {
				if validationErr.Severity == SeverityWarning {
					warnings = append(warnings, err)
				} else {
					errors = append(errors, err)
				}
			} else {
				errors = append(errors, err)
			}
		}
	}

	// Log warnings
	for _, warning := range warnings {
		logger.Warn("Summary validation warning",
			"warning", warning,
			"markdown", markdown)
	}

	// Return first error if any exist
	if len(errors) > 0 {
		return fmt.Errorf("validation failed: %v", errors)
	}

	return nil
}

// summarizeWithLLM summarizes the markdown content using the OpenAI API
func summarizeWithLLM(ctx context.Context, markdownContent string, feedURLs map[string]string) (string, error) {
	apiURL := "https://api.openai.com/v1/responses"
	apiKey := os.Getenv("OPENAI_API_KEY")

	if apiKey == "" {
		return "", fmt.Errorf("OPENAI_API_KEY environment variable is not set")
	}

	// Construct the exact prompt as requested
	promptText := `Create a brief morning briefing on these AI research papers, written in a conversational style for busy professionals. Focus on what's new and what it means for businesses and society.
Format the output in markdown:
## Morning Headline
(1 sentence)
## What's New 
(2-3 sentences, written like you're explaining it to a friend over coffee, with citations to papers using the full markdown link format: [Paper Title](URL))

 - Cover all papers in a natural, flowing narrative
 - Group related papers together
 - Include key metrics and outcomes
 - Keep the tone light and engaging

Keep it under 200 words. Start with the most impressive or important paper. Focus on outcomes and implications, not technical details. Write like you're explaining it to a friend over coffee. Do not write a word count.
Do not enclose in a markdown code block, just return the markdown.
Below are the paper abstracts and information in markdown format:

` + markdownContent

	// Construct the OpenAI request body
	request := OpenAIRequest{
		Model: "gpt-4.1-mini", // Use the specified model
		Input: []OpenAIMessage{
			{
				Role: "user",
				Content: []OpenAIContentBlock{
					{
						Type: "input_text",
						Text: promptText,
					},
				},
			},
		},
		Text: OpenAIText{
			Format: OpenAIFormat{
				Type: "text",
			},
		},
		Reasoning:       make(map[string]any), // Empty object
		Tools:           make([]any, 0),      // Empty array
		Temperature:     0.6,
		MaxOutputTokens: 4096, // Renamed from MaxTokens
		TopP:            0.95,
		Store:           true,
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("failed to marshal OpenAI request: %w", err)
	}

	// Create request with context
	req, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(requestBody))
	if err != nil {
		return "", fmt.Errorf("failed to create OpenAI request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey) // Use OpenAI key

	// Create an HTTP client with the LLM timeout
	client := &http.Client{
		Timeout: llmTimeout,
	}
	resp, err := client.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return "", fmt.Errorf("timeout calling OpenAI API: %w", err)
		}
		return "", fmt.Errorf("failed to send request to OpenAI API: %w", err)
	}
	defer resp.Body.Close()

	bodyBytes, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		logger.Error("Failed to read OpenAI response body", "error", readErr)
		// Return specific error about reading the body, but include original status code if not OK
		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("HTTP error %d from OpenAI API and failed to read body: %w", resp.StatusCode, readErr)
		}
		return "", fmt.Errorf("failed to read OpenAI response body: %w", readErr)
	}

	// Log the raw response body for debugging
	// logger.Info("Raw OpenAI API Response Body", "status_code", resp.StatusCode, "body", string(bodyBytes))

	if resp.StatusCode != http.StatusOK {
		// We already logged the body, just return the error
		return "", fmt.Errorf("HTTP error %d from OpenAI API: %s", resp.StatusCode, string(bodyBytes))
	}

	// Decode the single OpenAI response object from the read bytes
	var openAIResp OpenAIResponse // Decode into the struct, not a slice
	if err := json.Unmarshal(bodyBytes, &openAIResp); err != nil { // Use json.Unmarshal with the byte slice
		// Log the body again specifically on decode error
		logger.Error("Failed to decode OpenAI response JSON", "error", err, "raw_body", string(bodyBytes))
		return "", fmt.Errorf("failed to decode OpenAI response: %w", err)
	}

	// Extract the text content from the nested structure
	if len(openAIResp.Output) == 0 || openAIResp.Output[0].Role != "assistant" || len(openAIResp.Output[0].Content) == 0 || openAIResp.Output[0].Content[0].Type != "output_text" {
		// Log the parsed struct for better debugging if validation fails
		logger.Warn("OpenAI response structure unexpected or empty after parsing", "parsedResponse", openAIResp)
		return "", fmt.Errorf("invalid or empty response structure from OpenAI API")
	}

	// Extract the markdown text directly from the nested path
	markdownSummary := openAIResp.Output[0].Content[0].Text

	// Validate the summary content
	if err := validateSummaryContent(markdownSummary, feedURLs); err != nil {
		logger.Error("LLM summary validation failed",
			"error", err,
			"summary", markdownSummary)
		return "", fmt.Errorf("LLM summary validation failed: %w", err)
	}

	logger.Info("Successfully validated LLM summary",
		"summary_length", len(markdownSummary),
		"link_count", len(regexp.MustCompile(`\[([^\]]+)\]\([^)]+\)`).FindAllString(markdownSummary, -1)))

	return markdownSummary, nil
}

// extractLinksFromMarkdown parses the input markdown to find ## [Title](URL) lines
// and returns a map of title -> URL.
func extractLinksFromMarkdown(markdownContent string) map[string]string {
	links := make(map[string]string)
	// Regex to find lines like ## [Title](URL)
	// It captures the title (group 1) and the URL (group 2)
	re := regexp.MustCompile(`(?m)^##\s*\[([^\]]+)\]\(([^)]+)\)$`)
	matches := re.FindAllStringSubmatch(markdownContent, -1)
	for _, match := range matches {
		if len(match) == 3 {
			title := strings.TrimSpace(match[1])
			url := strings.TrimSpace(match[2])
			links[title] = url
			logger.Debug("Extracted link", "title", title, "url", url) // Optional: Debug log
		}
	}
	return links
}

// replacePlaceholdersWithLinks replaces placeholders like [Title] in the summary
// with actual markdown links using the provided title-URL map.
func replacePlaceholdersWithLinks(summaryMarkdown string, links map[string]string) string {
	// Regex to find placeholders like [Title]
	re := regexp.MustCompile(`\[([^\]]+)\]`)
	
	replacedMarkdown := re.ReplaceAllStringFunc(summaryMarkdown, func(match string) string {
		// Extract the title from the match (e.g., "[Title]" -> "Title")
		title := strings.Trim(match, "[]")
		// Look up the URL in the map
		if url, ok := links[title]; ok {
			// If found, return the proper markdown link
			return fmt.Sprintf("[%s](%s)", title, url)
		}
		// If not found, return the original placeholder unchanged
		return match
	})
	
	return replacedMarkdown
}

func generateSummaryRSS(summaryMarkdown string, requestURL string, originalMarkdown string) ([]byte, error) {
	now := time.Now().UTC()

	// 1. Extract links from the original markdown
	paperLinks := extractLinksFromMarkdown(originalMarkdown)

	// 2. Replace placeholders in the summary markdown with actual links
	linkedSummaryMarkdown := replacePlaceholdersWithLinks(summaryMarkdown, paperLinks)

	// 3. Convert the link-replaced markdown to HTML
	htmlBytes := md.ToHTML([]byte(linkedSummaryMarkdown), nil, nil)
	htmlSummary := string(htmlBytes)

	// 4. Wrap the HTML summary in a single div and place it in CDATA
	wrappedHtmlSummary := fmt.Sprintf("<div>%s</div>", htmlSummary)

	item := Item{
		Title:       "AI Research Papers Summary for " + now.Format("January 2, 2006"),
		Link:        liveURL,
		Description: CDATA{Text: wrappedHtmlSummary}, // Use div-wrapped HTML summary
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
	feedBytes, err := getCachedFeed(ctx, requestURL)
	if err != nil {
		return nil, fmt.Errorf("failed to get feed for summary generation: %w", err)
	}
	
	// Convert feed to markdown
	originalMarkdown, err := parseRSSToMarkdown(string(feedBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to parse RSS to originalMarkdown for summary generation: %w", err)
	}

	// Extract URLs from feed for validation
	feedURLs := extractLinksFromMarkdown(originalMarkdown)
	
	// Summarize with LLM, passing context
	summaryMarkdown, err := summarizeWithLLM(ctx, originalMarkdown, feedURLs)
	if err != nil {
		return nil, fmt.Errorf("failed to summarize markdown with LLM: %w", err)
	}
	
	// Use the original requestURL for the summary RSS self-link
	return generateSummaryRSS(summaryMarkdown, requestURL, originalMarkdown)
}

// Handler handles all requests
func Handler(w http.ResponseWriter, r *http.Request) {
	// Initialize Redis on first request (using background context for initialization)
	// initOnce.Do is now redundant as init() handles loading env and redis
	// initOnce.Do(func() {
	// 	initRedis() // initRedis now also handles godotenv loading via init()
	// })

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