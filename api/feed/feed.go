package feed

import (
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"regexp"
	"time"

	md "github.com/gomarkdown/markdown"
	"github.com/redis/go-redis/v9"

	constants "hf-papers-rss/api/constants"
	markdown "hf-papers-rss/api/markdown"
	models "hf-papers-rss/api/models"
	scraper "hf-papers-rss/api/scraper"
)

// GenerateRSS creates an RSS feed from a list of papers
func GenerateRSS(papers []models.Paper, requestURL string) ([]byte, error) {
	items := make([]models.Item, len(papers))
	for i, paper := range papers {
		items[i] = models.Item{
			Title:       paper.Title,
			Link:        paper.URL,
			Description: models.CDATA{Text: paper.Abstract},
			PubDate:     paper.PubDate.Format(time.RFC1123Z),
			GUID: models.GUID{
				IsPermaLink: true,
				Text:        paper.URL,
			},
		}
	}

	rss := models.RSS{
		Version: "2.0",
		XMLNS:   "http://www.w3.org/2005/Atom",
		Channel: models.Channel{
			Title:         "宝の知識: Hugging Face 論文フィード",
			Link:          constants.BaseURL,
			Description:   "最先端のAI論文をお届けする、Takara.aiの厳選フィード",
			LastBuildDate: time.Now().UTC().Format(time.RFC1123Z),
			AtomLink: models.AtomLink{
				Href: requestURL,
				Rel:  "self",
				Type: "application/rss+xml",
			},
			Items: items,
		},
	}

	output, err := xml.MarshalIndent(rss, "", "  ")
	if err != nil {
		return nil, err
	}

	return append([]byte(xml.Header), output...), nil
}

// GenerateFeedDirect generates a feed directly without caching
func GenerateFeedDirect(ctx context.Context, requestURL string) ([]byte, error) {
	papers, err := scraper.ScrapePapers(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed scraping papers: %w", err)
	}
	return GenerateRSS(papers, requestURL)
}

// GenerateSummaryRSS creates an RSS feed for the summary
func GenerateSummaryRSS(summaryMarkdown string, requestURL string, originalMarkdown string) ([]byte, error) {
	now := time.Now().UTC()

	paperLinks := markdown.ExtractLinksFromMarkdown(originalMarkdown)
	linkedSummaryMarkdown := markdown.ReplacePlaceholdersWithLinks(summaryMarkdown, paperLinks)
	htmlBytes := md.ToHTML([]byte(linkedSummaryMarkdown), nil, nil)
	htmlSummary := string(htmlBytes)
	wrappedHtmlSummary := fmt.Sprintf("<div>%s</div>", htmlSummary)

	item := models.Item{
		Title:       "AI Research Papers Summary for " + now.Format("January 2, 2006"),
		Link:        constants.LiveURL,
		Description: models.CDATA{Text: wrappedHtmlSummary},
		PubDate:     now.Format(time.RFC1123Z),
		GUID: models.GUID{
			IsPermaLink: false,
			Text:        fmt.Sprintf("summary-%s", now.Format("2006-01-02")),
		},
	}

	rss := models.RSS{
		Version: "2.0",
		XMLNS:   "http://www.w3.org/2005/Atom",
		Channel: models.Channel{
			Title:         "Takara TLDR",
			Link:          constants.LiveURL,
			Description:   "Daily summaries of AI research papers from takara.ai",
			LastBuildDate: now.Format(time.RFC1123Z),
			AtomLink: models.AtomLink{
				Href: requestURL,
				Rel:  "self",
				Type: "application/rss+xml",
			},
			Items: []models.Item{item},
		},
	}

	output, err := xml.MarshalIndent(rss, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to marshal summary RSS: %w", err)
	}

	return append([]byte(xml.Header), output...), nil
}

// GenerateSummaryDirect generates a summary directly without caching
func GenerateSummaryDirect(ctx context.Context, requestURL string) ([]byte, error) {
	feedBytes, err := GetCachedFeed(ctx, requestURL)
	if err != nil {
		return nil, fmt.Errorf("failed to get feed for summary generation: %w", err)
	}

	originalMarkdown, err := markdown.ParseRSSToMarkdown(string(feedBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to parse RSS to originalMarkdown for summary generation: %w", err)
	}

	feedURLs := markdown.ExtractLinksFromMarkdown(originalMarkdown)
	summaryMarkdown, err := SummarizeWithLLM(ctx, originalMarkdown, feedURLs)
	if err != nil {
		return nil, fmt.Errorf("failed to summarize markdown with LLM: %w", err)
	}

	return GenerateSummaryRSS(summaryMarkdown, requestURL, originalMarkdown)
}

// SummarizeWithLLM generates a summary using the LLM
func SummarizeWithLLM(ctx context.Context, markdownContent string, feedURLs map[string]string) (string, error) {
	apiURL := "https://api.openai.com/v1/responses"
	apiKey := os.Getenv("OPENAI_API_KEY")

	if apiKey == "" {
		return "", fmt.Errorf("OPENAI_API_KEY environment variable is not set")
	}

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

	request := models.OpenAIRequest{
		Model: "gpt-4.1-mini",
		Input: []models.OpenAIMessage{
			{
				Role: "user",
				Content: []models.OpenAIContentBlock{
					{
						Type: "input_text",
						Text: promptText,
					},
				},
			},
		},
		Text: models.OpenAIText{
			Format: models.OpenAIFormat{
				Type: "text",
			},
		},
		Reasoning:       make(map[string]any),
		Tools:           make([]any, 0),
		Temperature:     0.6,
		MaxOutputTokens: 4096,
		TopP:            0.95,
		Store:           true,
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("failed to marshal OpenAI request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(requestBody))
	if err != nil {
		return "", fmt.Errorf("failed to create OpenAI request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{
		Timeout: constants.LlmTimeout,
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
		constants.Logger.Error("Failed to read OpenAI response body", "error", readErr)
		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("HTTP error %d from OpenAI API and failed to read body: %w", resp.StatusCode, readErr)
		}
		return "", fmt.Errorf("failed to read OpenAI response body: %w", readErr)
	}

	var response models.OpenAIResponse
	if err := json.Unmarshal(bodyBytes, &response); err != nil {
		return "", fmt.Errorf("failed to unmarshal OpenAI response: %w", err)
	}

	if len(response.Output) == 0 {
		return "", fmt.Errorf("empty response from OpenAI API")
	}

	return response.Output[0].Content[0].Text, nil
}

// GetCachedFeed retrieves the feed from cache or generates it if missed
func GetCachedFeed(ctx context.Context, requestURL string) ([]byte, error) {
	if !constants.RedisConnected {
		return GenerateFeedDirect(ctx, requestURL)
	}

	cachedData, err := constants.RDB.Get(ctx, constants.CacheKey).Bytes()
	if err == nil {
		return cachedData, nil
	} else if !errors.Is(err, redis.Nil) {
		constants.Logger.Warn("Redis Get failed, generating feed directly", "key", constants.CacheKey, "error", err)
	}

	feed, err := GenerateFeedDirect(ctx, requestURL)
	if err != nil {
		return nil, fmt.Errorf("failed to generate direct feed: %w", err)
	}

	if constants.RedisConnected {
		err = constants.RDB.Set(ctx, constants.CacheKey, feed, constants.CacheDuration).Err()
		if err != nil {
			constants.Logger.Warn("Failed to cache feed", "key", constants.CacheKey, "error", err)
		}
	}

	return feed, nil
}

// GetCachedSummary retrieves the summary from cache or generates it if missed
func GetCachedSummary(ctx context.Context, requestURL string) ([]byte, error) {
	if !constants.RedisConnected {
		constants.Logger.Warn("Redis not connected, generating summary directly")
		return GenerateSummaryDirect(ctx, requestURL)
	}

	cachedData, err := constants.RDB.Get(ctx, constants.SummaryCacheKey).Bytes()
	if err == nil {
		constants.Logger.Info("Summary cache hit", "key", constants.SummaryCacheKey)
		return cachedData, nil
	} else if !errors.Is(err, redis.Nil) {
		constants.Logger.Warn("Redis Get failed for summary, generating summary directly", "key", constants.SummaryCacheKey, "error", err)
	}

	constants.Logger.Info("Summary cache miss, generating new summary")
	summary, err := GenerateSummaryDirect(ctx, requestURL)
	if err != nil {
		return nil, fmt.Errorf("failed to generate summary directly after cache miss: %w", err)
	}

	if constants.RedisConnected {
		err = constants.RDB.Set(ctx, constants.SummaryCacheKey, summary, constants.CacheDuration).Err()
		if err != nil {
			constants.Logger.Warn("Failed to cache summary", "key", constants.SummaryCacheKey, "error", err)
		} else {
			constants.Logger.Info("Successfully cached new summary")
		}
	}

	return summary, nil
}

func extractConversation(ctx context.Context, text string, maxRetries int) (*models.ConversationData, error) {
	var lastErr error

	for attempt := 1; attempt <= maxRetries; attempt++ {
		constants.Logger.Info("Attempting to generate conversation", "attempt", attempt, "maxRetries", maxRetries)

		// Create a context with timeout for this attempt
		attemptCtx, cancel := context.WithTimeout(ctx, constants.LlmTimeout)
		defer cancel()

		conversation, err := tryGenerateConversation(attemptCtx, text)
		if err == nil {
			return conversation, nil
		}

		lastErr = err
		constants.Logger.Warn("Conversation generation attempt failed",
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

func tryGenerateConversation(ctx context.Context, text string) (*models.ConversationData, error) {
	apiURL := "https://api.openai.com/v1/responses"
	apiKey := os.Getenv("OPENAI_API_KEY")

	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable is not set")
	}

	prompt := fmt.Sprintf(`Welcome to Daily Papers! Today, we're diving into the latest AI research through a short, engaging, 
	and insightful podcast-style conversation.
	
	Here are today's research papers:
	%s
	
	Convert this into a **brief, conversational dialogue** between two AI experts, Brian and Jenny. 
	
	Requirements:
	1. Use a natural back-and-forth tone, like a podcast—professional but casual.
	2. Include casual phrasing and occasional filler words (e.g., "you know", "kind of").
	3. Each speaker's response should be short—ideally 2-3 sentences.
	4. Cover **all papers concisely**, focusing on **practical takeaways** and **key insights**.
	5. Keep a **dynamic pace** with smooth transitions—no long monologues.
	6. Avoid name references in dialogue—use "you" and "I" only.
	7. The tone should be relaxed, clear, and easy to follow.
	
	Return in this exact JSON format:
	{
		"conversation": [
			{"speaker": "Brian", "text": ""},
			{"speaker": "Jenny", "text": ""}
		]
	}`, text)

	request := models.OpenAIRequest{
		Model: "gpt-4o-mini",
		Input: []models.OpenAIMessage{
			{
				Role: "user",
				Content: []models.OpenAIContentBlock{
					{
						Type: "input_text",
						Text: prompt,
					},
				},
			},
		},
		Text: models.OpenAIText{
			Format: models.OpenAIFormat{
				Type: "json_object",
			},
		},
		Reasoning:       make(map[string]any),
		Tools:           make([]any, 0),
		Temperature:     0.6,
		MaxOutputTokens: 4096,
		TopP:            0.95,
		Store:           true,
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

	client := &http.Client{Timeout: constants.LlmTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var llmResp models.OpenAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&llmResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(llmResp.Output) == 0 || llmResp.Output[0].Content[0].Text == "" {
		return nil, fmt.Errorf("no valid content in response")
	}

	content := llmResp.Output[0].Content[0].Text

	// Extract JSON using regex if needed
	re := regexp.MustCompile(`\{(?:[^{}]|(?:\{[^{}]*\}))*\}`)
	match := re.FindString(content)
	if match == "" {
		return nil, fmt.Errorf("no valid JSON found in response")
	}

	var conversation models.ConversationData
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

	if constants.RedisConnected {
		err = constants.RDB.Set(ctx, constants.ConversationCacheKey, result, constants.CacheDuration).Err()
		if err != nil {
			constants.Logger.Warn("Failed to cache conversation", "key", constants.ConversationCacheKey, "error", err)
		} else {
			constants.Logger.Info("Successfully cached new conversation")
		}
	}

	return string(result), nil
}

func Getcachedconversation(ctx context.Context, text string) (string, error) {
	// Check if Redis is connected
	if constants.RedisConnected {
		cachedData, err := constants.RDB.Get(ctx, constants.ConversationCacheKey).Bytes()
		if err == nil {
			constants.Logger.Info("Conversation cache hit", "key", constants.ConversationCacheKey)
			return string(cachedData), nil
		} else if !errors.Is(err, redis.Nil) {
			constants.Logger.Warn("Redis Get failed for conversation, generating conversation directly", "key", constants.ConversationCacheKey, "error", err)
		} else {
			constants.Logger.Info("Conversation cache miss, generating new conversation")
		}
	}

	conversation, err := generatePodcastConversation(ctx, text)
	if err != nil {
		return "", fmt.Errorf("failed to generate podcast conversation: %w", err)
	}

	return conversation, nil
}

// UpdateAllCaches updates both feed and summary caches
func UpdateAllCaches(ctx context.Context) error {
	if !constants.RedisConnected {
		return fmt.Errorf("redis not connected, cannot update caches")
	}

	constants.Logger.Info("Starting cache update for feed and summary")

	freshFeedBytes, err := GenerateFeedDirect(ctx, constants.BaseURL)
	if err != nil {
		return fmt.Errorf("failed to generate direct feed for cache update: %w", err)
	}

	err = constants.RDB.Set(ctx, constants.CacheKey, freshFeedBytes, constants.CacheDuration).Err()
	if err != nil {
		constants.Logger.Error("Failed to update feed cache", "key", constants.CacheKey, "error", err)
	} else {
		constants.Logger.Info("Successfully updated feed cache", "key", constants.CacheKey)
	}

	markdownContent, err := markdown.ParseRSSToMarkdown(string(freshFeedBytes))
	if err != nil {
		constants.Logger.Error("Failed to parse fresh feed to markdown for summary update", "error", err)
		return fmt.Errorf("failed to parse fresh feed to markdown: %w", err)
	}

	summaryCtx, cancel := context.WithTimeout(ctx, constants.LlmTimeout)
	defer cancel()
	feedURLs := markdown.ExtractLinksFromMarkdown(markdownContent)
	summaryContent, err := SummarizeWithLLM(summaryCtx, markdownContent, feedURLs)
	if err != nil {
		constants.Logger.Error("Failed to summarize markdown with LLM for cache update", "error", err)
		return fmt.Errorf("failed to summarize markdown with LLM: %w", err)
	}

	summaryRSSBytes, err := GenerateSummaryRSS(summaryContent, constants.BaseURL, markdownContent)
	if err != nil {
		constants.Logger.Error("Failed to generate summary RSS for cache update", "error", err)
		return fmt.Errorf("failed to generate summary RSS: %w", err)
	}

	err = constants.RDB.Set(ctx, constants.SummaryCacheKey, summaryRSSBytes, constants.CacheDuration).Err()
	if err != nil {
		constants.Logger.Error("Failed to update summary cache", "key", constants.SummaryCacheKey, "error", err)
		return fmt.Errorf("failed to update summary cache: %w", err)
	} else {
		constants.Logger.Info("Successfully updated summary cache", "key", constants.SummaryCacheKey)
	}

	constants.Logger.Info("Starting conversation cache update")

	conversation, err := generatePodcastConversation(ctx, string(summaryRSSBytes))
	if err != nil {
		constants.Logger.Error("Failed to generate podcast conversation", "error", err)
		return fmt.Errorf("failed to generate podcast conversation: %w", err)
	}

	err = constants.RDB.Set(ctx, constants.ConversationCacheKey, []byte(conversation), constants.CacheDuration).Err()
	if err != nil {
		constants.Logger.Error("Failed to update conversation cache", "key", constants.ConversationCacheKey, "error", err)
		return fmt.Errorf("failed to update conversation cache: %w", err)
	} else {
		constants.Logger.Info("Successfully updated conversation cache", "key", constants.ConversationCacheKey)
	}

	constants.Logger.Info("Starting podcast cache update")

	audioData, err := GenerateAudioPodcast(ctx, conversation)
	if err != nil {
		constants.Logger.Error("Failed to generate podcast audio", "error", err)
		return fmt.Errorf("failed to generate podcast audio: %w", err)
	}

	if constants.R2Ready {
		err = R2PutPodcast(ctx, "podcast-testing.wav", audioData)
		if err != nil {
			constants.Logger.Error("Failed to upload podcast to R2", "key", "podcast-testing.wav", "error", err)
			return fmt.Errorf("failed to upload podcast to R2: %w", err)
		}
	} else {
		constants.Logger.Warn("R2 not ready, podcast will not be stored in R2")
	}

	constants.Logger.Info("Successfully updated podcast cache",
		"key", "podcast-testing.wav",
		"size", len(audioData))

	constants.Logger.Info("Successfully updated all caches (feed, summary, conversation, and podcast)")
	return nil
}
