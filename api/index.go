package handler

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
	"golang.org/x/net/html"
	"os"
	"github.com/redis/go-redis/v9"
)

const (
	baseURL       = "https://huggingface.co/papers"
	scrapeTimeout = 30 * time.Second
	maxPapers     = 50
	cacheKey      = "hf_papers_cache"
	cacheDuration = 24 * time.Hour
)

type Paper struct {
	Title    string
	URL      string
	Abstract string
	PubDate  time.Time
}

type RSS struct {
	Version string `xml:"version"`
	Channel Channel `xml:"channel"`
}

type Channel struct {
	Title         string `xml:"title"`
	Link          string `xml:"link"`
	Description   string `xml:"description"`
	LastBuildDate string `xml:"lastBuildDate"`
	Items         []Item `xml:"item"`
}

type Item struct {
	Title       string `xml:"title"`
	Link        string `xml:"link"`
	Description string `xml:"description"`
	PubDate     string `xml:"pubDate"`
	GUID        string `xml:"guid"`
}

var (
	rdb *redis.Client
	ctx = context.Background()
	redisConnected bool
)

func scrapeAbstract(url string) (string, error) {
	client := &http.Client{
		Timeout: scrapeTimeout,
	}
	
	resp, err := client.Get(url)
	if err != nil {
		return "", fmt.Errorf("failed to fetch abstract: %w", err)
	}
	defer resp.Body.Close()

	doc, err := html.Parse(resp.Body)
	if err != nil {
		return "", err
	}

	var abstract string
	var crawler func(*html.Node)
	crawler = func(node *html.Node) {
		if node.Type == html.ElementNode && node.Data == "div" {
			for _, attr := range node.Attr {
				if attr.Key == "class" && strings.Contains(attr.Val, "pb-8 pr-4 md:pr-16") {
					abstract = extractText(node)
					return
				}
			}
		}
		for c := node.FirstChild; c != nil; c = c.NextSibling {
			crawler(c)
		}
	}
	crawler(doc)

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

func scrapePapers() ([]Paper, error) {
	client := &http.Client{
		Timeout: scrapeTimeout,
	}
	
	resp, err := client.Get(baseURL)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch papers: %w", err)
	}
	defer resp.Body.Close()

	doc, err := html.Parse(resp.Body)
	if err != nil {
		return nil, err
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
				abstract, err := scrapeAbstract(url)
				if err != nil {
					log.Printf("Failed to extract abstract for %s: %v", url, err)
					abstract = ""
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

func generateRSS(papers []Paper) ([]byte, error) {
	items := make([]Item, len(papers))
	for i, paper := range papers {
		items[i] = Item{
			Title:       paper.Title,
			Link:        paper.URL,
			Description: paper.Abstract,
			PubDate:     paper.PubDate.Format(time.RFC1123Z),
			GUID:        paper.URL,
		}
	}

	rss := RSS{
    Version: "2.0",
    Channel: Channel{
        Title:         "宝の知識: Hugging Face 論文フィード", // "Takara no Chishiki: Hugging Face Ronbun Fiido"
        Link:          baseURL,
        Description:   "最先端のAI論文をお届けする、Takara.aiの厳選フィード", // "Delivering cutting-edge AI papers, curated by Takara.ai."
        LastBuildDate: time.Now().UTC().Format(time.RFC1123Z),
        Items:         items,
    },
}


	return xml.MarshalIndent(rss, "", "  ")
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
		log.Printf("[ERROR] Error parsing Redis URL: %v", err)
		return
	}

	rdb = redis.NewClient(opt)
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Printf("[ERROR] Error connecting to Redis: %v", err)
		return
	}

	redisConnected = true
	log.Printf("[INFO] Successfully connected to Redis")
}

func getCachedFeed() ([]byte, error) {
	if !redisConnected {
		return generateFeedDirect()
	}

	// Try to get from cache first
	cachedData, err := rdb.Get(ctx, cacheKey).Bytes()
	if err == nil {
		return cachedData, nil
	}

	// Cache miss, generate new feed
	feed, err := generateFeedDirect()
	if err != nil {
		return nil, err
	}

	// Cache the new feed
	if redisConnected {
		rdb.Set(ctx, cacheKey, feed, cacheDuration)
	}

	return feed, nil
}

func generateFeedDirect() ([]byte, error) {
	papers, err := scrapePapers()
	if err != nil {
		return nil, err
	}
	return generateRSS(papers)
}

func updateCache() error {
	if !redisConnected {
		return fmt.Errorf("redis not connected")
	}

	// Generate new feed
	feed, err := generateFeedDirect()
	if err != nil {
		return fmt.Errorf("failed to generate feed: %w", err)
	}

	// Update cache
	err = rdb.Set(ctx, cacheKey, feed, cacheDuration).Err()
	if err != nil {
		return fmt.Errorf("failed to update cache: %w", err)
	}

	return nil
}

// Handler handles all requests
func Handler(w http.ResponseWriter, r *http.Request) {
	// Initialize Redis on first request
	if !redisConnected {
		initRedis()
	}

	// Remove trailing slash and normalize path
	path := strings.TrimSuffix(r.URL.Path, "/")
	if path == "" {
		path = "/api"  // Normalize empty path to /api
	}

	// Apply CORS middleware
	corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
		switch path {
		case "/api":
			// Health check endpoint
			w.Header().Set("Content-Type", "application/json")
			healthStatus := map[string]interface{}{
				"status":       "ok",
				"endpoints":    []string{"/api/feed"},
				"cache_status": redisConnected,
				"timestamp":    time.Now().UTC().Format(time.RFC3339),
				"version":      "1.0.0",
			}
			
			if err := json.NewEncoder(w).Encode(healthStatus); err != nil {
				http.Error(w, "Error encoding response", http.StatusInternalServerError)
			}
			return

		case "/api/feed":
			feed, err := getCachedFeed()
			if err != nil {
				http.Error(w, "Error generating feed", http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/rss+xml")
			w.Write(feed)
			return

		case "/api/update-cache":
			// Check for secret key to prevent unauthorized updates
			secretKey := r.Header.Get("X-Update-Key")
			expectedKey := os.Getenv("UPDATE_KEY")
			
			if expectedKey == "" || secretKey != expectedKey {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}

			err := updateCache()
			if err != nil {
				http.Error(w, fmt.Sprintf("Error updating cache: %v", err), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]string{
				"status": "Cache updated successfully",
				"timestamp": time.Now().UTC().Format(time.RFC3339),
			})
			return

		default:
			http.NotFound(w, r)
		}
	})(w, r)
}