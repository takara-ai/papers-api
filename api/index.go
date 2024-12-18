package handler

import (
	"context"
	"encoding/xml"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
	"golang.org/x/net/html"
	"os"
	"github.com/redis/go-redis/v9"
	"github.com/joho/godotenv"
)

const (
	baseURL        = "https://huggingface.co/papers"
	updateInterval = 24 * time.Hour
	cacheKey       = "hf_papers_cache"
	cacheDuration  = 24 * time.Hour
	envKeyRedisURL        = "KV_URL"
	envKeyRedisToken      = "KV_REST_API_TOKEN"
	envKeyRedisReadToken  = "KV_REST_API_READ_ONLY_TOKEN"
	envKeyRedisRestURL    = "KV_REST_API_URL"
	scrapeTimeout = 30 * time.Second
	maxPapers     = 50  // Limit number of papers to prevent excessive scraping
	redisTimeout = 5 * time.Second
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
}

type Channel struct {
	Title         string    `xml:"title"`
	Link          string    `xml:"link"`
	Description   string    `xml:"description"`
	LastBuildDate string    `xml:"lastBuildDate"`
	Items         []Item    `xml:"item"`
}

type Item struct {
	Title       string `xml:"title"`
	Link        string `xml:"link"`
	Description string `xml:"description"`
	PubDate     string `xml:"pubDate"`
	GUID        string `xml:"guid"`
}

type FeedManager struct {
	currentFeed []byte
	lastUpdate  time.Time
	mutex       sync.RWMutex
}

func NewFeedManager() *FeedManager {
	return &FeedManager{}
}

func (fm *FeedManager) getCurrentFeed() []byte {
	fm.mutex.RLock()
	defer fm.mutex.RUnlock()
	return fm.currentFeed
}

func (fm *FeedManager) updateFeed(feed []byte) {
	fm.mutex.Lock()
	defer fm.mutex.Unlock()
	fm.currentFeed = feed
	fm.lastUpdate = time.Now()
}

func (fm *FeedManager) getLastUpdate() time.Time {
	fm.mutex.RLock()
	defer fm.mutex.RUnlock()
	return fm.lastUpdate
}

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
			Title:         "Hugging Face Papers RSS Feed",
			Link:          baseURL,
			Description:   "Latest papers from Hugging Face",
			LastBuildDate: time.Now().UTC().Format(time.RFC1123Z),
			Items:         items,
		},
	}

	return xml.MarshalIndent(rss, "", "  ")
}

// LoggingMiddleware wraps an http.HandlerFunc and provides request logging
func LoggingMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		
		// Create a custom response writer to capture the status code
		rw := &responseWriter{
			ResponseWriter: w,
			statusCode:    http.StatusOK,
		}
		
		next(rw, r)
		
		duration := time.Since(start)
		
		// Format similar to Gin's logging
		log.Printf("[HTTP] %s | %d | %12v | %15s | %-7s %s\n",
			time.Now().Format("2006/01/02 - 15:04:05"),
			rw.statusCode,
			duration,
			r.RemoteAddr,
			r.Method,
			r.URL.Path,
		)
	}
}

// responseWriter is a custom ResponseWriter that captures the status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// Create a global mux that can be used by both local server and Vercel handler
var mux *http.ServeMux
var feedManager *FeedManager
var rdb *redis.Client
var ctx = context.Background()
var redisConnected bool

func validateEnv() error {
	required := []string{
		envKeyRedisURL,
		envKeyRedisToken,
		envKeyRedisReadToken,
		envKeyRedisRestURL,
	}

	missing := []string{}
	for _, env := range required {
		if value := os.Getenv(env); value == "" {
			missing = append(missing, env)
		} else {
			// Log that we found the environment variable (but not its value)
			log.Printf("[INFO] Found environment variable: %s", env)
		}
	}

	if len(missing) > 0 {
		return fmt.Errorf("missing required environment variables: %v", missing)
	}

	return nil
}

func initRedis() {
	// Parse the Redis URL from KV_URL environment variable
	redisURL := os.Getenv(envKeyRedisURL)
	log.Printf("[DEBUG] Attempting Redis connection with URL: %s", redisURL)
	
	if redisURL == "" {
		log.Printf("[ERROR] %s environment variable not set", envKeyRedisURL)
		return
	}

	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		log.Printf("[ERROR] Error parsing Redis URL: %v", err)
		log.Printf("[DEBUG] URL format should be: rediss://username:password@host:port")
		return
	}

	log.Printf("[DEBUG] Successfully parsed Redis URL. Connecting to %s", opt.Addr)
	
	rdb = redis.NewClient(opt)

	// Test the connection with context and timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Test the connection
	_, err = rdb.Ping(ctx).Result()
	if err != nil {
		log.Printf("[ERROR] Error connecting to Redis: %v", err)
		log.Printf("[DEBUG] Redis client options: %+v", opt)
		return
	}

	redisConnected = true
	log.Printf("[INFO] Successfully connected to Redis at %s", opt.Addr)
}

func getCachedFeed() ([]byte, error) {
	if !redisConnected {
		log.Printf("[INFO] Redis not connected, generating feed directly")
		return generateFeedDirect()
	}

	ctx, cancel := context.WithTimeout(context.Background(), redisTimeout)
	defer cancel()

	// Try to get from cache first
	cachedData, err := rdb.Get(ctx, cacheKey).Bytes()
	if err == nil {
		log.Printf("[DEBUG] Serving cached feed")
		return cachedData, nil
	}
	if err != redis.Nil {
		log.Printf("[ERROR] Redis error: %v", err)
	}

	// Cache miss or error, generate new feed
	feed, err := generateFeedDirect()
	if err != nil {
		return nil, err
	}

	// Try to cache the new feed
	if redisConnected {
		ctx, cancel := context.WithTimeout(context.Background(), redisTimeout)
		defer cancel()
		
		err = rdb.Set(ctx, cacheKey, feed, cacheDuration).Err()
		if err != nil {
			log.Printf("[ERROR] Failed to cache feed: %v", err)
		}
	}

	return feed, nil
}

func generateFeedDirect() ([]byte, error) {
	papers, err := scrapePapers()
	if err != nil {
		return nil, fmt.Errorf("failed to scrape papers: %w", err)
	}

	return generateRSS(papers)
}

func loadEnvFile() {
	// Try to load .env from current directory
	err := godotenv.Load()
	if err != nil {
		// Try to load from parent directory
		err = godotenv.Load("../.env")
		if err != nil {
			log.Printf("[DEBUG] No .env file found in current or parent directory: %v", err)
			return
		}
	}
	log.Printf("[INFO] Successfully loaded environment variables from .env file")
}

// Add CORS middleware
func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, X-Update-Key")
		
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		next(w, r)
	}
}

// Move initialization to a separate function that can be called from Handler
func initialize() {
	// Only initialize once
	if mux != nil {
		return
	}

	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)
	log.SetOutput(os.Stdout)
	
	// Load environment variables
	loadEnvFile()
	
	// Initialize Redis connection
	log.Printf("[DEBUG] Testing connection via environment variables...")
	initRedis()
	
	// Validate environment variables
	if err := validateEnv(); err != nil {
		log.Printf("[WARNING] Environment validation failed: %v", err)
	}
	
	// Initialize router
	mux = http.NewServeMux()

	// Root handler for API info
	mux.HandleFunc("/api", corsMiddleware(LoggingMiddleware(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		response := `{
			"status": "ok",
			"available_endpoints": [
				"/api/feed",
				"/api/status",
				"/api/health",
				"/api/update"
			],
			"documentation": "https://github.com/yourusername/hf-daily-papers-feeds"
		}`
		w.Write([]byte(response))
	})))

	// Feed handler
	mux.HandleFunc("/api/feed", corsMiddleware(LoggingMiddleware(func(w http.ResponseWriter, r *http.Request) {
		feed, err := getCachedFeed()
		if err != nil {
			log.Printf("Error getting feed: %v", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/rss+xml")
		w.Write(feed)
	})))

	// Status handler
	mux.HandleFunc("/api/status", corsMiddleware(LoggingMiddleware(func(w http.ResponseWriter, r *http.Request) {
		var lastUpdate time.Time
		var redisError string

		if redisConnected {
			var err error
			lastUpdate, err = rdb.Get(ctx, "last_update").Time()
			if err != nil {
				lastUpdate = time.Now()
				redisError = err.Error()
			}
		} else {
			lastUpdate = time.Now()
			redisError = "Redis not connected"
		}
		
		nextUpdate := lastUpdate.Add(updateInterval)
		
		w.Header().Set("Content-Type", "application/json")
		response := fmt.Sprintf(`{
			"last_update": "%s",
			"next_update": "%s",
			"status": "active",
			"redis_connected": %v,
			"redis_error": %q,
			"cache_enabled": %v
		}`,
			lastUpdate.Format(time.RFC3339),
			nextUpdate.Format(time.RFC3339),
			redisConnected,
			redisError,
			redisConnected && redisError == "")
		w.Write([]byte(response))
	})))

	// Health handler
	mux.HandleFunc("/api/health", corsMiddleware(LoggingMiddleware(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		
		var redisHealth string
		if redisConnected {
			_, err := rdb.Ping(ctx).Result()
			if err != nil {
				redisHealth = fmt.Sprintf("error: %v", err)
			} else {
				redisHealth = "ok"
			}
		} else {
			redisHealth = "not connected"
		}
		
		// Add environment check
		envStatus := "ok"
		if err := validateEnv(); err != nil {
			envStatus = fmt.Sprintf("error: %v", err)
		}
		
		response := fmt.Sprintf(`{
			"status": "ok",
			"redis": %q,
			"environment": %q,
			"timestamp": %q
		}`,
			redisHealth,
			envStatus,
			time.Now().Format(time.RFC3339))
		w.Write([]byte(response))
	})))

	// Update handler
	mux.HandleFunc("/api/update", corsMiddleware(LoggingMiddleware(func(w http.ResponseWriter, r *http.Request) {
		// Check if this is a Vercel cron job or an authenticated request
		isVercelCron := r.Header.Get("User-Agent") == "vercel-cron"
		hasValidToken := r.Header.Get("X-Update-Key") == os.Getenv(envKeyRedisToken)
		
		if !isVercelCron && !hasValidToken {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		papers, err := scrapePapers()
		if err != nil {
			log.Printf("Error scraping papers: %v", err)
			http.Error(w, fmt.Sprintf("Error scraping papers: %v", err), http.StatusInternalServerError)
			return
		}

		feed, err := generateRSS(papers)
		if err != nil {
			log.Printf("Error generating RSS: %v", err)
			http.Error(w, fmt.Sprintf("Error generating RSS: %v", err), http.StatusInternalServerError)
			return
		}

		var cacheError string
		if redisConnected {
			err = rdb.Set(ctx, cacheKey, feed, cacheDuration).Err()
			if err != nil {
				cacheError = err.Error()
				log.Printf("Error caching feed: %v", err)
			} else {
				err = rdb.Set(ctx, "last_update", time.Now(), 0).Err()
				if err != nil {
					log.Printf("Error updating last_update time: %v", err)
				}
			}
		} else {
			cacheError = "Redis not connected"
		}

		w.Header().Set("Content-Type", "application/json")
		response := fmt.Sprintf(`{
			"status": "updated",
			"cache_updated": %v,
			"cache_error": %q,
			"papers_count": %d
		}`,
			cacheError == "",
			cacheError,
			len(papers))
		w.Write([]byte(response))
	})))
}

// Add cleanup function
func cleanup() {
	if rdb != nil {
		if err := rdb.Close(); err != nil {
			log.Printf("[ERROR] Failed to close Redis connection: %v", err)
		}
	}
}

// Handler handles all requests routed by Vercel
func Handler(w http.ResponseWriter, r *http.Request) {
	log.Printf("[DEBUG] Received request for path: %s", r.URL.Path)
	
	// Initialize if not already initialized
	if mux == nil {
		initialize()
	}

	// Just serve the request, no path manipulation needed
	mux.ServeHTTP(w, r)
}