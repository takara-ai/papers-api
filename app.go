package main

import (
	"encoding/xml"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
	"golang.org/x/net/html"
	"os"
)

const (
	baseURL     = "https://huggingface.co/papers"
	updateInterval = 6 * time.Hour
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
	resp, err := http.Get(url)
	if err != nil {
		return "", err
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
	resp, err := http.Get(baseURL)
	if err != nil {
		return nil, err
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

func main() {
	// Configure logging
	log.SetFlags(0)
	log.SetOutput(os.Stdout)
	
	log.Println("[INFO] Starting HTTP server in development mode")
	log.Println("[INFO] Version: Go standard library HTTP server")
	
	feedManager := NewFeedManager()

	// Initial feed update
	papers, err := scrapePapers()
	if err != nil {
		log.Fatalf("Initial scrape failed: %v", err)
	}

	feed, err := generateRSS(papers)
	if err != nil {
		log.Fatalf("Failed to generate initial RSS: %v", err)
	}
	feedManager.updateFeed(feed)

	// Start background updater
	go func() {
		ticker := time.NewTicker(updateInterval)
		for range ticker.C {
			papers, err := scrapePapers()
			if err != nil {
				log.Printf("Failed to scrape papers: %v", err)
				continue
			}

			feed, err := generateRSS(papers)
			if err != nil {
				log.Printf("Failed to generate RSS: %v", err)
				continue
			}

			feedManager.updateFeed(feed)
			log.Printf("[INFO] Feed updated successfully with %d papers", len(papers))
		}
	}()

	// Set up HTTP server with standard library
	mux := http.NewServeMux()

	// Register routes with logging middleware
	mux.HandleFunc("/feed", LoggingMiddleware(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/rss+xml")
		w.Write(feedManager.getCurrentFeed())
	}))

	mux.HandleFunc("/status", LoggingMiddleware(func(w http.ResponseWriter, r *http.Request) {
		lastUpdate := feedManager.getLastUpdate()
		nextUpdate := lastUpdate.Add(updateInterval)
		
		w.Header().Set("Content-Type", "application/json")
		response := fmt.Sprintf(`{"last_update":"%s","next_update":"%s","status":"active"}`,
			lastUpdate.Format(time.RFC3339),
			nextUpdate.Format(time.RFC3339))
		w.Write([]byte(response))
	}))

	mux.HandleFunc("/health", LoggingMiddleware(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	server := &http.Server{
		Addr:    ":8080",
		Handler: mux,
	}

	// Log registered routes
	log.Println("[INFO] Registered routes:")
	log.Println("[INFO] GET    /feed")
	log.Println("[INFO] GET    /status")
	log.Println("[INFO] GET    /health")
	log.Printf("[INFO] Listening and serving HTTP on %s\n", server.Addr)

	log.Fatal(server.ListenAndServe())
}