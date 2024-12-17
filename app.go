package main

import (
	"encoding/xml"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/gin-gonic/gin"
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

	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return "", err
	}

	abstract := doc.Find("div.pb-8.pr-4.md\\:pr-16").Text()
	abstract = strings.TrimPrefix(abstract, "Abstract\n")
	abstract = strings.ReplaceAll(abstract, "\n", " ")
	return strings.TrimSpace(abstract), nil
}

func scrapePapers() ([]Paper, error) {
	resp, err := http.Get(baseURL)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return nil, err
	}

	var papers []Paper
	doc.Find("h3").Each(func(i int, s *goquery.Selection) {
		if a := s.Find("a"); a.Length() > 0 {
			title := strings.TrimSpace(a.Text())
			href, _ := a.Attr("href")
			url := fmt.Sprintf("https://huggingface.co%s", href)
			
			abstract, err := scrapeAbstract(url)
			if err != nil {
				log.Printf("Failed to extract abstract for %s: %v", url, err)
				abstract = ""
			}

			papers = append(papers, Paper{
				Title:    title,
				URL:      url,
				Abstract: abstract,
				PubDate:  time.Now().UTC(),
			})
		}
	})

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

func main() {
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
			log.Printf("Feed updated successfully with %d papers", len(papers))
		}
	}()

	// Set up HTTP server
	r := gin.Default()

	r.GET("/feed", func(c *gin.Context) {
		c.Data(http.StatusOK, "application/rss+xml", feedManager.getCurrentFeed())
	})

	r.GET("/status", func(c *gin.Context) {
		lastUpdate := feedManager.getLastUpdate()
		nextUpdate := lastUpdate.Add(updateInterval)
		
		c.JSON(http.StatusOK, gin.H{
			"last_update":  lastUpdate,
			"next_update":  nextUpdate,
			"status":       "active",
		})
	})

	// Add health check endpoint
	r.GET("/health", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	log.Fatal(r.Run(":8080"))
}