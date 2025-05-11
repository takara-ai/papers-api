package scraper

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	models "hf-papers-rss/api/models"
	constants "hf-papers-rss/api/constants"
	"golang.org/x/net/html"
)

func ScrapeAbstract(ctx context.Context, url string) (string, error) {
	client := &http.Client{
		Timeout: constants.ScrapeTimeout,
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
		constants.Logger.Warn("Abstract div not found", "class", "pb-8 pr-4 md:pr-16", "url", url)
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

func ScrapePapers(ctx context.Context) ([]models.Paper, error) {
	client := &http.Client{
		Timeout: constants.ScrapeTimeout,
	}

	req, err := http.NewRequestWithContext(ctx, "GET", constants.BaseURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request for %s: %w", constants.BaseURL, err)
	}

	resp, err := client.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, fmt.Errorf("timeout fetching papers from %s: %w", constants.BaseURL, err)
		}
		return nil, fmt.Errorf("failed to fetch papers from %s: %w", constants.BaseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to fetch papers from %s: status code %d", constants.BaseURL, resp.StatusCode)
	}

	doc, err := html.Parse(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to parse HTML from %s: %w", constants.BaseURL, err)
	}

	var papers []models.Paper
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
				abstract, err := ScrapeAbstract(ctx, url)
				if err != nil {
					constants.Logger.Error("Failed to extract abstract", "url", url, "error", err)
					abstract = "[Abstract not available]" // Placeholder
				}

				papers = append(papers, models.Paper{
					Title:    strings.TrimSpace(title),
					URL:      url,
					Abstract: abstract,
					PubDate:  time.Now().UTC(), // Consider parsing date if available in new structure
				})
			} else {
				// Log if details couldn't be found within an article element
				constants.Logger.Warn("Could not find paper details (h3 > a) within article element")
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
	if len(papers) > constants.MaxPapers {
		papers = papers[:constants.MaxPapers]
	}

	return papers, nil
}