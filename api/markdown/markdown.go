package markdown

import (
	"encoding/xml"
	"fmt"
	constants "hf-papers-rss/api/constants"
	models "hf-papers-rss/api/models"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/gomarkdown/markdown/ast"
	"github.com/gomarkdown/markdown/parser"
)

func ParseRSSToMarkdown(xmlContent string) (string, error) {
	var rss models.RSS
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
func ValidateMarkdownStructure(markdown string) error {
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
			constants.Logger.Error("Missing required section",
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
		constants.Logger.Error("Empty headline content", "details", "Extracted headline string was empty after AST parsing and trimming.")
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
		constants.Logger.Error("Headline too long",
			"headline", headlineText,
			"length", len(headlineText))
		return ValidationError{
			Field:    "headline",
			Message:  fmt.Sprintf("headline too long: %d characters (limit 200)", len(headlineText)),
			Details:  headlineText,
			Severity: SeverityError,
		}
	}

	constants.Logger.Debug("Markdown structure validation successful",
		"headline", headlineText,
		"headline_length", len(headlineText))

	return nil
}

// normalizeURL standardizes URL format for comparison
func NormalizeURL(url string) string {
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

	constants.Logger.Debug("URL normalization",
		"original", original,
		"normalized", url)
	return url
}

// validateMarkdownLinks checks if all markdown links are properly formatted and contain URLs
func ValidateMarkdownLinks(markdown string, feedURLs map[string]string) error {
	// Create normalized feed URLs map
	normalizedFeedURLs := make(map[string]string)
	for url := range feedURLs {
		normalized := NormalizeURL(url)
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
		normalizedURL := NormalizeURL(url)

		// Check if normalized URL exists in feed
		if _, exists := normalizedFeedURLs[normalizedURL]; !exists {
			warnings = append(warnings, fmt.Sprintf("%s: %s (not in feed)", text, url))
			continue
		}
	}

	// Log warnings but don't fail validation
	if len(warnings) > 0 {
		constants.Logger.Warn("Link validation warnings",
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
func ValidateSummaryLength(markdown string) error {
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
		constants.Logger.Warn("Summary approaching length limit",
			"word_count", len(words),
			"max_allowed", 1000)
	}

	return nil
}

// validateSummaryContent performs all validations on the summary in parallel
func ValidateSummaryContent(markdown string, feedURLs map[string]string) error {
	// Create channels for validation results
	errChan := make(chan error, 3)
	var wg sync.WaitGroup

	// Run validations in parallel
	wg.Add(3)
	go func() {
		defer wg.Done()
		if err := ValidateMarkdownStructure(markdown); err != nil {
			errChan <- err
		}
	}()

	go func() {
		defer wg.Done()
		if err := ValidateMarkdownLinks(markdown, feedURLs); err != nil {
			errChan <- err
		}
	}()

	go func() {
		defer wg.Done()
		if err := ValidateSummaryLength(markdown); err != nil {
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
		constants.Logger.Warn("Summary validation warning",
			"warning", warning,
			"markdown", markdown)
	}

	// Return first error if any exist
	if len(errors) > 0 {
		return fmt.Errorf("validation failed: %v", errors)
	}

	return nil
}

// extractLinksFromMarkdown parses the input markdown to find ## [Title](URL) lines
// and returns a map of title -> URL.
func ExtractLinksFromMarkdown(markdownContent string) map[string]string {
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
			constants.Logger.Debug("Extracted link", "title", title, "url", url) // Optional: Debug log
		}
	}
	return links
}

// replacePlaceholdersWithLinks replaces placeholders like [Title] in the summary
// with actual markdown links using the provided title-URL map.
func ReplacePlaceholdersWithLinks(summaryMarkdown string, links map[string]string) string {
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
