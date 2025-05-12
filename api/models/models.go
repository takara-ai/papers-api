package models

import (
	"encoding/xml"
	"time"
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

type ConversationData struct {
	Conversation []DialogueEntry `json:"conversation"`
}

type DialogueEntry struct {
	Speaker string `json:"speaker"`
	Text    string `json:"text"`
}
