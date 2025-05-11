package constants

import (
	"context"
	"log/slog"
	"os"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/redis/go-redis/v9"
)

const (
	BaseURL              = "https://huggingface.co/papers"
	LiveURL              = "https://tldr.takara.ai"
	ScrapeTimeout        = 180 * time.Second
	LlmTimeout           = 180 * time.Second
	MaxPapers            = 50
	CacheKey             = "hf_papers_cache"
	SummaryCacheKey      = "hf_papers_summary_cache"
	ConversationCacheKey = "hf_papers_conversation_cache"
	PodcastCacheKey      = "hf_papers_podcast_cache"
	CacheDuration        = 24 * time.Hour
)

var (
	Logger         = slog.New(slog.NewJSONHandler(os.Stdout, nil))
	RDB            *redis.Client
	Ctx            = context.Background()
	RedisConnected bool
	R2Client       *s3.Client
	R2Bucket       string
	R2BucketURL    string
	R2PublicURL    string
	R2Ready        bool
)
