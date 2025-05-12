package handler

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/joho/godotenv"
	"github.com/redis/go-redis/v9"

	constants "hf-papers-rss/api/constants"
	feed "hf-papers-rss/api/feed"
)

func init() {
	// Load .env file on package initialization
	err := godotenv.Load()
	if err != nil {
		// Log if .env is not found, but don't treat as fatal error
		// Environment variables might be set directly
		constants.Logger.Info("Error loading .env file (this is expected if using system env vars)", "error", err)
	}
	initRedis()
	initR2()
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
		constants.Logger.Warn("KV_URL not set, Redis connection skipped")
		return // Skip Redis connection if URL is not set
	}

	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		constants.Logger.Error("Error parsing Redis URL", "error", err)
		return
	}

	constants.RDB = redis.NewClient(opt)
	pingErr := constants.RDB.Ping(constants.Ctx).Err()
	if pingErr != nil {
		constants.Logger.Error("Error connecting to Redis", "error", pingErr)
		return
	}

	constants.RedisConnected = true
	constants.Logger.Info("Successfully connected to Redis")
}

func initR2() {
	endpoint := os.Getenv("R2_ENDPOINT")
	accessKey := os.Getenv("R2_ACCESS_KEY_ID")
	secretKey := os.Getenv("R2_SECRET_ACCESS_KEY")
	bucket := os.Getenv("R2_BUCKET_NAME")
	if endpoint == "" || accessKey == "" || secretKey == "" || bucket == "" {
		constants.Logger.Warn("Cloudflare R2 env vars missing, audio podcast will not be stored in R2")
		return
	}
	cfg, err := config.LoadDefaultConfig(constants.Ctx,
		config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(accessKey, secretKey, "")),
		config.WithRegion("auto"),
		config.WithEndpointResolverWithOptions(aws.EndpointResolverWithOptionsFunc(
			func(service, region string, options ...interface{}) (aws.Endpoint, error) {
				return aws.Endpoint{URL: endpoint, SigningRegion: "auto"}, nil
			},
		)),
	)
	if err != nil {
		constants.Logger.Error("Failed to init R2 S3 client", "error", err)
		return
	}
	constants.R2Client = s3.NewFromConfig(cfg)
	constants.R2Bucket = bucket
	constants.R2BucketURL = endpoint
	constants.R2Ready = true
	constants.R2PublicURL = os.Getenv("R2_PUBLIC_URL")
	constants.Logger.Info("Cloudflare R2 S3 client initialized")
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
		path = "/api" // Normalize empty path to /api
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
				"endpoints":    []string{"/api/feed", "/api/summary", "/api/conversation", "/api/podcast"},
				"cache_status": constants.RedisConnected,
				"timestamp":    time.Now().UTC().Format(time.RFC3339),
				"version":      "1.0.0",
			}

			if err := json.NewEncoder(w).Encode(healthStatus); err != nil {
				http.Error(w, "Error encoding response", http.StatusInternalServerError)
			}
			return

		case "/api/feed":
			// Pass request context to feed retrieval/generation
			feed, err := feed.GetCachedFeed(reqCtx, requestURL)
			if err != nil {
				constants.Logger.Error("Failed to get cached feed", "error", err)
				http.Error(w, "Error generating feed", http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/rss+xml")
			w.Write(feed)
			return

		case "/api/summary":
			// Pass request context to summary retrieval/generation
			summary, err := feed.GetCachedSummary(reqCtx, requestURL)
			if err != nil {
				constants.Logger.Error("Failed to get cached summary", "error", err)
				http.Error(w, fmt.Sprintf("Error generating summary: %v", err), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/rss+xml")
			w.Write(summary)
			return

		case "/api/conversation":
			// Pass request context to summary retrieval/generation
			summary, err := feed.GetCachedFeed(reqCtx, requestURL)
			if err != nil {
				constants.Logger.Error("Failed to get cached feed", "error", err)
				http.Error(w, fmt.Sprintf("Error getting Feed: %v", err), http.StatusInternalServerError)
				return
			}
			// Generate podcast conversation
			conversation, err := feed.Getcachedconversation(reqCtx, string(summary))
			if err != nil {
				constants.Logger.Error("Failed to generate podcast conversation", "error", err)
				http.Error(w, fmt.Sprintf("Error generating podcast conversation: %v", err), http.StatusInternalServerError)
				return
			}
			// Set content type to JSON
			w.Header().Set("Content-Type", "application/json")
			// Write the conversation response
			w.Write([]byte(conversation))
			return

		case "/api/podcast":
			feedBytes, err := feed.GetCachedFeed(reqCtx, requestURL)
			if err != nil {
				constants.Logger.Error("Failed to get cached feed", "error", err)
				http.Error(w, fmt.Sprintf("Error getting Feed: %v", err), http.StatusInternalServerError)
				return
			}

			// Ensure podcast is generated and cached
			final_url, err := feed.GetCachedPodcast(reqCtx, string(feedBytes))
			if err != nil {
				constants.Logger.Error("Failed to get/generate podcast", "error", err)
				http.Error(w, fmt.Sprintf("Error with podcast: %v", err), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			response := map[string]string{
				"url": fmt.Sprintf("%s/%s", constants.R2PublicURL, final_url),
			}
			if err := json.NewEncoder(w).Encode(response); err != nil {
				constants.Logger.Error("Failed to encode response", "error", err)
				http.Error(w, "Error encoding response", http.StatusInternalServerError)
			}
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
			err := feed.UpdateAllCaches(reqCtx)
			if err != nil {
				// Use a more specific error message if possible
				constants.Logger.Error("Failed to update caches via API", "error", err)
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
