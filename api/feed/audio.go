package feed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"

	constants "hf-papers-rss/api/constants"
	models "hf-papers-rss/api/models"
)

func extractWavData(data []byte) ([]byte, error) {
	if len(data) < 44 {
		return nil, fmt.Errorf("invalid WAV file: too short")
	}

	if string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, fmt.Errorf("invalid WAV file: missing RIFF/WAVE header")
	}

	dataStart := 44 
	for i := 12; i < len(data)-8; i++ {
		if string(data[i:i+4]) == "data" {
			dataStart = i + 8
			break
		}
	}

	return data[dataStart:], nil
}

func stripID3(data []byte) []byte {
	if len(data) > 10 && string(data[:3]) == "ID3" {
		sz := int(data[6]&0x7F)<<21 | int(data[7]&0x7F)<<14 | int(data[8]&0x7F)<<7 | int(data[9]&0x7F)
		end := 10 + sz
		if end < len(data) {
			return data[end:]
		}
	}
	if len(data) > 128 && string(data[len(data)-128:len(data)-125]) == "TAG" {
		return data[:len(data)-128]
	}
	return data
}

func GenerateAudioPodcast(ctx context.Context, text string) ([]byte, error) {
	var conversation models.ConversationData
	if err := json.Unmarshal([]byte(text), &conversation); err != nil {
		return nil, fmt.Errorf("failed to parse conversation: %w", err)
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable is not set")
	}

	url := "https://api.openai.com/v1/audio/speech"

	tempFiles := make([]string, 0, len(conversation.Conversation))
	defer func() {
		for _, f := range tempFiles {
			os.Remove(f)
		}
	}()

	client := &http.Client{
		Timeout: 150 * time.Second,
	}

	type result struct {
		index int
		file  string
		err   error
	}
	results := make(chan result, len(conversation.Conversation))

	maxWorkers := 3 
	semaphore := make(chan struct{}, maxWorkers)

	// Function to process a single entry with retries
	processEntry := func(i int, entry models.DialogueEntry) {
		const maxRetries = 3
		var lastErr error

		for attempt := 1; attempt <= maxRetries; attempt++ {
			if err := ctx.Err(); err != nil {
				results <- result{index: i, err: fmt.Errorf("context cancelled: %w", err)}
				return
			}

			voice := "alloy"
			if entry.Speaker == "Jenny" {
				voice = "alloy"
			} else if entry.Speaker == "Brian" {
				voice = "ash"
			}

			requestBody := map[string]interface{}{
				"model":           "gpt-4o-mini-tts",
				"input":           entry.Text,
				"voice":           voice,
				"response_format": "wav",
				"instructions":    "Voice: Enthusiastic, Casual, and Fun \n Tone: Exicted",
			}

			jsonBody, err := json.Marshal(requestBody)
			if err != nil {
				results <- result{index: i, err: fmt.Errorf("failed to marshal request body: %w", err)}
				return
			}

			req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonBody))
			if err != nil {
				results <- result{index: i, err: fmt.Errorf("failed to create request: %w", err)}
				return
			}

			req.Header.Set("Authorization", "Bearer "+apiKey)
			req.Header.Set("Content-Type", "application/json")

			start_time := time.Now()
			resp, err := client.Do(req)
			duration := time.Since(start_time).Seconds()
			constants.Logger.Info("Request duration", "duration", duration, "index", i, "attempt", attempt)

			if err != nil {
				lastErr = err
				if errors.Is(err, context.Canceled) {
					results <- result{index: i, err: fmt.Errorf("request cancelled: %w", err)}
					return
				}
				if attempt < maxRetries {
					time.Sleep(time.Duration(attempt) * time.Second)
					continue
				}
				results <- result{index: i, err: fmt.Errorf("failed to make request after %d attempts: %w", maxRetries, err)}
				return
			}

			if resp.StatusCode != http.StatusOK {
				bodyBytes, _ := io.ReadAll(resp.Body)
				resp.Body.Close()
				lastErr = fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(bodyBytes))
				if attempt < maxRetries {
					time.Sleep(time.Duration(attempt) * time.Second)
					continue
				}
				results <- result{index: i, err: lastErr}
				return
			}

			tempFile, err := os.CreateTemp("", "podcast_part_*.wav")
			if err != nil {
				results <- result{index: i, err: fmt.Errorf("failed to create temp file: %w", err)}
				return
			}
			tempFiles = append(tempFiles, tempFile.Name())
			_, err = io.Copy(tempFile, resp.Body)
			tempFile.Close()
			resp.Body.Close()
			if err != nil {
				lastErr = fmt.Errorf("failed to write audio data: %w", err)
				if attempt < maxRetries {
					time.Sleep(time.Duration(attempt) * time.Second)
					continue
				}
				results <- result{index: i, err: lastErr}
				return
			}

			results <- result{index: i, file: tempFile.Name()}
			return
		}

		results <- result{index: i, err: fmt.Errorf("all retries failed: %w", lastErr)}
	}

	for i, entry := range conversation.Conversation {
		semaphore <- struct{}{} 
		go func(i int, entry models.DialogueEntry) {
			defer func() { <-semaphore }() 
			processEntry(i, entry)
		}(i, entry)
	}

	orderedFiles := make([]string, len(conversation.Conversation))
	for i := 0; i < len(conversation.Conversation); i++ {
		result := <-results
		if result.err != nil {
			return nil, fmt.Errorf("error processing entry %d: %w", result.index, result.err)
		}
		orderedFiles[result.index] = result.file
	}

	var combined []byte
	var header []byte

	for i, f := range orderedFiles {
		b, err := os.ReadFile(f)
		if err != nil {
			return nil, fmt.Errorf("failed to read wav part: %w", err)
		}

		b = stripID3(b)

		if i == 0 {
			header = b[:44] 
			audioData, err := extractWavData(b)
			if err != nil {
				return nil, fmt.Errorf("failed to extract audio data from first file: %w", err)
			}
			combined = append(combined, audioData...)
		} else {
			audioData, err := extractWavData(b)
			if err != nil {
				return nil, fmt.Errorf("failed to extract audio data from file %d: %w", i, err)
			}
			combined = append(combined, audioData...)
		}
	}

	if len(combined) > 0 {
		fileSize := uint32(len(combined) + 36) 
		header[4] = byte(fileSize)
		header[5] = byte(fileSize >> 8)
		header[6] = byte(fileSize >> 16)
		header[7] = byte(fileSize >> 24)

		dataSize := uint32(len(combined))
		header[40] = byte(dataSize)
		header[41] = byte(dataSize >> 8)
		header[42] = byte(dataSize >> 16)
		header[43] = byte(dataSize >> 24)
	}

	return append(header, combined...), nil
}

func R2PutPodcast(ctx context.Context, key string, data []byte) error {
	if !constants.R2Ready {
		return fmt.Errorf("R2 not initialized")
	}

	uploadCtx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()

	metadata := map[string]string{
		"Content-Type":  "audio/mpeg",
		"Cache-Control": "public, max-age=86400",
		"Generated-At":  time.Now().UTC().Format(time.RFC3339),
	}

	constants.Logger.Info("Uploading podcast to R2", "key", key, "size", len(data))
	_, err := constants.R2Client.PutObject(uploadCtx, &s3.PutObjectInput{
		Bucket:      &constants.R2Bucket,
		Key:         &key,
		Body:        bytes.NewReader(data),
		ContentType: aws.String("audio/mpeg"),
		ACL:         types.ObjectCannedACLPublicRead,
		Metadata:    metadata,
	})
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			constants.Logger.Error("R2 upload timed out or was canceled", "key", key, "error", err)
			return fmt.Errorf("R2 upload timed out or was canceled: %w", err)
		}
		constants.Logger.Error("Failed to upload podcast to R2", "key", key, "error", err)
		return fmt.Errorf("failed to upload podcast to R2: %w", err)
	}

	constants.Logger.Info("Successfully uploaded podcast to R2", "key", key, "size", len(data))
	return nil
}

func r2GetPodcast(ctx context.Context, key string) ([]byte, error) {
	if !constants.R2Ready {
		return nil, fmt.Errorf("R2 not initialized")
	}
	resp, err := constants.R2Client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: &constants.R2Bucket,
		Key:    &key,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func GetCachedPodcast(ctx context.Context, text string) (string, error) {
	key := "podcast-testing.wav"
	final_url := key

	constants.Logger.Info("Getting cached podcast", "key", key, "url", final_url)

	if constants.R2Ready {
		_, err := r2GetPodcast(ctx, key)
		if err == nil {
			constants.Logger.Info("Podcast found in R2", "key", key)
			return final_url, nil
		}

		var notFoundErr *types.NoSuchKey
		if errors.As(err, &notFoundErr) {
			constants.Logger.Info("Podcast not found in R2, will generate", "key", key)
		} else {
			constants.Logger.Warn("Error checking R2 for podcast, will generate", "key", key, "error", err)
		}
	} else {
		constants.Logger.Warn("R2 not ready, will generate podcast without caching")
	}

	conversation, err := Getcachedconversation(ctx, text)
	if err != nil {
		return "", fmt.Errorf("failed to get conversation: %w", err)
	}

	audioData, err := GenerateAudioPodcast(ctx, conversation)
	if err != nil {
		return "", fmt.Errorf("failed to generate audio podcast: %w", err)
	}

	if constants.R2Ready {
		err = R2PutPodcast(ctx, key, audioData)
		if err != nil {
			constants.Logger.Error("Failed to upload podcast to R2", "key", key, "error", err)
			return "", fmt.Errorf("failed to upload podcast to R2: %w", err)
		}
		constants.Logger.Info("Successfully uploaded podcast to R2", "key", key)
	}

	return final_url, nil
}
