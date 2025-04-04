package main

import (
	"log/slog"
	"net/http"
	"os"
	handler "hf-papers-rss/api"
)

var logger = slog.New(slog.NewJSONHandler(os.Stderr, nil)) // Initialize logger

func main() {
	// Handle all requests at /api/* with the same handler that Vercel uses
	http.HandleFunc("/api/", handler.Handler)
	
	port := "3000"
	// log.Printf("Server starting at http://localhost:%s", port)
	logger.Info("Server starting", "address", "http://localhost:"+port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		// log.Fatal(err)
		logger.Error("Server failed to start", "error", err)
		os.Exit(1)
	}
} 