package handler

import (
	"net/http"
)

// Handler is the Vercel serverless function handler
func Handler(w http.ResponseWriter, r *http.Request) {
	// Initialize if not already initialized
	if mux == nil {
		initialize()
	}
	mux.ServeHTTP(w, r)
} 