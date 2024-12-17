package main

import (
	"net/http"
)

// Handler is the Vercel serverless function handler
var Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Forward the request to our router
	mux.ServeHTTP(w, r)
}) 