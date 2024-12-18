package main

import (
	"log"
	"net/http"
	handler "hf-papers-rss/api"
)

func main() {
	// Handle all requests at /api/* with the same handler that Vercel uses
	http.HandleFunc("/api/", handler.Handler)
	
	port := "3000"
	log.Printf("Server starting at http://localhost:%s", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal(err)
	}
} 