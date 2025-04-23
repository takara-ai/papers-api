# Hugging Face Papers RSS Feed

A serverless RSS feed for papers published on Hugging Face's [papers page](https://huggingface.co/papers). This service scrapes the papers page daily and generates an RSS feed that can be consumed by any RSS reader.

## Features

- Serverless deployment on Vercel
- Vercel Edge Caching to minimize scraping and provide low-latency responses globally
- Daily automatic cache refresh (target: 6 AM UTC)
- Clean RSS feed with paper titles, links and abstracts
- LLM-powered summary feed of the latest papers
- LLM-powered podcast feed (conversation + audio) of the latest papers
- Health check and status endpoints
- CORS enabled for cross-origin requests

## Deployment

### Prerequisites

- A [Vercel](https://vercel.com) account
- Go 1.22+ installed locally for development
- A Hugging Face API key (for LLM summaries and podcast generation)
- A DeepInfra API key (for podcast audio generation)

### Deploy to Vercel

1. Fork this repository

2. Create a new project on Vercel and import your forked repository

3. Set up the following environment variables in your Vercel project settings:

```env
# Required for LLM Summaries / Podcast Generation
HF_API_KEY=your_hugging_face_api_key

# Required for Podcast Audio Generation
DEEPINFRA_API_KEY=your_deepinfra_api_key

# Optional: Secret key for potentially adding manual cache purge in the future (not currently implemented)
# UPDATE_KEY=your_secret_key_here
```

4. Deploy! Vercel will automatically build and deploy your RSS feed service.

The feed will be available at: `https://your-project.vercel.app/api/feed`
The summary feed will be available at: `https://your-project.vercel.app/api/summary`
The podcast conversation (JSON) will be available at: `https://your-project.vercel.app/api/conversation`
The podcast audio (MP3) will be available at: `https://your-project.vercel.app/api/podcast`

### Local Development

1. Clone the repository:

```bash
git clone https://github.com/yourusername/hf-daily-papers-feeds.git # Make sure to use your repo name
cd hf-daily-papers-feeds
```

2. Create a `.env` file in the root directory with your Hugging Face and DeepInfra API keys:

```env
# Required for LLM Summaries / Podcast Generation
HF_API_KEY=your_hugging_face_api_key

# Required for Podcast Audio Generation
DEEPINFRA_API_KEY=your_deepinfra_api_key

# Optional: Secret key for potentially adding manual cache purge in the future (not currently implemented)
# UPDATE_KEY=your_secret_key_here
```

3. Run the development server:

```bash
vercel dev
```

Or run the Go server directly (listens on port 3000 by default):

```bash
go run main.go
```

The local server will be available at `http://localhost:3000`

## API Endpoints

- `/api` - Health check and status
- `/api/feed` - RSS feed of papers
- `/api/summary` - RSS feed summarizing the papers using an LLM
- `/api/conversation` - JSON feed containing the generated podcast conversation
- `/api/podcast` - MP3 audio feed of the generated podcast

## Caching

This service leverages Vercel's Edge Caching. Responses are cached globally with a `Cache-Control` header set to expire around 6 AM UTC daily. This means fresh content is generated approximately once per day.

- **Cache Duration:** Calculated dynamically to expire at the next 6:00 AM UTC.
- **Cache Control:** `public, max-age=0, s-maxage=<seconds_until_6am_utc>`
- **Invalidation:** Cache automatically expires based on `s-maxage` or when a new deployment occurs.

## Example Response

Health check (`/api`):

```json
{
  "status": "ok",
  "endpoints": [
    "/api/feed",
    "/api/summary",
    "/api/conversation",
    "/api/podcast"
  ],
  "timestamp": "2024-07-29T10:00:00Z",
  "version": "1.1.0"
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Thanks to [James (@capjamesg)](https://github.com/capjamesg) for the original inspiration
- [Hugging Face](https://huggingface.co) for their amazing papers platform
