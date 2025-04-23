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

## Caching

This service leverages Vercel's Edge Caching. Responses are cached globally with a `Cache-Control` header set to expire around 6 AM UTC daily. This means fresh content is generated approximately once per day.

- **Cache Duration:** Calculated dynamically to expire at the next 6:00 AM UTC.
- **Cache Control:** `public, max-age=0, s-maxage=<seconds_until_6am_utc>`
- **Invalidation:** Cache automatically expires based on `s-maxage` or when a new deployment occurs.

## Performance Comparison

### Sequential `curl` Benchmark

A basic sequential benchmark using `curl` from a single location provides a rough latency comparison.

| Version        | Request # | Time (seconds)  | Notes                |
| -------------- | --------- | --------------- | -------------------- |
| Redis (Legacy) | 1         | 0.022432        | @ `papers.takara.ai` |
| Redis (Legacy) | 2         | 0.020673        |                      |
| Redis (Legacy) | 3         | 0.021362        |                      |
| Redis (Legacy) | 4         | 0.022083        |                      |
| Redis (Legacy) | 5         | 0.016121        |                      |
| Redis (Legacy) | 6         | 0.018705        |                      |
| Redis (Legacy) | 7         | 0.017078        |                      |
| Redis (Legacy) | 8         | 0.018801        |                      |
| Redis (Legacy) | 9         | 0.018036        |                      |
| Redis (Legacy) | 10        | 0.026124        |                      |
| Vercel Edge    | _1-10_    | _(To be added)_ | _(New version)_      |

_Note: This sequential `curl` test does not represent performance under concurrent load. Vercel Edge Caching is expected to offer better global performance._

## Example Response

Health check (`/api`):

```json
{
  "status": "ok",
  "endpoints": ["/api/feed", "/api/summary"],
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
