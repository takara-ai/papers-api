# Hugging Face Papers RSS Feed

A serverless RSS feed service for papers published on Hugging Face's [papers page](https://huggingface.co/papers). This service scrapes the papers page, generates an RSS feed, and provides additional features like paper summaries and audio podcasts.

## Features

- Serverless deployment on Vercel
- Redis caching to minimize scraping
- Daily automatic updates via cron job
- Multiple feed formats:
  - Standard RSS feed with paper titles, links, and abstracts
  - LLM-powered summary feed of the latest papers
  - Conversation-style summaries for podcast generation
  - Audio podcast generation with Cloudflare R2 storage
- Health check and status endpoints
- CORS enabled for cross-origin requests
- Structured logging with slog

## Deployment

### Prerequisites

- A [Vercel](https://vercel.com) account
- A Redis database (e.g. [Upstash](https://upstash.com))
- Cloudflare R2 account (for podcast storage)
- Go 1.x installed locally for development

### Deploy to Vercel

1. Fork this repository

2. Create a new project on Vercel and import your forked repository

3. Set up the following environment variables in your Vercel project settings:

```env
# Redis Configuration
KV_URL=your_redis_url
KV_REST_API_TOKEN=your_redis_write_token
KV_REST_API_READ_ONLY_TOKEN=your_redis_read_token
KV_REST_API_URL=your_redis_rest_api_url

OPENAI_API_KEY=your_hugging_face_api_key

# Security
UPDATE_KEY=your_secret_key_here

# Cloudflare R2 Configuration (for podcast storage)
R2_ENDPOINT=your_r2_endpoint
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_BUCKET_NAME=your_r2_bucket_name
R2_PUBLIC_URL=your_r2_public_url
```

4. Deploy! Vercel will automatically build and deploy your service

The main feed will be available at: `https://your-project.vercel.app/api/feed`

### Local Development

1. Clone the repository:

```bash
git clone https://github.com/yourusername/hf-papers-rss.git
cd hf-papers-rss
```

2. Create a `.env` file in the root directory with your credentials (see environment variables above)

3. Run the development server:

```bash
go run main.go
```

The local server will be available at `http://localhost:3000`

## API Endpoints

- `/api` - Health check and status
- `/api/feed` - Standard RSS feed of papers
- `/api/summary` - RSS feed with LLM-generated summaries
- `/api/conversation` - JSON endpoint for conversation-style summaries
- `/api/podcast` - JSON endpoint returning the latest podcast URL
- `/api/update-cache` - Manually trigger feed update (requires authentication)

## Manual Cache Updates

To enable secure manual cache updates, you need to set an `UPDATE_KEY` environment variable:

1. Add to your Vercel environment variables:

```env
UPDATE_KEY=your_secret_key_here
```

2. To manually trigger a cache update, use the following curl command:

```bash
curl -X GET \
  https://your-project.vercel.app/api/update-cache \
  -H 'X-Update-Key: your_secret_key_here'
```

Successful response:

```json
{
  "status": "Cache updated successfully",
  "timestamp": "2024-03-20T15:30:45Z"
}
```

## Example Responses

Health check (`/api`):

```json
{
  "status": "ok",
  "endpoints": ["/api/feed", "/api/summary", "/api/conversation", "/api/podcast"],
  "cache_status": true,
  "timestamp": "2024-03-20T15:30:45Z",
  "version": "1.0.0"
}
```

Podcast endpoint (`/api/podcast`):

```json
{
  "url": "https://your-r2-public-url.com/latest-podcast.mp3"
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Thanks to [James (@capjamesg)](https://github.com/capjamesg) for the original inspiration
- [Hugging Face](https://huggingface.co) for their amazing papers platform