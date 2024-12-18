# Hugging Face Papers RSS Feed

A serverless RSS feed for papers published on Hugging Face's [papers page](https://huggingface.co/papers). This service scrapes the papers page daily and generates an RSS feed that can be consumed by any RSS reader.

Inspired by [James (@capjamesg)](https://github.com/capjamesg)'s [Hugging Face Papers RSS Feed](https://github.com/capjamesg/hugging-face-papers-rss).

## Features

- Serverless deployment on Vercel
- Redis caching to minimize scraping
- Daily automatic updates via cron job
- Clean RSS feed with paper titles, links and abstracts
- Health check and status endpoints
- CORS enabled for cross-origin requests

## Deployment

### Prerequisites

- A [Vercel](https://vercel.com) account
- A Redis database (e.g. [Upstash](https://upstash.com))
- Go 1.x installed locally for development

### Deploy to Vercel

1. Fork this repository

2. Create a new project on Vercel and import your forked repository

3. Set up the following environment variables in your Vercel project settings:
```env
KV_URL=your_redis_url
KV_REST_API_TOKEN=your_redis_write_token  
KV_REST_API_READ_ONLY_TOKEN=your_redis_read_token
KV_REST_API_URL=your_redis_rest_api_url
```

4. Deploy! Vercel will automatically build and deploy your RSS feed service

The feed will be available at: `https://your-project.vercel.app/api/feed`

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/daily-papers.git
cd daily-papers
```

2. Create a `.env` file in the root directory with your Redis credentials:
```env
KV_URL=your_redis_url
KV_REST_API_TOKEN=your_redis_write_token
KV_REST_API_READ_ONLY_TOKEN=your_redis_read_token
KV_REST_API_URL=your_redis_rest_api_url
```

3. Run the development server:
```bash
vercel dev
```

The local server will be available at `http://localhost:3000`

## API Endpoints

- `/api` - Health check and status
- `/api/feed` - RSS feed of papers
- `/api/update` - Manually trigger feed update (requires authentication)

## Example Response

Health check (`/api`):
```json
{
  "status": "ok",
  "endpoints": ["/api/feed"],
  "cache_status": true,
  "timestamp": "2024-03-20T15:30:45Z",
  "version": "1.0.0"
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Thanks to [James (@capjamesg)](https://github.com/capjamesg) for the original inspiration
- [Hugging Face](https://huggingface.co) for their amazing papers platform
