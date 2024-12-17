Here are some curl commands to test both endpoints of the Hugging Face papers RSS feed API:

1. Get the RSS Feed:
```bash
curl http://localhost:8080/feed -i
```
This will show you the full RSS XML feed with headers. The `-i` flag includes the HTTP response headers so you can verify the content type.

2. Check the Feed Status:
```bash
curl http://localhost:8080/feed -H "Accept: application/rss+xml"
```
This explicitly requests RSS XML format.

3. Get the Status Endpoint:
```bash
curl http://localhost:8080/status
```
This will return JSON with the last update time, status, and next scheduled update.

4. Test with Pretty-Printed Output (for status):
```bash
curl http://localhost:8080/status | json_pp
```
This pipes the JSON response through a pretty printer for better readability.

5. Save the RSS Feed to a File:
```bash
curl http://localhost:8080/feed -o huggingface_papers.xml
```
This saves the RSS feed to a local XML file.

6. Check Headers Only:
```bash
curl -I http://localhost:8080/feed
```
This shows just the HTTP headers of the response using the HEAD request.

You can test if the service is responding properly by comparing the timestamps between requests. The feed content should update every 6 hours according to the scheduler we set up.