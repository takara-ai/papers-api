from fastapi import FastAPI, Response
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import xml.etree.ElementTree as ET
from apscheduler.schedulers.background import BackgroundScheduler
from typing import Dict, List
import logging
from time import mktime, time
from email.utils import formatdate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global variable to store the feed data
current_feed: Dict = {}
last_update: datetime = None

class PaperScraper:
    BASE_URL = "https://huggingface.co/papers"

    @staticmethod
    def extract_abstract(url: str) -> str:
        """Extract abstract from paper page."""
        try:
            page = requests.get(url, timeout=10)
            soup = BeautifulSoup(page.content, "html.parser")
            abstract_div = soup.find("div", {"class": "pb-8 pr-4 md:pr-16"})
            if not abstract_div:
                return ""
            
            abstract = abstract_div.text
            if abstract.startswith("Abstract\n"):
                abstract = abstract[len("Abstract\n"):]
            return abstract.replace("\n", " ").strip()
        except Exception as e:
            logger.error(f"Failed to extract abstract for {url}: {e}")
            return ""

    @staticmethod
    def scrape_papers() -> List[Dict]:
        """Scrape papers from Hugging Face."""
        try:
            page = requests.get(PaperScraper.BASE_URL, timeout=10)
            soup = BeautifulSoup(page.content, "html.parser")
            h3s = soup.find_all("h3")
            
            papers = []
            for h3 in h3s:
                a = h3.find("a")
                if not a:
                    continue
                    
                title = a.text.strip()
                link = a["href"]
                url = f"https://huggingface.co{link}"
                abstract = PaperScraper.extract_abstract(url)
                
                papers.append({
                    "title": title,
                    "url": url,
                    "abstract": abstract,
                    "pub_date": datetime.utcnow().isoformat()
                })
            
            return papers
        except Exception as e:
            logger.error(f"Failed to scrape papers: {e}")
            return []

def generate_rss(papers: List[Dict]) -> str:
    """Generate RSS XML from papers data."""
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    
    # Add channel metadata
    title = ET.SubElement(channel, "title")
    title.text = "Hugging Face Papers RSS Feed"
    
    link = ET.SubElement(channel, "link")
    link.text = PaperScraper.BASE_URL
    
    last_build = ET.SubElement(channel, "lastBuildDate")
    last_build.text = formatdate(mktime(datetime.utcnow().timetuple()))
    
    desc = ET.SubElement(channel, "description")
    desc.text = "Latest papers from Hugging Face"
    
    # Add items
    for paper in papers:
        item = ET.SubElement(channel, "item")
        
        item_title = ET.SubElement(item, "title")
        item_title.text = paper["title"]
        
        item_link = ET.SubElement(item, "link")
        item_link.text = paper["url"]
        
        item_desc = ET.SubElement(item, "description")
        item_desc.text = paper["abstract"]
        
        item_date = ET.SubElement(item, "pubDate")
        dt = datetime.fromisoformat(paper["pub_date"])
        item_date.text = formatdate(mktime(dt.timetuple()))
        
        item_guid = ET.SubElement(item, "guid")
        item_guid.text = paper["url"]
    
    return ET.tostring(rss, encoding="unicode")

def update_feed():
    """Update the global feed data."""
    global current_feed, last_update
    
    logger.info("Starting feed update...")
    papers = PaperScraper.scrape_papers()
    
    if papers:
        current_feed = generate_rss(papers)
        last_update = datetime.utcnow()
        logger.info(f"Feed updated successfully with {len(papers)} papers")
    else:
        logger.warning("No papers found in update")

# Set up scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(update_feed, 'interval', hours=6)
scheduler.start()

# Initial feed update
update_feed()

@app.get("/feed")
async def get_feed():
    """Serve the RSS feed."""
    start_time = time()
    response = Response(
        content=current_feed,
        media_type="application/rss+xml"
    )
    elapsed_time = time() - start_time
    logger.info(f"Feed request processed in {elapsed_time * 1_000_000:.2f} microseconds")
    return response

@app.get("/status")
async def get_status():
    """Get the feed status."""
    start_time = time()
    response = {
        "last_update": last_update,
        "status": "active",
        "next_update": scheduler.get_jobs()[0].next_run_time
    }
    elapsed_time = time() - start_time
    logger.info(f"Status request processed in {elapsed_time * 1_000_000:.2f} microseconds")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)