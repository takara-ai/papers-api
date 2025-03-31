#!/usr/bin/env python3
"""
RSS to Markdown converter for AI research papers feed with LLM summarization
"""
import requests
from datetime import datetime
import xml.etree.ElementTree as ET
from groq import Groq


def fetch_rss(url):
    """Fetch RSS feed content from URL"""
    response = requests.get(url)
    response.raise_for_status()  # Raise exception for HTTP errors
    return response.text


def parse_rss_to_markdown(xml_content):
    """Parse RSS XML content and convert to markdown format"""
    root = ET.fromstring(xml_content)
    
    # Extract channel info
    channel = root.find('channel')
    title = channel.find('title').text
    description = channel.find('description').text
    last_build_date = channel.find('lastBuildDate').text
    
    # Format date
    try:
        parsed_date = datetime.strptime(last_build_date, '%a, %d %b %Y %H:%M:%S %z')
        formatted_date = parsed_date.strftime('%Y-%m-%d')
    except ValueError:
        formatted_date = last_build_date  # Fallback to original format
    
    # Create markdown header
    markdown = f"# {title}\n\n"
    markdown += f"*{description}*\n\n"
    markdown += f"*Last updated: {formatted_date}*\n\n"
    markdown += "---\n\n"
    
    # Process each item
    for item in channel.findall('item'):
        item_title = item.find('title').text.replace('\n', ' ').strip()
        item_link = item.find('link').text
        item_desc = item.find('description').text
        
        markdown += f"## [{item_title}]({item_link})\n\n"
        markdown += f"{item_desc}\n\n"
        markdown += "---\n\n"
    
    return markdown


def summarize_with_groq(markdown_content):
    """Summarize the markdown content using Groq API"""
    client = Groq(api_key="")
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "user",
                "content": """Create a brief morning briefing on these AI research papers, written in a conversational style for busy professionals. Focus on what's new and what it means for businesses and society.

Format:
1. Morning Headline (1 sentence)
2. What's New (2-3 sentences, written like you're explaining it to a friend over coffee, with citations to papers as [Paper Name](link))
   • Cover all papers in a natural, flowing narrative
   • Group related papers together
   • Include key metrics and outcomes
   • Keep the tone light and engaging

Keep it under 200 words. Focus on outcomes and implications, not technical details. Write like you're explaining it to a friend over coffee.

Below are the paper abstracts and information in markdown format:

""" + markdown_content
            }
        ],
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95,
        stream=False,
        stop=None,
    )
    
    # Extract only the content after <think> tags
    response = completion.choices[0].message.content
    if "<think>" in response:
        response = response.split("</think>")[-1].strip()
    return response


def main():
    RSS_URL = 'https://papers.takara.ai/api/feed'
    OUTPUT_FILE = 'papers_summary.md'
    
    try:
        xml_content = fetch_rss(RSS_URL)
        markdown = parse_rss_to_markdown(xml_content)
        summary = summarize_with_groq(markdown)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"Summarization successful! Output saved to {OUTPUT_FILE}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
