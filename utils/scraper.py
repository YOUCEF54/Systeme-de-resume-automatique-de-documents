import requests
from bs4 import BeautifulSoup
import re

def scrape_wikipedia_article(url: str) -> str:
    """
    Scrapes the content of an Arabic Wikipedia article.
    
    Args:
        url (str): The URL of the Wikipedia article.
        
    Returns:
        str: The extracted text content of the article.
    """
    try:
        headers = {
            'User-Agent': 'ArabicSummarizerBot/1.0 (contact@example.com)'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get the main content div
        content_div = soup.find('div', {'id': 'mw-content-text'})
        
        if not content_div:
            return ""

        # Extract paragraphs
        paragraphs = content_div.find_all('p')
        text = ""
        for p in paragraphs:
            text += p.get_text() + "\n"
            
        # Basic cleaning
        # Remove citation numbers like [1], [2]
        text = re.sub(r'\[\d+\]', '', text)
        
        return text.strip()

    except Exception as e:
        print(f"Error scraping URL {url}: {e}")
        return ""
