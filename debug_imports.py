import sys
import os

# Emulate the backend runtime path
sys.path.append(os.getcwd())

try:
    print("Attempting imports...")
    import sys
    print(f"Path: {sys.path}")
    
    from wiki_scraper import scrape_wikipedia_article
    print("Scraper imported successfully (from wiki_scraper).")

    from api.routes import router
    print("Routes imported successfully.")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Error: {e}")

try:
    import requests
    import bs4
    print("Dependencies (requests, bs4) are present.")
except ImportError as e:
    print(f"Missing dependency: {e}")
