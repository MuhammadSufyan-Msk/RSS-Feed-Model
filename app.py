import streamlit as st
import requests
from bs4 import BeautifulSoup

# Safe install check for feedparser
import sys, subprocess
try:
    import feedparser
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "feedparser"])
    import feedparser

from langdetect import detect

# ----------------------------
# Helper Functions
# ----------------------------

def fetch_rss_articles(feed_urls, limit=5):
    """Fetch news articles from RSS feeds"""
    articles = []
    for url in feed_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:limit]:
            content = fetch_article_content(entry.link)
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "content": content
            })
    return articles

def fetch_article_content(url):
    """Scrape main text content from article"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return " ".join(paragraphs[:10])  # shorter for testing
    except Exception as e:
        return f"Error fetching content: {e}"

def detect_language(text):
    """Detect language of text"""
    try:
        return detect(text)
    except:
        return "unknown"

# ----------------------------
# Streamlit App (Minimal Mode)
# ----------------------------

st.title("📰 RSS Feed Reader (Deploy-Test Mode)")

st.info("⚡ This is a lightweight deployment test. Summarization and sentiment analysis are disabled for now.")

feed_urls = [
    "https://techcrunch.com/feed/",
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml"
]

if st.button("Fetch Latest News"):
    articles = fetch_rss_articles(feed_urls, limit=3)
    for article in articles:
        st.subheader(article['title'])
        st.write(article['link'])

        lang = detect_language(article['content'])
        st.write(f"**Language Detected:** {lang}")
        st.write(article['content'][:500] + "...")
