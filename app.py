import streamlit as st
import feedparser
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast

# ----------------------------
# Load Models (once at startup)
# ----------------------------

# Multilingual summarization model
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
summarizer_model = MBartForConditionalGeneration.from_pretrained(model_name)

# Multilingual sentiment model (outputs 1–5 stars)
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

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
        return " ".join(paragraphs[:20])  # limit to first 20 paragraphs
    except Exception as e:
        return f"Error fetching content: {e}"

def detect_language(text):
    """Detect language of text"""
    try:
        return detect(text)
    except:
        return "unknown"

# Mapping from langdetect code → mBART language code
LANG_MAP = {
    "en": "en_XX",
    "fr": "fr_XX",
    "ar": "ar_AR",
    "ur": "ur_PK"
    # You can add more supported languages here
}

def summarize_text(text, max_len=130, min_len=30, lang_code="en"):
    """Summarize article text using mBART"""
    try:
        mbart_lang = LANG_MAP.get(lang_code, "en_XX")  # default English
        tokenizer.src_lang = mbart_lang
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        summary_ids = summarizer_model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=min_len,
            length_penalty=2.0
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error summarizing: {e}"

def analyze_sentiment(text):
    """Classify sentiment of news as Positive/Negative/Neutral"""
    try:
        result = sentiment_analyzer(text[:512])  # limit input length
        label = result[0]['label']  # e.g., "1 star", "5 stars"

        if label in ["1 star", "2 stars"]:
            news_sentiment = "Negative News (People may dislike it)"
        elif label == "3 stars":
            news_sentiment = "Neutral News"
        else:  # "4 stars" or "5 stars"
            news_sentiment = "Positive News (People may like it)"

        return {
            "label": news_sentiment,
            "score": result[0]['score']
        }
    except Exception as e:
        return {"label": "Error in sentiment", "score": 0.0, "error": str(e)}

# ----------------------------
# Streamlit App
# ----------------------------

st.title("📰 Multilingual News Summarization & Sentiment Analysis")

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
        summary = summarize_text(article['content'], lang_code=lang)
        sentiment = analyze_sentiment(article['content'])

        st.write(f"**Language Detected:** {lang}")
        st.write(f"**Summary:** {summary}")
        st.write(f"**Sentiment Analysis:** {sentiment['label']} (Confidence: {sentiment['score']:.2f})")
