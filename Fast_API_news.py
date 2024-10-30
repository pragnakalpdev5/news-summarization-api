from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import requests
import os
import json
from pydantic import BaseModel
import re
from datetime import datetime
from openai import OpenAI
import asyncio
from dotenv import load_dotenv

load_dotenv(".env")

app = FastAPI(title="Intelligent News Summarization and Analysis API")

# Configuration - API keys are set as environment variables for security
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
print("NEWS_API_KEY:", NEWS_API_KEY)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY:", OPENAI_API_KEY)

client = OpenAI(api_key=OPENAI_API_KEY,  # this is also the default, it can be omitted
)


# Data models to structure the article data and summarized responses
class Article(BaseModel):
    title: str
    date: datetime
    content: str
    source: str

class SummarizedArticle(BaseModel):
    title: str
    date: datetime
    summary: str
    sentiment: str
    topics: List[str]

# Configuring rate limiting for API calls
RATE_LIMIT = 100  # Requests per hour limit
requests_made = 0  # Counter to track requests made within the rate limit period

# Utility to manage rate limiting
async def rate_limit():
    global requests_made
    # If the limit is reached, wait for a minute to reset
    if requests_made >= RATE_LIMIT:
        await asyncio.sleep(60)
        requests_made = 0
    requests_made += 1

# Function to fetch articles from NewsAPI based on a search query
async def fetch_articles(query: str, page_size: int = 10) -> List[Article]:
    # Construct NewsAPI URL with query and API key
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&apiKey={NEWS_API_KEY}"
    print("url:", url)
    await rate_limit()  # Enforce rate limit
    
    # Perform the API request
    response = requests.get(url)
    
    # Handle non-successful responses
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch articles from NewsAPI")
    
    # Extract article data and structure it into Article objects
    articles_data = response.json().get('articles', [])
    return [
        Article(
            title=article['title'],
            date=article['publishedAt'],
            content=article['content'] or article['description'],
            source=article['source']['name']
        ) for article in articles_data
    ]

# Text preprocessing to clean HTML tags and unwanted characters
def preprocess_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)  # Remove any HTML tags
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = text.strip()  # Clean up extra whitespace
    return text

# Summarize news content function using chat completion with gpt-4o-mini model
async def summarize_content(content: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides concise summaries of news articles."},
        {"role": "user", "content": f"Summarize the following article:\n{content}"}
    ]
    try:
        # OpenAI API call with chat completion format for summarization
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150,
            temperature=0.5
        )
        print("response summary", response.choices[0].message.content)

        # Extract and return the summarized text
        return response.choices[0].message.content.strip()
    except client.error.OpenAIError:
        # Handle errors related to OpenAI API
        raise HTTPException(status_code=500, detail="Error occurred during article summarization")

# Sentiment analysis function using chat completion with gpt-4o-mini model 
async def analyze_sentiment(content: str) -> str:
    messages = [
        {"role": "system", "content": "You are an assistant specialized in analyzing sentiment of news articles (Such as Positive, Neutral, Negative)."},
        {"role": "user", "content": f"Analyze the sentiment of the following text:\n{content}"}
    ]
    try:
        # OpenAI API call for sentiment analysis
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=20
        )
        print("response analyze_sentiment:", response)

        # Extract and return the sentiment result
        return response.choices[0].message.content.strip()
    except client.error.OpenAIError:
        raise HTTPException(status_code=500, detail="Error occurred during sentiment analysis")

# Topic classification function using chat completion with gpt-4o-mini model
async def classify_topics(content: str) -> List[str]:
    messages = [
        {"role": "system", "content": "You are an assistant that identifies main topics in a text. Note: Key Topics should be comma seperated."},
        {"role": "user", "content": f"Identify key topics for the following text:\n{content}"}
    ]
    try:
        # OpenAI API call for topic identification
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=50
        )
        print("response classify_topics:", response)

        # Extract and split topics into a list
        return response.choices[0].message.content.strip().split(', ')
    except client.error.OpenAIError:
        raise HTTPException(status_code=500, detail="Error occurred during topic classification")
    
# Main endpoint to fetch and process articles
@app.get("/search", response_model=List[SummarizedArticle])
async def search_news(query: str = Query(..., min_length=3, max_length=100), page_size: int = 5):
    # Step 1: Fetch articles based on query and preprocess the content
    articles = await fetch_articles(query, page_size=page_size)
    
    # Step 2: Process each article using summarization, sentiment analysis, and topic classification
    summarized_articles = []
    for article in articles:
        processed_content = preprocess_text(article.content)
        summary = await summarize_content(processed_content)
        sentiment = await analyze_sentiment(processed_content)
        topics = await classify_topics(processed_content)
        
        # Compile processed data into SummarizedArticle model format
        summarized_articles.append(SummarizedArticle(
            title=article.title,
            date=article.date,
            summary=summary,
            sentiment=sentiment,
            topics=topics
        ))
    return summarized_articles
