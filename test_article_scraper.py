#!/usr/bin/env python3
"""
Test script for ArticleScraper functionality
"""
import os
import time
import sys
sys.path.append('.')
from src.article_scraper import ArticleScraper
from src.newsapi_verifier import NewsAPIVerifier

# Create cache directory if it doesn't exist
cache_dir = "./data/cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

def test_article_scraper():
    """Test basic article scraper functionality"""
    print("\n=== Testing ArticleScraper ===")
    
    # Initialize article scraper
    scraper = ArticleScraper(cache_dir=cache_dir)
    
    # Test URLs (mix of different news sources)
    test_urls = [
        "https://www.bbc.com/news/world-us-canada-68720274",  # BBC
        "https://www.theguardian.com/environment/2023/apr/27/climate-crisis-ipcc-carbon-emissions",  # Guardian
        "https://www.nytimes.com/2023/04/20/climate/global-warming-temperature-record.html",  # NYT
        "https://edition.cnn.com/2023/04/13/politics/us-military-climate-change/index.html",  # CNN
    ]
    
    # Test single article scraping
    print("\n--- Testing Single Article Scraping ---")
    start_time = time.time()
    article = scraper.scrape_article(test_urls[0])
    elapsed = time.time() - start_time
    
    print(f"URL: {article['url']}")
    print(f"Title: {article['title']}")
    print(f"Success: {article['success']}")
    if article['success']:
        print(f"Text length: {len(article['text'])} characters")
        print(f"Authors: {article['authors']}")
        print(f"Published date: {article['published_date']}")
        print(f"Summary: {article['summary'][:150]}..." if article['summary'] else "No summary available")
    else:
        print(f"Error: {article['error']}")
    print(f"Time taken: {elapsed:.2f} seconds")
    
    # Test cached retrieval (should be faster)
    print("\n--- Testing Cache Functionality ---")
    start_time = time.time()
    cached_article = scraper.scrape_article(test_urls[0])
    cached_elapsed = time.time() - start_time
    print(f"Cached retrieval time: {cached_elapsed:.2f} seconds")
    print(f"Speed improvement: {elapsed/cached_elapsed:.1f}x faster")
    
    # Test multiple article scraping
    print("\n--- Testing Multiple Article Scraping ---")
    start_time = time.time()
    articles = scraper.scrape_multiple(test_urls)
    multi_elapsed = time.time() - start_time
    
    print(f"Successfully scraped {len(articles)}/{len(test_urls)} articles")
    print(f"Average time per article: {multi_elapsed/len(test_urls):.2f} seconds")
    
    # Print summary of each article
    for i, article in enumerate(articles, 1):
        print(f"\nArticle {i}: {article['title']}")
        print(f"Source: {article['url']}")
        print(f"Text length: {len(article['text'])} characters")
        print(f"Keywords: {', '.join(article['keywords'][:5])}..." if article['keywords'] else "No keywords")

def test_integration_with_newsapi():
    """Test integration with NewsAPIVerifier"""
    print("\n=== Testing Integration with NewsAPIVerifier ===")
    
    # Initialize NewsAPIVerifier with article scraping enabled
    verifier = NewsAPIVerifier(cache_dir=cache_dir, use_full_content=True)
    
    # Test query
    query = "nobel prize winner"
    print(f"\nQuery: {query}")
    
    # Get verification score and articles
    score, articles = verifier.get_verification_score(query, max_results=3)
    
    print(f"Verification score: {score:.2f}")
    print(f"Found {len(articles)} supporting articles")
    
    # Print details of each article
    for i, article in enumerate(articles, 1):
        print(f"\nArticle {i}: {article['title']}")
        print(f"Source: {article['source']}")
        print(f"URL: {article['url']}")
        print(f"Has full content: {article.get('has_full_content', False)}")
        
        # Print snippet from full content if available
        if article.get('has_full_content') and article.get('full_text'):
            text_snippet = article['full_text'][:150] + "..." if len(article['full_text']) > 150 else article['full_text']
            print(f"Full text snippet: {text_snippet}")
        else:
            print(f"NewsAPI snippet: {article.get('content', 'No content available')[:150]}...")

if __name__ == "__main__":
    # Run tests
    test_article_scraper()
    test_integration_with_newsapi()