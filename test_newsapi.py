#!/usr/bin/env python3
"""
Test script for NewsAPI integration
"""
import os
import sys
from src.newsapi_verifier import NewsAPIVerifier

def main():
    """Test NewsAPI integration"""
    print("=== Testing NewsAPI Integration ===")
    
    # Create cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Create cache/api directory
    cache_api_dir = os.path.join(cache_dir, 'api')
    if not os.path.exists(cache_api_dir):
        os.makedirs(cache_api_dir)
    
    # Get API key from environment or use default
    api_key = os.environ.get('NEWSAPI_KEY')
    
    # Initialize NewsAPI verifier with source filtering enabled
    verifier = NewsAPIVerifier(api_key=api_key, cache_dir=cache_dir, use_sources=True)
    
    # Test search with reputable sources
    query = "climate change"
    print(f"\nSearching for: '{query}' (with reputable sources filter)")
    
    try:
        articles = verifier.search_news(query, max_results=5)
        
        # Also test without source filtering for comparison
        verifier_no_filter = NewsAPIVerifier(api_key=api_key, cache_dir=cache_dir, use_sources=False)
        print(f"\nSearching for: '{query}' (without source filtering)")
        articles_no_filter = verifier_no_filter.search_news(query, max_results=5)
        
        # Display results with source filtering
        if articles:
            print(f"Found {len(articles)} articles from reputable sources:")
            for i, article in enumerate(articles, 1):
                # Handle both dictionary and string article formats
                if isinstance(article, dict):
                    title = article.get('title', 'No title')
                    source = article.get('source', 'Unknown')
                    source_name = source if isinstance(source, str) else article.get('source', {}).get('name', 'Unknown')
                    published = article.get('published_at', article.get('publishedAt', 'Unknown'))
                    url = article.get('url', 'No URL')
                else:
                    # If article is a string, just print it
                    title = article
                    source_name = "Unknown"
                    published = "Unknown"
                    url = "No URL"
                
                print(f"\n{i}. {title}")
                print(f"   Source: {source_name}")
                print(f"   Published: {published}")
                print(f"   URL: {url}")
        else:
            print("No articles found from reputable sources.")
            
        # Display results without source filtering
        if articles_no_filter:
            print(f"\nFound {len(articles_no_filter)} articles without source filtering:")
            for i, article in enumerate(articles_no_filter, 1):
                if isinstance(article, dict):
                    title = article.get('title', 'No title')
                    source = article.get('source', 'Unknown')
                    source_name = source if isinstance(source, str) else article.get('source', {}).get('name', 'Unknown')
                    published = article.get('published_at', article.get('publishedAt', 'Unknown'))
                    url = article.get('url', 'No URL')
                else:
                    title = article
                    source_name = "Unknown"
                    published = "Unknown"
                    url = "No URL"
                
                print(f"\n{i}. {title}")
                print(f"   Source: {source_name}")
                print(f"   Published: {published}")
                print(f"   URL: {url}")
        else:
            print("\nNo articles found without source filtering.")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== NewsAPI Integration Test Complete ===")

if __name__ == "__main__":
    main()