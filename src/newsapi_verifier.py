"""
NewsAPI Integration Module for Real-Time Verification
"""
import os
import json
import time
from datetime import datetime, timedelta
import requests
from newsapi import NewsApiClient

# Import cache manager and article scraper
from .cache_manager import CacheManager
from .article_scraper import ArticleScraper

class NewsAPIVerifier:
    """NewsAPI integration for real-time verification of news"""
    
    # List of reputable news sources
    REPUTABLE_SOURCES = [
        # Global sources
        'bbc-news', 'cnn', 'the-washington-post', 'the-new-york-times', 
        'reuters', 'associated-press', 'the-economist', 'the-wall-street-journal',
        'al-jazeera-english', 'bloomberg', 'financial-times', 'the-guardian-uk',
        'abc-news', 'nbc-news', 'cbs-news',
        # Indian sources
        'the-hindu', 'the-times-of-india', 'the-indian-express', 'hindustan-times',
        'ndtv', 'india-today'
    ]
    
    def __init__(self, api_key=None, cache_dir=None, use_sources=True, use_full_content=True):
        """
        Initialize the NewsAPI verifier
        
        Args:
            api_key (str): NewsAPI API key
            cache_dir (str): Directory to cache API responses
            use_sources (bool): Whether to filter by reputable sources
            use_full_content (bool): Whether to scrape full article content
        """
        # Use environment variable if API key not provided, or use the default key
        self.api_key = api_key or os.environ.get('NEWSAPI_KEY') or 'ea70866cd4454c3db02541b742b4d3e0'
        
        if not self.api_key:
            print("Warning: NewsAPI key not provided. Using default key.")
        
        self.newsapi = NewsApiClient(api_key=self.api_key) if self.api_key else None
        
        # Set up caching with the new cache manager
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        
        # Whether to filter by reputable sources
        self.use_sources = use_sources
        
        # Whether to scrape full article content
        self.use_full_content = use_full_content
        
        # Initialize article scraper
        self.article_scraper = ArticleScraper(cache_manager=self.cache_manager)
        
        # Keep backward compatibility
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _load_cache(self):
        """Load cache from file"""
        # Define cache_file path if not already set
        if not hasattr(self, 'cache_file') and self.cache_dir:
            self.cache_file = os.path.join(self.cache_dir, 'newsapi_cache.json')
        
        if hasattr(self, 'cache_file') and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        # Define cache_file path if not already set
        if not hasattr(self, 'cache_file') and self.cache_dir:
            self.cache_file = os.path.join(self.cache_dir, 'newsapi_cache.json')
        
        if hasattr(self, 'cache_file') and self.cache_file:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache if hasattr(self, 'cache') else {}, f)
            except IOError:
                print("Warning: Could not save cache to file")
    
    def _cache_key(self, query, days=7):
        """Generate cache key for a query"""
        return f"{query}_{days}"
        
    def _calculate_relevance_score(self, query, title, description, content):
        """
        Calculate relevance score based on query terms in title, description, and content
        
        Args:
            query (str): Search query
            title (str): Article title
            description (str): Article description
            content (str): Article content
            
        Returns:
            float: Relevance score (0-1)
        """
        # Convert to lowercase for case-insensitive matching
        query_lower = query.lower()
        title_lower = title.lower()
        description_lower = description.lower() if description else ""
        content_lower = content.lower() if content else ""
        
        # Split query into terms
        query_terms = query_lower.split()
        
        # Calculate scores based on term presence and position
        title_score = 0
        desc_score = 0
        content_score = 0
        
        # Title has highest weight (exact match is best)
        if query_lower in title_lower:
            title_score = 1.0
        else:
            # Count matching terms in title
            for term in query_terms:
                if term in title_lower:
                    title_score += 0.2  # Each term match adds 0.2
        
        # Description has medium weight
        if description_lower:
            if query_lower in description_lower:
                desc_score = 0.8
            else:
                # Count matching terms in description
                for term in query_terms:
                    if term in description_lower:
                        desc_score += 0.1  # Each term match adds 0.1
        
        # Content has lowest weight
        if content_lower:
            if query_lower in content_lower:
                content_score = 0.6
            else:
                # Count matching terms in content
                for term in query_terms:
                    if term in content_lower:
                        content_score += 0.05  # Each term match adds 0.05
        
        # Combine scores with weights (title most important, then description, then content)
        combined_score = (title_score * 0.6) + (desc_score * 0.3) + (content_score * 0.1)
        
        # Normalize to 0-1 range
        return min(combined_score, 1.0)
    
    def _optimize_query(self, query):
        """
        Optimize the query for better NewsAPI results
        
        Args:
            query (str): Original search query
            
        Returns:
            str: Optimized query
        """
        # Remove common words that don't add search value
        stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                     'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 
                     'through', 'over', 'before', 'after', 'between', 'under']
        
        # Split query into terms
        terms = query.lower().split()
        
        # Filter out stop words
        filtered_terms = [term for term in terms if term not in stop_words]
        
        # If we have filtered out too many terms, use original query
        if len(filtered_terms) < 2 and len(terms) > 2:
            return query
            
        # Add quotes around multi-word phrases for exact matching
        if len(filtered_terms) > 1:
            optimized_query = f'"{" ".join(filtered_terms)}"'
            
            # Also add individual important terms for broader matching
            important_terms = [term for term in filtered_terms if len(term) > 3]
            if important_terms:
                optimized_query += " OR " + " OR ".join(important_terms)
                
            return optimized_query
        
        # If only one term or empty, return original query
        return query if filtered_terms else query
    
    def search_news(self, query, max_results=5, days=7):
        """
        Search for news articles related to the query
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            days (int): Number of days to look back
            
        Returns:
            list: List of news articles
        """
        if not self.newsapi:
            return []
        
        # Create cache key data
        cache_key_data = {
            'query': query,
            'days': days,
            'type': 'newsapi_search',
            'use_sources': self.use_sources
        }
        
        # Check cache first using the cache manager
        cached_result = self.cache_manager.get_api_cache(cache_key_data)
        if cached_result:
            return cached_result[:max_results]
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Optimize the query for better results
        optimized_query = self._optimize_query(query)
        
        try:
            # Prepare API call parameters
            api_params = {
                'q': optimized_query,
                'from_param': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sort_by': 'relevancy',
                'page_size': 100  # Increased from max_results to get more candidates for filtering
            }
            
            # Add sources parameter if filtering by reputable sources is enabled
            if self.use_sources and self.REPUTABLE_SOURCES:
                # Join sources with commas for the API call (limit to 20 sources as per API limits)
                sources_param = ','.join(self.REPUTABLE_SOURCES[:20])
                api_params['sources'] = sources_param
            
            # Search for articles
            response = self.newsapi.get_everything(**api_params)
            
            articles = response.get('articles', [])
            
            # Process articles and calculate relevance scores
            processed_articles = []
            for article in articles:
                # Extract article data
                title = article.get('title', '')
                description = article.get('description', '')
                content = article.get('content', '')
                
                # Calculate relevance score based on query terms in title, description, and content
                relevance_score = self._calculate_relevance_score(query, title, description, content)
                
                processed_article = {
                    'title': title,
                    'description': description,
                    'content': content,
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'relevance_score': relevance_score  # Add relevance score
                }
                processed_articles.append(processed_article)
                
            # Sort articles by relevance score (highest first)
            processed_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Cache the results using cache manager instead
            self.cache_manager.set_api_cache(cache_key_data, processed_articles)
            
            # Initialize self.cache if it doesn't exist
            if not hasattr(self, 'cache'):
                self.cache = {}
            
            # Return only the top max_results (5) most relevant articles
            return processed_articles[:max_results]
            
        except Exception as e:
            print(f"Error searching NewsAPI: {e}")
            return []
    
    def get_verification_score(self, query, max_results=5):
        """
        Get verification score based on NewsAPI results
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            tuple: (score, articles)
                score: 0-1 score (higher means more likely real)
                articles: List of supporting articles
        """
        articles = self.search_news(query, max_results=max_results)
        
        # If no articles found, return neutral score
        if not articles:
            return 0.5, []
        
        # Scrape full article content if enabled
        if self.use_full_content:
            # Extract URLs from articles
            urls = [article['url'] for article in articles if article.get('url')]
            
            # Scrape full content for each article
            try:
                # Limit to max_results to avoid excessive scraping
                scraped_articles = self.article_scraper.scrape_multiple(urls, max_articles=max_results)
                
                # Create a mapping of URL to scraped content
                scraped_content = {article['url']: article for article in scraped_articles if article['success']}
                
                # Enhance articles with full content
                for article in articles:
                    url = article.get('url', '')
                    if url in scraped_content:
                        # Add full text content
                        article['full_text'] = scraped_content[url].get('text', '')
                        article['full_title'] = scraped_content[url].get('title', '')
                        article['authors'] = scraped_content[url].get('authors', [])
                        article['published_date'] = scraped_content[url].get('published_date', '')
                        article['has_full_content'] = True
                    else:
                        # Mark as not having full content
                        article['has_full_content'] = False
            except Exception as e:
                print(f"Error scraping articles: {e}")
                # Continue with NewsAPI snippets if scraping fails
                for article in articles:
                    article['has_full_content'] = False
        
        # Simple scoring: more articles = higher score
        # This will be improved with similarity calculation
        score = min(len(articles) / max_results, 1.0)
        
        return score, articles


if __name__ == "__main__":
    # Test the NewsAPI verifier
    api_key = os.environ.get('NEWSAPI_KEY')
    cache_dir = "../data/cache"
    
    verifier = NewsAPIVerifier(api_key=api_key, cache_dir=cache_dir)
    
    # Test queries
    test_queries = [
        "Climate change causes extreme weather",
        "Vaccines cause autism",
        "COVID-19 pandemic"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        score, articles = verifier.get_verification_score(query)
        print(f"Verification score: {score:.2f}")
        print(f"Found {len(articles)} supporting articles:")
        
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   URL: {article['url']}")
            print(f"   Published: {article['published_at']}")