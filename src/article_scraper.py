"""
Article Scraper Module for extracting full article content from URLs
"""
import time
import logging
import hashlib
import random
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urlparse

# Using newspaper4k for article scraping
import newspaper
from newspaper import Article

# Fix for newspaper3k compatibility
try:
    from newspaper.settings import CACHE_DIRECTORY
except ImportError:
    from newspaper.settings import CF_CACHE_DIRECTORY as CACHE_DIRECTORY
from src.cache_manager import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('article_scraper')

class ArticleScraper:
    """
    Scrapes full article content from URLs using newspaper4k
    Includes caching, rate limiting, and error handling
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None, 
                 cache_dir: Optional[str] = None,
                 request_delay: float = 1.5,
                 max_retries: int = 3,
                 timeout: int = 30,
                 cache_expiry: int = 86400):
        """
        Initialize the article scraper
        
        Args:
            cache_manager: CacheManager instance (optional)
            cache_dir: Directory to store cache files (default: ./cache)
            request_delay: Delay between requests in seconds (default: 1.5)
            max_retries: Maximum number of retry attempts (default: 3)
            timeout: Request timeout in seconds (default: 30)
            cache_expiry: Cache expiry time in seconds (default: 24 hours)
        """
        # Use provided cache manager or create a new one
        self.cache_manager = cache_manager or CacheManager(cache_dir=cache_dir, expiry_time=cache_expiry)
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_request_time = 0
        
    def _cache_key(self, url: str) -> str:
        """
        Generate a cache key for a URL
        
        Args:
            url: Article URL
            
        Returns:
            str: Cache key
        """
        # Use domain and path for cache key to handle URL parameters
        parsed_url = urlparse(url)
        key_parts = f"{parsed_url.netloc}{parsed_url.path}"
        return hashlib.md5(key_parts.encode('utf-8')).hexdigest()
    
    def _enforce_rate_limit(self) -> None:
        """
        Enforce rate limiting between requests
        Adds a random delay between requests to avoid detection
        """
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            # Add a small random component to the delay
            delay = self.request_delay - elapsed + random.uniform(0.1, 0.5)
            time.sleep(delay)
        self.last_request_time = time.time()
    
    def scrape_article(self, url: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Scrape article content from URL with caching and error handling
        
        Args:
            url: Article URL
            force_refresh: Force refresh cache
            
        Returns:
            Dict: Article data including title, text, authors, etc.
        """
        # Check cache first unless force refresh is requested
        if not force_refresh:
            cache_key = self._cache_key(url)
            cached_data = self.cache_manager.get_api_cache(cache_key)
            if cached_data:
                logger.info(f"Using cached article for URL: {url}")
                return cached_data
        
        # Initialize result dictionary
        article_data = {
            'url': url,
            'title': None,
            'text': None,
            'authors': [],
            'published_date': None,
            'top_image': None,
            'summary': None,
            'keywords': [],
            'success': False,
            'error': None
        }
        
        # Implement retry logic
        for attempt in range(self.max_retries):
            try:
                # Enforce rate limiting
                self._enforce_rate_limit()
                
                # Download and parse article
                article = Article(url)
                article.download()
                article.parse()
                
                # Extract natural language features if available
                try:
                    article.nlp()
                except Exception as nlp_error:
                    logger.warning(f"NLP processing failed for {url}: {str(nlp_error)}")
                
                # Update article data
                article_data.update({
                    'title': article.title,
                    'text': article.text,
                    'authors': article.authors,
                    'published_date': article.publish_date.isoformat() if article.publish_date else None,
                    'top_image': article.top_image,
                    'summary': article.summary,
                    'keywords': article.keywords,
                    'success': True,
                    'error': None
                })
                
                # Cache the result
                cache_key = self._cache_key(url)
                self.cache_manager.set_api_cache(cache_key, article_data)
                
                logger.info(f"Successfully scraped article: {url}")
                return article_data
                
            except Exception as e:
                error_msg = f"Attempt {attempt+1}/{self.max_retries} failed for {url}: {str(e)}"
                logger.warning(error_msg)
                article_data['error'] = error_msg
                
                # Add increasing delay between retries
                if attempt < self.max_retries - 1:
                    retry_delay = (attempt + 1) * 2
                    time.sleep(retry_delay)
        
        # All attempts failed
        logger.error(f"Failed to scrape article after {self.max_retries} attempts: {url}")
        return article_data
    
    def scrape_multiple(self, urls: List[str], max_articles: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scrape multiple articles with rate limiting
        
        Args:
            urls: List of article URLs
            max_articles: Maximum number of articles to scrape (optional)
            
        Returns:
            List[Dict]: List of article data dictionaries
        """
        results = []
        
        # Limit number of articles if specified
        if max_articles and max_articles < len(urls):
            urls = urls[:max_articles]
        
        for url in urls:
            article_data = self.scrape_article(url)
            results.append(article_data)
        
        # Return only successful results
        successful_results = [r for r in results if r['success']]
        logger.info(f"Successfully scraped {len(successful_results)}/{len(urls)} articles")
        
        return successful_results