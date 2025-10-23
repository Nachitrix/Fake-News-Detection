"""
Integrated Verifier Module - Combines Article Scraper, NewsAPI, and Similarity Calculator
"""
import logging
import os
from typing import Dict, List, Optional, Any, Tuple

from src.article_scraper import ArticleScraper
from src.similarity_calculator import SimilarityCalculator
from src.newsapi_verifier import NewsAPIVerifier
from src.cache_manager import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('integrated_verifier')

class IntegratedVerifier:
    """
    Integrates NewsAPI, article scraping, and similarity calculation to verify claims
    against multiple sources.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 use_semantic_similarity: bool = True,
                 use_section_comparison: bool = True,
                 use_claim_decomposition: bool = True,
                 use_source_credibility: bool = True,
                 use_reputable_sources: bool = True):
        """
        Initialize the integrated verifier
        
        Args:
            api_key: NewsAPI API key (default: None, will use env var or default key)
            cache_dir: Directory to store cache files (default: ./cache)
            use_semantic_similarity: Whether to use semantic similarity (default: True)
            use_section_comparison: Whether to use section-based comparison (default: True)
            use_claim_decomposition: Whether to decompose claims (default: True)
            use_source_credibility: Whether to consider source credibility (default: True)
            use_reputable_sources: Whether to filter by reputable sources (default: True)
        """
        # Create shared cache manager
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        
        # Initialize components
        self.article_scraper = ArticleScraper(cache_manager=self.cache_manager)
        self.newsapi_verifier = NewsAPIVerifier(
            api_key=api_key,
            cache_dir=cache_dir,
            use_sources=use_reputable_sources,
            use_full_content=True
        )
        self.similarity_calculator = SimilarityCalculator(
            use_semantic=use_semantic_similarity
        )
    
    def verify_claim(self, claim: str, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Verify a claim against multiple article URLs
        
        Args:
            claim: The claim to verify
            urls: List of article URLs to compare against
            
        Returns:
            List[Dict]: List of verification results with similarity scores
        """
        # Scrape articles
        logger.info(f"Scraping {len(urls)} articles to verify claim")
        articles = self.article_scraper.scrape_multiple(urls)
        
        # Calculate similarity for each article
        results = []
        for article in articles:
            if not article['success'] or not article['text']:
                logger.warning(f"Skipping article with no content: {article['url']}")
                continue
                
            # Calculate similarity between claim and article
            similarity_score = self.similarity_calculator.calculate_similarity(
                claim, 
                article['text']
            )
            
            # Create result entry
            result = {
                'url': article['url'],
                'title': article['title'],
                'similarity_score': similarity_score,
                'article_summary': article['summary'],
                'article_keywords': article['keywords'],
                'published_date': article['published_date']
            }
            results.append(result)
        
        # Sort results by similarity score (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        logger.info(f"Completed verification against {len(results)} articles")
        return results
    
    def search_news_for_claim(self, claim: str, days: int = 7, max_results: int = 10) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Search for news articles related to a claim using NewsAPI
        
        Args:
            claim: The claim to search for
            days: Number of days to look back (default: 7)
            max_results: Maximum number of results to return (default: 10)
            
        Returns:
            Tuple[List[Dict], List[str]]: List of news articles and their URLs
        """
        logger.info(f"Searching for news articles related to claim: {claim}")
        
        try:
            # Use NewsAPI to search for articles
            articles = self.newsapi_verifier.search_news(
                query=claim,
                days=days,
                max_results=max_results
            )
            
            # Extract URLs for further processing
            urls = [article['url'] for article in articles if 'url' in article]
            
            logger.info(f"Found {len(urls)} news articles related to the claim")
            return articles, urls
        except Exception as e:
            logger.error(f"Error searching for news articles: {str(e)}")
            return [], []
    
    def verify_claim_with_newsapi(self, claim: str, days: int = 7, 
                                 max_results: int = 10, 
                                 threshold: float = 0.6) -> Dict[str, Any]:
        """
        Verify a claim using NewsAPI to find relevant articles and then calculate similarity
        
        Args:
            claim: The claim to verify
            days: Number of days to look back (default: 7)
            max_results: Maximum number of results to return (default: 10)
            threshold: Similarity threshold for considering a match (default: 0.6)
            
        Returns:
            Dict: Summary of verification results
        """
        try:
            # Search for news articles related to the claim
            articles, urls = self.search_news_for_claim(claim, days, max_results)
            
            if not urls:
                logger.warning("No articles found for claim verification")
                return {
                    "total_articles": 0,
                    "supporting_articles": 0,
                    "contradicting_articles": 0,
                    "average_similarity": 0.0,
                    "verification_confidence": "none",
                    "articles": []
                }
        except Exception as e:
            logger.error(f"Error in NewsAPI verification: {str(e)}")
            return {
                "error": str(e),
                "total_articles": 0,
                "supporting_articles": 0,
                "contradicting_articles": 0,
                "average_similarity": 0.0,
                "verification_confidence": "none",
                "articles": []
            }
        
        if not urls:
            logger.warning("No news articles found for the claim")
            return {
                'claim': claim,
                'total_articles': 0,
                'supporting_articles': 0,
                'contradicting_articles': 0,
                'average_similarity': 0,
                'verification_confidence': 'low',
                'detailed_results': []
            }
        
        # Verify the claim against the found articles
        return self.get_verification_summary(claim, urls, threshold)
    
    def get_verification_summary(self, claim: str, urls: List[str], 
                                threshold: float = 0.6) -> Dict[str, Any]:
        """
        Get a summary of verification results for a claim
        
        Args:
            claim: The claim to verify
            urls: List of article URLs to compare against
            threshold: Similarity threshold for considering a match (default: 0.6)
            
        Returns:
            Dict: Summary of verification results
        """
        # Get detailed results
        results = self.verify_claim(claim, urls)
        
        # Count articles above threshold
        supporting_articles = [r for r in results if r['similarity_score'] >= threshold]
        contradicting_articles = [r for r in results if r['similarity_score'] < threshold]
        
        # Calculate average similarity
        avg_similarity = sum(r['similarity_score'] for r in results) / len(results) if results else 0
        
        # Create summary
        summary = {
            'claim': claim,
            'total_articles': len(results),
            'supporting_articles': len(supporting_articles),
            'contradicting_articles': len(contradicting_articles),
            'average_similarity': avg_similarity,
            'top_match': results[0] if results else None,
            'verification_confidence': 'high' if avg_similarity > 0.8 else 
                                      'medium' if avg_similarity > 0.6 else 'low',
            'detailed_results': results
        }
        
        return summary