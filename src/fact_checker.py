"""
Main Fact Checker Module - Combines ML, NewsAPI, Similarity scoring, and Google Fact Check API
"""
import os
import json
import joblib
import numpy as np
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Import local modules
from src.preprocess import TextPreprocessor
from src.newsapi_verifier import NewsAPIVerifier
from src.similarity_calculator import SimilarityCalculator
# GoogleFactCheckAPI import removed

class FactChecker:
    """Main fact checker that combines multiple verification methods"""
    
    def __init__(self, model_path=None, newsapi_key=None, cache_dir=None):
        """
        Initialize the fact checker
        
        Args:
            model_path (str): Path to trained ML model
            newsapi_key (str): NewsAPI key
            cache_dir (str): Directory for caching
        """
        # Set up cache directory
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.newsapi = NewsAPIVerifier(api_key=newsapi_key, cache_dir=self.cache_dir)
        self.similarity_calculator = SimilarityCalculator(preprocessor=self.preprocessor)
        
        # Load ML model
        self.model_path = model_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'classifier.pkl')
        self.vectorizer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'tfidf_vectorizer.pkl')
        
        try:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.model_loaded = True
        except (FileNotFoundError, IOError):
            print(f"Warning: ML model not found at {self.model_path}")
            self.model_loaded = False
        
        # Scoring weights (redistributed as requested)
        self.weights = {
            'ml_score': 0.4,      # ML classification weight
            'newsapi_score': 0.2,  # NewsAPI verification weight
            'similarity_score': 0.4 # Content similarity weight
        }
    
    def set_weights(self, ml_score=0.4, newsapi_score=0.2, similarity_score=0.4):
        """
        Set custom weights for scoring components
        
        Args:
            ml_score (float): Weight for ML classifier (0-1)
            newsapi_score (float): Weight for NewsAPI verification (0-1)
            similarity_score (float): Weight for similarity calculation (0-1)
        """
        # Normalize weights to sum to 1
        total = ml_score + newsapi_score + similarity_score
        self.weights = {
            'ml_score': ml_score / total,
            'newsapi_score': newsapi_score / total,
            'similarity_score': similarity_score / total
        }
    
    def get_ml_score(self, text):
        """
        Get score from ML classifier
        
        Args:
            text (str): News text to classify
            
        Returns:
            float: Score between 0-1 (higher means more likely real)
        """
        if not self.model_loaded:
            return 0.5  # Neutral score if model not loaded
        
        # Preprocess text - handle tuple return value
        processed_result = self.preprocessor.preprocess(text)
        
        # The preprocess method returns a tuple of (tokens, preprocessed_text)
        # Extract the preprocessed text (second element)
        if isinstance(processed_result, tuple) and len(processed_result) >= 2:
            processed_text = processed_result[1]  # Get the preprocessed text
        else:
            processed_text = processed_result  # Use as is if not a tuple
        
        # Vectorize
        text_vector = self.vectorizer.transform([processed_text])
        
        # Get prediction probability
        proba = self.model.predict_proba(text_vector)[0]
        
        # Return probability of real news (assuming class 1 is real)
        # If class 1 is fake, use 1 - proba[1]
        return proba[1]
    
    def check_fact(self, text, title=None):
        """
        Check if a news article is likely true or false
        
        Args:
            text (str): News text to verify
            title (str, optional): News title
            
        Returns:
            dict: Results with scores and supporting evidence
        """
        # Prepare query text (title + text)
        query_text = f"{title} {text}" if title else text
        search_query = title or text[:100]  # Use title or first 100 chars for search
        
        # Get ML classification score
        ml_score = self.get_ml_score(query_text)
        
        # Get NewsAPI verification
        newsapi_score, articles = self.newsapi.get_verification_score(search_query)
        
        # Get similarity score for found articles
        similarity_score, scored_articles = self.similarity_calculator.get_similarity_score(query_text, articles)
        
        # Calculate weighted final score (without Google Fact Check API)
        final_score = (
            self.weights['ml_score'] * ml_score +
            self.weights['newsapi_score'] * newsapi_score +
            self.weights['similarity_score'] * similarity_score
        )
        
        # Determine verdict
        if final_score >= 0.7:
            verdict = "LIKELY TRUE"
            verdict_color = Fore.GREEN
        elif final_score <= 0.3:
            verdict = "LIKELY FALSE"
            verdict_color = Fore.RED
        else:
            verdict = "UNCERTAIN"
            verdict_color = Fore.YELLOW
        
        # Prepare result (without Google Fact Check API)
        result = {
            'text': text,
            'title': title,
            'verdict': verdict,
            'verdict_color': verdict_color,
            'final_score': final_score,
            'component_scores': {
                'ml_score': ml_score,
                'newsapi_score': newsapi_score,
                'similarity_score': similarity_score
            },
            'supporting_articles': scored_articles[:5],  # Top 5 articles
            'weights': self.weights
        }
        
        return result
    
    def print_result(self, result):
        """
        Print formatted result to terminal
        
        Args:
            result (dict): Result from check_fact
        """
        # Print header
        print("\n" + "="*80)
        print(f"FACT CHECK RESULT: {result['verdict_color']}{result['verdict']}{Style.RESET_ALL}")
        print("="*80)
        
        # Print title if available
        if result['title']:
            print(f"\nTitle: {Fore.CYAN}{result['title']}{Style.RESET_ALL}")
        
        # Print scores
        print(f"\nOverall Credibility Score: {self._format_score(result['final_score'])}")
        print("\nComponent Scores:")
        print(f"  - ML Classification: {self._format_score(result['component_scores']['ml_score'])} (Weight: {result['weights']['ml_score']:.2f})")
        print(f"  - NewsAPI Verification: {self._format_score(result['component_scores']['newsapi_score'])} (Weight: {result['weights']['newsapi_score']:.2f})")
        print(f"  - Content Similarity: {self._format_score(result['component_scores']['similarity_score'])} (Weight: {result['weights']['similarity_score']:.2f})")
        
        # Print supporting articles
        if result['supporting_articles']:
            print("\nSupporting Articles:")
            for i, article in enumerate(result['supporting_articles'], 1):
                similarity = article.get('similarity_score', 0)
                color = self._get_color_for_score(similarity)
                print(f"  {i}. {color}{article['title']}{Style.RESET_ALL}")
                print(f"     Source: {article['source']}")
                print(f"     Similarity: {self._format_score(similarity)}")
                print(f"     URL: {article['url']}")
                print()
        else:
            print("\nNo supporting articles found.")
        
        # Fact Check Claims section removed
        
        print("="*80)
    
    def _format_score(self, score):
        """Format score with color based on value"""
        color = self._get_color_for_score(score)
        return f"{color}{score:.2f}{Style.RESET_ALL}"
    
    def _get_color_for_score(self, score):
        """Get appropriate color for a score"""
        if score >= 0.7:
            return Fore.GREEN
        elif score <= 0.3:
            return Fore.RED
        else:
            return Fore.YELLOW


if __name__ == "__main__":
    # Test the fact checker
    model_path = "../models/fake_news_model.pkl"
    newsapi_key = os.environ.get('NEWSAPI_KEY')
    cache_dir = "../data/cache"
    
    checker = FactChecker(model_path=model_path, newsapi_key=newsapi_key, cache_dir=cache_dir)
    
    # Test with sample news
    test_news = [
        {
            'title': 'Global temperatures reach record high in 2023',
            'text': 'Scientists report that global temperatures have reached the highest levels ever recorded in 2023, providing further evidence of accelerating climate change. Multiple independent research groups confirmed the findings.'
        },
        {
            'title': 'New study finds chocolate prevents cancer',
            'text': 'A groundbreaking study has found that eating chocolate daily can prevent all types of cancer with 100% effectiveness. The miraculous properties of chocolate were discovered accidentally by researchers.'
        }
    ]
    
    for news in test_news:
        result = checker.check_fact(news['text'], news['title'])
        checker.print_result(result)