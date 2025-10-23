import unittest
import os
import sys
import logging
from unittest.mock import patch, MagicMock

# Configure logging for detailed test monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('end_to_end_tests')

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary modules
from src.fact_checker import FactChecker
from src.preprocess import TextPreprocessor
from src.newsapi_verifier import NewsAPIVerifier
from src.similarity_calculator import SimilarityCalculator
# GoogleFactCheckAPI import removed
from src.integrated_verifier import IntegratedVerifier
from src.cache_manager import CacheManager
from src.article_scraper import ArticleScraper

# Test data - Claims
TRUE_CLAIM = "NASA confirmed that 2020 was tied with 2016 for the warmest year on record."
FALSE_CLAIM = "The Earth is flat and NASA is hiding the truth from the public."
AMBIGUOUS_CLAIM = "Coffee has health benefits and risks."

# Test data - Articles
TRUE_ARTICLE = """
NASA and NOAA announced today that 2020 was tied with 2016 for the warmest year on record. 
The globally averaged temperature was 1.84 degrees Fahrenheit (1.02 degrees Celsius) warmer than the baseline 1951-1980 mean.
The analysis by NASA's Goddard Institute for Space Studies (GISS) compared global temperatures to the baseline period.
"The last seven years have been the warmest seven years on record," said GISS Director Gavin Schmidt.
Climate scientists track global temperature trends to understand how human activities are affecting Earth's climate system.
"""

FALSE_ARTICLE = """
Scientists have been lying to the public for decades about the shape of the Earth. 
The Earth is actually flat, and NASA has been covering up this truth with fake images from space.
All photos showing Earth as a globe are computer-generated images or heavily edited photos.
The Antarctic Treaty prevents ordinary citizens from exploring the ice wall at the edge of our flat Earth.
Gravity doesn't exist - objects fall because the flat Earth is constantly accelerating upward.
"""

AMBIGUOUS_ARTICLE = """
Coffee consumption has been the subject of numerous studies with mixed results.
Some research suggests coffee may reduce the risk of certain diseases like Parkinson's and type 2 diabetes.
However, other studies indicate that excessive coffee consumption may increase anxiety, disrupt sleep, and cause digestive issues.
The caffeine in coffee affects different people in different ways based on their genetics and tolerance.
Experts generally agree that moderate coffee consumption (3-4 cups daily) is safe for most adults, but individual responses vary.
"""

# Default weights (updated after removing factcheck_weight)
DEFAULT_WEIGHTS = {
    "ml_weight": 0.5,
    "newsapi_weight": 0.3,
    "similarity_weight": 0.2
}

class TestFactChecker(unittest.TestCase):
    """Test the FactChecker component and end-to-end verification"""
    
    def setUp(self):
        """Set up test environment"""
        logger.info("=== Setting up test environment ===")
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "classifier.pkl")
        self.vectorizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "tfidf_vectorizer.pkl")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Vectorizer path: {self.vectorizer_path}")
        
        # Initialize cache manager
        logger.info("Initializing CacheManager...")
        self.cache_manager = CacheManager(self.cache_dir)
        
        # Initialize preprocessor
        logger.info("Initializing TextPreprocessor...")
        self.preprocessor = TextPreprocessor()
        
        # Initialize components with mock API keys
        logger.info("Initializing NewsAPIVerifier with mock API key...")
        self.newsapi_verifier = NewsAPIVerifier(api_key="mock_newsapi_key", cache_dir=self.cache_dir)
        
        logger.info("Initializing SimilarityCalculator...")
        self.similarity_calculator = SimilarityCalculator()
        
        # Create patches for external API calls
        logger.info("Setting up mock patches for external dependencies...")
        self.newsapi_patch = patch('newsapi.NewsApiClient')  # Fix: patch the correct module path
        # factcheck_patch removed
        self.article_scraper_patch = patch('src.article_scraper.ArticleScraper.scrape_article')
        self.joblib_patch = patch('src.fact_checker.joblib.load')
        
        # Start the patches
        self.mock_newsapi = self.newsapi_patch.start()
        # mock_factcheck removed
        self.mock_article_scraper = self.article_scraper_patch.start()
        self.mock_joblib = self.joblib_patch.start()
        
        # Configure the mocks
        self.mock_newsapi_instance = MagicMock()
        self.mock_newsapi.return_value = self.mock_newsapi_instance
        
        # GoogleFactCheckAPI mock configuration removed
        
        self.mock_article_scraper.return_value = "Mocked article content for testing"
        
        # Configure ML model mocks
        self.mock_classifier = MagicMock()
        self.mock_vectorizer = MagicMock()
        self.mock_joblib.side_effect = [self.mock_classifier, self.mock_vectorizer]
        
        # Initialize fact checker with mock API keys
        self.fact_checker = FactChecker(
            newsapi_key="mock_newsapi_key",
            model_path=self.model_path,
            cache_dir=self.cache_dir
        )
    
    def tearDown(self):
        """Clean up after tests"""
        logger.info("=== Cleaning up after tests ===")
        # Stop the patches
        self.newsapi_patch.stop()
        # factcheck_patch.stop removed
        self.article_scraper_patch.stop()
        self.joblib_patch.stop()
        logger.info("All patches stopped successfully")
    
    def _configure_newsapi_mock(self, supporting=True):
        """Configure the NewsAPI mock to return articles that support or contradict the claim"""
        articles = []
        if supporting:
            articles = [
                {
                    'title': 'Supporting Article',
                    'description': 'This article supports the claim',
                    'url': 'https://example.com/article1',
                    'content': 'Full content supporting the claim'
                },
                {
                    'title': 'Another Supporting Article',
                    'description': 'This article also supports the claim',
                    'url': 'https://example.com/article2',
                    'content': 'More content supporting the claim'
                }
            ]
        else:
            articles = [
                {
                    'title': 'Contradicting Article',
                    'description': 'This article contradicts the claim',
                    'url': 'https://example.com/article1',
                    'content': 'Full content contradicting the claim'
                },
                {
                    'title': 'Another Contradicting Article',
                    'description': 'This article also contradicts the claim',
                    'url': 'https://example.com/article2',
                    'content': 'More content contradicting the claim'
                }
            ]
        
        self.mock_newsapi_instance.get_everything.return_value = {
            'status': 'ok',
            'totalResults': len(articles),
            'articles': articles
        }
    
    # _configure_factcheck_mock method removed
    
    @patch('joblib.load')
    def test_preprocessing(self, mock_load):
        """Test text preprocessing functionality"""
        logger.info("=== Starting test_preprocessing ===")
        # Mock the classifier and vectorizer
        logger.info("Setting up mocks for preprocessing test...")
        mock_classifier = MagicMock()
        mock_vectorizer = MagicMock()
        mock_load.side_effect = [mock_classifier, mock_vectorizer]
        
        # Test with a sample text
        sample_text = "This is a sample text with some UPPERCASE words and punctuation!!!"
        processed_text = self.preprocessor.preprocess(sample_text)
        
        # Check if preprocessing removes punctuation and converts to lowercase
        self.assertNotIn("!", str(processed_text))
        # The preprocessor returns a tuple with tokens and original text
        tokens, original = processed_text
        self.assertIn("sample", tokens)
        self.assertIn("text", tokens)
    
    @patch('joblib.load')
    def test_ml_classification(self, mock_load):
        """Test ML classification component"""
        # Mock the classifier and vectorizer
        mock_classifier = MagicMock()
        mock_vectorizer = MagicMock()
        
        # Set up the mock return values
        mock_classifier.predict_proba.return_value = [[0.3, 0.7]]  # 70% probability of being true
        mock_vectorizer.transform.return_value = "transformed_text"
        
        # Configure the mocks
        mock_load.side_effect = [mock_classifier, mock_vectorizer]
        
        # Create a fact checker with the mocked components
        with patch('src.fact_checker.joblib.load', side_effect=[mock_classifier, mock_vectorizer]):
            # Test ML classification
            ml_score = self.fact_checker.get_ml_score(TRUE_CLAIM)
            
            # Verify the score is as expected (0.7 for true class)
            self.assertEqual(ml_score, 0.7)
    
    @patch('joblib.load')
    def test_weight_configurations(self, mock_load):
        """Test different weight configurations"""
        # Mock the classifier and vectorizer
        mock_classifier = MagicMock()
        mock_vectorizer = MagicMock()
        
        # Set up the mock return values for ML model
        mock_classifier.predict_proba.return_value = [[0.2, 0.8]]  # 80% probability of being true
        mock_vectorizer.transform.return_value = "transformed_text"
        
        # Configure the ML mocks
        mock_load.side_effect = [mock_classifier, mock_vectorizer]
        
        # Configure the NewsAPI mock to return supporting articles
        self._configure_newsapi_mock(supporting=True)
        
        # Configure the FactCheck API mock to return true claims
        self._configure_factcheck_mock(rating="True")
        
        # Create a fact checker with the mocked components
        with patch('src.fact_checker.joblib.load', side_effect=[mock_classifier, mock_vectorizer]):
            # Run the fact checker
            result = self.fact_checker.check_fact(TRUE_CLAIM)
            
            # Verify the verdict is as expected
            self.assertEqual(result["verdict"], "LIKELY TRUE")
            
            # Verify the final score is above the threshold for "LIKELY TRUE"
            self.assertGreaterEqual(result["final_score"], 0.6)
    
    @patch('joblib.load')
    def test_end_to_end_false_claim(self, mock_load):
        """Test end-to-end verification with a known false claim"""
        # Mock the classifier and vectorizer
        mock_classifier = MagicMock()
        mock_vectorizer = MagicMock()
        
        # Set up the mock return values for ML model
        mock_classifier.predict_proba.return_value = [[0.7, 0.3]]  # 30% probability of being true
        mock_vectorizer.transform.return_value = "transformed_text"
        
        # Configure the ML mocks
        mock_load.side_effect = [mock_classifier, mock_vectorizer]
        
        # Configure the NewsAPI mock to return contradicting articles
        self._configure_newsapi_mock(supporting=False)
        
        # Configure the FactCheck API mock to return false claims
        self._configure_factcheck_mock(rating="False")
        
        # Create a fact checker with the mocked components
        with patch('src.fact_checker.joblib.load', side_effect=[mock_classifier, mock_vectorizer]):
            # Run the fact checker
            result = self.fact_checker.check_fact(FALSE_CLAIM)
            
            # Verify the verdict is as expected
            self.assertEqual(result["verdict"], "LIKELY FALSE")
            
            # Verify the final score is below the threshold for "LIKELY FALSE"
            self.assertLessEqual(result["final_score"], 0.4)
            
    def test_end_to_end_ambiguous_claim(self):
        """Test end-to-end verification with an ambiguous claim"""
        # Set up the mock return values for ML model
        self.mock_classifier.predict_proba.return_value = [[0.5, 0.5]]  # 50% probability of being true
        self.mock_vectorizer.transform.return_value = "transformed_text"
        
        # Configure the NewsAPI mock to return mixed articles
        articles = [
            {
                'title': 'Supporting Article',
                'description': 'This article supports the claim',
                'url': 'https://example.com/article1',
                'content': 'Full content supporting the claim'
            },
            {
                'title': 'Contradicting Article',
                'description': 'This article contradicts the claim',
                'url': 'https://example.com/article2',
                'content': 'Full content contradicting the claim'
            }
        ]
        
        self.mock_newsapi_instance.get_everything.return_value = {
            'status': 'ok',
            'totalResults': len(articles),
            'articles': articles
        }
        
        # Configure the FactCheck API mock to return mixed claims
        self.mock_factcheck_claims.search.return_value.execute.return_value = {
            'claims': [
                {
                    'text': 'Similar claim 1',
                    'claimReview': [
                        {
                            'textualRating': 'True',
                            'url': 'https://example.com/factcheck1'
                        }
                    ]
                },
                {
                    'text': 'Similar claim 2',
                    'claimReview': [
                        {
                            'textualRating': 'False',
                            'url': 'https://example.com/factcheck2'
                        }
                    ]
                }
            ]
        }
        
        # Run the fact checker
        result = self.fact_checker.check_fact(AMBIGUOUS_CLAIM)
        
        # Verify the verdict is as expected
        self.assertEqual(result["verdict"], "UNCERTAIN")
        
        # Verify the final score is in the uncertain range
        self.assertGreater(result["final_score"], 0.3)
        self.assertLess(result["final_score"], 0.7)
        
    def test_end_to_end_true_article(self):
        """Test end-to-end verification with a known true article"""
        # Set up the mock return values for ML model
        self.mock_classifier.predict_proba.return_value = [[0.2, 0.8]]  # 80% probability of being true
        self.mock_vectorizer.transform.return_value = "transformed_text"
        
        # Configure the NewsAPI mock to return supporting articles
        self._configure_newsapi_mock(supporting=True)
        
        # Configure the FactCheck API mock to return true claims
        self._configure_factcheck_mock(rating="True")
        
        # Run the fact checker
        result = self.fact_checker.check_fact(TRUE_ARTICLE)
        
        # Verify the verdict is as expected
        self.assertEqual(result["verdict"], "LIKELY TRUE")
        
        # Verify the final score is above the threshold for "LIKELY TRUE"
        self.assertGreaterEqual(result["final_score"], 0.6)
        
    def test_end_to_end_false_article(self):
        """Test end-to-end verification with a known false article"""
        # Set up the mock return values for ML model
        self.mock_classifier.predict_proba.return_value = [[0.7, 0.3]]  # 30% probability of being true
        self.mock_vectorizer.transform.return_value = "transformed_text"
        
        # Configure the NewsAPI mock to return contradicting articles
        self._configure_newsapi_mock(supporting=False)
        
        # Configure the FactCheck API mock to return false claims
        self._configure_factcheck_mock(rating="False")
        
        # Run the fact checker
        result = self.fact_checker.check_fact(FALSE_ARTICLE)
        
        # Verify the verdict is as expected
        self.assertEqual(result["verdict"], "LIKELY FALSE")
        
        # Verify the final score is below the threshold for "LIKELY FALSE"
        self.assertLessEqual(result["final_score"], 0.4)
        
    def test_end_to_end_ambiguous_article(self):
        """Test end-to-end verification with an ambiguous article"""
        # Set up the mock return values for ML model
        self.mock_classifier.predict_proba.return_value = [[0.5, 0.5]]  # 50% probability of being true
        self.mock_vectorizer.transform.return_value = "transformed_text"
        
        # Configure the NewsAPI mock to return mixed articles
        articles = [
            {
                'title': 'Supporting Article',
                'description': 'This article supports the claim',
                'url': 'https://example.com/article1',
                'content': 'Full content supporting the claim'
            },
            {
                'title': 'Contradicting Article',
                'description': 'This article contradicts the claim',
                'url': 'https://example.com/article2',
                'content': 'Full content contradicting the claim'
            }
        ]
        
        self.mock_newsapi_instance.get_everything.return_value = {
            'status': 'ok',
            'totalResults': len(articles),
            'articles': articles
        }
        
        # Configure the FactCheck API mock to return mixed claims
        self.mock_factcheck_claims.search.return_value.execute.return_value = {
            'claims': [
                {
                    'text': 'Similar claim 1',
                    'claimReview': [
                        {
                            'textualRating': 'True',
                            'url': 'https://example.com/factcheck1'
                        }
                    ]
                },
                {
                    'text': 'Similar claim 2',
                    'claimReview': [
                        {
                            'textualRating': 'False',
                            'url': 'https://example.com/factcheck2'
                        }
                    ]
                }
            ]
        }
        
        # Run the fact checker
        result = self.fact_checker.check_fact(AMBIGUOUS_ARTICLE)
        
        # Verify the verdict is as expected
        self.assertEqual(result["verdict"], "UNCERTAIN")
        
        # Verify the final score is in the uncertain range
        self.assertGreater(result["final_score"], 0.3)
        self.assertLess(result["final_score"], 0.7)
    
    @patch('src.train_model.joblib.load')
    def test_weight_configurations(self, mock_load):
        """Test different weight configurations"""
        # Mock the classifier and vectorizer
        mock_classifier = MagicMock()
        mock_vectorizer = MagicMock()
        
        # Set up the mock return values for ML model
        mock_classifier.predict_proba.return_value = [[0.2, 0.8]]  # 80% probability of being true
        mock_vectorizer.transform.return_value = "transformed_text"
        
        # Configure the ML mocks
        mock_load.side_effect = [mock_classifier, mock_vectorizer]
        
        # Configure the NewsAPI mock to return supporting articles
        self._configure_newsapi_mock(supporting=True)
        
        # Configure the FactCheck API mock to return true claims
        self._configure_factcheck_mock(rating="True")
        
        # Create a fact checker with the mocked components
        with patch('src.fact_checker.joblib.load', side_effect=[mock_classifier, mock_vectorizer]):
            # Test with default weights
            self.fact_checker.set_weights(
                ml_weight=DEFAULT_WEIGHTS["ml_weight"],
                newsapi_weight=DEFAULT_WEIGHTS["newsapi_weight"],
                similarity_weight=DEFAULT_WEIGHTS["similarity_weight"],
                factcheck_weight=DEFAULT_WEIGHTS["factcheck_weight"]
            )
            result_default = self.fact_checker.check_fact(TRUE_CLAIM)
            
            # Test with ML-focused weights
            self.fact_checker.set_weights(
                ml_weight=0.7,
                newsapi_weight=0.1,
                similarity_weight=0.1,
                factcheck_weight=0.1
            )
            result_ml_focused = self.fact_checker.check_fact(TRUE_CLAIM)
            
            # Verify that different weight configurations produce different final scores
            self.assertNotEqual(result_default["final_score"], result_ml_focused["final_score"])
    
    def test_caching_mechanism(self):
        """Test that caching mechanism works correctly"""
        # Create a unique test key
        test_key = f"test_cache_{os.urandom(8).hex()}"
        test_data = {"test": "data", "timestamp": "2023-01-01"}
        
        # Save data to cache
        self.cache_manager.save_to_cache(test_key, test_data, "api")
        
        # Verify data is in cache
        cached_data = self.cache_manager.load_from_cache(test_key, "api")
        self.assertIsNotNone(cached_data)
        self.assertEqual(cached_data["test"], "data")
        
        # Clean up
        cache_file = os.path.join(self.cache_dir, "api", f"{self.cache_manager._hash_key(test_key)}.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)


if __name__ == "__main__":
    unittest.main()