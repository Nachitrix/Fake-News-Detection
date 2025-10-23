"""
Unit tests for the enhanced similarity calculator
"""
import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the sentence_transformers module
sys.modules['sentence_transformers'] = MagicMock()
from src.similarity_calculator import SimilarityCalculator
from src.preprocess import TextPreprocessor

class TestEnhancedSimilarityCalculator(unittest.TestCase):
    """Test cases for the enhanced similarity calculator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = TextPreprocessor()
        
        # Create a calculator with semantic similarity disabled for consistent testing
        with patch('src.similarity_calculator.SentenceTransformer'):
            self.calculator = SimilarityCalculator(
                preprocessor=self.preprocessor,
                use_semantic=False
            )
        
        # Test texts
        self.claim = "Climate change is causing extreme weather events."
        self.similar_text = "Global warming has led to an increase in severe weather patterns."
        self.unrelated_text = "Football match ended with a score of 3-2 after extra time."
        
        # Test articles
        self.articles = [
            {
                'title': 'Climate Change Effects',
                'description': 'Global warming has led to an increase in severe weather patterns worldwide.',
                'content': 'Scientists have documented the relationship between climate change and extreme weather events.',
                'url': 'https://www.reuters.com/climate-change-report'
            },
            {
                'title': 'Sports News',
                'description': 'Football match results',
                'content': 'The match ended with a score of 3-2 after extra time.',
                'url': 'https://www.unknownsource.com/sports-news'
            }
        ]
    
    def test_basic_similarity(self):
        """Test the basic similarity calculation"""
        similarity = self.calculator.calculate_similarity(self.claim, self.similar_text)
        self.assertGreater(similarity, 0.0, "Similar texts should have similarity > 0")
        
        similarity = self.calculator.calculate_similarity(self.claim, self.unrelated_text)
        self.assertLess(similarity, 0.5, "Unrelated texts should have similarity < 0.5")
    
    def test_section_extraction(self):
        """Test the section extraction functionality"""
        text = """Climate Change Impact
        
        Recent studies have shown that climate change is causing more frequent and severe weather events globally.
        
        In conclusion, immediate action is needed to address this crisis."""
        
        sections = self.calculator.extract_sections(text)
        
        self.assertIn('title', sections)
        self.assertIn('body', sections)
        self.assertIn('conclusion', sections)
        
        self.assertIn("Climate Change Impact", sections['title'])
        self.assertIn("Recent studies", sections['body'])
        self.assertIn("immediate action", sections['conclusion'])
    
    def test_section_similarity(self):
        """Test the section-based similarity calculation"""
        # Mock the semantic similarity calculation
        with patch.object(self.calculator, 'calculate_semantic_similarity', return_value=0.7):
            text1 = """Climate Change Impact
            
            Recent studies have shown that climate change is causing more frequent and severe weather events globally.
            
            In conclusion, immediate action is needed to address this crisis."""
            
            text2 = """Global Warming Effects
            
            Scientists have documented increasing severe weather patterns worldwide due to global warming.
            
            Therefore, policy changes are urgently required."""
            
            similarity = self.calculator.calculate_section_similarity(text1, text2)
            self.assertGreater(similarity, 0.0, "Section similarity should be greater than 0")
            self.assertLessEqual(similarity, 1.0, "Section similarity should be less than or equal to 1")
    
    def test_claim_decomposition(self):
        """Test the claim decomposition functionality"""
        complex_claim = "Climate change is real, it's causing extreme weather, and it requires immediate action."
        statements = self.calculator.decompose_claim(complex_claim)
        
        self.assertGreaterEqual(len(statements), 2, "Complex claim should be decomposed into at least 2 statements")
        self.assertIn("Climate change is real", statements[0])
    
    def test_source_credibility(self):
        """Test the source credibility functionality"""
        credibility = self.calculator.get_source_credibility("https://www.reuters.com/article/123")
        self.assertGreater(credibility, 0.8, "Reuters should have high credibility")
        
        credibility = self.calculator.get_source_credibility("https://www.unknownsource.com/article")
        self.assertEqual(credibility, self.calculator.default_credibility, 
                         "Unknown source should have default credibility")
    
    def test_batch_similarity(self):
        """Test the batch similarity calculation"""
        # Mock the calculate_similarity method
        with patch.object(self.calculator, 'calculate_similarity', side_effect=[0.8, 0.2]):
            texts = [self.similar_text, self.unrelated_text]
            results = self.calculator.calculate_batch_similarity(self.claim, texts)
            
            self.assertEqual(len(results), 2, "Should return results for both texts")
    
    def test_overall_similarity_score(self):
        """Test the overall similarity score calculation"""
        # Mock the calculate_similarity method
        with patch.object(self.calculator, 'calculate_similarity', return_value=0.7):
            score, scored_articles = self.calculator.get_similarity_score(self.claim, self.articles)
            
            self.assertGreater(score, 0.0, "Overall score should be greater than 0")
            self.assertLessEqual(score, 1.0, "Overall score should be less than or equal to 1")
            self.assertEqual(len(scored_articles), 2, "Should return both articles with scores")
            
            # Check that source credibility is applied
            self.assertIn('source_credibility', scored_articles[0])
            self.assertIn('raw_similarity_score', scored_articles[0])
            self.assertIn('similarity_score', scored_articles[0])

if __name__ == '__main__':
    unittest.main()