#!/usr/bin/env python3
"""
Script to train the model and test the system with FakeNewsNet dataset and NewsAPI
"""
import os
import sys

# Add current directory to path to resolve import issues
sys.path.append('.')

# Import project modules
from src.dataset_handler import FakeNewsNetHandler
from src.train_model import FakeNewsClassifier
from src.newsapi_verifier import NewsAPIVerifier
from src.similarity_calculator import SimilarityCalculator
from src.fact_checker import FactChecker

def main():
    print("=== Fake News Detection System - Phase 1 Implementation ===")
    
    # Step 1: Set up directories
    data_dir = "data"
    models_dir = "models"
    cache_dir = os.path.join(data_dir, "cache")
    
    for directory in [data_dir, models_dir, cache_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 2: Process dataset
    print("\n[1/5] Processing FakeNewsNet dataset...")
    handler = FakeNewsNetHandler(data_dir)
    
    # Download dataset if not already downloaded
    handler.download_minimal_dataset()
    
    # Load and prepare dataset
    dataset = handler.load_dataset()
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Sample data:\n{dataset.head()}")
    
    # Step 3: Train ML model
    print("\n[2/5] Training ML classifier...")
    classifier = FakeNewsClassifier(models_dir)
    model_path = os.path.join(models_dir, "fake_news_classifier.pkl")
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        classifier.load_model(model_path)
    else:
        print(f"Training new model and saving to {model_path}")
        classifier.train_and_save(dataset)
    
    # Step 4: Test NewsAPI integration
    print("\n[3/5] Testing NewsAPI integration...")
    newsapi = NewsAPIVerifier(api_key="ea70866cd4454c3db02541b742b4d3e0", cache_dir=cache_dir)
    
    test_query = "climate change"
    print(f"Searching for news about '{test_query}'...")
    articles = newsapi.search_news(test_query, max_results=5)
    
    if articles:
        print(f"Found {len(articles)} articles")
        for i, article in enumerate(articles[:3], 1):
            print(f"  {i}. {article.get('title', 'No title')} - {article.get('source', {}).get('name', 'Unknown source')}")
    else:
        print("No articles found or API error")
    
    # Step 5: Test similarity calculation
    print("\n[4/5] Testing similarity calculation...")
    similarity = SimilarityCalculator()
    
    test_text1 = "Climate change is causing global temperatures to rise"
    test_text2 = "Global warming is increasing temperatures worldwide"
    
    sim_score = similarity.calculate_similarity(test_text1, test_text2)
    print(f"Similarity between two related texts: {sim_score:.4f}")
    
    # Step 6: Test fact checker
    print("\n[5/5] Testing fact checker...")
    fact_checker = FactChecker(
        model_path=model_path,
        newsapi_key="ea70866cd4454c3db02541b742b4d3e0",
        cache_dir=cache_dir
    )
    
    test_claim = "Climate change is a hoax created by scientists"
    result = fact_checker.check_fact(test_claim)
    
    print(f"\nClaim: '{test_claim}'")
    print(f"Verdict: {result['verdict']}")
    print(f"ML Score: {result['ml_score']:.4f}")
    print(f"NewsAPI Score: {result['newsapi_score']:.4f}")
    print(f"Similarity Score: {result['similarity_score']:.4f}")
    print(f"Final Score: {result['final_score']:.4f}")
    
    print("\n=== Phase 1 Implementation Test Complete ===")
    print("The system is now ready for use with the FakeNewsNet dataset and NewsAPI integration.")

if __name__ == "__main__":
    main()