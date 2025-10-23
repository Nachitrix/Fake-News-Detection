#!/usr/bin/env python3

from src.similarity_calculator import SimilarityCalculator
from src.preprocess import TextPreprocessor

def test_similarity_calculator():
    """
    Test the similarity calculator module separately to verify it's working correctly.
    """
    print("=== Testing Similarity Calculator Module ===\n")
    
    # Initialize the similarity calculator and text preprocessor
    similarity_calculator = SimilarityCalculator()
    preprocessor = TextPreprocessor()
    
    # Test cases with claim and articles
    test_cases = [
        {
            "claim": "Climate change is causing global temperatures to rise",
            "articles": [
                "Scientists confirm that global warming is leading to increased temperatures worldwide.",
                "Climate change deniers claim that global warming is a hoax.",
                "Unrelated article about sports events."
            ]
        },
        {
            "claim": "The new vaccine has been thoroughly tested",
            "articles": [
                "Clinical trials show the vaccine is safe and effective after rigorous testing.",
                "Some people are concerned about the speed of vaccine development.",
                "Completely unrelated article about cooking recipes."
            ]
        }
    ]
    
    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        claim = test_case["claim"]
        articles = test_case["articles"]
        
        print(f"\nTest Case {i}:")
        print(f"Claim: \"{claim}\"")
        
        # Calculate similarity for each article
        print("\nSimilarity scores:")
        for j, article in enumerate(articles, 1):
            # Calculate similarity directly using the original texts
            # The similarity calculator will handle preprocessing internally
            similarity = similarity_calculator.calculate_similarity(claim, article)
            
            print(f"  Article {j}: \"{article}\"")
            print(f"  Similarity Score: {similarity:.4f}")
            print()
        
        # Convert string articles to dictionary format for get_similarity_score
        article_dicts = [{"title": article, "description": "", "content": ""} for article in articles]
        
        # Calculate overall similarity score
        overall_score, _ = similarity_calculator.get_similarity_score(claim, article_dicts)
        print(f"Overall Similarity Score: {overall_score:.4f}")
    
    print("\n=== Similarity Calculator Test Complete ===")

if __name__ == "__main__":
    test_similarity_calculator()