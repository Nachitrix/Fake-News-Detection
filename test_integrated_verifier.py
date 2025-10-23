"""
Test script for the Integrated Verifier module
"""
import json
from src.integrated_verifier import IntegratedVerifier

def test_integrated_verifier():
    """Test the integrated verifier with a sample claim"""
    print("Initializing Integrated Verifier...")
    verifier = IntegratedVerifier()
    
    # Test claim
    claim = "Climate change is causing more frequent and severe hurricanes"
    print(f"\nTesting claim: '{claim}'")
    
    # Test with predefined URLs
    test_urls = [
        "https://www.noaa.gov/news-release/active-2022-atlantic-hurricane-season-ends",
        "https://www.nature.com/articles/s41558-020-0893-y",
        "https://climate.nasa.gov/news/3184/a-force-of-nature-hurricanes-in-a-changing-climate/"
    ]
    
    print(f"\nTesting with {len(test_urls)} predefined URLs...")
    results = verifier.verify_claim(claim, test_urls)
    
    # Print results
    print(f"\nResults for predefined URLs:")
    for i, result in enumerate(results):
        print(f"Article {i+1}: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print("---")
    
    # Get summary
    summary = verifier.get_verification_summary(claim, test_urls)
    print("\nVerification Summary:")
    print(f"Total Articles: {summary['total_articles']}")
    print(f"Supporting Articles: {summary['supporting_articles']}")
    print(f"Contradicting Articles: {summary['contradicting_articles']}")
    print(f"Average Similarity: {summary['average_similarity']:.4f}")
    print(f"Verification Confidence: {summary['verification_confidence']}")
    
    # Test with NewsAPI
    print("\nTesting with NewsAPI (limited to 3 results)...")
    try:
        newsapi_summary = verifier.verify_claim_with_newsapi(claim, days=30, max_results=3)
        print("\nNewsAPI Verification Summary:")
        print(f"Total Articles: {newsapi_summary['total_articles']}")
        print(f"Supporting Articles: {newsapi_summary['supporting_articles']}")
        print(f"Contradicting Articles: {newsapi_summary['contradicting_articles']}")
        print(f"Average Similarity: {newsapi_summary['average_similarity']:.4f}")
        print(f"Verification Confidence: {newsapi_summary['verification_confidence']}")
    except Exception as e:
        print(f"NewsAPI test failed: {str(e)}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_integrated_verifier()