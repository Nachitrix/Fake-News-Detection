#!/usr/bin/env python3
"""
Test script for verifying a specific claim using the NewsAPI integration
"""
from src.integrated_verifier import IntegratedVerifier
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Test the NewsAPI integration with a specific claim"""
    # Initialize the verifier
    print("Initializing IntegratedVerifier...")
    verifier = IntegratedVerifier()
    
    # Define the claim to test
    claim = "Donald Trump has won the Nobel Peace Prize"
    print(f"\nTesting claim: {claim}")
    
    # Verify the claim using NewsAPI
    print("\nVerifying claim with NewsAPI...")
    result = verifier.verify_claim_with_newsapi(claim, max_results=3)
    
    # Print the results in a readable format
    print("\nVerification Result:")
    print("="*50)
    print(f"Claim: {claim}\n")
    print(f"Total Articles: {result['total_articles']}")
    print(f"Supporting Articles: {result['supporting_articles']}")
    print(f"Contradicting Articles: {result['contradicting_articles']}")
    print(f"Average Similarity: {result.get('average_similarity', 0):.4f}")
    print(f"Verification Confidence: {result['verification_confidence']}\n")
    
    # Print article details
    print("Article Details:")
    for i, article in enumerate(result.get('articles', [])):
        print(f"\nArticle {i+1}:")
        print(f"Title: {article.get('title', 'N/A')}")
        print(f"URL: {article.get('url', 'N/A')}")
        print(f"Similarity: {article.get('similarity', 0):.4f}")
        print(f"Supports Claim: {article.get('supports_claim', False)}")
    
    # Print the full result as JSON for debugging
    print("\nFull Result (JSON):")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()