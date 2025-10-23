#!/usr/bin/env python3
"""
End-to-End Fake News Detection System
This script implements an end-to-end testing solution for the Fake News Detection system.
It accepts user input (either a claim or an article) and verifies its authenticity.
"""
import os
import sys
import joblib
import numpy as np
from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init(autoreset=True)

# Add current directory to path
sys.path.append('.')

# Import project modules
from src.preprocess import TextPreprocessor
from src.newsapi_verifier import NewsAPIVerifier
from src.similarity_calculator import SimilarityCalculator
from src.integrated_verifier import IntegratedVerifier

# Constants
MODELS_DIR = 'models'
CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'classifier.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
MAX_ARTICLES = 5  # Number of articles to retrieve from NewsAPI

def get_user_input():
    """
    Get input from the user (either a claim or an article)
    
    Returns:
        str: User input text
    """
    print(f"\n{Fore.CYAN}===== Fake News Detection System ====={Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Enter a claim or paste an article to verify:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}(Type or paste your text below and press Enter, then Ctrl+D to finish){Style.RESET_ALL}\n")
    
    # Collect multi-line input
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    
    text = '\n'.join(lines)
    return text.strip()

def is_article(text, min_length=200):
    """
    Determine if the input is an article or a claim based on length
    
    Args:
        text (str): Input text
        min_length (int): Minimum length to be considered an article
        
    Returns:
        bool: True if the text is likely an article, False if it's likely a claim
    """
    # Simple heuristic: if text is longer than min_length characters, it's likely an article
    return len(text) >= min_length

def load_ml_model():
    """
    Load the trained ML model and vectorizer
    
    Returns:
        tuple: (model, vectorizer) or (None, None) if loading fails
    """
    try:
        model = joblib.load(CLASSIFIER_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except (FileNotFoundError, IOError) as e:
        print(f"{Fore.RED}Error loading ML model: {str(e)}{Style.RESET_ALL}")
        return None, None

def get_ml_prediction(text, model, vectorizer):
    """
    Get prediction from ML model
    
    Args:
        text (str): Text to classify
        model: Trained classifier
        vectorizer: TF-IDF vectorizer
        
    Returns:
        tuple: (prediction, probability)
    """
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess text
    _, preprocessed_text = preprocessor.preprocess(text)
    
    # Vectorize text
    features = vectorizer.transform([preprocessed_text])
    
    # Get prediction and probability
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get probability of the predicted class
    if prediction == 'real':
        probability = probabilities[1]  # Probability of 'real' class
    else:
        probability = probabilities[0]  # Probability of 'fake' class
    
    return prediction, probability

def verify_claim(claim, model, vectorizer):
    """
    Verify a claim using ML model and NewsAPI
    
    Args:
        claim (str): Claim to verify
        model: Trained classifier
        vectorizer: TF-IDF vectorizer
        
    Returns:
        dict: Verification results
    """
    results = {}
    
    # Get ML prediction
    if model is not None and vectorizer is not None:
        prediction, probability = get_ml_prediction(claim, model, vectorizer)
        results['ml_prediction'] = {
            'prediction': prediction,
            'confidence': probability
        }
    else:
        results['ml_prediction'] = {
            'prediction': 'unknown',
            'confidence': 0.0
        }
    
    # Get NewsAPI verification
    try:
        verifier = IntegratedVerifier()
        newsapi_results = verifier.verify_claim_with_newsapi(claim, max_results=MAX_ARTICLES)
        results['newsapi_verification'] = newsapi_results
    except Exception as e:
        print(f"{Fore.RED}Error in NewsAPI verification: {str(e)}{Style.RESET_ALL}")
        results['newsapi_verification'] = {
            'error': str(e),
            'total_articles': 0
        }
    
    # Combine results
    if 'ml_prediction' in results and results['ml_prediction']['prediction'] != 'unknown':
        ml_score = 1.0 if results['ml_prediction']['prediction'] == 'real' else 0.0
        ml_confidence = results['ml_prediction']['confidence']
    else:
        ml_score = 0.5
        ml_confidence = 0.0
    
    if 'newsapi_verification' in results and 'average_similarity' in results['newsapi_verification']:
        newsapi_score = results['newsapi_verification']['average_similarity']
        supporting_articles = results['newsapi_verification'].get('supporting_articles', 0)
        total_articles = results['newsapi_verification'].get('total_articles', 0)
        newsapi_confidence = supporting_articles / total_articles if total_articles > 0 else 0.0
    else:
        newsapi_score = 0.5
        newsapi_confidence = 0.0
    
    # Calculate weighted score (60% ML, 40% NewsAPI)
    weighted_score = (0.6 * ml_score * ml_confidence) + (0.4 * newsapi_score * newsapi_confidence)
    
    # Determine final verdict
    if weighted_score >= 0.6:
        verdict = 'LIKELY REAL'
    elif weighted_score <= 0.4:
        verdict = 'LIKELY FAKE'
    else:
        verdict = 'UNCERTAIN'
    
    results['final_verdict'] = {
        'verdict': verdict,
        'confidence': weighted_score
    }
    
    return results

def verify_article(article, model, vectorizer):
    """
    Verify an article using ML model
    
    Args:
        article (str): Article to verify
        model: Trained classifier
        vectorizer: TF-IDF vectorizer
        
    Returns:
        dict: Verification results
    """
    results = {}
    
    # Get ML prediction
    if model is not None and vectorizer is not None:
        prediction, probability = get_ml_prediction(article, model, vectorizer)
        results['ml_prediction'] = {
            'prediction': prediction,
            'confidence': probability
        }
        
        # For articles, we rely primarily on the ML model
        if prediction == 'real':
            verdict = 'LIKELY REAL'
        else:
            verdict = 'LIKELY FAKE'
        
        results['final_verdict'] = {
            'verdict': verdict,
            'confidence': probability
        }
    else:
        results['ml_prediction'] = {
            'prediction': 'unknown',
            'confidence': 0.0
        }
        results['final_verdict'] = {
            'verdict': 'UNCERTAIN',
            'confidence': 0.0
        }
    
    return results

def format_verification_results(text, results, is_article_flag):
    """
    Format verification results for display
    
    Args:
        text (str): Original input text
        results (dict): Verification results
        is_article_flag (bool): Whether the input is an article
        
    Returns:
        str: Formatted results
    """
    # Determine text type
    text_type = "ARTICLE" if is_article_flag else "CLAIM"
    
    # Get verdict
    verdict = results['final_verdict']['verdict']
    confidence = results['final_verdict']['confidence']
    
    # Format header
    header = f"\n{Fore.CYAN}===== Verification Results ====={Style.RESET_ALL}"
    
    # Format input summary
    input_summary = f"\n{Fore.YELLOW}Input Type:{Style.RESET_ALL} {text_type}"
    input_summary += f"\n{Fore.YELLOW}Input Text:{Style.RESET_ALL} {text[:100]}..." if len(text) > 100 else f"\n{Fore.YELLOW}Input Text:{Style.RESET_ALL} {text}"
    
    # Format verdict
    if verdict == 'LIKELY REAL':
        verdict_color = Fore.GREEN
    elif verdict == 'LIKELY FAKE':
        verdict_color = Fore.RED
    else:
        verdict_color = Fore.YELLOW
    
    verdict_text = f"\n{Fore.YELLOW}Verdict:{Style.RESET_ALL} {verdict_color}{verdict}{Style.RESET_ALL}"
    verdict_text += f"\n{Fore.YELLOW}Confidence:{Style.RESET_ALL} {confidence:.2f}"
    
    # Format ML results
    ml_prediction = results['ml_prediction']['prediction']
    ml_confidence = results['ml_prediction']['confidence']
    
    if ml_prediction == 'real':
        ml_color = Fore.GREEN
    elif ml_prediction == 'fake':
        ml_color = Fore.RED
    else:
        ml_color = Fore.YELLOW
    
    ml_text = f"\n\n{Fore.CYAN}ML Classification:{Style.RESET_ALL}"
    ml_text += f"\n{Fore.YELLOW}Prediction:{Style.RESET_ALL} {ml_color}{ml_prediction.upper()}{Style.RESET_ALL}"
    ml_text += f"\n{Fore.YELLOW}Confidence:{Style.RESET_ALL} {ml_confidence:.2f}"
    
    # Format NewsAPI results (for claims only)
    newsapi_text = ""
    if not is_article_flag and 'newsapi_verification' in results:
        newsapi = results['newsapi_verification']
        
        newsapi_text = f"\n\n{Fore.CYAN}NewsAPI Verification:{Style.RESET_ALL}"
        
        if 'error' in newsapi:
            newsapi_text += f"\n{Fore.RED}Error:{Style.RESET_ALL} {newsapi['error']}"
        else:
            total_articles = newsapi.get('total_articles', 0)
            supporting_articles = newsapi.get('supporting_articles', 0)
            contradicting_articles = newsapi.get('contradicting_articles', 0)
            avg_similarity = newsapi.get('average_similarity', 0.0)
            
            newsapi_text += f"\n{Fore.YELLOW}Total Articles:{Style.RESET_ALL} {total_articles}"
            newsapi_text += f"\n{Fore.YELLOW}Supporting Articles:{Style.RESET_ALL} {supporting_articles}"
            newsapi_text += f"\n{Fore.YELLOW}Contradicting Articles:{Style.RESET_ALL} {contradicting_articles}"
            newsapi_text += f"\n{Fore.YELLOW}Average Similarity:{Style.RESET_ALL} {avg_similarity:.2f}"
            
            # Add top articles
            if 'detailed_results' in newsapi and newsapi['detailed_results']:
                newsapi_text += f"\n\n{Fore.CYAN}Top Related Articles:{Style.RESET_ALL}"
                
                for i, article in enumerate(newsapi['detailed_results'][:3], 1):
                    newsapi_text += f"\n{Fore.YELLOW}{i}. {article['title']}{Style.RESET_ALL}"
                    newsapi_text += f"\n   URL: {article['url']}"
                    newsapi_text += f"\n   Similarity: {article['similarity_score']:.2f}"
    
    # Combine all sections
    formatted_results = header + input_summary + verdict_text + ml_text + newsapi_text
    
    return formatted_results

def main():
    """Main function"""
    # Get user input
    text = get_user_input()
    
    if not text:
        print(f"{Fore.RED}Error: No input provided.{Style.RESET_ALL}")
        return
    
    # Determine if input is an article or a claim
    is_article_flag = is_article(text)
    
    # Load ML model
    model, vectorizer = load_ml_model()
    
    # Verify input
    if is_article_flag:
        print(f"\n{Fore.CYAN}Analyzing article...{Style.RESET_ALL}")
        results = verify_article(text, model, vectorizer)
    else:
        print(f"\n{Fore.CYAN}Verifying claim...{Style.RESET_ALL}")
        results = verify_claim(text, model, vectorizer)
    
    # Format and display results
    formatted_results = format_verification_results(text, results, is_article_flag)
    print(formatted_results)

if __name__ == "__main__":
    main()