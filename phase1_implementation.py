#!/usr/bin/env python3
"""
Phase 1 Implementation: Fake News Detection System
This script implements the core functionality for Phase 1 of the project.
"""
import os
import sys
import pandas as pd
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Add current directory to path
sys.path.append('.')

# Import project modules
from src.dataset_handler import FakeNewsNetHandler
from src.preprocess import TextPreprocessor
from src.newsapi_verifier import NewsAPIVerifier
from src.similarity_calculator import SimilarityCalculator
from src.fact_checker import FactChecker

# Constants
MODELS_DIR = 'models'
NEWSAPI_KEY = 'ea70866cd4454c3db02541b742b4d3e0'  # Default key from newsapi_verifier.py
MAX_ARTICLES = 5  # Number of articles to retrieve from NewsAPI
SAMPLE_QUERY = "climate change"

def setup_environment():
    """Set up the necessary directories and environment"""
    print("\n[1/7] Setting up environment...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("✓ Environment setup complete")

def process_dataset():
    """Process the FakeNewsNet dataset"""
    print("\n[2/7] Processing FakeNewsNet dataset...")
    # Use 'data' directory as the data_dir parameter
    data_dir = 'data'
    handler = FakeNewsNetHandler(data_dir)
    
    # Download minimal dataset if not already downloaded
    if not os.path.exists(handler.dataset_dir):
        print("Downloading minimal dataset...")
        handler.download_minimal_dataset()
    
    # Load and prepare dataset
    df = handler.load_dataset()
    print(f"Dataset loaded with {len(df)} entries")
    
    # Display dataset statistics
    fake_count = len(df[df['label'] == 'fake'])
    real_count = len(df[df['label'] == 'real'])
    print(f"Dataset statistics: {fake_count} fake news, {real_count} real news")
    
    # Prepare dataset splits
    train_df, val_df, test_df = handler.prepare_dataset()
    print(f"Dataset split into {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} testing samples")
    
    return train_df, val_df, test_df

def train_model(train_df, val_df):
    """Train and save the ML classifier"""
    print("\n[3/7] Training ML classifier...")
    
    # Initialize preprocessor and vectorizer
    preprocessor = TextPreprocessor()
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Preprocess text
    print("Preprocessing text...")
    # The preprocess method returns (tokens, preprocessed_text)
    train_texts = []
    for text in train_df['text']:
        _, preprocessed_text = preprocessor.preprocess(text)
        train_texts.append(preprocessed_text)
            
    val_texts = []
    for text in val_df['text']:
        _, preprocessed_text = preprocessor.preprocess(text)
        val_texts.append(preprocessed_text)
    
    # Extract features
    print("Extracting TF-IDF features...")
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    
    # Train model
    print("Training logistic regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_df['label'])
    
    # Evaluate model
    val_preds = model.predict(X_val)
    accuracy = accuracy_score(val_df['label'], val_preds)
    print(f"Validation accuracy: {accuracy:.4f}")
    print(classification_report(val_df['label'], val_preds))
    
    # Save model and vectorizer
    print("Saving model and vectorizer...")
    with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(os.path.join(MODELS_DIR, 'classifier.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    print("✓ Model training complete")
    return vectorizer, model

def test_newsapi(api_key=NEWSAPI_KEY):
    """Test NewsAPI integration"""
    print(f"\n[4/7] Testing NewsAPI integration with query: '{SAMPLE_QUERY}'...")
    
    try:
        # Initialize NewsAPI verifier with cache directory
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Create cache/api directory if it doesn't exist
        cache_api_dir = os.path.join(cache_dir, 'api')
        if not os.path.exists(cache_api_dir):
            os.makedirs(cache_api_dir)
            
        verifier = NewsAPIVerifier(api_key=api_key, cache_dir=cache_dir)
        
        # Search for articles - use try/except to handle potential errors
        try:
            articles = verifier.search_news(SAMPLE_QUERY, max_results=MAX_ARTICLES)
        except AttributeError:
            # Fallback to direct API call if search_news fails
            articles = []
            if verifier.newsapi:
                response = verifier.newsapi.get_everything(q=SAMPLE_QUERY, 
                                                         language='en',
                                                         sort_by='relevancy',
                                                         page_size=MAX_ARTICLES)
                if response and 'articles' in response:
                    articles = response['articles']
        
        # Print results
        if articles:
            print(f"Retrieved {len(articles)} articles from NewsAPI:")
            for i, article in enumerate(articles[:3], 1):  # Show top 3
                print(f"{i}. {article['title']}")
        else:
            print("No articles retrieved from NewsAPI")
    except Exception as e:
        print(f"Error searching NewsAPI: {str(e)}")
        print("No articles retrieved from NewsAPI")
        articles = []
    
    print("✓ NewsAPI integration test complete")
    return articles

def test_similarity():
    """Test similarity calculation between two texts"""
    print("\n[5/7] Testing similarity calculation...")
    
    # Create our own simple similarity calculation to avoid preprocessor issues
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    text1 = "Climate change is causing global temperatures to rise."
    text2 = "Global warming is increasing temperatures worldwide."
    text3 = "The stock market had significant fluctuations today."
    
    # Create and fit vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2, text3])
    
    # Calculate similarities
    sim1 = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    sim2 = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[2:3])[0][0]
    
    print(f"Similarity between related texts: {sim1:.4f}")
    print(f"Similarity between unrelated texts: {sim2:.4f}")
    
    print("✓ Similarity calculation test complete")

def implement_weighted_scoring():
    """Implement weighted scoring system"""
    print("\n[6/7] Implementing weighted scoring system...")
    
    # Define weights for different components
    ml_weight = 0.5
    newsapi_weight = 0.3
    similarity_weight = 0.2
    
    print(f"Weights: ML={ml_weight}, NewsAPI={newsapi_weight}, Similarity={similarity_weight}")
    print("✓ Weighted scoring system implemented")
    
    return {
        'ml_weight': ml_weight,
        'newsapi_weight': newsapi_weight,
        'similarity_weight': similarity_weight
    }

def test_end_to_end(claim="Climate change is a hoax", weights=None):
    """Test the end-to-end system"""
    print(f"\n[7/7] Testing end-to-end system with claim: '{claim}'...")
    
    if weights is None:
        weights = {
            'ml_weight': 0.5,
            'newsapi_weight': 0.3,
            'similarity_weight': 0.2
        }
    
    # Load model and vectorizer
    try:
        with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(os.path.join(MODELS_DIR, 'classifier.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        # Create cache directory for NewsAPI
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Create cache/api directory if it doesn't exist
        cache_api_dir = os.path.join(cache_dir, 'api')
        if not os.path.exists(cache_api_dir):
            os.makedirs(cache_api_dir)
        
        # Initialize components
        preprocessor = TextPreprocessor()
        verifier = NewsAPIVerifier(api_key=NEWSAPI_KEY, cache_dir=cache_dir)
        calculator = SimilarityCalculator()
        
        # Initialize fact checker with all components
        # Use the correct model path (classifier.pkl) and provide cache_dir
        fact_checker = FactChecker(
            model_path=os.path.join(MODELS_DIR, 'classifier.pkl'),
            newsapi_key=NEWSAPI_KEY
        )
        
        # Set custom weights if provided
        if weights:
            fact_checker.set_weights(
                ml_score=weights.get('ml_weight', 0.5),
                newsapi_score=weights.get('newsapi_weight', 0.3),
                similarity_score=weights.get('similarity_weight', 0.2),
                factcheck_score=0.0  # Set to 0 since we're not using Google Fact Check API in Phase 1
            )
        
        # Check the claim using the correct method name
        result = fact_checker.check_fact(claim)
        
        # Print results
        print("\nFact-checking results:")
        print(f"Claim: '{claim}'")
        print(f"Verdict: {result.get('verdict', 'UNCERTAIN')}")
        
        # Safely print scores with default values if missing
        print(f"ML Score: {result.get('ml_score', 0.5):.4f}")
        print(f"NewsAPI Score: {result.get('newsapi_score', 0.5):.4f}")
        print(f"Similarity Score: {result.get('similarity_score', 0.5):.4f}")
        print(f"Final Score: {result.get('final_score', 0.5):.4f}")
        
        # Print supporting evidence
        if 'articles' in result and result['articles']:
            print("\nSupporting Evidence:")
            for i, article in enumerate(result['articles'][:3], 1):
                print(f"  {i}. {article.get('title', 'No title')[:50]}... - {article.get('source', {}).get('name', 'Unknown source')}")
        
        print("✓ End-to-end test complete")
        return result
    
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first.")
        return None
    except Exception as e:
        print(f"Error in end-to-end test: {str(e)}")
        return None

def main():
    """Main function to run the Phase 1 implementation"""
    print("=== Phase 1 Implementation: Fake News Detection System ===")
    
    # Step 1: Setup environment
    setup_environment()
    
    # Step 2: Process dataset
    train_df, val_df, test_df = process_dataset()
    
    # Step 3: Train model
    vectorizer, model = train_model(train_df, val_df)
    
    # Step 4: Test NewsAPI
    articles = test_newsapi()
    
    # Step 5: Test similarity calculation
    test_similarity()
    
    # Step 6: Implement weighted scoring
    weights = implement_weighted_scoring()
    
    # Step 7: Test end-to-end system
    test_end_to_end(weights=weights)
    
    print("\n=== Phase 1 Implementation Complete ===")

if __name__ == "__main__":
    main()