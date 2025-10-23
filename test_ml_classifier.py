#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import TextPreprocessor
from src.train_model import FakeNewsClassifier

def test_ml_classifier():
    """
    Test the ML classifier module separately to verify it's working correctly.
    """
    print("=== Testing ML Classifier Module ===\n")
    
    # Initialize the classifier
    models_dir = 'models'
    classifier = FakeNewsClassifier(models_dir)
    
    # Load the model directly from files
    try:
        # First try to load from the standard location
        if os.path.exists(os.path.join(models_dir, 'classifier.pkl')):
            classifier.model = joblib.load(os.path.join(models_dir, 'classifier.pkl'))
            classifier.vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
            print("Successfully loaded ML model and vectorizer.")
        else:
            # If not found, try to train a new model
            print("Model files not found. Training a new model...")
            # Add the project root to sys.path
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from src.dataset_handler import FakeNewsNetHandler
            
            # Initialize dataset handler
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
            dataset_handler = FakeNewsNetHandler(data_dir)
            
            # Load and prepare dataset
            train_df, val_df, test_df = dataset_handler.load_and_prepare_dataset()
            
            # Train the model
            classifier.train(train_df, val_df)
            print("Model trained successfully.")
    except Exception as e:
        print(f"Error loading or training model: {e}")
        return False
    
    # Test with sample texts - balanced mix of real and fake news
    test_texts = [
        "Climate change is causing global temperatures to rise dramatically",  # Likely true
        "Aliens have been confirmed to be living among us by the government",  # Definitely fake
        "The Earth is flat and NASA is hiding the truth from everyone",        # Definitely fake
        "The sun rises in the east and sets in the west",                      # Definitely true
        "Vaccines are safe and effective according to scientific research",    # Definitely true
        "Breaking news: The president has been secretly replaced by a clone",  # Definitely fake
        "Scientists have discovered that drinking water causes cancer",        # Definitely fake
        "The stock market closed higher today after positive economic data"    # Likely true
    ]
    
    print("\nTesting classifier with sample texts:")
    
    for i, text in enumerate(test_texts, 1):
        # Use the classifier's predict method with confidence threshold
        prediction, probabilities, top_features, is_confident = classifier.predict(text)
        
        # Print results
        print(f"\nSample {i}: \"{text}\"")
        print(f"  Prediction: {'FAKE' if prediction == 1 else 'REAL'}")
        print(f"  Confidence: {max(probabilities):.4f}")
        print(f"  Is prediction confident: {'Yes' if is_confident else 'No'}")
        print(f"  Raw probabilities: FAKE={probabilities[1]:.4f}, REAL={probabilities[0]:.4f}")
        
        # Print top features
        print("  Top contributing features:")
        for j, (feature, contribution, _) in enumerate(top_features[:3], 1):
            print(f"    {j}. {feature}: {contribution:.4f}")
    
    print("\n=== ML Classifier Test Complete ===")
    return True

if __name__ == "__main__":
    test_ml_classifier()