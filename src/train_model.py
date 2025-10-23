"""
ML Classifier Training Module for Fake News Detection
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
# Fix import path when running directly
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import TextPreprocessor

class FakeNewsClassifier:
    """Fake News Classifier using TF-IDF features and Logistic Regression"""
    
    def __init__(self, models_dir):
        """
        Initialize the classifier
        
        Args:
            models_dir (str): Directory to save trained models
        """
        self.models_dir = models_dir
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=5,
            max_df=0.8
        )
        # Using a class_weight setting that favors fake news detection
        # We need to find a middle ground between the original bias (everything FAKE)
        # and our overcorrection (everything REAL)
        self.model = LogisticRegression(
            C=1.0,
            class_weight={0: 0.5, 1: 1.5},  # Strongly favor detecting fake news (class 1)
            max_iter=1000,
            random_state=42
        )
        
        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    
    def preprocess_data(self, texts):
        """
        Preprocess a list of texts
        
        Args:
            texts (list): List of text strings
            
        Returns:
            list: List of preprocessed text strings
        """
        preprocessed_texts = []
        for text in texts:
            _, preprocessed_text = self.preprocessor.preprocess(text)
            preprocessed_texts.append(preprocessed_text)
        
        return preprocessed_texts
    
    def train(self, train_df, val_df=None):
        """
        Train the classifier
        
        Args:
            train_df (pd.DataFrame): Training data with 'text' and 'label' columns
            val_df (pd.DataFrame): Validation data with 'text' and 'label' columns
            
        Returns:
            dict: Training metrics
        """
        # Preprocess training data
        X_train_raw = train_df['text'].values
        X_train_preprocessed = self.preprocess_data(X_train_raw)
        y_train = train_df['label'].values
        
        # Fit TF-IDF vectorizer and transform training data
        X_train_tfidf = self.vectorizer.fit_transform(X_train_preprocessed)
        
        # Apply SMOTE with a full balancing strategy
        print("Applying SMOTE with full balancing to the training dataset...")
        # Use sampling_strategy=1.0 to create a completely balanced dataset
        # This means the minority class will be resampled to match the majority class
        smote = SMOTE(sampling_strategy=1.0, random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
        print(f"Original class distribution: {np.bincount(y_train)}")
        print(f"Moderately balanced class distribution: {np.bincount(y_train_smote)}")
        
        # Train the model on moderately balanced data
        self.model.fit(X_train_smote, y_train_smote)
        
        # Get training predictions
        y_train_pred = self.model.predict(X_train_tfidf)
        
        # Calculate training metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred)
        }
        
        # Validate if validation data is provided
        val_metrics = None
        if val_df is not None:
            X_val_raw = val_df['text'].values
            X_val_preprocessed = self.preprocess_data(X_val_raw)
            y_val = val_df['label'].values
            
            # Transform validation data
            X_val_tfidf = self.vectorizer.transform(X_val_preprocessed)
            
            # Get validation predictions
            y_val_pred = self.model.predict(X_val_tfidf)
            
            # Calculate validation metrics
            val_metrics = {
                'accuracy': accuracy_score(y_val, y_val_pred),
                'precision': precision_score(y_val, y_val_pred),
                'recall': recall_score(y_val, y_val_pred),
                'f1': f1_score(y_val, y_val_pred),
                'confusion_matrix': confusion_matrix(y_val, y_val_pred).tolist()
            }
        
        # Save the model and vectorizer
        self.save_model()
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
    
    def predict(self, text, confidence_threshold=0.70):
        """
        Predict if a text is fake news with confidence threshold
        
        Args:
            text (str): Input text
            confidence_threshold (float): Minimum confidence required for a definitive prediction
            
        Returns:
            tuple: (prediction, probability, top_features, is_confident)
        """
        # Preprocess the text
        _, preprocessed_text = self.preprocessor.preprocess(text)
        
        # Transform to TF-IDF features
        X_tfidf = self.vectorizer.transform([preprocessed_text])
        
        # Get prediction and probability
        probability = self.model.predict_proba(X_tfidf)[0]
        
        # Apply confidence threshold
        max_prob = max(probability)
        is_confident = max_prob >= confidence_threshold
        
        # If confidence is below threshold, use a more conservative approach
        if is_confident:
            prediction = self.model.predict(X_tfidf)[0]
        else:
            # If not confident, return the class with higher probability
            # but also indicate low confidence
            prediction = np.argmax(probability)
        
        # Get top contributing features
        top_features = self._get_top_features(X_tfidf, n=10)
        
        return prediction, probability, top_features, is_confident
    
    def _get_top_features(self, X_tfidf, n=10):
        """
        Get top contributing features for a prediction
        
        Args:
            X_tfidf (scipy.sparse.csr.csr_matrix): TF-IDF features
            n (int): Number of top features to return
            
        Returns:
            list: List of (feature, weight) tuples
        """
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get coefficients from the model
        coefficients = self.model.coef_[0]
        
        # Get TF-IDF values for the input
        tfidf_values = X_tfidf.toarray()[0]
        
        # Calculate feature contributions (coefficient * tfidf_value)
        contributions = coefficients * tfidf_values
        
        # Get indices of non-zero TF-IDF values
        non_zero_indices = np.where(tfidf_values > 0)[0]
        
        # Get feature names, contributions, and TF-IDF values for non-zero indices
        features = [(feature_names[i], contributions[i], tfidf_values[i]) for i in non_zero_indices]
        
        # Sort by absolute contribution value (descending)
        features.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Return top n features
        return features[:n]
    
    def save_model(self):
        """Save the trained model and vectorizer"""
        model_path = os.path.join(self.models_dir, 'fake_news_model.pkl')
        vectorizer_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self):
        """Load the trained model and vectorizer"""
        model_path = os.path.join(self.models_dir, 'fake_news_model.pkl')
        vectorizer_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print(f"Model loaded from {model_path}")
            print(f"Vectorizer loaded from {vectorizer_path}")
            return True
        else:
            print("Model or vectorizer not found. Please train the model first.")
            return False


if __name__ == "__main__":
    # Test the classifier
    from dataset_handler import FakeNewsNetHandler
    
    # Initialize dataset handler and classifier
    data_dir = "../data"
    models_dir = "../models"
    
    dataset_handler = FakeNewsNetHandler(data_dir)
    classifier = FakeNewsClassifier(models_dir)
    
    # Download and prepare dataset if not already done
    if not os.path.exists(os.path.join(data_dir, 'fakenewsnet/processed/train.csv')):
        dataset_handler.download_minimal_dataset()
        train_df, val_df, test_df = dataset_handler.prepare_dataset()
    else:
        # Load prepared dataset
        train_df = pd.read_csv(os.path.join(data_dir, 'fakenewsnet/processed/train.csv'))
        val_df = pd.read_csv(os.path.join(data_dir, 'fakenewsnet/processed/val.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'fakenewsnet/processed/test.csv'))
    
    # Train the classifier
    metrics = classifier.train(train_df, val_df)
    
    # Print metrics
    print("\nTraining Metrics:")
    for metric, value in metrics['train_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nValidation Metrics:")
    for metric, value in metrics['val_metrics'].items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['val_metrics']['confusion_matrix'])
    
    # Test on a sample
    sample_text = "Breaking: Scientists discover that vaccines cause autism"
    prediction, probability, top_features, is_confident = classifier.predict(sample_text)
    
    print(f"\nSample text: {sample_text}")
    print(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
    print(f"Probability: Fake: {probability[1]:.4f}, Real: {probability[0]:.4f}")
    print(f"Confidence: {'High' if is_confident else 'Low'}")
    
    print("\nTop contributing features:")
    for feature, contribution, tfidf_value in top_features:
        print(f"{feature}: contribution={contribution:.4f}, tfidf={tfidf_value:.4f}")