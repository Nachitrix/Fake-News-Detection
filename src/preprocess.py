"""
NLP Preprocessing Module for Fake News Detection
"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    """Text preprocessing class for cleaning and normalizing text data"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Clean text by removing URLs, HTML tags, punctuation, and extra whitespace
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text):
        """
        Tokenize text into words
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            list: List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from list of tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: List of tokens with stopwords removed
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text):
        """
        Full preprocessing pipeline: clean, tokenize, remove stopwords, lemmatize
        
        Args:
            text (str): Raw input text
            
        Returns:
            list: Preprocessed tokens
            str: Preprocessed text (tokens joined by space)
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Remove stopwords
        tokens_no_stop = self.remove_stopwords(tokens)
        
        # Lemmatize
        lemmatized_tokens = self.lemmatize_tokens(tokens_no_stop)
        
        # Join tokens back to text
        preprocessed_text = ' '.join(lemmatized_tokens)
        
        return lemmatized_tokens, preprocessed_text


if __name__ == "__main__":
    # Test the preprocessing pipeline
    preprocessor = TextPreprocessor()
    sample_text = "This is a sample news article with some URLs https://example.com and HTML <b>tags</b>. It contains punctuation, stopwords, and needs to be lemmatized!"
    
    tokens, processed_text = preprocessor.preprocess(sample_text)
    print(f"Original text: {sample_text}")
    print(f"Processed text: {processed_text}")
    print(f"Tokens: {tokens}")