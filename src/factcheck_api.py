"""
Google Fact Check API Integration Module
"""
import os
import json
import time
import requests

class GoogleFactCheckAPI:
    """Google Fact Check API integration for claim verification"""
    
    def __init__(self, api_key=None, cache_dir=None):
        """
        Initialize the Google Fact Check API client
        
        Args:
            api_key (str): Google API key
            cache_dir (str): Directory to cache API responses
        """
        # Use environment variable if API key not provided
        self.api_key = api_key or os.environ.get('GOOGLE_FACTCHECK_KEY')
        
        if not self.api_key:
            print("Warning: Google Fact Check API key not provided. Please set GOOGLE_FACTCHECK_KEY environment variable.")
        
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        
        # Set up caching
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        self.cache_file = os.path.join(cache_dir, 'factcheck_cache.json') if cache_dir else None
        self.cache = self._load_cache() if self.cache_file else {}
    
    def _load_cache(self):
        """Load cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        if self.cache_file:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
    
    def _cache_key(self, query):
        """Generate cache key for a query"""
        return query.lower().strip()
    
    def search_claims(self, query, max_results=10):
        """
        Search for fact-checked claims related to the query
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of fact check results
        """
        if not self.api_key:
            return []
        
        # Check cache first
        cache_key = self._cache_key(query)
        if cache_key in self.cache:
            cache_time = self.cache[cache_key]['timestamp']
            # Cache is valid for 1 day
            if time.time() - cache_time < 86400:
                return self.cache[cache_key]['claims']
        
        # Prepare request parameters
        params = {
            'key': self.api_key,
            'query': query,
            'languageCode': 'en',
            'maxAgeDays': 365,  # Look back up to a year
            'pageSize': max_results
        }
        
        try:
            # Make API request
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            claims = data.get('claims', [])
            
            # Process claims
            processed_claims = []
            for claim in claims:
                processed_claim = {
                    'text': claim.get('text', ''),
                    'claimant': claim.get('claimant', ''),
                    'claim_date': claim.get('claimDate', ''),
                    'review_publisher': claim.get('claimReview', [{}])[0].get('publisher', {}).get('name', ''),
                    'review_title': claim.get('claimReview', [{}])[0].get('title', ''),
                    'review_url': claim.get('claimReview', [{}])[0].get('url', ''),
                    'rating': claim.get('claimReview', [{}])[0].get('textualRating', ''),
                    'rating_value': self._normalize_rating(claim.get('claimReview', [{}])[0].get('textualRating', ''))
                }
                processed_claims.append(processed_claim)
            
            # Cache the results
            if self.cache_file:
                self.cache[cache_key] = {
                    'timestamp': time.time(),
                    'claims': processed_claims
                }
                self._save_cache()
            
            return processed_claims
            
        except Exception as e:
            print(f"Error searching Google Fact Check API: {e}")
            return []
    
    def _normalize_rating(self, rating):
        """
        Normalize textual rating to a 0-1 score
        
        Args:
            rating (str): Textual rating
            
        Returns:
            float: Normalized score (0-1)
        """
        rating = rating.lower()
        
        # True/accurate ratings
        if any(term in rating for term in ['true', 'accurate', 'correct', 'fact', 'verified']):
            return 0.9
        
        # Mostly true ratings
        if any(term in rating for term in ['mostly true', 'mostly correct', 'mostly accurate']):
            return 0.7
        
        # Mixed/unclear ratings
        if any(term in rating for term in ['mixed', 'unclear', 'partially', 'half']):
            return 0.5
        
        # Mostly false ratings
        if any(term in rating for term in ['mostly false', 'mostly incorrect', 'misleading']):
            return 0.3
        
        # False ratings
        if any(term in rating for term in ['false', 'fake', 'incorrect', 'inaccurate', 'pants on fire']):
            return 0.1
        
        # Default to neutral if unknown
        return 0.5
    
    def get_verification_score(self, query):
        """
        Get verification score based on fact check results
        
        Args:
            query (str): Search query
            
        Returns:
            tuple: (score, claims)
                score: 0-1 score (higher means more likely real)
                claims: List of relevant fact check claims
        """
        claims = self.search_claims(query)
        
        # If no claims found, return neutral score
        if not claims:
            return 0.5, []
        
        # Calculate average rating value
        total_rating = sum(claim['rating_value'] for claim in claims)
        avg_rating = total_rating / len(claims)
        
        return avg_rating, claims


if __name__ == "__main__":
    # Test the Google Fact Check API
    api_key = os.environ.get('GOOGLE_FACTCHECK_KEY')
    cache_dir = "../data/cache"
    
    fact_checker = GoogleFactCheckAPI(api_key=api_key, cache_dir=cache_dir)
    
    # Test queries
    test_queries = [
        "COVID-19 vaccines contain microchips",
        "Climate change is a hoax",
        "The Earth is flat"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        score, claims = fact_checker.get_verification_score(query)
        print(f"Verification score: {score:.2f}")
        print(f"Found {len(claims)} fact checks:")
        
        for i, claim in enumerate(claims, 1):
            print(f"{i}. Claim: {claim['text']}")
            print(f"   Claimant: {claim['claimant']}")
            print(f"   Rating: {claim['rating']} (normalized: {claim['rating_value']:.2f})")
            print(f"   Fact-checked by: {claim['review_publisher']}")
            print(f"   URL: {claim['review_url']}")