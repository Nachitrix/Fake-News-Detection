"""
Cache Manager for API responses and computation results
"""
import os
import json
import hashlib
import time
import pickle
from typing import Any, Dict, Optional, Union

class CacheManager:
    """
    Manages caching for API responses and computation results
    to improve performance and reduce API calls
    """
    
    def __init__(self, cache_dir: Optional[str] = None, expiry_time: int = 86400):
        """
        Initialize the cache manager
        
        Args:
            cache_dir: Directory to store cache files (default: ./cache)
            expiry_time: Cache expiry time in seconds (default: 24 hours)
        """
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'cache')
        self.expiry_time = expiry_time
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create subdirectories for different cache types
        self.api_cache_dir = os.path.join(self.cache_dir, 'api')
        self.model_cache_dir = os.path.join(self.cache_dir, 'model')
        
        os.makedirs(self.api_cache_dir, exist_ok=True)
        os.makedirs(self.model_cache_dir, exist_ok=True)
    
    def _generate_key(self, data: Any) -> str:
        """
        Generate a unique cache key from input data
        
        Args:
            data: Input data to generate key from
            
        Returns:
            str: Unique cache key
        """
        if isinstance(data, str):
            serialized = data.encode('utf-8')
        else:
            serialized = json.dumps(data, sort_keys=True).encode('utf-8')
        
        return hashlib.md5(serialized).hexdigest()
    
    def get_api_cache(self, key_data: Any) -> Optional[Dict]:
        """
        Get cached API response
        
        Args:
            key_data: Data to generate cache key from
            
        Returns:
            Optional[Dict]: Cached response or None if not found/expired
        """
        key = self._generate_key(key_data)
        cache_file = os.path.join(self.api_cache_dir, f"{key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is expired
            if time.time() - cache_data['timestamp'] > self.expiry_time:
                return None
            
            return cache_data['data']
        except Exception:
            return None
    
    def set_api_cache(self, key_data: Any, data: Dict) -> None:
        """
        Cache API response
        
        Args:
            key_data: Data to generate cache key from
            data: Response data to cache
        """
        key = self._generate_key(key_data)
        cache_file = os.path.join(self.api_cache_dir, f"{key}.json")
        
        cache_data = {
            'timestamp': time.time(),
            'data': data
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception:
            pass  # Silently fail if caching fails
    
    def get_model_cache(self, key_data: Any) -> Optional[Any]:
        """
        Get cached model computation result
        
        Args:
            key_data: Data to generate cache key from
            
        Returns:
            Optional[Any]: Cached result or None if not found/expired
        """
        key = self._generate_key(key_data)
        cache_file = os.path.join(self.model_cache_dir, f"{key}.pkl")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if cache is expired
            if time.time() - cache_data['timestamp'] > self.expiry_time:
                return None
            
            return cache_data['data']
        except Exception:
            return None
    
    def set_model_cache(self, key_data: Any, data: Any) -> None:
        """
        Cache model computation result
        
        Args:
            key_data: Data to generate cache key from
            data: Result data to cache
        """
        key = self._generate_key(key_data)
        cache_file = os.path.join(self.model_cache_dir, f"{key}.pkl")
        
        cache_data = {
            'timestamp': time.time(),
            'data': data
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception:
            pass  # Silently fail if caching fails
    
    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Clear cache files
        
        Args:
            cache_type: Type of cache to clear ('api', 'model', or None for all)
        """
        if cache_type == 'api' or cache_type is None:
            for file in os.listdir(self.api_cache_dir):
                if file.endswith('.json'):
                    try:
                        os.remove(os.path.join(self.api_cache_dir, file))
                    except Exception:
                        pass
        
        if cache_type == 'model' or cache_type is None:
            for file in os.listdir(self.model_cache_dir):
                if file.endswith('.pkl'):
                    try:
                        os.remove(os.path.join(self.model_cache_dir, file))
                    except Exception:
                        pass

# Test the cache manager
if __name__ == "__main__":
    cache = CacheManager()
    
    # Test API cache
    test_data = {"query": "test query", "params": {"param1": "value1"}}
    test_response = {"results": [{"title": "Test Result"}]}
    
    cache.set_api_cache(test_data, test_response)
    cached_response = cache.get_api_cache(test_data)
    
    print(f"API Cache Test: {'Success' if cached_response == test_response else 'Failed'}")
    
    # Test model cache
    test_model_data = {"text": "This is a test text for model cache"}
    test_model_result = {"prediction": 0.75, "features": [0.1, 0.2, 0.3]}
    
    cache.set_model_cache(test_model_data, test_model_result)
    cached_model_result = cache.get_model_cache(test_model_data)
    
    print(f"Model Cache Test: {'Success' if cached_model_result == test_model_result else 'Failed'}")