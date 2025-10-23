"""
Dataset Handler for FakeNewsNet dataset
"""
import os
import pandas as pd
import requests
from tqdm import tqdm
import json
import time

class FakeNewsNetHandler:
    """Handler for downloading and processing FakeNewsNet dataset"""
    
    def __init__(self, data_dir):
        """
        Initialize the dataset handler
        
        Args:
            data_dir (str): Directory to store the dataset
        """
        self.data_dir = data_dir
        self.dataset_dir = os.path.join(data_dir, 'fakenewsnet')
        self.raw_dir = os.path.join(self.dataset_dir, 'raw')
        self.processed_dir = os.path.join(self.dataset_dir, 'processed')
        
        # Create directories if they don't exist
        for directory in [self.dataset_dir, self.raw_dir, self.processed_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def download_minimal_dataset(self):
        """
        Download the minimal FakeNewsNet dataset (CSV files with news IDs, URLs, titles)
        """
        # GitHub repository URLs for the minimal dataset
        base_url = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/"
        files = [
            "politifact_fake.csv",
            "politifact_real.csv",
            "gossipcop_fake.csv",
            "gossipcop_real.csv"
        ]
        
        print("Downloading minimal FakeNewsNet dataset...")
        
        for file in tqdm(files, desc="Downloading files"):
            url = base_url + file
            save_path = os.path.join(self.raw_dir, file)
            
            # Skip if file already exists
            if os.path.exists(save_path):
                print(f"File {file} already exists. Skipping download.")
                continue
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"Downloaded {file} successfully.")
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {file}: {e}")
    
    def load_dataset(self, source='all'):
        """
        Load the FakeNewsNet dataset
        
        Args:
            source (str): 'politifact', 'gossipcop', or 'all'
            
        Returns:
            pd.DataFrame: Combined dataset with label column
        """
        if source not in ['politifact', 'gossipcop', 'all']:
            raise ValueError("Source must be 'politifact', 'gossipcop', or 'all'")
        
        sources = [source] if source != 'all' else ['politifact', 'gossipcop']
        dfs = []
        
        for src in sources:
            # Load fake news
            fake_path = os.path.join(self.raw_dir, f"{src}_fake.csv")
            if os.path.exists(fake_path):
                fake_df = pd.read_csv(fake_path)
                fake_df['label'] = 'fake'
                fake_df['source'] = src
                dfs.append(fake_df)
            
            # Load real news
            real_path = os.path.join(self.raw_dir, f"{src}_real.csv")
            if os.path.exists(real_path):
                real_df = pd.read_csv(real_path)
                real_df['label'] = 'real'
                real_df['source'] = src
                dfs.append(real_df)
        
        if not dfs:
            raise FileNotFoundError("Dataset files not found. Please download the dataset first.")
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        return combined_df
    
    def prepare_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Prepare the dataset for training, validation, and testing
        
        Args:
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of data for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # Load the dataset
        df = self.load_dataset()
        
        # Extract features (title) and labels
        X = df['title']
        y = df['label'].map({'fake': 1, 'real': 0})
        
        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Split train+val into train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, 
            random_state=random_state, stratify=y_train_val
        )
        
        # Create dataframes
        train_df = pd.DataFrame({'text': X_train, 'label': y_train})
        val_df = pd.DataFrame({'text': X_val, 'label': y_val})
        test_df = pd.DataFrame({'text': X_test, 'label': y_test})
        
        # Save to processed directory
        train_df.to_csv(os.path.join(self.processed_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.processed_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(self.processed_dir, 'test.csv'), index=False)
        
        print(f"Dataset prepared and saved to {self.processed_dir}")
        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df


if __name__ == "__main__":
    # Test the dataset handler
    handler = FakeNewsNetHandler(data_dir="../data")
    handler.download_minimal_dataset()
    df = handler.load_dataset()
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")
    
    # Prepare the dataset
    train_df, val_df, test_df = handler.prepare_dataset()