#!/usr/bin/env python3
"""
Fake News Detection System - Terminal Interface
"""
import os
import sys
import argparse
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize colorama
init(autoreset=True)

# Import local modules
from src.fact_checker import FactChecker

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fake News Detection System')
    
    # Main arguments
    parser.add_argument('--text', type=str, help='News text to verify')
    parser.add_argument('--title', type=str, help='News title')
    parser.add_argument('--file', type=str, help='File containing news text')
    
    # Configuration
    parser.add_argument('--model', type=str, help='Path to ML model')
    parser.add_argument('--api-key', type=str, help='NewsAPI key (or set NEWSAPI_KEY env var)')
    parser.add_argument('--cache-dir', type=str, help='Cache directory')
    
    # Weights
    parser.add_argument('--ml-weight', type=float, default=0.4, help='Weight for ML classifier (0-1)')
    parser.add_argument('--newsapi-weight', type=float, default=0.2, help='Weight for NewsAPI verification (0-1)')
    parser.add_argument('--similarity-weight', type=float, default=0.4, help='Weight for similarity calculation (0-1)')
    
    return parser.parse_args()

def get_input_text(args):
    """Get input text from arguments or prompt user"""
    if args.text:
        return args.text, args.title
    
    if args.file:
        try:
            with open(args.file, 'r') as f:
                text = f.read().strip()
            return text, args.title
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    
    # Interactive mode
    print(f"{Fore.CYAN}=== Fake News Detection System ==={Style.RESET_ALL}")
    print("Enter news information to verify:")
    
    title = input(f"{Fore.YELLOW}Title (optional): {Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Text (press Enter twice when done):{Style.RESET_ALL}")
    
    lines = []
    while True:
        line = input()
        if not line and lines and not lines[-1]:
            break
        lines.append(line)
    
    text = '\n'.join(lines).strip()
    
    if not text:
        print("No text provided. Exiting.")
        sys.exit(1)
    
    return text, title or None

def main():
    """Main function"""
    args = parse_arguments()
    
    # Get API keys from args or environment
    newsapi_key = args.api_key or os.environ.get('NEWSAPI_KEY')
    if not newsapi_key:
        print(f"{Fore.YELLOW}Warning: NewsAPI key not provided. Set with --api-key or NEWSAPI_KEY environment variable.{Style.RESET_ALL}")
    
    # Set up fact checker
    checker = FactChecker(
        model_path=args.model,
        newsapi_key=newsapi_key,
        cache_dir=args.cache_dir
    )
    
    # Set custom weights if provided
    checker.set_weights(
        ml_score=args.ml_weight,
        newsapi_score=args.newsapi_weight,
        similarity_score=args.similarity_weight
    )
    
    # Get input text
    text, title = get_input_text(args)
    
    print(f"\n{Fore.CYAN}Analyzing news...{Style.RESET_ALL}")
    
    # Check fact
    result = checker.check_fact(text, title)
    
    # Print result
    checker.print_result(result)

if __name__ == "__main__":
    main()