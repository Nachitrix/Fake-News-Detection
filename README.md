# Hybrid Real-Time Fake News Detection System

## Project Title
**Hybrid Real-Time Fake News Detection System with Multi-Modal Verification**

A comprehensive terminal-based application that combines offline machine learning classification, real-time news verification through NewsAPI, content similarity analysis, and optional fact-checking scraping to provide accurate fake news detection with confidence scoring.

## Description
In an era of rampant misinformation, this project presents a robust solution for identifying fake news. It leverages a hybrid approach, integrating machine learning models, real-time external API verification, and advanced text similarity algorithms to provide a multi-faceted assessment of news credibility. The system is designed to be fast, accurate, and adaptable, offering a confidence score for each claim analyzed.

## Features
- **Multi-Modal Verification**: Combines ML classification, NewsAPI verification, and content similarity analysis.
- **Real-Time Analysis**: Integrates with NewsAPI for up-to-the-minute news cross-referencing.
- **Offline Capability**: Utilizes pre-trained machine learning models for rapid classification without internet dependency.
- **Configurable Weights**: Allows users to adjust the influence of different verification methods on the final score.
- **Intelligent Caching**: Optimizes performance and reduces API calls through a robust caching mechanism.
- **Comprehensive Text Preprocessing**: Handles cleaning, tokenization, lemmatization, and more.
- **Semantic Similarity**: Employs Sentence Transformers for deeper contextual understanding.
- **User-Friendly Interface**: Provides clear, colored output in the terminal for easy interpretation.
- **Modular Architecture**: Designed for extensibility and maintainability.

## Installation

### Prerequisites
- Python 3.8+
- `pip` (Python package installer)

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up NewsAPI Key:**
   - Obtain a free API key from [NewsAPI.org](https://newsapi.org/).
   - Create a `.env` file in the project root directory and add your API key:
     ```
     NEWS_API_KEY=YOUR_NEWS_API_KEY
     ```

5. **Download NLTK data:**
   The project uses NLTK for text processing. You'll need to download the `punkt` and `wordnet` corpora:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

6. **Train the ML Model (if not already provided):**
   The project includes a `train_model.py` script to train the machine learning classifier. If `classifier.pkl` and `tfidf_vectorizer.pkl` are not present in the `models/` directory, you can train them:
   ```bash
   python src/train_model.py
   ```
   This will download the FakeNewsNet dataset and train the model.

## Usage

To use the Fake News Detection system, run the `main.py` script with your desired text or URL.

### Basic Usage
```bash
python main.py --text "NASA confirmed that 2020 was tied with 2016 for the warmest year on record"
```

### Using a URL
```bash
python main.py --url "https://www.reuters.com/article/us-climate-change-nasa-idUSKBN29P20B"
```

### Customizing Weights
You can adjust the weights for ML, NewsAPI, and Similarity scores:
```bash
python main.py --text "Some claim the Earth is flat" --ml_weight 0.5 --news_weight 0.2 --sim_weight 0.3
```

### Example Output
```
🔍 FAKE NEWS DETECTION RESULTS 🔍
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 Input Text: NASA confirmed that 2020 was tied with 2016 for the warmest year on record

🤖 ML Classification:
   Prediction: REAL (Confidence: 0.89)

🌐 NewsAPI Verification:
   Found 8 supporting articles
   Average relevance: 0.92
   Top sources: Reuters, BBC News, CNN

📊 Similarity Analysis:
   Content similarity: 0.87
   Semantic similarity: 0.91

🎯 FINAL VERDICT: LIKELY TRUE
   Overall Confidence: 0.89
   Recommendation: This claim appears to be factually accurate
```

## Project Structure
```
.env
├── cache/                  # Stores cached API responses and model data
├── data/                   # Contains raw and processed datasets
├── models/                 # Stores trained ML models (classifier.pkl, tfidf_vectorizer.pkl)
├── src/
│   ├── article_scraper.py  # Extracts content from news articles
│   ├── cache_manager.py    # Manages caching for API requests
│   ├── dataset_handler.py  # Handles FakeNewsNet dataset loading and preprocessing
│   ├── fact_checker.py     # Main logic for combining verification methods
│   ├── factcheck_api.py    # (Optional) Integration with external fact-checking APIs
│   ├── integrated_verifier.py # Orchestrates multiple verification sources
│   ├── newsapi_verifier.py # Verifies claims using NewsAPI
│   ├── preprocess.py       # Text cleaning and preprocessing utilities
│   ├── similarity_calculator.py # Calculates content and semantic similarity
│   └── train_model.py      # Script for training the ML model
├── main.py                 # Entry point for the application
├── requirements.txt        # Project dependencies
├── Mini_Project_Report.md  # Detailed project report
├── README.md               # This file
├── test_*.py               # Unit and integration tests
└── venv/                   # Python virtual environment
```

## Technologies Used
- **Python 3.12**
- **Machine Learning**: `scikit-learn`, `numpy`, `pandas`
- **Natural Language Processing**: `nltk`, `spacy`, `sentence-transformers`
- **Web Scraping & APIs**: `requests`, `beautifulsoup4`, `newsapi-python`, `newspaper3k`
- **Utilities**: `joblib`, `colorama`, `tqdm`, `python-dotenv`

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes and ensure tests pass.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Create a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details. (Note: A LICENSE file is not provided in the current codebase, but is a standard practice.)