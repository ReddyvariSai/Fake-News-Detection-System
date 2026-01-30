# 🚨 Fake News Detection using Machine Learning

## 📋 Project Overview
This project implements a machine learning pipeline to detect fake news articles. The system processes textual data, extracts meaningful features, trains multiple classification models, and evaluates their performance.

## ✨ Features
- **Text Preprocessing**: Cleaning, tokenization, lemmatization
- **Feature Extraction**: TF-IDF, n-grams, text statistics
- **Multiple Models**: Random Forest, Logistic Regression, XGBoost
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score
- **Prediction Pipeline**: Real-time classification of new articles

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone repository
git clone <repository-url>
cd Machine-Learning-for-Truth-Discovery

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
