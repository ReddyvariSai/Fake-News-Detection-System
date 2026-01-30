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


```

Machine-Learning-for-Truth-Discovery/
│
├── README.md
│
├── data/
│   ├── raw/
│   │   └── fake_news.csv
│   │
│   └── processed/
│       └── cleaned_news.csv
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── prediction.py
│
├── models/
│   └── fake_news_model.pkl
│
├── results/
│   ├── accuracy_report.txt
│   └── confusion_matrix.png
│
├── docs/
│   ├── literature_review.md
│   ├── methodology.md
│   └── social_impact.md
│
├── requirements.txt
│
└── main.py                                            

```
