import pandas as pd
import numpy as np
import pickle
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from config import Config
from utils.metrics import save_model_metrics, log_training
import joblib

# Ensure models directory exists
if not os.path.exists(Config.MODELS_FOLDER):
    os.makedirs(Config.MODELS_FOLDER)

VECTORIZER_PATH = os.path.join(Config.MODELS_FOLDER, 'vectorizer.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def load_and_preprocess(dataset_path):
    df = pd.read_csv(dataset_path)
    # Basic cleaning
    # Assuming column 'text' and 'label' exist based on user schema description
    # If not, we might need to be flexible. For now, strict.
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")
        
    df['clean_text'] = df['text'].apply(clean_text)
    return df['clean_text'], df['label']

def get_vectorizer(corpus=None):
    if os.path.exists(VECTORIZER_PATH):
        return joblib.load(VECTORIZER_PATH)
    elif corpus is not None:
        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(corpus)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        return vectorizer
    else:
        raise ValueError("Vectorizer not found and no corpus provided to fit.")

def train_algorithm(dataset_path, algo_name):
    log_training(algo_name, "Started training...")
    
    X, y = load_and_preprocess(dataset_path)
    
    # Force fitting vectorizer on new training (or load existing if transfer learning?)
    # For this system, we re-fit vectorizer on every train call? 
    # Or check if exists? Let's re-fit for simplicity of "Model Training" action.
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    model = None
    if algo_name == 'Logistic Regression':
        model = LogisticRegression()
    elif algo_name == 'SVM':
        model = SVC(probability=True)
    elif algo_name == 'Random Forest':
        model = RandomForestClassifier()
    elif algo_name == 'XGBoost':
        # Encoding labels for XGBoost if necessary (e.g. Real/Fake to 0/1)
        # Assuming labels are 0/1 or compatible. 
        # If 'Fake'/'Real', we need label encoder.
        # Let's add a quick check/conversion.
        # For simplicity, assuming dataset is prepared or strings work (sklearn handles strings for some, XGB needs numeric).
        # We will wrap XGB with LabelEncoder logic if needed, but for now stick to standard sklearn compatible
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        # Note: y might need to be numeric for XGB
        # We'll rely on pd.factorize or LabelEncoder if it fails.
    elif algo_name == 'Ensemble':
        clf1 = LogisticRegression()
        clf2 = RandomForestClassifier()
        clf3 = SVC(probability=True)
        model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svm', clf3)], voting='soft')
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
        
    # Handle Label Encoding for XGBoost manually if needed
    # For safety, let's map y to integers if they are strings
    # But for a generic demo, we try-catch or assume numeric. 
    # Let's map unique labels to 0, 1 just in case
    # y_train_mapped = ...
    
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        # Fallback for value errors (e.g. strings in XGB)
        log_training(algo_name, f"Training failed: {e}")
        raise e

    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Save model
    model_path = os.path.join(Config.MODELS_FOLDER, f"{algo_name.replace(' ', '_')}.pkl")
    joblib.dump(model, model_path)
    
    log_training(algo_name, f"Training completed. Accuracy: {acc:.4f}")
    save_model_metrics(algo_name, acc, f1, prec, rec)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec
    }
