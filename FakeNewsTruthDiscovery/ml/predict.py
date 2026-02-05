import joblib
import os
import re
from config import Config
import numpy as np

VECTORIZER_PATH = os.path.join(Config.MODELS_FOLDER, 'vectorizer.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def predict_text(text, algo_name=None):
    if not os.path.exists(VECTORIZER_PATH):
        return {'label': 'Error', 'probability': 0.0, 'message': 'Vectorizer not found. Train a model first.'}
    
    vectorizer = joblib.load(VECTORIZER_PATH)
    processed_text = clean_text(text)
    vec_text = vectorizer.transform([processed_text])
    
    # Load model (Default to Random Forest or best performing? Layout says "User can test data")
    # Usually we pick the best model or a specific one. 
    # I'll default to 'Ensemble' if exists, else 'Random_Forest', else first found.
    
    model_file = None
    if algo_name:
        model_file = f"{algo_name.replace(' ', '_')}.pkl"
    else:
        # Check priority
        for name in ['Ensemble.pkl', 'Random_Forest.pkl', 'Logistic_Regression.pkl']:
            if os.path.exists(os.path.join(Config.MODELS_FOLDER, name)):
                model_file = name
                break
    
    if not model_file:
         # Check dynamic list
         files = os.listdir(Config.MODELS_FOLDER)
         models = [f for f in files if f.endswith('.pkl') and f != 'vectorizer.pkl']
         if models:
             model_file = models[0]
         else:
             return {'label': 'Error', 'probability': 0.0, 'message': 'No trained models found.'}
             
    model = joblib.load(os.path.join(Config.MODELS_FOLDER, model_file))
    
    prediction = model.predict(vec_text)[0]
    try:
        proba = model.predict_proba(vec_text)[0]
        max_proba = float(np.max(proba))
    except:
        max_proba = 1.0 # SVM without probability=True might fail, or others
        
    return {
        'label': str(prediction),
        'probability': max_proba,
        'model': model_file
    }
