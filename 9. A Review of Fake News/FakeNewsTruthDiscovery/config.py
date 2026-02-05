import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-very-secret'
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DB_PATH = os.path.join(BASE_DIR, 'database', 'fakenews.db')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MODELS_FOLDER = os.path.join(BASE_DIR, 'ml', 'models')
    LOGS_FOLDER = os.path.join(BASE_DIR, 'logs')
    
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
