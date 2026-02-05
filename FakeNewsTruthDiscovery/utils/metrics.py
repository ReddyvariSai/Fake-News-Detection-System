from utils.db import get_db_connection
from datetime import datetime

def log_training(algo, message):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('INSERT INTO training_logs (algo, message, time) VALUES (?, ?, ?)',
              (algo, message, datetime.now()))
    conn.commit()
    conn.close()

def save_model_metrics(name, accuracy, f1, precision, recall):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('INSERT INTO models (name, accuracy, f1, precision, recall, trained_at) VALUES (?, ?, ?, ?, ?, ?)',
              (name, accuracy, f1, precision, recall, datetime.now()))
    conn.commit()
    conn.close()

def get_latest_metrics():
    conn = get_db_connection()
    models = conn.execute('SELECT * FROM models ORDER BY trained_at DESC LIMIT 10').fetchall()
    conn.close()
    return models
