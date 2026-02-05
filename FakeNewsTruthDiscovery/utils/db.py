import sqlite3
import os
from config import Config
from datetime import datetime

def get_db_connection():
    conn = sqlite3.connect(Config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    if not os.path.exists(os.path.dirname(Config.DB_PATH)):
        os.makedirs(os.path.dirname(Config.DB_PATH))
    conn = get_db_connection()
    c = conn.cursor()
    
    # Users Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            phone TEXT,
            address TEXT,
            role TEXT NOT NULL DEFAULT 'user'
        )
    ''')
    
    # Datasets Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            rows INTEGER,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Data of Datasets Table
    # "id,text,label,author_followers,author_verified,retweets,likes,shares,credibility_score,sentiment"
    c.execute('''
        CREATE TABLE IF NOT EXISTS dataofdatasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            text TEXT,
            label TEXT,
            author_followers INTEGER,
            author_verified INTEGER,
            retweets INTEGER,
            likes INTEGER,
            shares INTEGER,
            credibility_score REAL,
            sentiment REAL,
            FOREIGN KEY (dataset_id) REFERENCES datasets (id)
        )
    ''')

    # Models Table
    # "id,name,accuracy,f1,precision,recall,trained_at"
    c.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            accuracy REAL,
            f1 REAL,
            precision REAL,
            recall REAL,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Training Logs Table
    # "id,algo,message,time"
    c.execute('''
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            algo TEXT,
            message TEXT,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Predictions Table
    # "id,user_id,text,predicted_label,prob,time"
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            text TEXT,
            predicted_label TEXT,
            prob REAL,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Tasks Table (for admin monitoring)
    # "id,name,status,start,end"
    c.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            status TEXT,
            start TIMESTAMP,
            end TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized successfully.")

if __name__ == '__main__':
    if not os.path.exists(os.path.dirname(Config.DB_PATH)):
        os.makedirs(os.path.dirname(Config.DB_PATH))
    init_db()
