from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import sqlite3
from config import Config
from utils.db import get_db_connection, init_db
from utils.auth import hash_password, verify_password
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime

# Initialize DB on start
init_db()

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Routes ---

@app.route('/')
def index():
    if 'user_id' in session:
        if session.get('role') == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('user_dashboard'))
    return redirect(url_for('login'))

# --- Auth ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        
        if user and verify_password(user['password'], password):
            session['user_id'] = user['id']
            session['name'] = user['name']
            session['role'] = user['role']
            
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        elif email == 'admin@svr.com' and password == 'admin123': # Fallback hardcoded admin if DB fail or init issue
             # Create admin if not exists? No, just rely on init_db in main
             pass
             
        flash('Invalid email or password')
            
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        phone = request.form.get('phone', '')
        address = request.form.get('address', '')
        role = 'user' 
        
        hashed_pw = hash_password(password)
        
        try:
            conn = get_db_connection()
            conn.execute('INSERT INTO users (name, email, password, phone, address, role) VALUES (?, ?, ?, ?, ?, ?)',
                         (name, email, hashed_pw, phone, address, role))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already registered.')
            
    return render_template('auth/register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- Admin ---
@app.route('/admin')
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    return render_template('admin/dashboard.html')

@app.route('/admin/upload', methods=['GET', 'POST'])
def upload_dataset():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Basic check for CSV
                df = pd.read_csv(filepath)
                rows = len(df)
                
                conn = get_db_connection()
                conn.execute('INSERT INTO datasets (filename, rows) VALUES (?, ?)', (filename, rows))
                conn.commit()
                conn.close()
                flash(f'Dataset {filename} uploaded successfully with {rows} rows.')
            except Exception as e:
                flash(f"Error processing file: {e}")
                
    return render_template('admin/upload.html')

@app.route('/admin/train')
def train_model_view():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    return render_template('admin/train.html')

@app.route('/admin/train/start', methods=['POST'])
def start_training():
    if session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    algo = request.form.get('algorithm')
    from ml.train import train_algorithm
    
    try:
        conn = get_db_connection()
        dataset = conn.execute('SELECT * FROM datasets ORDER BY id DESC LIMIT 1').fetchone()
        conn.close()
        
        if not dataset:
            return jsonify({'status': 'error', 'message': 'No dataset found. Upload one first.'})
            
        dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset['filename'])
        
        metrics = train_algorithm(dataset_path, algo)
        
        return jsonify({'status': 'success', 'metrics': metrics})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/admin/predictions')
def admin_predictions():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    return render_template('predict.html') 

@app.route('/admin/monitor')
def cloud_monitor():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    return render_template('admin/monitor.html')

@app.route('/admin/reports')
def performance_reports():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    models = conn.execute('SELECT * FROM models').fetchall()
    conn.close()
    return render_template('admin/reports.html', models=models)

@app.route('/admin/logs')
def view_logs():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    logs = conn.execute('SELECT * FROM training_logs ORDER BY time DESC').fetchall()
    conn.close()
    return render_template('admin/logs.html', logs=logs)

# --- User ---
@app.route('/user')
def user_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('user/dashboard.html')

@app.route('/user/test', methods=['GET', 'POST'])
def test_data():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    result = None
    if request.method == 'POST':
        text = request.form['text']
        from ml.predict import predict_text
        prediction = predict_text(text)
        
        result = prediction
        
        conn = get_db_connection()
        conn.execute('INSERT INTO predictions (user_id, text, predicted_label, prob) VALUES (?, ?, ?, ?)',
                     (session['user_id'], text, prediction['label'], prediction['probability']))
        conn.commit()
        conn.close()
        
    return render_template('user/test.html', result=result)

@app.route('/user/history')
def user_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    history = conn.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY time DESC', (session['user_id'],)).fetchall()
    conn.close()
    return render_template('user/history.html', history=history)

if __name__ == '__main__':
    # Auto-initialize DB and Admin
    try:
        init_db()
        conn = get_db_connection()
        if not conn.execute("SELECT * FROM users WHERE email='admin@svr.com'").fetchone():
            conn.execute("INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
                         ('Administrator', 'admin@svr.com', hash_password('admin123'), 'admin'))
            conn.commit()
            print("Admin account created: admin@svr.com / admin123")
        conn.close()
    except Exception as e:
        print(f"Startup error: {e}")
        
    app.run(debug=True)
