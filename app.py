from flask import Flask, render_template, request, redirect, url_for, session, Response, flash
import sqlite3
import os
import cv2
import datetime
from facerec.click_photos import click
from facerec.train_faces import trainer
from facerec.identify_faces import identify_faces
from facerec.faster_video_stream import generate_frames

app = Flask(__name__)
app.secret_key = 'secret_key_for_session'

DATABASE = 'database.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        admin = db.execute('SELECT * FROM admin WHERE username = ? AND password = ?', (username, password)).fetchone()
        if admin:
            session['role'] = 'admin'
            session['user'] = username
            return redirect(url_for('admin_dashboard'))
        flash('Invalid Credentials', 'error')
    return render_template('admin_login.html')

@app.route('/operator_login', methods=['GET', 'POST'])
def operator_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        op = db.execute('SELECT * FROM operator WHERE username = ? AND password = ?', (username, password)).fetchone()
        if op:
            session['role'] = 'operator'
            session['user'] = username
            return redirect(url_for('operator_dashboard'))
        flash('Invalid Credentials', 'error')
    return render_template('operator_login.html')

@app.route('/operator_register', methods=['GET', 'POST'])
def operator_register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        db = get_db()
        try:
            db.execute('INSERT INTO operator (username, password, name) VALUES (?, ?, ?)', (username, password, name))
            db.commit()
            flash('Registered Successfully', 'success')
            return redirect(url_for('operator_login'))
        except sqlite3.IntegrityError:
            flash('Username already exists', 'error')
    return render_template('operator_register.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    return render_template('admin_dashboard.html')

@app.route('/operator_dashboard')
def operator_dashboard():
    if session.get('role') != 'operator':
        return redirect(url_for('operator_login'))
    return render_template('operator_dashboard.html')

@app.route('/add_passenger', methods=['GET', 'POST'])
def add_passenger():
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    if request.method == 'POST':
        pid = request.form['pid']
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        address = request.form['address']
        contact = request.form['contact']
        nid = request.form['nid']
        status = request.form['status']
        db = get_db()
        try:
            db.execute('INSERT INTO passenger (pid, name, age, gender, address, contact, nid, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                       (pid, name, age, gender, address, contact, nid, status))
            db.commit()
            flash('Passenger added successfully', 'success')
            return redirect(url_for('admin_dashboard'))
        except sqlite3.IntegrityError:
            flash('Passenger ID already exists', 'error')
    return render_template('add_passenger.html')

@app.route('/view_passengers')
def view_passengers():
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    db = get_db()
    passengers = db.execute('SELECT * FROM passenger').fetchall()
    return render_template('view_passengers.html', passengers=passengers)

@app.route('/add_photos/<pid>')
def add_photos(pid):
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    db = get_db()
    passenger = db.execute('SELECT * FROM passenger WHERE pid = ?', (pid,)).fetchone()
    if passenger:
        cam = cv2.VideoCapture(0)
        ss="http://10.231.44.154:8080/videofeed"
        # cam = cv2.VideoCapture(0)  #wecam calling
        # cam = cv2.VideoCapture(ss) #Extransal
        # We need to adjust click function to take name and id
        # Original click function in click_photos.py uses 'app/facerec/dataset/...'
        # We might need to patch it or just call it if it works with root.
        click(passenger['name'], passenger['pid'], cam)
        flash(f'Photos captured for {passenger["name"]}', 'success')
    return redirect(url_for('view_passengers'))

@app.route('/train_model')
def train_model():
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    # Calls trainer from facerec/train_faces.py
    trainer()
    flash('Model trained successfully', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/performance_metrics')
def performance_metrics():
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    # For now, these metrics are from the abstract/description
    metrics = {
        'insightface_precision': 97.53,
        'insightface_recall': 100,
        'insightface_f1': 98.76,
        'yolov8_precision': 95.93,
        'yolov8_recall': 99
    }
    return render_template('performance_metrics.html', metrics=metrics)

@app.route('/view_operators')
def view_operators():
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    db = get_db()
    operators = db.execute('SELECT * FROM operator').fetchall()
    return render_template('view_operators.html', operators=operators)

@app.route('/search_passenger', methods=['GET', 'POST'])
def search_passenger():
    if session.get('role') != 'operator':
        return redirect(url_for('operator_login'))
    passenger = None
    if request.method == 'POST':
        pid = request.form['pid']
        db = get_db()
        passenger = db.execute('SELECT * FROM passenger WHERE pid = ?', (pid,)).fetchone()
        if not passenger:
            flash('Passenger not found', 'error')
    return render_template('search_passenger.html', passenger=passenger)

@app.route('/surveillance')
def surveillance():
    """Live surveillance page with in-browser MJPEG webcam stream."""
    if session.get('role') != 'operator':
        return redirect(url_for('operator_login'))
    return render_template('surveillance.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG streaming endpoint — consumed by <img src='/video_feed'> in the browser."""
    if session.get('role') != 'operator':
        return redirect(url_for('operator_login'))
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/recognition_history')
def recognition_history():
    if session.get('role') != 'operator' and session.get('role') != 'admin':
        return redirect(url_for('index'))
    db = get_db()
    history = db.execute('''
        SELECT rh.*, p.name, p.status 
        FROM recognition_history rh 
        JOIN passenger p ON rh.passenger_id = p.pid 
        ORDER BY rh.timestamp DESC
    ''').fetchall()
    return render_template('recognition_history.html', history=history)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists('facerec/dataset'):
        os.makedirs('facerec/dataset')
    if not os.path.exists('facerec/models'):
        os.makedirs('facerec/models')
    if not os.path.exists('facerec/detected'):
        os.makedirs('facerec/detected')
    if not os.path.exists('static/detected'):
        os.makedirs('static/detected')
    app.run(debug=True)
