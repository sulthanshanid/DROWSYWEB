from flask import Flask, jsonify, request, send_from_directory, render_template, redirect, url_for
import sqlite3
import os
import json
import numpy as np
from datetime import datetime
import face_recognition

app = Flask(__name__)

DB_NAME = 'drowsiness_data.db'
PHOTO_FOLDER = 'photos'

os.makedirs(PHOTO_FOLDER, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS drowsiness (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id TEXT NOT NULL,
            status TEXT NOT NULL,
            duration INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            photo_path TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS drivers (
            driver_id TEXT PRIMARY KEY,
            name TEXT,
            face_encoding TEXT,
            sample_photo TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM drowsiness ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    data = [
        {"id": row[0], "driver_id": row[1], "status": row[2], "duration": row[3],
         "timestamp": row[4], "photo_path": row[5]} for row in rows
    ]
    return jsonify(data)

@app.route('/api/data', methods=['POST'])
def add_data():
    status = request.form.get('status')
    duration = request.form.get('duration')
    timestamp = request.form.get('timestamp')
    image = request.files.get('image')
    driver_id = "Unknown"

    if image:
        filename = f"event_{timestamp}.jpg"
        path = os.path.join(PHOTO_FOLDER, filename)
        image.save(path)
        photo_path = f"/photos/{filename}"

        # Load image and encode
        img = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            new_encoding = encodings[0]
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("SELECT driver_id, face_encoding FROM drivers")
            rows = cursor.fetchall()

            match_found = False
            for row in rows:
                stored_driver_id, encoding_json = row
                known_encoding = np.array(json.loads(encoding_json))
                matches = face_recognition.compare_faces([known_encoding], new_encoding)
                if matches[0]:
                    driver_id = stored_driver_id
                    match_found = True
                    break

            if not match_found:
                driver_id = f"P{len(rows)+1}"
                encoding_str = json.dumps(new_encoding.tolist())
                cursor.execute('''
                    INSERT INTO drivers (driver_id, face_encoding, sample_photo)
                    VALUES (?, ?, ?)
                ''', (driver_id, encoding_str, photo_path))
                conn.commit()
            conn.close()

    else:
        photo_path = None

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO drowsiness (driver_id, status, duration, timestamp, photo_path)
        VALUES (?, ?, ?, ?, ?)
    ''', (driver_id, status, duration, timestamp, photo_path))
    conn.commit()
    conn.close()

    return jsonify({"message": "Data added successfully"}), 201

@app.route('/photos/<filename>')
def get_photo(filename):
    return send_from_directory(PHOTO_FOLDER, filename)

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT d.id, d.driver_id, COALESCE(dr.name, d.driver_id) as driver_name, d.status, d.duration, d.timestamp, d.photo_path
        FROM drowsiness d
        LEFT JOIN drivers dr ON d.driver_id = dr.driver_id
        ORDER BY d.id DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    data = [
        {"id": row[0], "driver_id": row[1], "driver_name": row[2], "status": row[3], "duration": row[4],
         "timestamp": row[5], "photo_path": row[6]} for row in rows
    ]
    return render_template("dashboard.html", entries=data)


@app.route('/stats')
def stats():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT driver_id, name, sample_photo FROM drivers")
    drivers = cursor.fetchall()

    face_stats = []
    for driver in drivers:
        driver_id, name, sample_photo = driver
        cursor.execute("SELECT COUNT(*), SUM(duration) FROM drowsiness WHERE driver_id=?", (driver_id,))
        count, total_duration = cursor.fetchone()
        face_stats.append({
            "driver_id": driver_id,
            "name": name if name else "Unnamed",
            "sample_photo": sample_photo,
            "events": count,
            "total_duration": total_duration if total_duration else 0
        })
    conn.close()
    return render_template("stats.html", face_stats=face_stats)

@app.route('/update-name/<driver_id>', methods=['POST'])
def update_name(driver_id):
    name = request.form.get('name')
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE drivers SET name = ? WHERE driver_id = ?", (name, driver_id))
    conn.commit()
    conn.close()
    return redirect(url_for('stats'))

import pickle
MODEL_PATH = 'driver_classification_model.pkl'

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        scaler, model = pickle.load(f)
    return scaler, model
import joblib

@app.route('/summary')
def summary():
    model = joblib.load("driver_classification_model.pkl")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT driver_id, COUNT(*), SUM(duration)
        FROM drowsiness
        GROUP BY driver_id
    """)
    data = cursor.fetchall()
    conn.close()

    summary_data = []
    for driver_id, event_count, total_duration in data:
        features = np.array([[event_count, total_duration]])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        confidence = max(proba) * 100
        classification = prediction  # Since your model predicts 'Good' or 'Bad' directly
        summary_data.append({
            'driver_id': driver_id,
            'classification': classification,
            'confidence': f"{confidence:.2f}%"
        })

    return render_template('summary.html', summary_data=summary_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
