import cv2
import pickle
import face_recognition
import datetime
import os
import sqlite3
from cachetools import TTLCache
import numpy as np

cache = TTLCache(maxsize=20, ttl=10)


def identify_and_save(frame, name, buf, buf_length, known_conf):
    """Save a snapshot and log to DB when a passenger is confidently recognised."""
    if name in cache:
        return
    count = sum(ele.count(name) for ele in buf)
    if count >= known_conf:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Detected: {name} at {timestamp}")
        cache[name] = 'detected'

        os.makedirs('static/detected', exist_ok=True)
        filename = '{}_{}.jpg'.format(name, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = 'static/detected/{}'.format(filename)
        cv2.imwrite(path, frame)
        try:
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute("SELECT pid FROM passenger WHERE name = ?", (name,))
            res = cursor.fetchone()
            if res:
                pid = res[0]
                cursor.execute(
                    "INSERT INTO recognition_history (passenger_id, image_path) VALUES (?, ?)",
                    (pid, filename)
                )
                conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving to DB: {e}")


def predict(rgb_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    """Run KNN-based face recognition on an RGB frame."""
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either through knn_clf or model_path")
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=1)
    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    return [
        (pred, loc) if rec else ("unknown", loc)
        for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)
    ]


def generate_frames():
    """
    MJPEG generator for Flask streaming Response.
    Yields multipart JPEG frames with face-recognition annotations.
    Stream this via:
        Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    """
    video_capture = cv2.VideoCapture(0)

    # ── Webcam unavailable ────────────────────────────────────────────────────
    if not video_capture.isOpened():
        err = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err, "Webcam not accessible!", (60, 230),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(err, "Check camera connection.", (80, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1)
        _, buf = cv2.imencode('.jpg', err)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        return

    model_path = "facerec/models/trained_model.clf"
    model_loaded = os.path.exists(model_path)

    buf_length = 10
    known_conf = 5
    buf = [[]] * buf_length
    idx = 0
    process_this_frame = True
    predictions = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # ── Face recognition (every other frame for performance) ─────────────
        if model_loaded:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb   = np.ascontiguousarray(small[:, :, ::-1])

            if process_this_frame:
                try:
                    predictions = predict(rgb, model_path=model_path)
                except Exception as e:
                    predictions = []
                    print(f"Prediction error: {e}")

            process_this_frame = not process_this_frame
            face_names = []

            for name, (top, right, bottom, left) in predictions:
                top *= 4; right *= 4; bottom *= 4; left *= 4
                color = (0, 200, 0) if name != "unknown" else (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                if name != "unknown":
                    identify_and_save(frame, name, buf, buf_length, known_conf)
                    face_names.append(name)

            buf[idx] = face_names
            idx = (idx + 1) % buf_length

        else:
            cv2.putText(frame, "Model not trained — showing raw feed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # ── Timestamp overlay ─────────────────────────────────────────────────
        ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # ── Encode and yield ──────────────────────────────────────────────────
        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    video_capture.release()
