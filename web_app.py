from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import base64
import io
from PIL import Image
import threading
import webbrowser

app = Flask(__name__)
app.secret_key = 'voting_system_secret_key'

# Global variables for face recognition
knn = None
LABELS = None
FACES = None
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_face_data():
    global knn, LABELS, FACES
    if os.path.exists('data/names.pkl') and os.path.exists('data/faces_data.pkl'):
        with open('data/names.pkl', 'rb') as f:
            LABELS = pickle.load(f)
        with open('data/faces_data.pkl', 'rb') as f:
            FACES = pickle.load(f)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(FACES, LABELS)

load_face_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/voter_login')
def voter_login():
    return render_template('voter_login.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/admin_stats')
def admin_stats():
    # Count total registered voters
    total_voters = 0
    if os.path.exists('data/names.pkl'):
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        total_voters = len(set(names))  # Unique Aadhar numbers

    # Count votes cast
    votes_cast = 0
    if os.path.exists('Votes.csv'):
        with open('Votes.csv', 'r') as f:
            votes_cast = sum(1 for line in f) - 1  # Subtract header

    # For now, duplicate attempts is a placeholder
    duplicate_attempts = 0

    return jsonify({
        'total_voters': total_voters,
        'votes_cast': votes_cast,
        'duplicate_attempts': duplicate_attempts
    })

@app.route('/enter_details', methods=['POST'])
def enter_details():
    name = request.form['name']
    dob = request.form['dob']
    aadhar = request.form['aadhar']
    voterid = request.form['voterid']
    phone = request.form['phone']

    # Validate Date of Birth and Age
    try:
        dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
        current_date = datetime.now().date()
        age = current_date.year - dob_date.year - ((current_date.month, current_date.day) < (dob_date.month, dob_date.day))
        if age < 18:
            flash('You must be at least 18 years old to vote.')
            return redirect(url_for('index'))
        if dob_date > current_date:
            flash('Date of Birth cannot be in the future.')
            return redirect(url_for('index'))
    except ValueError:
        flash('Invalid Date of Birth format.')
        return redirect(url_for('index'))

    # Validate Aadhar Card Number
    if not aadhar.isdigit() or len(aadhar) != 12:
        flash('Aadhar Card Number must be exactly 12 digits and contain only numbers.')
        return redirect(url_for('index'))

    # Validate Voter ID Card Number
    if not voterid.isalnum() or len(voterid) != 10:
        flash('Voter ID must be exactly 10 alphanumeric characters (A-Z, 0-9).')
        return redirect(url_for('index'))

    # Validate Phone Number
    if not phone.isdigit() or len(phone) != 10:
        flash('Phone Number must be exactly 10 digits.')
        return redirect(url_for('index'))

    # Check if Aadhar is already registered
    if os.path.exists('data/names.pkl'):
        with open('data/names.pkl', 'rb') as f:
            existing_names = pickle.load(f)
        if aadhar in existing_names:
            flash('This Aadhar Card Number is already registered. Please use a different one.')
            return redirect(url_for('index'))

    # Check if Voter ID is already registered
    if os.path.exists('data/voterids.pkl'):
        with open('data/voterids.pkl', 'rb') as f:
            existing_voterids = pickle.load(f)
        if voterid in existing_voterids:
            flash('This Voter ID is already registered. Please use a different one.')
            return redirect(url_for('index'))

    # Store details in session
    session['user_details'] = {
        'name': name,
        'dob': dob,
        'aadhar': aadhar,
        'voterid': voterid,
        'phone': phone
    }
    return redirect(url_for('capture_face', aadhar=aadhar))

@app.route('/capture_face/<aadhar>')
def capture_face(aadhar):
    return render_template('capture.html', aadhar=aadhar)

@app.route('/save_face/<aadhar>', methods=['POST'])
def save_face(aadhar):
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # Check for similar faces if KNN is trained
        if knn is not None:
            distances, indices = knn.kneighbors(resized_img, n_neighbors=1)
            if distances[0][0] < 0.05:  # Even stricter threshold for similarity to prevent duplicates
                return jsonify({'success': False, 'error': 'Face already registered. Please use a different face.'})

        # Save face data - replicate to match original structure (51 samples per person)
        frames_total = 51
        faces_data_single = resized_img
        faces_data = np.tile(faces_data_single, (frames_total, 1))

        # Get voterid from session
        user_details = session.get('user_details', {})
        voterid = user_details.get('voterid', '')

        # Save face data
        if not os.path.exists('data/'):
            os.makedirs('data/')

        if 'names.pkl' not in os.listdir('data/'):
            names = [aadhar] * frames_total
            voterids = [voterid] * frames_total
            all_faces_data = faces_data
        else:
            with open('data/names.pkl', 'rb') as f:
                names = pickle.load(f)
            names += [aadhar] * frames_total
            if 'voterids.pkl' not in os.listdir('data/'):
                voterids = [voterid] * frames_total
            else:
                with open('data/voterids.pkl', 'rb') as f:
                    voterids = pickle.load(f)
                voterids += [voterid] * frames_total
            with open('data/faces_data.pkl', 'rb') as f:
                all_faces_data = pickle.load(f)
            all_faces_data = np.append(all_faces_data, faces_data, axis=0)

        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
        with open('data/voterids.pkl', 'wb') as f:
            pickle.dump(voterids, f)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(all_faces_data, f)

        load_face_data()  # Reload data
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/vote')
def vote():
    return render_template('vote.html')

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0 and knn is not None:
        (x, y, w, h) = faces[0]
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        voter_exist = check_if_exists(output[0])
        if voter_exist:
            # Get vote details for already voted user
            vote_details = get_vote_details_for_aadhar(output[0])
            return jsonify({
                'recognized': True,
                'name': output[0],
                'already_voted': True,
                'vote_details': vote_details
            })
        return jsonify({'recognized': True, 'name': output[0], 'already_voted': False})
    return jsonify({'recognized': False})

@app.route('/submit_vote', methods=['POST'])
def submit_vote():
    data = request.get_json()
    name = data['name']
    vote = data['vote']
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

    exist = os.path.isfile("Votes.csv")
    if exist:
        with open("Votes.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            attendance = [name, vote, date, timestamp]
            writer.writerow(attendance)
    else:
        with open("Votes.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['NAME', 'VOTE', 'DATE', 'TIME'])
            attendance = [name, vote, date, timestamp]
            writer.writerow(attendance)

    # Get user details from session
    user_details = session.get('user_details', {})
    aadhar = user_details.get('aadhar', '')
    voterid = user_details.get('voterid', '')

    # Store vote details in session for popup
    session['vote_details'] = {
        'name': name,
        'voterid': voterid,
        'vote': vote,
        'date': date,
        'time': timestamp,
        'aadhar': aadhar
    }

    return jsonify({'success': True, 'date': date, 'time': timestamp, 'aadhar': aadhar})

@app.route('/get_vote_details')
def get_vote_details():
    vote_details = session.get('vote_details', {})
    return jsonify(vote_details)



def check_if_exists(value):
    try:
        with open("Votes.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == value:
                    return True
    except FileNotFoundError:
        pass
    return False

def get_vote_details_for_aadhar(aadhar):
    try:
        with open("Votes.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header
            for row in reader:
                if row and row[0] == aadhar:
                    return {
                        'vote': row[1],
                        'date': row[2],
                        'time': row[3]
                    }
    except FileNotFoundError:
        pass
    return None

if __name__ == '__main__':
    def open_browser():
        webbrowser.open_new('http://127.0.0.1:5000/')
    threading.Timer(1, open_browser).start()
    app.run(debug=True)
