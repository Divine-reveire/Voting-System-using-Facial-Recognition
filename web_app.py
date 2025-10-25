from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import cv2
import pickle
import _pickle
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
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(filename='admin_logs.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
        if len(FACES) > 0:
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(FACES, LABELS)
        else:
            knn = None
    else:
        knn = None
        LABELS = None
        FACES = None

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
    registered_aadhars = set()
    if os.path.exists('data/voters.pkl'):
        with open('data/voters.pkl', 'rb') as f:
            voters = pickle.load(f)
        total_voters = len(voters)
        registered_aadhars = {voter['aadhar'] for voter in voters}

    # Count unique votes cast (only from registered voters)
    votes_cast = 0
    if os.path.exists('Votes.csv'):
        unique_aadhars = set()
        with open('Votes.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 1 and row[0] in registered_aadhars:
                    unique_aadhars.add(row[0])
        votes_cast = len(unique_aadhars)

    return jsonify({
        'total_voters': total_voters,
        'votes_cast': votes_cast
    })

@app.route('/voting_results')
def voting_results():
    return render_template('voting_results.html')

@app.route('/get_voting_results')
def get_voting_results():
    try:
        results = {}
        total_votes = 0

        # Load registered voters to filter
        registered_aadhars = set()
        if os.path.exists('data/voters.pkl'):
            with open('data/voters.pkl', 'rb') as f:
                voters = pickle.load(f)
            registered_aadhars = {voter['aadhar'] for voter in voters}

        # Load candidates to get icons and filter to managed candidates
        candidate_icons = {}
        managed_candidates = set()
        if os.path.exists('data/candidates.pkl'):
            with open('data/candidates.pkl', 'rb') as f:
                candidates = pickle.load(f)
            for candidate in candidates:
                if isinstance(candidate, dict):
                    name = candidate.get('name', '')
                    managed_candidates.add(name)
                    icon = '👤'  # Default icon
                    if candidate.get('icon'):
                        icon = '📷'  # Use camera icon if custom icon exists
                    candidate_icons[name] = icon
                elif isinstance(candidate, str):
                    managed_candidates.add(candidate)
                    candidate_icons[candidate] = '👤'

        # Initialize results with 0 votes for all managed candidates
        for candidate in managed_candidates:
            results[candidate] = 0

        # Read votes from CSV and filter to only registered voters
        if os.path.exists('Votes.csv'):
            with open('Votes.csv', 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 2 and row[0] in registered_aadhars:
                        candidate = row[1]
                        if candidate in results:  # Only count if it's a managed candidate
                            results[candidate] += 1
                            total_votes += 1

        # Prepare results data - include all managed candidates
        results_data = []
        for name, votes in results.items():
            percentage = round((votes / total_votes * 100), 1) if total_votes > 0 else 0
            icon = candidate_icons.get(name, '👤')  # Default icon if not in candidates.pkl

            results_data.append({
                'name': name,
                'votes': votes,
                'percentage': percentage,
                'icon': icon
            })

        # Sort by votes descending
        results_data.sort(key=lambda x: x['votes'], reverse=True)

        return jsonify({'results': results_data})
    except Exception as e:
        print(f"Error in get_voting_results: {e}")
        return jsonify({'error': str(e)})

@app.route('/get_voters_list')
def get_voters_list():
    voters_list = []

    # Load registered voters to filter
    registered_aadhars = set()
    if os.path.exists('data/voters.pkl'):
        with open('data/voters.pkl', 'rb') as f:
            voters = pickle.load(f)
        registered_aadhars = {voter['aadhar'] for voter in voters}

    # Read votes from CSV and filter to only registered voters
    if os.path.exists('Votes.csv'):
        with open('Votes.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 4 and row[0] in registered_aadhars:
                    aadhar = row[0]
                    vote = row[1]
                    date = row[2]
                    time = row[3]
                    voters_list.append({
                        'aadhar': aadhar,
                        'vote': vote,
                        'date': date,
                        'time': time
                    })

    # Sort by date and time descending (most recent first)
    voters_list.sort(key=lambda x: (x['date'], x['time']), reverse=True)

    return jsonify({'voters': voters_list})

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
            return redirect(url_for('voter_login'))
        if dob_date > current_date:
            flash('Date of Birth cannot be in the future.')
            return redirect(url_for('voter_login'))
    except ValueError:
        flash('Invalid Date of Birth format.')
        return redirect(url_for('voter_login'))

    # Validate Aadhar Card Number
    if not aadhar.isdigit() or len(aadhar) != 12:
        flash('Aadhar Card Number must be exactly 12 digits and contain only numbers.')
        return redirect(url_for('voter_login'))

    # Validate Voter ID Card Number
    if not voterid.isalnum() or len(voterid) != 10:
        flash('Voter ID must be exactly 10 alphanumeric characters (A-Z, 0-9).')
        return redirect(url_for('voter_login'))

    # Validate Phone Number
    if not phone.isdigit() or len(phone) != 10:
        flash('Phone Number must be exactly 10 digits.')
        return redirect(url_for('voter_login'))

    # Load existing voters (only admin-registered voters allowed)
    voters = []
    if os.path.exists('data/voters.pkl'):
        with open('data/voters.pkl', 'rb') as f:
            voters = pickle.load(f)

    # Check if voter exists with exact matching details
    matching_voter = None
    for voter in voters:
        if (voter['name'].lower() == name.lower() and
            voter['dob'] == dob and
            voter['aadhar'] == aadhar and
            voter['voterid'] == voterid and
            voter['phone'] == phone):
            matching_voter = voter
            break

    if not matching_voter:
        flash('Your details do not match our records. Please ensure you are registered through the admin or contact support.')
        return redirect(url_for('voter_login'))

    # Store details in session
    session['user_details'] = {
        'name': matching_voter['name'],
        'dob': matching_voter['dob'],
        'aadhar': matching_voter['aadhar'],
        'voterid': matching_voter['voterid'],
        'phone': matching_voter['phone']
    }
    return redirect(url_for('capture_face', aadhar=aadhar))

@app.route('/capture_face/<aadhar>')
def capture_face(aadhar):
    return render_template('capture.html', aadhar=aadhar)

@app.route('/save_face/<aadhar>', methods=['POST'])
def save_face(aadhar):
    try:
        load_face_data()  # Reload data to ensure KNN is up to date
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        faces = facedetect.detectMultiScale(gray, 1.02, 1)

        print(f"Detected faces: {len(faces)}")  # Debug log

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

            # Check for similar faces if KNN is trained
            if knn is not None:
                distances, indices = knn.kneighbors(resized_img, n_neighbors=1)
                print(f"KNN distance: {distances[0][0]}")  # Debug log
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

            try:
                if 'names.pkl' not in os.listdir('data/'):
                    names = [aadhar] * frames_total
                else:
                    with open('data/names.pkl', 'rb') as f:
                        names = pickle.load(f)
                    names += [aadhar] * frames_total
            except (EOFError, _pickle.UnpicklingError):
                # If corrupted, start fresh
                names = [aadhar] * frames_total

            try:
                if 'voterids.pkl' not in os.listdir('data/'):
                    voterids = [voterid] * frames_total
                else:
                    with open('data/voterids.pkl', 'rb') as f:
                        voterids = pickle.load(f)
                    voterids += [voterid] * frames_total
            except (EOFError, _pickle.UnpicklingError):
                # If corrupted, start fresh
                voterids = [voterid] * frames_total

            try:
                if 'faces_data.pkl' not in os.listdir('data/'):
                    all_faces_data = faces_data
                else:
                    with open('data/faces_data.pkl', 'rb') as f:
                        all_faces_data = pickle.load(f)
                    all_faces_data = np.append(all_faces_data, faces_data, axis=0)
            except (EOFError, _pickle.UnpicklingError):
                # If corrupted, start fresh
                all_faces_data = faces_data

            with open('data/names.pkl', 'wb') as f:
                pickle.dump(names, f)
            with open('data/voterids.pkl', 'wb') as f:
                pickle.dump(voterids, f)
            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(all_faces_data, f)

            load_face_data()  # Reload data
            print("Face saved successfully")  # Debug log
            # After successful face registration, redirect to check vote status
            return jsonify({'success': True, 'redirect': '/check_vote_status'})
        print("No face detected")  # Debug log
        return jsonify({'success': False})
    except Exception as e:
        print(f"Error in save_face: {e}")  # Debug log
        return jsonify({'success': False, 'error': 'Failed to process face data'})

@app.route('/check_vote_status')
def check_vote_status():
    return render_template('check_vote_status.html')

@app.route('/vote')
def vote():
    admin_vote = session.get('admin_vote', False)
    return render_template('vote.html', admin_vote=admin_vote)

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    faces = facedetect.detectMultiScale(gray, 1.02, 1)

    if len(faces) > 0 and knn is not None:
        (x, y, w, h) = faces[0]
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        aadhar = output[0]

        # Get voter details
        voterid = 'N/A'
        name = aadhar
        if os.path.exists('data/voters.pkl'):
            with open('data/voters.pkl', 'rb') as f:
                voters = pickle.load(f)
            voter = next((v for v in voters if v['aadhar'] == aadhar), None)
            if voter:
                voterid = voter['voterid']
                name = voter['name']

        voter_exist = check_if_exists(aadhar)
        if voter_exist:
            # Get vote details for already voted user
            vote_details = get_vote_details_for_aadhar(aadhar)
            return jsonify({
                'recognized': True,
                'name': name,
                'voterid': voterid,
                'already_voted': True,
                'vote_details': vote_details
            })
        return jsonify({'recognized': True, 'name': name, 'voterid': voterid, 'already_voted': False})
    return jsonify({'recognized': False})

@app.route('/submit_vote', methods=['POST'])
def submit_vote():
    try:
        data = request.get_json()
    except Exception as e:
        return jsonify({'success': False, 'error': 'Invalid JSON data'})
    if not data or not isinstance(data, dict):
        return jsonify({'success': False, 'error': 'Invalid request data'})
    if 'name' not in data or 'vote' not in data:
        return jsonify({'success': False, 'error': 'Missing required fields: name and vote'})
    aadhar_from_data = data['name']  # This is actually aadhar from frontend
    vote = data['vote']
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

    # Check if voter is registered
    voters = []
    if os.path.exists('data/voters.pkl'):
        with open('data/voters.pkl', 'rb') as f:
            voters = pickle.load(f)
    registered_aadhars = {v['aadhar'] for v in voters}
    if aadhar_from_data not in registered_aadhars:
        return jsonify({'success': False, 'error': 'Voter not registered. Please register first.'})

    # Check if voter has already voted
    if check_if_exists(aadhar_from_data):
        return jsonify({'success': False, 'error': 'You have already voted. Multiple votes are not allowed.'})

    # Load managed candidates to validate vote
    managed_candidates = set()
    if os.path.exists('data/candidates.pkl'):
        with open('data/candidates.pkl', 'rb') as f:
            candidates = pickle.load(f)
        for candidate in candidates:
            if isinstance(candidate, dict):
                managed_candidates.add(candidate.get('name', ''))
            elif isinstance(candidate, str):
                managed_candidates.add(candidate)

    # Validate that the vote is for a managed candidate
    if vote not in managed_candidates:
        return jsonify({'success': False, 'error': 'Invalid candidate. Please select a valid candidate.'})

    exist = os.path.isfile("Votes.csv")
    if exist:
        with open("Votes.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            attendance = [aadhar_from_data, vote, date, timestamp]
            writer.writerow(attendance)
    else:
        with open("Votes.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['NAME', 'VOTE', 'DATE', 'TIME'])
            attendance = [aadhar_from_data, vote, date, timestamp]
            writer.writerow(attendance)

    # Get user details from session
    user_details = session.get('user_details', {})
    name = user_details.get('name', 'Unknown')
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

@app.route('/admin_manage')
def admin_manage():
    return render_template('admin_manage.html')

@app.route('/admin_logs')
def admin_logs():
    return render_template('logs.html')

@app.route('/get_logs')
def get_logs():
    logs = []
    if os.path.exists('admin_logs.txt'):
        with open('admin_logs.txt', 'r') as f:
            logs = f.readlines()
    return jsonify({'logs': logs})

@app.route('/admin_vote/<aadhar>')
def admin_vote(aadhar):
    # Load voter details
    voters = []
    if os.path.exists('data/voters.pkl'):
        with open('data/voters.pkl', 'rb') as f:
            voters = pickle.load(f)

    voter = next((v for v in voters if v['aadhar'] == aadhar), None)
    if voter:
        # Set session with voter details
        session['user_details'] = {
            'name': voter['name'],
            'dob': voter['dob'],
            'aadhar': voter['aadhar'],
            'voterid': voter['voterid'],
            'phone': voter['phone']
        }
        session['admin_vote'] = True
        # Redirect to capture face first, then proceed to vote
        return redirect(url_for('capture_face', aadhar=aadhar))
    else:
        flash('Voter not found.')
        return redirect(url_for('admin_manage'))

@app.route('/add_candidate', methods=['POST'])
def add_candidate():
    candidate_name = request.form.get('name', '').strip()
    position = request.form.get('position', '').strip()
    group = request.form.get('group', '').strip()
    icon_file = request.files.get('icon')

    if not candidate_name or not position or not group:
        return jsonify({'success': False, 'error': 'All fields are required'})

    # Load existing candidates
    candidates = []
    if os.path.exists('data/candidates.pkl'):
        with open('data/candidates.pkl', 'rb') as f:
            candidates = pickle.load(f)

    # Check if candidate already exists (by name)
    existing_names = [c.get('name', '') for c in candidates if isinstance(c, dict)]
    if candidate_name in existing_names:
        return jsonify({'success': False, 'error': 'Candidate already exists'})

    # Handle icon upload
    icon_path = None
    if icon_file and icon_file.filename:
        # Create icons directory if it doesn't exist
        icons_dir = 'static/icons'
        if not os.path.exists(icons_dir):
            os.makedirs(icons_dir)
        # Save the icon
        icon_filename = f"{candidate_name.replace(' ', '_')}_{icon_file.filename}"
        icon_path = os.path.join(icons_dir, icon_filename)
        icon_file.save(icon_path)

    # Create candidate object
    candidate = {
        'name': candidate_name,
        'position': position,
        'group': group,
        'icon': icon_path
    }

    # Add new candidate
    candidates.append(candidate)
    with open('data/candidates.pkl', 'wb') as f:
        pickle.dump(candidates, f)

    # Log admin action
    logging.info(f"Admin added candidate: {candidate_name}, Position: {position}, Group: {group}")

    return jsonify({'success': True})

@app.route('/remove_candidate', methods=['POST'])
def remove_candidate():
    data = request.get_json()
    candidate_name = data.get('name', '').strip()
    if not candidate_name:
        return jsonify({'success': False, 'error': 'Candidate name is required'})

    # Load existing candidates
    candidates = []
    if os.path.exists('data/candidates.pkl'):
        with open('data/candidates.pkl', 'rb') as f:
            candidates = pickle.load(f)

    # Find and remove candidate by name
    candidate_to_remove = None
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get('name') == candidate_name:
            candidate_to_remove = candidate
            break

    if candidate_to_remove:
        candidates.remove(candidate_to_remove)
        # Remove icon file if exists
        if candidate_to_remove.get('icon') and os.path.exists(candidate_to_remove['icon']):
            os.remove(candidate_to_remove['icon'])
        with open('data/candidates.pkl', 'wb') as f:
            pickle.dump(candidates, f)
        # Log admin action
        logging.info(f"Admin removed candidate: {candidate_name}")
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Candidate not found'})

@app.route('/get_candidates')
def get_candidates():
    candidates = []
    if os.path.exists('data/candidates.pkl'):
        with open('data/candidates.pkl', 'rb') as f:
            candidates = pickle.load(f)
    return jsonify({'candidates': candidates})

@app.route('/get_voters')
def get_voters():
    voters = []
    if os.path.exists('data/voters.pkl'):
        with open('data/voters.pkl', 'rb') as f:
            voters = pickle.load(f)

    # Check voting status for each voter
    for voter in voters:
        voter['has_voted'] = check_if_exists(voter['aadhar'])

    return jsonify({'voters': voters})

@app.route('/add_voter', methods=['POST'])
def add_voter():
    data = request.get_json()
    name = data.get('name', '').strip()
    dob = data.get('dob', '').strip()
    aadhar = data.get('aadhar', '').strip()
    voterid = data.get('voterid', '').strip()
    phone = data.get('phone', '').strip()

    if not name or not dob or not aadhar or not voterid or not phone:
        return jsonify({'success': False, 'error': 'All fields are required'})

    # Validate Aadhar: exactly 12 digits, numeric only
    if not aadhar.isdigit() or len(aadhar) != 12:
        return jsonify({'success': False, 'error': 'Aadhar must be exactly 12 digits and contain only numbers'})

    # Validate DOB
    try:
        dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
        current_date = datetime.now().date()
        age = current_date.year - dob_date.year - ((current_date.month, current_date.day) < (dob_date.month, dob_date.day))
        if age < 18:
            return jsonify({'success': False, 'error': 'Voter must be at least 18 years old'})
        if dob_date > current_date:
            return jsonify({'success': False, 'error': 'Date of Birth cannot be in the future'})
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid Date of Birth format'})

    # Validate Voter ID: alphanumeric, exactly 10 characters
    if not voterid.isalnum() or len(voterid) != 10:
        return jsonify({'success': False, 'error': 'Voter ID must be exactly 10 alphanumeric characters'})

    # Validate Phone: exactly 10 digits
    if not phone.isdigit() or len(phone) != 10:
        return jsonify({'success': False, 'error': 'Phone number must be exactly 10 digits'})

    # Load existing voters
    voters = []
    if os.path.exists('data/voters.pkl'):
        with open('data/voters.pkl', 'rb') as f:
            voters = pickle.load(f)

    # Check uniqueness of Aadhar
    if any(v['aadhar'] == aadhar for v in voters):
        return jsonify({'success': False, 'error': 'Aadhar number already exists'})

    # Check uniqueness of Voter ID
    if any(v['voterid'] == voterid for v in voters):
        return jsonify({'success': False, 'error': 'Voter ID already exists'})

    # Add new voter
    voters.append({
        'name': name,
        'dob': dob,
        'aadhar': aadhar,
        'voterid': voterid,
        'phone': phone
    })
    with open('data/voters.pkl', 'wb') as f:
        pickle.dump(voters, f)

    # Log admin action
    logging.info(f"Admin added voter: {name}, Aadhar: {aadhar}, VoterID: {voterid}")

    return jsonify({'success': True})

@app.route('/edit_voter', methods=['POST'])
def edit_voter():
    data = request.get_json()
    old_aadhar = data.get('old_aadhar', '').strip()
    name = data.get('name', '').strip()
    dob = data.get('dob', '').strip()
    aadhar = data.get('aadhar', '').strip()
    voterid = data.get('voterid', '').strip()
    phone = data.get('phone', '').strip()

    if not name or not dob or not aadhar or not voterid or not phone:
        return jsonify({'success': False, 'error': 'All fields are required'})

    # Validate Aadhar: exactly 12 digits, numeric only
    if not aadhar.isdigit() or len(aadhar) != 12:
        return jsonify({'success': False, 'error': 'Aadhar must be exactly 12 digits and contain only numbers'})

    # Validate DOB
    try:
        dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
        current_date = datetime.now().date()
        age = current_date.year - dob_date.year - ((current_date.month, current_date.day) < (dob_date.month, dob_date.day))
        if age < 18:
            return jsonify({'success': False, 'error': 'Voter must be at least 18 years old'})
        if dob_date > current_date:
            return jsonify({'success': False, 'error': 'Date of Birth cannot be in the future'})
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid Date of Birth format'})

    # Validate Voter ID: alphanumeric, exactly 10 characters
    if not voterid.isalnum() or len(voterid) != 10:
        return jsonify({'success': False, 'error': 'Voter ID must be exactly 10 alphanumeric characters'})

    # Validate Phone: exactly 10 digits
    if not phone.isdigit() or len(phone) != 10:
        return jsonify({'success': False, 'error': 'Phone number must be exactly 10 digits'})

    # Load existing voters
    if not os.path.exists('data/voters.pkl'):
        return jsonify({'success': False, 'error': 'No voter data found'})

    with open('data/voters.pkl', 'rb') as f:
        voters = pickle.load(f)

    # Find the voter to edit
    voter_index = None
    for i, v in enumerate(voters):
        if v['aadhar'] == old_aadhar:
            voter_index = i
            break

    if voter_index is None:
        return jsonify({'success': False, 'error': 'Voter not found'})

    # Check uniqueness of Aadhar (excluding current)
    if aadhar != old_aadhar and any(v['aadhar'] == aadhar for v in voters):
        return jsonify({'success': False, 'error': 'Aadhar number already exists'})

    # Check uniqueness of Voter ID (excluding current)
    if voterid != voters[voter_index]['voterid'] and any(v['voterid'] == voterid for v in voters):
        return jsonify({'success': False, 'error': 'Voter ID already exists'})

    # Update voter
    voters[voter_index] = {
        'name': name,
        'dob': dob,
        'aadhar': aadhar,
        'voterid': voterid,
        'phone': phone
    }
    with open('data/voters.pkl', 'wb') as f:
        pickle.dump(voters, f)

    # Log admin action
    logging.info(f"Admin edited voter: Old Aadhar: {old_aadhar}, New Aadhar: {aadhar}, Name: {name}, VoterID: {voterid}")

    # If Aadhar changed, update face data
    if aadhar != old_aadhar:
        if os.path.exists('data/names.pkl'):
            with open('data/names.pkl', 'rb') as f:
                names = pickle.load(f)
            # Replace all instances of old_aadhar with new aadhar
            names = [aadhar if name == old_aadhar else name for name in names]
            with open('data/names.pkl', 'wb') as f:
                pickle.dump(names, f)
        load_face_data()  # Reload KNN model

    return jsonify({'success': True})

@app.route('/remove_voter', methods=['POST'])
def remove_voter():
    data = request.get_json()
    aadhar = data.get('aadhar', '').strip()
    if not aadhar:
        return jsonify({'success': False, 'error': 'Aadhar is required'})

    # Remove from voters.pkl
    if os.path.exists('data/voters.pkl'):
        with open('data/voters.pkl', 'rb') as f:
            voters = pickle.load(f)
        voters = [v for v in voters if v['aadhar'] != aadhar]
        with open('data/voters.pkl', 'wb') as f:
            pickle.dump(voters, f)

    if not os.path.exists('data/names.pkl'):
        return jsonify({'success': False, 'error': 'No voter data found'})

    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)

    if aadhar not in names:
        return jsonify({'success': False, 'error': 'Voter not found'})

    # Remove all instances of this aadhar (51 samples per person)
    indices_to_remove = [i for i, name in enumerate(names) if name == aadhar]
    names = [name for i, name in enumerate(names) if i not in indices_to_remove]

    # Update names.pkl
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

    # Update voterids.pkl if exists
    if os.path.exists('data/voterids.pkl'):
        with open('data/voterids.pkl', 'rb') as f:
            voterids = pickle.load(f)
        voterids = [vid for i, vid in enumerate(voterids) if i not in indices_to_remove]
        with open('data/voterids.pkl', 'wb') as f:
            pickle.dump(voterids, f)

    # Update faces_data.pkl
    if os.path.exists('data/faces_data.pkl'):
        with open('data/faces_data.pkl', 'rb') as f:
            faces_data = pickle.load(f)
        faces_data = np.delete(faces_data, indices_to_remove, axis=0)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)

    # Log admin action
    logging.info(f"Admin removed voter: Aadhar: {aadhar}")

    load_face_data()  # Reload KNN model
    return jsonify({'success': True})

@app.route('/reset_data', methods=['POST'])
def reset_data():
    try:
        # Delete Votes.csv
        if os.path.exists('Votes.csv'):
            os.remove('Votes.csv')

        # Clear data/candidates.pkl
        with open('data/candidates.pkl', 'wb') as f:
            pickle.dump([], f)

        # Clear data/voters.pkl
        with open('data/voters.pkl', 'wb') as f:
            pickle.dump([], f)

        # Clear data/names.pkl
        with open('data/names.pkl', 'wb') as f:
            pickle.dump([], f)

        # Clear data/faces_data.pkl
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(np.array([]), f)

        # Clear data/voterids.pkl
        with open('data/voterids.pkl', 'wb') as f:
            pickle.dump([], f)

        # Remove all files in static/icons/ directory
        icons_dir = 'static/icons'
        if os.path.exists(icons_dir):
            for filename in os.listdir(icons_dir):
                file_path = os.path.join(icons_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Reload face data (set knn to None)
        load_face_data()

        # Log the reset action
        logging.info("Admin reset all data")

        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Error resetting data: {e}")
        return jsonify({'success': False, 'error': str(e)})



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
