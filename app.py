# app.py
import os
import hashlib
import datetime
from flask import (
    Flask, render_template, request, redirect, url_for, send_from_directory,
    jsonify, flash, session, g
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, login_required, current_user
)
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
import traceback # For detailed error logging

# --- App Setup ---
app = Flask(__name__)
# IMPORTANT: Change this secret key!
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_very_secret_key_change_this_later_923847')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 100 MB limit

# --- Database Setup (SQLite) ---
db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'site.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# --- Login Manager Setup ---
login_manager = LoginManager(app)
login_manager.login_view = 'main_page' # Redirect to main page if login required
login_manager.login_message = 'You must be logged in to access this page.'
login_manager.login_message_category = 'error' # Use 'error' category for flash messages

# --- Project Imports (Load after app setup) ---
# We import here to avoid circular dependencies and ensure app is created
try:
    from utils.predictor import predict_single_image, predict_video_sequence
    from blockchain.log_to_blockchain import log_detection_to_chain, query_detection_by_hash, get_all_detections
    print("[INFO] All utility and blockchain modules loaded.")
except ImportError as e:
    print(f"[FATAL ERROR] Failed to import modules: {e}")
    print("Please ensure all utility and blockchain files are in place.")
    # In a real app, you might exit or have this handled more gracefully
    
# --- Configuration ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS.union(VIDEO_EXTENSIONS)

# === User Models ===
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    # Use db.session.get for primary key lookup
    return db.session.get(User, int(user_id))

# === Helper Functions ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_hash(filepath):
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while True:
                data = f.read(65536)
                if not data: break
                sha256.update(data)
        return sha256.hexdigest()
    except Exception as e:
        print(f"[ERROR] Could not calculate hash: {e}")
        return None

# === Authentication Routes ===
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    user = User.query.filter_by(username=username).first()
    
    if user and user.check_password(password):
        login_user(user)
        print(f"[INFO] User '{username}' logged in successfully.")
        flash('Logged in successfully!', 'success')
        # Redirect to the upload tab after login
        return redirect(url_for('main_page', _anchor='upload'))
    else:
        print(f"[WARN] Failed login attempt for user '{username}'.")
        flash('Login failed. Check username and password.', 'error')
        return redirect(url_for('main_page'))

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    password = request.form.get('password')
    
    if not username or not password:
        flash('Username and password are required.', 'error')
        return redirect(url_for('main_page'))
    
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        print(f"[WARN] Registration failed: Username '{username}' already exists.")
        flash('Username already exists. Please choose another or log in.', 'error')
        return redirect(url_for('main_page'))
    
    try:
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        print(f"[INFO] New user '{username}' registered.")
        flash('Registration successful! Please log in.', 'success')
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Database error during registration: {e}")
        flash('An error occurred during registration. Please try again.', 'error')
        
    return redirect(url_for('main_page'))

@app.route('/logout')
@login_required
def logout():
    print(f"[INFO] User '{current_user.username}' logged out.")
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main_page'))

# === Main Application Routes ===
@app.route('/')
def main_page():
    """
    Renders the main single-page application.
    Checks for session data to display the last result if available.
    """
    # Retrieve last result from session and pass it to template
    result_data = session.get('last_result', None)
    error_data = session.get('last_error', None)
    # Clear them so they don't show up again on refresh
    if 'last_result' in session: session.pop('last_result')
    if 'last_error' in session: session.pop('last_error')
    
    # Render the main template, passing any previous result/error
    return render_template('index.html', result=result_data, error=error_data)

@app.route('/history')
@login_required
def history():
    """Renders the main page, pre-filled with history data."""
    print(f"[INFO] User '{current_user.username}' requesting history.")
    history_data = []
    error_msg = None
    try:
        logs = get_all_detections() # Get all logs
        history_data = logs
    except Exception as e:
        print(f"[ERROR] Could not fetch blockchain history: {e}")
        error_msg = f"Could not fetch blockchain history: {e}"
    
    # --- FIX: Pass default 'result' and 'error' values ---
    # The template needs 'result' and 'error' to be defined, even if they are None,
    # because it renders all tabs (even hidden ones).
    return render_template(
        'index.html', 
        history_logs=history_data, 
        history_error=error_msg, 
        initial_tab='history',
        result=None,  # <-- ADDED THIS LINE
        error=None    # <-- AND ADDED THIS LINE
    )
    # --- END FIX ---
@app.route('/upload_media', methods=['POST'])
@login_required
def upload_media():
    """
    Handles the file upload, prediction, and logging.
    This is an API-style endpoint. It saves the result to the session
    and redirects back to the main page to display it.
    """
    if 'media' not in request.files:
        flash("No file part selected.", "error")
        return redirect(url_for('main_page', _anchor='upload'))
    file = request.files['media']
    if file.filename == '':
        flash("No file selected.", "error")
        return redirect(url_for('main_page', _anchor='upload'))
    
    # Get options from form checkboxes
    use_blockchain = request.form.get('use_blockchain') == 'on' # Checkbox value is 'on'
    use_xai = request.form.get('use_xai') == 'on' # Checkbox value is 'on'
    print(f"[INFO] Upload request: blockchain={use_blockchain}, xai={use_xai}")


    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            print(f"[INFO] File saved to: {filepath}")
        except Exception as e:
            flash(f"Failed to save file: {e}", "error")
            return redirect(url_for('main_page', _anchor='upload'))

        label, confidence, media_hash, tx_hash = "N/A", 0.0, "N/A", "N/A"
        error_message, file_type = None, "Unknown"
        existing_record = None
        explanation_file = None

        try:
            media_hash = get_file_hash(filepath)
            if media_hash is None: raise ValueError("Could not calculate file hash.")
            print(f"[INFO] File hash: {media_hash}")

            file_ext = filename.rsplit('.', 1)[1].lower()
            file_type = "Image" if file_ext in IMAGE_EXTENSIONS else "Video"

            # 1. Check Blockchain (if user opted-in)
            if use_blockchain:
                existing_record = query_detection_by_hash(media_hash)
            
            if existing_record:
                print("[INFO] Using existing blockchain record.")
                label = existing_record["label"]
                confidence = existing_record["confidence"]
                explanation_file = existing_record.get("explanation_file")
                log_time = datetime.datetime.fromtimestamp(existing_record.get('timestamp', 0))
                existing_record['log_time_str'] = log_time.strftime('%Y-%m-%d %H:%M:%S')
                tx_hash = f"Record found (Logged: {existing_record['log_time_str']})"
            else:
                # 2. Run Prediction
                print("[INFO] No existing record. Proceeding with prediction...")
                if file_type == "Image":
                    label, confidence, explanation_file = predict_single_image(filepath, generate_xai=use_xai)
                elif file_type == "Video":
                    label, confidence, explanation_file = predict_video_sequence(filepath) # XAI not for video
                
                print(f"[INFO] Prediction: {label}, Conf: {confidence:.4f}")
                if explanation_file: print(f"[INFO] LIME explanation generated: {explanation_file}")

                # 3. Log to Blockchain (if user opted-in)
                if use_blockchain:
                    if label != "N/A" and label != "Error":
                        print("[INFO] Logging to blockchain...")
                        tx_hash = log_detection_to_chain(media_hash, label, confidence, explanation_file)
                    else: tx_hash = "Skipped (Prediction Error)"
                else:
                    tx_hash = "Skipped (User Opt-out)"

        except Exception as e:
            print(f"[ERROR] Unexpected error in upload_media: {e}")
            traceback.print_exc()
            error_message = f"An unexpected error occurred during processing: {e}"
            label = "Error"

        # 4. Store result in session and redirect
        result_data = {
            "type": file_type,
            "label": label,
            "confidence": confidence,
            "hash": media_hash,
            "tx_hash": tx_hash,
            "uploaded_filename": filename,
            "existing_record": existing_record,
            "explanation_file": explanation_file
        }
        session['last_result'] = result_data
        session['last_error'] = error_message # Store error too
        
        # Redirect back to the main page, which will render with the result
        return redirect(url_for('main_page', _anchor='result'))

    else:
        flash("File type not allowed.", "error")
        return redirect(url_for('main_page', _anchor='upload'))

@app.route('/uploads/<path:filename>')
@login_required
def serve_upload(filename):
    """Serves files from the UPLOAD_FOLDER (previews, explanations)."""
    safe_filename = secure_filename(filename)
    if safe_filename != filename: return "Invalid filename", 400
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename, as_attachment=False)
    except FileNotFoundError:
        return "File not found", 404

# --- Create Database and Run ---
if __name__ == '__main__':
    with app.app_context():
        print("Initializing database...")
        db.create_all()
        print("Database initialized.")
    print("Starting Flask app... Make sure Ganache is running.")
    app.run(debug=True, host='0.0.0.0', port=5000)