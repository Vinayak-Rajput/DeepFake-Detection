# app.py (Integrate Blockchain Query)
import os
import hashlib
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from utils.predictor import predict_single_image, predict_video_sequence
# --- Import BOTH blockchain functions ---
from blockchain.log_to_blockchain import log_detection_to_chain, query_detection_by_hash

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Ensure upload folder exists
IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS.union(VIDEO_EXTENSIONS)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 100 MB limit

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_hash(filepath):
    """Calculates the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while True:
                data = f.read(65536) # Read in 64k chunks
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()
    except Exception as e:
        print(f"[ERROR] Could not calculate hash for {filepath}: {e}")
        return None # Return None on error

@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles uploads, queries blockchain, predicts if needed, logs, returns results."""
    # Check if the post request has the file part
    if 'media' not in request.files:
        return render_template('index.html', error="No file part selected.")
    file = request.files['media']
    # If the user does not select a file, the browser submits an empty file without a filename.
    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename) # Ensure filename is safe
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the uploaded file
        try:
            file.save(filepath)
            print(f"File saved to: {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to save file {filename}: {e}")
            return render_template('index.html', error=f"Failed to save file: {e}")

        # Initialize variables for results and errors
        label, confidence, media_hash, tx_hash = "N/A", 0.0, "N/A", "N/A"
        error_message, file_type = None, "Unknown"
        existing_record = None # Flag/data for existing blockchain record

        try:
            # 1. Get file hash immediately
            media_hash = get_file_hash(filepath)
            if media_hash is None: # Handle hash calculation error
                raise ValueError("Could not calculate file hash.")
            print(f"File hash: {media_hash}")

            # --- 2. Query Blockchain BEFORE Prediction ---
            existing_record = query_detection_by_hash(media_hash)
            # --- End Query ---

            # Determine file type based on filename extension (needed in both cases)
            file_ext = filename.rsplit('.', 1)[1].lower()
            file_type = "Image" if file_ext in IMAGE_EXTENSIONS else "Video"

            if existing_record:
                # If found, use the existing data and skip prediction/logging
                print("Using existing blockchain record.")
                label = existing_record["label"]
                confidence = existing_record["confidence"]
                # Transaction hash isn't stored in this simple contract, indicate it was loaded
                log_time_str = existing_record.get('log_time_str', existing_record.get('timestamp', 'N/A'))
                tx_hash = f"Record found (Logged: {log_time_str})"

            else:
                # If NOT found, proceed with prediction and logging
                print("No existing record found. Proceeding with prediction...")

                if file_type == "Image":
                    print("Running prediction for Image...")
                    label, confidence = predict_single_image(filepath)
                elif file_type == "Video":
                    print("Running prediction for Video...")
                    label, confidence = predict_video_sequence(filepath)
                else:
                    # This case should ideally not be reached due to allowed_file check
                    raise ValueError("Internal Error: Unexpected file type passed checks.")

                print(f"Prediction result: Type={file_type}, Label={label}, Confidence={confidence:.4f}")

                # Log the NEW result to the blockchain
                if label != "N/A" and label != "Error":
                    print("Logging to blockchain...")
                    tx_hash = log_detection_to_chain(media_hash, label, confidence)
                    # Check the return status from the logger
                    if isinstance(tx_hash, str) and (tx_hash.startswith("[ERROR:") or tx_hash.startswith("[WARN:") or tx_hash.startswith("[SKIPPED:")):
                        print(f"Blockchain log status: {tx_hash}")
                else:
                    print("[WARN] Skipping blockchain logging due to prediction error.")
                    tx_hash = "Skipped (Prediction Error)"

        except ValueError as e: # Catch prediction or hash errors
            error_message = f"Processing Error: {e}"
            print(f"[ERROR] {error_message}")
            label = "Error" # Set label to Error
        except Exception as e: # Catch other potential errors
            error_message = f"An unexpected error occurred: {e}"
            print(f"[ERROR] {error_message}")
            label = "Error"
            # Ensure blockchain tx hash reflects potential init failure
            if tx_hash == "N/A":
                 tx_hash = "Logging Skipped/Failed"

        # 4. Prepare results dictionary for the template
        result_data = {
            "type": file_type,
            "label": label,
            "confidence": confidence,
            "hash": media_hash,
            "tx_hash": tx_hash,
            "uploaded_filename": filename,
            "existing_record": existing_record # Pass the existing record dict (or None)
        }

        # Render the page with results and/or error message
        return render_template('index.html', result=result_data, error=error_message)

    else:
        # Handle disallowed file types
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'unknown'
        return render_template('index.html', error=f"File type '{file_ext}' not allowed.")

# --- Route to serve uploaded files ---
@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serves files from the UPLOAD_FOLDER."""
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
        return "Invalid filename", 400
    try:
        # Use send_from_directory for security and proper handling
        return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename)
    except FileNotFoundError:
        print(f"[WARN] Uploaded file not found: {safe_filename}")
        return "File not found", 404

if __name__ == '__main__':
    print("Starting Flask app... Make sure Ganache is running.")
    # Models are loaded when utils.predictor is imported.
    app.run(debug=True, host='0.0.0.0', port=5000) # host='0.0.0.0' makes it accessible on your network