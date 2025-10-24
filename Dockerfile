# 1. Base Image: Use a stable Python version (adjust if needed)
FROM python:3.10-slim

# 2. Set Environment Variables
ENV PYTHONUNBUFFERED=1 \
    # Set locale to prevent potential encoding issues with OpenCV/Flask
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Indicate non-interactive frontend for apt-get
    DEBIAN_FRONTEND=noninteractive

# 3. Install System Dependencies
# Required for OpenCV (headless), basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # FFmpeg is needed by OpenCV for video processing
    ffmpeg \
    # Optional: git if needed during pip install, build tools if needed
    # git build-essential \
    && \
    # Clean up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. Set Working Directory
WORKDIR /app

# 5. Copy requirements and install Python packages
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
# Install dependencies (use --no-cache-dir to reduce image size)
# Use system site packages if preferred for some dependencies (like numpy)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code and Models
# Copy everything else from the host (respecting .dockerignore)
COPY . .

# 7. Expose Port
# Expose the port Flask runs on
EXPOSE 5000

# 8. Define Run Command
# Command to run the Flask application when the container starts
# Use Gunicorn in production instead of Flask's dev server
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
# For development/testing, Flask's server is okay:
CMD ["python", "app.py"]