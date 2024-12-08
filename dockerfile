# Use Python 3.11 as the base image
FROM python:3.11-slim

# Install required system dependencies
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application source code into the container
COPY . /app/

# Set environment variable to avoid output buffering issues
ENV PYTHONUNBUFFERED=1

# Set the environment variable for the application port
ENV PORT=8080

# Run the Flask application
CMD ["python", "app.py"]