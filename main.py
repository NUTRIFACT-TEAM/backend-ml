"""
Nutrifact API - Flask-based API for Nutrition Information Extraction and Health Grade Prediction

This API extracts nutrition information from an image of a product's nutrition label and predicts a health grade using a pre-trained ML model.

"""
import os
import json
import re  # For regex processing
from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
import tensorflow as tf
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set the Tesseract OCR executable path
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Bucket name (if using Google Cloud Storage)
bucket_name = 'bucket-nutrifact'  # Replace with your bucket name

# Load TensorFlow model for health grade prediction
model = tf.keras.models.load_model('nutrition_grade_model.h5')

# Function to load LabelEncoder from a JSON file
def load_label_encoder_from_json(file_path):
    """
    Loads LabelEncoder class mappings from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        LabelEncoder: An initialized LabelEncoder with classes loaded.
    """
    with open(file_path, 'r') as f:
        label_mapping = json.load(f)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_mapping["classes"])
    return label_encoder

# Function to load StandardScaler parameters from a JSON file
def load_scaler_from_json(file_path):
    """
    Loads StandardScaler parameters from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        StandardScaler: A scaler object with mean and scale set.
    """
    with open(file_path, 'r') as f:
        scaler_params = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params["mean"])
    scaler.scale_ = np.array(scaler_params["scale"])
    return scaler

# Load LabelEncoder and Scaler
le = load_label_encoder_from_json('label_encoder.json')
scaler = load_scaler_from_json('scaler.json')

# Function to perform OCR on an image and extract nutrition text
def extract_nutrition_from_image(file_storage):
    """
    Extracts text from an image file using OCR.

    Args:
        file_storage: File object from HTTP request.

    Returns:
        str: Extracted text.

    Raises:
        ValueError: If the image decoding fails.
    """
    # Read bytes from uploaded file
    in_memory_file = file_storage.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode bytes to image
    if img is None:
        raise ValueError("Image decoding failed")
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding for better OCR accuracy
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Extract text using Tesseract
    ocr_text = pytesseract.image_to_string(thresh, lang='eng')
    return ocr_text

# Function to extract sugar and fat values using regex
def extract_nutrition_info(ocr_text):
    """
    Extracts sugar and fat values from OCR text using regex.

    Args:
        ocr_text (str): Text extracted from OCR.

    Returns:
        tuple: (sugar, fat) extracted values.
    """
    print("Processed OCR Text: ", ocr_text)

    lines = ocr_text.split('\n')
    sugar = None
    fat = None

    for line in lines:
        if 'sugar' in line.lower() or 'gula' in line.lower():
            sugar_match = re.search(r'(\d+(\.\d+)?)\s*(g|%)', line)
            if sugar_match:
                sugar = sugar_match.group(1)

        elif 'fat' in line.lower() or 'lemak' in line.lower():
            fat_match = re.search(r'(\d+(\.\d+)?)\s*(g|%)', line)
            if fat_match:
                fat = fat_match.group(1)

    return sugar, fat

# Function to create feature vector for the ML model
def create_boundary_features(sugar_100g, fat_100g):
    """
    Generates feature vector based on sugar and fat values per 100g.

    Args:
        sugar_100g (float): Sugar per 100g.
        fat_100g (float): Fat per 100g.

    Returns:
        numpy.ndarray: Feature vector with boundaries.
    """
    features = np.zeros(6)
    features[0] = sugar_100g
    features[1] = fat_100g
    features[2] = abs(sugar_100g - 1)
    features[3] = abs(sugar_100g - 5)
    features[4] = abs(sugar_100g - 10)
    features[5] = abs(fat_100g - 2.8)
    return features

# Function to predict health grade using ML model
def predict_grade_with_model(sugar, fat, serving_size):
    """
    Predicts health grade using sugar and fat values.

    Args:
        sugar (str): Extracted sugar value.
        fat (str): Extracted fat value.
        serving_size (int): Serving size for normalization.

    Returns:
        str: Predicted health grade.
    """
    sugar_100g = (float(sugar) / float(serving_size)) * 100
    fat_100g = (float(fat) / float(serving_size)) * 100

    features = create_boundary_features(sugar_100g, fat_100g).reshape(1, -1)
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)
    grade = le.inverse_transform([np.argmax(pred)])[0]
    return grade

# Endpoint to process image and predict health grade
@app.route('/predict', methods=['POST'])
def predict():
    """
    Processes uploaded image to extract nutrition data and predict health grade.

    Returns:
        JSON response with prediction or error message.
    """
    try:
        if 'imageNutri' not in request.files or 'barcodeId' not in request.form:
            return jsonify({'error': 'No image or barcodeId provided'}), 400

        file = request.files['imageNutri']
        barcode_id = request.form['barcodeId']

        ocr_text = extract_nutrition_from_image(file)
        sugar, fat = extract_nutrition_info(ocr_text)

        if not sugar or not fat:
            return jsonify({'error': 'Failed to extract nutrition information'}), 400

        health_grade = predict_grade_with_model(sugar, fat, serving_size=100)

        return jsonify({
            'barcodeId': barcode_id,
            'fat': fat,
            'sugar': sugar,
            'healthGrade': health_grade
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Root endpoint for health check
@app.route('/', methods=['GET'])
def home():
    """
    Root endpoint for checking API status.

    Returns:
        JSON response with welcome message.
    """
    return jsonify({'message': 'Nutrifact-backend-ML is running'}), 200

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
