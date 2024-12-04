from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
from google.cloud import storage, firestore
import tensorflow as tf
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app)  # Enable CORS

# Konfigurasi Google Cloud Storage dan Firestore
storage_client = storage.Client()
firestore_client = firestore.Client()
bucket_name = 'bucket-nutrifact'  # Ganti dengan nama bucket Anda

# Load model untuk klasifikasi grade
model = tf.keras.models.load_model('nutrition_grade_model.h5')

# Fungsi untuk mengupload file ke Google Cloud Storage
def upload_to_gcs(file):
    bucket = storage_client.get_bucket(bucket_name)
    filename = secure_filename(file.filename)
    blob = bucket.blob(filename)
    blob.upload_from_file(file)
    return f'https://storage.googleapis.com/{bucket_name}/{filename}'

# Fungsi untuk mengunduh file dari Google Cloud Storage
def download_from_gcs(image_url):
    # Extract filename from URL or other logic if needed
    filename = image_url.split("/")[-1]

    # Tentukan path tempat file akan disimpan di local system
    local_path = os.path.join("downloaded_images", filename)

    # Create the folder if it doesn't exist
    if not os.path.exists('downloaded_images'):
        os.makedirs('downloaded_images')

    # Mengunduh file dari GCS
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.download_to_filename(local_path)

    return local_path

# Fungsi untuk ekstraksi teks dari gambar menggunakan Tesseract OCR
def extract_nutrition_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    ocr_text = pytesseract.image_to_string(thresh, lang='eng')
    return ocr_text

# Fungsi untuk mengekstraksi informasi gula dan lemak dari teks OCR
def extract_nutrition_info(ocr_text):
    lines = ocr_text.split('\n')
    sugar = None
    fat = None

    for line in lines:
        if 'sugar' in line.lower() or 'gula' in line.lower():
            sugar = ''.join([char for char in line if char.isdigit() or char == '.'])
        elif 'fat' in line.lower() or 'lemak' in line.lower():
            fat = ''.join([char for char in line if char.isdigit() or char == '.'])

    return sugar, fat

# Fungsi untuk menghitung Health Grade berdasarkan gula dan lemak
def calculate_health_grade(sugar, fat):
    sugar_100g = float(sugar) if sugar else 0
    fat_100g = float(fat) if fat else 0

    grade = 'A'
    if sugar_100g > 5 or fat_100g > 2.8:
        grade = 'D'
    elif sugar_100g > 3 or fat_100g > 1.5:
        grade = 'C'
    elif sugar_100g > 1 or fat_100g > 0.7:
        grade = 'B'

    return grade

# Endpoint untuk menerima gambar dan memprosesnya
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pastikan gambar dan barcode diupload
        if 'image' not in request.files or 'barcodeId' not in request.form:
            return jsonify({'error': 'No image or barcodeId provided'}), 400

        file = request.files['image']
        barcode_id = request.form['barcodeId']

        # Upload gambar ke Google Cloud Storage
        image_url = upload_to_gcs(file)

        # Unduh gambar dari URL Google Cloud Storage
        img = download_from_gcs(image_url)

        # Ekstraksi teks OCR dari gambar
        ocr_text = extract_nutrition_from_image(img)  # Ubah parameter menjadi img
        sugar, fat = extract_nutrition_info(ocr_text)

        if not sugar or not fat:
            return jsonify({'error': 'Failed to extract nutrition information from image'}), 400

        # Hitung grade kesehatan produk
        health_grade = calculate_health_grade(sugar, fat)

        # Simpan data produk ke Firestore
        product_data = {
            'barcodeId': barcode_id,
            'fat': fat,
            'sugar': sugar,
            'healthGrade': health_grade,
            'description': 'description',
            'brand': 'Sample Brand',
            'variant': 'Original 100g'
        }
        firestore_client.collection('products').document(barcode_id).set(product_data)

        # Kembalikan response dengan hasil prediksi
        return jsonify({
            'barcodeId': barcode_id,
            'fat': fat,
            'sugar': sugar,
            'healthGrade': health_grade
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Menambahkan route untuk menangani permintaan GET pada endpoint root ('/')
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to Nutrifact API'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
