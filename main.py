from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
from google.cloud import storage, firestore
import tensorflow as tf
import cv2
import numpy as np
import os
import json
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re  # Untuk regex

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app)  # Enable CORS

# Konfigurasi Google Cloud Storage dan Firestore
# storage_client = storage.Client()
# firestore_client = firestore.Client()
bucket_name = 'bucket-nutrifact'  # Ganti dengan nama bucket Anda

# Load model untuk klasifikasi grade
model = tf.keras.models.load_model('nutrition_grade_model.h5')


# Fungsi untuk memuat LabelEncoder dari file JSON
def load_label_encoder_from_json(file_path):
    with open(file_path, 'r') as f:
        label_mapping = json.load(f)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_mapping["classes"])
    return label_encoder


# Fungsi untuk memuat Scaler dari file JSON
def load_scaler_from_json(file_path):
    with open(file_path, 'r') as f:
        scaler_params = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params["mean"])
    scaler.scale_ = np.array(scaler_params["scale"])
    return scaler


# Memuat LabelEncoder dan Scaler
le = load_label_encoder_from_json('label_encoder.json')
scaler = load_scaler_from_json('scaler.json')


# Fungsi untuk mengupload file ke Google Cloud Storage
# def upload_to_gcs(file):
#     bucket = storage_client.get_bucket(bucket_name)
#     filename = secure_filename(file.filename)
#     blob = bucket.blob(filename)
#     blob.upload_from_file(file)
#     return f'https://storage.googleapis.com/{bucket_name}/{filename}'


# Fungsi untuk mengunduh file dari Google Cloud Storage
# def download_from_gcs(image_url):
#     filename = image_url.split("/")[-1]
#     local_path = os.path.join("downloaded_images", filename)
#     if not os.path.exists('downloaded_images'):
#         os.makedirs('downloaded_images')
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(filename)
#     blob.download_to_filename(local_path)
#     return local_path


# Fungsi untuk ekstraksi teks dari gambar menggunakan Tesseract OCR
def extract_nutrition_from_image(file_storage): # file_storage menerima objek file langsung dari permintaan HTTP
    in_memory_file = file_storage.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # byte jadi array
    if img is None:
        raise ValueError("Image decoding failed")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ocr_text = pytesseract.image_to_string(thresh, lang='eng')
    return ocr_text


# Fungsi untuk mengekstraksi informasi gula dan lemak dari teks OCR
def extract_nutrition_info(ocr_text):
    print("OCR Text setelah Proses: ", ocr_text)

    lines = ocr_text.split('\n')
    sugar = None
    fat = None

    for line in lines:
        # Ekstraksi gula menggunakan regex
        if 'sugar' in line.lower() or 'gula' in line.lower():
            sugar_match = re.search(r'(\d+(\.\d+)?)\s*(g|%)', line)  # Mencari angka dengan unit g atau %
            if sugar_match:
                sugar = sugar_match.group(1)

        # Ekstraksi lemak menggunakan regex
        elif 'fat' in line.lower() or 'lemak' in line.lower():
            fat_match = re.search(r'(\d+(\.\d+)?)\s*(g|%)', line)  # Mencari angka dengan unit g atau %
            if fat_match:
                fat = fat_match.group(1)

    return sugar, fat


# Fungsi untuk membuat fitur berdasarkan batasan (boundary) gula dan lemak
def create_boundary_features(sugar_100g, fat_100g):
    features = np.zeros(6)
    features[0] = sugar_100g
    features[1] = fat_100g
    features[2] = abs(sugar_100g - 1)
    features[3] = abs(sugar_100g - 5)
    features[4] = abs(sugar_100g - 10)
    features[5] = abs(fat_100g - 2.8)
    return features


# Fungsi untuk prediksi grade kesehatan menggunakan model ML
def predict_grade_with_model(sugar, fat, serving_size):
    # Normalisasi ke per 100g
    sugar_100g = (float(sugar) / float(serving_size)) * 100
    fat_100g = (float(fat) / float(serving_size)) * 100

    # Buat fitur
    features = create_boundary_features(sugar_100g, fat_100g).reshape(1, -1)

    # Normalisasi fitur
    features_scaled = scaler.transform(features)

    # Prediksi menggunakan model
    pred = model.predict(features_scaled)

    # Mengambil kelas dengan probabilitas tertinggi
    grade = le.inverse_transform([np.argmax(pred)])[0]

    return grade


# Endpoint untuk menerima gambar dan memprosesnya
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pastikan gambar dan barcode diupload
        if 'imageNutri' not in request.files or 'barcodeId' not in request.form:
            return jsonify({'error': 'No image or barcodeId provided'}), 400

        file = request.files['imageNutri']
        barcode_id = request.form['barcodeId']

        # # Upload gambar ke Google Cloud Storage
        # image_url = upload_to_gcs(file)
        # 
        # # Unduh gambar dari URL Google Cloud Storage
        # img = download_from_gcs(image_url)

        # Ekstraksi teks OCR dari gambar
        ocr_text = extract_nutrition_from_image(file)
        sugar, fat = extract_nutrition_info(ocr_text)

        if not sugar or not fat:
            return jsonify({'error': 'Failed to extract nutrition information from image'}), 400

        # Prediksi grade kesehatan menggunakan model
        health_grade = predict_grade_with_model(sugar, fat, serving_size=100)

        # # Simpan data produk ke Firestore
        # product_data = {
        #     'barcodeId': barcode_id,
        #     'fat': fat,
        #     'sugar': sugar,
        #     'healthGrade': health_grade,
        #     'description': 'description',
        #     'brand': 'Sample Brand',
        #     'variant': 'Original 100g'
        # }
        # firestore_client.collection('products').document(barcode_id).set(product_data)

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