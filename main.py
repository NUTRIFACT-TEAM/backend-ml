from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
import tensorflow as tf
import cv2
import numpy as np
import os
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re  # Untuk regex

# Menetapkan path ke executable Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app)  # Mengaktifkan Cross-Origin Resource Sharing (CORS)

# Nama bucket Google Cloud Storage (jika digunakan)
bucket_name = 'bucket-nutrifact'  # Ganti dengan nama bucket Anda

# Memuat model TensorFlow untuk klasifikasi grade kesehatan
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

# Memuat LabelEncoder dan Scaler dari file JSON
le = load_label_encoder_from_json('label_encoder.json')
scaler = load_scaler_from_json('scaler.json')

# Fungsi untuk ekstraksi teks dari gambar menggunakan Tesseract OCR
def extract_nutrition_from_image(file_storage):
    """
    Mengambil objek file dari permintaan HTTP, membaca byte-nya,
    mendekode gambar menggunakan OpenCV, dan mengekstrak teks menggunakan OCR.
    
    Args:
        file_storage: Objek file yang diunggah melalui permintaan HTTP.
    
    Returns:
        ocr_text: Teks yang diekstrak dari gambar.
    
    Raises:
        ValueError: Jika proses dekoding gambar gagal.
    """
    # Membaca byte dari file yang diunggah
    in_memory_file = file_storage.read()
    # Mengubah byte menjadi array numpy
    nparr = np.frombuffer(in_memory_file, np.uint8)
    # Mendekode gambar dari array numpy
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # byte menjadi array
    if img is None:
        raise ValueError("Image decoding failed")
    # Mengubah gambar menjadi grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Mengaplikasikan threshold untuk meningkatkan kontras teks
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Mengekstrak teks menggunakan Tesseract OCR
    ocr_text = pytesseract.image_to_string(thresh, lang='eng')
    return ocr_text

# Fungsi untuk mengekstraksi informasi gula dan lemak dari teks OCR
def extract_nutrition_info(ocr_text):
    """
    Mengekstraksi nilai gula dan lemak dari teks yang diperoleh dari OCR menggunakan regex.
    
    Args:
        ocr_text: Teks yang diekstrak dari gambar.
    
    Returns:
        sugar: Nilai gula yang diekstrak (string) atau None jika tidak ditemukan.
        fat: Nilai lemak yang diekstrak (string) atau None jika tidak ditemukan.
    """
    print("OCR Text setelah Proses: ", ocr_text)

    # Memisahkan teks menjadi baris-baris
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
    """
    Membuat vektor fitur berdasarkan nilai gula dan lemak per 100g serta jarak dari batas tertentu.
    
    Args:
        sugar_100g: Nilai gula per 100g.
        fat_100g: Nilai lemak per 100g.
    
    Returns:
        features: Array numpy dengan 6 elemen fitur.
    """
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
    """
    Memprediksi grade kesehatan berdasarkan kandungan gula dan lemak menggunakan model ML.
    
    Args:
        sugar: Nilai gula yang diekstrak dari OCR.
        fat: Nilai lemak yang diekstrak dari OCR.
        serving_size: Ukuran porsi (default 100).
    
    Returns:
        grade: Grade kesehatan yang diprediksi (string).
    """
    # Normalisasi ke per 100g
    sugar_100g = (float(sugar) / float(serving_size)) * 100
    fat_100g = (float(fat) / float(serving_size)) * 100

    # Membuat fitur
    features = create_boundary_features(sugar_100g, fat_100g).reshape(1, -1)

    # Normalisasi fitur menggunakan scaler yang telah dimuat
    features_scaled = scaler.transform(features)

    # Melakukan prediksi menggunakan model
    pred = model.predict(features_scaled)

    # Mengambil kelas dengan probabilitas tertinggi
    grade = le.inverse_transform([np.argmax(pred)])[0]

    return grade

# Endpoint untuk menerima gambar dan memprosesnya
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk menerima permintaan POST dengan gambar informasi nutrisi dan ID barcode.
    Memproses gambar untuk mengekstrak informasi nutrisi dan memprediksi grade kesehatan.
    
    Returns:
        JSON response dengan barcodeId, fat, sugar, dan healthGrade jika sukses.
        JSON response dengan error message jika gagal.
    """
    try:
        # Memastikan bahwa 'imageNutri' dan 'barcodeId' disertakan dalam permintaan
        if 'imageNutri' not in request.files or 'barcodeId' not in request.form:
            return jsonify({'error': 'No image or barcodeId provided'}), 400

        # Mengambil file gambar dan ID barcode dari permintaan
        file = request.files['imageNutri']
        barcode_id = request.form['barcodeId']

        # Ekstraksi teks OCR dari gambar
        ocr_text = extract_nutrition_from_image(file)
        sugar, fat = extract_nutrition_info(ocr_text)

        # Memeriksa apakah gula dan lemak berhasil diekstrak
        if not sugar or not fat:
            return jsonify({'error': 'Failed to extract nutrition information from image'}), 400

        # Prediksi grade kesehatan menggunakan model ML
        health_grade = predict_grade_with_model(sugar, fat, serving_size=100)

        # Mengembalikan response dengan hasil prediksi
        return jsonify({
            'barcodeId': barcode_id,
            'fat': fat,
            'sugar': sugar,
            'healthGrade': health_grade
        }), 200

    except Exception as e:
        # Menangani error dan mengembalikan response dengan pesan error
        return jsonify({'error': str(e)}), 500

# Endpoint root untuk memeriksa apakah API berjalan dengan benar
@app.route('/', methods=['GET'])
def home():
    """
    Endpoint root yang mengembalikan pesan selamat datang.
    
    Returns:
        JSON response dengan pesan selamat datang.
    """
    return jsonify({'message': 'Welcome to Nutrifact API'}), 200

# Menjalankan aplikasi Flask jika file ini dijalankan langsung
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
