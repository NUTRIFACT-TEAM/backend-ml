# Nutrifact Backend Model Machine Learning

## Daftar Isi

1. [Team C242-PS378](#C242-PS378---cc)
2. [Technology](#Technology)
3. [Requirement](#Requirement)
4. [Installation Steps](#InstallationSteps)

## C242-PS378 - CC

| Bangkit ID    | Nama                     | Learning Path       | Universitas                      |
|---------------|--------------------------|---------------------|----------------------------------|
| C117B4KY0554  | Anggit Nur Ridho         | Cloud Computing     | Institut Teknologi Nasional Bandung |
| C117B4KY2858  | Muhammad Ghaza Azhar Lesmana | Cloud Computing | Institut Teknologi Nasional Bandung |

## Technology

Nutrifact API Machine Learning dibangun menggunakan teknologi berikut:

- **Flask:** Framework web ringan untuk membangun API.
- **Flask-CORS:** Mengelola Cross-Origin Resource Sharing (CORS) untuk memungkinkan permintaan dari domain lain.
- **Pytesseract:** Menggunakan Tesseract OCR untuk ekstraksi teks dari gambar.
- **TensorFlow:** Library machine learning untuk memuat dan menjalankan model prediksi.
- **OpenCV:** Library untuk pemrosesan gambar.
- **NumPy:** Library untuk operasi numerik dan manipulasi array.
- **scikit-learn:** Library untuk pre-processing data seperti LabelEncoder dan StandardScaler.

## Requirement

Sebelum memulai, pastikan Anda telah menginstal perangkat lunak berikut di sistem Anda:

- **Python 3.7 atau lebih baru**
- **pip (package installer untuk Python)**
- **Virtual environment (opsional tetapi direkomendasikan)**
- **Tesseract OCR**

### Installation Steps

1. Clone the Repository
Clone the repository from GitHub to your computer.

```bash
git clone https://github.com/NUTRIFACT-TEAM/backend-ml.git
```

2. Create and Activate a Virtual Environment

```
python -m venv .venv
```

- Windows:

```
.venv\Scripts\activate
```

- Linux & MacOS:

```
source .venv/bin/activate
```

3. Install Dependencies

```
pip install -r requirements.txt
```
4. Ensure Model and Data Availability
Ensure the file model.h5, label_encoder.json, and scaler.json are available in the project's root directory.

5. Run the Application
Run the Flask application using the following command:

```bash
python main.py
```

Open your browser and go to http://127.0.0.1:8080 to check the API.
