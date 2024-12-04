const Hapi = require('@hapi/hapi');
const multer = require('multer');
const Tesseract = require('tesseract.js');
const { Firestore } = require('@google-cloud/firestore');
const path = require('path');
const { Storage } = require('@google-cloud/storage');

// Konfigurasi Firestore dan Google Cloud Storage
const firestore = new Firestore();
const storage = new Storage();
const bucketName = 'your-bucket-name'; // Ganti dengan nama bucket Anda

// Fungsi untuk mengupload file ke Google Cloud Storage
const uploadToGCS = async (file) => {
  const bucket = storage.bucket(bucketName);
  const blob = bucket.file(file.originalname);
  await blob.save(file.buffer);
  return `https://storage.googleapis.com/${bucketName}/${file.originalname}`;
};

// Fungsi untuk ekstraksi data dari gambar menggunakan Tesseract OCR
const extractNutritionFromImage = (imageUrl) => {
  return new Promise((resolve, reject) => {
    Tesseract.recognize(
      imageUrl,
      'eng',
      {
        logger: (m) => console.log(m),
      }
    ).then(({ data: { text } }) => {
      const nutritionData = extractNutritionInfo(text);
      resolve(nutritionData);
    }).catch(reject);
  });
};

// Fungsi untuk mengekstraksi informasi gula dan lemak dari teks OCR
const extractNutritionInfo = (ocrText) => {
  const lines = ocrText.split('\n');
  let sugar = null, fat = null;

  for (let line of lines) {
    if (line.toLowerCase().includes('sugar') || line.toLowerCase().includes('gula')) {
      sugar = line.match(/[\d.]+/g) ? line.match(/[\d.]+/g)[0] : null;
    }
    if (line.toLowerCase().includes('fat') || line.toLowerCase().includes('lemak')) {
      fat = line.match(/[\d.]+/g) ? line.match(/[\d.]+/g)[0] : null;
    }
  }

  return { sugar, fat };
};

// Fungsi untuk menghitung Health Grade berdasarkan gula dan lemak
const calculateHealthGrade = (sugar, fat) => {
  // Normalisasi nilai gula dan lemak
  const sugar100g = parseFloat(sugar) * 100;
  const fat100g = parseFloat(fat) * 100;

  let grade = 'A';  // Default grade is A
  if (sugar100g > 5 || fat100g > 2.8) {
    grade = 'D';  // Grade D for high sugar or fat
  } else if (sugar100g > 3 || fat100g > 1.5) {
    grade = 'C';
  } else if (sugar100g > 1 || fat100g > 0.7) {
    grade = 'B';
  }

  return grade;
};

// API Hapi.js
const init = async () => {
  const server = Hapi.server({
    port: 4000,
    host: 'localhost',
  });

  // Set up Multer for file uploads
  const storageEngine = multer.memoryStorage();
  const upload = multer({ storage: storageEngine }).single('label_image');

  // Route untuk menerima gambar dan memprosesnya
  server.route({
    method: 'POST',
    path: '/predict',
    handler: async (request, h) => {
      const file = request.payload.label_image;

      if (!file) {
        return h.response({ error: 'No image uploaded' }).code(400);
      }

      try {
        // Upload gambar ke Google Cloud Storage
        const imageUrl = await uploadToGCS(file);
        
        // Ekstrak data dari gambar menggunakan OCR
        const { sugar, fat } = await extractNutritionFromImage(imageUrl);

        if (!sugar || !fat) {
          return h.response({ error: 'Failed to extract nutrition information' }).code(400);
        }

        // Hitung Health Grade
        const grade = calculateHealthGrade(sugar, fat);

        // Simpan data ke Firestore
        const barcodeId = request.payload.barcodeId;  // Ambil ID Barcode
        const docRef = firestore.collection('products').doc(barcodeId);
        await docRef.set({
          barcodeId,
          fat,
          sugar,
          healthGrade: grade,
          description: 'Sample description for product',
          brand: 'Sample Brand',
          variant: 'Original 100g',
        });

        return h.response({
          message: 'Product data processed successfully',
          barcodeId,
          fat,
          sugar,
          healthGrade: grade,
        }).code(200);
      } catch (error) {
        console.error('Error processing image:', error);
        return h.response({ error: error.message }).code(500);
      }
    },
  });

  await server.start();
  console.log('Server running on %s', server.info.uri);
};

init();