<p align="center">
  <h1 align="center">🧠 Trans Neural Network (TNN)</h1>
  <p align="center">
    <strong>Hybrid Transformer + CNN Architecture for Malware Activity Detection</strong>
  </p>
  <p align="center">
    <a href="#overview">Overview</a> •
    <a href="#arsitektur">Arsitektur</a> •
    <a href="#alur-kerja">Alur Kerja</a> •
    <a href="#tech-stack">Tech Stack</a> •
    <a href="#instalasi">Instalasi</a> •
    <a href="#evaluasi">Evaluasi</a>
  </p>
</p>

---

## 📋 Overview

**Trans Neural Network (TNN)** adalah arsitektur hybrid yang menggabungkan kekuatan **Transformer** dan **Convolutional Neural Network (CNN)** untuk mendeteksi aktivitas malware pada lalu lintas jaringan (*network traffic*).

Model ini memanfaatkan:
- **Transformer** → untuk memahami pola sekuensial dan kontekstual dari fitur network traffic melalui mekanisme *self-attention*.
- **CNN** → untuk mengekstraksi fitur spasial dan lokal dari representasi data.
- **Feature Fusion** → menggabungkan fitur dari kedua arsitektur untuk klasifikasi yang lebih akurat.

### 🎯 Tujuan
Mengklasifikasikan lalu lintas jaringan menjadi **Benign (normal)** atau **Malware** secara akurat menggunakan pendekatan deep learning hybrid.

---

## 🏗️ Arsitektur

```
┌─────────────────┐
│  Data Collecting │  ← Dataset USTC-TFC2016 (Wireshark)
└────────┬────────┘
         ▼
┌─────────────────────┐
│  Data Preprocessing  │
│  • Feature Selection │
│  • Data Cleaning     │
│  • Data Normalization│
└────────┬────────────┘
         ▼
┌──────────────────────────────────────────────┐
│         TRANS NEURAL NETWORK MODELLING        │
│                                               │
│  Tokenization ──→ Transformer Modelling       │
│                        │                      │
│                   Frozen Weight               │
│                        ▼                      │
│                 Feature Extraction             │
│                        │                      │
│                        ▼                      │
│                  Feature Fusion               │
│                        │                      │
│                        ▼                      │
│                  CNN Modelling                 │
└──────────────────────┬───────────────────────┘
                       ▼
┌─────────────────────────────┐
│         Evaluation           │
│  • Accuracy    • F1-Score    │
│  • Precision   • Loss Func   │
│  • FLOPs       • Conf Matrix │
└─────────────────────────────┘
```

---

## 🔄 Alur Kerja

### 1️⃣ Data Collecting
- Menggunakan dataset **USTC-TFC2016** yang berisi data lalu lintas jaringan.
- Data dikumpulkan menggunakan **Wireshark** dalam format file capture jaringan.
- Terdiri dari sampel traffic **Benign** (normal) dan **Malware**.

### 2️⃣ Data Preprocessing
- **Feature Selection** — Memilih fitur-fitur yang relevan dari data network traffic.
- **Data Cleaning** — Membersihkan data dari noise, missing values, dan duplikasi.
- **Data Normalization** — Menormalkan data agar berada pada skala yang sama untuk mempercepat konvergensi model.

### 3️⃣ Trans Neural Network Modelling

#### 🔹 Tokenization
Data yang sudah dipreprocess dikonversi menjadi token menggunakan **BERT Tokenizer** agar dapat diproses oleh arsitektur Transformer.

#### 🔹 Transformer Modelling
- Menggunakan arsitektur **BERT (Bidirectional Encoder Representations from Transformers)**.
- Memanfaatkan mekanisme *self-attention* untuk menangkap hubungan kontekstual antar fitur.
- Setelah training, **weight di-freeze (Frozen Weight)** untuk mengekstrak fitur.

#### 🔹 Feature Extraction
Mengekstrak representasi fitur (*embeddings*) dari output Transformer yang telah dilatih.

#### 🔹 Feature Fusion
Menggabungkan fitur hasil ekstraksi Transformer dengan fitur asli untuk mendapatkan representasi yang lebih kaya.

#### 🔹 CNN Modelling
- Fitur gabungan diproses oleh **Convolutional Neural Network**.
- CNN mengekstrak pola spasial dan lokal untuk klasifikasi akhir.
- Output: **Benign** atau **Malware**.

### 4️⃣ Evaluation
Model dievaluasi menggunakan berbagai metrik:

| Metrik | Deskripsi |
|--------|-----------|
| **Accuracy** | Persentase prediksi yang benar secara keseluruhan |
| **Precision** | Ketepatan prediksi positif (malware) |
| **F1-Score** | Harmonic mean dari precision dan recall |
| **Loss Function** | Nilai loss selama training dan validasi |
| **FLOPs** | Floating Point Operations — kompleksitas komputasi model |
| **Confusion Matrix** | Visualisasi performa klasifikasi per kelas |

---

## 📁 Struktur Project

```
Trans-Neural-Network/
├── 01-malware-transformer.ipynb   # Notebook utama: Transformer + TNN pipeline
├── 02-tnn_cuda_improve.ipynb      # Notebook optimasi: CUDA acceleration & improvement
├── requirements.txt               # Daftar dependencies
└── README.md                      # Dokumentasi project
```

### 📓 Penjelasan Notebook

| Notebook | Deskripsi |
|----------|-----------|
| `01-malware-transformer.ipynb` | Pipeline lengkap TNN — dari data preprocessing, tokenization, Transformer modelling, feature fusion, hingga CNN classification |
| `02-tnn_cuda_improve.ipynb` | Versi optimasi dengan akselerasi **CUDA GPU** dan perbaikan arsitektur model |

---

## 🛠️ Tech Stack

| Kategori | Teknologi |
|----------|-----------|
| **Bahasa** | Python 3.x |
| **Deep Learning** | PyTorch, TensorFlow/Keras |
| **Transformer** | Hugging Face Transformers (BERT) |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Visualisasi** | Matplotlib, Seaborn |
| **Profiling** | thop (FLOPs calculation) |
| **Environment** | Jupyter Notebook, CUDA GPU |

---

## 🚀 Instalasi

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- pip package manager

### Langkah Instalasi

```bash
# 1. Clone repository
git clone https://github.com/adityadarmawann/Trans-Neural-Network.git
cd Trans-Neural-Network

# 2. Buat virtual environment (opsional tapi direkomendasikan)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan Jupyter Notebook
jupyter notebook
```

### 📦 Dependencies Utama

```
pandas, numpy, scikit-learn
torch, torchvision, tensorflow
transformers (Hugging Face)
thop, safetensors, chardet
matplotlib, seaborn
```

> Lihat [`requirements.txt`](requirements.txt) untuk daftar lengkap.

---

## 📊 Evaluasi

Model TNN dievaluasi secara komprehensif menggunakan metrik berikut:

- ✅ **Accuracy** — Akurasi klasifikasi keseluruhan
- ✅ **Precision** — Ketepatan deteksi malware
- ✅ **F1-Score** — Keseimbangan antara precision dan recall
- 📉 **Loss Function** — Kurva training & validation loss
- ⚡ **FLOPs** — Efisiensi komputasi model
- 📊 **Confusion Matrix** — Detail klasifikasi per kelas (Benign vs Malware)

---

## 📚 Dataset

**USTC-TFC2016** — Dataset benchmark untuk klasifikasi lalu lintas jaringan (*network traffic classification*) yang berisi:
- Data traffic **Benign** (normal)
- Data traffic **Malware** (berbahaya)
- Dikumpulkan menggunakan **Wireshark**

---

## 👤 Author

**M Aditya Darmawan**
- GitHub: [@adityadarmawann](https://github.com/adityadarmawann)

---

## 📄 License

Project ini dibuat untuk keperluan penelitian dan edukasi.

---

<p align="center">
  <sub>Built with ❤️ using Transformer + CNN Hybrid Architecture</sub>
</p>