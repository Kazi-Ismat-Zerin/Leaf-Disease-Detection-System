# 🌿 Smart Leaf Disease Detection & Management System

An end-to-end, AI-powered agricultural solution designed to help farmers and researchers identify plant health issues instantly. This system integrates a deep learning model with a high-speed API and a modern web interface to provide real-time diagnosis and treatment suggestions.

---

## 📖 Table of Contents
- Introduction
- System Architecture
- Key Features
- Technology Stack
- Installation Guide
- API Documentation
- Mobile Integration
- Future Scope

---

## 🌟 Introduction
Agriculture is the backbone of the economy, but plant diseases often reduce crop yields significantly. This project provides an accessible tool that uses Computer Vision to detect whether a leaf is **Healthy** or **Diseased** with high confidence.

---

## 🏗️ System Architecture

### 🔹 Data Processing
- Images are resized to **150x150 pixels**
- Rescaled for optimal model performance

### 🔹 Model Training
- 32-filter CNN (Convolutional Neural Network)
- Trained over 30 epochs

### 🔹 Backend Logic
- Built using **FastAPI**
- Handles image input, model inference, and JSON response

### 🔹 Frontend Interface
- Built using **PHP + AJAX**
- Smooth UI without page reloads

---

## ✨ Key Features

- 📷 **Dual Mode Scanner**
  - Scan using camera
  - Upload from gallery

- 📊 **Instant Accuracy Breakdown**
  - Confidence percentage displayed

- 🌐 **Cross-Platform Access**
  - Works on PC, Mobile, Tablet via browser

- ⚡ **Scalable Design**
  - Can be upgraded to detect specific diseases

---

## 🛠️ Technology Stack

- **Languages:** Python (3.x), PHP (8.x), JavaScript (ES6)
- **Deep Learning:** TensorFlow, Keras
- **Frameworks:** FastAPI, Bootstrap 5
- **Server Tools:** Uvicorn, XAMPP/WAMP

---

## ⚙️ Installation Guide

### 1. Requirements
```bash
pip install tensorflow fastapi uvicorn pillow numpy
```

### 2. Train the Model
```bash
python train.py
```
Output: `leaf_model.h5`

### 3. Run Backend
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Run Frontend
- Place PHP files in `htdocs`
- Update API URL in `index.php` using your IPv4 address

---

## 📡 API Documentation

### Endpoint
`POST /predict`

### Request
- Type: `multipart/form-data`
- Key: `file`

### Response
```json
{
  "status": "Success",
  "class": "Diseased",
  "confidence": "99.94%",
  "message": "The leaf is predicted to be Diseased."
}
```

---

## 📱 Mobile Integration

- Access using your PC's IP address from mobile browser
- Uses device camera for real-time scanning
- No app installation required

---

## 🔮 Future Scope

- 🌱 Multi-class disease detection (Blight, Rust, etc.)
- 📶 Offline mode using TensorFlow Lite (TFLite)
- 💊 Smart recommendation system for pesticides

---

## 👩‍💻 Author
**Your Name**
Kazi Ismat Zerin
Souravi Sultana Sumi

---

## 📄 License
This project is open-source and available under the MIT License.
