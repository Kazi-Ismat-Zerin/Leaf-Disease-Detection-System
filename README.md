# 🌿 Leaf Disease Detection System

An AI-powered application that detects plant leaf diseases from images using Deep Learning (CNN). The system provides real-time predictions through a FastAPI backend and a clean, user-friendly Streamlit interface—helping farmers and researchers make timely decisions.

---

## 🧾 Executive Summary

 This project focuses on developing an AI-powered solution to detect plant diseases using leaf images. By combining Deep Learning (CNN) with a robust FastAPI backend and a user-friendly Streamlit interface, the system delivers real-time diagnostic results to optimize crop health management.

---

## 🎯 Project Objectives

✅ Automatically classify leaf images into **Healthy** or **Diseased** categories.

🔗 Provide an **API-driven architecture** for seamless integration with web and mobile platforms.

📊 Deliver **instant feedback with confidence scores** to support better decision-making.

---

## ⭐ Core Features

🧠 **AI-Powered Classification**\
Uses a Convolutional Neural Network (CNN) trained on labeled datasets to identify disease patterns.

⚡ **Real-time API Endpoint**\
A high-performance `/predict` endpoint handles image uploads via HTTP POST requests.

🔧 **Preprocessing Pipeline**\
Automatically normalizes and resizes input images to **150 × 150 pixels** for model compatibility.

🖥️ **User Dashboard**\
A Streamlit-based web interface lets users upload images and view results without writing code.

📈 **Transparent Accuracy**\
Displays a confidence percentage for every prediction to indicate reliability.

---

## 🏗️ System Architecture & Methodology

🗂️ **Data Layer**\
Labeled image datasets organized into **Healthy** and **Diseased** folders for supervised learning.

🧠 **Model Layer**\
A TensorFlow-based model (`leaf_model.h5`) performs mathematical analysis of image features.

🚀 **Backend Layer**\
Built with **FastAPI**, manages API logic, handles CORS, and runs the inference engine.

🎨 **Frontend Layer**\
A **Streamlit** application acts as the client and displays final predictions.

---

## ⚙️ Tech Stack

-  **Language:** Python
-  **Deep Learning:** TensorFlow / Keras
-  **Backend Framework:** FastAPI
-  **Server:** Uvicorn
-  **Frontend:** Streamlit
-  **Image Processing:** Pillow (PIL), NumPy
-  **Model Format:** `.h5`

---

## 🔄 Implementation Workflow

📤 **Image Input**\
User uploads a `.jpg` or `.png` file via the Streamlit app.

🔧 **Processing**\
The image is converted into a NumPy array and resized to **150 × 150 pixels**.

🧠 **Inference**\
The processed array is passed to `MODEL.predict()`.

📡 **Output**\
The API returns a JSON response containing the predicted class and confidence score, which is rendered on the UI.

---

## ▶️ How to Run the Project

1. **Clone the repository**

   ```bash
   git clone <https://github.com/Kazi-Ismat-Zerin/Leaf-Disease-Detection-System>
   cd leaf-disease-detection
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI backend**

   ```bash
   uvicorn main:app --reload
   ```

4. **Run the Streamlit frontend**

   ```bash
   streamlit run app.py
   ```

5. **Open the app**\
   Visit the URL shown in the terminal to upload an image and view predictions.

---

## 🚀 Future Scope

  Expand datasets to improve accuracy and robustness.

 Detect **specific disease types** (e.g., rust, blight).

 Add **treatment recommendations** based on detected diseases.

 Deploy the API to the cloud for global accessibility.
