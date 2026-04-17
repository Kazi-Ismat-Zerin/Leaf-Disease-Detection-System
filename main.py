
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os

# 1. FastAPI App Setup
app = FastAPI(
    title="Leaf Disease Detection System",
    description="AI-powered system to identify diseases in Potato and Tomato leaves.",
    version="2.0.0"
)

# 2. Directory Configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(base_dir, "static")
templates_path = os.path.join(base_dir, "templates")

# 3. Static Files & Templates Configuration
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)

# 4. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Load AI Model
MODEL = tf.keras.models.load_model("leaf_model.h5")

# Class List (Matches your dataset folders)
CLASS_NAMES = [
    "Potato___Early_blight", 
    "Potato___Late_blight", 
    "Potato___healthy", 
    "Tomato___Early_blight", 
    "Tomato___Late_blight", 
    "Tomato___healthy"
]

# 6. Information & Cure Dictionary (Advice in Bengali, others in English)
LEAF_INFO = {
    "Potato___Early_blight": {
        "pata": "Potato", 
        "rog": "Early Blight", 
        "cure": "আক্রান্ত পাতা তুলে পুড়িয়ে ফেলুন। কপার ফাঙ্গিসাইড স্প্রে করুন।"
    },
    "Potato___Late_blight": {
        "pata": "Potato", 
        "rog": "Late Blight", 
        "cure": "জলাবদ্ধতা এড়িয়ে চলুন। অবিলম্বে ম্যানকোজেব বা রিডোমিল গোল্ড ছত্রাকনাশক ব্যবহার করুন।"
    },
    "Potato___healthy": {
        "pata": "Potato", 
        "rog": "Healthy", 
        "cure": "আপনার আলু গাছটি সুস্থ আছে! নিয়মিত যত্ন নিন।"
    },
    "Tomato___Early_blight": {
        "pata": "Tomato", 
        "rog": "Early Blight", 
        "cure": "আক্রান্ত পাতা ছাঁটাই করুন। গাছের গোড়ায় পর্যাপ্ত বাতাস চলাচলের ব্যবস্থা রাখুন।"
    },
    "Tomato___Late_blight": {
        "pata": "Tomato", 
        "rog": "Late Blight", 
        "cure": "আর্দ্রতা কমানোর ব্যবস্থা করুন এবং ট্রাইকোডার্মা ব্যবহার করুন।"
    },
    "Tomato___healthy": {
        "pata": "Tomato", 
        "rog": "Healthy", 
        "cure": "আপনার টমেটো গাছটি সম্পূর্ণ সুস্থ। অভিনন্দন!"
    }
}

# 7. Image Processing Function
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image = image.resize((128, 128)) 
    return np.array(image)

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", tags=["AI Prediction"])
async def predict(file: UploadFile = File(...)):
    # 1. Read and Process Image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    # 2. Get Prediction
    predictions = MODEL.predict(img_batch)
    
    # 3. Identify Class and Confidence Level
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = np.max(predictions[0])
    
    # 4. Confidence Threshold (৮০% এর নিচে হলে সেটি ফিল্টার হবে)
    THRESHOLD = 0.80 
    
    if confidence < THRESHOLD:
        return {
            'status': 'Success',
            'leaf_name': 'Unknown',
            'disease': 'Not Identified',
            'cure': 'এই ছবিটি আলু বা টমেটো পাতার সাথে মিলছে না। অনুগ্রহ করে পরিষ্কার ছবি দিয়ে পুনরায় চেষ্টা করুন।',
            'confidence': f"{round(confidence * 100, 2)}%",
            'original_class': 'None'
        }

    # 5. Get Data from Dictionary
    info = LEAF_INFO.get(predicted_class, {"pata": "Unknown", "rog": "Unknown", "cure": "তথ্য পাওয়া যায়নি।"})

    return {
        'status': 'Success',
        'leaf_name': info['pata'],
        'disease': info['rog'],
        'cure': info['cure'],
        'confidence': f"{round(confidence * 100, 2)}%",
        'original_class': predicted_class
    }

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)