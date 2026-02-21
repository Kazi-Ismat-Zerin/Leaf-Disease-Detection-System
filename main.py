import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf


app = FastAPI(
    title="Leaf Disease Detection System",
    description="This is an AI-powered system that identifies whether a leaf is Healthy or Diseased.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = tf.keras.models.load_model("leaf_model.h5")
CLASS_NAMES = ["Diseased", "Healthy"]


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    
    image = image.resize((150, 150))
    return np.array(image) / 255.0


@app.post("/predict", tags=["AI Prediction"])
async def predict(file: UploadFile = File(...)):
    
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
   
    predictions = MODEL.predict(img_batch)
    
    
    score = float(predictions[0][0])
    
    if score > 0.5:
        predicted_class = CLASS_NAMES[1] # Healthy
        confidence = score
    else:
        predicted_class = CLASS_NAMES[0] # Diseased
        confidence = 1 - score

    
    return {
        'status': 'Success',
        'class': predicted_class,
        'confidence': f"{round(confidence * 100, 2)}%",
        'message': f"The leaf is predicted to be {predicted_class}."
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)