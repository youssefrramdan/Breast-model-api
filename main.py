from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO
import time
import os

app = FastAPI()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["benign", "malignant"]

def preprocess_image(image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

@app.post("/predict")
async def predict(image_url: str = Query(..., description="Direct image URL")):
    total_start = time.time()

    # Step 1: Download image
    download_start = time.time()
    response = requests.get(image_url, timeout=10)
    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": f"Invalid content type: {content_type}"})
    img = Image.open(BytesIO(response.content))
    download_end = time.time()

    # Step 2: Predict
    pred_start = time.time()
    arr = preprocess_image(img)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], arr)

    # Run inference
    interpreter.invoke()

    # Get prediction
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
    label = class_names[1] if pred > 0.5 else class_names[0]
    confidence = float(pred) if pred > 0.5 else 1 - float(pred)
    pred_end = time.time()

    # Step 3: Return result
    total_end = time.time()

    return JSONResponse({
        "label": label,
        "confidence": round(confidence, 2),
    })

@app.get("/info")
async def info():
    return {
        "app_name": "Breast Cancer Classification API",
        "model": "CNN model using TensorFlow Lite",
        "classes": class_names,
        "endpoints": {
            "/predict": "POST - Provide image URL to classify",
            "/info": "GET - Get information about this API"
        },
    }

# For running locally or on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)
