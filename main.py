from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io
import logging
from typing import Dict, Any

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# إعداد CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يسمح بجميع المصادر
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل النموذج المدرب
MODEL_PATH = 'Breast.keras'

# التحقق من وجود النموذج
if not os.path.exists(MODEL_PATH):
    logger.error(f"ملف النموذج غير موجود: {MODEL_PATH}")
    model = None
else:
    try:
        logger.info("جاري تحميل النموذج...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("تم تحميل النموذج بنجاح")
    except Exception as e:
        logger.error(f"خطأ في تحميل النموذج: {str(e)}")
        model = None

# حجم الصورة الذي يتوقعه النموذج
IMG_SIZE = (224, 224)

def preprocess_image(image):
    """
    معالجة الصورة لتكون جاهزة للنموذج
    """
    # تحويل الصورة إلى الحجم المطلوب
    image = image.resize(IMG_SIZE)

    # تحويل الصورة إلى مصفوفة
    img_array = np.array(image)

    # التأكد من أن الصورة بتنسيق RGB
    if len(img_array.shape) == 2:  # صورة بالأبيض والأسود
        img_array = np.stack((img_array,) * 3, axis=-1)

    # إعادة تشكيل المصفوفة لتناسب مدخلات النموذج [1, height, width, channels]
    img_array = np.expand_dims(img_array, axis=0)

    # تطبيع البيانات (بين 0 و 1)
    img_array = img_array / 255.0

    return img_array

@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    نقطة نهاية API للتنبؤ باستخدام النموذج
    """
    try:
        # قراءة الصورة
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))

        # معالجة الصورة
        processed_image = preprocess_image(image)

        # إجراء التنبؤ
        prediction = model.predict(processed_image)

        # تفسير النتيجة
        if prediction.shape[1] == 1:
            probability = float(prediction[0][0])
            result = {
                'prediction': 'سرطاني' if probability > 0.5 else 'غير سرطاني',
                'probability': probability
            }
        else:
            class_index = np.argmax(prediction[0])
            probability = float(prediction[0][class_index])
            classes = ["benign", "malignant", "normal"]
            result = {
                'prediction': classes[class_index],
                'probability': probability
            }

        return result

    except Exception as e:
        logger.error(f"خطأ في المعالجة: {str(e)}")
        return {"error": str(e)}

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    نقطة نهاية للتحقق من حالة API
    """
    health_status = {
        'status': 'up' if model is not None else 'down',
        'model_loaded': model is not None
    }
    return health_status

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host='0.0.0.0', port=port)
