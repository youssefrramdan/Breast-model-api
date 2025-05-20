from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # السماح بطلبات من مصادر مختلفة

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
# قد تحتاج لتعديل هذه القيم حسب ما تم تدريب النموذج عليه
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

@app.route('/predict', methods=['POST'])
def predict():
    """
    نقطة نهاية API للتنبؤ باستخدام النموذج
    """
    # التحقق من وجود ملف صورة في الطلب
    if 'image' not in request.files:
        return jsonify({'error': 'لا توجد صورة في الطلب'}), 400

    image_file = request.files['image']

    try:
        # فتح الصورة
        image = Image.open(io.BytesIO(image_file.read()))

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

        return jsonify(result)

    except Exception as e:
        logger.error(f"خطأ في المعالجة: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    نقطة نهاية للتحقق من حالة API
    """
    health_status = {
        'status': 'up' if model is not None else 'down',
        'model_loaded': model is not None
    }
    return jsonify(health_status)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
