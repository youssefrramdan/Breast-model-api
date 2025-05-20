# Breast Cancer Classification API

This API provides breast cancer classification using a deep learning model. It can classify breast images into three categories:

- Benign
- Malignant
- Normal

## API Endpoints

### 1. Health Check

- **URL:** `/health`
- **Method:** GET
- **Response:** Status of the API

### 2. Predict

- **URL:** `/predict`
- **Method:** POST
- **Body:** Form-data with 'file' key containing the image
- **Response:** Classification results with confidence scores

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python main.py
```

## Model

The API uses a TensorFlow model trained on breast cancer images. The model file (`Breast.keras`) should be present in the root directory.
