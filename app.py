from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64

# =========================
# FASTAPI APP
# =========================
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL
# =========================
model = load_model("cnn_emotion_model.h5")

# Emotion labels
emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]

# =========================
# LOAD FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# =========================
# HOME ROUTE
# =========================
@app.get("/")
def home():
    return {
        "message": "Emotion Detection Backend Running"
    }

# =========================
# EMOTION DETECTION ROUTE
# =========================
@app.post("/detect-emotion")
async def detect_emotion(data: dict):

    try:

        # Get base64 image from frontend
        image_data = data["image"]

        # Remove base64 header
        image_data = image_data.split(",")[1]

        # Decode image
        image_bytes = base64.b64decode(image_data)

        # Convert to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)

        # Decode image using OpenCV
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        # Default emotion
        detected_emotion = "No Face Detected"

        # Process first face only
        for (x, y, w, h) in faces:

            # Extract face
            face_roi = gray[y:y+h, x:x+w]

            # Resize
            face_roi = cv2.resize(face_roi, (48, 48))

            # Normalize
            face_roi = face_roi / 255.0

            # Reshape
            face_roi = np.reshape(face_roi, (1, 48, 48, 1))

            # Predict
            prediction = model.predict(face_roi, verbose=0)

            # Get label
            detected_emotion = emotion_labels[np.argmax(prediction)]

            break

        return {
            "emotion": detected_emotion
        }

    except Exception as e:

        return {
            "error": str(e)
        }