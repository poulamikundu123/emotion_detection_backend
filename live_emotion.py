import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
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

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flip camera
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]

        # Resize according to model input size
        face_roi = cv2.resize(face_roi, (48, 48))

        # Normalize
        face_roi = face_roi / 255.0

        # Convert to array
        face_roi = np.reshape(face_roi, (1, 48, 48, 1))

        # Predict emotion
        prediction = model.predict(face_roi, verbose=0)

        # Get emotion label
        emotion = emotion_labels[np.argmax(prediction)]

        # Display emotion text
        cv2.putText(
            frame,
            emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # Show frame
    cv2.imshow("Live Emotion Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()