import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('emotion_detection_model.h5')

# Define the emotions that the model can detect
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture(0)

# Continuously process frames from the camera stream
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Preprocess the frame for prediction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))

    # Predict the emotion from the frame
    prediction = model.predict(reshaped)
    max_index = np.argmax(prediction)
    emotion = emotions[max_index]

    # Display the emotion prediction on the frame
    cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the display window
cap.release()
cv2.destroyAllWindows()
