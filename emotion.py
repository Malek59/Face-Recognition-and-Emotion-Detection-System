import cv2
from fer import FER

font = cv2.FONT_HERSHEY_SIMPLEX

# Create a VideoCapture object to read from the webcam
cap = cv2.VideoCapture(0)

# Creer FER object
detector = FER()
# Definir les emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Loop through frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Detecter emotions dans images
    results = detector.detect_emotions(frame)
    # parcourir les visages et mettre un rectangle
    for result in results:
        x, y, w, h = result['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # parcourir les emotions et ecrire l'emotion dominante avec plus de 50%
        for emotion, score in result['emotions'].items():
            if emotion in emotions:
                if score > 0.5:
                     cv2.putText(frame, str(emotion), (x,y-50), font, 0.7, (255,255,255), 2)

    # Display the frame with the detected faces and emotions
    cv2.imshow('frame', frame)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
        
# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()