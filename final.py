import cv2
from fer import FER

font = cv2.FONT_HERSHEY_SIMPLEX

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

id = 0
# nom des ids
names = ['Malek_kharoubi',"Ibrahim_salem","Sabrine_hassairi"]

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

# Create an FER object to detect emotions
detector = FER()

# Define the emotions to detect
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

while True:
    ret, img = cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if confidence < 100:
            id = names[id]
        else:
            id = "unknown"

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)

    # Detect emotions in the frame
    results = detector.detect_emotions(img)

    # Loop through the detected faces and draw rectangles around them
    for result in results:
        x, y, w, h = result['box']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Loop through the detected emotions and print the dominant one
        for emotion, score in result['emotions'].items():
            if emotion in emotions:
                if score > 0.5:
                     cv2.putText(img, str(emotion) +' ' +str("%.1f%%" % (score * 100)), (x,y-50), font, 0.7, (255,255,255), 2)

    cv2.imshow('camera',img)

    # Check for the Esc key press to break out of the loop
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# Do a bit of cleanup
cam.release()
cv2.destroyAllWindows()
