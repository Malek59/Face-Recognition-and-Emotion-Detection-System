import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


face_id = input('\n Enter user ID ')
print("\n Initializing face capture. Look the camera and wait ...")

count = 0
while True:
    #capturer image
    ret, img = cam.read()
    # transformer le frame en gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detecter le visage
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #dessiner un carré autour du visage
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        count += 1
        # enregister les visages capturer dans le dossier "dataset"
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 100: # Take 100 faces and stop video
        break

cam.release()
cv2.destroyAllWindows()