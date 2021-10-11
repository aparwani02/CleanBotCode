import numpy as np #NumPy library
import pandas as pd #pandas library
import cv2 #OpenCV

#make a video camera object
cap = cv2.VideoCapture(0)

#locate the opencv haar cascade identifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

while(True):
    #Capture frame-by-frame from video camera
    ret, frame = cap.read()

    #Convert image to gray-scale to deal with shadows
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    #detect face using haar cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        #draw circle around face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(200,200,0),2)
        # cv2.circle(frame,(int((x+x+w)/2),int((y+y+w)/2)),int((x+x+w)/2-x),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #detect eyes using haar cascade and draw circles around them
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.circle(roi_color, (int((ex+ex+ew)/2),int((ey+ey+eh)/2)),int((ex+ex+ew)/2-ex),(255,0,0),2)

        #detect smile using haar cascade and draw rectangle around it
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)

    #detects upper body using haar cascade
    body = upperbody_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    #draws rectangle over upper body
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
