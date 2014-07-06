import numpy as np 
import cv2 

# Be careful about file name: Typo of "H" does not return any error and I
# thought I am writing the correct code.

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

CameraIndex = 0;
cap =cv2.VideoCapture(CameraIndex) 



while(True):
    ret, img1=cap.read()

    img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img1_gray,1.3,5)
    for (x,y,w,h) in faces: 
        cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),1)
        roi_gray=img1_gray[y:y+h,x:x+w]
        roi_color=img1[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    cv2.imshow('Face Detection', img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
