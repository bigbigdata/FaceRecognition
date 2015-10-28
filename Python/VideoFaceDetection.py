import numpy as np 
import cv2 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

CameraIndex = 0;
cap = cv2.VideoCapture(CameraIndex) 

while(True):
    ret, stream=cap.read()
    stream_gray = cv2.cvtColor(stream,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(stream_gray,1.3,5)
    
    for (x,y,w,h) in faces: 
        cv2.rectangle(stream,(x,y),(x+w,y+h),(255,0,0),1)
        roi_gray=stream_gray[y:y+h,x:x+w]
        roi_color=stream[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    cv2.imshow('Face Detection', stream)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
