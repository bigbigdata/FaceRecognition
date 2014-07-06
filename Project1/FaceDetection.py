import numpy as np 
import cv2 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')
img3 = cv2.imread('3.jpg')

img1_gray = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
img3_gray = cv2.cvtColor(img3,cv2.COLOR_RGB2GRAY)

#img1 = img3 
#img1_gray = img3_gray


faces = face_cascade.detectMultiScale(img1_gray,1.3,5)
for (x,y,w,h) in faces: 
    cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),1)
    roi_gray=img1_gray[y:y+h,x:x+w]
    roi_color=img1[y:y+h,x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes: 
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('Face Detection',img1)
cv2.waitKey(0)
cv2.destroyAllWindows() 

# not perfect: addtional eye or face detection windows show up 
