import numpy as np 
import cv2 

CameraIndex = 0;
cap =cv2.VideoCapture(CameraIndex) 

while(True):
    ret, frame=cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#cv2.waitKey(0) doesn't help....
