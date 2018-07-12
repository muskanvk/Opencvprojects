import numpy as np
import cv2

cap = cv2.VideoCapture('people-walking.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
 
    cv2.imshow('frame',frame)
    cv2.imshow('frame1',fgmask)
    

    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
