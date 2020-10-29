import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
import pandas as pd
from joblib import load



svc= load('smile_d.sav')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vid = cv2.VideoCapture(0)
l= ['not smile','smile']
while True:
    ret, img = vid.read()
    cv2.waitKey(1)
    if not ret:
        break
    faces =  face_cascade.detectMultiScale(img, 1.1, 4)[0]
    (x, y, w, h) = faces
    roi = img[y:y+h, x:x+w]
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi,(64,64))
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    prediction=svc.predict(roi.ravel().reshape(1,-1))
    cv2.putText(img,'{}'.format(l[int(prediction)]),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
    cv2.imshow('img', img)
    
vid.release()    






