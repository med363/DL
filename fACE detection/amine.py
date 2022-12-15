# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 16:47:07 2021

@author: ASUS
"""

import cv2 
print(cv2.__version__)
import numpy as np
import matplotlib.pyplot as plt

img1=cv2.imread("test2.jpg")
def face_detect(img):    
    
    image_copy = img.copy()
    
    image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
    imgG=cv2.cvtColor(image_copy,cv2.COLOR_RGB2GRAY)
    plt.imshow(imgG, cmap="gray")
    haaracasade_face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faceRec=haaracasade_face.detectMultiScale(imgG,scaleFactor=1.2,minNeighbors=3)
    print("facefound:", len(faceRec))
    for (x,y,w,h) in faceRec:
        cv2.rectangle(image_copy,(x,y),(x+w,y+h),(255,0,0),5)
    return plt.imshow(image_copy)

image=cv2.imread("test1.jpg")
plt.imshow(image)
face_detect(image)