# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 14:51:57 2021

@author: ASUS
"""
import cv2 
print(cv2.__version__)
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("test.jpg")
print(type(img))
print("shape of bgr or rgb image")
print(img.shape)
#plt.imshow(img)

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#plt.imshow(img)

imgG=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
plt.imshow(imgG, cmap="gray")
print("shapeof gray image")
print(imgG.shape)
"""
haaracasade_face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceRec=haaracasade_face.detectMultiScale(imgG,scaleFactor=1.2,minNeighbors=5)
print("facefound:", len(faceRec))
for (x,y,w,h) in faceRec:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
"""
#plt.imshow(img)

