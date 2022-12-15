# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:57:34 2021

@author: ASUS
"""

import pandas as pd
import numpy as np
import cv2
from arch import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix

def load_fer2013(path):
    data = pd.read_csv(path)
    pixels = data['pixels'].tolist()
    faces = []
    for pix in pixels :
        face = [int(pixel) for pixel in pix.split(' ')]
        face = np.asarray(face).reshape(48,48)
        face = cv2.resize(face.astype('uint8'),(48,48))
        faces.append(face)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion'])
    print(emotions)
    return faces, emotions


a,b=load_fer2013('C:/Users/ASUS/Desktop/emotion_detection/fer2013.csv')