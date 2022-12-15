# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 14:25:02 2021

@author: ASUS
"""

import cv2
import numpy as np
import os 
import pandas as pd

known_emotion=("Angry", "Disgust","Fear", "Happy","Sad","Surprise","Neutral")
os.makedirs("DataSet",exist_ok=True)
for path in known_emotion:
    os.makedirs("Dataset/"+path,exist_ok=True)

data = pd.read_csv("fer2013.csv",delimiter=",")

for index,row in data.iterrows():
    pixels = np.asarray(list(row['pixels'].split(' ')), dtype=np.uint8)
    img = pixels.reshape(48,48)
    emotion_index=int(row['emotion'])
    #print(known_emotion[emotion_index])
    save_path="Dataset/"+known_emotion[emotion_index]+"/"+str(index)+".jpg"
    cv2.imwrite(save_path,img)
    print('image:',save_path)
