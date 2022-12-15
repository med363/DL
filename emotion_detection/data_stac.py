# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:20:17 2021

@author: ASUS
"""

from imutils import paths

known_emotion=("Angry", "Disgust","Fear", "Happy","Sad","Surprise","Neutral")

for emotion in known_emotion:
    list_path = list(paths.list_images("dataset/"+emotion))
    print("nombre d'image pour emotion  : {} , {} images".format(emotion, len(list_path)))