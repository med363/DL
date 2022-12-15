# -*- coding: utf-8 -*-
"""

"""
import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_model = load_model('_mini_XCEPTION.102-0.66.hdf5')
known_emotion = ("angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral")
cam = cv2.VideoCapture(0)
while True :
    image = cam.read()[1]
    faces = faceCascade.detectMultiScale(image, minNeighbors=5)
    for (x,y,w,h) in faces :
            face_image = image[y:y+h, x:x+w]
            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,255), 1)
            face_image = cv2.cvtColor(face_image,cv2.COLOR_BGR2GRAY)
            face_image = cv2.resize(face_image, (64,64))
            face_image = face_image.astype("float") / 255.0
            print(face_image.shape)
            face_image = img_to_array(face_image)
            print(face_image.shape)
            result = emotion_model.predict(np.expand_dims(face_image, axis=0))[0]
            result_index = np.argmax(result)
            emotion = known_emotion[result_index]
            print(emotion)
            
            cv2.putText(image, emotion, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv2.imshow("result", image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
