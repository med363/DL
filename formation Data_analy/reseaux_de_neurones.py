# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 15:09:33 2020

@author: ASUS
"""

import os
import zipfile
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
!wget --no-check-certificate \
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" \
    -O "/tmp/cats-and-dogs.zip"
local_zip = '/tmp/cats-and-dogs.zip'
zip_ref   = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
print(len(os.listdir("/tmp/PetImages/Cat/")))
print(len(os.listdir("/tmp/PetImages/Dog/")))
try:
  os.mkdir("/tmp/cats-v-dogs")
  os.mkdir("/tmp/cats-v-dogs/training")
  os.mkdir("/tmp/cats-v-dogs/training/cats")
  os.mkdir("/tmp/cats-v-dogs/training/dogs")
  os.mkdir("/tmp/cats-v-dogs/testing")
  os.mkdir("/tmp/cats-v-dogs/testing/dogs")
  os.mkdir("/tmp/cats-v-dogs/testing/cats")
except OSError:
  pass
def split_data(source,splite_size,TRAINING,TESTING):
  files=[]
  for i in os.listdir(source):
    file=source + i
    if os.path.getsize(file)>0:
      files.append(i)
    else:
      print(file,' is empty')
  training_length = int(len(files) * splite_size)
  testing_length = int(len(files) - training_length)
  shuffled_set = random.sample(files, len(files))
  training_set = shuffled_set[0:training_length]
  testing_set = shuffled_set[-testing_length:]
  for filename in training_set:
    this_file = source + filename
    destination = TRAINING + filename
    copyfile(this_file, destination)
  for filename in testing_set:
    this_file = source + filename
    destination = TESTING + filename
    copyfile(this_file, destination)

cat_source_dir="/tmp/PetImages/Cat/"
dog_source_dir="/tmp/PetImages/Dog/"
training_cat_dir="/tmp/cats-v-dogs/training/cats/"
testing_cat_dir="/tmp/cats-v-dogs/testing/cats/"
training_dog_dir="/tmp/cats-v-dogs/training/dogs/"
testing_dog_dir="/tmp/cats-v-dogs/testing/dogs/"

split_data(dog_source_dir,0.8,training_dog_dir,testing_dog_dir)
split_data(cat_source_dir,0.8,training_cat_dir,testing_cat_dir)
print(len(os.listdir("/tmp/cats-v-dogs/training/dogs/")))
print(len(os.listdir("/tmp/cats-v-dogs/training/cats/")))
print(len(os.listdir("/tmp/cats-v-dogs/testing/dogs/")))
print(len(os.listdir("/tmp/cats-v-dogs/testing/cats/")))
model=tf.keras.models.Sequential([  
    tf.keras.layers.Conv2D(16,(3,3),activation ='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation ='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation ='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='softmax')    
 ])
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
training_dir="/tmp/cats-v-dogs/training/"
train_data_gen=ImageDataGenerator(rescale=1.0/255.)
train_generator=train_data_gen.flow_from_directory(training_dir,batch_size=100,class_mode="binary",target_size=(150,150))
testing_dir="/tmp/cats-v-dogs/testing/"
test_data_gen=ImageDataGenerator(rescale=1.0/255.)
test_generator=train_data_gen.flow_from_directory(testing_dir,batch_size=100,class_mode="binary",target_size=(150,150))
x=model.fit(train_generator,epochs=40,steps_per_epoch=90,validation_data=test_generator,validation_steps=6)
Acr=x.history['acc']
print(Acr)
Los=x.history["loss"]
print(Los)
val_los=x.history["val_loss"]
print(val_los)
val_acr=x.history["val_acc"]
print(val_acr)
epochs=range(len(Acr))

plt.plot(epochs,Acr,"r","training_accuracy")
plt.plot(epochs,val_acr,"b","testing_accuracy")
plt.title("training_and_testing_accuracy")
plt.figure()

plt.plot(epochs,Los,"r","training_los")
plt.plot(epochs,val_los,"b","testing_los")
plt.title("training_and_testing_los")
plt.figure()

