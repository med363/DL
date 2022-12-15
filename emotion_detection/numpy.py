# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 17:17:36 2021

@author: ASUS
"""
import numpy as np
import pandas as pd


#x=np.zeros((2, 3))
#print(x)
#print(x.shape)

x1= np.array([[1, 2, 3], [4, 5, 6]])

print(x1.shape)

x2=np.array([[1,2,3],[4,5,6],[7,8,9]])

print(x2.shape)

x3=np.array([[[10,11,12],[25,23,24],[145,214,123]]])

print(x3.shape)
print(x3.shape[0])

print('on ajout un axe z==>(z,x,y)')
x33=np.expand_dims(x1, 0)
print(x33)
print(x33.shape)

print('on ajout un axe z==>(x,z,y)')
x33=np.expand_dims(x1, 1)
print(x33)
print(x33.shape)

print('on ajout un axe z==>(x,y,z)')
x33=np.expand_dims(x1, 2)
print(x33)
print(x33.shape)

print('demention -1 z==>(x,y,z)')
x33=np.expand_dims(x1, -1)
print(x33)
print(x33.shape)
print('on ajout un axe z==>(x,z,y)')
x33=np.expand_dims(x1, -2)
print(x33)
print(x33.shape)
print('on ajout un axe z==>(z,x,y)')
x33=np.expand_dims(x1, -3)
print(x33)
print(x33.shape)


x44=np.vstack([x33])
print(x44.shape)

a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
c=np.vstack((a,b))
print(c.shape)

