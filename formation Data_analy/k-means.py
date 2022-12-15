# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:12:26 2020

@author: ASUS
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
data = datasets.load_iris()
x=pd.DataFrame(data.data)
x.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
x.head()
y=pd.DataFrame(data.target)
y.columns=["target"]
from sklearn.cluster import KMeans
model=KMeans(n_clusters=3)
model.fit(x)
print(model.labels_)
colormap=np.array(["Red","green","blue"])
plt.scatter(x.PetalLengthCm,x.PetalWidthCm,c=colormap[y.target],s=40)
plt.suptitle("PetalOriginal")
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.show()
plt.scatter(x.SepalLengthCm,x.SepalWidthCm,c=colormap[y.target],s=40)
plt.suptitle("SepalOriginal")
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.show()
plt.scatter(x.PetalLengthCm,x.PetalWidthCm,c=colormap[model.labels_],s=40)
plt.suptitle("PetalModel")
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.show()
plt.scatter(x.SepalLengthCm,x.SepalWidthCm,c=colormap[model.labels_],s=40)
plt.suptitle("SepalModel")
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.show()

