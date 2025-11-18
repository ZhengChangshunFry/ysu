# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist=keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

print('train_images.shape=',train_images.shape)
print('test_images.shape=',test_images.shape)
print('train_labels.shape=',train_labels.shape)
print('test_labels.shape=',test_labels.shape)

num=train_images[6000]
for i in range(28):
  for j in range(28):
    print('{:4d}'.format(num[i][j]),end='')
  print()

fig=plt.figure()
ax1=fig.add_subplot(1,4,1)
ax2=fig.add_subplot(1,4,2)
ax3=fig.add_subplot(1,4,3)
ax4=fig.add_subplot(1,4,4)

ax1.imshow(train_images[1000],cmap='Greys')
ax2.imshow(train_images[2000],cmap='Greys')
ax3.imshow(train_images[3000],cmap='Greys')
ax4.imshow(train_images[4000],cmap='Greys')

print('train_labels[:4]=',train_labels[:4])

train_images,test_images=train_images/255,test_images/255

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(train_images,train_labels,epochs=5)

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
model.fit(train_images,train_labels,epochs=5)

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
model.fit(train_images,train_labels,epochs=2)