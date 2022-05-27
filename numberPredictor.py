import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import matplotlib
import tkinter as tk
import matplotlib.pyplot as plot
import numpy as np

# mnist = tf.keras.datasets.mnist
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
#
# x_train = tf.keras.utils.normalize(x_train,axis=1)
# x_test = tf.keras.utils.normalize(x_test,axis=1)
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# model = tf.keras.models.Sequential();
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
#
# model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
# model.fit(x_train,y_train,epochs=10)
# loss , accuracy   = model.evaluate(x_test,y_test);
#
# print(loss,accuracy)
# model.save('model.json')

model = keras.models.load_model('G:/CODE/Demos/NeuralNetwork/model.json')
for i in range(1, 6):
    imgI = cv.imread(f'G:/CODE/Demos/NeuralNetwork/Data/{i}.png')
    img = cv.imread(f'G:/CODE/Demos/NeuralNetwork/Data/{i}.png')[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(np.argmax(prediction))
    plot.imshow(img[0], cmap='binary')
    plot.show()
