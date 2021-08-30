import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import cv2
import PIL.Image as Image
from tensorflow import keras
import pathlib

from sklearn.metrics import confusion_matrix
import seaborn as sn

Train_Path = 'train/'
Test_Path = 'test/'
# dataset = tf.keras.utils.get_file(origin=Path)
data_dir = pathlib.Path(Train_Path)
data_dir1 = pathlib.Path(Test_Path)

train = {

   'NORMAL': list(data_dir.glob('NORMAL/*.jpeg')),
   'COVID': list(data_dir.glob('PNEUMONIA/*.jpeg'))

}

test = {
   'NORMAL': list(data_dir1.glob('NORMAL/*.jpeg')),
   'COVID': list(data_dir1.glob('PNEUMONIA/*.jpeg'))

}

train_labels = {

    'NORMAL': 0,
    'COVID': 1
}

test_labels = {

    'NORMAL': 0,
    'COVID': 1
}

x_train = []
y_train = []
x_test = []
y_test = []

for state, images in train.items():
    for image in images:

        img = cv2.imread(str(image))
        resize = cv2.resize(img, (224, 224))
        x_train.append(resize)
        y_train.append(train_labels[state])

for state, images in test.items():
    for image in images:
        img = cv2.imread(str(image))
        resize = cv2.resize(img, (224, 224))
        x_test.append(resize)
        y_test.append(test_labels[state])


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# print(x_train[0])

x_train = x_train/255
x_test = x_test/255

model = tf.keras.Sequential([

    keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),

    keras.layers.Flatten(),

    # Dense

    keras.layers.Dense(115, activation='relu'),

    keras.layers.Dense(2)

])

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']

)

model.fit(x_train, y_train, epochs=10)

model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

y_final = []
for i in range(len(y_pred)):
    score = tf.nn.softmax(y_pred[i])
    y_final.append(np.argmax(score))

y_final = np.array(y_final)

print("10 values of y_test is :", y_test[:10])
print("10 values of y_final is :", y_final[:10])


cm = confusion_matrix(y_test, y_final)

sn.heatmap(cm, annot=True, fmt='d')
plt.show()

my_img = cv2.imread('sample.jpg')
resized = cv2.resize(my_img, (224, 224))
resized = np.array(resized)/255
print(resized.shape)

print(resized[np.newaxis, ...].shape)

predicted = model.predict(resized[np.newaxis, ...])
score = np.argmax(tf.nn.softmax(predicted))
if score == 0:
    print("Normal")
    a = "Normal"
elif score == 1:
    print("COVID")
    a = "Covid"


plt.imshow(resized)
plt.title(a)
plt.axis('off')
plt.show()