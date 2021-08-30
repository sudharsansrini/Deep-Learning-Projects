import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('insurance_data.csv')

print(dataset.shape)
# print(dataset.head(5))

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=25)

x_train_cp = x_train.copy()
x_test_cp = x_test.copy()
x_train_cp['age'] = x_train_cp['age']/100
x_test_cp['age'] = x_test_cp['age']/100

print(x_train.shape)
print(y_train.shape)
model = keras.Sequential([

    keras.layers.Dense(10, input_shape=(2,), activation='relu', kernel_initializer='ones',
                       bias_initializer='zeros'),
    keras.layers.Dense(1, activation='sigmoid')

])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train_cp, y_train, epochs=1000)

model.evaluate(x_test_cp, y_test)
