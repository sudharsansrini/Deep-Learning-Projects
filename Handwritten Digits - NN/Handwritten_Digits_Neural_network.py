import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train/255
x_test = x_test/255

x_train_flattend = x_train.reshape(len(x_train), 28*28)
x_test_flattend = x_test.reshape(len(x_test), 28*28)

print(x_train_flattend.shape)
print(x_test_flattend.shape)

# print(x_train)
print(y_train)

model = tf.keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation = 'relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(x_train_flattend, y_train, epochs=50)
model.evaluate(x_test_flattend, y_test)

y_pred = model.predict(x_test_flattend)

n = 121
print(np.argmax(y_pred[n]))

plt.matshow(x_test[n])
plt.show()

y_pred_labels = [np.argmax(i) for i in y_pred]
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_labels)
plt.figure(figsize=(10, 10))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


