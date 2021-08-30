import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sn

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print(x_train.shape)
print(y_train.shape)

# print(x_train[:5])
# print(y_train[:5])

classes = ["T shirt/top", "Trousars", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneakers", "Bag", "Angle boot"
           ]
x_train = x_train / 255
x_test = x_test / 255

x_test_copy = x_test.copy()

x_train = x_train.reshape(len(x_train), 28 * 28)
x_test = x_test.reshape(len(x_test), 28 * 28)

y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')

model = keras.Sequential([
    # keras.layers.Flatten(input_shape =(28, 28)),
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train_categorical, epochs=100)

y_pred = model.predict(x_test)
n = []
for i in range(len(y_pred)):
    print(classes[np.argmax(y_pred[i])])
    print(classes[y_test[i]])
    print("\n")

y_final = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
y_final = np.array(y_final, dtype='uint8')

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_final)
sn.heatmap(cm, annot=True, fmt='d')
plt.show()

plt.figure(figsize=(10, 1))
plt.imshow(tf.squeeze(x_test_copy[1]))
plt.title(classes[y_final[1]])
plt.show()
