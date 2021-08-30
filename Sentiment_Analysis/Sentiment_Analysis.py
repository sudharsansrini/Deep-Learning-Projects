import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import numpy as np
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding

import pandas as pd

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

x_train = train['text']
y_train = train['label']

x_test = test['text']
y_test = test['label']

print(train.shape)
print(test.shape)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print(x_train.head(5))
print(y_train.head(5))

vocab_size = 500

encoded_reviews = [one_hot(i, vocab_size) for i in x_train]

print("--------------------------- ENCODED REVIEWS ----------------------------")
print(encoded_reviews)

lengths = []
for i in range(len(encoded_reviews)):
    lengths.append(len(encoded_reviews[i]))


print("--------------------------- ENCODED REVIEWS LENGTHS----------------------------")
print(lengths)
print(max(lengths))

max_length = max(lengths)

padded_reviews = pad_sequences(encoded_reviews, maxlen=max_length, padding='post')


print("--------------------------- PADDED REVIEWS ----------------------------")
print(padded_reviews)

embeded_vector_size = 4

model = Sequential()
model.add(Embedding(vocab_size, embeded_vector_size, input_length=max_length, name='embeddings'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)

print(model.evaluate(x_train, y_train))