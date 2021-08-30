import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def loss_and_errror(y_train, y_pred):
    return np.mean(np.square(y_train, y_pred))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def gradient_descent(age, affordability, y_train, epochs):
    w1 = w2 = 0
    learning_rate = 0.5
    bias = 1

    n = len(age)

    for i in range(epochs):
        sum = w1 * age + w2 * affordability + bias
        y_pred = sigmoid(sum)

        loss = loss_and_errror(y_train, y_pred)

        w1d = (1/n) * np.dot(np.transpose(age), (y_pred - y_train))
        w2d = (1/n) * np.dot(np.transpose(affordability), (y_pred - y_train))
        bias_d = np.mean(y_pred, y_train)

        w1 = w1 - learning_rate * w1d
        w2 = w2 - learning_rate * w2d
        bias = bias - learning_rate * bias_d

        print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}')

    return w1, w2, bias





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

gradient_descent(x_train_cp['age'], x_train_cp['affordibility'], y_train, 1000)