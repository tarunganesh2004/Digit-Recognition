import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


def load_data():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    print("Data loaded. Shapes:")
    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")
    return X_train, Y_train, X_test, Y_test


def preprocess_data(X_train, X_test):
    X_train, X_test = X_train / 255.0, X_test / 255.0
    return X_train, X_test


def display_sample_image(X_train, Y_train, idx=25):
    plt.imshow(X_train[idx], cmap="gray")
    plt.show()
    print(f"Label for displayed image: {Y_train[idx]}")
