import tensorflow as tf
from tensorflow import keras


def build_model():
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(50, activation="relu"),
            keras.layers.Dense(50, activation="relu"),
            keras.layers.Dense(10, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_model(model, X_train, Y_train, epochs=10):
    model.fit(X_train, Y_train, epochs=epochs)
    return model


def evaluate_model(model, X_test, Y_test):
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Accuracy: {accuracy}")
    return accuracy
