import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.math import confusion_matrix


def make_predictions(model, X_test):
    Y_pred = model.predict(X_test)
    Y_pred_labels = [np.argmax(i) for i in Y_pred]
    return Y_pred_labels


def plot_confusion_matrix(Y_test, Y_pred_labels):
    conf_mat = confusion_matrix(Y_test, Y_pred_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.show()
