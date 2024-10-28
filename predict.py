import numpy as np
import cv2
from PIL import Image
from google.colab.patches import cv2_imshow 


def preprocess_custom_image(image_path):
    input_image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(grayscale, (28, 28)) / 255.0
    reshaped = np.reshape(resized, (1, 28, 28))
    return reshaped


def predict_custom_image(model, processed_image):
    prediction = model.predict(processed_image)
    pred_label = np.argmax(prediction)
    print(f"The Handwritten Digit is recognized as: {pred_label}")
    return pred_label
