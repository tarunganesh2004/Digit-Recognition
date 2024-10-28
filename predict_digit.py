import numpy as np
import cv2
from tensorflow import keras
from google.colab.patches import cv2_imshow  # Use only if running in Google Colab


def load_model(model_path="mnist_model.h5"):
    """Load the pre-trained Keras model from the specified file path."""
    model = keras.models.load_model(model_path)
    print("Model loaded successfully.")
    return model


def preprocess_image(image_path):
    """
    Preprocess the image to the required input format:
    - Convert to grayscale
    - Resize to 28x28 pixels
    - Normalize pixel values to the range [0, 1]
    """
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    resized_image = cv2.resize(input_image, (28, 28))  # Resize to 28x28
    normalized_image = resized_image / 255.0  # Normalize pixel values
    reshaped_image = np.reshape(normalized_image, (1, 28, 28))  # Reshape for the model
    return reshaped_image


def predict_digit(model, processed_image):
    """Predict the digit from the processed image using the loaded model."""
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)
    print(f"The model predicts the digit is: {predicted_label}")
    return predicted_label


# Example Usage
if __name__ == "__main__":
    model = load_model("mnist_model.h5")  
    image_path = "/test.png" 
    processed_image = preprocess_image(image_path)  
    cv2_imshow(cv2.imread(image_path))  
    predict_digit(model, processed_image)  
