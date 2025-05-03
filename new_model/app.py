import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps, ImageChops

# Load trained model
model = load_model("model/digit_classifier1.h5")

st.title("📷 Digit Recognizer from Image Upload")
st.write("Upload an image of a digit (ideally on white background).")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])


def preprocess_image(image: Image.Image):
    # Convert to grayscale
    img = image.convert("L")

    # Invert if background is dark
    mean_intensity = np.array(img).mean()
    if mean_intensity < 127:
        img = ImageOps.invert(img)

    # Crop non-empty border
    img = ImageOps.invert(img)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img = ImageOps.invert(img)

    # Resize and center
    img = img.resize((20, 20))
    final_img = Image.new("L", (28, 28), 0)
    final_img.paste(img, (4, 4))

    # Normalize
    arr = np.array(final_img).astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1), final_img


if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=150)

    processed_tensor, processed_vis = preprocess_image(img)

    if st.button("Predict"):
        prediction = model.predict(processed_tensor)
        predicted_digit = np.argmax(prediction)
        confidence = 100 * np.max(prediction)

        st.subheader(f"Prediction: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.image(
            processed_vis.resize((140, 140)),
            caption="Preprocessed Input",
            use_column_width=False,
        )
        st.bar_chart(prediction[0])
