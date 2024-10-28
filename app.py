from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
from tensorflow import keras
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
model = keras.models.load_model("mnist_model.h5")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def preprocess_image(image_path):
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(input_image, (28, 28))
    normalized_image = resized_image / 255.0
    reshaped_image = np.reshape(normalized_image, (1, 28, 28))
    return reshaped_image


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            processed_image = preprocess_image(file_path)
            prediction = model.predict(processed_image)
            predicted_label = np.argmax(prediction)

            return render_template(
                "index.html", label=predicted_label, user_image=file_path
            )

    return render_template("index.html", label=None)


if __name__ == "__main__":
    app.run(debug=True)
