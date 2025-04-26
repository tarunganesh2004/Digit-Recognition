import os
import numpy as np
import cv2
from flask import Flask, request, render_template, flash, redirect, url_for
from tensorflow import keras
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key"  # Replace with a secure key
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the trained model
model = keras.models.load_model("mnist_model_updated.keras")

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(img_path):
    # Read and process the image
    input_image = cv2.imread(img_path)
    if input_image is None:
        return None, "Error: Could not load image."
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_image_resize = cv2.resize(grayscale, (28, 28))
    input_image_resize = input_image_resize / 255.0
    image_reshaped = np.reshape(input_image_resize, [1, 28, 28])
    return image_reshaped, None


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None
    processed_image_path = None  # Initialize to None

    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            flash("No file uploaded")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Process the image
            image_reshaped, error = process_image(file_path)
            if error:
                flash(error)
                return redirect(request.url)

            # Predict
            input_prediction = model.predict(image_reshaped)
            input_pred_label = np.argmax(input_prediction)

            # Save processed image for display
            processed_image = image_reshaped.reshape(28, 28) * 255
            processed_filename = "processed_" + filename
            cv2.imwrite(
                os.path.join(app.config["UPLOAD_FOLDER"], processed_filename),
                processed_image,
            )

            prediction = input_pred_label
            image_path = "uploads/" + filename
            processed_image_path = "uploads/" + processed_filename

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path,
        processed_image_path=processed_image_path,
    )


if __name__ == "__main__":
    app.run(debug=True)