from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# Load model once when app starts
model = load_model("sugarcane_disease_model.h5")

# Define class names
class_names = ['BrownRust', 'Dry', 'Healthy', 'Mawa', 'Mites', 'RedSpot', 'YellowLeaf']

@app.route("/", methods=["GET"])
def home():
    return "✅ Sugarcane Disease Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        file = request.files["file"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Required only for local testing (not needed in Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
