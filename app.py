from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = load_model("sugarcane_disease_model.h5")

# Define class names in same order as training
class_names = ['BrownRust', 'Dry', 'Healthy', 'Mawa', 'Mites', 'RedSpot', 'YellowLeaf']

@app.route("/", methods=["GET"])
def home():
    return "Sugarcane Disease Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
