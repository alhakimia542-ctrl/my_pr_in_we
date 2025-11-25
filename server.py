from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "final_skin_model.keras"  # تأكد أن الملف موجود في نفس المجلد

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        img = Image.open(file).resize((128, 128))
        img = np.array(img)/255.0
        img = img.reshape(1, 128, 128, 3)

        pred = model.predict(img)[0][0]
        label = "خلايا سلاطانية" if pred >= 0.5 else "طبيعي"

        return jsonify({"label": label, "score": float(pred)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
