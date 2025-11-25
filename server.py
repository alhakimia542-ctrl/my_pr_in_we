from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# -----------------------------
# ضع هنا مسار النموذج على سطح المكتب
# مثال: "C:/Users/YourName/Desktop/final_skin_model.keras"
MODEL_PATH = r"C:/Users/Owner/Desktop/final_skin_model.keras"
# -----------------------------

# تحميل النموذج
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    try:
        # فتح الصورة وتحويلها إلى مصفوفة NumPy
        img = Image.open(file).resize((128, 128))
        img = np.array(img)/255.0
        img = img.reshape(1, 128, 128, 3)
        
        # التنبؤ بالنموذج
        pred = model.predict(img)[0][0]
        label = "خلايا سلاطانية" if pred >= 0.5 else "طبيعي"
        
        return jsonify({"label": label, "score": float(pred)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # لجعل السيرفر متاح للآخرين على Render، نستخدم host="0.0.0.0"
    app.run(host="0.0.0.0", port=5000, debug=True)
