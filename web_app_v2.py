import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Load TFLite model
MODEL_PATH = "mobilenetv2_balanced.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ["Cescospora Leaf Spot", "Golden Mosaic", "Healthy Leaf"]

# Disease Information Dictionary
DISEASE_INFO = {
    "Cescospora Leaf Spot": {
        "description": "Cercospora leaf spot is a fungal disease that causes circular to oval spots with gray centers and dark brown borders on leaves. It can lead to significant defoliation and yield loss if left untreated.",
        "remedies": [
            "Remove and destroy infected plant debris.",
            "Apply fungicides containing copper or chlorothalonil.",
            "Ensure proper spacing between plants to improve air circulation.",
            "Rotate crops to prevent disease buildup in the soil."
        ]
    },
    "Golden Mosaic": {
        "description": "Golden Mosaic is a viral disease characterized by bright yellow mosaic patterns on the leaves. It is often transmitted by whiteflies and can stunt plant growth.",
        "remedies": [
            "Control whitefly populations using sticky traps or insecticides.",
            "Remove and destroy infected plants immediately to prevent spread.",
            "Use resistant plant varieties if available.",
            "Keep the field weed-free as weeds can host the virus."
        ]
    },
    "Healthy Leaf": {
        "description": "The leaf appears healthy with no visible signs of disease or distress. Keep up the good work!",
        "remedies": [
            "Continue regular watering and fertilization.",
            "Monitor for early signs of pests or diseases.",
            "Maintain good field hygiene."
        ]
    }
}

def predict_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_class_index = np.argmax(preds)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(preds[predicted_class_index] * 100)
        
        return predicted_class_name, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0.0

@app.route('/')
def index():
    # Rendering the new v2 template
    return render_template('Update index_v2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        prediction, confidence = predict_image(file.read())
        if prediction:
            # Fetch disease info
            info = DISEASE_INFO.get(prediction, {
                "description": "No description available.",
                "remedies": []
            })
            
            return jsonify({
                'class': prediction,
                'confidence': confidence,
                'description': info['description'],
                'remedies': info['remedies']
            })
        else:
            return jsonify({'error': 'Prediction failed'})

if __name__ == '__main__':
    app.run(debug=True, port=5001) # Running on port 5001 to avoid conflict if 5000 is busy
