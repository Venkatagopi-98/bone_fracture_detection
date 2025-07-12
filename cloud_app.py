from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='fracture_model.tflite')
interpreter.allocate_tensors()

# Get tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels (adjust according to training)
class_labels = [
    'avulsion fracture', 'Comminuted fracture', 'Fracture Dislocation',
    'Greenstick fracture', 'Hairline Fracture', 'Impacted fracture',
    'Longitudinal fracture', 'Oblique fracture', 'Pathological fracture', 'Spiral Fracture'
]

# Preprocess image
def preprocess(img_bytes):
    image = Image.open(io.BytesIO(img_bytes)).resize((200, 200)).convert('RGB')
    img_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_input = preprocess(file.read())

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction_idx = int(np.argmax(output_data[0]))
    confidence = float(output_data[0][prediction_idx]) * 100

    return jsonify({
        'predicted_class': class_labels[prediction_idx],
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
