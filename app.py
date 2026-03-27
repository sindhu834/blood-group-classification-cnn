from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
print("Loading trained model...")
try:
    model = keras.models.load_model('blood_group_classifier.h5')
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Blood group classes
BLOOD_GROUPS = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            return None
        
        # Resize to model input size
        img = cv2.resize(img, (180, 180))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and make prediction"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Use JPG, PNG, or BMP'}), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        processed_img = preprocess_image(filepath)
        
        if processed_img is None:
            return jsonify({'error': 'Could not process image'}), 400
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        blood_group = BLOOD_GROUPS[predicted_class_idx]
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'blood_group': blood_group,
            'confidence': round(confidence, 4),
            'confidence_percentage': round(confidence * 100, 2),
            'all_predictions': {BLOOD_GROUPS[i]: round(float(predictions[0][i]), 4) 
                               for i in range(len(BLOOD_GROUPS))}
        })
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'blood_groups': BLOOD_GROUPS
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Blood Group Classification Web App")
    print("="*60)
    print(f"Model Status: {'✓ Loaded' if model is not None else '❌ Not Found'}")
    print(f"Blood Groups: {', '.join(BLOOD_GROUPS)}")
    print("\nStarting Flask server...")
    print("Go to: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)