from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
import uuid
import time

app = Flask(__name__)

# Create directories for uploads and static files
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load model without loading optimizer state (to avoid warning)
model = load_model('./training/TSR.keras', compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Dictionary of class labels
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicle > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing vehicle > 3.5 tons'
}

def image_processing(img_path):
    try:
        # Load and prepare image for prediction
        image = Image.open(img_path)
        
        # Save original image dimensions for display purposes
        orig_width, orig_height = image.size
        
        # Resize for model input
        resized_image = image.resize((30, 30))
        image_array = img_to_array(resized_image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0
        
        # Make prediction
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class]) * 100
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Image processing error: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")

    try:
        # Generate a unique filename to prevent conflicts
        original_filename = secure_filename(file.filename)
        filename_parts = os.path.splitext(original_filename)
        unique_filename = f"{filename_parts[0]}_{uuid.uuid4().hex[:8]}{filename_parts[1]}"
        
        # Save file to uploads directory
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process image and get prediction
        predicted_class, confidence = image_processing(filepath)
        
        if predicted_class is None:
            return render_template('index.html', error="Error processing image")
        
        # Generate relative path for template
        image_path = f"{UPLOAD_FOLDER}/{unique_filename}"
        
        # Create result string
        result = f"Prediction: {classes[predicted_class]} (Confidence: {confidence:.2f}%)"
        
        return render_template('index.html', 
                               result=result, 
                               image_path=image_path,
                               filename=original_filename)
                               
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', error=f"An error occurred: {str(e)}")

# Clean up old files periodically (optional)
@app.before_request
def cleanup_old_files():
    try:
        # Delete files older than 1 hour
        now = time.time()
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath) and os.stat(filepath).st_mtime < now - 3600:
                os.remove(filepath)
    except Exception as e:
        print(f"Cleanup error: {e}")

if __name__ == '__main__':
    app.run(debug=True)