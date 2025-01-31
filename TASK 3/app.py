from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import tensorflow as tf
import numpy as np
import cv2
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = './static/uploads/'
MODEL_PATH = './model/plant_disease_model (4).h5'  # Make sure this is retrained or compatible with the changes
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Limit file size to 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model    
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (update to match your model's output classes)
CLASS_NAMES = [
'Tomato___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Target_Spot', 'Corn_(maize)___Northern_Leaf_Blight', 'Peach___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___healthy', 'Apple___Black_rot', 'Tomato___Bacterial_spot', 'Tomato___Late_blight', 'Grape___healthy', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Leaf_Mold', 'Cherry_(including_sour)___healthy', 'Apple___Cedar_apple_rust', 'Tomato___Tomato_mosaic_virus', 'Grape___Black_rot', 'Peach___Bacterial_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Corn_(maize)___Common_rust_', 'Raspberry___healthy', 'Potato___Late_blight', 'Pepper,_bell___healthy', 'Apple___Apple_scab', 'Squash___Powdery_mildew', 'Pepper,_bell___Bacterial_spot', 'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy', 'Corn_(maize)___healthy', 'Tomato___Septoria_leaf_spot', 'Tomato___Early_blight', 'Cherry_(including_sour)___Powdery_mildew', 'Grape___Esca_(Black_Measles)', 'Strawberry___healthy', 'Apple___healthy', 'Blueberry___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Potato___Early_blight'
]

# Helper function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess and segment the image
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Error: Unable to read image."

        # Segment leaf
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        segmented_image = cv2.bitwise_and(image, image, mask=mask)

        # Resize and normalize
        resized_image = cv2.resize(segmented_image, (150, 150))  # Updated size to match model
        normalized_image = resized_image / 255.0
        return np.expand_dims(normalized_image, axis=0), None
    except Exception as e:
        return None, f"Error during preprocessing: {str(e)}"

# Predict disease
def predict_disease(image_path):
    processed_image, error = preprocess_image(image_path)
    if error:
        return None, None, error

    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    
    # Validate the predicted index
    if predicted_class_index >= len(CLASS_NAMES):
        return None, None, "Error: Predicted index is out of range. Check your CLASS_NAMES or model output."

    confidence = np.max(predictions) * 100
    return CLASS_NAMES[predicted_class_index], confidence, None

# Classify severity
def classify_severity(confidence_score):
    if confidence_score >= 70:
        return 'Mild', 'green'
    elif 50 <= confidence_score < 70:
        return 'Moderate', 'orange'
    else:
        return 'Severe', 'red'

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/diseases')
def diseases():
    return render_template('diseases.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_class, confidence, error = predict_disease(filepath)
        if error:
            return jsonify({'error': error})

        severity, color = classify_severity(confidence)
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'severity': severity,
            'color': color,
            'image_url': url_for('static', filename='uploads/' + filename)
        })
    else:
        return jsonify({'error': 'Invalid file format'})

@app.route('/result/<filename>')
def result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    predicted_class, confidence, error = predict_disease(filepath)
    if error:
        return render_template('error.html', message=error)

    severity, color = classify_severity(confidence)
    return render_template(
        'result.html',
        filename=filename,
        predicted_class=predicted_class,
        confidence=f"{confidence:.2f}%",
        severity=severity,
        color=color
    )

if __name__ == '__main__':
    app.run(debug=True)
