from flask import Flask, request, jsonify, render_template
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import requests
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

# Load the pre-trained model
model = load_model(r"C:/Users/Deekshith Patel L L/OneDrive/Desktop/crop disease detection/crop disease detection/server/models/coffee.h5")

# Define the class names
class_names = ['Miner', 'NoDisease', 'Phoma', 'Rust']

# Define the disease descriptions and fertilizer recommendations
disease_info = {
    'Miner': {
        "description": "Miner disease is caused by leaf miners, which are larvae of certain insects that burrow into leaves, creating winding trails that damage plant tissue and reduce photosynthesis. This can weaken the plant and lower crop yield.",
        "fertilizers": [
            {
                "name": "Nitrogen-Rich Fertilizer (NPK 20-10-10)",
                "usage": "Apply 2-3 kg per acre every 3-4 weeks during growing season",
                "benefits": "Promotes rapid leaf regeneration and strengthens plant defense mechanisms",
                "image": "https://example.com/nitrogen-fertilizer.jpg"
            },
            {
                "name": "Organic Fish Emulsion",
                "usage": "Dilute 2 tablespoons per gallon of water, apply bi-weekly",
                "benefits": "Provides natural nitrogen and trace minerals for healthy leaf growth",
                "image": "https://example.com/fish-emulsion.jpg"
            },
            {
                "name": "Seaweed Extract",
                "usage": "Foliar spray every 2 weeks, 1 tablespoon per liter of water",
                "benefits": "Enhances plant immunity and provides micronutrients",
                "image": "https://example.com/seaweed-extract.jpg"
            }
        ]
    },
    'NoDisease': {
        "description": "A plant categorized as having no disease is healthy and free from any visible infections or damage. To maintain its optimal growth and prevent future issues, regular maintenance is essential.",
        "fertilizers": [
            {
                "name": "Balanced NPK (20-20-20)",
                "usage": "Apply 1-2 kg per acre monthly during growing season",
                "benefits": "Provides balanced nutrition for overall plant health and growth",
                "image": "https://example.com/balanced-npk.jpg"
            },
            {
                "name": "Micronutrient Mix",
                "usage": "Apply 500g per acre every 2-3 months",
                "benefits": "Prevents deficiencies and maintains plant vigor",
                "image": "https://example.com/micronutrient.jpg"
            },
            {
                "name": "Organic Compost",
                "usage": "Apply 2-3 tons per acre annually",
                "benefits": "Improves soil structure and provides slow-release nutrients",
                "image": "https://example.com/compost.jpg"
            }
        ]
    },
    'Phoma': {
        "description": "Phoma disease is a fungal infection that causes dark spots or lesions on stems, leaves, and fruits, often leading to plant decay and reduced yields.",
        "fertilizers": [
            {
                "name": "Potassium-Rich Fertilizer (NPK 13-0-44)",
                "usage": "Apply 1.5-2 kg per acre every 4-6 weeks",
                "benefits": "Strengthens cell walls and improves disease resistance",
                "image": "https://example.com/potassium-fertilizer.jpg"
            },
            {
                "name": "Calcium Nitrate",
                "usage": "Apply 1 kg per acre monthly",
                "benefits": "Strengthens plant tissue and improves disease resistance",
                "image": "https://example.com/calcium-nitrate.jpg"
            },
            {
                "name": "Silica Supplement",
                "usage": "Foliar spray every 2 weeks, 2ml per liter of water",
                "benefits": "Enhances plant defense mechanisms against fungal attacks",
                "image": "https://example.com/silica.jpg"
            }
        ]
    },
    'Rust': {
        "description": "Rust disease is a fungal infection characterized by orange or brown pustules on leaves, which hinder photosynthesis and weaken the plant over time.",
        "fertilizers": [
            {
                "name": "Sulfur-Based Fertilizer",
                "usage": "Apply 1-1.5 kg per acre every 4-6 weeks",
                "benefits": "Creates unfavorable conditions for rust development",
                "image": "https://example.com/sulfur-fertilizer.jpg"
            },
            {
                "name": "Copper Sulfate",
                "usage": "Foliar spray every 14 days, 3g per liter of water",
                "benefits": "Prevents spore germination and controls rust spread",
                "image": "https://example.com/copper-sulfate.jpg"
            },
            {
                "name": "Zinc Sulfate",
                "usage": "Apply 500g per acre every 2 months",
                "benefits": "Improves plant immunity and stress tolerance",
                "image": "https://example.com/zinc-sulfate.jpg"
            }
        ]
    }
}

# Prediction function
def predict(model, img, class_names):
    img = img.resize((256, 256))  # Resize the image
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence

# Convert the image to base64
def convert_img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' in request.files and request.files['file']:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded file temporarily
        img_path = os.path.join('uploads', file.filename)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        file.save(img_path)

        # Open the image
        img = Image.open(img_path)

    elif 'url' in request.form and request.form['url']:
        img_url = request.form['url']
        try:
            response = requests.get(img_url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        except Exception as e:
            return jsonify({"error": f"Failed to fetch image from URL: {str(e)}"}), 400

    else:
        return jsonify({"error": "No file or URL provided"}), 400

    # Get the prediction
    predicted_class, confidence = predict(model, img, class_names)

    # Get the disease information
    disease_data = disease_info.get(predicted_class, {})
    
    # Convert the image to base64
    img_base64 = convert_img_to_base64(img)

    # Return the complete response
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': confidence,
        'description': disease_data.get('description', 'No description available.'),
        'fertilizers': disease_data.get('fertilizers', []),
        'img_base64': img_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
