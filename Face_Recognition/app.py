# app.py
from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
from face_recognition_app import FaceRecognitionModel

app = Flask(__name__)

# Load the face recognition model (replace with your model path)
model = FaceRecognitionModel('face_recognition_model.py')

# Directory where known faces (JPGs) are stored
KNOWN_FACES_DIR = 'known_faces/'

def load_known_faces():
    """ Load known face images from the directory and return their embeddings. """
    known_embeddings = {}
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith('.jpg'):
            # Extract the name from the filename (e.g., "person_a.jpg" -> "person_a")
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            
            # Use the model to extract the embedding of the image
            image = cv2.imread(image_path)
            embedding = model.get_embedding(image)
            known_embeddings[name] = embedding
    return known_embeddings

@app.route('/recognize', methods=['POST'])
def recognize_faces():
    # Get the uploaded image from the request
    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Load the known faces from JPG images
    known_embeddings = load_known_faces()

    # Recognize faces using the model
    recognized_faces = model.recognize_faces(image, known_embeddings)

    # Return the recognized faces in JSON format
    return jsonify({
        'recognized': recognized_faces
    })

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
