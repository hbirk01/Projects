import cv2
import os
import mysql.connector
import pickle
import configparser
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import onnxruntime as rt

# COCO class names for YOLOv5
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", 
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
    "hair drier", "toothbrush", "phone", "marker"
]

# Load YOLOv5 model with ONNX Runtime
def load_yolo_model():
    session = rt.InferenceSession("yolov5s.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name

yolo_session, yolo_input_name, yolo_output_name = load_yolo_model()

# Perform object detection using YOLOv5
def detect_objects(frame, session, input_name, output_name, conf_threshold=0.5):
    blob = cv2.resize(frame, (640, 640)).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)
    outputs = session.run([output_name], {input_name: blob})
    objects = []
    for detection in outputs[0][0]:
        confidence = detection[4]
        if confidence > conf_threshold:
            class_id = int(detection[5])
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "Unknown"
            x, y, w, h = map(int, detection[:4])
            objects.append((class_name, confidence, (x, y, w, h)))
    return objects

# Load config for MySQL
config = configparser.ConfigParser()
config.read('config.ini')

# Connect to MySQL
def get_db_connection():
    return mysql.connector.connect(
        host=config['mysql']['host'],
        user=config['mysql']['user'],
        password=config['mysql']['password'],
        database=config['mysql']['database']
    )

# Load ResNet50 model for face recognition
def load_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return base_model

resnet_model = load_resnet_model()

# Generate face embedding
def get_face_embedding(image):
    image = cv2.resize(image, (224, 224))  
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = resnet_model.predict(img_array)
    return embedding[0]

# Store face embeddings in the database
def store_face_embedding_in_db(name, encoding):
    connection = get_db_connection()
    cursor = connection.cursor()
    encoding_blob = pickle.dumps(encoding)
    cursor.execute("INSERT INTO face_embeddings (name, embedding) VALUES (%s, %s)", (name, encoding_blob))
    connection.commit()
    cursor.close()
    connection.close()

# Load known faces and store in the database
def load_known_faces(known_faces_dir):
    known_face_encodings, known_face_names = [], []
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(known_faces_dir, filename)
            image = cv2.imread(img_path)
            embedding = get_face_embedding(image)
            name = os.path.splitext(filename)[0]
            store_face_embedding_in_db(name, embedding)
            known_face_encodings.append(embedding)
            known_face_names.append(name)
    return known_face_encodings, known_face_names

# Fetch known embeddings from the database
def fetch_face_embeddings_from_db():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT name, embedding FROM face_embeddings")
    rows = cursor.fetchall()
    known_face_encodings, known_face_names = [], []
    for name, embedding_blob in rows:
        encoding = pickle.loads(embedding_blob)
        if encoding.shape[0] == 2048: 
            known_face_encodings.append(encoding)
            known_face_names.append(name)
    cursor.close()
    connection.close()
    return known_face_encodings, known_face_names

# Compare embeddings using cosine similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1, norm2 = np.linalg.norm(embedding1), np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

# Recognize faces in the frame
def recognize_faces_in_frame(frame, known_face_encodings, known_face_names, threshold=0.5):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = rgb_frame[y:y+h, x:x+w]
        embedding = get_face_embedding(face)
        similarities = [cosine_similarity(embedding, enc) for enc in known_face_encodings]
        max_similarity = max(similarities) if similarities else 0
        if max_similarity >= threshold:
            best_match_index = np.argmax(similarities)
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

# Main function to run the application
if __name__ == "__main__":
    known_faces_dir = "./known_faces"
    known_face_encodings, known_face_names = fetch_face_embeddings_from_db()
    if not known_face_encodings:
        known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open video stream.")
        exit()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        objects = detect_objects(frame, yolo_session, yolo_input_name, yolo_output_name)
        for class_name, confidence, box in objects:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        frame = recognize_faces_in_frame(frame, known_face_encodings, known_face_names)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
