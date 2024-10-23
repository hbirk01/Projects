import face_recognition
import cv2
import os
import mysql.connector
import pickle
import configparser

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

# Store face embeddings in the database
def store_face_embedding_in_db(name, encoding):
    connection = get_db_connection()
    cursor = connection.cursor()
    encoding_blob = pickle.dumps(encoding)
    cursor.execute("INSERT INTO face_embeddings (name, embedding) VALUES (%s, %s)", (name, encoding_blob))
    connection.commit()
    cursor.close()
    connection.close()

# Fetch known embeddings
def fetch_face_embeddings_from_db():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT name, embedding FROM face_embeddings")
    rows = cursor.fetchall()

    known_face_encodings = []
    known_face_names = []
    for name, embedding_blob in rows:
        encoding = pickle.loads(embedding_blob)
        known_face_encodings.append(encoding)
        known_face_names.append(name)

    cursor.close()
    connection.close()
    return known_face_encodings, known_face_names

# Load known faces from directory and store embeddings in DB
def load_known_faces(known_faces_dir, store_in_db=True):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)

                if store_in_db:
                    store_face_embedding_in_db(name, encoding)
            else:
                print(f"Face encodings not found in image {filename}")

    return known_face_encodings, known_face_names

# Capture new embeddings based on side profile
def capture_side_profile(frame, name):
    encodings = face_recognition.face_encodings(frame)
    if encodings:
        encoding = encodings[0]
        store_face_embedding_in_db(name, encoding)
        print(f"Captured side profile for {name}")

# Track faces across frames
def track_face(frame, bbox, tracker):
    ok, bbox = tracker.update(frame)
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    return frame, bbox

def recognize_faces_in_frame(frame, known_face_encodings, known_face_names, model='hog', threshold=0.6):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model=model)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=threshold)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if face_distances.size > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # Start tracker to capture side profile later
                tracker = cv2.TrackerKCF_create()
                bbox = (left, top, right - left, bottom - top)
                tracker.init(frame, bbox)

                # Draw box around recognized face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                # Capture side profile when they look away
                if not matches[best_match_index]:
                    capture_side_profile(frame, name)

    return frame

if __name__ == "__main__":
    known_faces_dir = "./known_faces"

    # Fetch known faces from the database or load from directory
    known_face_encodings, known_face_names = fetch_face_embeddings_from_db()

    # If no embeddings are found, load from directory and store in DB
    if not known_face_encodings:
        print("No embeddings found in the database. Loading from directory and storing...")
        known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open video stream.")
        exit()

    tracker = None  # Initialize tracker for side profiles
    bbox = None  # Bounding box for face tracking

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        if tracker and bbox:
            frame, bbox = track_face(frame, bbox, tracker)

        frame_with_boxes = recognize_faces_in_frame(frame, known_face_encodings, known_face_names)

        cv2.imshow('Video', frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
