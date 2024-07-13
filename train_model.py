import face_recognition
import os
import pickle

def train_model():
    known_face_encodings = []
    known_face_names = []

    for user_dir in os.listdir('./faces'):
        user_path = os.path.join('./faces', user_dir)
        if os.path.isdir(user_path):
            for img_name in os.listdir(user_path):
                img_path = os.path.join(user_path, img_name)
                image = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    face_encoding = face_encodings[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(user_dir)

    with open('face_encodings.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

if __name__ == "__main__":
    train_model()
