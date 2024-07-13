import cv2
import face_recognition
import pickle
import numpy as np
import dlib
import os

def detect_faces():
    # Load trained model
    with open('face_encodings.pkl', 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)

    cap = cv2.VideoCapture(0)
    previous_frame = None

    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.isfile(predictor_path):
        raise RuntimeError(f"Unable to open {predictor_path}. Please ensure the file exists in the correct path.")
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    def detect_blink(landmarks):
        left_eye_ratio = (abs(landmarks[37][1] - landmarks[41][1]) + abs(landmarks[38][1] - landmarks[40][1])) / (2 * abs(landmarks[36][0] - landmarks[39][0]))
        right_eye_ratio = (abs(landmarks[43][1] - landmarks[47][1]) + abs(landmarks[44][1] - landmarks[46][1])) / (2 * abs(landmarks[42][0] - landmarks[45][0]))
        return left_eye_ratio < 0.2 and right_eye_ratio < 0.2

    def calculate_motion_score(previous_frame, current_frame):
        diff_frame = cv2.absdiff(previous_frame, current_frame)
        gray_diff = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
        _, thresh_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        motion_score = np.sum(thresh_diff)
        return motion_score

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw rectangle around face
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            # Detect facial landmarks for blink detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dlib_rect = dlib.rectangle(left, top, right, bottom)
            landmarks = predictor(gray_frame, dlib_rect)
            landmarks = [(p.x, p.y) for p in landmarks.parts()]

            if detect_blink(landmarks):
                cv2.putText(frame, "Blink Detected", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Check for motion (basic spoof detection)
        if previous_frame is not None:
            motion_score = calculate_motion_score(previous_frame, frame)
            print(motion_score)
            if motion_score > 0 and motion_score < 5000:  # Tweak threshold as needed
                cv2.putText(frame, "Spoofing Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if motion_score == 0:
                cv2.putText(frame, "Unknown User",  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        previous_frame = frame.copy()

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces()
