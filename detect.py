import cv2
import mediapipe as mp

# Inisialisasi MediaPipe dan OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk menghitung jumlah jari yang terangkat
def count_fingers(hand_landmarks):
    # Daftar titik referensi untuk setiap jari
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    # Hitung jumlah jari yang terangkat
    count = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            count += 1
    return count

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip gambar untuk tampilan seperti cermin dan konversi ke RGB
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Proses gambar untuk deteksi tangan
        results = hands.process(image_rgb)
        
        # Gambar anotasi tangan di gambar
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Hitung jumlah jari yang terangkat
                finger_count = count_fingers(hand_landmarks)
                
                # Tambahkan ibu jari sebagai jari terangkat jika di atas sendi pangkal ibu jari
                if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                    finger_count += 1
                
                # Tampilkan jumlah jari yang terangkat
                cv2.putText(image, str(finger_count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                
                # Cetak jumlah jari yang terangkat ke console
                print(f"Detected fingers: {finger_count}")
        
        # Tampilkan gambar
        cv2.imshow('MediaPipe Hands', image)
        
        if cv2.waitKey(5) & 0xFF == 27:  # Tekan ESC untuk keluar
            break

cap.release()
cv2.destroyAllWindows()
