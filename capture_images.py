import cv2
import os

def capture_images(name):
    cap = cv2.VideoCapture(0)
    count = 0
    os.makedirs(f'./faces/{name}', exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture Images", frame)
        k = cv2.waitKey(0) & 0xFF  # Masking untuk mendapatkan nilai 8-bit dari k

        if k == 27:  # ESC key
            break
        elif k == 32:  # SPACE key
            img_name = f'./faces/{name}/image_{count}.jpg'
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_name = input("Enter your name: ")
    capture_images(user_name)
