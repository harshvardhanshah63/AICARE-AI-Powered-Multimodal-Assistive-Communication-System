import cv2

print("Testing camera...")

for camera_index in range(5):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if cap.isOpened():
        print(f"Camera found at index: {camera_index}")

        ret, frame = cap.read()

        if ret:
            print("Camera is working!")
            cv2.imshow(f"Camera {camera_index}", frame)
            cv2.waitKey(3000)

        cap.release()
    else:
        print(f"No camera at index: {camera_index}")

cv2.destroyAllWindows()