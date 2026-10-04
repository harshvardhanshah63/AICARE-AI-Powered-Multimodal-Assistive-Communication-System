import cv2

for camera_index in range(5):

    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if camera.isOpened():

        print(f"Camera found at index: {camera_index}")

        ret, frame = camera.read()

        if ret:
            print(f"Camera {camera_index} is working")

        camera.release()

    else:
        print(f"Camera {camera_index} not available")