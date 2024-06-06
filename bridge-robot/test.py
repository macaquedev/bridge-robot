import cv2

def test_camera(index):
    # Open the camera
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print(f"Error opening camera with index {index}")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is Truek
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('Camera Test', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera(2)  # Change the index if you have multiple cameras