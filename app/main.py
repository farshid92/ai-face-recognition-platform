import cv2
from camera.camera import Camera

def main():
    camera = Camera()

    while True:

        frame = camera.read()

        cv2.imshow(
            "AI Face Recognition Platform",
            frame
        )

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()