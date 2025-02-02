import cv2


def crop_head(image_path, output_path=None, expansion=0.1):
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Read the input image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        # Expand the bounding box by 5%
        x1 = max(0, x - int(w * expansion))  # Expand left
        y1 = max(0, y - int(h * expansion))  # Expand upward
        x2 = min(image.shape[1], x + w + int(w * expansion))  # Expand right
        y2 = min(image.shape[0], y + h + int(h * expansion))  # Expand downward

        # Crop the head region
        head = image[y1:y2, x1:x2]

        # Save or return the cropped head
        if output_path:
            cv2.imwrite(output_path, head)
        return head

    print("No face detected!")
    return None


# Example usage
cropped_head = crop_head(
    "C:\\Users\\msria\\OneDrive\\Pictures\\young.jpg",
    "cropped_head.jpg",
    expansion=0.05,
)
