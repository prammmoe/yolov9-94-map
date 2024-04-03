import cv2
import math
from ultralytics import YOLO
import torch  # Assuming you're using PyTorch (modify if not)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Model path (replace with your actual path)
model_path = "yolo-Weights/yolov9n.pt"
model = YOLO(model_path)

# Object classes
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Define video writer parameters
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 compatibility
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))  # Adjust output filename and frame rate

while True:
    success, img = cap.read()

    if not success:  # Handle failed frame reading
        print("Error reading frame from webcam. Exiting...")
        break

    # Object detection
    results = model(img)

    # Process detection results
    for r in results:
        boxes = r.boxes  # Assuming results contain boxes

        for box in boxes:
            # Option A (using torch.tensor.item())
            try:
                x1, y1, x2, y2 = box.xyxy[0].cpu().item() for box in boxes
            except AttributeError:  # Handle potential errors during box access
                print("Error accessing box coordinates. Skipping...")
                continue

            # Option B (using direct conversion)
            # x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])

            # Bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence and class name
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cv2.putText(img, f"{classNames[cls]} {confidence:.2f}%", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Write frame to video
    out.write(img)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
