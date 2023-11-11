import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
import serial

plastic_category_map = {
    'plastic bag': 'Plastic',
    'plastic bottle': 'Plastic',
    'plastic container': 'Plastic',
    'plastic cup': 'Plastic',
    'plastic straw': 'Plastic',
    'plastic utensil': 'Plastic',
}

# # Open serial connection to Arduino
# arduino_port = '/dev/ttyUSB0'  # Replace with the appropriate port for your Arduino
# ser = serial.Serial(arduino_port, baudrate=115200)

plastic = torch.hub.load('ultralytics/yolov5', 'custom', path='model/weights/best.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plastic = plastic.to(device).eval()

confidence_threshold = 0.8
cap = cv2.VideoCapture(2)

# Set camera parameters (adjust as needed)
cap.set(3, 640)  # Set frame width
cap.set(4, 480)  # Set frame height

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame using OpenCV
        frame = cv2.resize(frame, (640, 480))

        # Perform inference with the YOLOv5 model
        results_plastic = plastic(frame)
        detections = results_plastic.xyxy[0].cpu().numpy()
        confidence_model = detections[:, 4]

        for detection in detections:
            x1, y1, x2, y2, confidence, class_index = detection

            if confidence > confidence_threshold:
                class_name = results_plastic.names[int(class_index)]

                if class_name in plastic_category_map:
                    bounding_box_data = f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}"
                    # Send bounding box data to Arduino (optimize as needed)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Object Detection', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
