import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import resize

plastic_category_map = {
    'plastic bag': 'Plastic',
    'plastic bottle': 'Plastic',
    'plastic container': 'Plastic',
    'plastic cup': 'Plastic',
    'plastic straw': 'Plastic',
    'plastic utensil': 'Plastic',
}

plastic = torch.hub.load('ultralytics/yolov5', 'custom', path='model/weights/best.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plastic = plastic.to(device).eval()

confidence_threshold = 0.8
image_dir = 'Plastic-Waste-Management-4/test/images'
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path).convert('RGB')

    # Assuming you want to resize each image to a fixed size (e.g., 600x600)
    new_height, new_width = 600, 600
    image = resize(image, (new_height, new_width))
    frame = np.array(image)

    # Perform inference with the YOLOv5 model
    results_plastic = plastic(frame)
    detections = results_plastic.xyxy[0].cpu().numpy()
    confidence_model = detections[:, 4]

    for detection in detections:
        x1, y1, x2, y2, confidence, class_index = detection

        if confidence > confidence_threshold:
            class_name = results_plastic.names[int(class_index)]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()
