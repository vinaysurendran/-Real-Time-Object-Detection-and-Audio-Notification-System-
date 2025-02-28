import cv2
from gtts import gTTS
import pygame
from ultralytics import YOLO
import os
from math import atan2, pi

# Initialize Pygame mixer
pygame.mixer.init()

# Define class names (this should match your model's labels)
class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
               "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
               "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
               "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
               "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"]

# Initialize YOLO model
model = YOLO('yolov8n.pt')

def calculate_relative_position(x, y, frame_width, frame_height):
    center_x, center_y = frame_width / 2, frame_height / 2
    dx = x - center_x
    dy = y - center_y
    angle = (atan2(dy, dx) * 180 / pi + 360) % 360
    hour = int((angle + 15) / 30) % 12
    if hour == 0:
        hour = 12
    return f"{hour} o'clock"

def calculate_distance(box_width, frame_width):
    return max(1, 10 - (box_width / frame_width) * 10)

def convert_labels_to_text(labels, positions, distances):
    descriptions = []
    for label, position, distance in zip(labels, positions, distances):
        descriptions.append(f"{label} detected at {position}, {distance} feet away.")
    return " ".join(descriptions)

# Main loop for video capture and processing
cap = cv2.VideoCapture(0)
wCam, hCam = 720, 1280
cap.set(3, wCam)
cap.set(4, hCam)

while True:
    try:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Get YOLO prediction
        results = model(frame)

        # Extract and annotate detected classes
        detected_classes = []
        positions = []
        distances = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls.item())
                detected_classes.append(class_names[cls])
                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                conf = box.conf.item()
                box_width = x2 - x1
                position = calculate_relative_position((x1 + x2) / 2, (y1 + y2) / 2, wCam, hCam)
                distance = calculate_distance(box_width, wCam)
                positions.append(position)
                distances.append(distance)
                label = f'{class_names[cls]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert detected labels to detailed audio message
        message = convert_labels_to_text(detected_classes, positions, distances)

        # Generate and play speech from text
        try:
            tts_engine = gTTS(text=message, lang='en', slow=False)
            audio_file = "output.mp3"
            tts_engine.save(audio_file)
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Error generating or playing audio: {e}")

        # Display the annotated frame with the message
        cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLOv8 Inference", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
if os.path.exists(audio_file):
    os.remove(audio_file)
