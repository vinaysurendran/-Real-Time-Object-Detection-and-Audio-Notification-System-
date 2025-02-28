# Real-Time Object Detection and Audio Notification System

This project uses the YOLOv8 model to perform real-time object detection through a webcam feed. Detected objects are annotated on the video feed, and their descriptions are converted to audio messages using Google Text-to-Speech (gTTS) and played back using Pygame.

## Requirements

- Python 3.x
- OpenCV
- gTTS
- Pygame
- Ultralytics YOLO

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/vinaysurendran/-Real-Time-Object-Detection-and-Audio-Notification-System-.git
    cd -Real-Time-Object-Detection-and-Audio-Notification-System-
    ```

2. Install the required Python packages:
    ```sh
    pip install opencv-python gtts pygame ultralytics
    ```

## Usage

1. Ensure you have the YOLOv8 model file (`yolov8n.pt`) in the same directory as [mini.py](http://_vscodecontentref_/0).
2. Run the [mini.py](http://_vscodecontentref_/1) script:
    ```sh
    python mini.py
    ```

## Code Overview

- [mini.py](http://_vscodecontentref_/2): Main script that captures video from the webcam, performs object detection using YOLOv8, and provides audio feedback for detected objects.

### Key Functions

- [calculate_relative_position(x, y, frame_width, frame_height)](http://_vscodecontentref_/3): Calculates the relative position of an object in the frame.
- [calculate_distance(box_width, frame_width)](http://_vscodecontentref_/4): Estimates the distance of an object from the camera.
- [convert_labels_to_text(labels, positions, distances)](http://_vscodecontentref_/5): Converts detected object labels, positions, and distances into a descriptive text message.

### Main Loop

1. Captures frames from the webcam.
2. Performs object detection using the YOLOv8 model.
3. Annotates detected objects on the video feed.
4. Converts detection results into an audio message and plays it.
5. Displays the annotated video feed.
6. Exits the loop when 'q' is pressed.

## Notes

- Ensure your webcam is connected and accessible.
- The class names should match the labels used by your YOLOv8 model.
- The script will generate and play an audio file (`output.mp3`) for each frame with detected objects.

## License

This project is licensed under the MIT License.