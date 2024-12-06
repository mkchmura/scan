import cv2
import time
from flask import Flask, Response, render_template
from PIL import Image
from transformers import pipeline

app = Flask(__name__)

# Initialize the object detection pipeline
object_detector = pipeline('object-detection', model='facebook/detr-resnet-50', device=0)

# URL of the live camera feed
url = 'https://wzmedia.dot.ca.gov/D8/LB-8_18_125.stream/playlist.m3u8'
#url = 'animals.mp4'
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Cannot open video stream")
    exit()

# Function to process and stream the video
def generate_frames():
    frame_counter = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps))  # Process 1 frame per second

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # Process every nth frame to reduce load
        if frame_counter % frame_interval == 0:
            # Convert OpenCV frame (BGR) to PIL Image (RGB)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = object_detector(image)

            # Draw detections on the frame
            for result in results:
                if result["score"] > 0.90:
                    box = result["box"]
                    label = result["label"]
                    score = result["score"]
                    cv2.rectangle(frame, 
                                  (int(box["xmin"]), int(box["ymin"])), 
                                  (int(box["xmax"]), int(box["ymax"])), 
                                  (0, 255, 0), 2)
                    cv2.putText(frame, 
                                f"{label} ({score:.2f})", 
                                (int(box["xmin"]), int(box["ymin"]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def live_feed():
    """Endpoint to serve the live video stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
