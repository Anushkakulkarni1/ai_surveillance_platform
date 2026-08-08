from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8m.pt")

# Open video
video_path = "videos/test.mp4"

cap = cv2.VideoCapture(video_path)

# Get video information
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Save output video
out = cv2.VideoWriter(
    "outputs/output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run AI detection
    results = model(frame, classes=[0])

    # Draw boxes
    annotated_frame = results[0].plot()

    # Show live window
    cv2.imshow("Person Detection", annotated_frame)

    # Save output
    out.write(annotated_frame)

    # Quit when q pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()