from ultralytics import YOLO
import cv2
import csv
from datetime import datetime


# LOAD MODEL

model = YOLO("yolov8m.pt")


# VIDEO

video_path = "videos/counting_test.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

print(f"FPS: {fps}")


# CSV SETUP

csv_file = "logs/behavior_analytics.csv"

with open(csv_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Timestamp",
        "Person_ID",
        "Dwell_Time_Seconds"
    ])


# TRACKING DATA


frame_counts = {}

# MAIN LOOP


while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.4,
        tracker="trackers/tuned_bytetrack.yaml",
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            if box.id is None:
                continue

            person_id = int(box.id)

            if person_id not in frame_counts:

                frame_counts[person_id] = 0

            frame_counts[person_id] += 1

    cv2.imshow(
        "Behavior Analytics",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# SAVE RESULTS


with open(csv_file, "a", newline="") as file:

    writer = csv.writer(file)

    for person_id, total_frames in frame_counts.items():

        dwell_time = (
            total_frames / fps
        )

        writer.writerow([
            datetime.now(),
            person_id,
            round(dwell_time, 2)
        ])

        print(
            f"Person {person_id} stayed "
            f"{round(dwell_time, 2)} seconds"
        )


# CLEANUP


cap.release()

cv2.destroyAllWindows()

print(
    f"Behavior analytics saved to {csv_file}"
)