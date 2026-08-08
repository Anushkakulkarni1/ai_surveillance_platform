from ultralytics import YOLO
import cv2
import csv
import time
from datetime import datetime


# LOAD POSE MODEL


model = YOLO("yolov8n-pose.pt")


# OPEN VIDEO


cap = cv2.VideoCapture(
    "videos/fall_test.mp4"
)


# SETTINGS


FALL_TIME = 3


# DATA STRUCTURES


horizontal_start = {}

alerted_ids = set()


# CSV LOG


csv_file = "logs/fall_events.csv"

with open(csv_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Timestamp",
        "Person_ID",
        "Event"
    ])


# MAIN LOOP


# FRAME-BASED TIMING

# See loitering_detection.py for why frame-count/fps is used instead of
# wall-clock time — it correctly reflects video-time duration regardless
# of how fast the file is actually processed.

video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        conf=0.4,
        classes=[0],
        tracker="trackers/tuned_bytetrack.yaml",
        verbose=False
    )

    annotated_frame = results[0].plot()

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            if box.id is None:
                continue

            person_id = int(box.id)

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            width = x2 - x1
            height = y2 - y1

            # FALL CHECK
           

            horizontal = (
                width > height
            )

            if horizontal:

                if person_id not in horizontal_start:

                    horizontal_start[
                        person_id
                    ] = frame_number

                elapsed = (
                    (frame_number - horizontal_start[person_id])
                    / video_fps
                )

                cv2.putText(
                    annotated_frame,
                    f"Horizontal: {int(elapsed)}s",
                    (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                if (
                    elapsed >= FALL_TIME
                    and
                    person_id not in alerted_ids
                ):

                    alerted_ids.add(
                        person_id
                    )

                    timestamp = datetime.now()

                    filename = (
                        f"evidence/fall_"
                        f"{person_id}_"
                        f"{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                    )

                    cv2.imwrite(
                        filename,
                        annotated_frame
                    )

                    with open(
                        csv_file,
                        "a",
                        newline=""
                    ) as file:

                        writer = csv.writer(
                            file
                        )

                        writer.writerow([
                            timestamp,
                            person_id,
                            "Fall Detected"
                        ])

                    cv2.putText(
                        annotated_frame,
                        f"FALL ALERT ID {person_id}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                    print(
                        f"FALL ALERT : "
                        f"Person {person_id}"
                    )

            else:

                if person_id in horizontal_start:

                    del horizontal_start[
                        person_id
                    ]

   
    # DISPLAY
   

    cv2.imshow(
        "Fall Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# CLEANUP


cap.release()
cv2.destroyAllWindows()