from ultralytics import YOLO
import cv2
import csv
from datetime import datetime


# LOAD MODEL


model = YOLO("yolov8m.pt")


# OPEN VIDEO


cap = cv2.VideoCapture("videos/intrusion_test.mp4")

print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# ZONES

# Computed as a fraction of the ACTUAL video resolution, not hardcoded
# absolute pixels. Hardcoded coordinates like (1200, 150, 1800, 850) only
# make sense for a ~1920px-wide feed — they silently never trigger on
# smaller footage (e.g. UCSD Ped2's 360x240 frames), since no detection
# center can ever fall inside a zone that starts past the frame's edge.

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

ZONE_A = (
    int(frame_width * 0.55), int(frame_height * 0.15),
    int(frame_width * 0.95), int(frame_height * 0.90),
)
ZONE_B = (
    int(frame_width * 0.05), int(frame_height * 0.15),
    int(frame_width * 0.45), int(frame_height * 0.90),
)


# LOGGING SETUP


alerted_ids = set()

csv_file = "logs/events.csv"

with open(csv_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Timestamp",
        "Person_ID",
        "Event",
        "Zone"
    ])


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

    annotated_frame = results[0].plot()

    
    # DRAW ZONE A
    

    cv2.rectangle(
        annotated_frame,
        (ZONE_A[0], ZONE_A[1]),
        (ZONE_A[2], ZONE_A[3]),
        (0, 0, 255),
        5
    )

    cv2.putText(
        annotated_frame,
        "ZONE A",
        (ZONE_A[0], ZONE_A[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    
    # DRAW ZONE B
    

    cv2.rectangle(
        annotated_frame,
        (ZONE_B[0], ZONE_B[1]),
        (ZONE_B[2], ZONE_B[3]),
        (255, 0, 0),
        5
    )

    cv2.putText(
        annotated_frame,
        "ZONE B",
        (ZONE_B[0], ZONE_B[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        3
    )

    
    # PERSON DETECTION
    

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            if box.id is None:
                continue

            person_id = int(box.id)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            cv2.circle(
                annotated_frame,
                (center_x, center_y),
                5,
                (255, 0, 0),
                -1
            )

            
            # CHECK ZONES
            

            inside_zone_a = (
                ZONE_A[0] < center_x < ZONE_A[2]
                and
                ZONE_A[1] < center_y < ZONE_A[3]
            )

            inside_zone_b = (
                ZONE_B[0] < center_x < ZONE_B[2]
                and
                ZONE_B[1] < center_y < ZONE_B[3]
            )

            zone_name = None

            if inside_zone_a:
                zone_name = "ZONE_A"

            elif inside_zone_b:
                zone_name = "ZONE_B"

            
            # ALERT
            

            if zone_name and person_id not in alerted_ids:

                alerted_ids.add(person_id)

                timestamp = datetime.now()

                filename = (
                    f"evidence/intrusion_{person_id}_"
                    f"{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                )

                cv2.imwrite(
                    filename,
                    annotated_frame
                )

                with open(csv_file, "a", newline="") as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        timestamp,
                        person_id,
                        "Intrusion",
                        zone_name
                    ])

                cv2.putText(
                    annotated_frame,
                    f"ALERT! ID {person_id}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                print(
                    f"ALERT: Person {person_id} entered {zone_name}"
                )

    
    # DISPLAY
    

    cv2.imshow(
        "Intrusion Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# CLEANUP


cap.release()
cv2.destroyAllWindows()