from ultralytics import YOLO
import cv2
import csv
from datetime import datetime


# LOAD MODEL


model = YOLO("yolov8m.pt")


# OPEN VIDEO


cap = cv2.VideoCapture(
    "videos/counting_test.mp4"
)


# COUNTING LINE

# Positioned as a fraction of the ACTUAL video width, not a hardcoded
# absolute pixel — see intrusion_detection.py for why this matters.

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920

COUNT_LINE_X = int(frame_width * 0.55)

OFFSET = max(10, int(frame_width * 0.015))


# COUNTERS


entry_count = 0
exit_count = 0


# TRACKING


previous_positions = {}

counted_entry_ids = set()
counted_exit_ids = set()


# CSV LOG


csv_file = "logs/counting_events.csv"

with open(csv_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Timestamp",
        "Person_ID",
        "Event",
        "Current_Occupancy"
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
        conf=0.25,
        tracker="trackers/tuned_bytetrack.yaml",
        verbose=False
    )

    annotated_frame = results[0].plot()

    
    # DRAW COUNT LINE
    

    cv2.line(
        annotated_frame,
        (COUNT_LINE_X, 0),
        (COUNT_LINE_X, annotated_frame.shape[0]),
        (0, 255, 255),
        3
    )

    cv2.putText(
        annotated_frame,
        "ENTRY / EXIT",
        (COUNT_LINE_X - 100, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    boxes = results[0].boxes

    current_ids = set()

    if boxes is not None:

        for box in boxes:

            if box.id is None:
                continue

            person_id = int(box.id)

            current_ids.add(person_id)

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            cv2.circle(
                annotated_frame,
                (center_x, center_y),
                5,
                (255, 0, 0),
                -1
            )

            
            # CROSSING LOGIC
            

            if person_id in previous_positions:

                previous_x = previous_positions[
                    person_id
                ]

                # -------------------------
                # ENTRY
                # Right -> Left
                # -------------------------

                if (
                    previous_x >
                    COUNT_LINE_X + OFFSET
                    and
                    center_x <
                    COUNT_LINE_X - OFFSET
                ):

                    if person_id not in counted_entry_ids:

                        counted_entry_ids.add(
                            person_id
                        )

                        entry_count += 1

                        timestamp = datetime.now()

                        filename = (
                            f"evidence/entry_"
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
                                "ENTRY",
                                len(current_ids)
                            ])

                        print(
                            f"ENTRY : Person {person_id}"
                        )

                # -------------------------
                # EXIT
                # Left -> Right
                # -------------------------

                elif (
                    previous_x <
                    COUNT_LINE_X - OFFSET
                    and
                    center_x >
                    COUNT_LINE_X + OFFSET
                ):

                    if person_id not in counted_exit_ids:

                        counted_exit_ids.add(
                            person_id
                        )

                        exit_count += 1

                        timestamp = datetime.now()

                        filename = (
                            f"evidence/exit_"
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
                                "EXIT",
                                len(current_ids)
                            ])

                        print(
                            f"EXIT : Person {person_id}"
                        )

            previous_positions[
                person_id
            ] = center_x

    
    # OCCUPANCY
    

    current_occupancy = len(
        current_ids
    )

    
    # DISPLAY STATS
    

    cv2.rectangle(
        annotated_frame,
        (10, 10),
        (350, 180),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        annotated_frame,
        f"Entries : {entry_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Exits : {exit_count}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Occupancy : {current_occupancy}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Tracked IDs : {len(current_ids)}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    
    # DISPLAY
    

    cv2.imshow(
        "People Counting & Occupancy",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# CLEANUP


cap.release()
cv2.destroyAllWindows()