from ultralytics import YOLO
import cv2
import numpy as np


# LOAD MODEL


model = YOLO("yolov8m.pt")


# VIDEO


cap = cv2.VideoCapture(
    "videos/counting_test.mp4"
)


# GET VIDEO SIZE


width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# HEATMAP CANVAS


heatmap = np.zeros(
    (height, width),
    dtype=np.float32
)


# PROCESS VIDEO


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

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            center_x = (
                x1 + x2
            ) // 2

            center_y = (
                y1 + y2
            ) // 2

            cv2.circle(
                heatmap,
                (center_x, center_y),
                25,
                1,
                -1
            )


# NORMALIZE


heatmap = cv2.GaussianBlur(
    heatmap,
    (0, 0),
    sigmaX=25
)

heatmap = cv2.normalize(
    heatmap,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

heatmap = np.uint8(
    heatmap
)


# APPLY COLORS


colored_heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)


# SAVE


output_path = (
    "outputs/heatmap.png"
)

cv2.imwrite(
    output_path,
    colored_heatmap
)

print(
    f"Heatmap saved to {output_path}"
)

cap.release()