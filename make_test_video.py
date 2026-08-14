
import cv2, os


frame_dir = "datasets/UCSDped2/Test/Test002"   # any TestXXX folder works
frames = sorted(os.path.join(frame_dir, f) for f in os.listdir(frame_dir))
first = cv2.imread(frames[0])
h, w = first.shape[:2]

writer = cv2.VideoWriter("test_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
for f in frames:
    writer.write(cv2.imread(f))
writer.release()
print("Done: test_video.mp4")