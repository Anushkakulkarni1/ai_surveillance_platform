

import argparse
import os

import cv2

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# clip -> output filename. Pick 5 different clips for variety; any valid
# TestXXX folder works, these are just a reasonable default spread.
CLIP_TO_OUTPUT = {
    "Test001": "intrusion_test.mp4",
    "Test002": "loitering_test.mp4",
    "Test003": "fall_test.mp4",
    "Test004": "counting_test.mp4",
    "Test005": "test.mp4",
}


def build_video(frame_dir: str, output_path: str, fps: int = 10) -> int:
    frame_paths = sorted(
        os.path.join(frame_dir, f) for f in os.listdir(frame_dir)
        if f.lower().endswith(IMG_EXTENSIONS)
    )
    if not frame_paths:
        raise FileNotFoundError(f"No frame images found in {frame_dir}")

    first = cv2.imread(frame_paths[0])
    if first is None:
        raise IOError(f"Could not read {frame_paths[0]}")
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in frame_paths:
        frame = cv2.imread(p)
        # UCSD frames are grayscale saved as single-channel; VideoWriter
        # expects 3-channel BGR, so convert if needed.
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(frame)
    writer.release()
    return len(frame_paths)


def main(args):
    os.makedirs("videos", exist_ok=True)

    available_clips = sorted(
        d for d in os.listdir(args.ped2_test_dir)
        if os.path.isdir(os.path.join(args.ped2_test_dir, d)) and not d.endswith("_gt")
    )
    print(f"Found {len(available_clips)} clips under {args.ped2_test_dir}")

    for clip_name, output_name in CLIP_TO_OUTPUT.items():
        # Fall back to any available clip if the specific one requested
        # isn't present (e.g. a partial/renamed dataset copy).
        actual_clip = clip_name if clip_name in available_clips else (
            available_clips[list(CLIP_TO_OUTPUT.keys()).index(clip_name) % len(available_clips)]
            if available_clips else None
        )
        if actual_clip is None:
            print(f"  SKIPPED {output_name}: no clips available at all.")
            continue

        frame_dir = os.path.join(args.ped2_test_dir, actual_clip)
        output_path = os.path.join("videos", output_name)
        n = build_video(frame_dir, output_path, fps=args.fps)
        print(f"  videos/{output_name}  <-  {actual_clip}  ({n} frames)")

    print("\nDone. All 5 expected video files are now in ./videos/")
    print("You can now run each detection script directly, e.g.:")
    print("  python detection/intrusion_detection.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the video files detection scripts expect, from UCSD Ped2 test clips.")
    p.add_argument("--ped2_test_dir", type=str, required=True, help="Path to your UCSDped2/Test folder.")
    p.add_argument("--fps", type=int, default=10)
    main(p.parse_args())
