# -*- coding: utf-8 -*-
"""
Webcam inference script using ultralytics YOLO model (best.pt).
Runs a persistent OpenCV window showing detections from the webcam.
Press 'q' to quit.
"""

import cv2
import time
import sys
from ultralytics import YOLO


def main(source=0, weights='best.pt', conf=0.15, imgsz=640):
    """Run webcam inference and display a persistent window."""
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"Failed to load model '{weights}': {e}")
        return

    print(f"Loaded model '{weights}'. Total classes: {len(model.names)}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open webcam (index 0). Check your camera and try again.")
        return

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam")
                break

            # Run inference on the frame (model accepts numpy arrays)
            results = model(frame, conf=conf, imgsz=imgsz)

            # Annotate frame with detections
            out_frame = frame
            for r in results:
                out_frame = r.plot()  # returns annotated image

            # Compute and display FPS for the inference loop
            fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0.0
            cv2.putText(out_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("TrashSort - Webcam (press 'q' to quit)", out_frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Exiting.")


if __name__ == '__main__':
    # Allow optional command-line args: python untitled1.py [camera_index] [weights]
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    weights = sys.argv[2] if len(sys.argv) > 2 else 'best.pt'
    main(source=cam, weights=weights)
