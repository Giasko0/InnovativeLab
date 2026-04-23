#!/usr/bin/env python3
"""
Simple webcam demo for TrashSort.

Flow:
- webcam frame
- YOLO detection
- local HK rule lookup
- Groq Vision sentence
- Groq TTS audio
"""

from __future__ import annotations

import asyncio
import time
import threading
from pathlib import Path

import cv2
from ultralytics import YOLO

from execution_pipeline import build_recycling_prompt, load_recycling_rules, stream_crop_vision_to_tts


def pick_detection(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    idx = int(boxes.conf.argmax().item())
    box = boxes[idx]
    xyxy = box.xyxy[0].cpu().numpy().astype(int)
    label = result.names[int(box.cls.item())]
    conf = float(box.conf.item())
    return label, conf, xyxy


def crop_frame(frame, xyxy):
    x1, y1, x2, y2 = xyxy
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def speak_item(crop, label, rules):
    prompt = build_recycling_prompt(label, rules)
    return asyncio.run(stream_crop_vision_to_tts(crop, prompt=prompt))


def main(weights="Code/train5/weights/best.pt", source=0, conf=0.35):
    model = YOLO(weights)
    rules = load_recycling_rules(Path(__file__).with_name("hk_recycling_rules.json"))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    last_spoken = 0.0
    last_label = None
    stable_count = 0
    spoken_text = ""
    speaking = False

    def speak_async(crop, label):
        nonlocal spoken_text, speaking, last_spoken
        try:
            spoken_text = speak_item(crop, label, rules)
        finally:
            last_spoken = time.time()
            speaking = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model(frame, conf=conf, imgsz=640, verbose=False)[0]
            det = pick_detection(result)
            display = result.plot()

            if det is not None:
                label, det_conf, xyxy = det
                if label == last_label:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_label = label

                cv2.putText(
                    display,
                    f"{label} {det_conf:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

                now = time.time()
                if stable_count >= 3 and now - last_spoken > 3.0 and not speaking:
                    crop = crop_frame(frame, xyxy)
                    if crop is not None:
                        speaking = True
                        threading.Thread(target=speak_async, args=(crop, label), daemon=True).start()

            if spoken_text:
                cv2.putText(
                    display,
                    spoken_text[:90],
                    (10, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

            cv2.imshow("TrashSort Demo", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
