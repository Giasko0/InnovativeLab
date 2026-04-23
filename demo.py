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
import os
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


def draw_status_bar(display, status_text, stable_count):
    """Draw the top status bar — overlay first, then text on top."""
    colors = {
        "Scanning": (34, 139, 34),   # green
        "Thinking": (0, 140, 255),   # orange
        "Ready":    (180, 180, 0),   # gold
    }
    color = colors.get(status_text, (80, 80, 80))
    cv2.rectangle(display, (0, 0), (display.shape[1], 50), color, -1)
    cv2.putText(
        display,
        f"TrashSort  |  {status_text}  |  stable={stable_count}",
        (10, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )


def draw_detection_label(display, label, conf):
    """Draw label just below the status bar so it's never hidden by the overlay."""
    cv2.putText(
        display,
        f"{label}  {conf:.2f}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 80),
        2,
    )


def draw_spoken_text(display, text):
    """Draw the last spoken sentence at the bottom of the frame."""
    if not text:
        return
    y = display.shape[0] - 16
    # dark background strip for readability
    cv2.rectangle(display, (0, y - 26), (display.shape[1], y + 6), (0, 0, 0), -1)
    cv2.putText(
        display,
        text[:90],
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 220, 60),
        2,
    )


def main(weights="Code/train5/weights/best.pt", source=0, conf=0.35):
    weights = os.getenv("TRASHSORT_WEIGHTS", weights)
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
    status_text = "Scanning"

    def speak_async(crop, label):
        nonlocal spoken_text, speaking, last_spoken, status_text
        try:
            spoken_text = speak_item(crop, label, rules)
        except Exception:
            spoken_text = ""
        finally:
            last_spoken = time.time()
            speaking = False
            status_text = "Ready"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model(frame, conf=conf, imgsz=640, verbose=False)[0]
            det = pick_detection(result)
            display = result.plot()

            # Draw status bar first (it's the background), then text on top
            draw_status_bar(display, status_text, stable_count)

            if det is not None:
                label, det_conf, xyxy = det
                if label == last_label:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_label = label

                draw_detection_label(display, label, det_conf)

                now = time.time()
                if stable_count >= 5 and now - last_spoken > 3.5 and not speaking:
                    crop = crop_frame(frame, xyxy)
                    if crop is not None:
                        speaking = True
                        status_text = "Thinking"
                        threading.Thread(target=speak_async, args=(crop, label), daemon=True).start()

            draw_spoken_text(display, spoken_text)

            cv2.imshow("TrashSort Demo", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
