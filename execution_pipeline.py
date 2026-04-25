#!/usr/bin/env python3
"""
Async Groq Vision -> sentence streaming -> Groq TTS -> non-blocking playback.

CLI usage:
- python execution_pipeline.py
- optional: python execution_pipeline.py --weights datasets/taco_hk_yolo26/runs/train_py/weights/best.pt --source 0

Requirements:
- GROQ_API_KEY in environment or .env file next to this script
- trained YOLO weights (default: datasets/taco_hk_yolo26/runs/train_py/weights/best.pt)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import AsyncGenerator

import aiohttp
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf


SENTENCE_RE = re.compile(r"(.+?[.!?])(?:\s+|$)", re.S)
RULES_PATH = Path(__file__).with_name("hk_recycling_rules.json")
DEFAULT_WEIGHTS_PATH = Path("datasets/taco_hk_yolo26/runs/train_py/weights/best.pt")
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720
DEFAULT_REPORT_PATH = Path("inference_report.json")
BIAS_LABELS = {"some sort of general waste item", "Corrugated carton"}
MIN_CONF_BY_LABEL = {
    "some sort of general waste item": 0.55,
    "Corrugated carton": 0.40,
}
NON_BIAS_MIN_CONF = 0.20
NON_BIAS_MARGIN = 0.08


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(slots=True)
class GroqStreamingConfig:
    api_key: str | None = None
    base_url: str = "https://api.groq.com/openai/v1"
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    tts_model: str = "canopylabs/orpheus-v1-english"
    tts_voice: str = "daniel"
    vision_max_tokens: int = 96
    vision_temperature: float = 0.1
    request_timeout_s: float = 45.0
    min_tts_interval_s: float = 0.2
    max_sentence_chars: int = 180
    min_sentence_chars: int = 6

    def resolved_api_key(self) -> str:
        key = self.api_key or os.getenv("GROQ_API_KEY")
        if not key:
            load_env_file(Path(__file__).with_name(".env"))
            key = self.api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Missing Groq API key. Set GROQ_API_KEY or pass api_key in config.")
        return key


@dataclass(slots=True)
class InferenceReport:
    frames: int = 0
    detections: int = 0
    speeches: int = 0
    runtime_s: float = 0.0
    frame_capture_ms: float = 0.0
    yolo_ms: float = 0.0
    postprocess_ms: float = 0.0
    speech_first_ms: float = 0.0
    speech_total_ms: float = 0.0
    loop_ms: float = 0.0
    loop_ms_max: float = 0.0

    def add_frame(
        self,
        *,
        capture_ms: float,
        yolo_ms: float,
        postprocess_ms: float,
        loop_ms: float,
        has_detection: bool,
    ) -> None:
        self.frames += 1
        self.frame_capture_ms += capture_ms
        self.yolo_ms += yolo_ms
        self.postprocess_ms += postprocess_ms
        self.loop_ms += loop_ms
        self.loop_ms_max = max(self.loop_ms_max, loop_ms)
        if has_detection:
            self.detections += 1

    def add_speech(self, *, first_sentence_ms: float | None, total_ms: float) -> None:
        self.speeches += 1
        self.speech_total_ms += total_ms
        if first_sentence_ms is not None:
            self.speech_first_ms += first_sentence_ms

    def as_dict(self, *, weights: Path, rules_path: Path, source: int) -> dict[str, object]:
        def avg(total: float, count: int) -> float:
            return total / count if count else 0.0

        return {
            "weights": str(weights),
            "rules_path": str(rules_path),
            "source": source,
            "frames": self.frames,
            "detections": self.detections,
            "speech_events": self.speeches,
            "runtime_s": self.runtime_s,
            "camera_loop_fps": (self.frames / self.runtime_s) if self.runtime_s > 0 else 0.0,
            "frame_capture_ms_avg": avg(self.frame_capture_ms, self.frames),
            "yolo_ms_avg": avg(self.yolo_ms, self.frames),
            "postprocess_ms_avg": avg(self.postprocess_ms, self.frames),
            "loop_ms_avg": avg(self.loop_ms, self.frames),
            "loop_ms_max": self.loop_ms_max,
            "speech_first_ms_avg": avg(self.speech_first_ms, self.speeches),
            "speech_total_ms_avg": avg(self.speech_total_ms, self.speeches),
        }


def load_recycling_rules(path: str | Path = RULES_PATH) -> dict[str, dict[str, str]]:
    rules_path = Path(path)
    if not rules_path.exists():
        raise FileNotFoundError(f"Missing recycling rules file: {rules_path}")
    return json.loads(rules_path.read_text(encoding="utf-8"))


def build_recycling_prompt(item_label: str, rules: dict[str, dict[str, str]]) -> str:
    rule = rules.get(item_label, rules["some sort of general waste item"])
    return (
        "You are TrashSort, a Hong Kong recycling assistant.\n"
        "Be natural, helpful, concise, and biased toward safe Hong Kong disposal.\n"
        "Treat the detected item label as fixed and do not contradict it.\n"
        "Never say the item is not the detected label, and never replace it with another item.\n"
        "When uncertain, prefer general waste or a designated collection point.\n"
        f"Detected item label: {item_label}\n"
        f"HK rule: throw it in {rule['bin']}.\n"
        f"Where: {rule['where']}.\n"
        f"Rule note: {rule['note']}\n"
        "Reply in one short, natural sentence of at most 18 words.\n"
        "Do not use a fixed opening or template."
    )


def encode_crop_to_data_url(crop: np.ndarray | bytes | str | Path) -> str:
    if isinstance(crop, np.ndarray):
        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise RuntimeError("Failed to JPEG-encode crop image")
        image_bytes = encoded.tobytes()
        mime = "image/jpeg"
    else:
        if isinstance(crop, (str, Path)):
            image_bytes = Path(crop).read_bytes()
        else:
            image_bytes = bytes(crop)
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif image_bytes.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        else:
            mime = "application/octet-stream"

    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


async def stream_vision_tokens(
    session: aiohttp.ClientSession,
    *,
    image_data_url: str,
    prompt: str,
    config: GroqStreamingConfig,
) -> AsyncGenerator[str, None]:
    api_key = config.resolved_api_key()
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": config.vision_model,
        "temperature": config.vision_temperature,
        "max_tokens": config.vision_max_tokens,
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are TrashSort, a concise recycling assistant for Hong Kong."
                    " Explain disposal decisions in short, practical sentences."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    timeout = aiohttp.ClientTimeout(total=config.request_timeout_s)
    async with session.post(url, headers=headers, json=payload, timeout=timeout) as response:
        if response.status != 200:
            body = await response.text()
            raise RuntimeError(f"Vision request failed ({response.status}): {body[:600]}")

        while True:
            raw_line = await response.content.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices", [])
            if not choices:
                continue
            token = choices[0].get("delta", {}).get("content")
            if token:
                yield token


async def stream_complete_sentences(
    token_stream: AsyncGenerator[str, None],
    *,
    min_chars: int,
) -> AsyncGenerator[str, None]:
    buffer = ""
    async for token in token_stream:
        buffer += token
        while True:
            match = SENTENCE_RE.search(buffer)
            if not match:
                break
            sentence = " ".join(match.group(1).split()).strip()
            buffer = buffer[match.end() :].lstrip()
            if len(sentence) >= min_chars:
                yield sentence

    tail = " ".join(buffer.split()).strip()
    if tail:
        yield tail


async def request_tts_wav(
    session: aiohttp.ClientSession,
    sentence: str,
    config: GroqStreamingConfig,
) -> bytes:
    api_key = config.resolved_api_key()
    url = f"{config.base_url.rstrip('/')}/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.tts_model,
        "voice": config.tts_voice,
        "input": sentence,
        "response_format": "wav",
    }

    timeout = aiohttp.ClientTimeout(total=config.request_timeout_s)
    async with session.post(url, headers=headers, json=payload, timeout=timeout) as response:
        if response.status >= 400:
            body = await response.text()
            raise RuntimeError(f"TTS request failed ({response.status}): {body[:600]}")
        audio = await response.read()
        if not audio:
            raise RuntimeError("TTS request returned empty audio")
        return audio


async def play_wav_non_blocking(wav_bytes: bytes) -> None:
    # sounddevice.play(..., blocking=False) starts playback immediately and returns.
    # We await sd.wait() in a worker task so the event loop stays responsive.
    data, samplerate = sf.read(BytesIO(wav_bytes), dtype="float32")
    if data.ndim == 1:
        data = data[:, np.newaxis]
    sd.play(data, samplerate, blocking=False)
    await asyncio.to_thread(sd.wait)


async def stream_crop_vision_to_tts(
    crop_image: np.ndarray | bytes | str | Path,
    *,
    prompt: str,
    config: GroqStreamingConfig | None = None,
    timings: dict[str, float] | None = None,
) -> str:
    """
    Run Groq Vision->TTS streaming on one detector crop.

    Args:
        crop_image: YOLO crop (OpenCV numpy array BGR) or image bytes/path.
        prompt: user prompt sent alongside the crop.
        config: GroqStreamingConfig (API throttling + model settings).
    Returns:
        Full concatenated model text used for spoken output.
    """
    load_env_file(Path(__file__).with_name(".env"))
    cfg = config or GroqStreamingConfig(
        api_key=os.getenv("GROQ_API_KEY"),
        vision_model=os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        tts_model=os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english"),
        tts_voice=os.getenv("GROQ_TTS_VOICE", "daniel"),
    )
    image_data_url = encode_crop_to_data_url(crop_image)

    full_text_parts: list[str] = []
    start = time.perf_counter()
    first_sentence_recorded = False

    async with aiohttp.ClientSession() as session:
        token_stream = stream_vision_tokens(
            session,
            image_data_url=image_data_url,
            prompt=prompt,
            config=cfg,
        )
        async for sentence in stream_complete_sentences(token_stream, min_chars=cfg.min_sentence_chars):
            sentence = sentence[: cfg.max_sentence_chars].strip()
            if not sentence:
                continue

            full_text_parts.append(sentence)

            wav_bytes = await request_tts_wav(session, sentence, cfg)
            await play_wav_non_blocking(wav_bytes)
            if timings is not None and not first_sentence_recorded:
                timings["first_sentence_ms"] = (time.perf_counter() - start) * 1000.0
                first_sentence_recorded = True
            if cfg.min_tts_interval_s > 0:
                await asyncio.sleep(cfg.min_tts_interval_s)
    if timings is not None:
        timings["total_ms"] = (time.perf_counter() - start) * 1000.0
    return " ".join(full_text_parts).strip()


def pick_detection(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    order = boxes.conf.argsort(descending=True).tolist()
    candidates: list[tuple[str, float, np.ndarray]] = []
    for idx in order:
        box = boxes[idx]
        label = result.names[int(box.cls.item())]
        conf = float(box.conf.item())
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        candidates.append((label, conf, xyxy))

    top_label, top_conf, top_xyxy = candidates[0]
    top_threshold = MIN_CONF_BY_LABEL.get(top_label, 0.0)

    if top_label in BIAS_LABELS:
        for label, conf, xyxy in candidates[1:]:
            if label in BIAS_LABELS:
                continue
            if conf >= max(NON_BIAS_MIN_CONF, top_conf - NON_BIAS_MARGIN):
                return label, conf, xyxy
        if top_conf < top_threshold:
            return None

    if top_conf < MIN_CONF_BY_LABEL.get(top_label, NON_BIAS_MIN_CONF):
        return None
    return top_label, top_conf, top_xyxy


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


def draw_status_bar(display, status_text, stable_count):
    colors = {
        "Scanning": (34, 139, 34),
        "Thinking": (0, 140, 255),
        "Ready": (180, 180, 0),
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


def wrap_text(text: str, max_width: int, font_scale: float = 0.6, thickness: int = 2) -> list[str]:
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_detection_label(display, label, conf):
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
    if not text:
        return
    max_width = display.shape[1] - 20
    lines = wrap_text(text, max_width=max_width, font_scale=0.6, thickness=2)[:3]
    if not lines:
        return

    _, line_height = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    line_height += 10
    padding = 10
    box_height = padding * 2 + line_height * len(lines)
    top = max(0, display.shape[0] - box_height)
    cv2.rectangle(display, (0, top), (display.shape[1], display.shape[0]), (0, 0, 0), -1)

    y = top + padding + line_height - 4
    for line in lines:
        cv2.putText(
            display,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 220, 60),
            2,
        )
        y += line_height


def speak_item(crop, label, rules, *, timings: dict[str, float] | None = None):
    prompt = build_recycling_prompt(label, rules)
    return asyncio.run(stream_crop_vision_to_tts(crop, prompt=prompt, timings=timings))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TrashSort webcam inference (YOLO + Groq Vision + Groq TTS)"
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--source", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--stable-frames", type=int, default=5, help="Frames before speaking")
    parser.add_argument("--cooldown", type=float, default=3.5, help="Minimum seconds between speeches")
    parser.add_argument("--window-width", type=int, default=DEFAULT_WINDOW_WIDTH, help="Initial inference window width")
    parser.add_argument("--window-height", type=int, default=DEFAULT_WINDOW_HEIGHT, help="Initial inference window height")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="Where to save the timing report")
    return parser.parse_args()


def run_inference(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for inference. Install with: pip install -U ultralytics"
        ) from exc

    weights = Path(os.getenv("TRASHSORT_WEIGHTS", str(args.weights))).resolve()
    if not weights.exists():
        raise FileNotFoundError(
            f"Missing weights file: {weights}. Run training_pipeline.py first or set TRASHSORT_WEIGHTS."
        )

    model = YOLO(str(weights))
    rules = load_recycling_rules(RULES_PATH)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam source {args.source}")

    window_name = "TrashSort Inference (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)

    last_spoken = 0.0
    last_label = None
    stable_count = 0
    spoken_text = ""
    speaking = False
    status_text = "Scanning"
    report = InferenceReport()
    report_lock = threading.Lock()
    run_start = time.perf_counter()

    def speak_async(crop, label):
        nonlocal spoken_text, speaking, last_spoken, status_text
        speech_timings: dict[str, float] = {}
        speech_start = time.perf_counter()
        try:
            spoken_text = speak_item(crop, label, rules, timings=speech_timings)
        except RuntimeError as exc:
            spoken_text = f"Inference/TTS error: {exc}"
        finally:
            last_spoken = time.time()
            speaking = False
            status_text = "Ready"
            with report_lock:
                report.add_speech(
                    first_sentence_ms=speech_timings.get("first_sentence_ms"),
                    total_ms=(speech_timings.get("total_ms") or ((time.perf_counter() - speech_start) * 1000.0)),
                )

    try:
        while True:
            loop_start = time.perf_counter()
            capture_start = loop_start
            ok, frame = cap.read()
            capture_ms = (time.perf_counter() - capture_start) * 1000.0
            if not ok:
                break

            yolo_start = time.perf_counter()
            result = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            yolo_ms = (time.perf_counter() - yolo_start) * 1000.0

            post_start = time.perf_counter()
            det = pick_detection(result)
            display = result.plot()
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
                if (
                    stable_count >= args.stable_frames
                    and now - last_spoken > args.cooldown
                    and not speaking
                ):
                    crop = crop_frame(frame, xyxy)
                    if crop is not None:
                        speaking = True
                        status_text = "Thinking"
                        threading.Thread(target=speak_async, args=(crop, label), daemon=True).start()

            draw_spoken_text(display, spoken_text)
            postprocess_ms = (time.perf_counter() - post_start) * 1000.0
            cv2.imshow(window_name, display)
            should_exit = (cv2.waitKey(1) & 0xFF) == ord("q")
            with report_lock:
                report.add_frame(
                    capture_ms=capture_ms,
                    yolo_ms=yolo_ms,
                    postprocess_ms=postprocess_ms,
                    loop_ms=(time.perf_counter() - loop_start) * 1000.0,
                    has_detection=det is not None,
                )
            if should_exit:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        report.runtime_s = time.perf_counter() - run_start
        summary = report.as_dict(weights=weights, rules_path=RULES_PATH, source=args.source)
        report_path = args.report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("\nInference report")
        print(f"Output: {report_path}")
        print(f"Frames: {summary['frames']} | Detections: {summary['detections']} | Speeches: {summary['speech_events']}")
        print(
            f"Frame capture avg: {summary['frame_capture_ms_avg']:.1f} ms | "
            f"YOLO avg: {summary['yolo_ms_avg']:.1f} ms | "
            f"Post avg: {summary['postprocess_ms_avg']:.1f} ms"
        )
        print(
            f"Speech first avg: {summary['speech_first_ms_avg']:.1f} ms | "
            f"Speech total avg: {summary['speech_total_ms_avg']:.1f} ms"
        )
        print(f"Loop FPS: {summary['camera_loop_fps']:.2f} | Loop max: {summary['loop_ms_max']:.1f} ms")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
