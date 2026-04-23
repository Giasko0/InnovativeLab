#!/usr/bin/env python3
"""
Async Groq Vision -> sentence streaming -> Groq TTS -> non-blocking playback.

This module is designed for the TrashSort loop:
1) detector (YOLO11n/YOLO26n) provides a cropped image of the detected item
2) Vision model explains disposal action
3) each completed sentence is sent immediately to TTS
4) audio is played sequentially for smooth voice streaming
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import AsyncGenerator, Callable

import aiohttp
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf


SENTENCE_RE = re.compile(r"(.+?[.!?])(?:\s+|$)", re.S)


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
    vision_model: str = "meta-llama/llama-3.2-11b-vision-instruct"
    tts_model: str = "playai-tts"
    tts_voice: str = "Fritz-PlayAI"
    vision_max_tokens: int = 220
    vision_temperature: float = 0.2
    request_timeout_s: float = 45.0
    tts_concurrency: int = 1
    min_tts_interval_s: float = 0.8
    tts_retries: int = 2
    max_sentences: int = 8
    max_sentence_chars: int = 220
    max_total_chars: int = 1200
    min_sentence_chars: int = 6

    def resolved_api_key(self) -> str:
        key = self.api_key or os.getenv("GROQ_API_KEY")
        if not key:
            load_env_file(Path(__file__).with_name(".env"))
            key = self.api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Missing Groq API key. Set GROQ_API_KEY or pass api_key in config.")
        return key


class RequestPacer:
    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def wait_turn(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait_s = self._next_allowed - now
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            self._next_allowed = time.monotonic() + self.min_interval_s


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
    pacer: RequestPacer,
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

    attempt = 0
    while True:
        await pacer.wait_turn()
        try:
            timeout = aiohttp.ClientTimeout(total=config.request_timeout_s)
            async with session.post(url, headers=headers, json=payload, timeout=timeout) as response:
                if response.status == 429 and attempt < config.tts_retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    attempt += 1
                    continue
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(f"TTS request failed ({response.status}): {body[:600]}")
                audio = await response.read()
                if not audio:
                    raise RuntimeError("TTS request returned empty audio")
                return audio
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt >= config.tts_retries:
                raise
            await asyncio.sleep(1.0 * (attempt + 1))
            attempt += 1


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
    on_sentence: Callable[[str], None] | None = None,
) -> str:
    """
    Run Groq Vision->TTS streaming on one detector crop.

    Args:
        crop_image: YOLO crop (OpenCV numpy array BGR) or image bytes/path.
        prompt: user prompt sent alongside the crop.
        config: GroqStreamingConfig (API throttling + model settings).
        on_sentence: optional callback fired when each complete sentence is formed.

    Returns:
        Full concatenated model text used for spoken output.
    """
    load_env_file(Path(__file__).with_name(".env"))
    cfg = config or GroqStreamingConfig(
        api_key=os.getenv("GROQ_API_KEY"),
        vision_model=os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-3.2-11b-vision-instruct"),
        tts_model=os.getenv("GROQ_TTS_MODEL", "playai-tts"),
        tts_voice=os.getenv("GROQ_TTS_VOICE", "Fritz-PlayAI"),
    )
    image_data_url = encode_crop_to_data_url(crop_image)

    sentence_queue: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue()
    audio_queue: asyncio.Queue[tuple[int, bytes] | None] = asyncio.Queue()
    pacer = RequestPacer(cfg.min_tts_interval_s)
    tts_workers = 1

    async def tts_worker() -> None:
        try:
            while True:
                item = await sentence_queue.get()
                if item is None:
                    await audio_queue.put(None)
                    return
                idx, sentence = item
                wav = await request_tts_wav(session, sentence, cfg, pacer)
                await audio_queue.put((idx, wav))
        except Exception:
            await audio_queue.put(None)
            raise

    async def audio_worker() -> None:
        next_idx = 0
        pending: dict[int, bytes] = {}
        while True:
            item = await audio_queue.get()
            if item is None:
                break

            idx, wav_bytes = item
            pending[idx] = wav_bytes
            while next_idx in pending:
                to_play = pending.pop(next_idx)
                await play_wav_non_blocking(to_play)
                next_idx += 1

    full_text_parts: list[str] = []
    sent_count = 0
    total_chars = 0

    async with aiohttp.ClientSession() as session:
        tts_tasks = [asyncio.create_task(tts_worker())]
        audio_task = asyncio.create_task(audio_worker())
        try:
            token_stream = stream_vision_tokens(
                session,
                image_data_url=image_data_url,
                prompt=prompt,
                config=cfg,
            )
            async for sentence in stream_complete_sentences(
                token_stream,
                min_chars=cfg.min_sentence_chars,
            ):
                if sent_count >= cfg.max_sentences or total_chars >= cfg.max_total_chars:
                    break

                sentence = sentence[: cfg.max_sentence_chars].strip()
                if not sentence:
                    continue

                full_text_parts.append(sentence)
                total_chars += len(sentence)
                sent_count += 1

                if on_sentence is not None:
                    on_sentence(sentence)

                await sentence_queue.put((sent_count - 1, sentence))
        finally:
            await sentence_queue.put(None)
            tts_results = await asyncio.gather(*tts_tasks, return_exceptions=True)
            tts_error = next((r for r in tts_results if isinstance(r, Exception)), None)
            if tts_error is not None:
                audio_task.cancel()
                await asyncio.gather(audio_task, return_exceptions=True)
                raise tts_error
            await audio_task

    return " ".join(full_text_parts).strip()


async def _example() -> None:
    # Example: pass crop from detector loop
    crop = cv2.imread("sample_crop.jpg")
    if crop is None:
        raise RuntimeError("sample_crop.jpg not found")

    spoken_text = await stream_crop_vision_to_tts(
        crop,
        prompt=(
            "Identify the item in this crop and explain how to dispose of it in Hong Kong. "
            "Keep the answer short and practical."
        ),
        config=GroqStreamingConfig(),
    )
    print(spoken_text)


if __name__ == "__main__":
    load_env_file(Path(__file__).with_name(".env"))
    asyncio.run(_example())
