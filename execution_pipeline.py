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
    vision_max_tokens: int = 96
    vision_temperature: float = 0.2
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


def load_recycling_rules(path: str | Path = RULES_PATH) -> dict[str, dict[str, str]]:
    rules_path = Path(path)
    if not rules_path.exists():
        raise FileNotFoundError(f"Missing recycling rules file: {rules_path}")
    return json.loads(rules_path.read_text(encoding="utf-8"))


def build_recycling_prompt(item_label: str, rules: dict[str, dict[str, str]]) -> str:
    rule = rules.get(item_label, rules["some sort of general waste item"])
    return (
        "You are TrashSort, a Hong Kong recycling assistant.\n"
        f"Detected item label: {item_label}\n"
        f"HK rule: throw it in {rule['bin']}.\n"
        f"Where: {rule['where']}.\n"
        f"Rule note: {rule['note']}\n"
        "Reply in one short, natural sentence of at most 15 words.\n"
        "Start with: 'Looks like you are showing me a ...,' and end with the disposal advice.\n"
        "Do not add extra explanations."
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
        vision_model=os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-3.2-11b-vision-instruct"),
        tts_model=os.getenv("GROQ_TTS_MODEL", "playai-tts"),
        tts_voice=os.getenv("GROQ_TTS_VOICE", "Fritz-PlayAI"),
    )
    image_data_url = encode_crop_to_data_url(crop_image)

    full_text_parts: list[str] = []

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
            if cfg.min_tts_interval_s > 0:
                await asyncio.sleep(cfg.min_tts_interval_s)
    return " ".join(full_text_parts).strip()
