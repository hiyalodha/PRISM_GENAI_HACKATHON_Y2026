from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent

UTTERANCES = {
    "find_flights_tokyo": "Find flights from London to Tokyo on Wednesday.",
    "make_it_thursday": "Actually, make it Thursday instead.",
}

FRAMES = {
    "router_x200": {
        "label": "X200",
        "led": (220, 30, 30),
        "blur": False,
        "sidecar": {
            "description": "A white home router with the model label X200 on the front; the power LED is blinking red.",
            "confidence": 0.92,
            "entities": {"device_model": "X200", "light_color": "red", "light_state": "blinking"},
        },
    },
    "router_blurry": {
        "label": "X200",
        "led": (220, 30, 30),
        "blur": True,
        "sidecar": {
            "description": "A blurry photo of what may be a router with a red light; no label is readable.",
            "confidence": 0.3,
            "entities": {},
        },
    },
}


def write_tone(path: Path, seconds: float = 1.0, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(int(seconds * rate)))
        w.writeframes(frames)


def make_audio() -> None:
    have_say = shutil.which("say") is not None
    for name, text in UTTERANCES.items():
        wav = HERE / f"{name}.wav"
        (HERE / f"{name}.wav.txt").write_text(text + "\n")
        if have_say:
            subprocess.run(["say", "-o", str(wav), "--data-format=LEI16@16000", text], check=True)
        else:
            write_tone(wav)
            print(f"'say' not available; wrote a placeholder tone to {wav.name}", file=sys.stderr)


def make_frames() -> None:
    try:
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError:
        print("Pillow not installed; skipping frame generation", file=sys.stderr)
        return
    for name, spec in FRAMES.items():
        img = Image.new("RGB", (640, 480), (40, 44, 52))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((120, 160, 520, 360), radius=24, fill=(235, 235, 235), outline=(180, 180, 180), width=4)
        for i in range(3):
            d.line((180 + i * 140, 160, 160 + i * 150, 60), fill=(200, 200, 200), width=8)
        d.ellipse((160, 300, 190, 330), fill=spec["led"])
        d.ellipse((210, 300, 240, 330), fill=(60, 60, 60))
        d.text((360, 300), spec["label"], fill=(20, 20, 20))
        d.text((160, 200), "HOME ROUTER", fill=(90, 90, 90))
        if spec["blur"]:
            img = img.filter(ImageFilter.GaussianBlur(14))
        png = HERE / f"{name}.png"
        img.save(png)
        (HERE / f"{name}.png.json").write_text(json.dumps(spec["sidecar"], indent=2) + "\n")


if __name__ == "__main__":
    make_audio()
    make_frames()
    print("assets written to", HERE)
