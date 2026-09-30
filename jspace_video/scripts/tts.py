"""Render narration.json to per-scene WAVs with Kokoro TTS and write durations.

Usage: python3 scripts/tts.py <kokoro.onnx> <voices.bin>
Outputs public/vo/<id>.wav and src/vo-durations.json (seconds).
"""
import json
import sys
from pathlib import Path

import soundfile as sf
from kokoro_onnx import Kokoro

ROOT = Path(__file__).resolve().parent.parent
VOICE = "bm_george"
SPEED = 1.0

kokoro = Kokoro(sys.argv[1], sys.argv[2])
lines = json.loads((ROOT / "scripts" / "narration.json").read_text())
durations = {}
for line in lines:
    samples, sr = kokoro.create(line["text"], voice=VOICE, speed=SPEED, lang="en-gb")
    sf.write(ROOT / "public" / "vo" / f"{line['id']}.wav", samples, sr)
    durations[line["id"]] = round(len(samples) / sr, 3)
    print(line["id"], durations[line["id"]])
(ROOT / "src" / "vo-durations.json").write_text(json.dumps(durations, indent=2) + "\n")
