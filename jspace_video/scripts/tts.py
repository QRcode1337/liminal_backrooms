"""Render a narration script to per-scene WAVs with Kokoro TTS and write durations.

Usage:
  python3 scripts/tts.py <kokoro.onnx> <voices.bin>                      # long cut
  python3 scripts/tts.py <kokoro.onnx> <voices.bin> short 1.1            # 75 s short
Long cut  -> public/vo/<id>.wav,       src/vo-durations.json
Short cut -> public/vo-short/<id>.wav, src/short-durations.json
"""
import json
import sys
from pathlib import Path

import soundfile as sf
from kokoro_onnx import Kokoro

ROOT = Path(__file__).resolve().parent.parent
VOICE = "bm_george"

variant = sys.argv[3] if len(sys.argv) > 3 else "long"
speed = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
script = "narration.json" if variant == "long" else f"{variant}_narration.json"
out_dir = ROOT / "public" / ("vo" if variant == "long" else f"vo-{variant}")
dur_file = ROOT / "src" / ("vo-durations.json" if variant == "long" else f"{variant}-durations.json")
out_dir.mkdir(parents=True, exist_ok=True)

kokoro = Kokoro(sys.argv[1], sys.argv[2])
lines = json.loads((ROOT / "scripts" / script).read_text())
durations = {}
for line in lines:
    samples, sr = kokoro.create(line["text"], voice=VOICE, speed=speed, lang="en-gb")
    sf.write(out_dir / f"{line['id']}.wav", samples, sr)
    durations[line["id"]] = round(len(samples) / sr, 3)
    print(line["id"], durations[line["id"]])
dur_file.write_text(json.dumps(durations, indent=2) + "\n")
