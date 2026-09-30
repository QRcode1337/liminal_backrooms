"""Synthesize a low ambient drone bed (public/drone.wav) long enough for the video."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR = 44100
seconds = sum(json.loads((ROOT / "src" / "vo-durations.json").read_text()).values()) + 30
t = np.arange(int(SR * seconds)) / SR
wobble = 0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t)
sig = (
    0.5 * np.sin(2 * np.pi * 55 * t)
    + 0.3 * np.sin(2 * np.pi * 82.4 * t + 0.3 * np.sin(2 * np.pi * 0.1 * t))
    + 0.15 * wobble * np.sin(2 * np.pi * 110.3 * t)
    + 0.05 * np.sin(2 * np.pi * 60 * t)  # mains hum, backrooms fluorescent vibe
)
rng = np.random.default_rng(0)
noise = np.convolve(rng.standard_normal(len(t)), np.ones(200) / 200, mode="same")
sig = sig + 0.4 * noise
fade = np.minimum(1, np.minimum(t / 3, (seconds - t) / 3))
sig = 0.3 * sig / np.abs(sig).max() * fade
sf.write(ROOT / "public" / "drone.wav", sig.astype(np.float32), SR)
print(f"drone.wav {seconds:.1f}s")
