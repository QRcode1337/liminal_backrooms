"""Synthesize the short cut's beat bed (public/beat.wav) and SFX (public/sfx/*.wav)."""
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR = 44100
rng = np.random.default_rng(7)
(ROOT / "public" / "sfx").mkdir(parents=True, exist_ok=True)


def t(sec):
    return np.arange(int(SR * sec)) / SR


def env(n, attack=0.002, decay=0.2):
    x = np.arange(n) / SR
    return np.minimum(1, x / attack) * np.exp(-x / decay)


def lowpass(x, k):
    return np.convolve(x, np.ones(k) / k, mode="same")


def norm(x, peak=0.9):
    return (peak * x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def kick(sec=0.35):
    tt = t(sec)
    f = 45 + 110 * np.exp(-tt * 28)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(len(tt), 0.001, 0.12)


def noise(sec, decay, smooth=1):
    n = rng.standard_normal(int(SR * sec))
    if smooth > 1:
        n = lowpass(n, smooth)
    return n * env(len(n), 0.001, decay)


# --- SFX ---
impact = kick(0.9) * 1.4 + 0.5 * noise(0.9, 0.08, 3) + 0.6 * np.sin(2 * np.pi * 38 * t(0.9)) * env(int(SR * 0.9), 0.001, 0.35)
sf.write(ROOT / "public/sfx/impact.wav", norm(impact), SR)

w = rng.standard_normal(int(SR * 0.6))
sweep = np.concatenate([lowpass(w[i:i + 1000], max(2, int(40 - 38 * i / len(w)))) for i in range(0, len(w), 1000)])[: len(w)]
wt = np.linspace(0, 1, len(sweep))
sf.write(ROOT / "public/sfx/whoosh.wav", norm(sweep * np.sin(np.pi * wt) ** 2, 0.7), SR)

g = np.zeros(int(SR * 0.3))
for i in range(6):
    s, n = int(rng.integers(0, len(g) - 2000)), int(rng.integers(300, 2000))
    g[s:s + n] += np.sign(np.sin(2 * np.pi * rng.integers(200, 2400) * np.arange(n) / SR)) * 0.5
g = np.round(g * 6) / 6  # bitcrush
sf.write(ROOT / "public/sfx/glitch.wav", norm(g, 0.6), SR)

rt = t(1.6)
riser = rng.standard_normal(len(rt)) * (rt / 1.6) ** 2 * 0.5 + np.sin(2 * np.pi * (200 + 900 * (rt / 1.6) ** 2) * rt) * (rt / 1.6) ** 1.5
sf.write(ROOT / "public/sfx/riser.wav", norm(riser, 0.6), SR)

tick = np.sin(2 * np.pi * 2400 * t(0.04)) * env(int(SR * 0.04), 0.0005, 0.008)
sf.write(ROOT / "public/sfx/tick.wav", norm(tick, 0.5), SR)

# --- Beat bed: 100 BPM, A-minor darkwave pulse ---
BPM = 100
beat = 60 / BPM
bars = 34
total = int(SR * beat * 4 * bars) + SR
out = np.zeros(total)


def put(sig, sec, gain=1.0):
    s = int(sec * SR)
    e = min(total, s + len(sig))
    out[s:e] += gain * sig[: e - s]


roots = [55.0, 55.0, 43.65, 49.0]  # A, A, F, G
k, sn, hh = kick(), noise(0.25, 0.06, 2), noise(0.05, 0.012)
hh = hh - lowpass(hh, 8)  # crude highpass
for bar in range(bars):
    b0 = bar * 4 * beat
    root = roots[bar % 4]
    for q in range(4):
        put(k, b0 + q * beat, 0.9)
        if q in (1, 3):
            put(sn, b0 + q * beat, 0.35)
    for e in range(8):
        put(hh, b0 + e * beat / 2 + beat / 4, 0.18)
        bt = t(beat / 2 * 0.9)
        bass = np.sign(np.sin(2 * np.pi * root * bt)) * env(len(bt), 0.003, 0.12)
        put(lowpass(bass, 30), b0 + e * beat / 2, 0.35)
pad_t = np.arange(total) / SR
out += 0.06 * np.sin(2 * np.pi * 110 * pad_t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * pad_t))
sf.write(ROOT / "public/beat.wav", norm(out, 0.8), SR)
print("beat.wav", round(total / SR, 1), "s; sfx:", sorted(p.name for p in (ROOT / "public/sfx").iterdir()))
