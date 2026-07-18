"""Build a repeatable 60-second hackathon monitoring scenario."""

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
SR = 32000
DURATION = 60
OUTPUT_DIR = ROOT / "data/demo"


def load(path):
    audio, _ = librosa.load(ROOT / path, sr=SR, mono=True)
    peak = max(float(np.max(np.abs(audio))), 1e-6)
    return audio.astype(np.float32) / peak


def fit(audio, seconds):
    samples = int(seconds * SR)
    repeats = int(np.ceil(samples / len(audio)))
    result = np.tile(audio, repeats)[:samples].copy()
    fade = min(int(0.25 * SR), samples // 4)
    result[:fade] *= np.linspace(0, 1, fade)
    result[-fade:] *= np.linspace(1, 0, fade)
    return result


def place(track, audio, start, gain=1.0):
    offset = int(start * SR)
    end = min(len(track), offset + len(audio))
    track[offset:end] += audio[:end - offset] * gain


def main():
    rng = np.random.default_rng(7)
    track = rng.normal(0, 0.002, DURATION * SR).astype(np.float32)
    sources = {
        "bird": load("data/birdclef_sample/Aves__XC558005.ogg"),
        "frog": load("data/birdclef_sample/Amphibia__iNat936019.ogg"),
        "insect": load("data/birdclef_sample/Insecta__iNat792679.ogg"),
        "mammal": load("data/birdclef_sample/Mammalia__iNat742666.ogg"),
        "chainsaw": load("data/threat_samples/1-116765-A-41.wav"),
        "gunshot": load("data/threat_samples/gunshot_ccby.mp3"),
    }
    place(track, fit(sources["bird"], 10), 0, 0.55)
    place(track, fit(sources["frog"], 10), 10, 0.55)
    place(track, fit(sources["insect"], 10), 20, 0.50)
    place(track, fit(sources["bird"], 10), 30, 0.34)
    place(track, fit(sources["frog"], 10), 30, 0.28)
    place(track, fit(sources["insect"], 10), 30, 0.24)
    place(track, fit(sources["mammal"], 20), 40, 0.28)
    place(track, fit(sources["chainsaw"], 8), 42, 0.72)
    place(track, sources["gunshot"], 53, 0.95)
    place(track, sources["gunshot"], 57, 0.90)
    track /= max(float(np.max(np.abs(track))) / 0.96, 1.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(OUTPUT_DIR / "forest_monitoring_scenario.wav", track, SR, subtype="PCM_16")
    timeline = [
        {"start": 0, "end": 10, "label": "Bird habitat"},
        {"start": 10, "end": 20, "label": "Amphibian habitat"},
        {"start": 20, "end": 30, "label": "Insect habitat"},
        {"start": 30, "end": 40, "label": "Mixed biophony"},
        {"start": 40, "end": 60, "label": "Mammal background"},
        {"start": 42, "end": 50, "label": "Chainsaw", "threat": True},
        {"start": 53, "end": 55.1, "label": "Gunshot", "threat": True},
        {"start": 57, "end": 59.1, "label": "Gunshot", "threat": True},
    ]
    (OUTPUT_DIR / "forest_monitoring_scenario.json").write_text(
        json.dumps(timeline, indent=2), encoding="utf-8"
    )
    print(OUTPUT_DIR / "forest_monitoring_scenario.wav")


if __name__ == "__main__":
    main()
