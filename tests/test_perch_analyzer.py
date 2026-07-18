from pathlib import Path

import numpy as np
import pytest
import librosa

from src.audio.perch_analyzer import PerchAnalyzer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/models/perch"
SAMPLE_DIR = ROOT / "data/birdclef_sample"


@pytest.fixture(scope="module")
def analyzer():
    model = MODEL_DIR / "perch_v2.onnx"
    if not model.exists():
        pytest.skip("Perch model is an optional local artifact")
    return PerchAnalyzer(
        model,
        MODEL_DIR / "labels.csv",
        SAMPLE_DIR / "taxonomy.csv",
        MODEL_DIR / "coarse_taxa_probe.npz",
    )


def test_five_second_window_contract(analyzer):
    audio = np.zeros(22050 * 2, dtype=np.float32)
    windows, times, energies = analyzer._window_audio(audio, 22050)
    assert windows.shape == (1, 160000)
    assert times.tolist() == [0.0]
    assert energies.shape == (1,)


def test_model_summary_schema(analyzer):
    seconds = 5
    t = np.arange(PerchAnalyzer.SAMPLE_RATE * seconds) / PerchAnalyzer.SAMPLE_RATE
    audio = (0.1 * np.sin(2 * np.pi * 1800 * t)).astype(np.float32)
    summary = analyzer.analyze(audio, PerchAnalyzer.SAMPLE_RATE)
    assert summary["mode"] == "model"
    assert summary["windows_analyzed"] == 1
    assert 0 <= summary["match_score"] <= 100
    assert 0 <= summary["health_score"] <= 100
    assert "Perch v2 ONNX" in summary["model_source"]


def test_unseen_frog_clip_is_grouped_as_amphibian(analyzer):
    path = SAMPLE_DIR / "iNat1269019.ogg"
    if not path.exists():
        pytest.skip("Optional BirdCLEF smoke-test clip is unavailable")
    audio, sample_rate = librosa.load(path, sr=None, mono=True)
    summary = analyzer.analyze(audio, sample_rate)
    assert summary["dominant_label"] == "Amphibian call"
    assert summary["threat_detected"] is False


def test_esc50_chainsaw_triggers_review_alert(analyzer):
    path = ROOT / "data/threat_samples/1-116765-A-41.wav"
    if not path.exists():
        pytest.skip("Optional ESC-50 chainsaw clip is unavailable")
    audio, sample_rate = librosa.load(path, sr=None, mono=True)
    summary = analyzer.analyze(audio, sample_rate)
    assert summary["threat_detected"] is True
    assert summary["threat_type"] == "chainsaw"
    assert summary["dominant_label"] == "Chainsaw / power-tool threat"
    assert summary["threat_events"][0]["time"] == 0.0
    assert summary["threat_events"][0]["end_time"] == 5.0
    chainsaw_points = [p for p in summary["threat_timeline"] if p["type"] == "chainsaw"]
    assert chainsaw_points[0]["detected"] is True
    assert chainsaw_points[0]["anomaly_score"] >= summary["threat_threshold"]
