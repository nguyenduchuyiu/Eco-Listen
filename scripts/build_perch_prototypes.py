"""Build and cross-validate a coarse-taxon linear probe on Perch embeddings."""

import argparse
from pathlib import Path

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from src.audio.perch_analyzer import PerchAnalyzer


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data/birdclef_sample"
MODEL_DIR = ROOT / "data/models/perch"
GROUPS = list(PerchAnalyzer.GROUP_DISPLAY)


def load_embedding(analyzer, path):
    audio, sample_rate = librosa.load(path, sr=None, mono=True)
    return analyzer.embed_clip(audio, sample_rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "perch_v2.onnx")
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "coarse_taxa_probe.npz")
    args = parser.parse_args()
    analyzer = PerchAnalyzer(
        args.model,
        MODEL_DIR / "labels.csv",
        SAMPLE_DIR / "taxonomy.csv",
    )
    embeddings, labels, paths = [], [], []
    for group in GROUPS:
        group_paths = sorted(SAMPLE_DIR.glob(f"{group}__*.ogg"))
        if len(group_paths) < 2:
            raise RuntimeError(f"Need at least two samples for {group}; found {len(group_paths)}")
        for path in group_paths:
            embedding = load_embedding(analyzer, path)
            embedding /= max(np.linalg.norm(embedding), 1e-8)
            embeddings.append(embedding)
            labels.append(group)
            paths.append(path.name)

    x = np.nan_to_num(np.stack(embeddings).astype(np.float64))
    y = np.array(labels)
    estimator = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced")
    predicted = cross_val_predict(estimator, x, y, cv=LeaveOneOut())
    correct = int(np.sum(predicted == y))
    estimator.fit(x, y)

    # Reorder the learned rows to the stable dashboard group order.
    row_order = [int(np.where(estimator.classes_ == group)[0][0]) for group in GROUPS]
    output = args.output
    np.savez_compressed(
        output,
        groups=np.array(GROUPS),
        weights=estimator.coef_[row_order],
        bias=estimator.intercept_[row_order],
        validation_correct=np.array(correct),
        validation_total=np.array(len(y)),
    )
    print("Leave-one-recording-out evaluation:")
    for path, expected, actual in zip(paths, y, predicted):
        print(f"  {path:34} expected={expected:9} predicted={actual:9}")
    print(f"LOOCV accuracy: {correct}/{len(y)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
