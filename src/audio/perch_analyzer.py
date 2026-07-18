"""Perch v2 inference adapter for the Forest Sentinel MVP."""

from pathlib import Path
from time import perf_counter

import librosa
import numpy as np
import pandas as pd


class PerchAnalyzer:
    """Run Perch ONNX over five-second windows and aggregate coarse taxa."""

    SAMPLE_RATE = 32000
    WINDOW_SECONDS = 5
    WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
    GROUP_DISPLAY = {
        "Aves": "Bird vocalisation",
        "Amphibia": "Amphibian call",
        "Insecta": "Insect chorus",
        "Mammalia": "Mammal / primate call",
    }

    def __init__(self, model_path, labels_path, taxonomy_path=None, probe_path=None):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.labels = pd.read_csv(labels_path).iloc[:, 0].astype(str).tolist()
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.taxon_labels = self._load_taxon_labels(taxonomy_path)
        self.group_indices = self._build_group_indices(taxonomy_path)
        self.groups = list(self.GROUP_DISPLAY)
        self.probe_weights = None
        self.probe_bias = None
        self.probe_validation = None
        if probe_path and Path(probe_path).exists():
            data = np.load(probe_path, allow_pickle=False)
            saved_groups = data["groups"].astype(str).tolist()
            if saved_groups == self.groups:
                self.probe_weights = data["weights"].astype(np.float64)
                self.probe_bias = data["bias"].astype(np.float64)
                self.probe_validation = (int(data["validation_correct"]), int(data["validation_total"]))

        self.threat_indices = {
            "gunshot": self._indices(["Gunshot_and_gunfire"]),
            "chainsaw": self._indices(["Power_tool", "Engine", "Sawing"]),
        }

    def _load_taxon_labels(self, taxonomy_path):
        """Map Perch label indices to display metadata for ranked candidates."""
        if not taxonomy_path or not Path(taxonomy_path).exists():
            return []
        taxonomy = pd.read_csv(taxonomy_path)
        rows = []
        for row in taxonomy.itertuples(index=False):
            index = self.label_to_index.get(str(row.scientific_name))
            if index is not None:
                rows.append({
                    "index": index,
                    "scientific_name": str(row.scientific_name),
                    "common_name": str(row.common_name),
                    "group": str(row.class_name),
                })
        return rows

    def _indices(self, names):
        return [self.label_to_index[name] for name in names if name in self.label_to_index]

    def _build_group_indices(self, taxonomy_path):
        result = {group: [] for group in self.GROUP_DISPLAY}
        if not taxonomy_path or not Path(taxonomy_path).exists():
            return result
        taxonomy = pd.read_csv(taxonomy_path)
        for group in result:
            names = taxonomy.loc[taxonomy["class_name"].eq(group), "scientific_name"]
            result[group] = [self.label_to_index[name] for name in names if name in self.label_to_index]
        return result

    @staticmethod
    def _normalize(values):
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        return values / np.maximum(norms, 1e-8)

    @staticmethod
    def _softmax(values, temperature=1.0):
        scaled = values / temperature
        scaled -= scaled.max(axis=-1, keepdims=True)
        exp = np.exp(scaled)
        return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-8)

    def _window_audio(self, audio, sample_rate, hop_seconds=2.5, max_windows=60):
        y = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sample_rate, target_sr=self.SAMPLE_RATE)
        if not np.any(np.isfinite(y)):
            raise ValueError("Audio does not contain finite samples")
        y = np.nan_to_num(y)
        hop = int(hop_seconds * self.SAMPLE_RATE)
        if len(y) <= self.WINDOW_SAMPLES:
            window = np.pad(y, (0, self.WINDOW_SAMPLES - len(y)))
            return window[None, :].astype(np.float32), np.array([0.0]), np.array([float(np.sqrt(np.mean(y ** 2)))])

        starts = np.arange(0, len(y) - self.WINDOW_SAMPLES + 1, hop, dtype=int)
        if starts[-1] != len(y) - self.WINDOW_SAMPLES:
            starts = np.append(starts, len(y) - self.WINDOW_SAMPLES)
        energies = np.array([
            np.sqrt(np.mean(y[start:start + self.WINDOW_SAMPLES] ** 2)) for start in starts
        ])
        if len(starts) > max_windows:
            # Preserve temporal coverage, then include the strongest windows.
            coverage = np.linspace(0, len(starts) - 1, max_windows // 2, dtype=int)
            strongest = np.argsort(energies)[-(max_windows - len(coverage)):]
            selected = np.unique(np.concatenate([coverage, strongest]))
            starts, energies = starts[selected], energies[selected]
        windows = np.stack([y[start:start + self.WINDOW_SAMPLES] for start in starts]).astype(np.float32)
        return windows, starts / self.SAMPLE_RATE, energies

    def _infer(self, windows, batch_size=1):
        embeddings, logits = [], []
        started = perf_counter()
        for offset in range(0, len(windows), batch_size):
            batch = windows[offset:offset + batch_size]
            embedding, label = self.session.run(["embedding", "label"], {"inputs": batch})
            embeddings.append(embedding)
            logits.append(label)
        return np.concatenate(embeddings), np.concatenate(logits), perf_counter() - started

    def embed_clip(self, audio, sample_rate):
        windows, _, _ = self._window_audio(audio, sample_rate, max_windows=12)
        embeddings, _, _ = self._infer(windows)
        return self._normalize(embeddings).mean(axis=0)

    def _native_group_probabilities(self, logits):
        scores = np.full((len(logits), len(self.groups)), -12.0, dtype=np.float32)
        for column, group in enumerate(self.groups):
            indices = self.group_indices.get(group, [])
            if indices:
                scores[:, column] = logits[:, indices].max(axis=1)
        return self._softmax(scores, temperature=1.7)

    def analyze(self, audio, sample_rate):
        windows, times, energies = self._window_audio(audio, sample_rate)
        embeddings, logits, latency = self._infer(windows)
        native_probs = self._native_group_probabilities(logits)

        if self.probe_weights is not None:
            normalized = self._normalize(embeddings).astype(np.float64)
            probe_logits = np.einsum("ij,kj->ik", normalized, self.probe_weights) + self.probe_bias
            probe_probs = self._softmax(probe_logits, temperature=0.85)
            probabilities = 0.75 * probe_probs + 0.25 * native_probs
            correct, total = self.probe_validation
            model_source = f"Perch v2 ONNX + linear probe ({correct}/{total} LOOCV)"
        else:
            probabilities = native_probs
            model_source = "Perch v2 ONNX native taxonomy"

        winners = probabilities.argmax(axis=1)
        peak_by_group = probabilities.max(axis=0)
        dominant_idx = int(np.argmax(probabilities.mean(axis=0)))
        energy_floor = max(float(np.percentile(energies, 30)), 0.01)
        active = energies > 0.01 if len(energies) == 1 else energies > energy_floor
        activity_rate = int(round(float(active.mean()) * 100))

        candidate_events = []
        for i, winner in enumerate(winners):
            score = float(probabilities[i, winner])
            if score >= 0.35 and (active[i] or len(windows) == 1):
                candidate_events.append({
                    "time": round(float(times[i]), 1),
                    "label": self.GROUP_DISPLAY[self.groups[winner]],
                    "match_score": int(round(score * 100)),
                })
        events = sorted(candidate_events, key=lambda item: item["match_score"], reverse=True)[:8]
        events.sort(key=lambda item: item["time"])

        threat_scores = {}
        threat_timeline = []
        for threat, indices in self.threat_indices.items():
            if indices:
                subset = logits[:, indices]
                flat_index = int(np.argmax(subset))
                window_index = flat_index // len(indices)
                threat_scores[threat] = (float(subset.flat[flat_index]), window_index)
                per_window = subset.max(axis=1)
                for window_number, raw_score in enumerate(per_window):
                    # Raw Perch logits are mapped to a stable 0–100 display scale.
                    # The alert threshold remains the validated raw-logit value 7.
                    threat_timeline.append({
                        "time": round(float(times[window_number]), 1),
                        "duration": self.WINDOW_SECONDS,
                        "type": threat,
                        "raw_score": round(float(raw_score), 2),
                        "anomaly_score": int(round(float(np.clip((raw_score + 2) / 12 * 100, 0, 100)))),
                        "detected": bool(raw_score >= 7.0),
                    })
            else:
                threat_scores[threat] = (-20.0, 0)
        threat, (threat_logit, threat_window) = max(threat_scores.items(), key=lambda item: item[1][0])
        threat_detected = threat_logit >= 7.0
        threat_match_score = int(np.clip(round(50 + (threat_logit - 7.0) * 12), 0, 99))
        threat_status = (
            f"Potential {threat} signature — review evidence clip"
            if threat_detected else "No high-confidence threat signature in this clip"
        )

        # Collapse overlapping positive windows into evidence events with an
        # explicit start/end time for the alert feed.
        threat_events = []
        for threat_name in self.threat_indices:
            positives = [item for item in threat_timeline if item["type"] == threat_name and item["detected"]]
            current = None
            for item in positives:
                if current is None or item["time"] > current["end_time"] + 0.1:
                    if current:
                        threat_events.append(current)
                    current = {
                        "time": item["time"],
                        "end_time": round(item["time"] + item["duration"], 1),
                        "label": "Chainsaw / power-tool threat" if threat_name == "chainsaw" else "Gunshot threat",
                        "threat_type": threat_name,
                        "match_score": int(np.clip(round(50 + (item["raw_score"] - 7.0) * 12), 50, 99)),
                        "anomaly_score": item["anomaly_score"],
                        "is_threat": True,
                    }
                else:
                    current["end_time"] = round(item["time"] + item["duration"], 1)
                    current["anomaly_score"] = max(current["anomaly_score"], item["anomaly_score"])
                    current["match_score"] = max(current["match_score"], int(np.clip(round(50 + (item["raw_score"] - 7.0) * 12), 50, 99)))
            if current:
                threat_events.append(current)
        threat_events.sort(key=lambda item: item["time"])

        # Compact payload for the browser: one taxon distribution per five-second
        # inference window. This drives the end-to-end timeline and 3D field.
        window_predictions = []
        for i, start in enumerate(times):
            group_scores = {
                self.groups[column]: int(round(float(probabilities[i, column]) * 100))
                for column in range(len(self.groups))
            }
            window_predictions.append({
                "time": round(float(start), 1),
                "duration": self.WINDOW_SECONDS,
                "group": self.groups[int(winners[i])],
                "scores": group_scores,
                "activity": int(round(float(energies[i] / max(float(energies.max()), 1e-8)) * 100)),
            })

        group_distribution = {
            group: int(round(float(probabilities[:, column].mean()) * 100))
            for column, group in enumerate(self.groups)
        }

        species_candidates = []
        if self.taxon_labels:
            dominant_group = self.groups[dominant_idx]
            eligible_taxa = [item for item in self.taxon_labels if item["group"] == dominant_group]
            taxon_indices = np.array([item["index"] for item in eligible_taxa], dtype=int)
            taxon_logits = logits[:, taxon_indices].max(axis=0)
            top_positions = np.argsort(taxon_logits)[-5:][::-1]
            top_values = taxon_logits[top_positions]
            # Candidate scores are explicitly a rank display, not calibrated
            # species probabilities.
            rank_scores = self._softmax(top_values[None, :], temperature=2.2)[0]
            for position, score in zip(top_positions, rank_scores):
                item = eligible_taxa[int(position)]
                species_candidates.append({
                    "common_name": item["common_name"],
                    "scientific_name": item["scientific_name"],
                    "group": item["group"],
                    "rank_score": int(round(float(score) * 100)),
                })

        if threat_detected:
            dominant_label = "Chainsaw / power-tool threat" if threat == "chainsaw" else "Gunshot threat"
            dominant_match_score = threat_match_score
            events = threat_events
        else:
            dominant_label = self.GROUP_DISPLAY[self.groups[dominant_idx]]
            dominant_match_score = int(round(float(probabilities[:, dominant_idx].mean()) * 100))

        richness = int(np.sum(peak_by_group >= 0.40))
        health_score = int(np.clip(34 + richness * 11 + activity_rate * 0.28 - (25 if threat_detected else 0), 10, 95))
        return {
            "mode": "model",
            "model_source": model_source,
            "health_score": health_score,
            "activity_rate": activity_rate,
            "dominant_label": dominant_label,
            "match_score": dominant_match_score,
            "richness": richness,
            "threat_status": threat_status,
            "threat_detected": threat_detected,
            "threat_type": threat if threat_detected else None,
            "threat_match_score": threat_match_score if threat_detected else None,
            "inference_ms": int(round(latency * 1000)),
            "windows_analyzed": len(windows),
            "events": events,
            "group_distribution": group_distribution,
            "window_predictions": window_predictions,
            "species_candidates": species_candidates,
            "threat_timeline": threat_timeline,
            "threat_events": threat_events,
            "threat_threshold": 75,
        }
