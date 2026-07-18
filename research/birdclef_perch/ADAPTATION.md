# BirdCLEF 2026 Perch notebook — adaptation notes

Source notebook: <https://www.kaggle.com/code/lakhindarpal/birdclef-2026-perch-inference>

## What the notebook actually does

1. Resamples/loads 32 kHz mono audio and splits each 60-second soundscape into twelve 5-second windows.
2. Runs Google Perch v2 and reads a 1,536-dimensional embedding plus 14,795 class logits for every window.
3. Maps Perch taxonomy indices into the 234 BirdCLEF competition labels.
4. Applies site/hour priors, separate handling for short events versus texture-like taxa, temporal smoothing, and rank-aware scaling.
5. Writes one multi-label prediction row per 5-second window.

## Important audit finding

The useful learned component in this version is Perch. `ProtoSSM` and `ResidualSSM` are instantiated with random parameters and no checkpoint is loaded. `Y_SC` is initialized to all zeros, so the fitted priors are not learned from positive targets either. Those pieces should not be copied into Forest Sentinel as if they were trained inference models.

## Smart adaptation for Forest Sentinel

```text
32 kHz mono stream
  → overlapping 5 s windows (2.5 s hop)
  → Perch ONNX embedding (1,536 dimensions)
  ├─ native Perch label head: bird candidates
  ├─ small calibrated probe: bird / frog / primate / insect / other
  ├─ dedicated threat head: chainsaw / gunshot / vehicle / human
  └─ temporal smoothing + event hysteresis
      → metadata, alert clip, occupancy history, health index
```

### Why use embeddings instead of training a large network

- A small linear or prototype head can be trained from tens of local examples while Perch supplies the expensive acoustic representation.
- The 1,536-dimensional embedding can support new Vietnamese taxa without retraining the entire backbone.
- Perch ONNX accepts a batch of 160,000 samples (5 seconds at 32 kHz), so the windowing contract is simple.
- The backbone can run locally, while the dashboard receives only predictions and short evidence clips.

### Keep threats separate

Perch is specialized for bioacoustics. Chainsaws and gunshots should use a separate general sound-event model or a small supervised head trained with negative forest backgrounds. A high-confidence alert should require two conditions: model score above threshold and temporal/event-shape validation. This avoids treating every impulsive branch crack as a rifle shot.

### Replace the current health score

Do not derive habitat health directly from classifier confidence. Use a rolling, site-calibrated index:

- taxonomic richness or number of occupied acoustic groups;
- biophony activity and temporal coverage;
- acoustic evenness/diversity;
- persistence relative to the site's own baseline;
- penalties for anthropogenic noise and verified threat events.

Display the components beside the combined index so judges can see that the score is explainable.

## One-day implementation priority

1. Use Perch ONNX directly; skip the random SSM heads.
2. Run 5-second windows and show top-k detections on a timeline.
3. Rename the current heuristic percentage to `acoustic match score` until a validation set exists.
4. Add one threat demo clip and a deterministic alert flow.
5. Present the edge deployment as ONNX + quantization; benchmark actual hardware later.

## Local sample

`data/birdclef_sample/` contains the BirdCLEF metadata and five small recordings covering Aves, Amphibia, Insecta, Mammalia, and Reptilia. It is intentionally small and is for pipeline inspection, not training or accuracy claims.

## Implemented MVP result

- Perch v2 ONNX model: 409,148,616 bytes, loaded with ONNX Runtime on CPU.
- Inference contract: 32 kHz mono, 5-second windows, 2.5-second hop.
- Output used: 1,536-dimensional embedding and 14,795-label head.
- Local probe validation: 11/18 leave-one-recording-out on a tiny four-group subset.
- Observed inference: roughly 100 ms for one window and about 1 second for ten windows on the development machine.
- Runtime fallback: the old heuristic is used only if Perch cannot load, and the UI exposes the source.

Run `scripts/setup_perch_mvp.sh` to recreate the local model assets.
