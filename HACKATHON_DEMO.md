# Forest Sentinel — Hackathon Demo

## One-line pitch

Forest Sentinel turns continuous forest audio into compact ecological signals at the edge: animal activity, threat alerts, and a habitat-health trend without streaming 24/7 raw audio.

## 90-second demo flow

1. **Problem (15s):** Forest surveys are expensive and intermittent. A recorder hears the ecosystem continuously, but sending and reviewing all audio does not scale.
2. **Import (10s):** Upload a short forest recording. Explain that the production device would process the same window locally on solar/battery power.
3. **Acoustic map (20s):** Show the 3D trajectory. X is spectral centroid, Y is pitch, and Z is spectral bandwidth. Colour and motion reveal changing acoustic niches.
4. **Intelligence (20s):** Point to the dominant acoustic class, model-match score, activity rate, habitat-health index, event timeline, and threat scan.
5. **Edge story (15s):** Only event metadata and short evidence clips need to leave the forest. Raw recordings remain local unless an alert is raised.
6. **Roadmap (10s):** Quantize the Perch backbone, expand the probe with Vietnamese recordings, and train a dedicated chainsaw/gunshot head.

## What is real in this prototype

- Audio upload, resampling, normalization, and five-minute trimming.
- Pitch (pYIN), spectral centroid, bandwidth, energy, and spectral-flatness extraction.
- Interactive 3D acoustic trajectory and downloadable ecological report.
- Perch v2 ONNX inference over real 5-second windows, with a small coarse-taxon probe.
- Model provenance, measured inference latency, window count, and model-match score shown in the UI.
- Fast local processing with no cloud dependency after the UI libraries load.

## What is the next model milestone

The current coarse-taxon probe was trained on only 18 BirdCLEF recordings and scored 11/18 under leave-one-recording-out validation. It demonstrates transfer learning, not production accuracy. Perch's native species head is real, but Vietnamese field recordings, site calibration, more negative backgrounds, quantization, and false-alert evaluation are still required. The chainsaw signal is currently a proxy assembled from Perch's general `Power_tool`, `Engine`, and `Sawing` outputs; it is not a dedicated verified detector.

## Architecture answer for judges

```text
Solar recorder → 5-second audio windows → denoise/features → edge classifier
                                                    ├─ animal event metadata
                                                    ├─ chainsaw/gunshot alert + evidence clip
                                                    └─ rolling habitat-health statistics
```

The dashboard receives small JSON events instead of a continuous audio stream. This is the core bandwidth and battery-saving design decision.
