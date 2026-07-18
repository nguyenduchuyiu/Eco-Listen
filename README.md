# Forest Sentinel

Forest Sentinel is an edge-ready ecological acoustics platform for monitoring biodiversity and detecting illegal activity from continuous forest audio.

**Live:** [forest-sentinel-beige.vercel.app](https://forest-sentinel-beige.vercel.app)

## The idea

Vietnam's forests are rich in biodiversity, but inventories still depend heavily on camera traps and manual surveys. Forest Sentinel converts 24/7 acoustic recordings into an interactive stream of ecological intelligence:

- Detect birds, amphibians, insects, and mammals.
- Rank likely species candidates.
- Locate every detection on a synchronized timeline.
- Alert on chainsaw and gunshot signatures with evidence timestamps.
- Visualize the living soundscape in an interactive 3D acoustic space.
- Track acoustic activity, richness, and ecosystem health over time.

## End-to-end pipeline

```text
Solar forest sensor
  → edge preprocessing
  → overlapping 5-second windows
  → Perch v2 acoustic embeddings
  → biodiversity + threat detection
  → timeline, 3D soundscape, alerts, health index
```

The dashboard includes a one-click 60-second scenario containing birds, frogs, insects, mammals, mixed biophony, chainsaw, and gunshots. During playback, the live detector changes with the current inference window and raises alerts exactly where threat evidence appears.

## Stack

- **AI:** Perch v2 ONNX, 1536-dimensional embeddings, coarse-taxon linear probe
- **Audio:** Librosa, SoundFile, overlapping five-second edge windows
- **Backend:** Flask, Gunicorn, ONNX Runtime
- **Visualization:** Three.js, Canvas timelines, synchronized anomaly graphs
- **Deployment:** Vercel frontend + Railway inference service

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/setup_perch_mvp.sh
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

```bash
pytest -q
```

## Deployment

The full FP32 Perch model runs as a single-worker Railway service to keep memory predictable. Vercel serves the dashboard and communicates directly with the inference API.

- Dashboard: [forest-sentinel-beige.vercel.app](https://forest-sentinel-beige.vercel.app)
- API: [forest-sentinel-production.up.railway.app](https://forest-sentinel-production.up.railway.app)

## MVP scope

Species rankings, threat thresholds, and the health index are hackathon-stage outputs designed for rapid field validation and adaptation with Vietnamese recordings.

Built on the open-source SoundPlot framework. MIT licensed; see [LICENSE](LICENSE).
