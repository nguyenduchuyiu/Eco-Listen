"""Forest Sentinel ecological-acoustics demo server."""

import secrets
import threading
import uuid
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import librosa
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import backend modules
from src.audio import AudioLoader, AudioPreprocessor, PerchAnalyzer

app = Flask(__name__, static_folder="ui")
CORS(app)

# Configuration
UPLOAD_FOLDER = Path("data/uploads")
RESULTS_FOLDER = Path("data/results")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Global task storage
TASKS = {}
_PERCH_ANALYZER = None
_PERCH_LOCK = threading.Lock()


def get_perch_analyzer():
    """Lazily load the 409 MB model once per server process."""
    global _PERCH_ANALYZER
    if _PERCH_ANALYZER is None:
        with _PERCH_LOCK:
            if _PERCH_ANALYZER is None:
                model_dir = Path("data/models/perch")
                _PERCH_ANALYZER = PerchAnalyzer(
                    model_dir / "perch_v2.onnx",
                    model_dir / "labels.csv",
                    Path("data/birdclef_sample/taxonomy.csv"),
                    model_dir / "coarse_taxa_probe.npz",
                )
    return _PERCH_ANALYZER

# --- Helper Functions ---

def normalize_features(features):
    """Normalize features to 0-10 range for 3D visualization."""
    if not features:
        return []
    
    # Extract arrays
    centroids = np.array([f['centroid'] for f in features])
    bandwidths = np.array([f['bandwidth'] for f in features])
    pitches = np.array([f['pitch'] for f in features])
    
    # Helper to normalize array
    def norm(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.full_like(arr, 5.0)
        return (arr - min_val) / (max_val - min_val) * 10
    
    centroids_norm = norm(centroids)
    bandwidths_norm = norm(bandwidths)
    pitches_norm = norm(pitches)
    
    # Reassemble
    result = []
    for i, f in enumerate(features):
        result.append({
            'time': f['time'],
            'x': float(centroids_norm[i]),  # Spectral Centroid -> X
            'y': float(bandwidths_norm[i]), # Bandwidth -> Y
            'z': float(pitches_norm[i]),    # Pitch -> Z
            'raw_centroid': float(f['centroid']),
            'raw_pitch': float(f['pitch'])
        })
    return result


def build_heuristic_summary(audio, sr):
    """Small, explainable baseline for the hackathon dashboard.

    This is deliberately labelled as a heuristic, not a trained species model.
    It converts acoustic activity into a useful demo signal while a proper
    labelled classifier is being collected for deployment.
    """
    hop = 512
    rms = librosa.feature.rms(y=audio, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=audio, hop_length=hop)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop)
    energy_floor = max(float(np.percentile(rms, 35)), 0.025)
    active = rms > energy_floor
    activity_rate = float(np.mean(active))
    mean_centroid = float(np.mean(centroid[active])) if np.any(active) else float(np.mean(centroid))
    mean_flatness = float(np.mean(flatness[active])) if np.any(active) else float(np.mean(flatness))

    if mean_centroid > 2600 and mean_flatness < 0.42:
        label = "Bird vocalisation"
    elif mean_centroid < 1500:
        label = "Amphibian / low-frequency call"
    else:
        label = "Insect chorus / mixed biophony"

    confidence = int(np.clip(56 + activity_rate * 28 + (1 - mean_flatness) * 14, 55, 92))
    # Choose widely separated high-activity moments for a compact event timeline.
    ranked = np.argsort(rms)[::-1]
    event_indices = []
    min_gap = max(1, int(8 * sr / hop))
    for idx in ranked:
        if all(abs(int(idx) - old) > min_gap for old in event_indices):
            event_indices.append(int(idx))
        if len(event_indices) == 3:
            break
    events = [{"time": round(float(times[i]), 1), "label": label, "match_score": max(50, confidence - n * 4)}
              for n, i in enumerate(sorted(event_indices))]
    health_score = int(np.clip(42 + activity_rate * 34 + (1 - mean_flatness) * 24, 20, 96))
    return {
        "mode": "fallback",
        "model_source": "Acoustic heuristic fallback — Perch unavailable",
        "health_score": health_score,
        "activity_rate": int(activity_rate * 100),
        "dominant_label": label,
        "match_score": confidence,
        "richness": 1,
        "threat_status": "No high-confidence threat signature in this clip",
        "events": events,
        "inference_ms": 0,
        "windows_analyzed": len(rms),
        "group_distribution": {
            "Aves": 70 if "Bird" in label else 10,
            "Amphibia": 70 if "Amphibian" in label else 10,
            "Insecta": 70 if "Insect" in label else 10,
            "Mammalia": 70 if "Mammal" in label else 10,
        },
        "window_predictions": [],
        "species_candidates": [],
        "threat_timeline": [],
        "threat_events": [],
        "threat_threshold": 75,
    }

def run_analysis_task(task_id, file_path, save_name):
    """Heavy processing task run in a separate thread."""
    try:
        def update_status(msg, progress):
            TASKS[task_id]['status'] = msg
            TASKS[task_id]['progress'] = progress

        # 1. Load
        update_status("Loading audio file...", 5)
        loader = AudioLoader(target_sr=22050)
        audio, sr = loader.load(file_path)
        
        # 2. Enforce 5-minute limit
        MAX_DURATION = 300  # 5 minutes in seconds
        duration = len(audio) / sr
        if duration > MAX_DURATION:
            update_status(f"Trimming audio to 5 minutes (was {duration:.1f}s)...", 8)
            audio = audio[:int(MAX_DURATION * sr)]
            duration = MAX_DURATION
        
        # 3. Create session folder
        session_name = Path(file_path).stem
        session_folder = Path("data/sessions") / f"{session_name}_{task_id[:8]}"
        session_folder.mkdir(parents=True, exist_ok=True)
        
        # 4. Preprocess
        update_status("Preprocessing & Normalizing...", 15)
        preprocessor = AudioPreprocessor(sample_rate=sr)
        audio = preprocessor.normalize(audio)
        
        # Save original to session
        original_path = session_folder / "original.wav"
        sf.write(str(original_path), audio, sr)
        
        # 5. Extract time-series features for visualization
        update_status("Mapping acoustic habitat...", 45)
        
        def extract_time_series(signal):
            """Vectorized extraction of time-series features."""
            hop_length = int(sr * 0.05) # 50ms hop
            n_fft = int(sr * 0.1) # 100ms window
            
            try:
                centroids = librosa.feature.spectral_centroid(
                    y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length
                )[0]
                
                bandwidths = librosa.feature.spectral_bandwidth(
                    y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length
                )[0]
                
                f0, voiced_flag, _ = librosa.pyin(
                    signal, fmin=100, fmax=8000, sr=sr, hop_length=hop_length
                )
                
                times = librosa.times_like(centroids, sr=sr, hop_length=hop_length)
                min_len = min(len(centroids), len(f0))
                
                chunk_features = []
                for i in range(min_len):
                    pitch = f0[i] if voiced_flag[i] else 0
                    if np.isnan(pitch): pitch = 0
                    
                    chunk_features.append({
                        'time': float(times[i]),
                        'centroid': float(centroids[i]),
                        'bandwidth': float(bandwidths[i]),
                        'pitch': float(pitch)
                    })
                return normalize_features(chunk_features)
            except Exception as e:
                print(f"Extraction error: {e}")
                import traceback
                traceback.print_exc()
                return []

        original_points = extract_time_series(audio)
        update_status("Running Perch edge inference...", 65)
        try:
            forest_summary = get_perch_analyzer().analyze(audio, sr)
        except Exception as model_error:
            print(f"Perch inference unavailable, using fallback: {model_error}")
            forest_summary = build_heuristic_summary(audio, sr)
            forest_summary["model_error"] = str(model_error)
        
        # 6. Generate ecological visualizations.
        update_status("Generating ecological report...", 75)
        comparison_image_url = None
        analysis_image_url = None
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Forest Sentinel — Ecological Acoustic Report", fontsize=14, fontweight='bold')

            times_orig = np.arange(len(audio)) / sr
            axes[0, 0].plot(times_orig, audio, color='#2196F3', linewidth=0.5)
            axes[0, 0].set_title('Habitat Waveform')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')

            rms = librosa.feature.rms(y=audio, hop_length=512)[0]
            rms_times = librosa.times_like(rms, sr=sr, hop_length=512)
            axes[0, 1].fill_between(rms_times, rms, color='#8aad22', alpha=0.8)
            axes[0, 1].set_title('Acoustic Activity')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('RMS Energy')

            S_orig = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, hop_length=1024)
            axes[1, 0].imshow(librosa.power_to_db(S_orig, ref=np.max), aspect='auto', origin='lower', cmap='magma')
            axes[1, 0].set_title('Mel Spectrogram')
            axes[1, 0].set_ylabel('Mel Bin')

            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=512)[0]
            centroid_times = librosa.times_like(centroid, sr=sr, hop_length=512)
            axes[1, 1].plot(centroid_times, centroid, color='#7b5cff', linewidth=0.8)
            axes[1, 1].set_title('Spectral Centroid / Brightness')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Frequency (Hz)')

            metric_text = (f"Health: {forest_summary['health_score']}/100 | "
                           f"Activity: {forest_summary['activity_rate']}% | "
                           f"Dominant: {forest_summary['dominant_label']}")
            fig.text(0.5, 0.02, metric_text, ha='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            comparison_path = session_folder / "comparison.png"
            plt.savefig(str(comparison_path), dpi=100, bbox_inches='tight')
            plt.close(fig)
            comparison_image_url = f"/data/sessions/{session_folder.name}/comparison.png"

            # Save metadata
            metadata = {
                "original_file": save_name,
                "duration_seconds": duration,
                "sample_rate": sr,
                "forest_summary": forest_summary
            }
            with open(session_folder / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as viz_error:
            print(f"Visualization error (non-fatal): {viz_error}")
            import traceback
            traceback.print_exc()
        
        # 10. Finalize
        update_status("Complete!", 100)
        
        TASKS[task_id].update({
            "status": "Complete",
            "progress": 100,
            "result": {
                "original_points": original_points,
                "original_audio_url": f"/data/sessions/{session_folder.name}/original.wav",
                "comparison_image_url": comparison_image_url,
                "analysis_image_url": analysis_image_url,
                "forest_summary": forest_summary,
                "duration_seconds": duration,
                "session_folder": str(session_folder)
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        TASKS[task_id]['status'] = f"Error: {str(e)}"
        TASKS[task_id]['error'] = True

# --- Routes ---

@app.route('/')
def index():
    """Serve the main UI."""
    return send_from_directory('ui', 'index.html')


@app.route('/api/health')
def health():
    """Cheap container health check; model remains lazily loaded."""
    return jsonify({"status": "ok", "service": "forest-sentinel"})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Handle audio upload and start async analysis."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        save_name = f"{Path(filename).stem}_{secrets.token_hex(4)}{Path(filename).suffix}"
        file_path = UPLOAD_FOLDER / save_name
        file.save(str(file_path))
        
        # Create Task
        task_id = str(uuid.uuid4())
        TASKS[task_id] = {
            'status': 'Uploaded. Starting task...',
            'progress': 0,
            'result': None,
            'error': False
        }
        
        # Start Thread
        thread = threading.Thread(target=run_analysis_task, args=(task_id, file_path, save_name))
        thread.start()
        
        return jsonify({'taskId': task_id})

@app.route('/api/status/<task_id>')
def task_status(task_id):
    """Retrieve the status of a specific analysis task."""
    task = TASKS.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)


@app.route('/api/demo', methods=['POST'])
def analyze_demo():
    """Run the prepared multi-species and threat monitoring scenario."""
    demo_path = Path("data/demo/forest_monitoring_scenario.wav")
    if not demo_path.exists():
        return jsonify({"error": "Demo missing; run scripts/build_demo_soundscape.py"}), 404
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        'status': 'Loading 60-second forest scenario...',
        'progress': 0,
        'result': None,
        'error': False
    }
    thread = threading.Thread(
        target=run_analysis_task,
        args=(task_id, demo_path, demo_path.name),
    )
    thread.start()
    return jsonify({'taskId': task_id})

@app.route('/data/sessions/<path:filepath>')
def serve_sessions(filepath):
    return send_from_directory('data/sessions', filepath)

@app.route('/data/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/data/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    print("Starting Forest Sentinel Server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000, threaded=True)
