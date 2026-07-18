#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="$project_root/data/models/perch"
model_path="$model_dir/perch_v2.onnx"
expected_bytes=409148616
python_bin="$project_root/.venv/bin/python"

if [[ ! -x "$python_bin" ]]; then
  python_bin=python3
fi

mkdir -p "$model_dir"
"$python_bin" -m pip install 'onnxruntime>=1.19.0'

current_bytes=0
if [[ -f "$model_path" ]]; then
  current_bytes=$(wc -c < "$model_path" | tr -d ' ')
fi
if [[ "$current_bytes" != "$expected_bytes" ]]; then
  curl -L --fail --show-error \
    'https://huggingface.co/justinchuby/Perch-onnx/resolve/main/perch_v2.onnx' \
    -o "$model_path.download"
  downloaded_bytes=$(wc -c < "$model_path.download" | tr -d ' ')
  if [[ "$downloaded_bytes" != "$expected_bytes" ]]; then
    echo "Unexpected model size: $downloaded_bytes (expected $expected_bytes)" >&2
    exit 1
  fi
  mv "$model_path.download" "$model_path"
fi

curl -L --fail --silent --show-error \
  'https://huggingface.co/cgeorgiaw/Perch/resolve/main/assets/labels.csv' \
  -o "$model_dir/labels.csv"

if compgen -G "$project_root/data/birdclef_sample/*__*.ogg" >/dev/null; then
  cd "$project_root"
  PYTHONPATH=. "$python_bin" -W ignore scripts/build_perch_prototypes.py
else
  echo "Perch installed. No local labelled clips found; native taxonomy mode will be used."
fi

