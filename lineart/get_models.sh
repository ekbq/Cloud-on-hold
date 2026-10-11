#!/usr/bin/env bash
# Downloads the ONNX models used by the pipeline into ./models
# (rembg and diffusers download their own models on first run).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
curl -L -o models/rocca_lineart.onnx \
  https://huggingface.co/rocca/informative-drawings-line-art-onnx/resolve/main/model.onnx
curl -L -o models/edge_detection_dexined_2024sep.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/edge_detection_dexined/edge_detection_dexined_2024sep.onnx
