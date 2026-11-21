#!/usr/bin/env bash
set -euo pipefail

# Software rasterizer: works everywhere (no GPU/display required),
# unlike egl which depends on the host's driver capabilities.
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

solution_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$solution_root/render_oracle.py" --output-dir "${LBT_OUTPUT_DIR:-/tmp/output}"
