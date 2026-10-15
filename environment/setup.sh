#!/usr/bin/env bash
set -euo pipefail
project_dir=/workspace/final_cloud_code
runtime_dir=/workspace/pinn-venv
cd "$project_dir"
test -f requirements.txt
if [ ! -x "$runtime_dir/bin/python" ]; then
  python -m venv --system-site-packages "$runtime_dir"
fi
mkdir -p /workspace/.cache/parcel-pinn-pip
"$runtime_dir/bin/python" -m pip install --no-input --disable-pip-version-check --cache-dir /workspace/.cache/parcel-pinn-pip --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
"$runtime_dir/bin/python" -m pip install --no-input --disable-pip-version-check --cache-dir /workspace/.cache/parcel-pinn-pip -r requirements.txt
"$runtime_dir/bin/python" -m pip check
"$runtime_dir/bin/python" -m unittest discover -s tests -v
