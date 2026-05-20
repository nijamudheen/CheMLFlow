#!/usr/bin/env bash
# =============================================================================
# run_first_case_locally.sh
#
# Runs the first valid DOE case directly on this machine (no Slurm). Use this
# when you have a local GPU and want to test the full pipeline end-to-end
# without going through the cluster's queue.
#
# What it does:
#   1. Writes a runtime DOE YAML with absolute paths (same as submit_gpu.sb).
#   2. Generates DOE cases (writes manifest.jsonl).
#   3. Picks the first VALID case from the manifest.
#   4. Runs a GPU sanity check.
#   5. Runs main.py against that case with CHEMLFLOW_CONFIG set.
#
# Usage:
#     ./run_first_case_locally.sh                        # uses defaults
#     CASE_INDEX=3 ./run_first_case_locally.sh           # run the 3rd valid case
#     RUN_PROJECT_DIR=/path/to/CheMLFlow-main ./run_first_case_locally.sh
#     PY=/path/to/python ./run_first_case_locally.sh
#     SKIP_DOE_GEN=1 ./run_first_case_locally.sh         # reuse existing manifest
#
# The script is intentionally a near-copy of the SLURM submit_gpu.sb's body
# (DOE generation + child execution) so debugging in one transfers to the other.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration — override via environment variables, not edits.
# -----------------------------------------------------------------------------
RUN_PROJECT_DIR="${RUN_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PY:-python}"
DOE_SRC="${DOE_SRC:-$RUN_PROJECT_DIR/config/doe_timeseries.yaml}"
DOE_RUN="${DOE_RUN:-$RUN_PROJECT_DIR/doe_timeseries.runtime.yaml}"
DOE_OUT="${DOE_OUT:-$RUN_PROJECT_DIR/A-NVAR_doe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_PROJECT_DIR/output}"
LOG_DIR="${LOG_DIR:-$RUN_PROJECT_DIR/logs_local}"
CASE_INDEX="${CASE_INDEX:-1}"            # which valid case to run (1-based)
SKIP_DOE_GEN="${SKIP_DOE_GEN:-0}"        # set to 1 to reuse an existing manifest

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
mkdir -p "$LOG_DIR" "$DOE_OUT" "$OUTPUT_ROOT"

if ! command -v "$PY" >/dev/null 2>&1 && [[ ! -x "$PY" ]]; then
  echo "Python not found/executable: $PY" >&2
  echo "Hint: export PY=/path/to/python (e.g. your conda env's python)." >&2
  exit 2
fi
if [[ ! -f "$RUN_PROJECT_DIR/main.py" ]]; then
  echo "main.py not found in $RUN_PROJECT_DIR" >&2
  exit 2
fi
if [[ ! -f "$RUN_PROJECT_DIR/scripts/generate_doe.py" ]]; then
  echo "DOE generator not found: $RUN_PROJECT_DIR/scripts/generate_doe.py" >&2
  exit 2
fi
if [[ ! -f "$DOE_SRC" ]]; then
  echo "DOE source not found: $DOE_SRC" >&2
  exit 2
fi

echo "=== Local runner configuration ==="
echo "  RUN_PROJECT_DIR = $RUN_PROJECT_DIR"
echo "  PY              = $PY"
echo "  DOE_SRC         = $DOE_SRC"
echo "  DOE_OUT         = $DOE_OUT"
echo "  OUTPUT_ROOT     = $OUTPUT_ROOT"
echo "  CASE_INDEX      = $CASE_INDEX"
echo "  SKIP_DOE_GEN    = $SKIP_DOE_GEN"
echo "==================================="

# -----------------------------------------------------------------------------
# Step 1+2: write runtime DOE YAML + generate cases
# -----------------------------------------------------------------------------
if [[ "$SKIP_DOE_GEN" != "1" ]]; then
  "$PY" - "$DOE_SRC" "$DOE_RUN" "$DOE_OUT" "$OUTPUT_ROOT" "$RUN_PROJECT_DIR" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, doe_out, output_root, project_dir = sys.argv[1:6]
src_path = Path(src).resolve()
project_dir = Path(project_dir).resolve()

spec = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
spec.setdefault("output", {})["dir"] = str(Path(doe_out).resolve())
spec.setdefault("defaults", {})["global.base_dir"] = f"{Path(output_root).resolve()}/data"
spec.setdefault("defaults", {})["global.run_dir"]  = f"{Path(output_root).resolve()}/runs"

dataset = spec.setdefault("dataset", {})
source = dataset.setdefault("source", {})
if source.get("type") in {"local_csv", "local_npy", "local_npz"} and "path" in source:
    p = Path(source["path"])
    if not p.is_absolute():
        candidate = (src_path.parent / p).resolve()
        if not candidate.exists():
            candidate = (project_dir / p).resolve()
        source["path"] = str(candidate)

Path(dst).write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
print(f"Wrote runtime DOE: {dst}")
print(f"Resolved dataset path: {source.get('path')}")
PY

  cd "$RUN_PROJECT_DIR"
  "$PY" scripts/generate_doe.py --doe "$DOE_RUN"
fi

MANIFEST="$DOE_OUT/manifest.jsonl"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  echo "Hint: rerun without SKIP_DOE_GEN=1 to regenerate it." >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# Step 3: pick the Nth valid case from the manifest
# -----------------------------------------------------------------------------
CHOSEN_CFG="$(
  "$PY" - "$MANIFEST" "$CASE_INDEX" <<'PY'
import json, sys
from pathlib import Path

manifest, case_index = sys.argv[1], int(sys.argv[2])
valid = []
for line in Path(manifest).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    if str(rec.get("status", "")).lower() != "valid":
        continue
    cfg = rec.get("config_path")
    if cfg:
        valid.append(cfg)

if not valid:
    sys.exit("No valid cases in manifest.")
if case_index < 1 or case_index > len(valid):
    sys.exit(f"CASE_INDEX={case_index} out of range; manifest has {len(valid)} valid cases.")
print(valid[case_index - 1])
PY
)"

echo "=== Chosen case ==="
echo "  $CHOSEN_CFG"
echo "==================="

# -----------------------------------------------------------------------------
# Step 4: GPU sanity check (mirrors the child preamble in submit_gpu.sb)
# -----------------------------------------------------------------------------
echo "=== GPU CHECK ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || echo "nvidia-smi -L failed"
else
  echo "nvidia-smi not on PATH"
fi
"$PY" - <<'PYCHK'
import os, sys
try:
    import torch
except Exception as exc:
    print(f"torch import failed: {exc}", file=sys.stderr)
    sys.exit(0)
print(f"  torch.__version__         = {torch.__version__}")
print(f"  torch.version.cuda        = {torch.version.cuda}")
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"  torch.cuda.device_count() = {torch.cuda.device_count()}")
print(f"  CUDA_VISIBLE_DEVICES env  = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
PYCHK

# -----------------------------------------------------------------------------
# Step 5: actually run main.py against the chosen case.
#
# We mirror the child job's environment hygiene: unset PYTHONPATH so we use
# only the conda env, and clear PYTHONNOUSERSITE so ~/.local doesn't leak in.
# Output is teed to both the terminal and a per-run log file under LOG_DIR.
# -----------------------------------------------------------------------------
case_name="$(basename "$CHOSEN_CFG" .yaml)"
log_out="$LOG_DIR/local_${case_name}.out"
log_err="$LOG_DIR/local_${case_name}.err"

echo "=== RUN ==="
echo "  CHEMLFLOW_CONFIG = $CHOSEN_CFG"
echo "  stdout -> $log_out"
echo "  stderr -> $log_err"
echo "==========="

export PYTHONNOUSERSITE=1
unset PYTHONPATH

cd "$RUN_PROJECT_DIR"
CHEMLFLOW_CONFIG="$CHOSEN_CFG" "$PY" -u main.py \
  > >(tee "$log_out") \
  2> >(tee "$log_err" >&2)
