#!/usr/bin/env bash
set -euo pipefail

ROOT="data/raw"
OUT_ROOT="data/mGLI_embeddings"
PYTHON_BIN="${PYTHON_BIN:-/Users/annasu/miniconda3/envs/InsightProt/bin/python}"

EXPECTED_DIM=81
idx=0

find "$ROOT" -type f -name "*.pdb" -print0 | while IFS= read -r -d '' pdb_file; do
  pid="$(basename "$(dirname "$pdb_file")")"
  fname="$(basename "$pdb_file")"
  mkdir -p "${OUT_ROOT}/${pid}"

  out_pt="${OUT_ROOT}/${pid}/${fname%.pdb}_mGLI.pt"

  # --------- check if already computed & valid ----------
  if [[ -f "$out_pt" ]]; then
    is_valid=$(
      "$PYTHON_BIN" - <<EOF
import torch, sys
try:
    obj = torch.load("$out_pt", map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        x = obj
    sys.exit(0 if len(x) == $EXPECTED_DIM else 1)
except Exception:
    sys.exit(1)
EOF
    ) || true

    if [[ "$is_valid" == "" ]]; then
      echo "[SKIP] ${pid}/${fname} (embedding exists and dim=$EXPECTED_DIM)"
      continue
    else
      echo "[RECOMPUTE] ${pid}/${fname} (invalid embedding)"
    fi
  fi
  # -----------------------------------------------------

  idx=$((idx + 1))
  echo "[RUN] Computing #${idx}: ${pid}/${fname}"

  "$PYTHON_BIN" data/embed_mGLI.py "$pdb_file"

  src="${pdb_file%.pdb}_mGLI.pt"
  mv -f "$src" "$out_pt"
done