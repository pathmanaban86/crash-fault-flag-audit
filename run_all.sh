#!/usr/bin/env bash
# Reproduce the full pipeline from the raw data snapshot to the submission folder.
# Usage:  bash run_all.sh            (writes everything to ./outputs)
#         OUT=/some/dir bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
OUT="${OUT:-outputs}"
mkdir -p "$OUT"

# 0. raw data snapshot (see data/README.md for provenance and checksum)
if [ ! -f data/raw/export.csv ]; then
  gunzip -c data/raw/export.csv.gz > data/raw/export.csv
fi
python - <<'PY'
import hashlib
h = hashlib.sha256(open("data/raw/export.csv", "rb").read()).hexdigest()
expected = "0ab950c6e88ccbf74d908fbc1ffff4b77b034c13a8feb6573c89b33d9081c8b7"
print("raw snapshot SHA-256:", h, "(matches the paper's snapshot)" if h == expected else "(DIFFERENT from the paper's snapshot)")
PY

export CRASH_DATA_PATH="data/raw/export.csv"
export CRASH_PREP_DIR="$OUT/harmonised"
export CRASH_EXP_DIR="$OUT/experiments"
export CRASH_LAT_DIR="$OUT/experiments_latent"
export CRASH_MS_DIR="$OUT/manuscript"
export MPLBACKEND=Agg

python src/01_preprocess_harmonise.py      # format assignment, harmonisation, drift diagnostics
python src/02_drift_experiments.py         # E1-E6
python src/03_latent_label_model.py        # E7-E10
python src/04_manuscript_assets.py         # figures (PNG, 1000 dpi) and LaTeX tables

# the two draw.io diagrams are Figs 1-2; data figures start at Fig 3
cp manuscript/diagrams/Fig_01.png manuscript/diagrams/Fig_01.tex \
   manuscript/diagrams/Fig_02.png manuscript/diagrams/Fig_02.tex "$CRASH_MS_DIR/figures/"

# fill every number in the manuscript from the outputs and write one flat submission folder
python manuscript/build_manuscript.py "$CRASH_PREP_DIR" "$CRASH_EXP_DIR" "$CRASH_LAT_DIR" "$CRASH_MS_DIR" "$OUT/submission"
echo "Done. Compare $OUT/experiments*/ with results/ to check the reproduction."
