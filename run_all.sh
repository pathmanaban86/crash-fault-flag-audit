#!/usr/bin/env bash
# Run the full pipeline from the raw data file to the results, figures and tables.
# Usage:  bash run_all.sh                                   (raw file at data/raw/export.csv, outputs to ./outputs)
#         CRASH_DATA_PATH=/path/to/export.csv OUT=/some/dir bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
OUT="${OUT:-outputs}"
export CRASH_DATA_PATH="${CRASH_DATA_PATH:-data/raw/export.csv}"

if [ ! -f "$CRASH_DATA_PATH" ]; then
  echo "Raw data not found at $CRASH_DATA_PATH."
  echo "Download the CSV export of 'Crash Reporting - Drivers Data' from"
  echo "  https://data.montgomerycountymd.gov/Public-Safety/Crash-Reporting-Drivers-Data/mmzv-x632"
  echo "and save it as data/raw/export.csv (or set CRASH_DATA_PATH)."
  exit 1
fi

python - <<'PY'
import hashlib, os
h = hashlib.sha256(open(os.environ["CRASH_DATA_PATH"], "rb").read()).hexdigest()
expected = "0ab950c6e88ccbf74d908fbc1ffff4b77b034c13a8feb6573c89b33d9081c8b7"
if h == expected:
    print("Raw file is the snapshot analysed in the paper: results should match results/.")
else:
    print("Raw file differs from the snapshot analysed in the paper (the portal is updated weekly):")
    print("results will differ from results/.")
PY

mkdir -p "$OUT"
export CRASH_PREP_DIR="$OUT/harmonised"
export CRASH_EXP_DIR="$OUT/experiments"
export CRASH_LAT_DIR="$OUT/experiments_latent"
export CRASH_MS_DIR="$OUT/manuscript"
export MPLBACKEND=Agg

python src/01_preprocess_harmonise.py      # format assignment, harmonisation, drift diagnostics
python src/02_drift_experiments.py         # E1-E6
python src/03_latent_label_model.py        # E7-E10
python src/04_manuscript_assets.py         # data figures (PNG, 1000 dpi) and LaTeX tables

echo "Done. Results are in $OUT/; compare $OUT/harmonised, $OUT/experiments and $OUT/experiments_latent with results/."
