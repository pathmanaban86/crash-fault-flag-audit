# Reporting-format change and police at-fault flags (Montgomery County, Maryland)

Code and results for the manuscript

> Pugazhendi, P. *Reporting-format change and the reliability of police at-fault flags: evidence from interrupted time series and a pair-count measurement model.* Submitted to *Accident Analysis & Prevention* (2026).

On 1 January 2024 Maryland revised its electronic crash report (ACRS 2.0). The public Montgomery County
*Crash Reporting – Drivers Data* file contains records in both the legacy and the new coding. This repository
uses that change as a natural experiment on the stability of police-reported labels. It provides:

* a style-based assignment of every record to the legacy or new format and an auditable, confidence-graded
  crosswalk that harmonises 12 coded fields;
* interrupted time-series tests, a placebo scan of break months and distribution-drift diagnostics;
* temporal-transfer and adaptation experiments with gradient-boosted models (E1–E6);
* a pair-count measurement model that estimates the sensitivity *s* and false-flag rate *f* of the at-fault
  flag from two-driver crashes, with a covariate model, robustness checks, a simulation with injected noise and
  a noise-corrected classifier (E7–E10);
* a script that regenerates the data figures and the tables of the manuscript and its supplement.

Main results of the reported run (snapshot of 222,133 driver records in 125,413 crashes, 2015–2026): the share of
drivers flagged at fault fell from 55.2% (legacy format) to 34.6% (new format), whereas the serious/fatal injury
share showed no detectable break; under a one-culprit assumption the estimated sensitivity of the at-fault flag
fell from 0.994 to 0.632 (95% CI 0.625–0.639). See the manuscript for the full results and their limitations.

## Repository structure

```
.
├── run_all.sh                  one command: raw data -> results -> figures and tables
├── requirements.txt
├── CITATION.cff                citation metadata (GitHub shows a "Cite this repository" button)
├── LICENSE, LICENSE-docs.md    MIT (code); CC BY 4.0 (documentation and result tables)
├── src/                        pipeline scripts, run in this order
│   ├── 01_preprocess_harmonise.py   format assignment, crosswalk, cleaning, coverage/PSI/detectability
│   ├── 02_drift_experiments.py      E1–E6 (transfer, ablation, strategies, paired counts, ITS, agencies)
│   ├── 03_latent_label_model.py     E7–E10 (s and f, simulation, covariate model, noise-corrected models)
│   └── 04_manuscript_assets.py      data figures (PNG, 1000 dpi) and LaTeX tables, main and supplementary
├── notebooks/                  the same four steps as Google Colab notebooks
├── results/                    result tables of the reported run (reference for checking a reproduction)
│   ├── harmonised/             crosswalk, data dictionary, coverage, PSI, format detectability
│   ├── experiments/            E1–E6
│   └── experiments_latent/     E7–E10
└── docs/
    ├── pipeline.md             which step produces which figure, table and section of the paper
    └── reproducibility.md      environment, seeds, run time, how to compare a rerun with results/
```

The raw data are not included (see [Data](#data)).

## Quick start

```bash
git clone https://github.com/pathmanaban86/crash-fault-flag-audit.git
cd crash-fault-flag-audit
python -m pip install -r requirements.txt
mkdir -p data/raw
# download the CSV export of the data set (see Data) and save it as data/raw/export.csv
bash run_all.sh                 # writes everything to ./outputs
```

`run_all.sh` reports whether the raw file is the snapshot analysed in the paper (by its SHA-256 checksum), runs
the four pipeline steps, and writes the results to `outputs/`, with the same file names as in `results/`.
A different path to the raw file can be given with `CRASH_DATA_PATH=/path/to/export.csv bash run_all.sh`.

### Google Colab

Open the notebooks in `notebooks/` in order. By default they read and write
`/content/drive/MyDrive/Research/crash driver data/`; set the environment variables below (or edit the first
cell) to use another location.

| Variable | Used by | Meaning |
|---|---|---|
| `CRASH_DATA_PATH` | 01 | path to the raw `export.csv` |
| `CRASH_PREP_DIR` | 01–04 | harmonised outputs |
| `CRASH_EXP_DIR` | 02, 04 | outputs of E1–E6 |
| `CRASH_LAT_DIR` | 03, 04 | outputs of E7–E10 |
| `CRASH_MS_DIR` | 04 | figures and tables |
| `N_BOOT`, `N_BOOT_MODEL` | 02, 03 | bootstrap sizes (defaults 200 for E1–E3; 500 and 40 for E7–E10) |

## Data

The source data are the public *Crash Reporting – Drivers Data* file (dataset identifier `mmzv-x632`) published
by Montgomery County, Maryland, at
<https://data.montgomerycountymd.gov/Public-Safety/Crash-Reporting-Drivers-Data/mmzv-x632>
(use *Export* → *CSV*). The data are collected through the Automated Crash Reporting System of the Maryland State
Police.

| Snapshot analysed in the paper | |
|---|---|
| Retrieved | October 2026 |
| Crash dates | 1 January 2015 – 14 October 2026 |
| Size | 222,133 driver records × 39 columns (125,413 crashes); 120,616,253 bytes |
| SHA-256 of `export.csv` | `0ab950c6e88ccbf74d908fbc1ffff4b77b034c13a8feb6573c89b33d9081c8b7` |

The portal file is updated weekly and its records are preliminary, so a new download will contain more records
and will not reproduce the reported numbers exactly. The snapshot analysed in the paper is available from the
author on request. The file contains no names and no information on the sex, gender or age of drivers.

As required by the portal's Terms of Use (<https://data.montgomerycountymd.gov/terms-of-use>):

> The data made available here has been modified for use from its original source, which is Montgomery County, Maryland ("the County").  THE COUNTY MAKES NO REPRESENTATIONS OR WARRANTIES AS TO THE COMPLETENESS, ACCURACY, OR TIMELINESS OF ANY DATA. THE COUNTY EXPRESSLY DISCLAIMS ALL WARRANTIES, WHETHER EXPRESS OR IMPLIED, INCLUDING ANY IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  THE COUNTY MAKES NO REPRESENTATION OR WARRANTY AS TO THE WEBISTE OR APPLICATION USING DATA DRAWN FROM DATA.MONTGOMERYCOUNTYMD.GOV OR FOR THE USEFULNESS OR INTEGRITY OF THE WEB SITE OR APPLICATION. The data is subject to change and possible deletion, temporarily or permanently. It is understood that the information contained in the web feed is being used at one's own risk.

This repository is not affiliated with, sponsored or endorsed by Montgomery County, Maryland.

### Editing the crosswalk

The crosswalk used in the paper is in `results/harmonised/crosswalk_proposed.csv` (columns `column`,
`raw_pattern`, `harmonised`, `confidence`, `note`, and the number of legacy and new-format rows each rule covers).
To change a mapping, save the rows to change as `crosswalk_edited.csv` (same columns) in the harmonised output
folder (`outputs/harmonised/` with `run_all.sh`) and rerun all steps; edited rows override the proposal.

## Reproducibility

All random processes use fixed seeds (42 for models and bootstraps). Gradient-boosting results can differ in the
last decimal between machines because of multi-threading. See [`docs/reproducibility.md`](docs/reproducibility.md).

## Citation

Please cite the article (details will be updated on publication) and this repository. Citation metadata are in
[`CITATION.cff`](CITATION.cff); GitHub's *Cite this repository* button on the repository page uses them.

## Licence

* Code (`src/`, `notebooks/`, `run_all.sh`): MIT licence, see [`LICENSE`](LICENSE).
* Documentation and result tables: Creative Commons Attribution 4.0 International (CC BY 4.0), see
  [`LICENSE-docs.md`](LICENSE-docs.md).
* The source data are not part of this repository and remain subject to the Montgomery County Terms of Use
  quoted above.

## Use of AI tools

Claude (Anthropic) was used to assist with writing the analysis and figure-generation code and drafting the
documentation. The author reviewed and edited all material and is responsible for its content.

## Contact

Pathmanaban Pugazhendi, Department of Automobile Engineering, Easwari Engineering College, Chennai, India —
princepathu@gmail.com
