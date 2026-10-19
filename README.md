# Reporting-format change and police at-fault flags (Montgomery County, Maryland)

Code, derived data and results for the manuscript

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
* scripts that regenerate every figure and table and fill every number in the manuscript from the results.

Main results of the reported run (snapshot of 222,133 driver records in 125,413 crashes, 2015–2026): the share of
drivers flagged at fault fell from 55.2% (legacy format) to 34.6% (new format), whereas the serious/fatal injury
share showed no detectable break; under a one-culprit assumption the estimated sensitivity of the at-fault flag
fell from 0.994 to 0.632 (95% CI 0.625–0.639). See the manuscript for the full results and their limitations.

## Repository structure

```
.
├── run_all.sh                  one command: raw snapshot -> results -> figures/tables -> submission folder
├── requirements.txt
├── CITATION.cff, .zenodo.json  citation and archiving metadata
├── LICENSE, LICENSE-docs.md    MIT (code); CC BY 4.0 (documentation, results, figures)
├── src/                        pipeline scripts, run in this order
│   ├── 01_preprocess_harmonise.py   format assignment, crosswalk, cleaning, coverage/PSI/detectability
│   ├── 02_drift_experiments.py      E1–E6 (transfer, ablation, strategies, paired counts, ITS, agencies)
│   ├── 03_latent_label_model.py     E7–E10 (s and f, simulation, covariate model, noise-corrected models)
│   └── 04_manuscript_assets.py      manuscript figures (PNG, 1000 dpi) and LaTeX tables, main and supplementary
├── notebooks/                  the same four steps as Google Colab notebooks
├── data/
│   ├── README.md               provenance, checksum, required disclaimer, field descriptions
│   ├── raw/export.csv.gz       snapshot of the public file used in the paper
│   └── harmonised/             harmonised analysis file (.csv.gz), crosswalk, data dictionary, make map
├── results/                    result tables of the reported run (reference for checking a reproduction)
│   ├── harmonised/             coverage, PSI, format detectability, crosswalk confidence shares
│   ├── experiments/            E1–E6
│   └── experiments_latent/     E7–E10
├── manuscript/
│   ├── manuscript_template.tex numbers appear as @tokens@ and are filled from results
│   ├── build_manuscript.py     fills the tokens, checks the wording against the numbers, writes the submission folder
│   ├── references.bib, highlights.txt
│   └── diagrams/               Figs 1–2 as editable draw.io files, their generator and PNG exports
└── docs/
    ├── pipeline.md             which step produces which figure, table and section of the paper
    └── reproducibility.md      environment, seeds, run times, how to compare a rerun with results/
```

## Quick start

```bash
git clone https://github.com/<your-account>/crash-fault-flag-audit.git
cd crash-fault-flag-audit
python -m pip install -r requirements.txt
bash run_all.sh                 # writes everything to ./outputs
```

`run_all.sh` decompresses the raw snapshot, checks its SHA-256 against the paper's snapshot, runs the four
pipeline steps, and builds the submission folder `outputs/submission/`. Building the manuscript PDF additionally
needs a LaTeX installation with the Elsevier `elsarticle` class; without LaTeX all other steps still run.

To start from the harmonised file instead of the raw snapshot, decompress
`data/harmonised/crash_drivers_harmonised.csv.gz` into `outputs/harmonised/` and run steps 02–04 with the same
environment variables as in `run_all.sh`.

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
| `CRASH_MS_DIR` | 04 | figures and tables for the manuscript |
| `N_BOOT`, `N_BOOT_MODEL` | 02, 03 | bootstrap sizes (defaults 200 for E1–E3; 500 and 40 for E7–E10) |

## Data

The source data are the public *Crash Reporting – Drivers Data* file (dataset identifier `mmzv-x632`) published
by Montgomery County, Maryland, at
<https://data.montgomerycountymd.gov/Public-Safety/Crash-Reporting-Drivers-Data/mmzv-x632>. The file is updated
weekly and its records are preliminary, so the snapshot used in the paper is included in `data/raw/`.
Details, the checksum and the field descriptions are in [`data/README.md`](data/README.md).

As required by the portal's Terms of Use (<https://data.montgomerycountymd.gov/terms-of-use>):

> The data made available here has been modified for use from its original source, which is Montgomery County, Maryland ("the County").  THE COUNTY MAKES NO REPRESENTATIONS OR WARRANTIES AS TO THE COMPLETENESS, ACCURACY, OR TIMELINESS OF ANY DATA. THE COUNTY EXPRESSLY DISCLAIMS ALL WARRANTIES, WHETHER EXPRESS OR IMPLIED, INCLUDING ANY IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  THE COUNTY MAKES NO REPRESENTATION OR WARRANTY AS TO THE WEBISTE OR APPLICATION USING DATA DRAWN FROM DATA.MONTGOMERYCOUNTYMD.GOV OR FOR THE USEFULNESS OR INTEGRITY OF THE WEB SITE OR APPLICATION. The data is subject to change and possible deletion, temporarily or permanently. It is understood that the information contained in the web feed is being used at one's own risk.

This repository is not affiliated with, sponsored or endorsed by Montgomery County, Maryland.

## Reproducibility

All random processes use fixed seeds (42 for models and bootstraps). Gradient-boosting results can differ in the
last decimal between machines because of multi-threading. `manuscript/build_manuscript.py` stops with an error if
any qualitative statement in the manuscript (for example, which strategy has the smallest bias) no longer holds
for a rerun's numbers. See [`docs/reproducibility.md`](docs/reproducibility.md).

## Citation

Please cite the article (details will be updated on publication) and this repository; citation metadata are in
[`CITATION.cff`](CITATION.cff). Archived releases with a DOI are made through Zenodo
(DOI to be added after the first release).

### Archiving a release with a DOI

1. Sign in to <https://zenodo.org> with GitHub and switch this repository on under *GitHub* in the Zenodo settings.
2. On GitHub, create a release (for example `v1.0.0`). Zenodo archives it and mints a DOI, using the metadata
   in `.zenodo.json`.
3. Add the DOI to this README, to `CITATION.cff` (`doi:` field) and to the manuscript's data availability statement.

## Licence

* Code (`src/`, `notebooks/`, `manuscript/*.py`, `manuscript/diagrams/make_diagrams.py`, `run_all.sh`): MIT
  licence, see [`LICENSE`](LICENSE).
* Documentation, result tables, figures and diagrams: Creative Commons Attribution 4.0 International (CC BY 4.0),
  see [`LICENSE-docs.md`](LICENSE-docs.md).
* Data in `data/`: the raw snapshot and the derived harmonised file remain subject to the Montgomery County
  Terms of Use quoted above.

## Use of AI tools

Claude (Anthropic) was used to assist with writing the analysis and figure-generation code and drafting the
documentation and diagrams. The author reviewed and edited all material and is responsible for its content.

## Contact

Pathmanaban Pugazhendi, Department of Automobile Engineering, Easwari Engineering College, Chennai, India —
princepathu@gmail.com
