# Pipeline and mapping to the manuscript

Steps run in this order; each reads the outputs of the previous ones.

| Step | Script | Reads | Writes |
|---|---|---|---|
| 01 | `src/01_preprocess_harmonise.py` | raw `export.csv` (not included; see README) | harmonised file, crosswalk, data dictionary, coverage, PSI, format detectability |
| 02 | `src/02_drift_experiments.py` | harmonised file | E1–E6 result tables |
| 03 | `src/03_latent_label_model.py` | harmonised file | E7–E10 result tables |
| 04 | `src/04_manuscript_assets.py` | outputs of 01–03 | `Fig_03`–`Fig_07`, supplementary figures, LaTeX tables |
| 05 | `src/05_qie_analysis.py` | harmonised file; raw `export.csv` (Circumstance field) | E11 result tables and figures (`Fig_E11*`) |

Figs 1 and 2 of the manuscript are schematic diagrams (analytical framework and pair-count model) and are not
produced by this code; the data figures produced by step 04 are therefore numbered from 3.

## Analyses

| Code | Analysis | Result files | Manuscript |
|---|---|---|---|
| — | format assignment, harmonisation, cleaning | `results/harmonised/*` | Section 3; Table 1; Tables S1–S3; Figs S2–S3 |
| E1 | legacy-trained model scored by half-year | `E1_temporal_transfer_by_halfyear.csv` | Section 5.4; Fig. 4; Table S4 |
| E2 | field ablation and feature reliance | `E2_feature_set_ablation.csv`, `E2_reliance_shift.csv` | Section 5.4; Table S5; Figs S4–S5 |
| E3 | adaptation strategies S1–S7 | `E3_mitigation_strategies.csv` | Sections 5.4 and 5.7; Table 2; Fig. 6 |
| E4 | zero/one/two flagged drivers in two-driver crashes | `E4_paired_driver_consistency.csv` | Table S6; Fig. S6 |
| E5 | interrupted time series and placebo break scan | `E5_break_tests.csv`, `E5_break_scan.csv` | Sections 5.1–5.2; Fig. 3; Table S7; Fig. S1 |
| E6 | at-fault share by agency; logistic model of distraction coupling | `E6_agency_fault_rates.csv`, `E6_logit_MCPD_regime_coupling.csv` | Sections 5.4–5.5; Tables S8–S9; Fig. S7 |
| E7 | closed-form sensitivity *s* and false-flag rate *f*; κ relaxation | `E7_sensitivity_estimates.csv`, `E7_kappa_sensitivity.csv` | Sections 5.5–5.6; Table 3; Fig. 5; Tables S10–S11 |
| E8 | simulation with injected label noise | `E8_simulation_recovery.csv`, `E8_heterogeneous_sensitivity_recovery.csv` | Section 5.6; Table S12; Fig. S8 |
| E9 | covariate model for *s* | `E9_sensitivity_by_group.csv`, `E9_sensitivity_model_coefficients.csv` | Section 5.5; Fig. 5; Table S13 |
| E10 | noise-corrected versus naive classifiers | `E10_corrected_vs_naive.csv` | Section 5.7; Table 4; Fig. 7 |
| E11 | quasi-induced exposure: relative involvement ratios (naive, clean-pair, corrected) by period, change across the format change and placebo change, model check, year-by-year series; flag share of unidentified drivers; driver contributing actions in new-format crashes | `E11_qie_by_period.csv`, `E11_qie_change.csv`, `E11_qie_model_check.csv`, `E11_qie_by_year.csv`, `E11_unidentified_drivers.csv`, `E11_unidentified_by_year.csv`, `E11_contributing_actions.csv` | not in the current manuscript draft |

E1–E6 files are in `results/experiments/`, E7–E10 files in `results/experiments_latent/`, E11 files in
`results/experiments_qie/`.

## Key settings

* Format: share of mixed-case values across 12 style-sensitive fields above 0.5 = new format; crash date
  (1 January 2024) as fallback.
* Gradient-boosted trees (LightGBM): learning rate 0.05, 31 leaves, at least 50 observations per leaf, feature
  and row subsampling 0.8, L2 penalty 1.0, seed 42; 300 trees in E1–E3 (100 extra trees for fine-tuning), 250 in
  E9–E10 and 150 in the E8 simulation.
* Transfer: trained on legacy records before 2022; adaptation data are new-format records from 2024; test data
  are new-format records from 2025 onwards. Every driver of a crash gets the crash's earliest timestamp.
* Intervals: crash-clustered bootstrap, 200 resamples (E1–E3), 500 (E7, E10), 40 refits (E9), 1000 (E11 by
  period; 300 for the E11 year-by-year series); Wilson intervals for the flag share of unidentified drivers.
* E11 sample: two-driver crashes with both flags known and no parked vehicle; for each group, crashes in which
  the group is known for both drivers. Periods: legacy 2020–2021 and 2022–2023 (placebo), new 2024–2026.
  Groups: licence state other than Maryland; vehicle 15 or more years old; truck or bus (versus car, SUV, pickup
  or van). Unidentified driver: neither licence state nor vehicle year or make recorded.
* Interrupted time series: monthly OLS with a common linear trend and a level-change indicator from January 2024,
  Newey–West standard errors with three lags; placebo months at least 24 months from either end of the series.
