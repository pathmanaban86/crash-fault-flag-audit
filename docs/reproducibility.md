# Reproducibility

## Environment

Python 3 with the packages in `requirements.txt`; the pipeline has been run locally and on Google Colab. The
reported results were produced with the pipeline in this repository on Python 3.13 with pandas 3.0, NumPy 2.5, scikit-learn 1.9, LightGBM 4.7,
statsmodels 0.15, SciPy 1.18 and Matplotlib 3.11; the Colab notebooks install LightGBM and statsmodels if they
are missing. To record the exact versions of your own run:

```bash
python -m pip freeze > outputs/pip_freeze.txt
```

## Determinism

* Seeds are fixed (42) for model training, bootstrap resampling and the simulation (seeds 1000–1002 for the
  three noise draws).
* LightGBM uses all available threads, so floating-point results can differ in the last digits between
  machines. Rounded values in the manuscript (three decimals) are expected to agree.
* The raw portal file changes weekly. The reported numbers are reproduced only with the snapshot analysed in the
  paper (SHA-256 `0ab950c6e88ccbf74d908fbc1ffff4b77b034c13a8feb6573c89b33d9081c8b7`; `run_all.sh` checks it).

## Checking a rerun

1. Run `bash run_all.sh`.
2. Compare `outputs/harmonised/`, `outputs/experiments/` and `outputs/experiments_latent/` with the
   corresponding folders in `results/`, for example:

   ```python
   import pandas as pd
   a = pd.read_csv("results/experiments_latent/E7_sensitivity_estimates.csv")
   b = pd.read_csv("outputs/experiments_latent/E7_sensitivity_estimates.csv")
   print((a.select_dtypes("number") - b.select_dtypes("number")).abs().max())
   ```

## Run time

On a two-core test machine, `run_all.sh` took about 9 minutes. Step 02 (bootstrapped gradient-boosting models)
takes the longest. Bootstrap sizes can be reduced for a quick check with the environment variables `N_BOOT` and
`N_BOOT_MODEL`, at the cost of wider and slightly different intervals.

## Verification of this repository

A clean copy of this repository was run end to end with `bash run_all.sh` on the snapshot analysed in the
paper. Every table in `results/` was reproduced with a maximum absolute difference of 0.
