# %% [markdown]
# # Step 3 - Drift experiments (Montgomery County crash drivers)
# Reads `harmonised/crash_drivers_harmonised.csv` (from the preprocessing step) and writes everything to
# `<data folder>/experiments/`. Runtime about 5-10 min on Colab CPU.
#
# E1  Temporal transfer: train on legacy years, test by half-year (fault vs serious/fatal injury)
# E2  Which fields carry the drift: feature-set ablation + reliance shift
# E3  Mitigation strategies tested on 2025+ (new regime)
# E4  Paired-driver consistency (exactly one at-fault driver per two-driver crash)
# E5  Structural-break tests on monthly series (+ placebo break scan)
# E6  Agency heterogeneity

# %%
# 1. SETUP
import os, sys, json, warnings, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm"], check=True)
    import lightgbm as lgb
try:
    import statsmodels.api as sm
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "statsmodels"], check=True)
    import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

PREP_DIR = os.environ.get("CRASH_PREP_DIR", "/content/drive/MyDrive/Research/crash driver data/harmonised")
OUT_DIR = os.environ.get("CRASH_EXP_DIR", os.path.join(os.path.dirname(PREP_DIR.rstrip("/")), "experiments"))
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_BOOT = int(os.environ.get("N_BOOT", 200))       # cluster bootstrap (by crash report)
SEED = 42
ADAPT_START, ADAPT_END = "2024-01-01", "2025-01-01"   # new-regime rows used for adaptation
TEST_START = "2025-01-01"                             # strategy test window (new regime)
ADAPT_WEIGHT = 5.0                                    # weight on adaptation rows in the weighted strategy
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=SEED, num_threads=0)
N_TREES, N_TREES_FT = 300, 100

# Consistent figure style: white background, no grid, minimal text
C_FAULT, C_KSI, C_GREY = "#1f4e79", "#c0504d", "#7f7f7f"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": False,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

def save_fig(name):
    plt.savefig(os.path.join(FIG_DIR, name + ".png")); plt.show(); plt.close()

def log(*a):
    print(*a, flush=True)

# %%
# 2. LOAD AND FEATURE SETS
t0 = time.time()
src = os.path.join(PREP_DIR, "crash_drivers_harmonised.csv")
df = pd.read_csv(src, low_memory=False, parse_dates=["crash_datetime"])
log(f"Loaded {len(df):,} rows, {df.report_number.nunique():,} crashes from {src}")

df["ym"] = df.crash_datetime.dt.to_period("M")
df["half"] = df.crash_datetime.dt.year.astype(str) + "H" + np.where(df.crash_datetime.dt.month <= 6, "1", "2")
# every driver of a crash gets the crash's earliest timestamp, so a crash is never split across time windows
df["t_crash"] = df.groupby("report_number").crash_datetime.transform("min")
df["is_new"] = (df.regime == "new").astype(int)
df["gid"] = pd.factorize(df.report_number)[0]

# keep make to the 40 most common values
top_make = df.make_h.value_counts().head(40).index
df["make_top"] = df.make_h.where(df.make_h.isin(top_make), "other")

CAT = ["collision_h", "movement_h", "route_h", "light_h", "control_h", "direction_h", "agency_h", "weather_h",
       "surface_h", "damage_h", "body_h", "distraction_h", "impact_zone_h", "alcohol_susp", "drug_susp",
       "parked_vehicle", "license_state_group", "make_top"]
NUM = ["speed_limit", "vehicle_age", "crash_hour", "crash_dow", "crash_month"]
# deliberately excluded: report_type (derived from injury severity -> leakage), injury_ord/y_* (targets),
# n_fault_yes_in_crash / zero_fault_two_driver (derived from the fault label), ids, free-text, coordinates.
for c in CAT:
    df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
    df[c] = pd.Categorical(df[c], categories=sorted(df[c].dropna().unique()))   # same categories everywhere

FEATURE_SETS = {
    "full": CAT + NUM,
    "no_distraction": [c for c in CAT + NUM if c != "distraction_h"],
    "no_distraction_agency": [c for c in CAT + NUM if c not in ("distraction_h", "agency_h")],
    "no_distraction_agency_substance": [c for c in CAT + NUM if c not in ("distraction_h", "agency_h", "alcohol_susp", "drug_susp")],
}
TARGETS = {"fault": "y_fault", "ksi": "y_ksi"}

legacy = df[(df.regime == "legacy")]
adapt = df[(df.regime == "new") & (df.t_crash >= ADAPT_START) & (df.t_crash < ADAPT_END)]
test = df[(df.regime == "new") & (df.t_crash >= TEST_START)]
log(f"Legacy train {len(legacy):,} | adapt (new, 2024) {len(adapt):,} | test (new, 2025+) {len(test):,}")

# %%
# 3. HELPERS
def fit(train, feats, tgt, weight=None, n_trees=N_TREES, init_model=None, extra=None):
    d = train[train[tgt].notna()]
    ds = lgb.Dataset(d[feats], label=d[tgt].astype(int), weight=None if weight is None else weight.loc[d.index],
                     categorical_feature=[f for f in feats if f in CAT], free_raw_data=False)
    p = dict(LGB_PARAMS); p.update(extra or {})
    return lgb.train(p, ds, num_boost_round=n_trees, init_model=init_model)

def ece(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop"); g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True)
    return float((g.size() / len(y) * (g.p.mean() - g.y.mean()).abs()).sum())

def metrics(y, p):
    return dict(auroc=roc_auc_score(y, p), ap=average_precision_score(y, p), brier=float(np.mean((p - y) ** 2)),
                citl=float(p.mean() - y.mean()), ece=ece(y, p), n=len(y), events=int(y.sum()), prevalence=float(y.mean()))

def boot_ci(y, p, gid, n_boot=N_BOOT, seed=SEED):
    """cluster bootstrap by crash: resample crashes with replacement via integer weights"""
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p)
    u, inv = np.unique(gid, return_inverse=True); G = len(u)
    out = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, G, G), minlength=G)[inv]
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0: continue
        out.append((roc_auc_score(y, p, sample_weight=w), np.average((p - y) ** 2, weights=w),
                    np.average(p, weights=w) - np.average(y, weights=w)))
    a = np.array(out)
    return {k: (np.percentile(a[:, i], 2.5), np.percentile(a[:, i], 97.5)) for i, k in enumerate(["auroc", "brier", "citl"])}

def evaluate(model, d, feats, tgt, label, extra=None, ci=False):
    d = d[d[tgt].notna()]
    p = model.predict(d[feats]) if not isinstance(model, np.ndarray) else model
    m = metrics(d[tgt].values, p)
    if ci:
        b = boot_ci(d[tgt].values, p, d.gid.values)
        for k, (lo, hi) in b.items(): m[f"{k}_lo"], m[f"{k}_hi"] = lo, hi
    m.update(label); m.update(extra or {})
    return m, p

# %%
# 4. E1 - TEMPORAL TRANSFER: legacy-trained model, scored by half-year
log("\nE1 temporal transfer"); rows = []; models_full = {}
feats = FEATURE_SETS["full"]
for tn, tgt in TARGETS.items():
    # train on 2015-2021 legacy rows, so 2022 and 2023 are in-regime controls for "ordinary" time decay
    tr = legacy[legacy.t_crash < "2022-01-01"]
    mdl = fit(tr, feats, tgt); models_full[tn] = mdl
    for h, d in df[df.t_crash >= "2022-01-01"].groupby("half"):
        d = d[d[tgt].notna()]
        if d[tgt].nunique() < 2 or len(d) < 500: continue
        m, _ = evaluate(mdl, d, feats, tgt, dict(target=tn, half=h, regime=("new" if d.is_new.mean() > .5 else "legacy"),
                                                 share_new=float(d.is_new.mean())), ci=True)
        rows.append(m)
e1 = pd.DataFrame(rows); e1.to_csv(os.path.join(OUT_DIR, "E1_temporal_transfer_by_halfyear.csv"), index=False)
print(e1.pivot(index="half", columns="target", values=["auroc", "citl"]).round(3))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for tn, col, lab in [("fault", C_FAULT, "At-fault"), ("ksi", C_KSI, "Serious/fatal")]:
    d = e1[e1.target == tn]
    ax[0].errorbar(d.half, d.auroc, yerr=[d.auroc - d.auroc_lo, d.auroc_hi - d.auroc], marker="o", color=col, label=lab, capsize=2)
    ax[1].errorbar(d.half, d.citl, yerr=[d.citl - d.citl_lo, d.citl_hi - d.citl], marker="o", color=col, capsize=2)
ax[0].set_ylabel("AUROC"); ax[1].set_ylabel("Mean predicted - observed"); ax[1].axhline(0, color=C_GREY, lw=.8)
for a in ax:
    a.axvline(list(d.half).index("2024H1") - .5, color=C_GREY, ls="--", lw=.8); a.tick_params(axis="x", rotation=45)
ax[0].legend(frameon=False); plt.tight_layout(); save_fig("fig03_temporal_transfer")

# %%
# 5. E2 - WHICH FIELDS CARRY THE DRIFT
log("\nE2 feature-set ablation (train legacy <2022, test 2025+ new regime; control = legacy 2022-2023)")
ctrl = legacy[(legacy.t_crash >= "2022-01-01")]
rows = []
for tn, tgt in TARGETS.items():
    tr = legacy[legacy.t_crash < "2022-01-01"]
    for fs, ff in FEATURE_SETS.items():
        mdl = fit(tr, ff, tgt)
        for where, d in [("legacy_2022_23_control", ctrl), ("new_2025plus", test)]:
            m, _ = evaluate(mdl, d, ff, tgt, dict(target=tn, feature_set=fs, eval_set=where), ci=True)
            rows.append(m)
e2 = pd.DataFrame(rows); e2.to_csv(os.path.join(OUT_DIR, "E2_feature_set_ablation.csv"), index=False)
print(e2.pivot_table(index="feature_set", columns=["target", "eval_set"], values="auroc").round(3))

# reliance shift: mean |SHAP-like contribution| of the legacy-trained fault model, legacy vs new rows
sh = []
for tn, tgt in TARGETS.items():
    mdl = models_full[tn]
    for where, d in [("legacy", ctrl), ("new", test)]:
        s = d.sample(min(15000, len(d)), random_state=SEED)
        c = mdl.predict(s[FEATURE_SETS["full"]], pred_contrib=True)[:, :-1]
        sh.append(pd.DataFrame({"target": tn, "eval_set": where, "feature": FEATURE_SETS["full"], "mean_abs_contrib": np.abs(c).mean(0)}))
sh = pd.concat(sh); sh.to_csv(os.path.join(OUT_DIR, "E2_reliance_shift.csv"), index=False)
print(sh[sh.target == "fault"].pivot(index="feature", columns="eval_set", values="mean_abs_contrib").round(3)
      .sort_values("legacy", ascending=False).head(8))

# %%
# 6. E3 - MITIGATION STRATEGIES (test = new regime, 2025+)
log("\nE3 mitigation strategies")
rows = []; preds = {}
ff = FEATURE_SETS["full"] + ["regime_flag"]
for d_ in (legacy, adapt, test): d_["regime_flag"] = d_.is_new.astype(int)
for tn, tgt in TARGETS.items():
    feats_ = FEATURE_SETS["full"]
    pooled = pd.concat([legacy, adapt])
    wts = pd.Series(np.where(pooled.is_new == 1, ADAPT_WEIGHT, 1.0), index=pooled.index)
    strategies = {}
    base = fit(legacy, feats_, tgt)
    strategies["S1 legacy only"] = (base, feats_)
    strategies["S2 pooled"] = (fit(pooled, feats_, tgt), feats_)
    strategies["S3 pooled + regime feature"] = (fit(pooled, ff, tgt), ff)
    strategies[f"S4 pooled, new rows x{ADAPT_WEIGHT:g}"] = (fit(pooled, feats_, tgt, weight=wts), feats_)
    strategies["S5 new only (2024)"] = (fit(adapt, feats_, tgt), feats_)
    strategies["S6 legacy model + fine-tune"] = (fit(adapt, feats_, tgt, n_trees=N_TREES_FT, init_model=base), feats_)
    for name, (mdl, f_) in strategies.items():
        m, p = evaluate(mdl, test, f_, tgt, dict(target=tn, strategy=name), ci=True); rows.append(m); preds[(tn, name)] = p
    # S7: legacy model + logistic recalibration fitted on the adaptation year (cheap, keeps ranking)
    pa = base.predict(adapt[adapt[tgt].notna()][feats_]); ya = adapt[adapt[tgt].notna()][tgt].astype(int)
    lr = LogisticRegression(C=1e6).fit(np.log(np.clip(pa, 1e-6, 1 - 1e-6) / (1 - np.clip(pa, 1e-6, 1 - 1e-6))).reshape(-1, 1), ya)
    pt = base.predict(test[test[tgt].notna()][feats_]); lg = np.log(np.clip(pt, 1e-6, 1 - 1e-6) / (1 - np.clip(pt, 1e-6, 1 - 1e-6)))
    p7 = lr.predict_proba(lg.reshape(-1, 1))[:, 1]
    m, _ = evaluate(p7, test, feats_, tgt, dict(target=tn, strategy="S7 legacy model + recalibration"), ci=True); rows.append(m)
e3 = pd.DataFrame(rows); e3.to_csv(os.path.join(OUT_DIR, "E3_mitigation_strategies.csv"), index=False)
print(e3.pivot(index="strategy", columns="target", values=["auroc", "brier", "citl"]).round(3))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for a, (tn, col, lab) in zip(ax, [("fault", C_FAULT, "At-fault"), ("ksi", C_KSI, "Serious/fatal")]):
    d = e3[e3.target == tn].reset_index(drop=True)
    a.barh(range(len(d)), d.auroc, xerr=[d.auroc - d.auroc_lo, d.auroc_hi - d.auroc], color=col, capsize=2)
    a.set_xlim(max(0.5, d.auroc_lo.min() - .03), min(1, d.auroc_hi.max() + .02)); a.set_xlabel(f"AUROC ({lab})")
    a.set_yticks(range(len(d))); a.set_yticklabels(d.strategy if tn == "fault" else [""] * len(d)); a.invert_yaxis()
plt.tight_layout(); save_fig("fig04_mitigation_strategies")

# %%
# 7. E4 - PAIRED-DRIVER CONSISTENCY
log("\nE4 paired-driver consistency")
two = df[(df.n_drivers_in_crash == 2)].drop_duplicates("report_number").copy()
known = df[(df.n_drivers_in_crash == 2)].groupby("report_number").y_fault.agg(["count", "sum"])
known = known[known["count"] == 2]
tw = df[df.report_number.isin(known.index)].drop_duplicates("report_number").set_index("report_number")
tw["n_fault"] = known["sum"]
tw["year"] = tw.t_crash.dt.year
e4 = (tw.groupby(["regime", "year"]).n_fault.value_counts(normalize=True).unstack(fill_value=0)
      .rename(columns={0.0: "zero_at_fault", 1.0: "exactly_one", 2.0: "both"}))
e4["n_crashes"] = tw.groupby(["regime", "year"]).size()
e4.round(4).to_csv(os.path.join(OUT_DIR, "E4_paired_driver_consistency.csv"))
print(e4.round(3))

# %%
# 8. E5 - STRUCTURAL BREAK TESTS ON MONTHLY SERIES (+ placebo scan)
log("\nE5 structural break tests")
# share of two-driver crashes (both fault flags known) in which neither driver is flagged at fault
_tk = df[(df.n_drivers_in_crash == 2) & df.y_fault.notna()].groupby("report_number").agg(nf=("y_fault", "sum"), n=("y_fault", "size"), ym=("ym", "first"))
_tk = _tk[_tk.n == 2]
m = (df.groupby("ym").agg(fault=("y_fault", "mean"), ksi=("y_ksi", "mean"), n=("y_fault", "size"))
       .join(_tk.groupby("ym").nf.apply(lambda v: (v == 0).mean()).rename("zero_fault_2dr")))
m = m[m.n >= 200].copy(); m["t"] = np.arange(len(m)); m.index = m.index.to_timestamp()
BREAK = pd.Timestamp("2024-01-01")

def its(series, brk=BREAK, start=None):
    d = m if start is None else m[m.index >= start]
    X = pd.DataFrame({"const": 1.0, "t": d.t - d.t.mean(), "post": (d.index >= brk).astype(float)}, index=d.index)
    r = sm.OLS(d[series], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return r

rows = []
for s, lab in [("fault", "At-fault rate"), ("zero_fault_2dr", "Two-driver crashes with no at-fault"), ("ksi", "Serious/fatal rate")]:
    r = its(s); ci = r.conf_int().loc["post"]
    rows.append(dict(series=lab, level_shift=r.params["post"], ci_lo=ci[0], ci_hi=ci[1], t_hac=r.tvalues["post"], p=r.pvalues["post"],
                     pre_mean=m.loc[m.index < BREAK, s].mean(), post_mean=m.loc[m.index >= BREAK, s].mean()))
e5 = pd.DataFrame(rows); e5.to_csv(os.path.join(OUT_DIR, "E5_break_tests.csv"), index=False); print(e5.round(4))

# scan every candidate break month (>=24 months from the ends); the true break should dominate
scan = []
for k in m.index[24:-24]:
    for s in ("fault", "ksi"):
        r = its(s, brk=k); scan.append(dict(break_month=k, series=s, t_hac=r.tvalues["post"], shift=r.params["post"]))
scan = pd.DataFrame(scan); scan.to_csv(os.path.join(OUT_DIR, "E5_break_scan.csv"), index=False)
best = scan.loc[scan[scan.series == "fault"].t_hac.abs().idxmax()]
log(f"Strongest break in the at-fault series is at {best.break_month:%Y-%m} (|t|={abs(best.t_hac):.1f}); "
    f"serious/fatal series max |t| = {scan[scan.series == 'ksi'].t_hac.abs().max():.1f}")

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
ax[0].plot(m.index, m.fault, color=C_FAULT, label="At-fault"); ax[0].plot(m.index, m.zero_fault_2dr, color=C_FAULT, ls=":", label="Zero at-fault (2-driver)")
ax[0].plot(m.index, m.ksi * 10, color=C_KSI, label="Serious/fatal x10"); ax[0].axvline(BREAK, color=C_GREY, ls="--", lw=.8)
ax[0].set_ylabel("Monthly rate"); ax[0].legend(frameon=False, fontsize=8)
for s, col in [("fault", C_FAULT), ("ksi", C_KSI)]:
    d = scan[scan.series == s]; ax[1].plot(d.break_month, d.t_hac.abs(), color=col)
ax[1].axvline(BREAK, color=C_GREY, ls="--", lw=.8); ax[1].set_ylabel("|t| of level shift (HAC)")
plt.tight_layout(); save_fig("fig05_structural_break")

# %%
# 9. E6 - AGENCY HETEROGENEITY (new regime) + driver-level logit with crash-clustered SEs
log("\nE6 agency heterogeneity")
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z ** 2 / n; c = p + z ** 2 / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (c - h) / d, (c + h) / d
g = df[df.y_fault.notna()].groupby(["regime", "agency_h"]).y_fault.agg(["sum", "size"]).reset_index()
g = g[g["size"] >= 300]; g["rate"] = g["sum"] / g["size"]
g["lo"], g["hi"] = zip(*[wilson(k, n) for k, n in zip(g["sum"], g["size"])])
g.to_csv(os.path.join(OUT_DIR, "E6_agency_fault_rates.csv"), index=False); print(g.round(3))

d = df[(df.y_fault.notna()) & (df.agency_h == "montgomery_county_police")].copy()
d["post"] = d.is_new; d["tt"] = (d.t_crash - pd.Timestamp("2024-01-01")).dt.days / 365.25
d["tt_post"] = d.tt * d.post
d["lbds"] = (d.distraction_h == "looked_but_did_not_see").astype(int)
d["named_distr"] = d.distraction_h.isin(["inattentive_other", "phone_or_device", "in_vehicle_activity", "external_distraction"]).astype(int)
d["dist_x_post"] = d.named_distr * d.post
cols = ["post", "tt", "tt_post", "named_distr", "dist_x_post", "lbds"]
cols = [c for c in cols if d[c].nunique() > 1]          # 'lbds' is constant if preprocessing predates the separate category
if "lbds" not in cols: log("Note: no 'looked_but_did_not_see' rows found - rerun the updated preprocessing notebook to include it.")
X = sm.add_constant(d[cols].astype(float))
lg = sm.Logit(d.y_fault.astype(float), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.gid})
res = pd.DataFrame({"coef": lg.params, "se": lg.bse, "OR": np.exp(lg.params), "p": lg.pvalues}); res.to_csv(os.path.join(OUT_DIR, "E6_logit_MCPD_regime_coupling.csv"))
print(res.round(4))
log("Reading: 'named_distr' = coupling of a named distraction with at-fault in legacy; 'dist_x_post' = change of that coupling in the new regime.")

# %%
# 10. REPORT
lines = ["# Drift experiments - summary", "", f"Runtime {time.time() - t0:.0f}s. Train/adapt/test: legacy {len(legacy):,} / new-2024 {len(adapt):,} / new-2025+ {len(test):,} rows.", "",
         "## E1 AUROC by half-year (train legacy 2015-2021)", e1.pivot(index="half", columns="target", values="auroc").round(3).to_string(), "",
         "## E1 calibration-in-the-large (mean predicted - observed)", e1.pivot(index="half", columns="target", values="citl").round(3).to_string(), "",
         "## E2 feature-set ablation (AUROC)", e2.pivot_table(index="feature_set", columns=["target", "eval_set"], values="auroc").round(3).to_string(), "",
         "## E3 strategies on 2025+ new-regime test", e3.set_index(["target", "strategy"])[["auroc", "auroc_lo", "auroc_hi", "brier", "citl", "ece"]].round(3).to_string(), "",
         "## E4 paired-driver consistency", e4.round(3).to_string(), "",
         "## E5 break tests", e5.round(4).to_string(index=False), "",
         "## E6 logit (MCPD, crash-clustered SE)", res.round(4).to_string()]
open(os.path.join(OUT_DIR, "drift_experiments_report.md"), "w").write("\n".join(lines))
log(f"\nAll outputs saved in: {OUT_DIR}")
