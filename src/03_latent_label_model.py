# %% [markdown]
# # Step 4 - Latent-label model: how much of the at-fault drift is under-flagging?
# **Idea.** In a two-driver crash with one culprit, the recorded at-fault flag A is a noisy measurement of the true fault F:
# sensitivity s = P(A=1 | F=1), false-flag rate f = P(A=1 | F=0). If the crash has exactly one culprit, the pair counts
# (zero / one / two flagged) identify s and f in closed form. Legacy data give s close to 1; the new format gives the drift as an
# estimated sensitivity loss, with intervals, instead of only a drop in AUROC.
#
# E7  Closed-form estimates of s and f by regime, year and agency (+ sensitivity to crashes that truly have no culprit)
# E8  Simulation check: inject known label noise into legacy data and see if the estimator and the correction recover the truth
# E9  Who stops flagging? A transport model where s depends on agency, year, injury and collision type
# E10 Noise-corrected classifier vs naive models on 2025+ data (pair concordance, calibration)
#
# Reads `harmonised/crash_drivers_harmonised.csv`; writes to `<data folder>/experiments_latent/`. Runtime about 5 min.

# %%
# 1. SETUP
import os, sys, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm"], check=True); import lightgbm as lgb
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

PREP_DIR = os.environ.get("CRASH_PREP_DIR", "/content/drive/MyDrive/Research/crash driver data/harmonised")
OUT_DIR = os.environ.get("CRASH_LAT_DIR", os.path.join(os.path.dirname(PREP_DIR.rstrip("/")), "experiments_latent"))
FIG_DIR = os.path.join(OUT_DIR, "figures"); os.makedirs(FIG_DIR, exist_ok=True)
N_BOOT = int(os.environ.get("N_BOOT", 500))          # crash-level bootstrap for closed-form estimates
N_BOOT_MODEL = int(os.environ.get("N_BOOT_MODEL", 40))  # refits of the transport model
SEED = 42; N_TREES = 250
ADAPT = ("2024-01-01", "2025-01-01"); TEST_START = "2025-01-01"
LGB = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=0)
C1, C2, CG = "#1f4e79", "#c0504d", "#7f7f7f"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": False, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "savefig.dpi": 200, "savefig.bbox": "tight"})
def save_fig(n): plt.savefig(os.path.join(FIG_DIR, n + ".png")); plt.show(); plt.close()
def log(*a): print(*a, flush=True)
rng = np.random.default_rng(SEED)

# %%
# 2. LOAD, FEATURES, CRASH-LEVEL PAIR TABLE
t0 = time.time()
df = pd.read_csv(os.path.join(PREP_DIR, "crash_drivers_harmonised.csv"), low_memory=False, parse_dates=["crash_datetime"])
df["t_crash"] = df.groupby("report_number").crash_datetime.transform("min")
df["is_new"] = (df.regime == "new").astype(int)
df["year"] = df.t_crash.dt.year
df["gid"] = pd.factorize(df.report_number)[0]
top_make = df.make_h.value_counts().head(40).index
df["make_top"] = df.make_h.where(df.make_h.isin(top_make), "other")

CAT = ["collision_h", "movement_h", "route_h", "light_h", "control_h", "direction_h", "weather_h", "surface_h", "damage_h",
       "body_h", "impact_zone_h", "alcohol_susp", "drug_susp", "parked_vehicle", "license_state_group", "make_top"]
NUM = ["speed_limit", "vehicle_age", "crash_hour", "crash_dow", "crash_month"]
FEATS = CAT + NUM          # distraction_h and agency_h are deliberately excluded: they are the fields whose meaning changed
for c in CAT:
    df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
    df[c] = pd.Categorical(df[c], categories=sorted(df[c].dropna().unique()))

# two-driver crashes where both fault flags are known
two = df[(df.n_drivers_in_crash == 2) & df.y_fault.notna()]
cnt = two.groupby("report_number").y_fault.agg(["size", "sum"])
two = two[two.report_number.isin(cnt[cnt["size"] == 2].index)].copy()
cr = two.groupby("report_number").agg(nflag=("y_fault", "sum"), regime=("regime", "first"), year=("year", "first"),
                                      agency=("agency_h", "first"), t=("t_crash", "first"),
                                      injury=("report_type", lambda s: int((s != "property_damage").any())))
log(f"Two-driver crashes with both flags known: {len(cr):,} (legacy {int((cr.regime == 'legacy').sum()):,}, new {int((cr.regime == 'new').sum()):,})")

# %%
# 3. E7 - CLOSED-FORM ESTIMATES OF SENSITIVITY s AND FALSE-FLAG RATE f
def solve_sf(nflag):
    """exactly one true culprit: P00=(1-s)(1-f), P11=s*f  ->  s+f = 1+P11-P00, s*f = P11; take s >= f"""
    n = len(nflag); p00 = np.mean(nflag == 0); p11 = np.mean(nflag == 2)
    S = 1 + p11 - p00; disc = max(S * S - 4 * p11, 0.0); r = np.sqrt(disc)
    return (S + r) / 2, (S - r) / 2

def s_given_f(nflag, f, kappa=0.0):
    """fix f; allow a share kappa of crashes with truly no culprit: P00 = kappa(1-f)^2 + (1-kappa)(1-s)(1-f)"""
    p00 = np.mean(nflag == 0)
    return 1 - (p00 - kappa * (1 - f) ** 2) / ((1 - kappa) * (1 - f))

def est_row(nflag, label, n_boot=N_BOOT):
    nflag = np.asarray(nflag); s, f = solve_sf(nflag)
    b = np.array([solve_sf(nflag[rng.integers(0, len(nflag), len(nflag))]) for _ in range(n_boot)])
    return dict(**label, n_crashes=len(nflag), p_zero=np.mean(nflag == 0), p_one=np.mean(nflag == 1), p_both=np.mean(nflag == 2),
                s=s, s_lo=np.percentile(b[:, 0], 2.5), s_hi=np.percentile(b[:, 0], 97.5),
                f=f, f_lo=np.percentile(b[:, 1], 2.5), f_hi=np.percentile(b[:, 1], 97.5))

rows = []
for reg, d in cr.groupby("regime"): rows.append(est_row(d.nflag, dict(by="regime", level=reg)))
for (reg, y), d in cr.groupby(["regime", "year"]):
    if len(d) >= 500: rows.append(est_row(d.nflag, dict(by="regime_year", level=f"{reg}-{y}")))
for (reg, a), d in cr.groupby(["regime", "agency"]):
    if len(d) >= 300: rows.append(est_row(d.nflag, dict(by="regime_agency", level=f"{reg}-{a}")))
for (reg, k), d in cr.groupby(["regime", "injury"]):
    if len(d) >= 300: rows.append(est_row(d.nflag, dict(by="regime_injury", level=f"{reg}-{'injury' if k else 'property_damage'}")))
e7 = pd.DataFrame(rows); e7.to_csv(os.path.join(OUT_DIR, "E7_sensitivity_estimates.csv"), index=False)
print(e7[["by", "level", "n_crashes", "p_zero", "p_both", "s", "s_lo", "s_hi", "f"]].round(3).to_string(index=False))

# robustness: what if some crashes truly have no culprit?
f_leg = solve_sf(cr[cr.regime == "legacy"].nflag.values)[1]; kap = []
for k in [0, 0.005, 0.02, 0.05, 0.10, 0.20]:
    for reg in ["legacy", "new"]:
        kap.append(dict(regime=reg, kappa_true_no_culprit=k, s=s_given_f(cr[cr.regime == reg].nflag.values, f_leg, k)))
kap = pd.DataFrame(kap); kap.to_csv(os.path.join(OUT_DIR, "E7_kappa_sensitivity.csv"), index=False)
print("\nSensitivity of s to the share of crashes that truly have no culprit (f fixed at the legacy value):")
print(kap.pivot(index="kappa_true_no_culprit", columns="regime", values="s").round(3))

# %%
# 4. E8 - SIMULATION: DOES THE ESTIMATOR AND THE CORRECTION RECOVER KNOWN NOISE?
log("\nE8 simulation (legacy crashes with exactly one flag = taken as truth; noise injected by us)")
sig = lambda z: expit(z)

def make_obj(y, s, f):
    def obj(preds, ds):
        p = sig(preds); q = np.clip(f + (s - f) * p, 1e-6, 1 - 1e-6); dq = (s - f) * p * (1 - p)
        return (-y / q + (1 - y) / (1 - q)) * dq, np.maximum(dq ** 2 / (q * (1 - q)), 1e-6)
    return obj

def fit_corrected(d, feats, s_vec, f_vec, n_trees=N_TREES):
    d = d[d.y_fault.notna()]; y = d.y_fault.values.astype(float); s = np.asarray(s_vec, float); f = np.asarray(f_vec, float)
    p0 = np.clip((y.mean() - f.mean()) / (s.mean() - f.mean()), 0.02, 0.98); z0 = float(logit(p0))
    ds = lgb.Dataset(d[feats], label=y, init_score=np.full(len(d), z0), categorical_feature=[c for c in feats if c in CAT], free_raw_data=False)
    p = dict(LGB); p["objective"] = make_obj(y, s, f)
    return lgb.train(p, ds, num_boost_round=n_trees), z0

def predict_corrected(m, z0, d, feats): return sig(m.predict(d[feats], raw_score=True) + z0)

def fit_naive(d, feats, n_trees=N_TREES, weight=None):
    d = d[d.y_fault.notna()]
    ds = lgb.Dataset(d[feats], label=d.y_fault.astype(int), categorical_feature=[c for c in feats if c in CAT], free_raw_data=False)
    return lgb.train(dict(LGB, objective="binary"), ds, num_boost_round=n_trees)

one = cr[(cr.regime == "legacy") & (cr.nflag == 1)].index
clean = two[two.report_number.isin(one)].copy()
clean["truth"] = clean.y_fault.astype(int)
tr0, te0 = clean[clean.t_crash < "2020-01-01"], clean[clean.t_crash >= "2020-01-01"]
sim = []
for s_true in [0.9, 0.7, 0.5]:
    for seed in range(3):
        r = np.random.default_rng(1000 + seed); f_true = 0.02
        def corrupt(d):
            u = r.random(len(d)); return np.where(d.truth.values == 1, u < s_true, u < f_true).astype(float)
        tr, te = tr0.copy(), te0.copy(); tr["y_fault"] = corrupt(tr); te["y_fault"] = corrupt(te)
        # estimate (s, f) from the training pair counts only
        nf = tr.groupby("report_number").y_fault.sum().values; s_hat, f_hat = solve_sf(nf)
        nm = fit_naive(tr, FEATS, 150); pn = nm.predict(te[FEATS])
        cm, z0 = fit_corrected(tr, FEATS, np.full(len(tr), s_hat), np.full(len(tr), f_hat), 150); pc = predict_corrected(cm, z0, te, FEATS)
        for name, p in [("naive", pn), ("corrected", pc)]:
            sim.append(dict(s_true=s_true, seed=seed, model=name, s_hat=s_hat, f_hat=f_hat, auroc_vs_truth=roc_auc_score(te.truth, p),
                            brier_vs_truth=np.mean((p - te.truth) ** 2), mean_p=p.mean(), true_prevalence=te.truth.mean()))
sim = pd.DataFrame(sim); sim.to_csv(os.path.join(OUT_DIR, "E8_simulation_recovery.csv"), index=False)
print(sim.groupby(["s_true", "model"])[["s_hat", "f_hat", "auroc_vs_truth", "brier_vs_truth", "mean_p"]].mean().round(3))

# heterogeneous sensitivity: injury crashes keep s, property-damage-only crashes lose 0.2
het = []
for s_inj in [0.9, 0.7]:
    r = np.random.default_rng(7); d = cr[(cr.regime == "legacy") & (cr.nflag == 1)].copy()
    s_row = np.where(d.injury == 1, s_inj, s_inj - 0.2)
    # simulate pair flags for a crash with exactly one culprit
    a = (r.random(len(d)) < s_row).astype(int) + (r.random(len(d)) < 0.02).astype(int); d["nsim"] = a
    for k, nm_ in [(1, "injury"), (0, "property_damage")]:
        sh, fh = solve_sf(d[d.injury == k].nsim.values)
        het.append(dict(s_true=s_inj if k else s_inj - 0.2, group=nm_, s_hat=sh, f_hat=fh))
het = pd.DataFrame(het); het.to_csv(os.path.join(OUT_DIR, "E8_heterogeneous_sensitivity_recovery.csv"), index=False); print(het.round(3))

# %%
# 5. E9 - WHO STOPS FLAGGING?  transport model: P(F|x) from legacy, s depends on covariates in the new format
log("\nE9 transport model for sensitivity")
leg_tr = two[two.regime == "legacy"]
base = fit_naive(leg_tr, FEATS)            # legacy flags ~ truth (97% of legacy two-driver crashes have exactly one flag)
new2 = two[two.regime == "new"].copy(); new2["p_hat"] = base.predict(new2[FEATS])
s_all, f_new = solve_sf(cr[cr.regime == "new"].nflag.values)
log(f"New format: pooled s = {s_all:.3f}, f = {f_new:.3f}; mean transported P(F|x) = {new2.p_hat.mean():.3f}")

top_col = new2.collision_h.value_counts().head(6).index.tolist()
W = pd.DataFrame({"const": 1.0}, index=new2.index)
for y in (2025, 2026): W[f"year_{y}"] = (new2.year == y).astype(float)
for a in ["gaithersburg_police", "rockville_police", "takoma_park_police", "park_police"]: W[f"agency_{a}"] = (new2.agency_h == a).astype(float)
W["injury_crash"] = (new2.report_type != "property_damage").astype(float)
for c in top_col[1:]: W[f"collision_{c}"] = (new2.collision_h == c).astype(float)
A = new2.y_fault.values.astype(float); P = new2.p_hat.values; Wm = W.values; gid = new2.gid.values
log("Reference level: Montgomery County Police, 2024, property-damage crash, collision = " + str(top_col[0]))

def fit_s(weights, f, g0=None):
    def nll(g):
        s = expit(Wm @ g); q = np.clip(f + (s - f) * P, 1e-6, 1 - 1e-6)
        ll = -(A * np.log(q) + (1 - A) * np.log(1 - q)); dq = P * s * (1 - s)
        dl = (-A / q + (1 - A) / (1 - q)) * dq
        return (weights * ll).sum(), (Wm * (weights * dl)[:, None]).sum(0)
    g0 = np.zeros(Wm.shape[1]) if g0 is None else g0
    return minimize(nll, g0, jac=True, method="L-BFGS-B").x

g_hat = fit_s(np.ones(len(A)), f_new)
ug, inv = np.unique(gid, return_inverse=True); G = len(ug)
boots = []
for b in range(N_BOOT_MODEL):
    w = np.bincount(rng.integers(0, G, G), minlength=G)[inv].astype(float); boots.append(fit_s(w, f_new, g_hat))
boots = np.array(boots)
new2["s_i"] = expit(Wm @ g_hat)
coef = pd.DataFrame({"coef": g_hat, "se_boot": boots.std(0), "odds_ratio": np.exp(g_hat)}, index=W.columns)
coef["or_lo"], coef["or_hi"] = np.exp(np.percentile(boots, 2.5, 0)), np.exp(np.percentile(boots, 97.5, 0))
coef.to_csv(os.path.join(OUT_DIR, "E9_sensitivity_model_coefficients.csv")); print(coef.round(3))

def group_s(col):
    out = []
    for lv, idx in new2.groupby(col).groups.items():
        pos = new2.index.get_indexer(idx); sb = [expit(Wm[pos] @ bb).mean() for bb in boots]
        out.append(dict(variable=col, level=lv, n=len(pos), s=new2.loc[idx, "s_i"].mean(), lo=np.percentile(sb, 2.5), hi=np.percentile(sb, 97.5)))
    return out
new2["injury_crash"] = np.where(new2.report_type != "property_damage", "injury", "property_damage")
e9 = pd.DataFrame(sum([group_s(c) for c in ["year", "agency_h", "injury_crash", "collision_h"]], []))
e9 = e9[e9.n >= 300]; e9.to_csv(os.path.join(OUT_DIR, "E9_sensitivity_by_group.csv"), index=False); print(e9.round(3).to_string(index=False))

# %%
# 6. E10 - NOISE-CORRECTED CLASSIFIER vs NAIVE MODELS (test: new format, 2025+, two-driver crashes)
log("\nE10 corrected vs naive models")
adapt = two[(two.regime == "new") & (two.t_crash >= ADAPT[0]) & (two.t_crash < ADAPT[1])]
test = two[(two.regime == "new") & (two.t_crash >= TEST_START)].copy()
leg = two[two.regime == "legacy"]
s_L, f_L = solve_sf(cr.loc[leg.report_number.unique()].nflag.values)
s_N, f_N = solve_sf(cr.loc[adapt.report_number.unique()].nflag.values)
log(f"Plug-in noise rates: legacy s={s_L:.3f} f={f_L:.3f} | new-2024 s={s_N:.3f} f={f_N:.3f}")

def concordance(d, p):
    t = d[["report_number", "y_fault"]].copy(); t["p"] = p
    t = t[t.groupby("report_number").y_fault.transform("sum") == 1]
    fl = t[t.y_fault == 1].set_index("report_number").p; un = t[t.y_fault == 0].set_index("report_number").p.reindex(fl.index)
    diff = (fl - un).values; c = (diff > 0).astype(float) + 0.5 * (diff == 0)
    b = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(N_BOOT)]
    return c.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(c)

pool = pd.concat([leg, adapt])
s_pool = np.where(pool.is_new == 1, s_N, s_L); f_pool = np.where(pool.is_new == 1, f_N, f_L)
models = {}
models["M1 legacy only, naive"] = ("naive", fit_naive(leg, FEATS))
models["M2 pooled, naive"] = ("naive", fit_naive(pool, FEATS))
models["M3 new-2024 only, naive"] = ("naive", fit_naive(adapt, FEATS))
models["M4 pooled, noise-corrected"] = ("corr",) + fit_corrected(pool, FEATS, s_pool, f_pool)
models["M5 new-2024 only, noise-corrected"] = ("corr",) + fit_corrected(adapt, FEATS, np.full(len(adapt), s_N), np.full(len(adapt), f_N))
rows = []
for name, spec in models.items():
    p = spec[1].predict(test[FEATS]) if spec[0] == "naive" else predict_corrected(spec[1], spec[2], test, FEATS)
    c, lo, hi, n_pairs = concordance(test, p)
    ok = test.y_fault.notna()
    rows.append(dict(model=name, pair_concordance=c, conc_lo=lo, conc_hi=hi, n_exactly_one_flag_crashes=n_pairs,
                     auroc_vs_flags=roc_auc_score(test.y_fault, p), mean_p=p.mean(), gap_from_half=p.mean() - 0.5))
e10 = pd.DataFrame(rows); e10.to_csv(os.path.join(OUT_DIR, "E10_corrected_vs_naive.csv"), index=False)
print(e10.round(3).to_string(index=False))
log("Reading: in a two-driver crash with one culprit the true mean P(fault) is 0.5, so 'gap_from_half' shows calibration to the TRUE label;")
log("'pair_concordance' = how often the model ranks the flagged driver above the unflagged one in crashes with exactly one flag.")

# %%
# 7. FIGURE
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
d = e7[e7.by == "regime_year"].copy(); d["x"] = d.level
ax[0].errorbar(range(len(d)), d.s, yerr=[d.s - d.s_lo, d.s_hi - d.s], fmt="o-", color=C1, capsize=2)
ax[0].set_xticks(range(len(d))); ax[0].set_xticklabels([l.replace("legacy-", "L").replace("new-", "N") for l in d.level], rotation=60)
ax[0].set_ylabel("Estimated sensitivity s"); ax[0].set_ylim(0.3, 1.02)
g = e9[e9.variable == "agency_h"].sort_values("s")
ax[1].errorbar(g.s, range(len(g)), xerr=[g.s - g.lo, g.hi - g.s], fmt="o", color=C1, capsize=2)
ax[1].set_yticks(range(len(g))); ax[1].set_yticklabels([x.replace("_police", "") for x in g.level]); ax[1].set_xlabel("s, new format (by agency)")
sm_ = sim[sim.model == "naive"].groupby("s_true").s_hat.agg(["mean", "std"]).reset_index()
ax[2].errorbar(sm_.s_true, sm_["mean"], yerr=sm_["std"], fmt="o", color=C2, capsize=2); ax[2].plot([.4, 1], [.4, 1], color=CG, lw=.8)
ax[2].set_xlabel("True s (simulation)"); ax[2].set_ylabel("Estimated s")
plt.tight_layout(); save_fig("fig06_latent_sensitivity")

# %%
# 8. REPORT
L = ["# Latent-label experiments - summary", "", f"Runtime {time.time() - t0:.0f}s", "",
     "## E7 sensitivity estimates", e7[["by", "level", "n_crashes", "p_zero", "p_both", "s", "s_lo", "s_hi", "f"]].round(3).to_string(index=False), "",
     "## E7 robustness to crashes with truly no culprit", kap.pivot(index="kappa_true_no_culprit", columns="regime", values="s").round(3).to_string(), "",
     "## E8 simulation", sim.groupby(["s_true", "model"])[["s_hat", "auroc_vs_truth", "brier_vs_truth", "mean_p"]].mean().round(3).to_string(), "",
     "## E9 sensitivity by group (new format)", e9.round(3).to_string(index=False), "",
     "## E10 corrected vs naive", e10.round(3).to_string(index=False)]
open(os.path.join(OUT_DIR, "latent_label_report.md"), "w").write("\n".join(L))
log(f"\nAll outputs saved in: {OUT_DIR}")
