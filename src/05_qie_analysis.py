# %% [markdown]
# # Step 6 - Quasi-induced exposure across the format change (E11)
# **Question.** Does the 2024 change in the at-fault flag change the answer of a standard safety method?
# Quasi-induced exposure (QIE) uses two-vehicle crashes: the at-fault drivers describe who causes crashes and the
# not-at-fault drivers serve as a sample of drivers exposed to traffic. The relative involvement ratio (RAR) of a group g is
#     RAR_g = P(g | at fault) / P(g | not at fault).
# RAR > 1 means the group is over-represented among drivers who cause crashes.
#
# Three estimators are compared on two-driver crashes (both flags known, no parked vehicle, group known for both drivers):
# * **Naive**      every driver as recorded: flagged = at fault, unflagged = not at fault.
# * **Clean-pair** the usual QIE restriction to crashes with exactly one flagged driver.
# * **Corrected**  every driver, with the recorded flag corrected for the sensitivity s and false-flag rate f estimated
#                  from the pair counts of the same sample (one culprit per crash, flags independent given true fault):
#                  RAR = [(1-f) N1 - f N0] / [s N0 - (1-s) N1],  N1 / N0 = flagged / unflagged drivers in group g.
# Periods: legacy 2020-2021 and 2022-2023, and the new format 2024-2026. The legacy-legacy comparison is a placebo that shows
# how much RARs move without a format change. Groups use only fields whose coding did not change between formats:
# non-Maryland licence, vehicle 15 or more years old, heavy vehicle (truck or bus, compared with cars, SUVs, pickups and vans).
#
# A final section locates the lost flags: drivers with neither a licence state nor vehicle details (typically drivers who
# could not be identified at the scene) and their share flagged at fault, by period and by year.
# A last check uses the driver contributing actions that the new format records in the raw `Circumstance` field.
#
# Reads `harmonised/crash_drivers_harmonised.csv` and the raw `export.csv` (Circumstance field only); writes to `<data folder>/experiments_qie/`. Runtime about 1-2 min.

# %%
# 1. SETUP
import os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

PREP_DIR = os.environ.get("CRASH_PREP_DIR", "/content/drive/MyDrive/Research/crash driver data/harmonised")
OUT_DIR = os.environ.get("CRASH_QIE_DIR", os.path.join(os.path.dirname(PREP_DIR.rstrip("/")), "experiments_qie"))
FIG_DIR = os.path.join(OUT_DIR, "figures"); os.makedirs(FIG_DIR, exist_ok=True)
N_BOOT = int(os.environ.get("N_BOOT_QIE", 1000))     # crash-level bootstrap resamples
SEED = 42
PERIODS = {"L1": ("legacy", 2020, 2021, "2020–21\n(legacy)"),
           "L2": ("legacy", 2022, 2023, "2022–23\n(legacy)"),
           "N":  ("new",    2024, 2026, "2024–26\n(new)")}
GROUPS = {"nonmd": "Non-Maryland licence", "old": "Vehicle aged 15 years or more", "heavy": "Heavy vehicle (truck or bus)"}
EST = ["naive", "clean", "corrected"]

# plot style: white background, no grid, Times-style serif ~14 pt, colour-blind-safe (Okabe-Ito) colours plus distinct markers
COL = {"naive": "#D55E00", "clean": "#0072B2", "corrected": "#009E73"}
MRK = {"naive": "o", "clean": "s", "corrected": "^"}
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})

# %%
# 2. SAMPLE: two-driver crashes, both flags known, no parked vehicle
d = pd.read_csv(os.path.join(PREP_DIR, "crash_drivers_harmonised.csv"), low_memory=False, parse_dates=["crash_datetime"])
two = d[d.two_driver_crash == 1].copy()
crash = two.groupby("report_number")
two = two[crash.y_fault.transform(lambda s: s.notna().all()) & ~crash.parked_vehicle.transform(lambda s: (s == "yes").any())].copy()
two["year"] = two.groupby("report_number").crash_datetime.transform("min").dt.year
lic = two.license_state_group
two["nonmd"] = np.where(lic == "MD", 0.0, np.where(lic.isin(["DC", "VA", "other_us_or_foreign"]), 1.0, np.nan))
two["old"] = np.where(two.vehicle_age.isna(), np.nan, (two.vehicle_age >= 15).astype(float))
heavy = two.body_h.isin(["truck_medium_heavy", "bus_transit", "bus_school", "bus_other"])
light = two.body_h.isin(["passenger_car", "suv", "pickup", "van"])
two["heavy"] = np.where(heavy, 1.0, np.where(light, 0.0, np.nan))
print(f"Two-driver crashes in the QIE sample: {two.report_number.nunique():,}")


def crash_arrays(sub, g):
    """Per crash (group known for both drivers): flagged and unflagged drivers in group g, and number of flagged drivers."""
    x = sub[["report_number", g, "y_fault"]]
    x = x[x.groupby("report_number")[g].transform(lambda s: s.notna().all())]
    x = x.assign(a1=(x.y_fault == 1) & (x[g] == 1), a0=(x.y_fault == 0) & (x[g] == 1))
    c = x.groupby("report_number").agg(a1=("a1", "sum"), a0=("a0", "sum"), nflag=("y_fault", "sum"))
    return c.a1.to_numpy(float), c.a0.to_numpy(float), c.nflag.to_numpy(int)


def estimates(a1, a0, nf, w):
    """RARs of the three estimators and (s, f) for crash weights w (w = 1 for the point estimate)."""
    C = w.sum(); N1 = (w * a1).sum(); N0 = (w * a0).sum()
    m1 = (w * nf).sum(); m0 = 2 * C - m1                        # flagged / unflagged drivers
    naive = (N1 / m1) / (N0 / m0)
    cl = nf == 1
    clean = (w * a1 * cl).sum() / (w * a0 * cl).sum()           # one culprit and one non-culprit per clean crash
    p0, p2 = (w * (nf == 0)).sum() / C, (w * (nf == 2)).sum() / C
    S = 1 + p2 - p0; r = np.sqrt(max(S * S - 4 * p2, 0.0)); s, f = (S + r) / 2, (S - r) / 2
    corrected = ((1 - f) * N1 - f * N0) / (s * N0 - (1 - s) * N1)
    return {"naive": naive, "clean": clean, "corrected": corrected, "s": s, "f": f}


def boot(a1, a0, nf, rng, n=N_BOOT):
    k = len(a1); out = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
        out.append(estimates(a1, a0, nf, w))
    return pd.DataFrame(out)

# %%
# 3. RAR BY PERIOD, CHANGE ACROSS THE FORMAT CHANGE AND PLACEBO CHANGE
rng = np.random.default_rng(SEED)
rows, reps = [], {}
for g in GROUPS:
    for p, (reg, y0, y1, _) in PERIODS.items():
        sub = two[(two.regime == reg) & two.year.between(y0, y1)]
        a1, a0, nf = crash_arrays(sub, g)
        pt = estimates(a1, a0, nf, np.ones(len(a1))); bt = boot(a1, a0, nf, rng); reps[(g, p)] = bt
        for e in EST:
            rows.append(dict(group=g, period=p, estimator=e, rar=pt[e], lo=bt[e].quantile(0.025), hi=bt[e].quantile(0.975),
                             n_crashes=len(a1), s=pt["s"], f=pt["f"], share_clean=(nf == 1).mean()))
by_period = pd.DataFrame(rows)
by_period.to_csv(os.path.join(OUT_DIR, "E11_qie_by_period.csv"), index=False)

chg = []
for g in GROUPS:
    for e in EST:
        for name, (a, b) in {"format change (2024-26 new vs 2022-23 legacy)": ("N", "L2"),
                             "placebo (2022-23 vs 2020-21, both legacy)": ("L2", "L1")}.items():
            pa = by_period.query("group==@g and period==@a and estimator==@e").rar.iloc[0]
            pb = by_period.query("group==@g and period==@b and estimator==@e").rar.iloc[0]
            ratio_b = reps[(g, a)][e].to_numpy() / reps[(g, b)][e].to_numpy()     # independent samples
            chg.append(dict(group=g, estimator=e, comparison=name, ratio=pa / pb,
                            lo=np.quantile(ratio_b, 0.025), hi=np.quantile(ratio_b, 0.975)))
change = pd.DataFrame(chg)
change["excludes_1"] = (change.lo > 1) | (change.hi < 1)
change.to_csv(os.path.join(OUT_DIR, "E11_qie_change.csv"), index=False)
print(change.round(3).to_string())

# %%
# 4. MODEL CHECK: does the measurement model predict the naive estimate in the new format?
# If the legacy clean-pair shares p1 = P(g | culprit), p0 = P(g | non-culprit) still held in 2024-26, recording with the
# new-format (s, f) would give a naive RAR of [(s p1 + f p0)/(s + f)] / [((1-s) p1 + (1-f) p0)/(2 - s - f)].
chk = []
for g in GROUPS:
    legx = two[(two.regime == "legacy") & two.year.between(2022, 2023)]; newx = two[(two.regime == "new") & two.year.between(2024, 2026)]
    la1, la0, lnf = crash_arrays(legx, g); na1, na0, nnf = crash_arrays(newx, g)
    def pred(wl, wn):
        cl = lnf == 1; Cc = (wl * cl).sum(); p1 = (wl * la1 * cl).sum() / Cc; p0 = (wl * la0 * cl).sum() / Cc
        e = estimates(na1, na0, nnf, wn); s, f = e["s"], e["f"]
        return ((s * p1 + f * p0) / (s + f)) / (((1 - s) * p1 + (1 - f) * p0) / (2 - s - f)), e["naive"]
    pr, ob = pred(np.ones(len(la1)), np.ones(len(na1)))
    bp = [pred(np.bincount(rng.integers(0, len(la1), len(la1)), minlength=len(la1)).astype(float),
               np.bincount(rng.integers(0, len(na1), len(na1)), minlength=len(na1)).astype(float)) for _ in range(N_BOOT)]
    bp = np.array(bp)
    chk.append(dict(group=g, predicted_naive=pr, pred_lo=np.quantile(bp[:, 0], .025), pred_hi=np.quantile(bp[:, 0], .975),
                    observed_naive=ob, obs_lo=np.quantile(bp[:, 1], .025), obs_hi=np.quantile(bp[:, 1], .975)))
model_check = pd.DataFrame(chk)
model_check.to_csv(os.path.join(OUT_DIR, "E11_qie_model_check.csv"), index=False)
print(model_check.round(3).to_string())

# %%
# 5. YEAR-BY-YEAR RAR (supplementary series)
yr_rows = []
for g in GROUPS:
    for y in range(2015, 2027):
        for reg in ("legacy", "new"):
            sub = two[(two.year == y) & (two.regime == reg)]
            if sub.report_number.nunique() < 1000:
                continue
            a1, a0, nf = crash_arrays(sub, g)
            pt = estimates(a1, a0, nf, np.ones(len(a1))); bt = boot(a1, a0, nf, rng, n=min(N_BOOT, 300))
            for e in EST:
                yr_rows.append(dict(group=g, year=y, regime=reg, estimator=e, rar=pt[e], lo=bt[e].quantile(.025),
                                    hi=bt[e].quantile(.975), n_crashes=len(a1), s=pt["s"]))
by_year = pd.DataFrame(yr_rows)
by_year.to_csv(os.path.join(OUT_DIR, "E11_qie_by_year.csv"), index=False)

# %%
# 6. WHERE THE FLAG WAS LOST: drivers who could not be identified
# A driver with neither a licence state nor vehicle details (vehicle year or make) is typical of a driver who left the
# scene or could not otherwise be identified. Drivers are split into three classes:
#   unidentified  - no licence state and no vehicle details
#   licence only  - no licence state, vehicle details recorded
#   identified    - licence state recorded
# For each class and period: drivers, drivers per year, share flagged at fault (Wilson 95% interval), and, for crashes with
# one driver of the class, the share in which the OTHER driver is flagged and the share with no flagged driver. If the flag
# moved to the other driver, the first would rise; if it was simply not recorded, the second rises.
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / den, (c + h) / den
lic_miss = two.license_state_group.isin(["missing", "unknown"])
veh_miss = two.vehicle_year.isna() | two.make_h.isna()
two["id_class"] = np.select([lic_miss & veh_miss, lic_miss], ["unidentified", "licence only"], "identified")
two["nflag"] = two.groupby("report_number").y_fault.transform("sum")
CLASSES = ["unidentified", "licence only", "identified"]
ud = []
for p, (reg, y0, y1, _) in PERIODS.items():
    sub = two[(two.regime == reg) & two.year.between(y0, y1)]
    n_years = sub.year.nunique() if p != "N" else (sub.crash_datetime.max() - pd.Timestamp("2024-01-01")).days / 365.25
    for c in CLASSES:
        x = sub[sub.id_class == c]; k, n = int(x.y_fault.sum()), len(x); lo, hi = wilson(k, n)
        one = sub[sub.groupby("report_number").id_class.transform(lambda v: (v == c).sum() == 1)]   # crashes with one such driver
        oth = one[one.id_class != c]
        ud.append(dict(period=p, driver_class=c, drivers=n, drivers_per_year=n / n_years, share_of_drivers=n / len(sub),
                       flagged_at_fault=k / n, lo=lo, hi=hi,
                       other_driver_flagged=oth.y_fault.mean() if c != "identified" else np.nan,
                       crashes_no_flag=(one[one.id_class == c].nflag == 0).mean() if c != "identified" else np.nan))
unident = pd.DataFrame(ud)
unident.to_csv(os.path.join(OUT_DIR, "E11_unidentified_drivers.csv"), index=False)
print(unident.round(3).to_string())

# share of all new-format zero-flag crashes that involve an unidentified driver
newp = two[(two.regime == "new") & two.year.between(2024, 2026)]
zc = newp[newp.nflag == 0].groupby("report_number").id_class.agg(lambda v: ((v == "unidentified").any(), (v != "identified").any()))
print(f"New-format zero-flag crashes: {len(zc):,}; with an unidentified driver: {zc.str[0].mean():.1%}; "
      f"with a driver without a licence state: {zc.str[1].mean():.1%}")

# year-by-year share flagged at fault by class (supplementary series)
uy = []
for (y, reg, c), x in two.groupby(["year", "regime", "id_class"]):
    if len(x) < 50:
        continue
    k, n = int(x.y_fault.sum()), len(x); lo, hi = wilson(k, n)
    uy.append(dict(year=y, regime=reg, driver_class=c, drivers=n, flagged_at_fault=k / n, lo=lo, hi=hi))
unident_year = pd.DataFrame(uy)
unident_year.to_csv(os.path.join(OUT_DIR, "E11_unidentified_by_year.csv"), index=False)

# %%
# 7. A SECOND INDICATOR IN THE NEW FORMAT: DRIVER CONTRIBUTING ACTIONS
# In the new format the raw `Circumstance` field lists driver contributing actions (for example "Failed to Yield
# Right-of-Way", "Followed Too Closely"); in the legacy format it describes road and weather conditions only. If the
# zero-flag crashes of the new format were crashes without a culprit, a contributing action would be recorded in them about
# as rarely as for unflagged drivers. The evasive action "Swerved or Avoided Due to ..." is not counted.
RAW_PATH = os.environ.get("CRASH_DATA_PATH", os.path.join(os.path.dirname(PREP_DIR.rstrip("/")), "export.csv"))
EVASIVE = "Swerved or Avoided Due to Wind, Slippery Surface, Motor Vehicle, Object, Non-Motorist in Roadway, etc"
if os.path.exists(RAW_PATH):
    circ = pd.read_csv(RAW_PATH, usecols=["Person ID", "Circumstance"], low_memory=False).rename(
        columns={"Person ID": "person_id", "Circumstance": "circumstance"})
    rest = circ.circumstance.fillna("").str.replace(EVASIVE, "", regex=False).str.replace(r"[,\s]", "", regex=True)
    circ["dca"] = rest.str.len() > 0
    two = two.drop(columns=["dca"], errors="ignore").merge(circ[["person_id", "dca"]], on="person_id", how="left")
    newp = two[(two.regime == "new") & two.year.between(2024, 2026)].copy()
    newp["ndca"] = newp.groupby("report_number").dca.transform("sum")
    newp["unid_in_crash"] = newp.groupby("report_number").id_class.transform(lambda v: (v == "unidentified").any())
    groups = {"flagged drivers": newp.y_fault == 1,
              "unflagged drivers, one-flag crashes": (newp.y_fault == 0) & (newp.nflag == 1),
              "drivers in zero-flag crashes": newp.nflag == 0,
              "unidentified drivers": newp.id_class == "unidentified",
              "other driver, crashes with an unidentified driver": newp.unid_in_crash & (newp.id_class != "unidentified")}
    rows = []
    for name, m in groups.items():
        k, n = int(newp.loc[m, "dca"].sum()), int(m.sum()); lo, hi = wilson(k, n)
        rows.append(dict(level="driver", group=name, year="2024-26", n=n, share_with_action=k / n, lo=lo, hi=hi))
    cr = newp.groupby("report_number").agg(nflag=("nflag", "first"), ndca=("ndca", "first"), year=("year", "first"))
    for yr in ["2024-26", 2024, 2025, 2026]:
        c = cr if yr == "2024-26" else cr[cr.year == yr]
        for nf in (0, 1, 2):
            x = c[c.nflag == nf]; k, n = int((x.ndca > 0).sum()), len(x); lo, hi = wilson(k, n)
            rows.append(dict(level="crash", group=f"{nf} flagged", year=yr, n=n, share_with_action=k / n, lo=lo, hi=hi,
                             share_one_action=(x.ndca == 1).mean(), share_two_actions=(x.ndca == 2).mean()))
    one = newp[(newp.nflag == 1) & (newp.ndca == 1) & newp.dca]
    rows.append(dict(level="crash", group="one flagged, one action: action on flagged driver", year="2024-26", n=len(one),
                     share_with_action=one.y_fault.mean(), lo=wilson(int(one.y_fault.sum()), len(one))[0],
                     hi=wilson(int(one.y_fault.sum()), len(one))[1]))
    actions = pd.DataFrame(rows)
    actions.to_csv(os.path.join(OUT_DIR, "E11_contributing_actions.csv"), index=False)
    print(actions.round(3).to_string())
else:
    print("Raw file not found at", RAW_PATH, "- contributing-action check skipped (set CRASH_DATA_PATH).")

# %%
# 8. FIGURES: one file per panel, plus a stacked vertical version
def draw(ax, g, label):
    xs = np.arange(len(PERIODS)); off = {"naive": -0.18, "clean": 0.0, "corrected": 0.18}
    for e in EST:
        q = by_period[(by_period.group == g) & (by_period.estimator == e)].set_index("period").loc[list(PERIODS)]
        ax.errorbar(xs + off[e], q.rar, yerr=[q.rar - q.lo, q.hi - q.rar], fmt=MRK[e], color=COL[e], ms=8,
                    capsize=4, lw=1.6, mfc=COL[e] if e != "naive" else "white", mew=1.6)
    ax.axhline(1, color="#555555", lw=1, ls=":")
    ax.axvline(1.5, color="#999999", lw=1, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels([v[3] for v in PERIODS.values()])
    ax.set_xlim(-0.5, len(PERIODS) - 0.5); ax.set_ylabel("Relative involvement ratio")
    ax.set_title(label, loc="left", fontsize=15, pad=8)   # label above the axes: never overlaps data or reference lines

for i, g in enumerate(GROUPS):
    fig, ax = plt.subplots(figsize=(6.3, 4.6)); draw(ax, g, f"({'abc'[i]})")
    fig.savefig(os.path.join(FIG_DIR, f"Fig_E11{'abc'[i]}.png")); plt.close(fig)
fig, axs = plt.subplots(3, 1, figsize=(6.3, 13.0))
for i, g in enumerate(GROUPS):
    draw(axs[i], g, f"({'abc'[i]})")
fig.tight_layout(h_pad=2.0); fig.savefig(os.path.join(FIG_DIR, "Fig_E11_stacked.png")); plt.close(fig)

def draw_year(ax, g, label):
    for e in EST:
        q = by_year[(by_year.group == g) & (by_year.estimator == e)].sort_values("year")
        for reg, ls in (("legacy", "-"), ("new", "-")):
            r = q[q.regime == reg]
            ax.errorbar(r.year + {"naive": -0.15, "clean": 0, "corrected": 0.15}[e], r.rar, yerr=[r.rar - r.lo, r.hi - r.rar],
                        fmt=MRK[e] + ls, color=COL[e], ms=6, capsize=2.5, lw=1.3, mfc=COL[e] if e != "naive" else "white")
    ax.axvline(2023.5, color="#999999", lw=1, ls="--"); ax.axhline(1, color="#555555", lw=1, ls=":")
    ax.set_xticks(range(2015, 2027, 2)); ax.set_xlabel("Crash year"); ax.set_ylabel("Relative involvement ratio")
    ax.set_title(label, loc="left", fontsize=15, pad=8)   # label above the axes: never overlaps data or reference lines

for i, g in enumerate(GROUPS):
    fig, ax = plt.subplots(figsize=(7.5, 4.6)); draw_year(ax, g, f"({'abc'[i]})")
    fig.savefig(os.path.join(FIG_DIR, f"Fig_E11_year_{'abc'[i]}.png")); plt.close(fig)
fig, axs = plt.subplots(3, 1, figsize=(7.5, 13.0))
for i, g in enumerate(GROUPS):
    draw_year(axs[i], g, f"({'abc'[i]})")
fig.tight_layout(h_pad=2.0); fig.savefig(os.path.join(FIG_DIR, "Fig_E11_year_stacked.png")); plt.close(fig)
# share flagged at fault by driver class and crash year (single panel)
CCOL = {"unidentified": "#CC79A7", "licence only": "#E69F00", "identified": "#56B4E9"}
CMRK = {"unidentified": "D", "licence only": "v", "identified": "o"}
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for c in CLASSES:
    q = unident_year[unident_year.driver_class == c].sort_values("year")
    for reg in ("legacy", "new"):
        r = q[q.regime == reg]
        ax.errorbar(r.year, r.flagged_at_fault, yerr=[r.flagged_at_fault - r.lo, r.hi - r.flagged_at_fault], fmt=CMRK[c] + "-",
                    color=CCOL[c], ms=6, capsize=2.5, lw=1.3)
ax.axvline(2023.5, color="#999999", lw=1, ls="--")
ax.set_ylim(0, 1.02); ax.set_xticks(range(2015, 2027, 2))
ax.set_xlabel("Crash year"); ax.set_ylabel("Share flagged at fault")
fig.savefig(os.path.join(FIG_DIR, "Fig_E11_unidentified.png")); plt.close(fig)
print("Saved in:", OUT_DIR)
