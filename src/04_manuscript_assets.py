# %% [markdown]
# # Step 5 - Manuscript figures and tables (Elsevier style)
# One shared style and ONE label dictionary drive every figure, table and caption, so notation is identical across the main text and the supplement.
#
# **Reads** `harmonised/`, `experiments/`, `experiments_latent/` (from steps 2-4). **Writes** `<data folder>/manuscript/`:
# - `figures/Fig_03..07` (.png at 1000 dpi, AAP line-art requirement) and `figures/main_figures.tex` (figure environments to `\input`)
#   (Figs 1-2 of the manuscript are schematic diagrams that are not produced by this script)
# - `tables/Table_1..4.tex` (booktabs, one file each) and `tables/main_tables.tex`
# - `supplementary/` - every supplementary table and figure as its own standalone `.tex` (+ figure files)
# - `labels_and_symbols.csv`, `README_assets.md`
#
# Specs: widths 90 / 140 / 190 mm (Elsevier single / 1.5 / double column), sans-serif 7 pt, white background, no grid lines,
# vector PDF with embedded fonts, colour-blind-checked palette plus marker/line-style redundancy so figures survive greyscale print.

# %%
# 1. SETUP, STYLE AND THE SINGLE SOURCE OF LABELS
import os, sys, re, warnings, textwrap
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
warnings.filterwarnings("ignore")
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

BASE = os.environ.get("CRASH_BASE", "/content/drive/MyDrive/Research/crash driver data")
PREP_DIR = os.environ.get("CRASH_PREP_DIR", os.path.join(BASE, "harmonised"))
EXP_DIR = os.environ.get("CRASH_EXP_DIR", os.path.join(BASE, "experiments"))
LAT_DIR = os.environ.get("CRASH_LAT_DIR", os.path.join(BASE, "experiments_latent"))
OUT = os.environ.get("CRASH_MS_DIR", os.path.join(BASE, "manuscript"))
D_FIG, D_TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
D_SUP, D_SFIG = os.path.join(OUT, "supplementary"), os.path.join(OUT, "supplementary", "figures")
for d in (D_FIG, D_TAB, D_SUP, D_SFIG): os.makedirs(d, exist_ok=True)
FIG_OFFSET = int(os.environ.get("CRASH_FIG_OFFSET", 2))     # Figs 1-2 are schematic diagrams, so data figures start at Fig_03
FN = lambda k: f"Fig_{k + FIG_OFFSET:02d}"

# ---- fonts: Arial if present, else a metric-compatible sans
avail = {f.name for f in fm.fontManager.ttflist}
FONT = next((f for f in ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"] if f in avail), "DejaVu Sans")
print("Figure font:", FONT)

# ---- palette (checked with the colour-blind validator; marker + line-style redundancy added for greyscale print)
BLUE, ORANGE = "#2a78d6", "#eb6834"          # at-fault ; serious/fatal  (the same two hues in EVERY figure)
INK, MID, LIGHT = "#1a1a1a", "#6b6b6b", "#b8b8b8"
COL = {"fault": BLUE, "ksi": ORANGE}
MK = {"fault": "o", "ksi": "s"}               # circle = at-fault, square = serious/fatal
LS = {"fault": "-", "ksi": "--"}

mm = lambda x: x / 25.4
W = {"single": mm(90), "onehalf": mm(140), "double": mm(190)}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "axes.grid": False,
    "font.family": FONT, "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0, "lines.markersize": 3.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular", "savefig.dpi": 600,
    "legend.frameon": False, "axes.titlesize": 7,
})

# ---- THE label dictionary: change a name here and every figure/table/caption follows
L = {
    "fault": "At-fault", "ksi": "Serious/fatal",
    "legacy": "Legacy format", "new": "New format",
    "auroc": "AUROC", "citl": "Mean predicted \u2212 observed", "brier": "Brier score",
    "sens": "Sensitivity, $\\it{s}$", "fpr": "False-flag rate, $\\it{f}$",
    "concord": "Pair concordance", "gap": "Mean $\\it{p}$ \u2212 0.5",
}
STRAT = {"S1": "Legacy only", "S2": "Pooled", "S3": "Pooled + format indicator", "S4": "Pooled, new rows \u00d75",
         "S5": "New only", "S6": "Legacy + fine-tuning", "S7": "Legacy + recalibration"}
MODEL10 = {"M1": "Legacy only", "M2": "Pooled", "M3": "New only", "M4": "Pooled, noise-corrected", "M5": "New only, noise-corrected"}
FSET = {"full": "All fields", "no_distraction": "\u2212 distraction", "no_distraction_agency": "\u2212 distraction, agency",
        "no_distraction_agency_substance": "\u2212 distraction, agency, substance"}
EVAL = {"legacy_2022_23_control": "Legacy 2022\u201323", "new_2025plus": "New format 2025+"}
AGENCY = {"montgomery_county_police": "Montgomery Co. Police", "gaithersburg_police": "Gaithersburg Police",
          "rockville_police": "Rockville Police", "takoma_park_police": "Takoma Park Police", "park_police": "Park Police"}
VAR = {"collision_h": "Collision type", "movement_h": "Vehicle movement", "route_h": "Route type", "light_h": "Light condition",
       "control_h": "Traffic control", "direction_h": "Travel direction", "agency_h": "Reporting agency", "weather_h": "Weather",
       "surface_h": "Road surface", "damage_h": "Vehicle damage", "body_h": "Vehicle body type", "distraction_h": "Driver distraction",
       "impact_zone_h": "Impact zone", "alcohol_susp": "Alcohol suspected", "drug_susp": "Drugs suspected", "speed_limit": "Speed limit",
       "vehicle_age": "Vehicle age", "report_type": "Report severity", "crash_hour": "Hour of day", "crash_dow": "Day of week",
       "crash_month": "Month", "parked_vehicle": "Parked vehicle", "license_state_group": "Licence state", "make_top": "Vehicle make",
       "y_fault (target)": "At-fault flag (target)", "y_injury_any (target)": "Any injury (target)", "y_ksi (target)": "Serious/fatal (target)"}
COLLISION = {"angle_turning": "Angle/turning", "rear_end": "Rear-end", "sideswipe_same": "Sideswipe, same dir.",
             "sideswipe_opposite": "Sideswipe, opposite dir.", "head_on": "Head-on", "other_unknown": "Other/unknown"}
pretty = lambda k, d: d.get(k, str(k).replace("_", " ").capitalize())

# ---- helpers
def ci(a, lo, hi, d=3): return f"{a:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"
def esc(s):  # LaTeX escape for plain text
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
            .replace("\u2212", "$-$").replace("\u00d7", r"$\times$").replace("\u2013", "--"))
def panel(ax, tag, dx=-0.16, dy=1.04):  # identical panel lettering everywhere: bold (a), (b), ...
    ax.text(dx, dy, f"({tag})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
def save(fig, name, outdir):
    # AAP guide: bitmapped line drawings (charts) need >= 1000 dpi; PNG only, as requested for the submission folder
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=1000, bbox_inches="tight", pad_inches=0.03)
    plt.show(); plt.close(fig)
def newfig(kind, h_mm, ncols=1, **kw):
    fig, ax = plt.subplots(1, ncols, figsize=(W[kind], mm(h_mm)), **kw); return fig, ax
def legend_target(ax, loc="best"):
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=COL[t], marker=MK[t], ls=LS[t], label=L[t]) for t in ("fault", "ksi")], loc=loc, handlelength=1.6)
def dotrow(ax, y, est, lo, hi, color, marker, filled=True, ms=3.6, lw=0.9):
    ax.errorbar(est, y, xerr=[np.asarray(est) - np.asarray(lo), np.asarray(hi) - np.asarray(est)], fmt=marker, color=color, ecolor=color,
                elinewidth=lw, capsize=1.5, capthick=lw, markersize=ms, markerfacecolor=color if filled else "white", markeredgewidth=lw)
def hv(path): return pd.read_csv(path)

# %%
# 2. LOAD DATA
df = pd.read_csv(os.path.join(PREP_DIR, "crash_drivers_harmonised.csv"), low_memory=False, parse_dates=["crash_datetime"])
df["ym"] = df.crash_datetime.dt.to_period("M")
E1, E2 = hv(f"{EXP_DIR}/E1_temporal_transfer_by_halfyear.csv"), hv(f"{EXP_DIR}/E2_feature_set_ablation.csv")
E2s, E3 = hv(f"{EXP_DIR}/E2_reliance_shift.csv"), hv(f"{EXP_DIR}/E3_mitigation_strategies.csv")
E4, E5s, E5t = hv(f"{EXP_DIR}/E4_paired_driver_consistency.csv"), hv(f"{EXP_DIR}/E5_break_scan.csv"), hv(f"{EXP_DIR}/E5_break_tests.csv")
E6, E6l = hv(f"{EXP_DIR}/E6_agency_fault_rates.csv"), pd.read_csv(f"{EXP_DIR}/E6_logit_MCPD_regime_coupling.csv", index_col=0)
E7, E7k = hv(f"{LAT_DIR}/E7_sensitivity_estimates.csv"), hv(f"{LAT_DIR}/E7_kappa_sensitivity.csv")
E8, E8h = hv(f"{LAT_DIR}/E8_simulation_recovery.csv"), hv(f"{LAT_DIR}/E8_heterogeneous_sensitivity_recovery.csv")
E9, E9c = hv(f"{LAT_DIR}/E9_sensitivity_by_group.csv"), pd.read_csv(f"{LAT_DIR}/E9_sensitivity_model_coefficients.csv", index_col=0)
E10 = hv(f"{LAT_DIR}/E10_corrected_vs_naive.csv")
COV, PSI = hv(f"{PREP_DIR}/coverage_before_after.csv"), hv(f"{PREP_DIR}/drift_psi_after_harmonisation.csv")
XWC = pd.read_csv(f"{PREP_DIR}/crosswalk_confidence_share_by_column.csv", index_col=0)
print("Loaded all inputs;", len(df), "driver rows")

# %%
# 3. MAIN FIGURES
CAP, SCAP = {}, {}       # captions (written into the .tex blocks)

# ---- Fig 1: the reporting-format break in three monthly series
_tk = df[(df.n_drivers_in_crash == 2) & df.y_fault.notna()].groupby("report_number").agg(nf=("y_fault", "sum"), n=("y_fault", "size"), ym=("ym", "first"))
_tk = _tk[_tk.n == 2]            # two-driver crashes with both fault flags known
m = (df.groupby("ym").agg(fault=("y_fault", "mean"), ksi=("y_ksi", "mean"), n=("y_fault", "size"))
       .join(_tk.groupby("ym").nf.apply(lambda v: (v == 0).mean()).rename("zero")))
m = m[m.n >= 200]; x = m.index.to_timestamp(); brk = pd.Timestamp("2024-01-01")
fig, axs = newfig("double", 52, 3, constrained_layout=True)
for ax, col, key, ylab, ylim in [(axs[0], "fault", "fault", L["fault"] + " rate", (0, 0.7)),
                                 (axs[1], "fault", "zero", "Two-driver crashes with\nno driver flagged", (0, 0.6)),
                                 (axs[2], "ksi", "ksi", L["ksi"] + " rate", (0, 0.02))]:
    ax.plot(x, m[key], color=COL[col], lw=1.0, ls="-")
    ax.axvline(brk, color=MID, lw=0.7, ls=":"); ax.set_ylim(*ylim); ax.set_ylabel(ylab)
    ax.set_xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2026-12-31"))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in (2015, 2018, 2021, 2024)]); ax.set_xticklabels(["2015", "2018", "2021", "2024"])
for a, t in zip(axs, "abc"): panel(a, t, dx=-0.28)
axs[0].text(brk, 0.67, " Format\n change", fontsize=6, color=MID, va="top", ha="left")
CAP["fig:format_break"] = ("Monthly series for Montgomery County driver records, 2015--2026. (a) Share of drivers flagged at fault. (b) Share of two-driver crashes with both flags known in which "
    "neither driver is flagged at fault. (c) Share of drivers with serious or fatal injury. The dotted line marks the change of reporting format (1 January 2024).")
save(fig, FN(1), D_FIG)

# ---- Fig 2: temporal transfer of a legacy-trained model
def hx(h): return int(h[:4]) + (0.5 if h.endswith("H2") else 0.0)
fig, axs = newfig("onehalf", 56, 2, constrained_layout=True)
for ax, key, ylab in [(axs[0], "auroc", L["auroc"]), (axs[1], "citl", L["citl"])]:
    for t in ("fault", "ksi"):
        d0 = E1[E1.target == t]
        for seg in (d0.half.map(hx) < 2024, d0.half.map(hx) >= 2024):      # no line drawn across the format change
            d = d0[seg]; xx = d.half.map(hx) + (0.04 if t == "ksi" else -0.04)
            ax.errorbar(xx, d[key], yerr=[d[key] - d[key + "_lo"], d[key + "_hi"] - d[key]], color=COL[t], marker=MK[t], ls=LS[t], lw=0.9,
                        elinewidth=0.7, capsize=1.2, markersize=3.4)
    ax.axvline(2023.75, color=MID, lw=0.7, ls=":"); ax.set_ylabel(ylab)
    ax.set_xticks(range(2022, 2027)); ax.set_xlim(2021.8, 2026.8)
axs[1].axhline(0, color=LIGHT, lw=0.6); legend_target(axs[0], "lower left"); panel(axs[0], "a", -0.2); panel(axs[1], "b", -0.2)
CAP["fig:transfer"] = ("Transfer of a model trained on legacy-format records (2015--2021) to later half-years. (a) Discrimination. (b) Calibration-in-the-large. "
    "Bars show 95\\% crash-clustered bootstrap intervals; the dotted line marks the format change.")
save(fig, FN(2), D_FIG)

# ---- Fig 3: estimated sensitivity s of the at-fault flag
yr = E7[E7.by == "regime_year"].copy(); yr["year"] = yr.level.str[-4:].astype(int); yr["new"] = yr.level.str.startswith("new")
ag = E7[(E7.by == "regime_agency") & E7.level.str.startswith("new-")].copy(); ag["name"] = ag.level.str[4:].map(lambda k: pretty(k, AGENCY))
inj = E7[(E7.by == "regime_injury") & E7.level.str.startswith("new-")].copy()
inj["name"] = inj.level.str[4:].map({"injury": "Injury crashes", "property_damage": "Property-damage crashes"})
col9 = E9[E9.variable == "collision_h"].copy(); col9["name"] = col9.level.map(lambda k: pretty(k, COLLISION))
fig, axs = newfig("double", 60, 3, constrained_layout=True, gridspec_kw={"width_ratios": [1.1, 1, 1]})
a = axs[0]
for new_, filled in [(False, False), (True, True)]:
    d = yr[yr.new == new_]; dotrow_x = d.year.values
    a.errorbar(dotrow_x, d.s, yerr=[d.s - d.s_lo, d.s_hi - d.s], fmt="o", color=BLUE, markerfacecolor=BLUE if filled else "white",
               elinewidth=0.8, capsize=1.2, markersize=3.4, markeredgewidth=0.8)
a.axvline(2023.5, color=MID, lw=0.7, ls=":"); a.set_ylim(0.4, 1.02); a.set_ylabel(L["sens"]); a.set_xticks([2015, 2018, 2021, 2024])
a.text(2019, 0.45, L["legacy"], fontsize=6, color=MID, ha="center"); a.text(2025, 0.45, L["new"], fontsize=6, color=MID, ha="center")
b = axs[1]; g = pd.concat([ag[["name", "s", "s_lo", "s_hi"]]]).sort_values("s"); yy = np.arange(len(g))
dotrow(b, yy, g.s, g.s_lo, g.s_hi, BLUE, "o"); b.set_yticks(yy); b.set_yticklabels(g.name); b.set_xlim(0, 1); b.set_ylim(-0.6, len(g) - 0.4); b.set_xlabel(L["sens"] + ", new format")
c = axs[2]; g2 = pd.concat([inj[["name", "s", "s_lo", "s_hi"]], col9.rename(columns={"lo": "s_lo", "hi": "s_hi"})[["name", "s", "s_lo", "s_hi"]]]).sort_values("s")
yy = np.arange(len(g2)); dotrow(c, yy, g2.s, g2.s_lo, g2.s_hi, BLUE, "o"); c.set_yticks(yy); c.set_yticklabels(g2.name); c.set_xlim(0, 1)
c.set_xlabel(L["sens"] + ", new format")
for ax, t, dx in zip(axs, "abc", (-0.2, -0.62, -0.62)): panel(ax, t, dx)
CAP["fig:sensitivity"] = ("Estimated sensitivity $s$ of the at-fault flag, i.e.\\ the probability that a truly at-fault driver is flagged, assuming one at-fault driver per two-driver crash. "
    "(a) By year (open: legacy format; filled: new format). (b) By reporting agency, new format. (c) By crash severity and collision type, new format. "
    "Bars show 95\\% bootstrap intervals; (a) and (b) are closed-form estimates, (c) uses the covariate model in Supplementary Table S13.")
save(fig, FN(3), D_FIG)

# ---- Fig 4: mitigation strategies
E3["k"] = E3.strategy.str[:2]; E3["name"] = E3.k.map(STRAT)
order = list(STRAT.values())[::-1]
fig, axs = newfig("onehalf", 62, 2, constrained_layout=True, sharey=True)
for ax, key, xl in [(axs[0], "auroc", L["auroc"]), (axs[1], "citl", L["citl"])]:
    for t, off in (("fault", 0.14), ("ksi", -0.14)):
        d = E3[E3.target == t].set_index("name").loc[order]
        dotrow(ax, np.arange(len(order)) + off, d[key].values, d[key + "_lo"].values, d[key + "_hi"].values, COL[t], MK[t], ms=3.2)
    ax.set_xlabel(xl)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(order); axs[1].axvline(0, color=LIGHT, lw=0.6)
legend_target(axs[1], "lower right"); panel(axs[0], "a", -0.62); panel(axs[1], "b", -0.06)
from matplotlib.ticker import MultipleLocator
axs[0].xaxis.set_major_locator(MultipleLocator(0.05))
CAP["fig:strategies"] = ("Strategies for coping with the format change, evaluated on new-format records from 2025 onwards. (a) Discrimination. (b) Calibration-in-the-large. "
    "Bars show 95\\% crash-clustered bootstrap intervals.")
save(fig, FN(4), D_FIG)

# ---- Fig 5: noise-corrected vs naive models
E10["k"] = E10.model.str[:2]; E10["name"] = E10.k.map(MODEL10); d = E10.iloc[::-1].reset_index(drop=True)
fig, axs = newfig("onehalf", 50, 2, constrained_layout=True, sharey=True)
dotrow(axs[0], d.index.values, d.pair_concordance, d.conc_lo, d.conc_hi, BLUE, "o"); axs[0].set_xlabel(L["concord"])
axs[1].plot(d.gap_from_half, d.index.values, "o", color=BLUE); axs[1].axvline(0, color=LIGHT, lw=0.6); axs[1].set_xlabel(L["gap"])
axs[0].set_yticks(d.index.values); axs[0].set_yticklabels(d.name); panel(axs[0], "a", -0.62); panel(axs[1], "b", -0.06)
CAP["fig:corrected"] = ("Noise-corrected versus naive at-fault models on two-driver crashes, new format, 2025 onwards. (a) Share of crashes with exactly one flagged driver in which the "
    "model ranks the flagged driver higher (95\\% bootstrap interval). (b) Deviation of the mean predicted at-fault probability from 0.5, the expected value when each crash has one at-fault driver.")
save(fig, FN(5), D_FIG)

# %%
# 4. SUPPLEMENTARY FIGURES
# S1 break scan
fig, ax = newfig("single", 52)
for t in ("fault", "ksi"):
    d = E5s[E5s.series == t]; ax.plot(pd.to_datetime(d.break_month), d.t_hac.abs(), color=COL[t], ls=LS[t], lw=1.0)
ax.axvline(brk, color=MID, lw=0.7, ls=":"); ax.set_ylabel("|t| of level shift"); legend_target(ax, "upper left")
SCAP["sfig:breakscan"] = "Level-shift statistic (absolute HAC $t$) for every candidate break month in the monthly at-fault and serious/fatal series. The dotted line marks 1 January 2024."
save(fig, "S_Fig_01", D_SFIG)

# S2 label coverage
cv = COV.copy(); cv["name"] = cv.harmonised.map(lambda k: pretty(k, VAR | {"alcohol_susp": "Alcohol suspected", "drug_susp": "Drugs suspected"})); cv = cv.sort_values("raw_overlap_pct")
fig, ax = newfig("onehalf", 72); yy = np.arange(len(cv))
ax.hlines(yy, cv.raw_overlap_pct, cv.harmonised_overlap_pct, color=LIGHT, lw=1.0)
ax.plot(cv.raw_overlap_pct, yy, "o", mfc="white", color=INK, mew=0.8, label="Raw labels"); ax.plot(cv.harmonised_overlap_pct, yy, "o", color=INK, label="Harmonised labels")
ax.set_yticks(yy); ax.set_yticklabels(cv.name); ax.set_xlabel("Rows whose label exists in both formats (%)"); ax.set_xlim(-3, 103); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
SCAP["sfig:coverage"] = "Share of records carrying a label that occurs in both reporting formats, before and after harmonisation, by field."
save(fig, "S_Fig_02", D_SFIG)

# S3 PSI
p = PSI.copy(); p["name"] = p.variable.map(lambda k: pretty(k, VAR)); p = p.sort_values("psi"); yy = np.arange(len(p))
fig, ax = newfig("onehalf", 98)
ax.plot(p.psi, yy, "o", color=INK)
for v in (0.1, 0.25): ax.axvline(v, color=LIGHT, lw=0.6, ls="--")
ax.set_yticks(yy); ax.set_yticklabels(p.name); ax.set_xlabel("Population stability index, legacy vs new format")
SCAP["sfig:psi"] = "Population stability index after harmonisation. Dashed lines mark the conventional thresholds of 0.10 (moderate) and 0.25 (major)."
save(fig, "S_Fig_03", D_SFIG)

# S4 ablation
fig, axs = newfig("onehalf", 52, 2, constrained_layout=True, sharey=True); fo = list(FSET)[::-1]
for ax, t in zip(axs, ("fault", "ksi")):
    for ev, filled in (("legacy_2022_23_control", False), ("new_2025plus", True)):
        d = E2[(E2.target == t) & (E2.eval_set == ev)].set_index("feature_set").loc[fo]
        dotrow(ax, np.arange(len(fo)) + (0.13 if filled else -0.13), d.auroc.values, d.auroc_lo.values, d.auroc_hi.values, COL[t], MK[t], filled=filled)
    ax.set_xlabel(f"{L['auroc']} ({L[t]})")
axs[0].set_yticks(range(len(fo))); axs[0].set_yticklabels([FSET[k] for k in fo]); panel(axs[0], "a", -0.8); panel(axs[1], "b", -0.06)
from matplotlib.lines import Line2D
axs[1].legend(handles=[Line2D([], [], color=INK, marker="o", mfc="white", ls="", label=EVAL["legacy_2022_23_control"]),
                       Line2D([], [], color=INK, marker="o", ls="", label=EVAL["new_2025plus"])], loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
SCAP["sfig:ablation"] = "Discrimination of a legacy-trained model as fields are removed, evaluated on same-format data (open) and on new-format data (filled). (a) At-fault flag. (b) Serious/fatal injury."
save(fig, "S_Fig_04", D_SFIG)

# S5 reliance shift (at-fault model)
r = E2s[E2s.target == "fault"].pivot(index="feature", columns="eval_set", values="mean_abs_contrib").sort_values("legacy").tail(8)
fig, ax = newfig("single", 58); yy = np.arange(len(r))
ax.hlines(yy, r["new"], r["legacy"], color=LIGHT, lw=1.0); ax.plot(r["legacy"], yy, "o", mfc="white", color=BLUE, mew=0.8, label=L["legacy"]); ax.plot(r["new"], yy, "o", color=BLUE, label=L["new"])
ax.set_yticks(yy); ax.set_yticklabels([pretty(k, VAR) for k in r.index]); ax.set_xlabel("Mean |contribution| to log-odds"); ax.legend(loc="lower right")
SCAP["sfig:reliance"] = "Mean absolute feature contribution of the legacy-trained at-fault model on legacy (open) and new-format (filled) records; the eight largest fields are shown."
save(fig, "S_Fig_05", D_SFIG)

# S6 paired-driver consistency
pp = E4[E4.n_crashes >= 500]
fig, ax = newfig("single", 54)
for col, nm, c, mk, ls in (("exactly_one", "One flagged", BLUE, "o", "-"), ("zero_at_fault", "None flagged", INK, "s", "--"), ("both", "Both flagged", MID, "^", ":")):
    for seg in (pp.year <= 2023, pp.year >= 2024):
        ax.plot(pp.year[seg], pp[col][seg], color=c, marker=mk, ls=ls, lw=0.9, ms=3.2, label=nm if seg.iloc[0] else None)
ax.axvline(2023.5, color=MID, lw=0.7, ls=":"); ax.set_ylabel("Share of two-driver crashes"); ax.set_xticks([2015, 2018, 2021, 2024]); ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.55))
SCAP["sfig:paired"] = "Number of drivers flagged at fault in two-driver crashes, by year. Years with fewer than 500 crashes in a format are omitted."
save(fig, "S_Fig_06", D_SFIG)

# S7 agency fault rates
e6 = E6.copy(); e6["name"] = e6.agency_h.map(lambda k: pretty(k, AGENCY)); names = [AGENCY[k] for k in AGENCY if AGENCY[k] in set(e6.name)][::-1]
fig, ax = newfig("single", 56)
for reg, off, filled in (("legacy", 0.14, False), ("new", -0.14, True)):
    d = e6[e6.regime == reg].set_index("name").reindex(names).dropna(subset=["rate"])
    dotrow(ax, [names.index(n) + off for n in d.index], d.rate.values, d.lo.values, d.hi.values, BLUE, "o", filled=filled)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.set_xlabel(L["fault"] + " rate"); ax.set_xlim(0, 0.75)
ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", mfc="white", ls="", label=L["legacy"]), Line2D([], [], color=BLUE, marker="o", ls="", label=L["new"])],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
SCAP["sfig:agency"] = "At-fault rate by reporting agency and format (95\\% Wilson intervals). Agencies with fewer than 300 flagged-or-unflagged drivers in a format are omitted."
save(fig, "S_Fig_07", D_SFIG)

# S8 simulation
fig, axs = newfig("onehalf", 50, 2, constrained_layout=True)
sm_ = E8[E8.model == "naive"].groupby("s_true").s_hat.agg(["mean", "std"]).reset_index()
axs[0].plot([0.4, 1], [0.4, 1], color=LIGHT, lw=0.7); axs[0].errorbar(sm_.s_true, sm_["mean"], yerr=sm_["std"], fmt="o", color=BLUE, capsize=1.5, elinewidth=0.8)
axs[0].set_xlabel("True " + L["sens"].lower().replace("sensitivity", "sensitivity")); axs[0].set_ylabel("Estimated " + L["sens"].lower())
for nm, mk, filled in (("naive", "o", False), ("corrected", "o", True)):
    d = E8[E8.model == nm].groupby("s_true").brier_vs_truth.mean()
    axs[1].plot(d.index, d.values, marker=mk, mfc=BLUE if filled else "white", color=BLUE, ls="-" if filled else "--", lw=0.9, label=nm.capitalize())
axs[1].set_xlabel("True " + L["sens"].lower()); axs[1].set_ylabel(L["brier"] + " vs true fault"); axs[1].legend(loc="upper right")
panel(axs[0], "a", -0.2); panel(axs[1], "b", -0.2)
SCAP["sfig:simulation"] = ("Simulation check. Known noise was injected into legacy crashes with one flagged driver. (a) Recovery of the sensitivity from pair counts. "
    "(b) Brier score against the true fault label for naive and noise-corrected models.")
save(fig, "S_Fig_08", D_SFIG)

# %%
# 5. TABLE BUILDERS (booktabs, no vertical rules, notation from the same dictionary)
def sg(v, d=3): return f"{v:+.{d}f}".replace("-", "$-$")        # typographic minus
def tabular(rows, header, align, wide=False, hdr_raw=None):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    hdr = hdr_raw if hdr_raw else f"{' & '.join(header)} \\\\"
    t = f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{hdr}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}"
    return f"\\resizebox{{\\textwidth}}{{!}}{{%\n{t}}}" if wide else t
def fit(tab):   # shrink a table only if it is wider than the text block (needs \\usepackage{adjustbox})
    return f"\\begin{{adjustbox}}{{max width=\\textwidth}}\n{tab}\n\\end{{adjustbox}}"
def table_env(tab, caption, label, note=None, size="\\small\\setlength{\\tabcolsep}{4pt}"):
    n = f"\n\\par\\vspace{{2pt}}{{\\footnotesize {note}}}" if note else ""
    return f"\\begin{{table}}[!t]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n{fit(tab)}{n}\n\\end{{table}}\n"
def standalone(body, kind, number, extra=""):
    ctr = "table" if kind == "table" else "figure"
    return ("\\documentclass[11pt]{article}\n\\usepackage[a4paper,margin=2.5cm]{geometry}\n\\usepackage{booktabs,graphicx,caption,amsmath,float,adjustbox}\n"
            f"\\captionsetup{{labelfont=bf}}\n\\renewcommand{{\\the{ctr}}}{{S\\arabic{{{ctr}}}}}\n\\setcounter{{{ctr}}}{{{number - 1}}}\n{extra}\n"
            f"\\begin{{document}}\n{body}\n\\end{{document}}\n")
n3 = lambda v: f"{v:.3f}"; pc = lambda v: f"{100 * v:.1f}"

MAIN_T, SUPP_T = [], []      # (label, caption, tabular, note, wide)
# ---- Table 1: data description by format
def desc(d):
    cr = d.drop_duplicates("report_number"); known = d[d.y_fault.notna()]
    tw = d[(d.n_drivers_in_crash == 2) & d.y_fault.notna()].groupby("report_number").y_fault.agg(["size", "sum"]); tw = tw[tw["size"] == 2]
    return [f"{len(d):,}", f"{len(cr):,}", f"{d.crash_datetime.min():%b %Y}\u2013{d.crash_datetime.max():%b %Y}", pc(known.y_fault.mean()),
            pc(d.y_fault.isna().mean()), f"{len(tw):,}", pc((tw['sum'] == 0).mean()), pc((tw['sum'] == 1).mean()), f"{int(d.y_ksi.sum()):,}", pc(d.y_ksi.mean())]
labs = ["Driver records", "Crashes", "Period", f"{L['fault']} drivers (\\%)", "Fault unknown (\\%)", "Two-driver crashes with both flags known",
        "\\quad no driver flagged (\\%)", "\\quad exactly one driver flagged (\\%)", f"{L['ksi']} drivers", f"{L['ksi']} drivers (\\%)"]
dl, dn = desc(df[df.regime == "legacy"]), desc(df[df.regime == "new"])
rows = [[a, esc(b), esc(c)] if i == 2 else [a, b, c] for i, (a, b, c) in enumerate(zip(labs, dl, dn))]
MAIN_T.append(("tab:data", "Driver records by reporting format.", tabular(rows, ["", L["legacy"], L["new"]], "lrr"),
               "Format assigned from the coding style of each record; records between 28 December 2023 and 3 January 2024 can belong to either format.", False))
# ---- Table 2: strategies
r2 = []
for k in STRAT:
    row = [esc(STRAT[k])]
    for t in ("fault", "ksi"):
        d = E3[(E3.target == t) & (E3.k == k)].iloc[0]; row += [ci(d.auroc, d.auroc_lo, d.auroc_hi), n3(d.brier), sg(d.citl)]
    r2.append(row)
MAIN_T.append(("tab:strategies", "Strategies for the format change, evaluated on new-format records from 2025 onwards.",
               tabular(r2, [], "lcccccc", hdr_raw=("& \\multicolumn{3}{c}{" + L["fault"] + "} & \\multicolumn{3}{c}{" + L["ksi"] + "} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
                       "Strategy & AUROC [95\\% CI] & Brier & Bias & AUROC [95\\% CI] & Brier & Bias \\\\")),
               "Bias is the mean predicted minus the observed rate. Intervals are crash-clustered bootstrap intervals.", False))
# ---- Table 3: s and f
e7 = E7[E7.by.isin(["regime", "regime_year"])].copy()
r3 = [[esc(r.level.replace("legacy", "Legacy").replace("new", "New").replace("-", " ")), f"{int(r.n_crashes):,}", pc(r.p_zero), pc(r.p_one), pc(r.p_both),
       ci(r.s, r.s_lo, r.s_hi), n3(r.f)] for r in e7.itertuples()]
MAIN_T.append(("tab:sens", "Pair counts and estimated sensitivity $s$ and false-flag rate $f$ of the at-fault flag, by format and year.",
               tabular(r3, ["Format / year", "Crashes", "None (\\%)", "One (\\%)", "Both (\\%)", "$s$ [95\\% CI]", "$f$"], "lrrrrcc"),
               "Columns \\emph{None}, \\emph{One}, \\emph{Both} give the share of two-driver crashes with that number of drivers flagged at fault.", False))
# ---- Table 4: corrected vs naive
r4 = [[esc(MODEL10[r.k]), ci(r.pair_concordance, r.conc_lo, r.conc_hi), n3(r.auroc_vs_flags), f"{sg(r.gap_from_half)}"] for r in E10.itertuples()]
MAIN_T.append(("tab:corrected", "Noise-corrected versus naive at-fault models, two-driver crashes, new format, 2025 onwards.",
               tabular(r4, ["Model", "Pair concordance [95\\% CI]", "AUROC vs.\\ flags", "Mean $p$ $-$ 0.5"], "lccc"),
               "Pair concordance is evaluated on crashes with exactly one flagged driver.", False))

# ---- supplementary tables
x = XWC.reset_index().rename(columns={"index": "col"}); x.columns = ["col", "high", "low", "medium"]
SUPP_T.append(("stab:xwalk", "Share of records by confidence of the harmonisation mapping.",
               tabular([[esc(pretty(r.col, VAR)), f"{r.high:.1f}", f"{r.medium:.1f}", f"{r.low:.1f}"] for r in x.itertuples()], ["Field", "High (\\%)", "Medium (\\%)", "Low (\\%)"], "lrrr"), None, False))
SUPP_T.append(("stab:coverage", "Share of records whose label occurs in both formats, before and after harmonisation.",
               tabular([[esc(pretty(r.harmonised, VAR)), f"{r.raw_overlap_pct:.1f}", f"{r.harmonised_overlap_pct:.1f}"] for r in COV.sort_values("raw_overlap_pct").itertuples()],
                       ["Field", "Raw (\\%)", "Harmonised (\\%)"], "lrr"), None, False))
SUPP_T.append(("stab:psi", "Population stability index between formats after harmonisation.",
               tabular([[esc(pretty(r.variable, VAR)), n3(r.psi), esc(r.reading)] for r in PSI.itertuples()], ["Variable", "PSI", "Reading"], "lrl"), None, False))
SUPP_T.append(("stab:halfyear", "Legacy-trained model by half-year (95\\% crash-clustered bootstrap intervals).",
               tabular([[r.half, L[r.target], ci(r.auroc, r.auroc_lo, r.auroc_hi), ci(r.citl, r.citl_lo, r.citl_hi)] for r in E1.sort_values(["target", "half"]).itertuples()],
                       ["Half-year", "Target", "AUROC", "Mean predicted $-$ observed"], "llcc"), None, False))
SUPP_T.append(("stab:ablation", "Discrimination as fields are removed from the legacy-trained model.",
               tabular([[L[r.target], esc(FSET[r.feature_set]), esc(EVAL[r.eval_set]), ci(r.auroc, r.auroc_lo, r.auroc_hi)] for r in E2.sort_values(["target", "eval_set", "feature_set"]).itertuples()],
                       ["Target", "Fields", "Evaluation", "AUROC"], "lllc"), None, False))
SUPP_T.append(("stab:paired", "Number of drivers flagged at fault in two-driver crashes.",
               tabular([[L[r.regime], str(r.year), f"{int(r.n_crashes):,}", pc(r.zero_at_fault), pc(r.exactly_one), pc(r.both)] for r in E4[E4.n_crashes >= 500].itertuples()],
                       ["Format", "Year", "Crashes", "None (\\%)", "One (\\%)", "Both (\\%)"], "llrrrr"), None, False))
SUPP_T.append(("stab:break", "Interrupted time-series level shift at 1 January 2024 (HAC standard errors).",
               tabular([[esc(r.series), sg(r.level_shift), f"[{sg(r.ci_lo)}, {sg(r.ci_hi)}]", f"{r.t_hac:.1f}", n3(r.pre_mean), n3(r.post_mean)] for r in E5t.itertuples()],
                       ["Series", "Level shift", "95\\% CI", "$t$", "Mean before", "Mean after"], "lrcrrr"), None, False))
SUPP_T.append(("stab:agency", "At-fault rate by agency and format.",
               tabular([[pretty(r.agency_h, AGENCY), L[r.regime], f"{int(r.size):,}", ci(r.rate, r.lo, r.hi)] for r in E6.itertuples()], ["Agency", "Format", "Drivers", "Rate [95\\% CI]"], "llrc"), None, False))
SUPP_T.append(("stab:logit", "Logistic model of the at-fault flag, Montgomery County Police records (crash-clustered standard errors).",
               tabular([[esc(k.replace("_", " ")), sg(r.coef), f"{r.se:.3f}", f"{r.OR:.2f}", "<0.001" if r.p < 0.001 else f"{r.p:.3f}"] for k, r in E6l.iterrows()],
                       ["Term", "Coefficient", "SE", "Odds ratio", "$p$"], "lrrrr"), None, False))
e7b = E7[E7.by.isin(["regime_agency", "regime_injury"])]
SUPP_T.append(("stab:sens_groups", "Closed-form sensitivity $s$ and false-flag rate $f$ by agency and crash severity.",
               tabular([[esc(r.level.replace("legacy", "Legacy").replace("new", "New").replace("-", " ").replace("_police", "").replace("_", " ")), f"{int(r.n_crashes):,}", ci(r.s, r.s_lo, r.s_hi), n3(r.f)] for r in e7b.itertuples()],
                       ["Format / group", "Crashes", "$s$ [95\\% CI]", "$f$"], "lrcc"), None, False))
kp = E7k.pivot(index="kappa_true_no_culprit", columns="regime", values="s").reset_index()
SUPP_T.append(("stab:kappa", "Sensitivity $s$ if a share $\\kappa$ of crashes truly has no at-fault driver ($f$ fixed at the legacy value).",
               tabular([[f"{r.kappa_true_no_culprit:.3f}", n3(r.legacy), n3(r.new)] for r in kp.itertuples()], ["$\\kappa$", L["legacy"], L["new"]], "lrr"), None, False))
sm = E8.groupby(["s_true", "model"])[["s_hat", "auroc_vs_truth", "brier_vs_truth", "mean_p"]].mean().reset_index()
SUPP_T.append(("stab:sim", "Simulation: known noise injected into legacy crashes (mean of three seeds).",
               tabular([[f"{r.s_true:.1f}", r.model.capitalize(), n3(r.s_hat), n3(r.auroc_vs_truth), n3(r.brier_vs_truth), n3(r.mean_p)] for r in sm.itertuples()],
                       ["True $s$", "Model", "Estimated $s$", "AUROC vs.\\ truth", "Brier vs.\\ truth", "Mean $p$"], "llrrrr"), "True prevalence of fault is 0.5.", False))
SUPP_T.append(("stab:cov", "Covariate model for sensitivity, new format (odds ratios with 95\\% bootstrap intervals; reference: Montgomery County Police, 2024, property-damage crash, angle/turning).",
               tabular([[esc(k.replace("_", " ")), f"{r.odds_ratio:.2f}", f"[{r.or_lo:.2f}, {r.or_hi:.2f}]"] for k, r in E9c.iterrows()], ["Term", "Odds ratio", "95\\% CI"], "lrc"), None, False))

# %%
# 6. WRITE LaTeX FILES
_figtex = {i: f"\\begin{{figure}}[!t]\n\\centering\n\\includegraphics[width=\\textwidth]{{figures/Fig_{i:02d}}}\n\\caption{{{CAP[lab]}}}\n\\label{{{lab}}}\n\\end{{figure}}\n"
           for i, lab in enumerate(CAP, 1 + FIG_OFFSET)}
for i, t in _figtex.items(): open(os.path.join(D_FIG, f"Fig_{i:02d}.tex"), "w").write(t)      # one file per figure: \\input{figures/Fig_01}
open(os.path.join(D_FIG, "main_figures.tex"), "w").write("\n".join(_figtex.values()))
for i, (lab, cap, tab, note, wide) in enumerate(MAIN_T, 1):
    open(os.path.join(D_TAB, f"Table_{i}.tex"), "w").write(table_env(tab, cap, lab, note))
open(os.path.join(D_TAB, "main_tables.tex"), "w").write("\n".join(f"\\input{{tables/Table_{i}}}" for i in range(1, len(MAIN_T) + 1)) + "\n")
for i, (lab, cap, tab, note, wide) in enumerate(SUPP_T, 1):
    body = f"\\begin{{table}}[H]\n\\centering\n\\small\n\\caption{{{cap}}}\n\\label{{{lab}}}\n{fit(tab)}\n" + (f"\\par\\vspace{{2pt}}{{\\footnotesize {note}}}\n" if note else "") + "\\end{table}"
    open(os.path.join(D_SUP, f"S_Table_{i:02d}.tex"), "w").write(standalone(body, "table", i))
for i, lab in enumerate(SCAP, 1):
    body = f"\\begin{{figure}}[H]\n\\centering\n\\includegraphics[width=\\textwidth]{{figures/S_Fig_{i:02d}}}\n\\caption{{{SCAP[lab]}}}\n\\label{{{lab}}}\n\\end{{figure}}"
    open(os.path.join(D_SUP, f"S_Fig_{i:02d}.tex"), "w").write(standalone(body, "figure", i))

# labels and README
rows = [("target", k, v) for k, v in L.items()] + [("strategy", k, v) for k, v in STRAT.items()] + [("model", k, v) for k, v in MODEL10.items()] + \
       [("field", k, v) for k, v in VAR.items()] + [("agency", k, v) for k, v in AGENCY.items()] + [("collision", k, v) for k, v in COLLISION.items()]
pd.DataFrame(rows, columns=["group", "key", "label"]).to_csv(os.path.join(OUT, "labels_and_symbols.csv"), index=False)
open(os.path.join(OUT, "README_assets.md"), "w").write(textwrap.dedent("""\
    # Manuscript assets
    Preamble: `\\usepackage{graphicx,booktabs,amsmath,adjustbox}` (wide tables are fitted to the text width with adjustbox).
    Main text: `\\input{figures/main_figures}` and `\\input{tables/main_tables}` (or `\\input{tables/Table_1}` etc.).
    Supplement: each `supplementary/S_Table_NN.tex` and `S_Fig_NN.tex` is a standalone, compilable file numbered S1, S2, ...
    Figures: Fig_NN.png at 1000 dpi (AAP minimum for line art). Widths: 90 mm (single), 140 mm (1.5), 190 mm (double).
    To rename anything (e.g. "Serious/fatal"), edit the dictionary `L` / `STRAT` / `VAR` in the script and rerun: all outputs change together.
    """))
print("Main figures:", len(CAP), "| supplementary figures:", len(SCAP), "| main tables:", len(MAIN_T), "| supplementary tables:", len(SUPP_T))
print("Saved in:", OUT)
