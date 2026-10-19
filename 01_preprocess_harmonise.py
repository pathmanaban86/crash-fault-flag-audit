# %% [markdown]
# # Step 2: Harmonisation and preprocessing - Montgomery County crash drivers data
# Builds one analysis-ready file from `export.csv` using what the EDA found:
# * regime (reporting format) assigned per row from the coding style, not just the date;
# * an **editable crosswalk** that maps legacy and new labels to common categories;
# * invalid values flagged/set to missing (no rows are dropped);
# * targets (at-fault, injury) and crash-level fields;
# * validation: label coverage before/after, drift (PSI), and how well a model can still tell the regimes apart.
# Everything is written to `harmonised/` next to the CSV.

# %% 0. Setup and configuration
import os, re, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_PATH = os.environ.get("CRASH_DATA_PATH", "/content/drive/MyDrive/Research/crash driver data/export.csv")
OUT_DIR = os.environ.get("CRASH_PREP_DIR", os.path.join(os.path.dirname(DATA_PATH), "harmonised"))
USE_EDITED_CROSSWALK = True        # if harmonised/crosswalk_edited.csv exists, its rows override the proposal
RUN_DOMAIN_CLASSIFIER = True       # regime-detectability check (about 1-2 minutes)
LAT_RANGE, LON_RANGE = (38.93, 39.36), (-77.55, -76.88)   # approximate county box
SPEED_VALID = (5, 70)              # mph; 0 is treated as missing
VEHICLE_YEAR_MIN = 1950            # earlier years (and years after crash year + 1) are set to missing

try:
    from google.colab import drive  # type: ignore
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except ImportError:
    pass

FIG_DIR = os.path.join(OUT_DIR, "figures")
for d in (OUT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

BLUE, GREY, ORANGE = "#0B5CAD", "#9AA5B1", "#D9822B"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.frameon": False,
})
LEGACY, NEW = "legacy", "new"
REPORT = []   # lines for preprocessing_report.md


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def log(msg):
    print(msg)
    REPORT.append(msg)


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=300)
    plt.show()
    plt.close(fig)


_NORM_CACHE = {}


def norm_col(df_, col):
    """cached normalised version of a raw column"""
    if col not in _NORM_CACHE:
        _NORM_CACHE[col] = norm(df_[col])
    return _NORM_CACHE[col]


def norm(s):
    """lower-case, collapse spaces, decode &lt;; missing stays NaN (works on pandas 2 and 3)."""
    out = s.astype("string").str.lower().str.replace(r"\s+", " ", regex=True).str.strip().str.replace("&lt;", "<", regex=False)
    return out.astype(object).where(out.notna(), np.nan)


# %% 1. Load, parse dates, basic keys
section("1. LOAD")
raw = pd.read_csv(DATA_PATH, low_memory=False)
raw.columns = [c.strip() for c in raw.columns]
N = len(raw)
log(f"Loaded {N:,} rows x {raw.shape[1]} columns from {DATA_PATH}")

dt = pd.to_datetime(raw["Crash Date/Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
if dt.isna().mean() > 0.01:
    dt = pd.to_datetime(raw["Crash Date/Time"], errors="coerce")
log(f"Unparsed dates: {int(dt.isna().sum())}; range {dt.min()} to {dt.max()}")

# %% 2. Regime assignment from coding style (case of coded values), with date as fallback
section("2. REGIME ASSIGNMENT")
STYLE_COLS = ["Weather", "Surface Condition", "Light", "Collision Type", "Vehicle Movement", "Traffic Control",
              "Injury Severity", "Vehicle Damage Extent", "Vehicle Body Type", "Vehicle First Impact Location",
              "Driver Distracted By", "Driver Substance Abuse"]
STYLE_COLS = [c for c in STYLE_COLS if c in raw.columns]
mixed = pd.DataFrame({c: raw[c].astype("string").str.contains("[a-z]", regex=True).astype(float).where(raw[c].notna()) for c in STYLE_COLS})
style_score = mixed.mean(axis=1)                                     # share of mixed-case values in the row (0 = legacy UPPER-CASE)
regime_by_date = np.where(dt >= pd.Timestamp("2024-01-01"), NEW, LEGACY)
regime = np.where(style_score.isna(), regime_by_date, np.where(style_score > 0.5, NEW, LEGACY))
regime = pd.Series(regime, index=raw.index)
agree = (regime == pd.Series(regime_by_date, index=raw.index))
first_new = dt[regime == NEW].min()
last_legacy = dt[regime == LEGACY].max()
in_overlap = (dt >= first_new) & (dt <= last_legacy)
log(f"Style-based regime: legacy {int((regime == LEGACY).sum()):,}, new {int((regime == NEW).sum()):,}")
log(f"First new-style row: {first_new}; last legacy-style row: {last_legacy}; rows in overlap window: {int(in_overlap.sum())}")
log(f"Rows where style and the 1 Jan 2024 date rule disagree: {int((~agree).sum())} (all should sit inside the overlap window: "
    f"{int((~agree & ~in_overlap).sum())} outside)")

# %% 3. Crosswalk: proposed mapping of normalised raw values -> common categories
section("3. CROSSWALK")
XW = []   # (column, raw_pattern, harmonised, confidence, note); a trailing * in raw_pattern means "starts with"


def X(col, harm, values, conf="high", note=""):
    for v in values:
        XW.append((col, v, harm, conf, note))


# --- Collision Type (legacy detailed turning types vs new coarse types)
C = "Collision Type"
X(C, "rear_end", ["same dir rear end", "front to rear"])
X(C, "rear_end", ["same dir rend left turn", "same dir rend right turn"], "medium", "legacy rear-end with turning vehicle")
X(C, "angle_turning", ["straight movement angle", "angle"])
X(C, "angle_turning", ["angle meets left turn", "angle meets right turn", "angle meets left head on", "rear to side"], "medium", "angle / turning conflicts")
X(C, "angle_turning", ["head on left turn", "same direction left turn", "same direction right turn", "same dir both left turn", "opposite dir both left turn"], "low",
  "legacy turning types have no exact equivalent in the new coarse list")
X(C, "head_on", ["head on", "front to front"])
X(C, "sideswipe_same", ["same direction sideswipe", "sideswipe, same direction"])
X(C, "sideswipe_opposite", ["opposite direction sideswipe", "sideswipe, opposite direction"])
X(C, "single_vehicle", ["single vehicle"])
X(C, "other_unknown", ["other", "unknown", "rear to rear"], "medium")
# --- Vehicle Movement
M = "Vehicle Movement"
X(M, "constant_speed", ["moving constant speed"])
X(M, "slowing_stopping", ["slowing or stopping"])
X(M, "stopped_in_lane", ["stopped in traffic lane", "stopped in traffic"])
X(M, "turning_left", ["making left turn", "turning left"])
X(M, "turning_right", ["making right turn", "turning right", "right turn on red"])
X(M, "u_turn", ["making u turn", "making u-turn"])
X(M, "accelerating_starting", ["accelerating"]); X(M, "accelerating_starting", ["starting from lane"], "medium")
X(M, "backing", ["backing"])
X(M, "changing_lanes", ["changing lanes"])
X(M, "passing", ["passing", "overtaking/passing"])
X(M, "parked_parking", ["parked"]); X(M, "parked_parking", ["parking", "starting from parked"], "medium")
X(M, "entering_leaving_lane", ["entering traffic lane", "leaving traffic lane"])
X(M, "curve", ["negotiating a curve"])
X(M, "other_unknown", ["other", "unknown", "skidding", "driverless moving veh."], "medium")
# --- Route Type
R = "Route Type"
X(R, "interstate", ["interstate (state)"]); X(R, "us_route", ["us (state)"])
X(R, "state", ["maryland (state)", "maryland (state) route"]); X(R, "county", ["county", "county route"])
X(R, "municipality", ["municipality", "municipality route"]); X(R, "ramp", ["ramp"]); X(R, "local", ["local route"])
X(R, "other", ["other public roadway", "government", "government route", "service road", "unknown", "bicycle route", "spur", "private route", "crossover"], "medium")
# --- Light
L = "Light"
X(L, "daylight", ["daylight"]); X(L, "dark_lighted", ["dark lights on", "dark - lighted"])
X(L, "dark_unlit", ["dark no lights", "dark - not lighted"]); X(L, "dark_unknown", ["dark -- unknown lighting", "dark - unknown lighting"])
X(L, "dusk", ["dusk"]); X(L, "dawn", ["dawn"]); X(L, "other_unknown", ["other", "unknown"])
# --- Traffic Control
T = "Traffic Control"
X(T, "none", ["no controls"])
X(T, "signal", ["traffic signal", "traffic control signal"]); X(T, "signal", ["lane use control signal", "other signal", "ramp meter signal"], "medium")
X(T, "flashing_signal", ["flashing traffic signal", "flashing traffic control signal", "flashing school zone signal"])
X(T, "stop_sign", ["stop sign"]); X(T, "yield_sign", ["yield sign"])
X(T, "warning_or_school_sign", ["warning sign", "other warning sign", "intersection ahead warning sign", "curve ahead warning sign",
  "reduce speed ahead warning sign", "school zone sign device", "school zone sign", "school zone", "pedestrian crossing sign",
  "pedestrian crossing", "bicycle crossing sign"], "medium")
X(T, "railroad", ["railway crossing device", "flashing railroad crossing signal*", "railroad crossing"])
X(T, "person", ["person*"]); X(T, "other_unknown", ["other", "unknown", "other pavement marking*"], "medium")
# --- Vehicle Going Dir
D = "Vehicle Going Dir"
X(D, "north", ["north", "northbound"]); X(D, "south", ["south", "southbound"])
X(D, "east", ["east", "eastbound"]); X(D, "west", ["west", "westbound"]); X(D, "unknown_or_off_road", ["unknown", "not on roadway"], "medium")
# --- Agency Name (name styles differ; truncated names in legacy)
A = "Agency Name"
X(A, "montgomery_county_police", ["montgomery county police", "montgomery"]); X(A, "rockville_police", ["rockville police departme", "rockville"])
X(A, "gaithersburg_police", ["gaithersburg police depar", "gaithersburg"]); X(A, "takoma_park_police", ["takoma park police depart", "takoma"])
X(A, "park_police", ["maryland-national capital", "mcpark"])
# --- Weather
W = "Weather"
X(W, "clear", ["clear"]); X(W, "cloudy", ["cloudy"]); X(W, "rain", ["raining", "rain"])
X(W, "winter_precip", ["snow", "blowing snow", "wintry mix", "sleet", "sleet or hail", "freezing rain or freezing drizzle"])
X(W, "fog", ["foggy", "fog, smog, smoke"]); X(W, "wind_debris", ["severe winds", "severe crosswinds", "blowing sand, soil, dirt"])
X(W, "other_unknown", ["other", "unknown"])
# --- Surface Condition
S = "Surface Condition"
X(S, "dry", ["dry"]); X(S, "wet", ["wet", "water(standing/moving)", "water (standing, moving)"])
X(S, "snow_ice", ["ice", "ice/frost", "snow", "slush"]); X(S, "other_unknown", ["other", "unknown", "mud, dirt, gravel", "oil", "sand"], "medium")
# --- Vehicle Damage Extent ('destroyed' does not exist in the new list)
V = "Vehicle Damage Extent"
X(V, "no_damage", ["no damage"]); X(V, "superficial", ["superficial"]); X(V, "functional", ["functional"]); X(V, "disabling_or_destroyed", ["disabling"])
X(V, "disabling_or_destroyed", ["destroyed"], "medium", "category absent in the new format; probably merged into 'disabling'")
X(V, "unknown_or_not_at_scene", ["unknown", "other", "vehicle not at scene"], "medium")
# --- Vehicle First Impact Location (clock positions parsed separately; non-clock values here)
I = "Vehicle First Impact Location"
X(I, "underside_top", ["underside", "roof top", "top"]); X(I, "non_collision_cargo", ["non-collision", "cargo loss"])
X(I, "unknown_or_not_at_scene", ["unknown", "vehicle not at scene"])
# --- Vehicle Body Type (class boundaries differ; SUV share shifts, so several rows are medium)
B = "Vehicle Body Type"
X(B, "passenger_car", ["passenger car", "station wagon", "limousine", "large limo"]); X(B, "suv", ["(sport) utility vehicle", "sport utility vehicle"], "medium")
X(B, "pickup", ["pickup truck", "pickup"]); X(B, "pickup", ["other light trucks*"], "medium")
X(B, "van", ["van", "van - passenger*", "van - cargo"])
X(B, "truck_medium_heavy", ["medium/heavy trucks*", "truck tractor", "single-unit truck", "other trucks", "cargo van/light truck 2 axles*"], "medium")
X(B, "bus_transit", ["transit bus", "bus - transit"]); X(B, "bus_school", ["school bus", "bus - school"])
X(B, "bus_other", ["other bus", "cross country bus", "bus - other type", "bus - mini", "bus - cross country"])
X(B, "motorcycle_moped", ["motorcycle", "moped", "autocycle", "motorcycle - 2 wheeled", "motorcycle - 3 wheeled", "moped or motorized bicycle"])
X(B, "emergency_vehicle", ["police vehicle/*", "ambulance/*", "fire vehicle/*"])
X(B, "special_other", ["recreational vehicle", "motor home", "snowmobile", "all terrain vehicle (atv)", "all-terrain vehicle/*", "recreational off-highway*",
  "farm vehicle", "farm equipment*", "construction equipment*", "golf cart", "low speed vehicle"], "medium")
X(B, "other_unknown", ["other", "unknown"])
# --- Driver Distracted By (named categories differ a lot between formats)
G = "Driver Distracted By"
X(G, "not_distracted", ["not distracted"]); X(G, "unknown", ["unknown", "no driver present"])
X(G, "phone_or_device", ["other cellular phone related", "other electronic device (navigational palm pilot)", "talking or listening to cellular phone",
  "texting from a cellular phone", "dialing cellular phone", "using device object brought into vehicle", "talking/listening", "manually operating (dialing, playing game, etc.)"])
X(G, "in_vehicle_activity", ["by other occupants", "by moving object in vehicle", "eating or drinking", "adjusting audio and or climate controls",
  "using other device controls integral to vehicle", "smoking related"])
X(G, "external_distraction", ["distracted by outside person object or event"])
X(G, "inattentive_other", ["inattentive or lost in thought", "other distraction", "other action (looking away from task, etc.)"], "medium")
X(G, "looked_but_did_not_see", ["looked but did not see"], "medium", "legacy-only category (21k rows, ~none in new format); kept separate so it can be merged or dropped as a sensitivity choice")

XW_DF = pd.DataFrame(XW, columns=["column", "raw_pattern", "harmonised", "confidence", "note"])
edited_path = os.path.join(OUT_DIR, "crosswalk_edited.csv")
if USE_EDITED_CROSSWALK and os.path.exists(edited_path):
    ed = pd.read_csv(edited_path)
    XW_DF = pd.concat([XW_DF, ed.assign(confidence=ed.get("confidence", "edited"), note=ed.get("note", "edited by user"))], ignore_index=True)
    XW_DF = XW_DF.drop_duplicates(["column", "raw_pattern"], keep="last")
    log(f"Applied {len(ed)} rows from {edited_path}")

H = pd.DataFrame(index=raw.index)       # harmonised columns
conf = pd.DataFrame(index=raw.index)    # confidence of the mapping per row
H_NAMES = {C: "collision_h", M: "movement_h", R: "route_h", L: "light_h", T: "control_h", D: "direction_h", A: "agency_h",
           W: "weather_h", S: "surface_h", V: "damage_h", B: "body_h", G: "distraction_h"}
unmapped_rows, count_rows = [], []
for col, hname in {**H_NAMES, I: "impact_other_h"}.items():
    n = norm_col(raw, col)
    sub = XW_DF[XW_DF.column == col]
    exact = {r.raw_pattern: (r.harmonised, r.confidence) for r in sub.itertuples() if not r.raw_pattern.endswith("*")}
    prefix = [(r.raw_pattern[:-1], r.harmonised, r.confidence) for r in sub.itertuples() if r.raw_pattern.endswith("*")]
    h = n.map(lambda v: exact[v][0] if v in exact else np.nan)
    cf = n.map(lambda v: exact[v][1] if v in exact else np.nan)
    for p, hv, cv in prefix:
        m = h.isna() & n.notna() & n.astype("string").str.startswith(p).fillna(False).astype(bool)
        h = h.where(~m, hv); cf = cf.where(~m, cv)
    H[hname], conf[hname] = h, cf
    if col == I:   # clock positions are mapped programmatically below, so they are not "unmapped"
        continue
    um = n.notna() & h.isna()
    if um.any():
        t = n[um].value_counts()
        unmapped_rows += [dict(column=col, raw_value=k, n_rows=int(v)) for k, v in t.items()]

# clock positions (identical meaning in both formats; 'o clock' and 'oclock' spellings)
WORDS = {w: i + 1 for i, w in enumerate(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"])}
ni = norm(raw[I])
clock = ni.astype("string").str.extract(r"^(\w+) o ?clock$")[0].map(WORDS)
H["impact_clock"] = clock.astype(float)
zone = pd.Series(np.select([clock.isin([11, 12, 1]), clock.isin([2, 3, 4]), clock.isin([5, 6, 7]), clock.isin([8, 9, 10])],
                            ["front", "right", "rear", "left"], default=None), index=raw.index).replace({None: np.nan})
H["impact_zone_h"] = zone.where(zone.notna(), H["impact_other_h"])
H = H.drop(columns="impact_other_h"); conf = conf.drop(columns="impact_other_h")
um_clock = ni.notna() & H["impact_zone_h"].isna()
if um_clock.any():
    unmapped_rows += [dict(column=I, raw_value=k, n_rows=int(v)) for k, v in ni[um_clock].value_counts().items()]

unm = pd.DataFrame(unmapped_rows, columns=["column", "raw_value", "n_rows"])
unm.to_csv(os.path.join(OUT_DIR, "unmapped_values.csv"), index=False)
log(f"Unmapped raw values: {len(unm)} distinct ({int(unm.n_rows.sum()) if len(unm) else 0} rows). "
    + ("None - every value is covered by the crosswalk." if len(unm) == 0 else "See unmapped_values.csv and add rows to crosswalk_edited.csv."))
if len(unm):
    print(unm.to_string(index=False))

# crosswalk file with counts per regime for review/editing
rows = []
for r in XW_DF.itertuples():
    n = norm_col(raw, r.column)
    m = n.astype("string").str.startswith(r.raw_pattern[:-1]).fillna(False).astype(bool) if r.raw_pattern.endswith("*") else (n == r.raw_pattern)
    rows.append(dict(column=r.column, raw_pattern=r.raw_pattern, harmonised=r.harmonised, confidence=r.confidence, note=r.note,
                     n_legacy=int((m & (regime == LEGACY)).sum()), n_new=int((m & (regime == NEW)).sum())))
xw_out = pd.DataFrame(rows)
xw_out.to_csv(os.path.join(OUT_DIR, "crosswalk_proposed.csv"), index=False, encoding="utf-8-sig")
share_conf = (conf.apply(lambda s: s.value_counts(normalize=True)).T.fillna(0) * 100).round(1)
share_conf.to_csv(os.path.join(OUT_DIR, "crosswalk_confidence_share_by_column.csv"))
print("\nShare of rows (%) by mapping confidence:\n", share_conf.to_string())

# --- Driver Substance Abuse -> alcohol_susp / drug_susp (yes/no/unknown)
sub = norm(raw["Driver Substance Abuse"])
LEG_SUB = {"none detected": ("no", "no"), "unknown": ("unknown", "unknown"), "alcohol present": ("yes", "unknown"), "alcohol contributed": ("yes", "unknown"),
           "illegal drug present": ("unknown", "yes"), "illegal drug contributed": ("unknown", "yes"), "medication present": ("unknown", "yes"),
           "medication contributed": ("unknown", "yes"), "combined substance present": ("yes", "yes"), "combination contributed": ("yes", "yes"), "other": ("unknown", "unknown")}
STATE = {"suspect of alcohol use": "yes", "not suspect of alcohol use": "no", "suspect of drug use": "yes", "not suspect of drug use": "no", "unknown": "unknown"}


def parse_sub(v):
    if v is np.nan or v is None or (isinstance(v, float) and np.isnan(v)):
        return (np.nan, np.nan)
    if v in LEG_SUB:
        return LEG_SUB[v]
    parts = [p.strip() for p in str(v).split(", ")]
    if len(parts) == 2 and all(p in STATE for p in parts):
        return (STATE[parts[0]], STATE[parts[1]])
    return ("__unmapped__", "__unmapped__")


pairs = sub.map(parse_sub)
H["alcohol_susp"], H["drug_susp"] = pairs.map(lambda t: t[0]), pairs.map(lambda t: t[1])
bad_sub = int((H["alcohol_susp"] == "__unmapped__").sum())
log(f"Driver Substance Abuse values not understood: {bad_sub}")
H.loc[H["alcohol_susp"] == "__unmapped__", ["alcohol_susp", "drug_susp"]] = np.nan

# %% 4. Direct columns, numeric cleaning and quality flags
section("4. CLEANING AND FLAGS")
out = pd.DataFrame(index=raw.index)
out["report_number"], out["local_case_number"] = raw["Report Number"], raw["Local Case Number"]
out["person_id"], out["vehicle_id"] = raw["Person ID"], raw["Vehicle ID"]
out["crash_datetime"], out["crash_year"] = dt, dt.dt.year
out["crash_quarter"] = dt.dt.to_period("Q").astype(str)
out["crash_month"], out["crash_dow"], out["crash_hour"] = dt.dt.month, dt.dt.dayofweek, dt.dt.hour
out["regime"], out["regime_by_date"], out["in_overlap_window"] = regime, regime_by_date, in_overlap.astype(int)

rt = norm(raw["ACRS Report Type"]).map({"property damage crash": "property_damage", "injury crash": "injury", "fatal crash": "fatal"})
out["report_type"] = rt
for c in H.columns:
    out[c] = H[c]
out["parked_vehicle"] = norm(raw["Parked Vehicle"])
out["driverless_vehicle"] = norm(raw["Driverless Vehicle"])
ls = raw["Drivers License State"].astype("string").str.upper().str.strip()
_b = lambda x: x.fillna(False).to_numpy(dtype=bool)
out["license_state_group"] = np.select([_b(ls.isna()), _b(ls == "MD"), _b(ls == "DC"), _b(ls == "VA"), _b(ls == "XX")],
                                       ["missing", "MD", "DC", "VA", "unknown"], "other_us_or_foreign")

# numeric cleaning (original values are never overwritten in the raw file)
sl = pd.to_numeric(raw["Speed Limit"], errors="coerce")
out["speed_limit"] = sl.where(sl.between(*SPEED_VALID))
out["flag_speed_limit_invalid"] = (~sl.between(*SPEED_VALID)).astype(int)
vy = pd.to_numeric(raw["Vehicle Year"], errors="coerce")
vy_bad = (vy < VEHICLE_YEAR_MIN) | (vy > out["crash_year"] + 1)
out["vehicle_year"] = vy.where(~vy_bad)
out["flag_vehicle_year_invalid"] = vy_bad.fillna(False).astype(int)
out["vehicle_age"] = (out["crash_year"] - out["vehicle_year"]).clip(lower=0)
lat, lon = pd.to_numeric(raw["Latitude"], errors="coerce"), pd.to_numeric(raw["Longitude"], errors="coerce")
bad_ll = ~(lat.between(*LAT_RANGE) & lon.between(*LON_RANGE))
out["latitude"], out["longitude"] = lat.where(~bad_ll), lon.where(~bad_ll)
out["flag_coord_outside_county"] = bad_ll.astype(int)
log(f"Speed limit invalid/zero -> missing: {int(out.flag_speed_limit_invalid.sum()):,}; vehicle year invalid -> missing: {int(out.flag_vehicle_year_invalid.sum()):,}; "
    f"coordinates outside county -> missing: {int(out.flag_coord_outside_county.sum()):,}")

# vehicle make: explicit aliases + prefix rule against major makes (map saved for review)
ALIAS = {"TOYT": "TOYOTA", "HOND": "HONDA", "CHEV": "CHEVROLET", "CHEVY": "CHEVROLET", "HYUN": "HYUNDAI", "NISS": "NISSAN", "ACUR": "ACURA", "SUBA": "SUBARU",
         "MERZ": "MERCEDES-BENZ", "MERCEDES": "MERCEDES-BENZ", "VOLK": "VOLKSWAGEN", "VOLKSWAGON": "VOLKSWAGEN", "VW": "VOLKSWAGEN", "VOLKS": "VOLKSWAGEN",
         "GILL": "GILLIG", "DODG": "DODGE", "MAZD": "MAZDA", "FRHT": "FREIGHTLINER", "THOMAS BUILT": "THOMAS", "THOM": "THOMAS", "LEXS": "LEXUS", "LEXU": "LEXUS",
         "CHRY": "CHRYSLER", "MITS": "MITSUBISHI", "CADI": "CADILLAC", "INFI": "INFINITI", "VOLV": "VOLVO"}
AMBIGUOUS = {"MERC"}           # could be MERCURY or MERCEDES; left unchanged
NO_PREFIX_RULE = {"MINI", "RAM", "MACK"}
mk = raw["Vehicle Make"].astype("string").str.upper().str.strip()
mk = mk.where(mk.notna() & (mk != "UNKNOWN"), pd.NA)
mk1 = mk.map(lambda v: ALIAS.get(v, v) if v is not pd.NA and v == v else v)
cnt = mk1.value_counts()
majors = [m for m, c in cnt.items() if c >= 300 and len(m) >= 4 and m not in NO_PREFIX_RULE]


def prefix_fix(v):
    if v is pd.NA or v != v or v in AMBIGUOUS or cnt.get(v, 0) >= 300:
        return v
    c = [m for m in majors if v.startswith(m) and v != m]
    return c[0] if len(c) == 1 else v


mk2 = mk1.map(prefix_fix)
out["make_h"] = mk2.astype(object).where(mk2.notna(), np.nan)
mk_map = pd.DataFrame({"raw": mk, "mapped": mk2}).dropna().groupby(["raw", "mapped"]).size().reset_index(name="n_rows")
mk_map = mk_map[mk_map.raw != mk_map.mapped].sort_values("n_rows", ascending=False)
mk_map.to_csv(os.path.join(OUT_DIR, "vehicle_make_map.csv"), index=False)
log(f"Vehicle make: {mk.nunique():,} raw spellings -> {mk2.nunique():,} after mapping; {int(mk_map.n_rows.sum()):,} rows re-labelled (see vehicle_make_map.csv)")

out["road_name"] = raw["Road Name"].astype("string").str.upper().str.strip()
out["cross_street_name"] = raw["Cross-Street Name"].astype("string").str.upper().str.strip()

# %% 5. Targets and crash-level fields
section("5. TARGETS AND CRASH-LEVEL FIELDS")
fault = norm(raw["Driver At Fault"])
out["y_fault"] = fault.map({"yes": 1.0, "no": 0.0})                       # 'unknown' -> NaN (kept, not dropped)
inj = norm(raw["Injury Severity"])
ORD = {"no apparent injury": 0, "possible injury": 1, "suspected minor injury": 2, "suspected serious injury": 3, "fatal injury": 4}
out["injury_ord"] = inj.map(ORD).astype(float)
out["y_injury_any"] = (out["injury_ord"] >= 1).astype(float).where(out["injury_ord"].notna())
out["y_ksi"] = (out["injury_ord"] >= 3).astype(float).where(out["injury_ord"].notna())
out["flag_fault_unknown"] = out["y_fault"].isna().astype(int)
g = out.groupby("report_number")
out["n_drivers_in_crash"] = g["person_id"].transform("size")
out["n_fault_yes_in_crash"] = g["y_fault"].transform("sum")
out["two_driver_crash"] = (out["n_drivers_in_crash"] == 2).astype(int)
out["zero_fault_two_driver"] = ((out["two_driver_crash"] == 1) & (out["n_fault_yes_in_crash"] == 0)).astype(int)
log(f"Targets: y_fault non-missing {int(out.y_fault.notna().sum()):,}; y_injury_any non-missing {int(out.y_injury_any.notna().sum()):,}; "
    f"y_ksi events {int(out.y_ksi.sum()):,} (legacy {int(out.y_ksi[out.regime == LEGACY].sum()):,}, new {int(out.y_ksi[out.regime == NEW].sum()):,})")

# %% 6. Save harmonised dataset and data dictionary
section("6. SAVE")
assert len(out) == N, "row count changed"
assert out["person_id"].is_unique, "Person ID not unique"
out_path = os.path.join(OUT_DIR, "crash_drivers_harmonised.csv")
out.to_csv(out_path, index=False, encoding="utf-8-sig")
try:
    out.to_parquet(os.path.join(OUT_DIR, "crash_drivers_harmonised.parquet"), index=False)
except Exception as e:
    print("Parquet not written:", type(e).__name__)
DESC = {
    "regime": "Reporting format assigned per row from coding style (legacy UPPER-CASE vs new mixed-case); date is only a fallback",
    "regime_by_date": "Regime from the 1 Jan 2024 date rule (for comparison)", "in_overlap_window": "1 if the crash falls between the first new-style and last legacy-style row",
    "y_fault": "1 = driver flagged at fault, 0 = not; NaN when 'Unknown' (legacy only)", "y_injury_any": "1 if possible/minor/serious/fatal injury; NaN if injury missing",
    "y_ksi": "1 if suspected serious or fatal injury; NaN if injury missing", "injury_ord": "0 none ... 4 fatal",
    "n_fault_yes_in_crash": "Number of drivers flagged at fault in the crash", "zero_fault_two_driver": "Two-driver crash where no driver is flagged at fault",
    "speed_limit": "Speed limit; 0 and values outside 5-70 set to missing", "vehicle_year": "Vehicle year; <1950 or > crash year + 1 set to missing",
    "vehicle_age": "Crash year minus vehicle year (floored at 0)", "latitude": "Set to missing when outside the approximate county box", "longitude": "Same as latitude",
    "make_h": "Vehicle make with abbreviations mapped (see vehicle_make_map.csv)", "alcohol_susp": "yes/no/unknown parsed from both formats of Driver Substance Abuse",
    "drug_susp": "yes/no/unknown parsed from both formats of Driver Substance Abuse", "impact_clock": "First impact location as clock position 1-12",
    "impact_zone_h": "front (11-1), right (2-4), rear (5-7), left (8-10), or non-clock categories",
}
for raw_col, hname in H_NAMES.items():
    DESC[hname] = f"Common categories for '{raw_col}' (see crosswalk_proposed.csv)"
dd = pd.DataFrame({"column": out.columns, "dtype": [str(t) for t in out.dtypes], "missing_pct": (out.isna().mean() * 100).round(2).values,
                   "n_unique": [out[c].nunique() for c in out.columns], "description": [DESC.get(c, "") for c in out.columns]})
dd.to_csv(os.path.join(OUT_DIR, "data_dictionary.csv"), index=False, encoding="utf-8-sig")
log(f"Saved {out_path} ({len(out):,} rows x {out.shape[1]} columns); no rows were dropped.")

# %% 7. Validation: label coverage before/after, drift (PSI), regime detectability
section("7. VALIDATION")
CATS = list(H_NAMES.values()) + ["impact_zone_h", "alcohol_susp", "drug_susp", "report_type", "body_h"]
CATS = list(dict.fromkeys(CATS))
RAW_OF = {**{v: k for k, v in H_NAMES.items()}, "impact_zone_h": I, "alcohol_susp": "Driver Substance Abuse", "drug_susp": "Driver Substance Abuse"}


def coverage(series, reg):
    """Share of rows whose category occurs in BOTH regimes (a model can only compare those)."""
    s = pd.DataFrame({"v": series, "r": reg}).dropna()
    both = set(s[s.r == LEGACY].v) & set(s[s.r == NEW].v)
    return 100 * s.v.isin(both).mean()


cov = []
for h, rc in RAW_OF.items():
    if h in out.columns:
        cov.append(dict(column=rc if h in ("alcohol_susp", "drug_susp") else rc, harmonised=h, raw_overlap_pct=round(coverage(norm(raw[rc]), regime), 1),
                        harmonised_overlap_pct=round(coverage(out[h], out["regime"]), 1)))
cov = pd.DataFrame(cov).drop_duplicates("harmonised").sort_values("raw_overlap_pct")
cov.to_csv(os.path.join(OUT_DIR, "coverage_before_after.csv"), index=False)
print("Rows (%) whose label exists in both regimes - raw vs harmonised:\n", cov.to_string(index=False))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
y = np.arange(len(cov))
ax.hlines(y, cov.raw_overlap_pct, cov.harmonised_overlap_pct, color=GREY, lw=2)
ax.scatter(cov.raw_overlap_pct, y, color=GREY, s=36, label="raw labels", zorder=3)
ax.scatter(cov.harmonised_overlap_pct, y, color=BLUE, s=36, label="harmonised", zorder=3)
ax.set_yticks(y); ax.set_yticklabels(cov.harmonised); ax.set_xlabel("Rows with a label present in both regimes (%)"); ax.legend(loc="lower left")
save_fig(fig, "fig01_label_coverage.png")


def psi(a, b, eps=1e-4):
    cats = pd.Index(a.dropna().unique()).union(b.dropna().unique())
    pa = a.value_counts(normalize=True).reindex(cats).fillna(0) + eps
    pb = b.value_counts(normalize=True).reindex(cats).fillna(0) + eps
    return float(((pb - pa) * np.log(pb / pa)).sum())


leg, new = out[out.regime == LEGACY], out[out.regime == NEW]
rows = [dict(variable=c, psi=round(psi(leg[c], new[c]), 3)) for c in CATS if c in out.columns]
for c in ["speed_limit", "vehicle_age", "crash_hour"]:
    q = pd.qcut(out[c], 8, duplicates="drop")
    rows.append(dict(variable=c, psi=round(psi(q[out.regime == LEGACY], q[out.regime == NEW]), 3)))
for c in ["y_fault", "y_injury_any", "y_ksi"]:
    rows.append(dict(variable=c + " (target)", psi=round(psi(leg[c].astype("string"), new[c].astype("string")), 3)))
psi_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
psi_df["reading"] = np.where(psi_df.psi >= 0.25, "major drift", np.where(psi_df.psi >= 0.10, "moderate", "stable"))
psi_df.to_csv(os.path.join(OUT_DIR, "drift_psi_after_harmonisation.csv"), index=False)
print("\nPopulation stability index (legacy vs new) after harmonisation:\n", psi_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(6.5, 5))
d = psi_df.sort_values("psi")
ax.barh(d.variable, d.psi, color=np.where(d.psi >= 0.25, ORANGE, BLUE))
ax.axvline(0.25, color=GREY, ls="--", lw=1); ax.axvline(0.10, color=GREY, ls=":", lw=1)
ax.set_xlabel("PSI, legacy vs new")
save_fig(fig, "fig02_drift_psi.png")

if RUN_DOMAIN_CLASSIFIER:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score

    def domain_auc(cat_cols, num_cols, name):
        X = pd.concat([out[cat_cols].astype("string").fillna("missing").apply(lambda s: s.astype("category").cat.codes), out[num_cols]], axis=1)
        yv = (out.regime == NEW).astype(int)
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(X, yv, out.report_number))
        mdl = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0,
                                             categorical_features=[True] * len(cat_cols) + [False] * len(num_cols))
        mdl.fit(X.iloc[tr], yv.iloc[tr])
        auc = roc_auc_score(yv.iloc[te], mdl.predict_proba(X.iloc[te])[:, 1])
        log(f"Regime detectability (AUROC, grouped split) - {name}: {auc:.3f}")
        return auc

    num_all = ["speed_limit", "vehicle_age", "crash_hour", "crash_dow", "latitude", "longitude", "impact_clock"]
    cat_all = [c for c in CATS if c in out.columns and c != "report_type"] + ["license_state_group"]
    res = {
        "all harmonised predictors": domain_auc(cat_all, num_all, "all harmonised predictors"),
        "without agency, distraction and substance": domain_auc([c for c in cat_all if c not in ("agency_h", "distraction_h", "alcohol_susp", "drug_susp")], num_all,
                                                                "without agency, distraction and substance"),
        "environment only": domain_auc(["weather_h", "surface_h", "light_h"], ["crash_hour", "crash_dow"], "environment only (weather, surface, light, hour, weekday)"),
    }
    pd.Series(res, name="auroc").to_csv(os.path.join(OUT_DIR, "regime_detectability_auroc.csv"))

# %% 8. Report
section("8. REPORT")
log("\nLabel check (these should still show the drift, because the fault label is NOT altered here):")
fr = out.groupby("regime").agg(fault_rate=("y_fault", "mean"), zero_fault_two_driver=("zero_fault_two_driver", "mean"), ksi_rate=("y_ksi", "mean"))
log(fr.round(4).to_string())
with open(os.path.join(OUT_DIR, "preprocessing_report.md"), "w", encoding="utf-8") as f:
    f.write("# Preprocessing report\n\n```\n" + "\n".join(REPORT) + "\n```\n")
print(f"\nAll outputs saved in: {OUT_DIR}")
