"""
Brooklyn Bus Equity Explorer

Two ways to run this file:
    python app1.py --build      Reads the raw folders once and writes small files to data/processed/
    streamlit run app1.py       Starts the app (reads only data/processed/)

The build step needs geopandas, shapely and pyproj.
The app needs streamlit, pandas, numpy, pyarrow and scipy (the map is the components/brooklyn_map folder).
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import zipfile
import re
from html import escape as esc, unescape
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
logging.getLogger("streamlit").setLevel(logging.ERROR)
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

# =============================================================================
# Configuration: edit this block to match your folders
# =============================================================================
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"

TRACT_DIR = ROOT / "tl_2025_36"
BG_DIR = ROOT / "tl_2025_36_bg"
ACS_TRACT_DIR = ROOT / "05_boundaries_demographics" / "ACS_2024_tract_equity"
ACS_BG_DIR = ROOT / "05_boundaries_demographics" / "ACS_2024_block_groups"
MONTHLY_FILE = ROOT / "03_stop_level_ridership" / "mta_bus_stop_level_ridership_full.csv"
RIDERSHIP_FILE = "mta_bus_stop_level_ridership_full.csv"      # used only if the monthly file is missing

EQUITY_UNIT = "tract"        # "tract" or "block group". Tracts are built from block-group tables when needed.

# The run looks in these folders in order and uses the first one that holds a GTFS feed.
# A feed can be an extracted folder (with stop_times.txt) or a .zip file.
RUN = {
    "key": "brooklyn_current",
    # The MTA download archive; only gtfs_b and gtfs_busco are used for Brooklyn.
    "gtfs": ["Jun to Sep GTFS"],
    "ref_date": "2026-08-12",                      # picks the bundle and the typical Wednesday (None = middle of bundle)
    "months": ("2026-08", "2026-08"),              # default ridership months
    "folder": "03_stop_level_ridership/august_2026",    # fallback if the monthly file is missing
    "period": "August 2026",
}

COUNTY = {"Brooklyn": "047"}
AM_START, AM_END = 7 * 3600, 10 * 3600         # AM peak: 7 to 10 AM
PM_START, PM_END = 16 * 3600, 19 * 3600        # PM peak: 4 to 7 PM
DAY_START, DAY_END = 6 * 3600, 21 * 3600       # all-day frequent-service window
FREQ_MIN = 10.0                                # frequent = average headway of 10 minutes or better
WALK_FT = 1320                                 # quarter mile, in feet
WALK_M = 402.3                                 # quarter mile, in meters
CRS_FT = 2263                                  # NY State Plane Long Island, feet
MIN_POP = 100                                  # areas below this get no equity score
KEYS = ["stop_id", "route_id", "direction_id"]
DEFAULT_DATE = "2026-08-12"
TOP_TRACTS, TOP_N = 10, 5      # rows in the top lists
MIN_TRACT_POP = 500            # tracts with fewer residents are left out of the top-tracts list
MIN_STOP_BOARDINGS = 50        # stops with fewer boardings on the selected date are left out of the top-stops list
SCHEDULE_DIR = PROC / RUN["key"] / "daily_schedule"
COMPONENT_DIR = ROOT / "components" / "brooklyn_map"

# Eight indicators, in the order and with the weights of MTA's Equity Priority Area table.
WEIGHTS = {"minority": 10, "zero_vehicle": 10, "poverty": 9, "commute": 9,
           "disability": 6, "lep": 5, "age": 3, "education": 1}
INDICATOR_LABEL = {"minority": "Minority population", "zero_vehicle": "Zero-vehicle households",
                   "poverty": "Population in poverty", "commute": "Commute over 45 minutes",
                   "disability": "Disability (under 65)", "lep": "Limited-English households",
                   "age": "Under 18 or 75 and over", "education": "High school or lower (25+)"}
# ACS variable numbers: table, [numerator], [denominator]
SPECS = {
    "zero_vehicle": ("B25044", [3, 10], [1]),
    "poverty": ("B17001", [2], [1]),
    "commute": ("B08303", [11, 12, 13], [1]),
    "disability": ("B18101", [4, 7, 10, 13, 23, 26, 29, 32], [3, 6, 9, 12, 22, 25, 28, 31]),
    "lep": ("C16002", [4, 7, 10, 13], [1]),
    "age": ("B01001", [3, 4, 5, 6, 23, 24, 25, 27, 28, 29, 30, 47, 48, 49], [1]),
    "education": ("B15003", list(range(2, 19)), [1]),
}
# Phrases each variable's label should contain (checked when a Column-Metadata file is present)
EXPECT = {
    "B03002": {3: "white alone"},
    "B25044": {3: "no vehicle", 10: "no vehicle"},
    "B17001": {2: "below poverty"},
    "B08303": {11: "45 to 59", 12: "60 to 89", 13: "90 or more"},
    "B18101": {3: "under 5", 4: "with a disability", 12: "35 to 64", 13: "with a disability",
               22: "under 5", 23: "with a disability", 31: "35 to 64", 32: "with a disability"},
    "C16002": {4: "limited", 7: "limited", 10: "limited", 13: "limited"},
    "B01001": {3: "under 5", 6: "15 to 17", 23: "75 to 79", 25: "85 years", 27: "under 5", 30: "15 to 17",
               47: "75 to 79", 49: "85 years"},
    "B15003": {2: "no schooling", 17: "regular high school diploma", 18: "ged", 19: "some college"},
}

# Light basemap: your own OpenStreetMap-based layers, drawn once into map tiles (with street names) by render_basemap_tiles.py.
# The tiles sit in the static/tiles folder and Streamlit serves them itself (.streamlit/config.toml turns that on), so the map
# uses no tile service and no API key, and the page stays small because the tiles load separately and are cached by the browser.
TILE_DIR = ROOT / "static" / "tiles"
TILE_URL = "https://mkp-7.github.io/brooklyn-bus-equity-explorer/static/tiles/{z}/{x}/{y}.png"
TILE_ZOOM = (10, 16)                 # zooms that were drawn; closer views enlarge the last one
TILES = {"Light": TILE_URL, "None": None}
TILE_ATTR = "&copy; OpenStreetMap contributors"
RAMP_LO, RAMP_HI = (232, 241, 252), (8, 48, 140)       # light to dark blue
NO_DATA = [205, 205, 205, 90]


# The only routes this app uses, exactly as listed. A slash joins routes that are listed together: SIM1/SIM1C means SIM1 and SIM1C.
BROOKLYN_ROUTES = """
B1, B2, B3, B4, B6, B7, B8, B9, B11, B12, B13, B14, B15, B16, B17, B20, B24, B25, B26, B31, B32, B35, B36, B37, B38, B39,
B41, B42, B43, B44, B44 SBS, B45, B46, B46 SBS, B47, B48, B49, B52, B54, B57, B60, B61, B62, B63, B64, B65, B67, B68, B69,
B70, B74, B82, B82 SBS, B83, B84, B96, B100, B103, BM1, BM2, BM3, BM4, BM5, X27, X28, X37, X38,
SIM1/SIM1C, SIM2, SIM3/SIM3C, SIM4/SIM4C, SIM33/SIM33C
"""
ROUTE_NAMES = [r.strip() for part in BROOKLYN_ROUTES.replace("\n", " ").split(",") for r in part.split("/") if r.strip()]


def _route_key(s) -> str:
    """Spelling-proof key. B44+, B44-SBS and B44 SBS are one route, and case and spaces do not matter."""
    return str(s).strip().upper().replace(" ", "").replace("-", "").replace("+", "SBS")


ROUTE_BY_KEY = {_route_key(n): n for n in ROUTE_NAMES}


def canon_route(route_id):
    """The route's name as listed above (for example 'B44 SBS'), or None when it is not one of the listed routes."""
    return ROUTE_BY_KEY.get(_route_key(route_id))


def canon_series(s: pd.Series) -> pd.Series:
    """canon_route for a whole column: listed routes get their listed name, every other route becomes NaN."""
    key = (s.astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
           .str.replace("-", "", regex=False).str.replace("+", "SBS", regex=False))
    return key.map(ROUTE_BY_KEY)


def canon_frame(df: pd.DataFrame, col: str = "route_id") -> pd.DataFrame:
    """Rename routes to their listed names and drop every other route."""
    if col not in df.columns:
        return df
    out = df.copy()
    out[col] = canon_series(out[col])
    return out[out[col].notna()]


def is_brooklyn(route_id) -> bool:
    """True only for the listed routes. Bronx, Queens and Manhattan routes, and Brooklyn routes not on the list, are False."""
    return canon_route(route_id) is not None


# =============================================================================
# BUILD STEP  (python app1.py --build)
# =============================================================================
# ---- GTFS reading ------------------------------------------------------------
def _csv(d, name, cols=None):
    """Read one GTFS file from an extracted folder or a .zip. Returns None if it is not there."""
    d = Path(d)
    keep = (lambda c: c in cols) if cols else None
    if d.suffix.lower() == ".zip":
        with zipfile.ZipFile(d) as z:
            names = {Path(n).name: n for n in z.namelist()}
            if name not in names:
                return None
            with z.open(names[name]) as f:
                return pd.read_csv(f, dtype=str, usecols=keep, encoding="utf-8-sig")
    p = d / name
    if not p.exists():
        return None
    return pd.read_csv(p, dtype=str, usecols=keep, encoding="utf-8-sig")


def _zip_is_gtfs(z) -> bool:
    try:
        with zipfile.ZipFile(z) as f:
            return any(Path(n).name == "stop_times.txt" for n in f.namelist())
    except zipfile.BadZipFile:
        return False


def _label(d) -> str:
    d = Path(d)
    try:
        return str(d.relative_to(ROOT))
    except ValueError:
        return str(d)


def _feed_range(d):
    cal = _csv(d, "calendar.txt", {"start_date", "end_date"})
    if cal is None or cal.empty:
        return None
    return (pd.to_datetime(cal["start_date"].min(), format="%Y%m%d"),
            pd.to_datetime(cal["end_date"].max(), format="%Y%m%d"))


def discover_feeds(roots):
    """First root that holds GTFS: extracted folders (with stop_times.txt), else .zip files."""
    for rel in roots:
        root = ROOT / rel
        if not root.exists():
            continue
        dirs = sorted({p.parent for p in root.rglob("stop_times.txt")})
        zips = [] if dirs else [z for z in sorted(root.rglob("*.zip")) if _zip_is_gtfs(z)]
        if dirs or zips:
            return rel, dirs + zips
    return None, []


def select_by_date(feeds, ref):
    """Keep feeds whose service dates include the reference date. If none do, keep all and say so."""
    info = [(f, _feed_range(f)) for f in feeds]
    if not ref:
        return list(feeds), True, info
    ref = pd.Timestamp(ref)
    inside = [f for f, r in info if r and r[0] <= ref <= r[1]]
    return (inside or list(feeds)), bool(inside), info


def _secs(s: pd.Series) -> pd.Series:
    """GTFS time text (can pass 24:00:00) to seconds. Bad values become NaN."""
    s = s.astype("string").str.strip()
    ok = s.str.match(r"^\d+:\d{2}:\d{2}$").fillna(False).astype(bool)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    if not ok.any():
        return out
    parts = s[ok].str.split(":", expand=True).astype("int64")
    out.loc[ok] = (parts[0] * 3600 + parts[1] * 60 + parts[2]).to_numpy()
    return out


def _target_wednesday(ref, lo, hi):
    """Only used when no reference date is given: a Wednesday near the middle of the bundle."""
    t = pd.Timestamp(ref) if ref else lo + (hi - lo) / 2
    if lo is not None:
        t = min(max(t, lo), hi)
    t = t - pd.Timedelta(days=(t.weekday() - 2) % 7)
    if lo is not None and t < lo:
        t += pd.Timedelta(days=7)
    return t.normalize()


def service_ids_from_tables(cal, cal_dates, date):
    """Service IDs running on one exact date: calendar.txt weekday flags and date range, then calendar_dates.txt
    exceptions (type 1 adds a service, type 2 removes one). An empty set means no service ran that day."""
    date = pd.Timestamp(date).normalize()
    ids = set()
    if cal is not None and not cal.empty:
        start = pd.to_datetime(cal["start_date"], format="%Y%m%d")
        end = pd.to_datetime(cal["end_date"], format="%Y%m%d")
        day = date.day_name().lower()
        ids = set(cal[(cal[day] == "1") & (start <= date) & (end >= date)]["service_id"])
    if cal_dates is not None and not cal_dates.empty:
        d = cal_dates[cal_dates["date"] == date.strftime("%Y%m%d")]
        ids |= set(d[d["exception_type"] == "1"]["service_id"])
        ids -= set(d[d["exception_type"] == "2"]["service_id"])
    return ids


def _active_services(cal, cal_dates, ref):
    """Service IDs for the reference date (exact). With no date, a Wednesday in the middle of the bundle."""
    if cal is None or cal.empty:
        return None, None
    lo = pd.to_datetime(cal["start_date"].min(), format="%Y%m%d")
    hi = pd.to_datetime(cal["end_date"].max(), format="%Y%m%d")
    target = pd.Timestamp(ref).normalize() if ref else _target_wednesday(None, lo, hi)
    return service_ids_from_tables(cal, cal_dates, target), target


def _waits(st_: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Expected wait in minutes for a rider arriving at a random moment inside each window, from scheduled departures.
    A window is a list of (start, end) spans in seconds, for example the AM peak and the PM peak together.
    Only the time between two scheduled departures counts, so hours before the first bus or after the last do not.
    For even headway h this is h / 2. In general it is sum(h^2) / (2 * sum(h)), with gaps cut off at the window edges."""
    d = st_.sort_values(KEYS + ["t"])[KEYS + ["t"]].copy()
    d["prev"] = d.groupby(KEYS)["t"].shift(1)
    d = d[d["prev"].notna() & (d["t"] > d["prev"])]
    cur, prev = d["t"].to_numpy(), d["prev"].to_numpy()
    out = None
    for name, spans in windows.items():
        pieces = []
        for t0, t1 in spans:
            a, b = np.maximum(prev, t0), np.minimum(cur, t1)
            ok = b > a
            w = d.loc[ok, KEYS].copy()
            w["num"] = ((cur[ok] - a[ok]) ** 2 - (cur[ok] - b[ok]) ** 2) / 2
            w["den"] = b[ok] - a[ok]
            pieces.append(w)
        g = pd.concat(pieces).groupby(KEYS, as_index=False)[["num", "den"]].sum()
        g[name] = g["num"] / g["den"] / 60
        g = g[KEYS + [name]]
        out = g if out is None else out.merge(g, on=KEYS, how="outer")
    return out


def _service_metrics(st_: pd.DataFrame) -> pd.DataFrame:
    """Per stop, route and direction on one date: trips, expected waits (AM peak, PM peak, both), overnight trips."""
    out = st_.groupby(KEYS).size().rename("all_trips").reset_index()
    am = st_[(st_["t"] >= AM_START) & (st_["t"] < AM_END)].groupby(KEYS).size().rename("am_trips").reset_index()
    pm = st_[(st_["t"] >= PM_START) & (st_["t"] < PM_END)].groupby(KEYS).size().rename("pm_trips").reset_index()
    day = st_[(st_["t"] >= DAY_START) & (st_["t"] < DAY_END)].groupby(KEYS).size().rename("day_trips").reset_index()
    night_mask = ((st_["t"] >= 0) & (st_["t"] < 4 * 3600)) | ((st_["t"] >= 24 * 3600) & (st_["t"] < 28 * 3600))
    night = st_[night_mask].groupby(KEYS).size().rename("night_trips").reset_index()
    out = (out.merge(am, on=KEYS, how="left").merge(pm, on=KEYS, how="left")
              .merge(day, on=KEYS, how="left").merge(night, on=KEYS, how="left"))
    waits = _waits(st_, {"am_wait_min": [(AM_START, AM_END)], "pm_wait_min": [(PM_START, PM_END)],
                         "peak_wait_min": [(AM_START, AM_END), (PM_START, PM_END)]})
    if waits is None:
        for c in ("am_wait_min", "pm_wait_min", "peak_wait_min"):
            out[c] = np.nan
    else:
        out = out.merge(waits, on=KEYS, how="left")
    for c in ("am_trips", "pm_trips", "day_trips", "night_trips"):
        out[c] = out[c].fillna(0).astype(int)
    out["day_headway_min"] = np.where(out["day_trips"] > 0, (DAY_END - DAY_START) / 60 / out["day_trips"].clip(lower=1), np.nan)
    out["frequent"] = (out["day_headway_min"] <= FREQ_MIN).fillna(False)
    return out


def _bearing_dir(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians(lat1))
    dy = lat2 - lat1
    ang = math.degrees(math.atan2(dx, dy))
    if -45 <= ang < 45:
        return "N"
    if 45 <= ang < 135:
        return "E"
    if -135 <= ang < -45:
        return "W"
    return "S"


def _process_feed(d, ref_date):
    from shapely.geometry import LineString

    stops = _csv(d, "stops.txt", {"stop_id", "stop_name", "stop_lat", "stop_lon"})
    trips = _csv(d, "trips.txt", {"route_id", "service_id", "trip_id", "direction_id", "shape_id"})
    st_ = _csv(d, "stop_times.txt", {"trip_id", "stop_id", "stop_sequence", "departure_time", "arrival_time"})
    if stops is None or trips is None or st_ is None:
        raise FileNotFoundError(f"{_label(d)} needs stops.txt, trips.txt and stop_times.txt")
    cal = _csv(d, "calendar.txt")
    ids, target = _active_services(cal, _csv(d, "calendar_dates.txt"), ref_date)
    if ids is not None and not ids:
        raise ValueError(f"No scheduled service on {target.date()} in {_label(d)}")
    rng = None
    if cal is not None and not cal.empty:
        rng = [str(cal["start_date"].min()), str(cal["end_date"].max())]

    trips["route_id"] = canon_series(trips["route_id"])            # B44+ becomes 'B44 SBS'; routes that are not listed are dropped
    trips = trips[trips["route_id"].notna()]
    if trips.empty:
        raise ValueError(f"None of the listed routes run in {_label(d)}")
    if "direction_id" not in trips:
        trips["direction_id"] = "0"
    trips["direction_id"] = trips["direction_id"].fillna("0")
    if ids is not None:
        trips = trips[trips["service_id"].isin(ids)]
    if trips.empty:                                                  # for example, weekday-only express routes on a Saturday
        raise ValueError(f"None of the listed routes run on {target.date() if target is not None else 'this date'} in {_label(d)}")
    weekday_trips = (trips.groupby(["route_id", "direction_id"])["trip_id"].nunique()
                     .rename("weekday_trips").reset_index())
    tcol = "departure_time" if "departure_time" in st_ else "arrival_time"
    st_ = st_[st_["trip_id"].isin(trips["trip_id"])].copy()
    st_["t"] = _secs(st_[tcol])
    st_ = st_.dropna(subset=["t"])
    st_["seq"] = st_["stop_sequence"].astype(int)
    st_ = st_.merge(trips[["trip_id", "route_id", "direction_id"]], on="trip_id")
    svc = _service_metrics(st_[KEYS + ["t"]])

    # Compass direction of each route and direction, from the first and last stop of its longest trip
    n = st_.groupby("trip_id").size().rename("n")
    rep = trips.merge(n, on="trip_id").sort_values("n", ascending=False).drop_duplicates(["route_id", "direction_id"])
    seq = st_[st_["trip_id"].isin(rep["trip_id"])][["trip_id", "stop_id", "seq"]].merge(
        stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id")
    seq["lat"] = seq["stop_lat"].astype(float)
    seq["lon"] = seq["stop_lon"].astype(float)
    seq = seq.sort_values(["trip_id", "seq"])
    grp = seq.groupby("trip_id")
    first, last = grp.first(), grp.last()
    rd = rep[["trip_id", "route_id", "direction_id"]].copy()
    rd["dir"] = [_bearing_dir(first.loc[t, "lon"], first.loc[t, "lat"], last.loc[t, "lon"], last.loc[t, "lat"])
                 for t in rd["trip_id"]]
    rd = rd.drop(columns="trip_id")

    shapes = []
    sh = _csv(d, "shapes.txt", {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"})
    if sh is not None and "shape_id" in trips:
        use = trips[["route_id", "direction_id", "shape_id"]].dropna().drop_duplicates(["route_id", "shape_id"])
        sh = sh[sh["shape_id"].isin(use["shape_id"])].copy()
        sh["seq"] = sh["shape_pt_sequence"].astype(int)
        sh["lat"] = sh["shape_pt_lat"].astype(float)
        sh["lon"] = sh["shape_pt_lon"].astype(float)
        sh = sh.sort_values(["shape_id", "seq"])
        lines = {sid: LineString(list(zip(g["lon"], g["lat"]))).simplify(0.00003)
                 for sid, g in sh.groupby("shape_id") if len(g) > 1}
        shapes = [(r.route_id, r.direction_id, lines[r.shape_id]) for r in use.itertuples() if r.shape_id in lines]

    return {"svc": svc, "stops": stops, "routedir": rd, "shapes": shapes, "weekday_trips": weekday_trips,
            "target": None if target is None else str(target.date()), "range": rng}


# ---- Geography and equity ----------------------------------------------------
def _union(gs):
    return gs.union_all() if hasattr(gs, "union_all") else gs.unary_union


def _coverage(area_ft, pts_ft):
    if len(pts_ft) == 0:
        return np.zeros(len(area_ft))
    u = _union(pts_ft.geometry.buffer(WALK_FT))
    share = area_ft.geometry.intersection(u).area / area_ft.geometry.area
    return share.clip(0, 1).to_numpy()


def _find_shp(folder: Path):
    found = sorted(folder.glob("*.shp")) if folder.exists() else []
    return found[0] if found else None


def _read_acs(p: Path):
    """Read an ACS download. Handles data.census.gov files (GEO_ID plus a label row) and cleaned files (GEOID)."""
    try:
        df = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str).str.strip()
    elif "GEO_ID" in df.columns:
        df = df[df["GEO_ID"].str.contains("US", na=False)].copy()
        df["GEOID"] = df["GEO_ID"].str.split("US").str[-1]
    else:
        return None
    return df.drop_duplicates("GEOID")


def _acs_candidates(table: str, dirs):
    """Candidate files, preferring earlier folders in `dirs`, then cleaned files, then data.census.gov Data files."""
    out = []
    for i, d in enumerate(dirs):
        if not d.exists():
            continue
        for p in d.rglob("*.csv"):
            n = p.name.lower()
            if table.lower() in n and "metadata" not in n and "notes" not in n:
                out.append((i, "clean" not in n, "data" not in n, len(str(p)), p))
    return [p for *_, p in sorted(out)]


def _load_acs(table: str, dirs, want_len: int):
    """First candidate file for this table whose rows are at the wanted geography (12 digits block group, 11 tract)."""
    for p in _acs_candidates(table, dirs):
        df = _read_acs(p)
        if df is None or df.empty:
            continue
        if (df["GEOID"].str.len() == want_len).mean() > 0.9:
            prefixes = tuple("36" + c for c in COUNTY.values())
            return df[df["GEOID"].str.startswith(prefixes)].copy(), p
    return None, None


def _check_labels(table: str, dirs):
    """If a Column-Metadata file sits next to the data, confirm the variable numbers match their labels."""
    meta = None
    for d in dirs:
        if d.exists():
            for p in d.rglob("*.csv"):
                n = p.name.lower()
                if table.lower() in n and "metadata" in n and "column" in n:
                    meta = p
                    break
        if meta:
            break
    if meta is None:
        print(f"   {table}: label check skipped (no Column-Metadata file found)")
        return
    m = pd.read_csv(meta, dtype=str)
    m.columns = [c.strip() for c in m.columns]
    lab = dict(zip(m["Column Name"], m["Label"]))
    bad = []
    for num, phrase in EXPECT.get(table, {}).items():
        col = f"{table}_{num:03d}E"
        if col in lab and phrase not in str(lab[col]).lower():
            bad.append(f"{col} is '{lab[col]}', expected to contain '{phrase}'")
    print(f"   {table}: label check " + ("OK" if not bad else "WARNING: " + "; ".join(bad)))


def build_areas():
    """Equity layer for Brooklyn: tracts (or block groups) with an equity score built from eight ACS indicators."""
    import geopandas as gpd

    unit = EQUITY_UNIT
    shp = _find_shp(TRACT_DIR if unit == "tract" else BG_DIR)
    if shp is None:
        raise FileNotFoundError(f"No shapefile found for {unit}s in {TRACT_DIR if unit == 'tract' else BG_DIR}")
    unit_len = 11 if unit == "tract" else 12
    print(f"   Equity unit: {unit}  (shapefile: {_label(shp)})")
    g = gpd.read_file(shp)
    g["GEOID"] = g["GEOID"].astype(str)
    g = g[g["GEOID"].str[:5].isin(["36" + c for c in COUNTY.values()])].copy()

    def load(table):
        own = ACS_TRACT_DIR if unit == "tract" else ACS_BG_DIR
        search_dirs = [own, ACS_BG_DIR, ROOT / table]     # tract downloads may sit in the folder named for block groups
        df, p = _load_acs(table, search_dirs, unit_len)
        if df is not None:
            return df, p, False
        if unit == "tract":                                  # fall back to block groups and add them up to tracts
            df, p = _load_acs(table, [ACS_BG_DIR, ROOT / table], 12)
            if df is not None:
                return df, p, True
        return None, None, False

    def numeric(df, table, nums):
        cols = [f"{table}_{n:03d}E" for n in nums]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{table} file lacks columns {missing}")
        return df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)

    def at_unit(df, rolled, num, den):
        nd = pd.DataFrame({"GEOID": df["GEOID"].to_numpy(), "num": num.to_numpy(), "den": den.to_numpy()})
        if rolled:
            nd["GEOID"] = nd["GEOID"].str[:unit_len]
            return nd.groupby("GEOID").sum(min_count=1)
        return nd.set_index("GEOID")

    meta_dirs = [ACS_BG_DIR, ACS_TRACT_DIR]
    df, p, rolled = load("B03002")
    if df is None:
        raise FileNotFoundError(f"Could not find B03002 data for Brooklyn {unit}s. Checked {ACS_BG_DIR} and {ACS_TRACT_DIR}.")
    print(f"   B03002: {_label(p)}" + ("  (block groups added up to tracts)" if rolled else ""))
    _check_labels("B03002", meta_dirs + [ROOT / "B03002"])
    pop_s, white_s = numeric(df, "B03002", [1]), numeric(df, "B03002", [3])
    nd = at_unit(df, rolled, pop_s - white_s, pop_s)
    base = pd.DataFrame({"pop": nd["den"], "minority_pct": nd["num"] / nd["den"].where(nd["den"] > 0) * 100})
    used = ["minority"]
    for name, (table, num, den) in SPECS.items():
        df, p, rolled = load(table)
        if df is None:
            print(f"   {table}: not found at {unit} or block-group level, so '{name}' is skipped")
            continue
        print(f"   {table}: {_label(p)}" + ("  (block groups added up to tracts)" if rolled else ""))
        _check_labels(table, meta_dirs + [ROOT / table])
        try:
            nd = at_unit(df, rolled, numeric(df, table, num), numeric(df, table, den))
        except KeyError as e:
            print(f"   {name}: skipped ({e})")
            continue
        base[f"{name}_pct"] = nd["num"] / nd["den"].where(nd["den"] > 0) * 100
        used.append(name)
    base = base.reset_index().rename(columns={"index": "GEOID"})
    g = g.merge(base, on="GEOID", how="left")
    for name in WEIGHTS:
        if f"{name}_pct" not in g:
            g[f"{name}_pct"] = np.nan
    ok = g["pop"] >= MIN_POP
    for name in WEIGHTS:
        g.loc[~ok, f"{name}_pct"] = np.nan
    if len(used) < 2:
        raise ValueError("Need at least two equity indicators; only found: " + ", ".join(used))
    print("   Indicators used:", ", ".join(used))

    # Scale each indicator 0 to 1, then weight. Areas missing some indicators use the rest (needs 60% of the weight).
    w = pd.Series({k: WEIGHTS[k] for k in used})
    z = pd.DataFrame({k: (g[f"{k}_pct"] - g[f"{k}_pct"].min()) / (g[f"{k}_pct"].max() - g[f"{k}_pct"].min()) for k in used})
    have = z.notna().mul(w).sum(axis=1)
    g["equity_score"] = 100 * z.mul(w).sum(axis=1, skipna=True) / have.where(have > 0)
    g.loc[have < 0.6 * w.sum(), "equity_score"] = np.nan
    g["tier"] = np.nan
    valid = g["equity_score"].notna()
    if valid.sum() >= 3:
        g.loc[valid, "tier"] = pd.qcut(g.loc[valid, "equity_score"], 3, labels=[3, 2, 1]).astype(int)
    scored = int(valid.sum())
    print(f"   {scored:,} of {len(g):,} {unit}s have an equity score")
    if scored < 0.5 * len(g):
        print("   WARNING: fewer than half of the areas are scored. Check that the ACS files cover Kings County "
              f"at {unit} or block-group level and that GEOIDs match the shapefile.")

    g4 = g.to_crs(4326)
    pts = g4.geometry.representative_point()
    g["lon"], g["lat"] = pts.x.to_numpy(), pts.y.to_numpy()
    simp = g4.copy()
    simp["geometry"] = simp.geometry.simplify(0.00005)
    polys = []
    for r in simp.explode(index_parts=False).itertuples():
        if r.geometry is None or r.geometry.is_empty or r.geometry.geom_type != "Polygon":
            continue
        polys.append({"GEOID": r.GEOID, "polygon": [[round(x, 5), round(y, 5)] for x, y in r.geometry.exterior.coords]})

    PROC.mkdir(parents=True, exist_ok=True)
    cols = (["GEOID", "pop"] + [f"{k}_pct" for k in WEIGHTS] + ["equity_score", "tier", "lon", "lat"])
    g[cols].to_csv(PROC / "areas.csv", index=False)
    (PROC / "area_polys.json").write_text(json.dumps(polys))
    (PROC / "areas_meta.json").write_text(json.dumps({"unit": unit, "indicators": used, "built": str(pd.Timestamp.now().date())}))
    return g


def load_area_geometries():
    """Tract (or block-group) shapes with their need tier, read back from the files build_areas wrote."""
    import geopandas as gpd

    meta = json.loads((PROC / "areas_meta.json").read_text())
    shp = _find_shp(TRACT_DIR if meta["unit"] == "tract" else BG_DIR)
    g = gpd.read_file(shp)
    g["GEOID"] = g["GEOID"].astype(str)
    a = pd.read_csv(PROC / "areas.csv", dtype={"GEOID": str})
    return g.merge(a[["GEOID", "tier"]], on="GEOID", how="inner")


def pattern_outputs(parts, areas):
    """Everything that depends on which service runs on a date: service rows, the stops that run, how much of each
    area is near a stop, and the route shapes. `parts` come from _process_feed; `areas` from build_areas or
    load_area_geometries."""
    import geopandas as gpd

    svc = pd.concat([p["svc"] for p in parts], ignore_index=True)
    dup = int(svc.duplicated(KEYS).sum())
    if dup:
        print(f"   WARNING: {dup:,} stop-route rows appear in more than one feed (overlapping bundles). Keeping the first.")
        svc = svc.drop_duplicates(KEYS)
    stops = pd.concat([p["stops"] for p in parts], ignore_index=True).drop_duplicates("stop_id")
    rd = pd.concat([p["routedir"] for p in parts], ignore_index=True).drop_duplicates(["route_id", "direction_id"])
    wt = pd.concat([p["weekday_trips"] for p in parts], ignore_index=True).drop_duplicates(["route_id", "direction_id"])
    shapes = [s for p in parts for s in p["shapes"]]

    svc = svc.merge(rd[["route_id", "direction_id", "dir"]], on=["route_id", "direction_id"], how="left")
    svc["dir"] = svc["dir"].fillna("?")
    svc["in_borough"] = svc["route_id"].map(is_brooklyn)
    stops["lat"] = stops["stop_lat"].astype(float)
    stops["lon"] = stops["stop_lon"].astype(float)
    stops = stops[stops["stop_id"].isin(svc["stop_id"])].copy()

    a4 = gpd.GeoDataFrame(areas[["GEOID", "tier"]], geometry=areas.geometry, crs=areas.crs).to_crs(4326)
    pts = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops["lon"], stops["lat"]), crs=4326)
    j = gpd.sjoin(pts, a4[["GEOID", "tier", "geometry"]], how="left", predicate="within").drop_duplicates("stop_id")
    stops_out = j[["stop_id", "stop_name", "lat", "lon", "GEOID", "tier"]].rename(columns={"GEOID": "area"})

    # How much of each area lies within a quarter mile of any stop, of frequent service, of overnight service
    ab = areas.to_crs(CRS_FT)
    flags = svc.groupby("stop_id").agg(freq=("frequent", "any"), night=("night_trips", "sum")).reset_index()
    pts_ft = gpd.GeoDataFrame(stops.merge(flags, on="stop_id"),
                              geometry=gpd.points_from_xy(stops["lon"], stops["lat"]), crs=4326).to_crs(CRS_FT)
    cov = pd.DataFrame({
        "GEOID": ab["GEOID"].to_numpy(),
        "cover_any": _coverage(ab, pts_ft),
        "cover_freq": _coverage(ab, pts_ft[pts_ft["freq"]]),
        "cover_night": _coverage(ab, pts_ft[pts_ft["night"] > 0]),
    })

    rows, seen = [], set()
    for r, d, ln in shapes:
        if not is_brooklyn(r):
            continue
        k = (r, d, tuple(np.round(ln.coords[0], 4)), len(ln.coords))
        if k in seen:
            continue
        seen.add(k)
        rows.append({"route_id": r, "direction_id": d, "path": [[round(x, 5), round(y, 5)] for x, y in ln.coords]})
    return {"svc_all": svc, "svc": svc[svc["in_borough"]].copy(), "stops": stops_out, "cov": cov,
            "route_trips": wt[wt["route_id"].map(is_brooklyn)].copy(), "routedir": rd, "routes": rows}


def brooklyn_feeds(found):
    """The MTA archive holds every borough, and the express routes sit in other feeds. Keep each feed that has a listed route,
    and say which listed routes no feed has."""
    keep, have = [], set()
    for f in found:
        r = _csv(f, "routes.txt", {"route_id"})
        if r is None:
            continue
        ids = set(canon_series(r["route_id"]).dropna())
        if ids:
            keep.append(f)
            have |= ids
    missing = [n for n in ROUTE_NAMES if n not in have]
    print(f"   Feeds with listed routes: {', '.join(Path(f).name for f in keep) or 'none'}")
    print(f"   Listed routes found in the GTFS: {len(have)} of {len(ROUTE_NAMES)}"
          + (f". Not in any feed (add its GTFS folder, or it may not run in this period): {', '.join(missing)}" if missing else ""))
    return keep or list(found)


def build_run(areas):
    """Static outputs for the reference date (a fallback, plus the build log). The per-date files come from
    build_daily_schedule_parquet.py."""
    cfg = RUN
    out_dir = PROC / cfg["key"]
    root_used, found = discover_feeds(cfg["gtfs"])
    if not found:
        raise FileNotFoundError(f"No GTFS found. Looked in: {cfg['gtfs']}")
    chosen, matched, info = select_by_date(brooklyn_feeds(found), cfg["ref_date"])
    print(f"   GTFS folder: {root_used}  (reference date {cfg['ref_date'] or 'middle of the bundle'})")
    for f, r in info:
        span = "no calendar.txt" if r is None else f"{r[0].date()} to {r[1].date()}"
        print(f"     {'USE ' if f in chosen else 'skip'} {_label(f)}   service {span}")
    if not matched:
        print("   WARNING: no feed covers the reference date.")
    parts = []
    for f in chosen:
        try:
            parts.append(_process_feed(f, cfg["ref_date"]))
        except ValueError as e:
            print(f"   skipped: {e}")
    if not parts:
        raise ValueError("No feed has service on the reference date.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = pattern_outputs(parts, areas)
    out["svc_all"].to_csv(out_dir / "service.csv", index=False)
    out["stops"].to_csv(out_dir / "stops.csv", index=False)
    out["cov"].to_csv(out_dir / "cov.csv", index=False)
    out["route_trips"].to_csv(out_dir / "route_trips.csv", index=False)
    out["routedir"].to_csv(out_dir / "routedir.csv", index=False)
    (out_dir / "routes.json").write_text(json.dumps(out["routes"]))
    meta = {"feeds_used": [_label(f) for f in chosen], "gtfs_root": root_used, "ref_date": cfg["ref_date"],
            "ref_date_matched": matched,
            "service_sample_dates": [p["target"] for p in parts], "service_ranges": [p["range"] for p in parts],
            "built": str(pd.Timestamp.now().date())}
    (out_dir / "meta.json").write_text(json.dumps(meta))


# ---- Ridership ---------------------------------------------------------------
def build_ridership_monthly():
    """Citywide monthly file to one small Brooklyn file with a month column (YYYY-MM)."""
    want = {"route_id", "direction", "stop_id", "boardings", "alightings", "trips"}
    parts = []
    for ch in pd.read_csv(MONTHLY_FILE, dtype=str, chunksize=300_000):
        ch.columns = [c.strip().lower().replace(" ", "_") for c in ch.columns]
        ch = ch.rename(columns={"sum_boardings": "boardings", "sum_alightings": "alightings", "sum_trips": "trips"})
        if not want <= set(ch.columns):
            raise KeyError(f"{MONTHLY_FILE.name} is missing columns: {sorted(want - set(ch.columns))}")
        ch["route_id"] = canon_series(ch["route_id"])
        ch = ch[ch["route_id"].notna()].copy()
        ch["direction"] = ch["direction"].str.strip().str.upper()
        date_col = "by_month_date" if "by_month_date" in ch.columns else "date"
        ts = pd.to_datetime(ch[date_col], format="%Y %b %d %I:%M:%S %p", errors="coerce")
        if ts.isna().all():
            ts = pd.to_datetime(ch[date_col], errors="coerce")
        ch["month"] = ts.dt.strftime("%Y-%m")
        for c in ("boardings", "alightings", "trips"):
            ch[c] = pd.to_numeric(ch[c], errors="coerce").fillna(0)
        parts.append(ch.dropna(subset=["month"]).groupby(["route_id", "direction", "stop_id", "month"], as_index=False)[
            ["boardings", "alightings", "trips"]].sum())
    out = pd.concat(parts).groupby(["route_id", "direction", "stop_id", "month"], as_index=False)[
        ["boardings", "alightings", "trips"]].sum()
    (PROC / "ridership").mkdir(parents=True, exist_ok=True)
    out.to_csv(PROC / "ridership" / "monthly.csv", index=False)
    return out["month"].min(), out["month"].max(), len(out)


def build_ridership_folder(folder_rel: str):
    d = ROOT / folder_rel
    p = d / RIDERSHIP_FILE
    if not p.exists():
        csvs = sorted(d.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No ridership CSV in {d}")
        p = csvs[0]
    want = {"route_id", "direction", "stop_id", "boardings", "alightings", "trips"}
    parts = []
    for ch in pd.read_csv(p, usecols=lambda c: c.strip().lower() in want, dtype=str, chunksize=250_000):
        ch.columns = [c.strip().lower() for c in ch.columns]
        for c in ("boardings", "alightings", "trips"):
            ch[c] = pd.to_numeric(ch[c], errors="coerce").fillna(0)
        ch["route_id"] = canon_series(ch["route_id"])
        ch["direction"] = ch["direction"].str.strip().str.upper()
        ch = ch[ch["route_id"].notna()]
        parts.append(ch.groupby(["route_id", "direction", "stop_id"], as_index=False)[["boardings", "alightings", "trips"]].sum())
    out = pd.concat(parts).groupby(["route_id", "direction", "stop_id"], as_index=False)[["boardings", "alightings", "trips"]].sum()
    (PROC / "ridership").mkdir(parents=True, exist_ok=True)
    out.to_csv(PROC / "ridership" / f"{Path(folder_rel).name}.csv", index=False)


def build_all():
    print("Building the equity layer ...")
    areas = build_areas()
    print("Building the bus network ...")
    build_run(areas)
    if MONTHLY_FILE.exists():
        print(f"Aggregating monthly ridership: {MONTHLY_FILE.name}")
        lo, hi, n = build_ridership_monthly()
        print(f"   {n:,} rows, months {lo} to {hi}")
    else:
        print("Monthly ridership file not found; using the per-period folder.")
        build_ridership_folder(RUN["folder"])
    print("Done. Files are in", PROC)


# =============================================================================
# APP
# =============================================================================
def _stamp(p: Path) -> float:
    """File modified time. Passing it to the cached loaders makes them reload after a rebuild."""
    return p.stat().st_mtime if p.exists() else 0.0


def _listed_shapes(shapes):
    """Route shapes for the listed routes only, under their listed names."""
    out = []
    for p in shapes:
        name = canon_route(p["route_id"])
        if name:
            out.append({**p, "route_id": name})
    return out


@st.cache_data(show_spinner=False)
def load_areas(stamp: float):
    a = pd.read_csv(PROC / "areas.csv", dtype={"GEOID": str})
    polys = json.loads((PROC / "area_polys.json").read_text())
    meta = json.loads((PROC / "areas_meta.json").read_text())
    return a, polys, meta


@st.cache_data(show_spinner=False)
def load_run(stamp: float):
    d = PROC / RUN["key"]
    ids = {"stop_id": str, "route_id": str, "GEOID": str, "area": str, "direction_id": str}
    out = {n: pd.read_csv(d / f"{n}.csv", dtype=ids) for n in ("service", "stops", "cov", "route_trips", "routedir")}
    for n in ("service", "route_trips", "routedir"):
        out[n] = canon_frame(out[n])
    out["routes"] = _listed_shapes(json.loads((d / "routes.json").read_text()))
    out["routes_enc"] = [polyline_encode(p["path"]) for p in out["routes"]]
    out["meta"] = json.loads((d / "meta.json").read_text())
    return out


@st.cache_data(show_spinner=False)
def load_polys_enc(stamp: float):
    """Tract outlines as encoded strings, built once."""
    polys = json.loads((PROC / "area_polys.json").read_text())
    return [p["GEOID"] for p in polys], [polyline_encode(p["polygon"]) for p in polys]


@st.cache_data(show_spinner=False)
def load_ridership(name: str, stamp: float):
    return canon_frame(pd.read_csv(PROC / "ridership" / f"{name}.csv", dtype={"route_id": str, "direction": str, "stop_id": str}))


@st.cache_data(show_spinner=False)
def load_daily_ridership(path: str, stamp: float):
    return canon_frame(pd.read_parquet(path))


@st.cache_data(show_spinner=False)
def load_manifest(stamp: float):
    p = SCHEDULE_DIR / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(show_spinner=False)
def load_pattern(folder: str, stamp: float):
    """The service files for one pattern of service (for example 'weekday' or 'Saturday')."""
    d = SCHEDULE_DIR / folder
    routes = _listed_shapes(json.loads((d / "routes.json").read_text()))
    return {"service": canon_frame(pd.read_parquet(d / "service.parquet")), "route_trips": canon_frame(pd.read_parquet(d / "route_trips.parquet")),
            "routedir": canon_frame(pd.read_parquet(d / "routedir.parquet")), "stops": pd.read_parquet(d / "stops.parquet"),
            "cov": pd.read_parquet(d / "cov.parquet"), "routes": routes, "routes_enc": [polyline_encode(p["path"]) for p in routes]}


@st.cache_data(show_spinner=False)
def load_universe(s_stamp: float, r_stamp: float):
    """Every route, direction and stop seen on any date of the month, so filter options do not change with the date.
    Uses the index files when present, else scans the month's files."""
    sp = SCHEDULE_DIR / "universe.parquet"
    rp = ROOT / "03_stop_level_ridership" / "daily_parquet" / "universe.parquet"
    if sp.exists():
        s_ = pd.read_parquet(sp)
    else:
        frames = []
        for d in sorted(SCHEDULE_DIR.glob("pattern_*")):
            if (d / "service.parquet").exists() and (d / "stops.parquet").exists():
                svc = pd.read_parquet(d / "service.parquet")[["route_id", "dir", "stop_id"]].drop_duplicates()
                frames.append(svc.merge(pd.read_parquet(d / "stops.parquet")[["stop_id", "stop_name", "lat", "lon"]], on="stop_id", how="left"))
        s_ = (pd.concat(frames, ignore_index=True).drop_duplicates(["route_id", "dir", "stop_id"]) if frames
              else pd.DataFrame(columns=["route_id", "dir", "stop_id", "stop_name", "lat", "lon"]))
    if rp.exists():
        r_ = pd.read_parquet(rp)
    else:
        frames = [pd.read_parquet(f)[["route_id", "direction", "stop_id"]].drop_duplicates()
                  for f in sorted((ROOT / "03_stop_level_ridership" / "daily_parquet").glob("*/ridership.parquet"))]
        r_ = (pd.concat(frames, ignore_index=True).drop_duplicates() if frames
              else pd.DataFrame(columns=["route_id", "direction", "stop_id"]))
    return canon_frame(s_), canon_frame(r_)


def sticky_selectbox(label, options, name, format_func=str):
    """A dropdown that keeps its value when its option list changes, on any Streamlit version.
    Older Streamlit versions treat a dropdown whose options changed as a new widget and reset it, which is what happens
    when the date changes the available routes or stops. This one remembers its own value and feeds it back as the default."""
    opts = list(options)
    saved = st.session_state.get(f"saved_{name}", opts[0])
    if saved not in opts:
        saved = opts[0]
    value = st.selectbox(label, opts, index=opts.index(saved), format_func=format_func)
    st.session_state[f"saved_{name}"] = value
    return value


def sidebar_label(text):
    st.markdown(f"<div class='side-h'>{esc(text)}</div>", unsafe_allow_html=True)


def pills(items):
    """Small labelled chips. items is a list of (label, value) pairs; empty values are skipped."""
    return "".join(f"<span class='pl'><i>{esc(str(k))}</i> {esc(str(v))}</span>" for k, v in items if v not in (None, ""))


def row_html(rank, title, lines, value, value_sub="", bar=None, hint="", height=48, wrap_title=False, chips="", pad=6):
    """One ranked row: rank badge, name, detail lines and chips, a figure on the right, and an optional meter.
    The row is `height` pixels tall with `pad` pixels of breathing space around it, so a list can be spread to fill its panel."""
    def _fit(text, base, limit):
        n = len(unescape(re.sub(r"<[^>]+>", "", str(text))))
        return base if n <= limit else max(base * 0.72, round(base * limit / n, 1))
    meter = f"<i class='mt'><s style='width:{max(0.0, min(100.0, bar)):.0f}%'></s></i>" if bar is not None else ""
    body = "".join(f"<div class='ls' style='font-size:{_fit(ln, 11, 36)}px'>{ln}</div>" for ln in lines)
    cls = "lt wrap" if wrap_title else "lt"
    half = max(0, int(pad // 2))
    return (f"<div class='lw' style='padding:{half}px 0'><div class='lrow' style='height:{height}px' title='{esc(hint, quote=True)}'>"
            f"<span class='rk'>{rank}</span><div class='lm'><div class='{cls}' style='font-size:{_fit(title, 13, 24)}px'>{title}</div>{body}<div class='pls'>{chips}</div></div>"
            f"<div class='lv'><b>{value}</b><span>{value_sub}</span>{meter}</div></div></div>")


def bounds_of(coords):
    """[[south, west], [north, east]] for a list of [lon, lat] pairs."""
    arr = np.asarray(coords, dtype=float)
    return [[float(arr[:, 1].min()), float(arr[:, 0].min())], [float(arr[:, 1].max()), float(arr[:, 0].max())]]


def _hex(rgba):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgba[:3]), (rgba[3] / 255 if len(rgba) > 3 else 1.0)


_MAP = components.declare_component("brooklyn_map", path=str(COMPONENT_DIR)) if COMPONENT_DIR.exists() else None


def _enc_num(n, out):
    n = ~(n << 1) if n < 0 else (n << 1)
    while n >= 0x20:
        out.append(chr((0x20 | (n & 0x1F)) + 63))
        n >>= 5
    out.append(chr(n + 63))


def polyline_encode(points):
    """[[lon, lat], ...] as one short string (Google's encoded polyline, 5 decimals). The browser decodes it."""
    out, plat, plon = [], 0, 0
    for lon, lat in points:
        la, lo = round(lat * 1e5), round(lon * 1e5)
        _enc_num(la - plat, out)
        _enc_num(lo - plon, out)
        plat, plon = la, lo
    return "".join(out)


# Buttons in the lists change the selection in a callback, so the app runs once per click instead of twice.
def _bump_view():
    st.session_state["view_nonce"] = st.session_state.get("view_nonce", 0) + 1


def _zoom_tract(geoid, lat, lon):
    st.session_state["map_focus"] = {"type": "tract", "geoid": geoid, "lat": lat, "lon": lon}
    _bump_view()


def _go_stop(route, sid, lat, lon):
    st.session_state["pending_filter"] = {"route": route, "direction": "All directions", "stop": sid,
                                          "focus": {"type": "stop", "lat": lat, "lon": lon}}
    _bump_view()


def _go_route(route):
    st.session_state["pending_filter"] = {"route": route, "direction": "All directions", "stop": "All stops",
                                          "focus": {"type": "route", "route_id": route}}
    _bump_view()


def _reset_filters():
    st.session_state.update(saved_route="All routes", saved_direction="All directions", saved_stop="All stops", map_focus=None)
    _bump_view()


def show_df(df: pd.DataFrame):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True)


def month_label(m: str) -> str:
    return pd.Timestamp(m + "-01").strftime("%b %Y")


def _xy(lon, lat):
    """Local flat-earth meters for NYC. Accurate to a fraction of a percent at this scale."""
    lat0 = math.radians(40.65)
    return np.column_stack([np.asarray(lon) * math.cos(lat0) * 111320.0, np.asarray(lat) * 110540.0])


def nearest_m(centers_lon, centers_lat, stops_lon, stops_lat):
    """Straight-line distance in meters from each center to its nearest stop. NaN when there are no stops."""
    if len(stops_lon) == 0:
        return np.full(len(centers_lon), np.nan)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(_xy(stops_lon, stops_lat)).query(_xy(centers_lon, centers_lat))
    return d


def fmt_dist(m) -> str:
    if pd.isna(m):
        return "n/a"
    ft = m * 3.28084
    return f"{ft:,.0f} ft" if ft < 528 else f"{m / 1609.34:.2f} mi"


def ramp_color(score, lo, hi):
    if pd.isna(score):
        return NO_DATA
    t = min(max((score - lo) / (hi - lo), 0.0), 1.0) if hi > lo else 0.5
    return [int(round(RAMP_LO[i] + (RAMP_HI[i] - RAMP_LO[i]) * t)) for i in range(3)] + [195]


def view_for(a: pd.DataFrame):
    lon0, lon1 = a["lon"].min(), a["lon"].max()
    lat0, lat1 = a["lat"].min(), a["lat"].max()
    span = max(lon1 - lon0, (lat1 - lat0) * 1.3, 0.02)
    zoom = math.log2(360 * 900 / 512 / span) - 0.2
    return (lat0 + lat1) / 2, (lon0 + lon1) / 2, float(min(max(zoom, 8), 13))


def main():
    st.set_page_config(page_title="Brooklyn Bus Equity Explorer", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    /* page frame: tight padding so the whole view fits one small screen. The toolbar stays, because it holds the running indicator;
       only its menu and deploy button are hidden. */
    header[data-testid="stHeader"]{background:transparent;height:2rem}
    #MainMenu,[data-testid="stMainMenu"],.stDeployButton,[data-testid="stAppDeployButton"],div[data-testid="stDecoration"],footer,
    [data-testid="stToolbarActions"],[data-testid="manage-app-button"],[class*="viewerBadge"],[class*="_profileContainer"]{display:none !important}
    [data-testid="stMainBlockContainer"],.block-container{padding:2.1rem .9rem .3rem .9rem !important;max-width:100% !important}
    /* Streamlit pulls the element after a markdown block up by 1rem to tidy paragraph spacing. Our blocks are custom HTML,
       so that pull made the banner, the list descriptions and the sidebar headings overlap what follows them. */
    [data-testid="stMarkdownContainer"]{margin-bottom:0 !important}
    div[data-testid="stVerticalBlock"]{gap:6px}
    /* centre the content of each list row and the banner, but never the outer map/list row (that pushed the map down by half
       the height difference with the lists, so the map moved over the KPIs whenever the lists got shorter) */
    div[data-testid="stHorizontalBlock"]:has(.lrow):not(:has(div[data-testid="stHorizontalBlock"])),
    div[data-testid="stHorizontalBlock"]:has(.banner){align-items:center}
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stHorizontalBlock"]){align-items:flex-start}
    /* sidebar: category titles, then control titles, are larger than the values inside the controls */
    section[data-testid="stSidebar"]{width:260px !important;min-width:260px !important;background:#eef3f9}
    /* sidebar content starts at the same height as the title banner (the page padding above is 2.1rem) */
    section[data-testid="stSidebar"] [data-testid="stSidebarHeader"]{height:2.1rem !important;min-height:2.1rem !important;padding:0 .5rem !important}
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding:0 .85rem .6rem .85rem !important}
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]{gap:8px}
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"]{min-height:0;margin-bottom:2px}
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{font-size:14px !important;font-weight:700 !important;color:#16335c !important}
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] div[data-baseweb="input"] > div{min-height:34px}
    section[data-testid="stSidebar"] div[data-baseweb="select"] div,
    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] input{font-size:12.5px !important;font-weight:400 !important}
    ul[role="listbox"] li{font-size:12.5px !important}
    section[data-testid="stSidebar"] [data-testid="stCheckbox"] p{font-size:13.5px !important}
    .side-h{font-size:16px;font-weight:800;color:#0039A6;margin:14px 0 12px;padding-bottom:4px;border-bottom:2px solid #c9d7ee}
    .side-h.first{margin-top:2px}
    /* styled title banner */
    .banner{display:flex;align-items:center;gap:12px;background:linear-gradient(100deg,#0039A6 0%,#0b4a95 55%,#0f5257 100%);
            color:#fff;border-radius:10px;padding:6px 14px;min-height:46px;box-shadow:0 2px 8px rgba(0,57,166,.22);margin-bottom:8px}
    .banner .logo{width:34px;height:34px;border-radius:9px;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.32);
                  display:flex;align-items:center;justify-content:center;flex:0 0 auto}
    .banner .b1{font-size:19px;font-weight:800;letter-spacing:-.3px;line-height:1.1}
    .banner .b1 span{font-weight:500;opacity:.92}
    .banner .b2{font-size:11px;opacity:.82;margin-top:1px}
    .banner .bchips{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
    .banner .bchip{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.3);border-radius:999px;padding:2px 10px;
                   font-size:12px;font-weight:600;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .banner .bchip.day{background:#fff;color:#0039A6;border-color:#fff}
    div[data-testid="stPopover"] button{height:38px;border-radius:9px;border:1px solid #c8d4e3;background:#fff;color:#0039A6;font-weight:700}
    /* KPI strip: single values and stacked pairs */
    .kpis{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:2px}
    .kpi{flex:1 1 120px;min-width:112px;background:#fff;border:1px solid #d8e0ec;border-top:3px solid var(--c,#0039A6);
         border-radius:8px;padding:3px 10px 4px;display:flex;flex-direction:column;justify-content:center}
    .kpi.stack{flex:1.15 1 140px}
    .kpi .kt{font-size:10.5px;font-weight:700;color:#5b6b7f;margin-bottom:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .kpi .kv{font-size:18px;line-height:1.1;font-weight:700;color:#132238}
    .kpi .kr{display:flex;justify-content:space-between;align-items:baseline;gap:8px;line-height:1.3}
    .kpi .kr span{font-size:11px;color:#5b6b7f}
    .kpi .kr b{font-size:13.5px;color:#132238;font-weight:700}
    /* top lists: row sizes and spacing are set in Python to fill the panel */
    div[data-baseweb="tab-list"]{gap:0}
    button[data-baseweb="tab"]{padding:2px 10px;height:32px;font-size:13.5px;font-weight:600}
    div[data-baseweb="tab-panel"]{padding-top:.4rem}
    .cap{font-size:11px;color:#5b6b7f;margin:0 0 6px;line-height:1.35}
    .lw{box-sizing:content-box}
    .lrow{display:grid;grid-template-columns:22px minmax(0,1fr) 98px;gap:8px;align-items:center;background:#fff;
          border:1px solid #e1e8f2;border-radius:8px;padding:3px 9px;overflow:hidden;box-sizing:border-box}
    .lrow .rk{width:21px;height:21px;border-radius:50%;background:#0039A6;color:#fff;font-size:11px;font-weight:700;
              display:flex;align-items:center;justify-content:center}
    .lrow .lm{min-width:0;overflow:visible}
    .lrow .lt{font-size:13px;font-weight:700;color:#132238;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.2}
    .lrow .lt.wrap{white-space:normal;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;line-height:1.2}
    .lrow .ls{font-size:11px;color:#4f6075;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.35}
    .lrow .pls{display:flex;flex-wrap:nowrap;gap:3px;margin-top:2px;overflow:hidden}
    .lrow .pl{background:#eef3fb;color:#26456e;border-radius:999px;padding:0 7px;font-size:10.5px;line-height:1.55;white-space:nowrap}
    .lrow .pl i{font-style:normal;color:#6b7f99}
    .lrow .lv{text-align:right;line-height:1.15}
    .lrow .lv b{display:block;font-size:16px;color:#0039A6}
    .lrow .lv span{display:block;font-size:10px;color:#5b6b7f;white-space:nowrap}
    .lrow .mt{display:block;height:5px;background:#e3eaf5;border-radius:3px;margin-top:4px}
    .lrow .mt s{display:block;height:5px;background:#0039A6;border-radius:3px;text-decoration:none}
    .about{background:#f6f8fb;border:1px dashed #b9c6d8;border-radius:8px;padding:8px 10px;font-size:11px;color:#4f6075;line-height:1.45;margin-top:4px}
    .stButton > button{min-height:28px;height:28px;padding:0 6px;font-size:13px;line-height:1;border-radius:7px;border:1px solid #c8d4e3}
    section[data-testid="stSidebar"] .stButton > button{width:100%}
    /* the small zoom arrows beside each list row */
    div[data-testid="stColumn"]{min-width:0 !important}
    div[data-testid="stColumn"] .stButton > button{min-height:24px;height:24px;width:100%;min-width:0;padding:0;font-size:12px}
    section[data-testid="stSidebar"] div[data-testid="stColumn"] .stButton > button{min-height:28px;height:28px;font-size:13px}
    </style>
    """, unsafe_allow_html=True)

    run_meta = PROC / RUN["key"] / "meta.json"
    if not (PROC / "areas.csv").exists() or not run_meta.exists():
        st.error("No processed data found.")
        st.write("Run this once in a terminal, from the project folder, then reload the page:")
        st.code("python app1.py --build")
        st.stop()

    areas, polys, ameta = load_areas(_stamp(PROC / "areas.csv"))
    poly_ids, polys_enc = load_polys_enc(_stamp(PROC / "area_polys.json"))
    if _MAP is None:
        st.error("The map component folder is missing. Put the components/brooklyn_map folder next to app1.py.")
        st.stop()
    unit = ameta["unit"]
    U = unit.capitalize()
    R = load_run(_stamp(run_meta))
    meta = R["meta"]

    # ---- The date: ridership for that date, and the schedule that ran on it -------------------
    daily_root = ROOT / "03_stop_level_ridership" / "daily_parquet"
    day_files = sorted(daily_root.glob("*/ridership.parquet"))
    selected_day = None
    rid_cols = ["route_id", "direction", "stop_id", "boardings", "alightings", "trips"]
    with st.sidebar:
        st.markdown("<div class='side-h first'>Date</div>", unsafe_allow_html=True)
        if day_files:
            days = [pd.Timestamp(p.parent.name).date() for p in day_files]
            default_day = pd.Timestamp(DEFAULT_DATE).date()
            if not (days[0] <= default_day <= days[-1]):
                default_day = days[0]
            st.session_state.setdefault("ridership_date", default_day)
            if not (days[0] <= st.session_state["ridership_date"] <= days[-1]):
                st.session_state["ridership_date"] = default_day
            selected_day = st.date_input("Date", min_value=days[0], max_value=days[-1], key="ridership_date", label_visibility="collapsed")
            path = daily_root / str(selected_day) / "ridership.parquet"
            rid = load_daily_ridership(str(path), _stamp(path)) if path.exists() else pd.DataFrame(columns=rid_cols)
            stamp = pd.Timestamp(selected_day)
            # %-d is not available on Windows, so the day is formatted as an integer
            period = f"{stamp.strftime('%A, %B')} {stamp.day}, {stamp.year}"
        else:
            monthly_path = PROC / "ridership" / "monthly.csv"
            if not monthly_path.exists():
                st.warning("Daily ridership files are missing. Run: python build_daily_ridership_parquet.py")
                st.stop()
            rid = load_ridership("monthly", _stamp(monthly_path))
            period = RUN["period"]
            st.write(period)
    rid = rid[rid["route_id"].map(is_brooklyn)]

    manifest = load_manifest(_stamp(SCHEDULE_DIR / "manifest.json"))
    pattern = manifest.get("dates", {}).get(str(selected_day)) if selected_day else None
    if pattern:
        S = load_pattern(pattern, _stamp(SCHEDULE_DIR / pattern / "service.parquet"))
        svc_all, day_stops, cov_day, routes_day = S["service"], S["stops"], S["cov"], S["routes"]
        routes_enc_day = S["routes_enc"]
        route_trips_day, routedir_day = S["route_trips"], S["routedir"]
    else:
        svc_all = R["service"][R["service"]["in_borough"]]
        day_stops, cov_day, routes_day = R["stops"], R["cov"], R["routes"]
        routes_enc_day = R["routes_enc"]
        route_trips_day, routedir_day = R["route_trips"], R["routedir"]
        if selected_day:
            st.warning("No schedule has been built for this date, so the reference-date schedule is shown. "
                       "Run: python build_daily_schedule_parquet.py")
    stops_tbl = day_stops.set_index("stop_id")
    # the need tier of the tract each stop sits in (the reference-date table fills in stops that do not run on the selected date)
    tier_of = (pd.concat([R["stops"][["stop_id", "tier"]], day_stops[["stop_id", "tier"]]])
               .drop_duplicates("stop_id", keep="last").set_index("stop_id")["tier"])

    # ---- Filter options come from the whole month, so a selection survives a change of date ------
    uni_s, uni_r = load_universe(_stamp(SCHEDULE_DIR / "universe.parquet"), _stamp(daily_root / "universe.parquet"))
    name_of = {}
    if "stop_name" in uni_s:
        name_of.update({k: v for k, v in zip(uni_s["stop_id"], uni_s["stop_name"]) if isinstance(v, str)})
    if "stop_name" in day_stops:
        name_of.update(dict(zip(day_stops["stop_id"], day_stops["stop_name"])))
    # always include what exists on the selected date, so the lists are never smaller than that day's data
    uni_s = pd.concat([uni_s[["route_id", "dir", "stop_id"]], svc_all[["route_id", "dir", "stop_id"]]], ignore_index=True).drop_duplicates()
    uni_r = pd.concat([uni_r[["route_id", "direction", "stop_id"]], rid[["route_id", "direction", "stop_id"]]], ignore_index=True).drop_duplicates()

    rid_routes, svc_routes = set(rid["route_id"]), set(svc_all["route_id"])

    def notes(has_rid, has_svc):
        n = ([] if has_rid else ["no ridership data available"]) + ([] if has_svc else ["no scheduled service"])
        return f" ({'; '.join(n)})" if n else ""

    pending = st.session_state.pop("pending_filter", None)
    if pending:
        st.session_state["saved_route"] = pending.get("route", "All routes")
        st.session_state["saved_direction"] = pending.get("direction", "All directions")
        st.session_state["saved_stop"] = pending.get("stop", "All stops")
        st.session_state["map_focus"] = pending.get("focus")
    with st.sidebar:
        sidebar_label("Filters")
        route_universe = sorted(set(uni_s["route_id"]) | set(uni_r["route_id"]))
        route = sticky_selectbox("Route", ["All routes"] + route_universe, "route",
                                 format_func=lambda r: r if r == "All routes" else r + notes(r in rid_routes, r in svc_routes))
        if st.session_state.get("prev_route") not in (None, route) and not pending:
            st.session_state["saved_direction"] = "All directions"      # a different route starts from all directions and stops
            st.session_state["saved_stop"] = "All stops"
        st.session_state["prev_route"] = route
        direction = "All directions"
        if route != "All routes":                                  # direction only exists once a route is chosen
            # Ridership direction codes lead. The schedule's compass directions are only used for a route with no ridership.
            dirs = sorted(set(uni_r.loc[uni_r["route_id"] == route, "direction"].dropna().astype(str)))
            if not dirs:
                dirs = sorted(set(uni_s.loc[uni_s["route_id"] == route, "dir"].dropna().astype(str)) - {"?"})
            rid_dirs = set(zip(rid["route_id"], rid["direction"]))
            direction = sticky_selectbox("Direction", ["All directions"] + dirs, "direction",
                                         format_func=lambda d: d if d == "All directions" else d + notes((route, d) in rid_dirs, route in svc_routes))
        else:
            st.session_state["saved_direction"] = "All directions"
        r_u = uni_r if route == "All routes" else uni_r[uni_r["route_id"] == route]
        s_u = uni_s if route == "All routes" else uni_s[uni_s["route_id"] == route]
        if direction != "All directions":
            r_u, s_u = r_u[r_u["direction"] == direction], s_u[s_u["dir"] == direction]
        stop_ids = sorted(set(r_u["stop_id"]) | set(s_u["stop_id"]), key=lambda s: (str(name_of.get(s, "~")), s))
        rid_today = rid if route == "All routes" else rid[rid["route_id"] == route]
        if direction != "All directions":
            rid_today = rid_today[rid_today["direction"] == direction]
        svc_today = svc_all if route == "All routes" else svc_all[svc_all["route_id"] == route]
        rid_stops, svc_stops = set(rid_today["stop_id"]), set(svc_today["stop_id"])
        stop = sticky_selectbox("Stop", ["All stops"] + stop_ids, "stop",
                                format_func=lambda s: s if s == "All stops"
                                else f"{name_of.get(s, 'Unknown stop')} ({s})" + notes(s in rid_stops, s in svc_stops))
        st.button("Reset filters", key="reset_filters", help="Back to all routes and all stops", on_click=_reset_filters)
        sidebar_label("Layers")
        show_basemap = st.checkbox("Light basemap", True, key="l_basemap")
        basemap = "Light" if show_basemap else "None"
        show_areas = st.checkbox("Tracts (equity score)", True, key="l_areas")
        show_routes = st.checkbox("Routes", True, key="l_routes")
        show_stops = st.checkbox("Stops", True, key="l_stops")

    # ---- The selection, as ridership rows and schedule rows ------------------------------
    r_rid = rid if route == "All routes" else rid[rid["route_id"] == route]
    svc_rd = svc_all if route == "All routes" else svc_all[svc_all["route_id"] == route]
    dir_note = ""
    if direction != "All directions":
        r_rid = r_rid[r_rid["direction"] == direction]
        by_dir = svc_rd[svc_rd["dir"] == direction]
        if by_dir.empty and not svc_rd.empty:
            dir_note = ("The schedule's direction could not be matched to the ridership direction for this route, "
                        "so wait and trip figures cover both directions.")
        else:
            svc_rd = by_dir
    rid_rd = r_rid                                                  # route and direction, any stop
    svc_sel = svc_rd if stop == "All stops" else svc_rd[svc_rd["stop_id"] == stop]
    rid_sel = rid_rd if stop == "All stops" else rid_rd[rid_rd["stop_id"] == stop]

    # scheduled trips on the selected date, respecting the direction filter
    rt_full = route_trips_day.merge(routedir_day[["route_id", "direction_id", "dir"]], on=["route_id", "direction_id"], how="left")
    rt_full = rt_full[rt_full["route_id"].map(is_brooklyn)]
    rt = rt_full
    if route != "All routes":
        rt = rt[rt["route_id"] == route]
    if direction != "All directions" and not rt[rt["dir"] == direction].empty:
        rt = rt[rt["dir"] == direction]
    scheduled_trips = int(svc_sel["all_trips"].sum()) if stop != "All stops" else int(rt["weekday_trips"].sum())

    # ---- Tracts: nearest selected stop, residents nearby ---------------------------------
    sel_stops = stops_tbl.reindex(sorted(set(svc_sel["stop_id"]))).dropna(subset=["lat"])
    t = areas.merge(cov_day, on="GEOID", how="left")
    t["near_m"] = nearest_m(t["lon"].to_numpy(), t["lat"].to_numpy(), sel_stops["lon"].to_numpy(), sel_stops["lat"].to_numpy())
    near = t[t["near_m"] <= WALK_M]
    near_pop = float(near["pop"].sum())
    near_score = (np.average(near["equity_score"].dropna(), weights=near.loc[near["equity_score"].notna(), "pop"].clip(lower=1))
                  if near["equity_score"].notna().any() else np.nan)

    # ---------- values shared by the title bar, the KPIs, the map and the lists ----------
    scored = areas["equity_score"].dropna()
    lo_s, hi_s = (float(scored.quantile(0.02)), float(scored.quantile(0.98))) if len(scored) else (0.0, 100.0)
    scope = "selected stops" if (route != "All routes" or stop != "All stops") else "bus stop"

    # tract layer: outlines are encoded once and cached, so only short text is rebuilt when the selection changes
    focus = st.session_state.get("map_focus")
    t["fill"] = [ramp_color(sc_, lo_s, hi_s) for sc_ in t["equity_score"]]
    fills = [_hex(c) for c in t["fill"]]
    tr = pd.DataFrame({
        "GEOID": t["GEOID"].to_numpy(),
        "score": ["n/a" if pd.isna(v) else f"{v:.1f}" for v in t["equity_score"]],
        "pop": [f"{v:,.0f}" for v in t["pop"]],
        "minority": [_pct(v) for v in t["minority_pct"]], "poverty": [_pct(v) for v in t["poverty_pct"]],
        "near": [fmt_dist(v) for v in t["near_m"]], "cover": [_pct(v * 100) for v in t["cover_any"]],
        "fill": [f[0] for f in fills], "fop": [round(f[1], 3) for f in fills]})
    pdf = pd.DataFrame({"GEOID": poly_ids, "enc": polys_enc}).merge(tr, on="GEOID")
    fg = focus.get("geoid") if focus else None
    tract_rows = [[g, enc, fill, fop, "#ffd400" if g == fg else "#ffffff", 1.0 if g == fg else 0.47, 4 if g == fg else 1,
                   sc_, po, mi, pv, ne, co]
                  for g, enc, fill, fop, sc_, po, mi, pv, ne, co in zip(
                      pdf["GEOID"], pdf["enc"], pdf["fill"], pdf["fop"], pdf["score"], pdf["pop"], pdf["minority"],
                      pdf["poverty"], pdf["near"], pdf["cover"])]

    # route layer
    rb = rid_rd.groupby("route_id")[["boardings", "alightings"]].sum()
    rw_am = svc_rd.groupby("route_id")["am_wait_min"].median()
    rw_pm = svc_rd.groupby("route_id")["pm_wait_min"].median()
    rs = svc_rd.groupby("route_id")["stop_id"].nunique()
    rtrips = rt.groupby("route_id")["weekday_trips"].sum()
    shown_routes = set(svc_sel["route_id"])
    dir_of = {(r.route_id, r.direction_id): r.dir for r in routedir_day.itertuples()}
    dir_filter = direction != "All directions" and bool(rt["dir"].eq(direction).any())
    route_rows, route_paths = [], []
    for p, enc in zip(routes_day, routes_enc_day):
        rid_ = p["route_id"]
        if rid_ not in shown_routes:
            continue
        if dir_filter and dir_of.get((rid_, p["direction_id"])) != direction:
            continue
        route_rows.append([rid_, enc, int(rb["boardings"].get(rid_, 0)), int(rb["alightings"].get(rid_, 0)), int(rtrips.get(rid_, 0)),
                           int(rs.get(rid_, 0)), _wait(rw_am.get(rid_)), _wait(rw_pm.get(rid_))])
        route_paths.append(p["path"])

    # stop layer: the text for every stop is built in a few table operations, not one query per stop
    sp = sel_stops.rename_axis("stop_id").reset_index().copy()
    sb = rid_sel.groupby("stop_id")[["boardings", "alightings"]].sum()
    sp["boardings"] = sp["stop_id"].map(sb["boardings"]).fillna(0)
    sp["alightings"] = sp["stop_id"].map(sb["alightings"]).fillna(0)
    routes_at = (svc_sel[["stop_id", "route_id"]].drop_duplicates().sort_values(["stop_id", "route_id"])
                 .groupby("stop_id")["route_id"].agg(", ".join))
    w = svc_sel.groupby(["stop_id", "route_id"])[["am_wait_min", "pm_wait_min"]].min()
    w = w[w.notna().any(axis=1)].reset_index()
    w["txt"] = w["route_id"] + " " + w["am_wait_min"].map(_wait) + " AM, " + w["pm_wait_min"].map(_wait) + " PM"
    first4 = w[w.groupby("stop_id").cumcount() < 4].groupby("stop_id")["txt"].agg("; ".join)
    n_routes = w.groupby("stop_id").size()
    waits_at = {sid_: txt + (f"; +{n_routes[sid_] - 4} more" if n_routes[sid_] > 4 else "") for sid_, txt in first4.items()}
    top = max(float(sp["boardings"].max()), 1.0) if len(sp) else 1.0
    sp["routes"] = sp["stop_id"].map(routes_at).fillna("n/a")
    sp["waits"] = sp["stop_id"].map(waits_at).fillna("n/a")
    sp["px"] = 3 + 5 * np.sqrt(sp["boardings"] / top)
    stop_rows = [[round(float(lo_), 5), round(float(la_), 5), round(float(px_), 2), nm_, sid_, int(b_), int(a_), rts_, wts_]
                 for lo_, la_, px_, nm_, sid_, b_, a_, rts_, wts_ in zip(
                     sp["lon"], sp["lat"], sp["px"], sp["stop_name"], sp["stop_id"], sp["boardings"], sp["alightings"],
                     sp["routes"], sp["waits"])]

    # Where to look: the most specific active selection, else all of Brooklyn
    if stop != "All stops" and stop in stops_tbl.index:
        q = stops_tbl.loc[stop]
        view = {"center": [float(q["lat"]), float(q["lon"])], "zoom": 16}
    elif route != "All routes" and route_paths:
        view = {"bounds": bounds_of([xy for path_ in route_paths for xy in path_])}
    elif focus and focus.get("type") == "tract":
        view = {"center": [focus["lat"], focus["lon"]], "zoom": 15}
    elif focus and focus.get("type") == "stop":
        view = {"center": [focus["lat"], focus["lon"]], "zoom": 16}
    elif focus and focus.get("type") == "route" and any(p["route_id"] == focus.get("route_id") for p in routes_day):
        view = {"bounds": bounds_of([xy for p in routes_day if p["route_id"] == focus.get("route_id") for xy in p["path"]])}
    else:
        view = {"bounds": [[float(areas["lat"].min()), float(areas["lon"].min())], [float(areas["lat"].max()), float(areas["lon"].max())]]}

    # ---------- title bar: name, the date and filters in use, and the method note ----------
    chips = [f"<span class='bchip day'>{esc(period)}</span>"]
    if route != "All routes":
        chips.append(f"<span class='bchip'>Route {esc(route)}</span>")
    if direction != "All directions":
        chips.append(f"<span class='bchip'>Direction {esc(direction)}</span>")
    if stop != "All stops":
        chips.append(f"<span class='bchip'>{esc(str(name_of.get(stop, 'Stop')))}</span>")
    bus_icon = ("<svg viewBox='0 0 24 24' width='22' height='22' fill='none' stroke='#fff' stroke-width='1.8' stroke-linecap='round' "
                "stroke-linejoin='round'><rect x='4' y='3' width='16' height='14' rx='3'/><path d='M4 11h16'/><path d='M8 7h8' opacity='.7'/>"
                "<circle cx='8' cy='19' r='1.7'/><circle cx='16' cy='19' r='1.7'/></svg>")
    bar_l, bar_r = st.columns([9, 1.4])
    bar_l.markdown(
        f"<div class='banner'><div class='logo'>{bus_icon}</div>"
        f"<div><div class='b1'>Brooklyn <span>Bus Equity Explorer</span></div>"
        f"<div class='b2'>Where bus service reaches the people who need it most</div></div>"
        f"<div class='bchips'>{''.join(chips)}</div></div>", unsafe_allow_html=True)
    with bar_r:
        try:
            box = st.popover("ⓘ Method")
        except AttributeError:                       # older Streamlit versions
            box = st.expander("ⓘ Method")
    with box:
        used = ", ".join(INDICATOR_LABEL[k].lower() for k in ameta["indicators"])
        st.markdown(f"""
**What this shows.** For Brooklyn, how much each {unit} needs bus service (one equity score) and what service reaches it.
It screens for places to look at. It does not say where service should change, and it is not MTA's Title VI analysis.

**The selected date.** Ridership is that day's counts. The schedule is the GTFS service that ran that day: the calendar's weekday and date
rules, then its exceptions. Stops, routes, trips, waits and the "within a quarter mile" coverage are all computed from that day's service,
so a weekend shows weekend service.

**Equity score.** Built for each {unit} from: {used}.
Each indicator is scaled 0 to 1 across Brooklyn {unit}s and combined with weights following MTA's published Equity Priority Area table
({", ".join(f"{INDICATOR_LABEL[k].lower()} {WEIGHTS[k]}" for k in ameta["indicators"])}).
A {unit} missing some indicators is scored on the rest if at least 60% of the weight is available. Areas under {MIN_POP} residents are not scored.
Need tiers are thirds of the score. The map color runs from the 2nd to the 98th percentile of scores so a few extremes do not wash it out.

**Expected wait.** How long a rider who arrives at a random moment waits for the next scheduled bus, on average, on the selected date.
It is shown separately for the AM peak ({AM_START // 3600} to {AM_END // 3600} AM) and the PM peak ({PM_START // 3600 - 12} to {PM_END // 3600 - 12} PM), in the KPIs and in
the route and stop tooltips. Only the time between two scheduled departures counts, so hours before the first bus or after the last do not.
For buses every h minutes it is h / 2. A stop takes the shortest wait among its routes, and a selection shows the median across its stops.

**Limits.**
- Brooklyn routes are chosen by route name. Some Q and X routes that serve Brooklyn are not counted as Brooklyn.
- Schedules are scheduled service. They do not show late or cancelled buses.
- ACS values are estimates with margins of error.
- Distances are straight-line from the {unit} center, not walking routes.

**Build details.** Schedule files used: {', '.join(meta.get('feeds_used', [])) or 'none'}. Built {meta.get('built', '')}.
""")
        ctx = coverage_context(t, unit)
        if ctx:
            st.caption(ctx)

    notes_ = [dir_note] if dir_note else []
    if basemap != "None" and not TILE_DIR.exists():
        notes_.append("Basemap tiles not found. Run render_basemap_tiles.py to draw them.")
    if (route != "All routes" or stop != "All stops") and rid_sel.empty:
        notes_.append("No ridership data is available for this selection on the selected date.")
    if (route != "All routes" or stop != "All stops") and svc_sel.empty:
        notes_.append("No scheduled service for this selection on the selected date.")

    # ---------- KPI strip: six cards, three of them stacked pairs ----------
    wait_am, wait_pm = svc_sel["am_wait_min"].median(), svc_sel["pm_wait_min"].median()
    am_hours = f"{AM_START // 3600} and {AM_END // 3600} AM"
    pm_hours = f"{PM_START // 3600 - 12} and {PM_END // 3600 - 12} PM"
    BLUE, TEAL, SLATE = "#0039A6", "#0f5257", "#7a8aa0"

    def kpi_one(label, value, hint, tone):
        return (f"<div class='kpi' style='--c:{tone}' title='{esc(hint, quote=True)}'>"
                f"<div class='kt'>{label}</div><div class='kv'>{value}</div></div>")

    def kpi_pair(label, pair, hint, tone):
        rows = "".join(f"<div class='kr'><span>{k}</span><b>{v}</b></div>" for k, v in pair)
        return (f"<div class='kpi stack' style='--c:{tone}' title='{esc(hint, quote=True)}'>"
                f"<div class='kt'>{label}</div>{rows}</div>")

    st.markdown("<div class='kpis'>" + "".join([
        kpi_pair("In this view", [("Routes", f"{svc_sel['route_id'].nunique():,}"), ("Stops", f"{svc_sel['stop_id'].nunique():,}")],
                 "Routes and stops in the current selection.", SLATE),
        kpi_pair("Ridership", [("Boardings", f"{rid_sel['boardings'].sum():,.0f}"), ("Alightings", f"{rid_sel['alightings'].sum():,.0f}")],
                 f"Observed boardings and alightings on {period}.", BLUE),
        kpi_pair("Expected wait", [("AM peak", "n/a" if pd.isna(wait_am) else f"{wait_am:.1f} min"),
                                   ("PM peak", "n/a" if pd.isna(wait_pm) else f"{wait_pm:.1f} min")],
                 f"How long a rider who arrives at a random moment waits for the next scheduled bus, on average, on {period}: AM peak is "
                 f"between {am_hours}, PM peak between {pm_hours}. Median across the selected stops and routes.", TEAL),
        kpi_one("Residents within ¼ mile", f"{near_pop:,.0f}",
                f"Population of {unit}s whose center is within a quarter mile of the selected stops.", SLATE),
        kpi_one("Scheduled trips", f"{scheduled_trips:,}", f"Distinct GTFS trips scheduled on {period}. For one stop, trips that serve it.", BLUE),
        kpi_one("Avg equity score", "n/a" if pd.isna(near_score) else f"{near_score:.0f}",
                "Population-weighted average equity score of those areas. Higher means greater need.", SLATE),
    ]) + "</div>", unsafe_allow_html=True)

    # ---------- measure the browser window, so the map and the lists fill exactly the space that is left ----------
    # Keep the map compact enough that the ranking rail is visible on a normal
    # desktop viewport. The rail is allowed to grow instead of clipping cards.
    map_h = 480

    # ---------- the map (left) and the top lists (right), both map_h tall ----------
    # The map is a component that keeps its basemap and view and swaps only the three data layers, so a new date, route or
    # list selection does not redraw the map. Clicks and hovers stay in the browser and never rerun the app.
    tile_url = TILES[basemap]
    payload = {
        "height": map_h,
        "tiles": ({"url": tile_url, "attr": TILE_ATTR, "minZoom": TILE_ZOOM[0], "maxZoom": 19, "maxNative": TILE_ZOOM[1]} if tile_url else None),
        "view": view,
        "viewKey": json.dumps(view, sort_keys=True) + f"|{st.session_state.get('view_nonce', 0)}",
        "legend": {"lo": lo_s, "hi": hi_s, "loRgb": list(RAMP_LO), "hiRgb": list(RAMP_HI)},
        "notes": notes_,
        "ctx": {"unit": U, "scope": scope, "period": period, "direction": direction if direction != "All directions" else ""},
        "tracts": tract_rows if show_areas else [],
        "routes": route_rows if show_routes else [],
        "stops": stop_rows if show_stops else [],
    }
    
    map_col, rail_col = st.columns([1.8, 1], gap="small")
    with map_col:
        _MAP(payload=payload, key="map", default=None)

    GAP0 = 6                               # space Streamlit puts between stacked elements

    def fit(n, area, min_h, max_h):
        """Row height and extra breathing space so that n rows fill `area` pixels. Rows never get shorter than min_h."""
        unit = (area - (n - 1) * GAP0) / n
        h = int(min(max_h, max(min_h, unit - 8)))
        return h, int(min(22, max(4, unit - h)))

    list_h = map_h - 42                  # tab bar is about 40 px, so the list box ends level with the map's bottom edge
    ROW_H, ROW_PAD = 64, 5              # every row in every list has this exact size, so nothing is cut and items line up between tabs
    h10 = h5 = ROW_H
    p10 = p5 = ROW_PAD

    def scroll_box():
        return st.container(height=list_h, border=False)
    panel = rail_col.container()
    with panel:
        need_short = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}
        lists = st.tabs(["Priority Tracts", "Priority Stops", "Priority Routes"])

        # ---- Top tracts: highest equity score ----
        with lists[0], scroll_box():
            st.markdown(f"<div class='cap'>Highest equity score, {MIN_TRACT_POP:,}+ residents. Ranking is fixed; nearest-stop distance follows your filters.</div>", unsafe_allow_html=True)
            tbl = t[t["equity_score"].notna() & (t["pop"] >= MIN_TRACT_POP)].sort_values("equity_score", ascending=False).head(TOP_TRACTS)
            if tbl.empty:
                st.markdown(f"<div class='cap'>No {unit}s have a score and at least {MIN_TRACT_POP:,} residents.</div>", unsafe_allow_html=True)
            for i, (_, row) in enumerate(tbl.iterrows(), 1):
                a_, b_ = st.columns([11, 1.4], gap="small")
                a_.markdown(row_html(
                    i, f"{U} {row['GEOID']}",
                    [f"{row['pop']:,.0f} residents"],
                    f"{row['equity_score']:.1f}", "equity score", bar=row["equity_score"], height=h10, pad=p10, 
                    hint=f"{U} {row['GEOID']}: equity score {row['equity_score']:.1f}, {row['pop']:,.0f} residents, nearest {scope} "
                         f"{fmt_dist(row['near_m'])}, {_pct(row['cover_any'] * 100)} of the {unit} within a quarter mile of a stop."),
                    unsafe_allow_html=True)
                b_.button("↗", key=f"zoom_tract_{row['GEOID']}", help="Zoom to this tract", on_click=_zoom_tract,
                          args=(row["GEOID"], float(row["lat"]), float(row["lon"])))

        # ---- Top stops: longest wait at the worse of the two peaks ----
        with lists[1], scroll_box():
            st.markdown(f"<div class='cap'>Longest wait at the worse peak, stops with {MIN_STOP_BOARDINGS}+ boardings. Shortest wait among a stop's routes. Follows route and direction.</div>",
                        unsafe_allow_html=True)
            by_stop = svc_rd.groupby("stop_id").agg(am=("am_wait_min", "min"), pm=("pm_wait_min", "min"),
                                                     routes=("route_id", lambda x: ", ".join(sorted(set(x)))))
            by_stop["boardings"] = by_stop.index.to_series().map(rid_rd.groupby("stop_id")["boardings"].sum()).fillna(0).to_numpy()
            by_stop["worst"] = by_stop[["am", "pm"]].max(axis=1, skipna=True)
            eligible = by_stop[(by_stop["boardings"] >= MIN_STOP_BOARDINGS) & by_stop["worst"].notna()]
            by_stop = eligible.sort_values(["worst", "boardings"], ascending=False).head(TOP_N)
            if by_stop.empty:
                st.markdown(f"<div class='cap'>No stops with {MIN_STOP_BOARDINGS}+ boardings and scheduled peak service for this selection and date.</div>",
                            unsafe_allow_html=True)
            for i, (sid, row) in enumerate(by_stop.iterrows(), 1):
                tier = tier_of.get(sid)
                need = need_short.get(int(tier), "Not scored") if pd.notna(tier) else "Not scored"
                name = str(stops_tbl["stop_name"].get(sid, "Unknown stop"))
                a_, b_ = st.columns([11, 1.4], gap="small")
                a_.markdown(row_html(
                    i, esc(name), [f"Stop {esc(sid)} · routes {esc(row['routes'])}"],
                    f"{row['worst']:.1f} min", "longer peak wait", height=h5, pad=p5, 
                    chips=pills([("AM", f"{_pk(row['am'])} min"), ("PM", f"{_pk(row['pm'])} min")]),
                    hint=f"{name} ({sid}): AM peak {_peak(row['am'])}, PM peak {_peak(row['pm'])}, {row['boardings']:,.0f} boardings, "
                         f"routes {row['routes']}, {need}."), unsafe_allow_html=True)
                q = stops_tbl.loc[sid]
                chosen_route = str(svc_rd.loc[svc_rd["stop_id"] == sid, "route_id"].iloc[0]) if route == "All routes" else route
                b_.button("↗", key=f"zoom_wait_{sid}", help="Filter and zoom to this stop", on_click=_go_stop,
                          args=(chosen_route, sid, float(q["lat"]), float(q["lon"])))
            st.markdown(f"<div class='about'><b>How this list works.</b> {len(eligible):,} stops meet the {MIN_STOP_BOARDINGS}-boarding minimum "
                        f"on this date{'' if route == 'All routes' else ' for the selected route'}. Waits are scheduled, assuming riders arrive at random.</div>",
                        unsafe_allow_html=True)

        # ---- Top routes: boardings, with the equity lens ----
        with lists[2], scroll_box():
            st.markdown("<div class='cap'>Most boardings that day, both directions unless one is chosen. Ignores the route filter.</div>", unsafe_allow_html=True)
            rid_dir = rid if direction == "All directions" else rid[rid["direction"] == direction]
            have_dir = direction != "All directions"
            svc_dir = svc_all[svc_all["dir"] == direction] if have_dir and svc_all["dir"].eq(direction).any() else svc_all
            rt_dir = rt_full[rt_full["dir"] == direction] if have_dir and rt_full["dir"].eq(direction).any() else rt_full
            boards = rid_dir.groupby("route_id")["boardings"].sum()
            need_boards = rid_dir[rid_dir["stop_id"].map(tier_of) == 1].groupby("route_id")["boardings"].sum()
            waits_r = svc_dir.groupby("route_id")[["am_wait_min", "pm_wait_min"]].median()
            trips_r = rt_dir.groupby("route_id")["weekday_trips"].sum()
            top_routes = boards[boards > 0].sort_values(ascending=False).head(TOP_N)
            if top_routes.empty:
                st.markdown("<div class='cap'>No ridership on the selected date.</div>", unsafe_allow_html=True)
            for i, (rid_top, value) in enumerate(top_routes.items(), 1):
                share = need_boards.get(rid_top, 0) / value if value else 0
                am_r, pm_r = waits_r["am_wait_min"].get(rid_top), waits_r["pm_wait_min"].get(rid_top)
                a_, b_ = st.columns([11, 1.4], gap="small")
                a_.markdown(row_html(
                    i, f"Route {esc(str(rid_top))}", [f"{trips_r.get(rid_top, 0):,.0f} scheduled trips on this date"],
                    f"{value:,.0f}", "boardings", bar=share * 100, height=h5, pad=p5, 
                    chips=pills([("AM", f"{_pk(am_r)} min"), ("PM", f"{_pk(pm_r)} min")]),
                    hint=f"Route {rid_top}: {value:,.0f} boardings, {share:.0%} at stops in the highest-need third, "
                         f"{trips_r.get(rid_top, 0):,.0f} scheduled trips, AM peak wait {_peak(am_r)}, PM peak wait {_peak(pm_r)}."),
                    unsafe_allow_html=True)
                b_.button("↗", key=f"zoom_route_{rid_top}", help="Filter and zoom to this route", on_click=_go_route,
                          args=(str(rid_top),))
            st.markdown("<div class='about'><b>How to read the bar.</b> The blue bar under each figure is the share of that route's boardings "
                        "that happen at stops in the highest-need third of tracts. A longer bar means the route serves more high-need riders.</div>",
                        unsafe_allow_html=True)


def _pct(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:.0f}%"


def _pk(v) -> str:
    return "–" if pd.isna(v) else f"{v:.1f}"


def _peak(v) -> str:
    return "no service" if pd.isna(v) else f"{v:.1f} min"


def _wait(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:.1f} min"


def coverage_context(t: pd.DataFrame, unit: str) -> str:
    d = t[t["tier"] == 1].dropna(subset=["pop", "cover_freq", "cover_any"])
    if d.empty or d["pop"].sum() <= 0:
        return ""
    f = np.average(d["cover_freq"], weights=d["pop"])
    a = np.average(d["cover_any"], weights=d["pop"])
    return (f"Across all Brooklyn routes, in the highest-need third of {unit}s: about {a:.0%} of the area is within a quarter mile "
            f"of any stop and {f:.0%} is within a quarter mile of frequent service (10 minutes or better, 6 AM to 9 PM).")


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if "--build" in sys.argv and not _in_streamlit():
        build_all()
    elif _in_streamlit():
        main()
    else:
        print("Run the app with:  streamlit run app1.py\nBuild data with:   python app1.py --build")
