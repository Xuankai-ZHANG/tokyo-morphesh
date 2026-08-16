# -*- coding: utf-8 -*-
"""Multi-scenario inter-community energy-sharing simulation engine.

Simulates rooftop-PV generation, demand, and local surplus sharing across a
mesh network for a set of technical-potential fractions. Surplus is allocated
to nearby deficit meshes under a conditional donor-normalized rule.
"""

import argparse
import yaml
import hashlib
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as csgraph_cc
from pathlib import Path
import time
import sys
import calendar
import json
import glob
from collections import OrderedDict
from datetime import datetime, timezone

from allocation_rules import conditional_donor_normalized
from simulation_utils import restore_lambda_totals

# Force UTF-8 output on Windows
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ─── CLI ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Inter-community Energy-Sharing Simulation")
parser.add_argument("--config", type=str, default=None,
                    help="Path to YAML config file")
parser.add_argument("--radius-km", type=float, default=None,
                    help="Candidate graph radius (km). Overrides config.")
parser.add_argument("--mode", type=str, default="primary",
                    choices=["smoke", "primary", "radius_sensitivity", "weight_robustness"],
                    help="Run mode: smoke (2-day validation), primary (7-alpha R=5 λ=5)")
parser.add_argument("--alpha", type=float, nargs="*", default=None,
                    help="Technical-potential fraction(s) to run (overrides mode default)")
parser.add_argument("--lambda-km", type=str, nargs="*", default=None,
                    help="Decay length(s) in km, or 'uniform_local' (overrides mode default)")
args = parser.parse_args()

# ─── Paths ─────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data" / "input" / "mesh"
from formal_input_contract import load_technical_potential, validate_technical_potential
STATIC_DATA_DIR = PROJ_ROOT / "data" / "input" / "shapes"  # gen_shapes, dem_shapes, season_factors
OUT_DIR_BASE = PROJ_ROOT / "data" / "results"
OUT_DIR = OUT_DIR_BASE  # may be overridden after RADIUS_KM resolved
MANIFEST_DIR = OUT_DIR / "manifests"

# ─── Config loading ────────────────────────────────────────────────
_config = {}
if args.config:
    with open(args.config, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)

# Resolve radius_km: CLI > config > default(5)
RADIUS_KM = args.radius_km
if RADIUS_KM is None:
    try:
        RADIUS_KM = _config["primary"]["radius_km"]
    except (KeyError, TypeError):
        RADIUS_KM = 5.0

# Resolve mode
MODE = args.mode

# Resolve lambdas
UNIFORM_LOCAL_LABEL = "uniform_local"
if args.lambda_km is not None:
    _user_lambdas = []
    for v in args.lambda_km:
        if v == UNIFORM_LOCAL_LABEL:
            _user_lambdas.append(UNIFORM_LOCAL_LABEL)
        else:
            _user_lambdas.append(float(v))
    LAMBDAS_KM = [v for v in _user_lambdas if isinstance(v, float)]
    HAS_UNIFORM_LOCAL = UNIFORM_LOCAL_LABEL in _user_lambdas
else:
    if MODE == "weight_robustness":
        LAMBDAS_KM = [1.0, 5.0]
        HAS_UNIFORM_LOCAL = True
    else:
        LAMBDAS_KM = [5.0]
        HAS_UNIFORM_LOCAL = False

LAMBDA_INF_LABEL = "inf"
UNIFORM_LOCAL_LABEL_CONST = "uniform_local"
if HAS_UNIFORM_LOCAL:
    ALL_LAMBDAS = LAMBDAS_KM + [UNIFORM_LOCAL_LABEL_CONST, LAMBDA_INF_LABEL]
else:
    ALL_LAMBDAS = LAMBDAS_KM + [LAMBDA_INF_LABEL]
# Weight-based lambdas (all except inf — use weight matrices)
WEIGHT_LAMBDAS = [lam for lam in ALL_LAMBDAS if lam != LAMBDA_INF_LABEL]
# Primary lambda for summary display (use first float lambda, or fallback)
_float_lams = [lam for lam in ALL_LAMBDAS if isinstance(lam, float)]
PRIMARY_LAMBDA = 5.0 if 5.0 in _float_lams else (_float_lams[0] if _float_lams else 5.0)

# Resolve technical-potential scenarios
if args.alpha is not None:
    ALPHA_SCENARIOS = OrderedDict()
    for a in args.alpha:
        label = f"tp{int(a*100):03d}" if a < 1.0 else "tp100"
        ALPHA_SCENARIOS[label] = a
elif MODE == "smoke":
    ALPHA_SCENARIOS = OrderedDict([("tp030", 0.30)])
elif MODE == "radius_sensitivity" or MODE == "weight_robustness":
    ALPHA_SCENARIOS = OrderedDict([("tp030", 0.30)])
else:
    ALPHA_SCENARIOS = OrderedDict([
        ("tp010", 0.10),
        ("tp015", 0.15),
        ("tp020", 0.20),
        ("tp030", 0.30),
        ("tp040", 0.40),
        ("tp050", 0.50),
        ("tp100", 1.00),
    ])

# ─── Smoke mode: pre-select 2 days ─────────────────────────────────
# 2024-01-07 = Sunday (winter weekend), 2024-07-10 = Wednesday (summer weekday)
SMOKE_DAYS = {(1, 7), (7, 10)}  # (month, day)

# ─── Helpers ───────────────────────────────────────────────────────
def log(msg: str, **kwargs):
    print(msg, flush=True, **kwargs)


# Finalize output directory (after all params resolved)
if MODE == "smoke":
    OUT_DIR = OUT_DIR_BASE / f"smoke_R{RADIUS_KM:.0f}"
elif MODE == "radius_sensitivity":
    OUT_DIR = OUT_DIR_BASE / "sensitivity" / f"radius_R{RADIUS_KM:.0f}"
elif MODE == "weight_robustness":
    OUT_DIR = OUT_DIR_BASE / "sensitivity" / "weight"
# primary mode: OUT_DIR stays at OUT_DIR_BASE (root)
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR = OUT_DIR / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# Summary tables → summary/; per-mesh daily tables → daily/ (primary mode).
# smoke/sensitivity modes keep a flat OUT_DIR.
if MODE == "primary":
    SUMMARY_DIR = OUT_DIR_BASE / "summary"
    DAILY_DIR = OUT_DIR_BASE / "daily"
else:
    SUMMARY_DIR = OUT_DIR
    DAILY_DIR = OUT_DIR
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)

log(f"Mode: {MODE}, R={RADIUS_KM} km, λ={ALL_LAMBDAS}, α={list(ALPHA_SCENARIOS.keys())}")
log(f"Output: {OUT_DIR}")
if MODE == "smoke":
    log(f"Smoke days: {sorted(SMOKE_DAYS)} (weekend + weekday)")


def build_weight_matrix_raw(dist_i, dist_j, dist_d, lam_km, n_meshes):
    """Build receiver-by-donor distance-decay weights on the R-filtered graph."""
    if isinstance(lam_km, str) and lam_km == UNIFORM_LOCAL_LABEL_CONST:
        w_raw = np.ones(len(dist_d), dtype=np.float32)
    else:
        w_raw = np.exp(-dist_d / lam_km).astype(np.float32)
    return csr_matrix(
        (w_raw, (dist_i, dist_j)),
        shape=(n_meshes, n_meshes),
        dtype=np.float32,
    )


def graph_audit(dist_i, dist_j, dist_d, n_meshes, radius_km):
    """Compute graph diagnostics for the R-filtered candidate graph."""
    n_edges = len(dist_d)
    if n_edges == 0:
        return {"n_edges": 0, "avg_degree": 0.0, "n_components": n_meshes,
                "n_isolates": n_meshes, "max_distance_km": 0.0,
                "radius_km": radius_km}
    degree = np.bincount(dist_i, minlength=n_meshes) + np.bincount(dist_j, minlength=n_meshes)
    avg_degree = float(np.mean(degree))
    n_isolates = int((degree == 0).sum())
    # Build adjacency for component count
    adj = csr_matrix(
        (np.ones(n_edges, dtype=np.int8), (dist_i, dist_j)),
        shape=(n_meshes, n_meshes), dtype=np.int8
    )
    n_components, _ = csgraph_cc(adj, directed=False)
    return {
        "n_edges": int(n_edges),
        "avg_degree": round(avg_degree, 4),
        "n_components": int(n_components),
        "n_isolates": int(n_isolates),
        "max_distance_km": round(float(dist_d.max()), 6),
        "radius_km": radius_km,
    }


def system_equalization_cssr(surplus, deficit, dem):
    """λ → ∞: all surplus pooled, distributed proportional to deficit."""
    total_surplus = surplus.sum()
    total_deficit = deficit.sum()
    if total_deficit > 0 and total_surplus > 0:
        allocation_ratio = min(1.0, total_surplus / total_deficit)
        imported = deficit * allocation_ratio
    else:
        imported = np.zeros_like(deficit)
    self_consumed = dem - deficit
    served = self_consumed + np.minimum(deficit, imported)
    cssr = np.where(dem > 0, served / dem, 0.0)
    return cssr


def gini(x):
    """Gini coefficient of array x."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n


def make_accumulators(n_meshes, n_cats, n_days):
    """Create a fresh set of accumulators for one scenario."""
    accum = {
        "demand": np.zeros(n_meshes, dtype=np.float64),
        "self_consumed": np.zeros(n_meshes, dtype=np.float64),
        "served": {lam: np.zeros(n_meshes, dtype=np.float64) for lam in ALL_LAMBDAS},
        "surplus": np.zeros(n_meshes, dtype=np.float64),
        "imported": {lam: np.zeros(n_meshes, dtype=np.float64) for lam in ALL_LAMBDAS},
        "total_surplus_all": {lam: 0.0 for lam in ALL_LAMBDAS},
        "total_imported_all": {lam: 0.0 for lam in ALL_LAMBDAS},
        # Daily (subset of λ only)
        "daily_demand": np.zeros((n_meshes, n_days), dtype=np.float32),
        "daily_served": {
            lam: np.zeros((n_meshes, n_days), dtype=np.float32)
            for lam in [1.0, 5.0, 10.0, LAMBDA_INF_LABEL]
        },
        # Category profiles
        "cat_gen": np.zeros((n_cats, 48), dtype=np.float64),
        "cat_dem": np.zeros((n_cats, 48), dtype=np.float64),
        "cat_served": {
            lam: np.zeros((n_cats, 48), dtype=np.float64) for lam in ALL_LAMBDAS
        },
        "cat_self_consumed": np.zeros((n_cats, 48), dtype=np.float64),
        # Stats
        "total_steps": 0,
        "skipped_steps": 0,
        "active_steps": 0,
    }
    return accum


# ─── 1. Load Base Data ─────────────────────────────────────────────
log("=" * 70)
log("Conditional donor-normalized inter-community energy-sharing simulation")
log("=" * 70)

t0_total = time.time()

log("\n[1/7] Loading base data and building mesh index...")
t0 = time.time()

# Annual totals + mesh attributes
tp_df, formal_potential_manifest = load_technical_potential()
base_meta = pd.read_parquet(
    DATA_DIR / "mesh_attributes.parquet",
    columns=["mesh_code", "unit_type"]
)
base = tp_df.merge(base_meta, on="mesh_code", how="inner")
base = base[["mesh_code", "potential_kwh", "annual_dem_kwh",
             "category_3", "urbanization_index", "unit_type"]].copy()

# Sparse distance matrix
dist = pd.read_parquet(DATA_DIR / "sparse_distances.parquet")

# Season factors
season = pd.read_parquet(STATIC_DATA_DIR / "season_factors.parquet")
season_map = dict(zip(season["month"], season["season_factor"]))

# DEM_NORM: monthly days × season_factor, ensures annual demand sum correct
DEM_NORM = sum(
    calendar.monthrange(2024, m)[1] * season_map[m]
    for m in range(1, 13)
)
log(f"  DEM_NORM = {DEM_NORM:.4f} (annual demand normalization constant)")

# Dem shapes
dem_shapes_all = pd.read_parquet(STATIC_DATA_DIR / "dem_shapes.parquet")

log(f"  technical_potential: {len(tp_df)} meshes")
log(f"  sparse_distances: {len(dist):,} pairs")
log(f"  dem_shapes:        {len(dem_shapes_all):,} rows")
log(f"  season_factors:    {len(season)} months")
log(f"  technical-potential scenarios: {list(ALPHA_SCENARIOS.keys())}")

# ─── 2. Build Unified Mesh Index ───────────────────────────────────
log(f"\n[2/7] Building mesh index...")
base_meshes = set(base["mesh_code"])

gen_files = sorted(glob.glob(str(STATIC_DATA_DIR / "gen_shapes" / "month=*.parquet")))
assert len(gen_files) == 12, f"Expected 12 monthly gen_shapes files, found {len(gen_files)}"
gs_meshes = set()
for gf in gen_files:
    gs_part = pd.read_parquet(gf, columns=["mesh_code"])
    gs_meshes.update(gs_part["mesh_code"].unique())

dem_meshes = set(dem_shapes_all["mesh_code"])
dist_meshes_from = set(dist["mesh_idx_i"])
dist_meshes_to = set(dist["mesh_idx_j"])

common_meshes = sorted(base_meshes & gs_meshes & dem_meshes)
n_meshes = len(common_meshes)
formal_contract = validate_technical_potential(
    tp_df, formal_potential_manifest, set(common_meshes)
)
log(f"  FORMAL POTENTIAL CONTRACT: {formal_contract['potential_kwh']/1e9:.6f} TWh")
log(f"  base meshes:       {len(base_meshes)}")
log(f"  gen_shapes meshes: {len(gs_meshes)}")
log(f"  dem_shapes meshes: {len(dem_meshes)}")
log(f"  common (simulation): {n_meshes}")

# Mesh → positional index
mesh_to_idx = {m: i for i, m in enumerate(common_meshes)}
mesh_codes = np.array(common_meshes, dtype=np.int64)

# ─── 3. Extract and Align Arrays ───────────────────────────────────
log(f"\n[3/7] Aligning arrays to {n_meshes}-mesh index...")
t0 = time.time()

base_idx = base.set_index("mesh_code").loc[common_meshes]
potential = base_idx["potential_kwh"].values.astype(np.float64)
annual_dem = base_idx["annual_dem_kwh"].values.astype(np.float64)
category_3 = base_idx["category_3"].values
urban_idx = base_idx["urbanization_index"].values.astype(np.float64)

# Filter out meshes with zero demand
valid_dem = annual_dem > 0
if not valid_dem.all():
    n_invalid = (~valid_dem).sum()
    log(f"  WARNING: {n_invalid} meshes have zero annual demand — filtering out")
    # Rebuild mesh index excluding zero-demand meshes
    valid_indices = np.where(valid_dem)[0]
    common_meshes_valid = [common_meshes[i] for i in valid_indices]
    mesh_to_idx = {m: i for i, m in enumerate(common_meshes_valid)}
    mesh_codes = np.array(common_meshes_valid, dtype=np.int64)
    n_meshes = len(mesh_codes)
    potential = potential[valid_indices]
    annual_dem = annual_dem[valid_indices]
    category_3 = category_3[valid_indices]
    urban_idx = urban_idx[valid_indices]
    log(f"  After filtering: {n_meshes} meshes")

cats = np.unique(category_3)
cat_to_int = {c: i for i, c in enumerate(cats)}
cat_int = np.array([cat_to_int[c] for c in category_3], dtype=np.int32)
n_cats = len(cats)

pot_total = potential.sum()
dem_total = annual_dem.sum()
log(f"  potential total:      {pot_total:.2e} kWh = {pot_total/1e9:.1f} TWh")
log(f"  annual_dem total:     {dem_total:.2e} kWh = {dem_total/1e9:.1f} TWh")
log(f"  gen/dem ratio (α=1.0): {pot_total/dem_total:.2f}")
for a_name, a_val in ALPHA_SCENARIOS.items():
    log(f"  gen/dem ratio ({a_name}): {a_val*pot_total/dem_total:.2f}")
log(f"  categories:           {list(cats)}")

# ─── 4. Build Dem Shape Lookup ─────────────────────────────────────
log(f"\n[4/7] Building demand shape lookup...")
t0 = time.time()

dem_lookup = {}
dem_mesh_set = set(mesh_codes)
dem_filtered = dem_shapes_all[dem_shapes_all["mesh_code"].isin(dem_mesh_set)]

for (month, dt, slot), group in dem_filtered.groupby(["month", "day_type", "slot"]):
    arr = np.zeros(n_meshes, dtype=np.float64)
    for m, v in zip(group["mesh_code"].values, group["dem_shape"].values):
        if m in mesh_to_idx:
            arr[mesh_to_idx[m]] = v
    dem_lookup[(month, dt, slot)] = arr

log(f"  dem_lookup keys: {len(dem_lookup)} (12 months × 2 day_types × 48 slots)")
log(f"  Build time: {time.time() - t0:.1f}s")

# ─── 5. Build Weight Matrices (with radius filtering) ───────────────
log(f"\n[5/7] Building weight matrices (R={RADIUS_KM} km)...")
t0 = time.time()

orig_base = pd.read_parquet(DATA_DIR / "mesh_attributes.parquet")
orig_mesh_list = orig_base["mesh_code"].values
orig_to_new = {int(m): mesh_to_idx[int(m)] for m in orig_mesh_list if int(m) in mesh_to_idx}

dist_i_orig = dist["mesh_idx_i"].values.astype(np.int32)
dist_j_orig = dist["mesh_idx_j"].values.astype(np.int32)
dist_d = dist["distance_km"].values.astype(np.float32)

dist_mesh_i = orig_mesh_list[dist_i_orig]
dist_mesh_j = orig_mesh_list[dist_j_orig]

keep_i = np.array([int(m) in mesh_to_idx for m in dist_mesh_i])
keep_j = np.array([int(m) in mesh_to_idx for m in dist_mesh_j])
keep = keep_i & keep_j

dist_i_new = np.array([mesh_to_idx[int(m)] for m in dist_mesh_i[keep]], dtype=np.int32)
dist_j_new = np.array([mesh_to_idx[int(m)] for m in dist_mesh_j[keep]], dtype=np.int32)
dist_d_new = dist_d[keep]

log(f"  Mesh-filtered pairs: {len(dist_d_new):,} / {len(dist_d):,} "
    f"({len(dist_d_new)/len(dist_d)*100:.1f}%)")

# ── Radius filtering (P0-B fix) ──
radius_mask = dist_d_new <= RADIUS_KM
dist_i_r = dist_i_new[radius_mask]
dist_j_r = dist_j_new[radius_mask]
dist_d_r = dist_d_new[radius_mask]

n_dropped = len(dist_d_new) - len(dist_d_r)
log(f"  Radius filter (≤{RADIUS_KM} km): {len(dist_d_r):,} pairs kept, "
    f"{n_dropped:,} dropped ({n_dropped/max(1,len(dist_d_new))*100:.1f}%)")

# ⚠ Stop condition: no edge must exceed radius
if len(dist_d_r) > 0:
    max_d = dist_d_r.max()
    assert max_d <= RADIUS_KM + 1e-9, \
        f"STOP: edge with distance_km={max_d:.6f} > R={RADIUS_KM} found!"
    log(f"  ✓ Max distance in graph: {max_d:.6f} km ≤ R={RADIUS_KM}")

# Graph audit
_audit = graph_audit(dist_i_r, dist_j_r, dist_d_r, n_meshes, RADIUS_KM)
log(f"  Graph audit: edges={_audit['n_edges']:,}, avg_deg={_audit['avg_degree']:.2f}, "
    f"components={_audit['n_components']}, isolates={_audit['n_isolates']}, "
    f"max_d={_audit['max_distance_km']:.4f} km")

weight_matrices = {}
for lam in ALL_LAMBDAS:
    if lam == LAMBDA_INF_LABEL:
        continue  # system pooling doesn't use weight matrix
    t_lam = time.time()
    W_raw = build_weight_matrix_raw(dist_i_r, dist_j_r, dist_d_r, lam, n_meshes)
    weight_matrices[lam] = W_raw
    nnz = W_raw.nnz
    lam_label = lam if isinstance(lam, str) else f"{lam:.1f} km"
    log(f"  λ={lam_label}: {nnz:,} non-zeros, "
        f"density={nnz/(n_meshes*n_meshes)*100:.3f}%, "
        f"build time={time.time()-t_lam:.1f}s")

log(f"  Total weight matrix build: {time.time()-t0:.1f}s")

# ─── 6. Main Simulation Loop ───────────────────────────────────────
log(f"\n[6/7] Running multi-scenario simulation...")
t0 = time.time()

# Daily lambdas for per-day output (subset to keep file size manageable)
_float_daily = [lam for lam in ALL_LAMBDAS if isinstance(lam, float) and lam in [1.0, 5.0]]
if PRIMARY_LAMBDA not in _float_daily and isinstance(PRIMARY_LAMBDA, float):
    _float_daily.append(PRIMARY_LAMBDA)
DAILY_LAMBDAS = _float_daily + [LAMBDA_INF_LABEL]
n_days_in_year = 366  # 2024 is leap year

# ── Create accumulators per scenario ──
accumulators = {
    name: make_accumulators(n_meshes, n_cats, n_days_in_year)
    for name in ALPHA_SCENARIOS
}

# ── Checkpoint setup (per scenario, per radius) ──
CHECKPOINT_ROOT = OUT_DIR / f"checkpoints_R{RADIUS_KM:.0f}"
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

# Load any existing checkpoints
for scenario_name in ALPHA_SCENARIOS:
    ckpt_dir = CHECKPOINT_ROOT / scenario_name
    state_path = ckpt_dir / "sim_state.json"
    if state_path.exists():
        accum = accumulators[scenario_name]
        with open(state_path) as f:
            ckpt_meta = json.load(f)
        # Load accumulators
        ckpt = np.load(ckpt_dir / "accumulators.npz")
        accum["demand"] = ckpt["accum_demand"]
        accum["self_consumed"] = ckpt["accum_self_consumed"]
        accum["surplus"] = ckpt["accum_surplus"]
        for lam in ALL_LAMBDAS:
            key = f"accum_served_{str(lam).replace('.', '_')}"
            if key in ckpt: accum["served"][lam] = ckpt[key]
            key_imp = f"accum_imported_{str(lam).replace('.', '_')}"
            if key_imp in ckpt: accum["imported"][lam] = ckpt[key_imp]
        ckpt.close()

        ckpt_d = np.load(ckpt_dir / "accumulators_daily.npz")
        accum["daily_demand"] = ckpt_d["accum_daily_demand"]
        for lam in DAILY_LAMBDAS:
            key = f"daily_served_{str(lam).replace('.', '_')}"
            if key in ckpt_d: accum["daily_served"][lam] = ckpt_d[key]
        ckpt_d.close()

        ckpt_p = np.load(ckpt_dir / "accumulators_profiles.npz")
        accum["cat_gen"] = ckpt_p["cat_gen_profile"]
        accum["cat_dem"] = ckpt_p["cat_dem_profile"]
        if "cat_self_consumed" in ckpt_p:
            accum["cat_self_consumed"] = ckpt_p["cat_self_consumed"]
        for lam in ALL_LAMBDAS:
            key = f"cat_served_{str(lam).replace('.', '_')}"
            if key in ckpt_p: accum["cat_served"][lam] = ckpt_p[key]
        ckpt_p.close()

        if (ckpt_dir / "sur_totals.json").exists():
            with open(ckpt_dir / "sur_totals.json") as f:
                sur_data = json.load(f)
                accum["total_surplus_all"] = restore_lambda_totals(
                    sur_data["surplus"], ALL_LAMBDAS
                )
                accum["total_imported_all"] = restore_lambda_totals(
                    sur_data["imported"], ALL_LAMBDAS
                )

        if (ckpt_dir / "stats.json").exists():
            with open(ckpt_dir / "stats.json") as f:
                stats = json.load(f)
                accum["total_steps"] = stats.get("total_steps", 0)
                accum["skipped_steps"] = stats.get("skipped_steps", 0)
                accum["active_steps"] = stats.get("active_steps", 0)

        log(f"  [{scenario_name}] Loaded checkpoint: months {sorted(ckpt_meta.get('completed_months', []))} complete")

# ── 2024 calendar ──
day_offset = {1: 0, 2: 31, 3: 60, 4: 91, 5: 121, 6: 152,
              7: 182, 8: 213, 9: 244, 10: 274, 11: 305, 12: 335}
day_type_lookup = {}
for m in range(1, 13):
    n_days = calendar.monthrange(2024, m)[1]
    for d in range(1, n_days + 1):
        doy = day_offset[m] + (d - 1)
        wd = calendar.weekday(2024, m, d)
        day_type_lookup[doy] = "we" if wd >= 5 else "wd"

# ── Monthly iteration (outer loop) ──
for month in range(1, 13):
    # Check if all scenarios completed this month
    all_done = all(
        month in json.loads(
            (CHECKPOINT_ROOT / name / "sim_state.json").read_text()
        ).get("completed_months", [])
        if (CHECKPOINT_ROOT / name / "sim_state.json").exists()
        else False
        for name in ALPHA_SCENARIOS
    )
    if all_done:
        log(f"  Month {month:2d}: all scenarios complete, skipping...")
        continue

    log(f"\n  ── Month {month:2d}/12 ──")
    t_month = time.time()

    # ── Load gen_shapes ONCE for this month ──
    gen_file = STATIC_DATA_DIR / "gen_shapes" / f"month={month:02d}.parquet"
    gen_month = pd.read_parquet(gen_file)
    gen_month = gen_month[gen_month["mesh_code"].isin(mesh_to_idx)]
    gen_month["mesh_idx"] = gen_month["mesh_code"].map(mesh_to_idx)
    gen_month = gen_month.dropna(subset=["mesh_idx"])
    gen_month["mesh_idx"] = gen_month["mesh_idx"].astype(np.int32)

    n_days_in_month = calendar.monthrange(2024, month)[1]
    season_factor = season_map[month]

    # Pre-group gen_shapes by (day, slot)
    gen_by_day_slot = {}
    for (day, slot), group in gen_month.groupby(["day", "slot"]):
        arr = np.zeros(n_meshes, dtype=np.float64)
        idx = group["mesh_idx"].values
        val = group["gen_shape"].values
        arr[idx] = val
        gen_by_day_slot[(int(day), int(slot))] = arr
    all_zero = np.zeros(n_meshes, dtype=np.float64)

    # ── Scenario loop (inner: share gen_shapes) ──
    for scenario_name, alpha in ALPHA_SCENARIOS.items():
        accum = accumulators[scenario_name]
        ckpt_dir = CHECKPOINT_ROOT / scenario_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Check if this scenario already completed this month
        state_path = ckpt_dir / "sim_state.json"
        completed_months = set()
        if state_path.exists():
            with open(state_path) as f:
                completed_months = set(json.load(f).get("completed_months", []))
        if month in completed_months:
            continue

        # Free gen_month memory: drop the DataFrame reference after pre-grouping
        # (gen_by_day_slot is shared, gen_month is not needed per-scenario)

        for day in range(1, n_days_in_month + 1):
            # Smoke mode: skip non-selected days
            if MODE == "smoke" and (month, day) not in SMOKE_DAYS:
                continue
            doy = day_offset[month] + (day - 1)         # 0-indexed day of year
            gen_doy = day_offset[month] + day            # 1-indexed (matches gen_shapes "day" column)
            day_type = day_type_lookup[doy]

            for slot in range(48):
                accum["total_steps"] += 1

                gen_shape_arr = gen_by_day_slot.get((gen_doy, slot), all_zero)

                dem_key = (month, day_type, slot)
                dem_shape_arr = dem_lookup.get(dem_key)
                if dem_shape_arr is None:
                    continue

                dem = annual_dem * dem_shape_arr / DEM_NORM

                # Gen with technical-potential scaling
                gen = potential * gen_shape_arr * alpha

                accum["demand"] += dem
                accum["daily_demand"][:, doy] += dem.astype(np.float32)

                self_consumed = np.minimum(gen, dem)
                accum["self_consumed"] += self_consumed

                surplus = np.maximum(gen - dem, 0)
                deficit = np.maximum(dem - gen, 0)

                # ── Nighttime / no-surplus skip ──
                if surplus.sum() <= 0:
                    accum["skipped_steps"] += 1
                    for lam in ALL_LAMBDAS:
                        accum["served"][lam] += self_consumed
                    for lam in DAILY_LAMBDAS:
                        accum["daily_served"][lam][:, doy] += self_consumed.astype(np.float32)
                    for ci in range(n_cats):
                        mask = cat_int == ci
                        if mask.any():
                            accum["cat_gen"][ci, slot] += gen[mask].sum()
                            accum["cat_dem"][ci, slot] += dem[mask].sum()
                            accum["cat_self_consumed"][ci, slot] += self_consumed[mask].sum()
                    for lam in ALL_LAMBDAS:
                        for ci in range(n_cats):
                            mask = cat_int == ci
                            if mask.any():
                                accum["cat_served"][lam][ci, slot] += self_consumed[mask].sum()
                    continue

                accum["active_steps"] += 1
                accum["surplus"] += surplus

                # Category gen/dem profiles
                for ci in range(n_cats):
                    mask = cat_int == ci
                    if mask.any():
                        accum["cat_gen"][ci, slot] += gen[mask].sum()
                        accum["cat_dem"][ci, slot] += dem[mask].sum()
                        accum["cat_self_consumed"][ci, slot] += self_consumed[mask].sum()

                # ── inter-community sharing for each finite λ ──
                for lam in WEIGHT_LAMBDAS:
                    W_raw = weight_matrices[lam]
                    received, imported = conditional_donor_normalized(
                        W_raw, surplus, deficit
                    )

                    served = self_consumed + imported
                    accum["served"][lam] += served
                    accum["imported"][lam] += imported

                    accum["total_surplus_all"][lam] += surplus.sum()
                    accum["total_imported_all"][lam] += imported.sum()

                    for ci in range(n_cats):
                        mask = cat_int == ci
                        if mask.any():
                            accum["cat_served"][lam][ci, slot] += served[mask].sum()

                    if lam in DAILY_LAMBDAS:
                        accum["daily_served"][lam][:, doy] += served.astype(np.float32)

                # ── System equalization (λ → ∞) ──
                cssr_inf = system_equalization_cssr(surplus, deficit, dem)
                served_inf = cssr_inf * dem
                imported_inf = np.minimum(deficit, served_inf - self_consumed)

                accum["served"][LAMBDA_INF_LABEL] += served_inf
                accum["imported"][LAMBDA_INF_LABEL] += imported_inf
                accum["total_surplus_all"][LAMBDA_INF_LABEL] += surplus.sum()
                accum["total_imported_all"][LAMBDA_INF_LABEL] += imported_inf.sum()

                if LAMBDA_INF_LABEL in DAILY_LAMBDAS:
                    accum["daily_served"][LAMBDA_INF_LABEL][:, doy] += served_inf.astype(np.float32)

                for ci in range(n_cats):
                    mask = cat_int == ci
                    if mask.any():
                        accum["cat_served"][LAMBDA_INF_LABEL][ci, slot] += served_inf[mask].sum()

        # ── Save checkpoint for this (month, scenario) ──
        completed_months.add(month)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump({"completed_months": sorted(completed_months)}, f)

        np.savez_compressed(
            ckpt_dir / "accumulators.npz",
            accum_demand=accum["demand"],
            accum_self_consumed=accum["self_consumed"],
            accum_surplus=accum["surplus"],
            **{f"accum_served_{str(lam).replace('.', '_')}": accum["served"][lam] for lam in ALL_LAMBDAS},
            **{f"accum_imported_{str(lam).replace('.', '_')}": accum["imported"][lam] for lam in ALL_LAMBDAS},
        )
        np.savez_compressed(
            ckpt_dir / "accumulators_daily.npz",
            accum_daily_demand=accum["daily_demand"],
            **{f"daily_served_{str(lam).replace('.', '_')}": accum["daily_served"][lam] for lam in DAILY_LAMBDAS},
        )
        np.savez_compressed(
            ckpt_dir / "accumulators_profiles.npz",
            cat_gen_profile=accum["cat_gen"],
            cat_dem_profile=accum["cat_dem"],
            cat_self_consumed=accum["cat_self_consumed"],
            **{f"cat_served_{str(lam).replace('.', '_')}": accum["cat_served"][lam] for lam in ALL_LAMBDAS},
        )
        with open(ckpt_dir / "sur_totals.json", "w") as f:
            json.dump({
                "surplus": accum["total_surplus_all"],
                "imported": accum["total_imported_all"],
            }, f)
        with open(ckpt_dir / "stats.json", "w") as f:
            json.dump({
                "total_steps": accum["total_steps"],
                "skipped_steps": accum["skipped_steps"],
                "active_steps": accum["active_steps"],
            }, f)

    month_elapsed = time.time() - t_month
    log(f"  Month {month:2d} complete in {month_elapsed/60:.1f} min")

total_sim_time = time.time() - t0
log(f"\n  Simulation complete in {total_sim_time/60:.1f} min")

# ─── 7. Post-processing and Output (per scenario) ──────────────────
log(f"\n[7/7] Computing summaries and saving outputs...")

# ── Trajectory summary (cross-scenario) ──
trajectory_rows = []

for scenario_name in ALPHA_SCENARIOS:
    accum = accumulators[scenario_name]
    alpha = ALPHA_SCENARIOS[scenario_name]

    log(f"\n── Processing {scenario_name} (α={alpha}) ──")

    # Skip scenarios with no active steps
    if accum["active_steps"] == 0:
        log(f"  WARNING: 0 active steps — skipping")
        continue

    log(f"  Steps: {accum['total_steps']:,} total, "
        f"{accum['skipped_steps']:,} skipped ({accum['skipped_steps']/max(1,accum['total_steps'])*100:.1f}%), "
        f"{accum['active_steps']:,} active")

    # SSR
    acc_demand = accum["demand"]
    ssr_annual = np.where(acc_demand > 0, accum["self_consumed"] / acc_demand, 0.0)

    # CSSR, CPI, SUR per λ
    cssr_annual = {}
    cpi_annual = {}
    sur_annual = {}
    for lam in ALL_LAMBDAS:
        cssr = np.where(acc_demand > 0, accum["served"][lam] / acc_demand, 0.0)
        cssr_annual[lam] = cssr
        cpi_annual[lam] = cssr - ssr_annual
        if accum["total_surplus_all"][lam] > 0:
            sur_annual[lam] = accum["total_imported_all"][lam] / accum["total_surplus_all"][lam]
        else:
            sur_annual[lam] = 0.0

    # Donor/receiver
    annual_surplus = accum["surplus"]
    annual_deficit = acc_demand - accum["self_consumed"]
    net_position = annual_surplus - annual_deficit
    donor_mask = net_position > 0
    receiver_mask = net_position < 0

    # ── Build annual summary DataFrame ──
    summary_data = {
        "mesh_code": mesh_codes,
        "alpha_label": scenario_name,
        "alpha": alpha,
        "category_3": category_3,
        "urbanization_index": urban_idx,
        "unit_type": base_idx.loc[mesh_codes]["unit_type"].values,
        "annual_gen_kwh": potential * alpha,
        "annual_dem_kwh": annual_dem,
        "annual_surplus_kwh": annual_surplus,
        "annual_deficit_kwh": annual_deficit,
        "is_donor": donor_mask,
        "is_receiver": receiver_mask,
        "ssr": ssr_annual,
    }
    for lam in ALL_LAMBDAS:
        lam_str = f"{lam:.1f}" if isinstance(lam, float) else lam
        summary_data[f"cssr_{lam_str}"] = cssr_annual[lam]
        summary_data[f"cpi_{lam_str}"] = cpi_annual[lam]

    summary_df = pd.DataFrame(summary_data)
    out_path = SUMMARY_DIR / f"cssr_summary_{scenario_name}.parquet"
    summary_df.to_parquet(out_path, index=False)
    log(f"  [OK] {out_path.name}: {summary_df.shape}")

    # ── Build daily CSSR ──
    daily_data = {"mesh_code": mesh_codes}
    for lam in DAILY_LAMBDAS:
        lam_str = f"{lam:.1f}" if isinstance(lam, float) else lam
        daily_cssr = np.where(
            accum["daily_demand"] > 0,
            accum["daily_served"][lam] / accum["daily_demand"],
            0.0,
        )
        for d in range(n_days_in_year):
            daily_data[f"cssr_{lam_str}_day{d+1:03d}"] = daily_cssr[:, d]

    daily_df = pd.DataFrame(daily_data)
    out_path = DAILY_DIR / f"cssr_daily_{scenario_name}.parquet"
    daily_df.to_parquet(out_path, index=False)
    log(f"  [OK] {out_path.name}: {daily_df.shape}")

    # ── Category × λ summary ──
    cpi_cat_rows = []
    for lam in ALL_LAMBDAS:
        lam_str = f"{lam:.1f}" if isinstance(lam, float) else lam
        cpi = cpi_annual[lam]
        cssr = cssr_annual[lam]
        imported_arr = accum["imported"][lam]

        for cat in cats:
            mask = category_3 == cat
            if not mask.any():
                continue
            cpi_cat = cpi[mask]
            cssr_cat = cssr[mask]
            ssr_cat = ssr_annual[mask]
            donor_frac = donor_mask[mask].mean()

            cpi_cat_rows.append({
                "alpha_label": scenario_name,
                "alpha": alpha,
                "lambda_km": lam if isinstance(lam, float) else np.inf,
                "lambda_label": lam_str,
                "category_3": cat,
                "n_meshes": int(mask.sum()),
                "ssr_mean": float(ssr_cat.mean()),
                "ssr_median": float(np.median(ssr_cat)),
                "cssr_mean": float(cssr_cat.mean()),
                "cssr_median": float(np.median(cssr_cat)),
                "cpi_mean": float(cpi_cat.mean()),
                "cpi_median": float(np.median(cpi_cat)),
                "cpi_std": float(cpi_cat.std()),
                "cpi_gini": float(gini(cpi_cat)),
                "ssr_gini": float(gini(ssr_cat)),
                "cssr_gini": float(gini(cssr_cat)),
                "donor_fraction": float(donor_frac),
                "receiver_fraction": float(1 - donor_frac),
                "total_surplus_kwh": float(annual_surplus[mask].sum()),
                "total_deficit_kwh": float(annual_deficit[mask].sum()),
                "total_imported_kwh": float(imported_arr[mask].sum()),
                "sur": float(sur_annual[lam]),
            })

    cpi_cat_df = pd.DataFrame(cpi_cat_rows)
    out_path = SUMMARY_DIR / f"cpi_by_category3_{scenario_name}.parquet"
    cpi_cat_df.to_parquet(out_path, index=False)

    # ── λ sensitivity summary ──
    lambda_rows = []
    for lam in ALL_LAMBDAS:
        lam_str = f"{lam:.1f}" if isinstance(lam, float) else lam
        cpi = cpi_annual[lam]
        cssr = cssr_annual[lam]
        lambda_rows.append({
            "alpha_label": scenario_name,
            "alpha": alpha,
            "lambda_km": lam if isinstance(lam, float) else np.inf,
            "lambda_label": lam_str,
            "cssr_mean": float(cssr.mean()),
            "cssr_median": float(np.median(cssr)),
            "cpi_mean": float(cpi.mean()),
            "cpi_median": float(np.median(cpi)),
            "cpi_std": float(cpi.std()),
            "cpi_p10": float(np.percentile(cpi, 10)),
            "cpi_p90": float(np.percentile(cpi, 90)),
            "cpi_gini": float(gini(cpi)),
            "ssr_mean": float(ssr_annual.mean()),
            "donor_fraction": float(donor_mask.mean()),
            "sur": float(sur_annual[lam]),
            "total_surplus_kwh": float(annual_surplus.sum()),
            "total_imported_kwh": float(accum["imported"][lam].sum()),
        })
    lambda_sensitivity_df = pd.DataFrame(lambda_rows)
    out_path = SUMMARY_DIR / f"lambda_sensitivity_{scenario_name}.csv"
    lambda_sensitivity_df.to_csv(out_path, index=False)

    # ── 30-min category profiles ──
    profile_rows = []
    for lam in ALL_LAMBDAS:
        lam_str = f"{lam:.1f}" if isinstance(lam, float) else lam
        for ci, cat in enumerate(cats):
            for hh in range(48):
                total_dem_hh = accum["cat_dem"][ci, hh]
                total_served_hh = accum["cat_served"][lam][ci, hh]
                cssr_val = total_served_hh / total_dem_hh if total_dem_hh > 0 else 0.0
                profile_rows.append({
                    "alpha_label": scenario_name,
                    "alpha": alpha,
                    "lambda_km": lam if isinstance(lam, float) else np.inf,
                    "lambda_label": lam_str,
                    "category_3": cat,
                    "halfhour_bin": int(hh),
                    "gen_kw": float(accum["cat_gen"][ci, hh] / n_days_in_year),
                    "dem_kw": float(accum["cat_dem"][ci, hh] / n_days_in_year),
                    "served_kw": float(total_served_hh / n_days_in_year),
                    "self_consumed_kw": float(accum["cat_self_consumed"][ci, hh] / n_days_in_year),
                    "cssr": float(cssr_val),
                })
    profiles_df = pd.DataFrame(profile_rows)
    out_path = SUMMARY_DIR / f"category_profiles_30min_{scenario_name}.parquet"
    profiles_df.to_parquet(out_path, index=False)

    # ── Accumulate trajectory rows ──
    for _, row in lambda_sensitivity_df.iterrows():
        trajectory_rows.append(row.to_dict())

    # ── Quick validation per scenario ──
    log(f"\n  ── Validation ({scenario_name}) ──")
    lam_display = WEIGHT_LAMBDAS[0] if WEIGHT_LAMBDAS else PRIMARY_LAMBDA
    lam_first = lam_display
    cpi_first = cpi_annual[lam_first]
    log(f"  CPI(λ={lam_first}): mean={cpi_first.mean():.6f}, median={np.median(cpi_first):.6f}")
    log(f"  SSR mean: {ssr_annual.mean():.4f}")
    log(f"  CSSR(λ={PRIMARY_LAMBDA}) mean: {cssr_annual[PRIMARY_LAMBDA].mean():.4f}")
    log(f"  CPI(λ={PRIMARY_LAMBDA}) mean: {cpi_annual[PRIMARY_LAMBDA].mean():.4f}")
    log(f"  Donors: {donor_mask.sum():,} ({donor_mask.mean()*100:.1f}%), "
        f"Receivers: {receiver_mask.sum():,} ({receiver_mask.mean()*100:.1f}%)")
    log(f"  SUR(λ={PRIMARY_LAMBDA}): {sur_annual[PRIMARY_LAMBDA]:.4f}")
    log(f"  Nighttime skip: {accum['skipped_steps']/max(1,accum['total_steps'])*100:.1f}%")

# ── Save trajectory summary (cross-scenario) ──
trajectory_df = pd.DataFrame(trajectory_rows)
out_path = SUMMARY_DIR / "trajectory_summary.parquet"
trajectory_df.to_parquet(out_path, index=False)
log(f"\n  [OK] trajectory_summary.parquet: {trajectory_df.shape}")

# ─── 8. Print Results Tables ───────────────────────────────────────
log(f"\n{'='*70}")
log("RESULTS SUMMARY (Multi-Scenario Trajectory)")
log("=" * 70)

# System-level CPI across scenarios
log(f"\nSystem-level CPI(λ={PRIMARY_LAMBDA}) across technical-potential scenarios:")
log(f"{'alpha_label':>11s}  {'alpha':>6s}  {'SSR':>8s}  {'CSSR':>8s}  {'CPI':>8s}  "
    f"{'SUR':>8s}  {'Donor%':>8s}  {'Gini':>8s}  {'Skip%':>8s}")
log("-" * 90)
for scenario_name in ALPHA_SCENARIOS:
    accum = accumulators[scenario_name]
    if accum["active_steps"] == 0:
        continue
    alpha = ALPHA_SCENARIOS[scenario_name]
    acc_demand = accum["demand"]
    ssr_val = np.where(acc_demand > 0, accum["self_consumed"] / acc_demand, 0.0).mean()
    cssr_val = np.where(acc_demand > 0, accum["served"][PRIMARY_LAMBDA] / acc_demand, 0.0).mean()
    cpi_val = cssr_val - ssr_val
    sur_val = (accum["total_imported_all"][PRIMARY_LAMBDA] / accum["total_surplus_all"][PRIMARY_LAMBDA]
               if accum["total_surplus_all"][PRIMARY_LAMBDA] > 0 else 0.0)
    n_donor = (accum["surplus"] > (acc_demand - accum["self_consumed"])).sum()
    donor_pct = n_donor / n_meshes * 100
    skip_pct = accum["skipped_steps"] / max(1, accum["total_steps"]) * 100
    log(f"{scenario_name:>11s}  {alpha:6.2f}  {ssr_val:8.4f}  {cssr_val:8.4f}  "
        f"{cpi_val:8.4f}  {sur_val:8.4f}  {donor_pct:7.1f}%  "
        f"{gini(np.where(acc_demand > 0, accum['served'][PRIMARY_LAMBDA]/acc_demand, 0.0) - np.where(acc_demand > 0, accum['self_consumed']/acc_demand, 0.0)):8.4f}  "
        f"{skip_pct:7.1f}%")

# CPI by category_3 across scenarios
log(f"\nCPI(λ={PRIMARY_LAMBDA}) by category_3:")
log(f"{'alpha_label':>11s}  {'alpha':>6s}  {'metro CPI':>10s}  {'regional CPI':>10s}  "
    f"{'rural CPI':>10s}  {'metro Donor%':>12s}  {'rural Donor%':>12s}")
log("-" * 86)
for scenario_name in ALPHA_SCENARIOS:
    accum = accumulators[scenario_name]
    if accum["active_steps"] == 0:
        continue
    alpha = ALPHA_SCENARIOS[scenario_name]
    acc_demand = accum["demand"]
    ssr_a = np.where(acc_demand > 0, accum["self_consumed"] / acc_demand, 0.0)
    cssr5 = np.where(acc_demand > 0, accum["served"][PRIMARY_LAMBDA] / acc_demand, 0.0)
    cpi5 = cssr5 - ssr_a

    vals = {}
    for cat in ["metropolitan_core", "regional_city", "rural"]:
        mask = category_3 == cat
        if mask.any():
            vals[f"{cat}_cpi"] = cpi5[mask].mean()
            vals[f"{cat}_donor"] = ((accum["surplus"] > (acc_demand - accum["self_consumed"]))[mask]).mean() * 100

    log(f"{scenario_name:>11s}  {alpha:6.2f}  "
        f"{vals.get('metropolitan_core_cpi', float('nan')):10.4f}  "
        f"{vals.get('regional_city_cpi', float('nan')):10.4f}  "
        f"{vals.get('rural_cpi', float('nan')):10.4f}  "
        f"{vals.get('metropolitan_core_donor', float('nan')):11.1f}%  "
        f"{vals.get('rural_donor', float('nan')):11.1f}%")

total_elapsed = time.time() - t0_total
log(f"\n{'='*70}")
log(f"Simulation complete. Total elapsed: {total_elapsed/60:.1f} min")
log(f"Outputs saved to: {OUT_DIR}")
log(f"{'='*70}")

# ─── 9. Run Manifest ────────────────────────────────────────────────
log(f"\n[9/9] Writing run manifest and energy conservation audit...")

# Energy conservation audit
_audit_rows = []
for scenario_name in ALPHA_SCENARIOS:
    accum = accumulators[scenario_name]
    if accum["active_steps"] == 0:
        continue
    total_dem = accum["demand"].sum()
    total_gen = potential.sum() * ALPHA_SCENARIOS[scenario_name] * (accum["total_steps"] / accum["total_steps"])  # approximate
    # More accurate: sum across all steps
    for lam in ALL_LAMBDAS:
        _served_sum = accum["served"][lam].sum()
        _imported_sum = accum["imported"][lam].sum()
        _self_consumed_sum = accum["self_consumed"].sum()
        _surplus_sum = accum["surplus"].sum()
        _total_surplus_all = accum["total_surplus_all"][lam]
        _total_imported_all = accum["total_imported_all"][lam]

        # Conservation: served = self_consumed + imported
        _served_from_parts = _self_consumed_sum + _imported_sum
        _served_error = abs(_served_sum - _served_from_parts)
        _served_error_rel = _served_error / max(1.0, _served_sum)

        # Conservation: imported ≤ deficit
        _deficit_sum = accum["demand"].sum() - _self_consumed_sum

        # Conservation: sum(imported) ≤ sum(surplus)
        _imported_le_surplus = _total_imported_all <= _total_surplus_all + 1e-9

        _audit_rows.append({
            "alpha_label": scenario_name,
            "alpha": ALPHA_SCENARIOS[scenario_name],
            "lambda_label": str(lam),
            "total_demand_kwh": float(total_dem),
            "total_self_consumed_kwh": float(_self_consumed_sum),
            "total_imported_kwh": float(_imported_sum),
            "total_surplus_kwh": float(_surplus_sum),
            "total_surplus_all_kwh": float(_total_surplus_all),
            "total_imported_all_kwh": float(_total_imported_all),
            "served_closure_error_kwh": float(_served_error),
            "served_closure_error_rel": float(_served_error_rel),
            "imported_le_surplus": bool(_imported_le_surplus),
            "transfer_twh": float(_total_imported_all / 1e9),
        })

audit_df = pd.DataFrame(_audit_rows)
audit_path = OUT_DIR / "audits" / "energy_conservation.csv"
audit_path.parent.mkdir(parents=True, exist_ok=True)
audit_df.to_csv(audit_path, index=False)
log(f"  [OK] {audit_path}: {audit_df.shape}")

# Energy conservation assertion
for _, row in audit_df.iterrows():
    assert row["served_closure_error_rel"] < 1e-9, \
        f"STOP: Energy conservation failed for {row['alpha_label']}/{row['lambda_label']}: error={row['served_closure_error_kwh']:.3f}"
    # Float64 accumulation tolerance: ~1e-16 relative over 17,568 steps
    _imp = row["total_imported_all_kwh"]
    _sur = row["total_surplus_all_kwh"]
    _ok = _imp <= _sur + max(1e-6, _sur * 1e-12)
    if not _ok:
        log(f"  ⚠ imported={_imp:.15g} > surplus={_sur:.15g} "
            f"(diff={_imp-_sur:.2e}) for {row['alpha_label']}/{row['lambda_label']} — "
            f"checking if float64 artifact...")
    assert _ok, (
        f"STOP: imported > surplus for {row['alpha_label']}/{row['lambda_label']}: "
        f"imported={_imp:.15g}, surplus={_sur:.15g}, diff={_imp-_sur:.2e}"
    )
log("  ✓ Energy conservation: served = self_consumed + imported (all scenarios)")
log("  ✓ Imported ≤ surplus (all scenarios)")

# Build run manifest
_sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else "MISSING"

manifest = {
    "run_id": f"{MODE}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "mode": MODE,
    "config": {
        "radius_km": RADIUS_KM,
        "decay_km": [lam if isinstance(lam, str) else lam for lam in ALL_LAMBDAS],
        "alpha_scenarios": list(ALPHA_SCENARIOS.keys()),
        "alpha_levels": list(ALPHA_SCENARIOS.values()),
        "mesh_support_n": n_meshes,
        "year": 2024,
        "timestep": "30min",
        "allocation": "conditional_donor_normalized",
        "transshipment": False,
        "reallocation_after_receiver_cap": False,
    },
    "graph_audit": _audit,
    "category_distribution": {str(c): int((category_3 == c).sum()) for c in cats},
    "input_sha256": {
        "technical_potential": _sha(str(PROJ_ROOT / "data" / "input" / "potential" / "technical_potential.parquet")),
        "mesh_attributes": _sha(str(DATA_DIR / "mesh_attributes.parquet")),
        "sparse_distances": _sha(str(DATA_DIR / "sparse_distances.parquet")),
    },
    "formula": {
        "dem_norm": DEM_NORM,
        "gen": "potential_kwh × gen_shape × alpha",
        "dem": "annual_dem_kwh × dem_shape / DEM_NORM",
        "surplus": "max(gen - dem, 0)",
        "deficit": "max(dem - gen, 0)",
        "inter_community_sharing": "conditional_donor_normalized(W_raw, surplus, deficit)",
        "served": "min(gen, dem) + imported",
    },
    "stop_conditions_checked": [
        "distance_km ≤ radius_km (R-filtered graph)",
        "energy_conservation: served = self_consumed + imported",
        "imported_le_surplus: sum(imported) ≤ sum(surplus)",
    ],
    "output_files": sorted([str(p.relative_to(OUT_DIR)) for p in OUT_DIR.rglob("*") if p.is_file()]),
}

manifest_path = MANIFEST_DIR / f"run_manifest_{MODE}.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
log(f"  [OK] {manifest_path}")
log(f"  Run ID: {manifest['run_id']}")
log(f"  R={RADIUS_KM} km, λ={ALL_LAMBDAS}, α={list(ALPHA_SCENARIOS.keys())}")
log(f"{'='*70}")
