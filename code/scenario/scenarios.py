# -*- coding: utf-8 -*-
"""Urban-form counterfactuals — build scenario geometry, then run the annual
inter-community energy-sharing simulation.

Stage order: run_build() -> run_simulate().
"""





def run_build():
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import sys

    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    PROJ_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJ_ROOT / "data" / "input" / "mesh"
    sys.path.insert(0, str(PROJ_ROOT / "code" / "simulation"))
    from formal_input_contract import load_technical_potential
    OUT_DIR = Path(__file__).resolve().parent / "results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("Build: urban form scenario geometry")
    print("=" * 70)

    # ── 1. Load baseline ──
    print("\n[1/5] Loading baseline data...")
    arv2, formal_potential_manifest = load_technical_potential()
    base = pd.read_parquet(DATA_DIR / "mesh_attributes.parquet")

    # Merge to get all needed columns
    df = arv2[["mesh_code", "potential_kwh", "annual_dem_kwh", "category_3",
               "urbanization_index", "lon", "lat"]].merge(
        base[["mesh_code", "avg_floors", "sr_ceiling", "building_density",
              "floor_area_ratio", "unit_type", "parent_muni_code",
              "parent_municipality", "annual_gen_kwh"]],
        on="mesh_code", how="inner"
    )
    df = df[df["annual_dem_kwh"] > 0].copy()
    n = len(df)
    print(f"  Baseline meshes: {n}")

    # Extract arrays
    mesh_codes = df["mesh_code"].values
    avg_floors_orig = df["avg_floors"].values.astype(np.float64)
    sr_ceiling_orig = df["sr_ceiling"].values.astype(np.float64)
    gen_orig = df["potential_kwh"].values.astype(np.float64)      # 100% potential
    dem_orig = df["annual_dem_kwh"].values.astype(np.float64)
    category_3 = df["category_3"].values
    lon = df["lon"].values
    lat = df["lat"].values

    print(f"  gen_orig total: {gen_orig.sum()/1e9:.1f} TWh")
    print(f"  dem_orig total: {dem_orig.sum()/1e9:.1f} TWh")
    print(f"  sr_ceiling: mean={sr_ceiling_orig.mean():.4f}, "
          f"median={np.median(sr_ceiling_orig):.4f}")
    for cat in ["metropolitan_core", "regional_city", "rural"]:
        mask = category_3 == cat
        if mask.any():
            print(f"  {cat}: n={mask.sum()}, sr_ceiling median={np.median(sr_ceiling_orig[mask]):.4f}, "
                  f"avg_floors median={np.median(avg_floors_orig[mask]):.1f}")

    # ── 2. Define scenario transforms ──
    print("\n[2/5] Computing scenario geometries...")

    scenarios = {}

    # ── S0: Status Quo ──
    scenarios["S0"] = {
        "label": "Status Quo",
        "gen_kwh": gen_orig.copy(),
        "dem_kwh": dem_orig.copy(),
        "avg_floors": avg_floors_orig.copy(),
        "sr_ceiling": sr_ceiling_orig.copy(),
    }

    # ── S1: Compact Vertical City (Morphology + BIPV) ──
    # metropolitan_core, avg_floors >= 5: grow upward x1.5, cap at 30F
    # BIPV compensates: gen_roof_loss × 1.3
    # Literature: Tokyo 2040, Hong Kong TA, Li et al. 2025
    print("\n  S1 — Compact Vertical City (morphology + BIPV):")
    floors_s1 = avg_floors_orig.copy()
    gen_s1 = gen_orig.copy()

    mask_s1 = (category_3 == "metropolitan_core") & (avg_floors_orig >= 5)
    floors_new_s1 = np.minimum(avg_floors_orig[mask_s1] * 1.5, 30.0)
    floors_s1[mask_s1] = floors_new_s1

    # gen: roof area ∝ 1/floors, BIPV ×1.3
    gen_s1[mask_s1] = gen_orig[mask_s1] * (avg_floors_orig[mask_s1] / floors_new_s1) * 1.3
    # dem: ∝ floors
    dem_s1 = dem_orig * (floors_s1 / avg_floors_orig)
    dem_s1 = np.maximum(dem_s1, 1.0)
    sr_s1 = 1.0 / np.maximum(floors_s1, 1.0)

    n_s1 = mask_s1.sum()
    scenarios["S1"] = {
        "label": "Compact Vertical City",
        "gen_kwh": gen_s1, "dem_kwh": dem_s1,
        "avg_floors": floors_s1, "sr_ceiling": sr_s1,
        "n_modified": n_s1,
    }
    print(f"    {n_s1} meshes modified (metro_core, ≥5F)")
    if n_s1 > 0:
        print(f"    floors: {avg_floors_orig[mask_s1].mean():.1f} → {floors_s1[mask_s1].mean():.1f}")
        print(f"    gen factor: {(gen_s1[mask_s1]/gen_orig[mask_s1]).mean():.2f}")
        print(f"    dem factor: {(dem_s1[mask_s1]/dem_orig[mask_s1]).mean():.2f}")

    # ── S1m: Compact Vertical City (Morphology only, no BIPV) ──
    # Same floor changes as S1, but PV multiplier = 1.0
    # Isolates pure morphology effect of vertical densification
    print("\n  S1m — Compact Vertical City (morphology only):")
    floors_s1m = floors_s1.copy()  # same floor changes
    gen_s1m = gen_orig.copy()
    gen_s1m[mask_s1] = gen_orig[mask_s1] * (avg_floors_orig[mask_s1] / floors_new_s1)  # no BIPV boost
    dem_s1m = dem_s1.copy()  # same demand
    sr_s1m = sr_s1.copy()
    scenarios["S1m"] = {
        "label": "Compact Vertical (morphology only)",
        "gen_kwh": gen_s1m, "dem_kwh": dem_s1m,
        "avg_floors": floors_s1m, "sr_ceiling": sr_s1m,
        "n_modified": n_s1,
    }
    print(f"    gen factor: {(gen_s1m[mask_s1]/gen_orig[mask_s1]).mean():.2f} (vs S1: {(gen_s1[mask_s1]/gen_orig[mask_s1]).mean():.2f})")

    # ── S2: TOD Polycentric Network ──
    # Density redistribution along rail corridors — no new construction.
    # Four spatial layers:
    #   A. Tokyo Core (<15km from Tokyo Station): de-densify 15%, BIPV ×1.4
    #   B. Secondary Nodes (<5km from 11 major stations): target 4-6F, PV ×1.5
    #   C. Transit Corridors (<2km from rail lines, non-A/B): cap >5F→5F, PV ×1.3
    #   D. Peripheral: >3F reduce 10%, PV ×1.2
    print("\n  S2 — TOD Polycentric Network:")

    def dist_km(lon1, lat1, lon2, lat2):
        lat_mid = (np.asarray(lat1) + lat2) / 2
        dx = (np.asarray(lon1) - lon2) * 111 * np.cos(np.radians(lat_mid))
        dy = (np.asarray(lat1) - lat2) * 111
        return np.sqrt(dx**2 + dy**2)

    # A. Tokyo Core
    TOKYO_STN = (139.7671, 35.6812)
    d_tokyo = dist_km(lon, lat, TOKYO_STN[0], TOKYO_STN[1])
    mask_tokyo_core = d_tokyo < 15

    # B. Secondary nodes
    SECONDARY = {
        "Yokohama": (139.622, 35.443), "Omiya": (139.624, 35.906),
        "Chiba": (140.114, 35.613), "Tachikawa": (139.414, 35.698),
        "Hachioji": (139.339, 35.656), "Kawasaki": (139.697, 35.531),
        "Kashiwa": (139.971, 35.862), "Machida": (139.448, 35.545),
        "Fujisawa": (139.487, 35.339), "Kawagoe": (139.483, 35.907),
        "Odawara": (139.156, 35.256),
    }
    d_secondary = np.full(n, np.inf)
    for slon, slat in SECONDARY.values():
        d = dist_km(lon, lat, slon, slat)
        d_secondary = np.minimum(d_secondary, d)
    mask_secondary = d_secondary < 5

    # C. Transit corridors: approximate with distance to major station ring
    # (Full rail-line buffer needs rail/station GIS data — use midpoint distances as proxy)
    # Use distance bands from Tokyo + secondary nodes
    mask_corridor = (d_tokyo < 30) & (~mask_tokyo_core) & (~mask_secondary)

    # D. Peripheral
    mask_peripheral = ~(mask_tokyo_core | mask_secondary | mask_corridor)

    floors_s2 = avg_floors_orig.copy()
    gen_s2 = gen_orig.copy()

    # A: de-densify 15%, BIPV ×1.4
    floors_s2[mask_tokyo_core] = np.maximum(avg_floors_orig[mask_tokyo_core] * 0.85, 2.0)
    gen_s2[mask_tokyo_core] = (gen_orig[mask_tokyo_core]
                               * (avg_floors_orig[mask_tokyo_core] / floors_s2[mask_tokyo_core])
                               * 1.4)

    # B: target 4-6F, PV ×1.5
    nf_b = avg_floors_orig[mask_secondary]
    fn_b = nf_b.copy()
    fn_b[nf_b < 4] = 4.0
    fn_b[nf_b > 6] = 6.0
    floors_s2[mask_secondary] = fn_b
    gen_s2[mask_secondary] = (gen_orig[mask_secondary]
                              * (nf_b / fn_b)
                              * 1.5)

    # C: cap >5F→5F, PV ×1.3
    corr_mod = mask_corridor & (avg_floors_orig > 5)
    floors_s2[corr_mod] = 5.0
    gen_s2[mask_corridor] = (gen_orig[mask_corridor]
                             * (avg_floors_orig[mask_corridor] / floors_s2[mask_corridor])
                             * 1.3)

    # D: >3F reduce 10%, PV ×1.2
    periph_mod = mask_peripheral & (avg_floors_orig > 3)
    floors_s2[periph_mod] = avg_floors_orig[periph_mod] * 0.9
    gen_s2[mask_peripheral] = (gen_orig[mask_peripheral]
                               * (avg_floors_orig[mask_peripheral] / floors_s2[mask_peripheral])
                               * 1.2)

    dem_s2 = dem_orig * (floors_s2 / avg_floors_orig)
    dem_s2 = np.maximum(dem_s2, 1.0)
    sr_s2 = 1.0 / np.maximum(floors_s2, 1.0)

    n_s2_mod = (floors_s2 != avg_floors_orig).sum()
    scenarios["S2"] = {
        "label": "TOD Polycentric Network",
        "gen_kwh": gen_s2, "dem_kwh": dem_s2,
        "avg_floors": floors_s2, "sr_ceiling": sr_s2,
        "n_modified": n_s2_mod,
        "_mask_tokyo": mask_tokyo_core,
        "_mask_secondary": mask_secondary,
        "_mask_corridor": mask_corridor,
        "_mask_peripheral": mask_peripheral,
    }
    print(f"    {n_s2_mod} meshes modified")
    for label, mask in [("Tokyo Core", mask_tokyo_core), ("Secondary Nodes", mask_secondary),
                         ("Transit Corridors", mask_corridor), ("Peripheral", mask_peripheral)]:
        m = mask.sum()
        if m > 0:
            print(f"    {label:20s}: {m:5d} meshes, "
                  f"floors {avg_floors_orig[mask].mean():.1f}→{floors_s2[mask].mean():.1f}, "
                  f"gen ×{(gen_s2[mask]/gen_orig[mask]).mean():.2f}")

    # ── S2m: TOD Polycentric Network (Morphology only, no PV boost) ──
    # Same floor changes as S2, but all PV multipliers = 1.0
    # Isolates pure density redistribution effect
    print("\n  S2m — TOD Polycentric Network (morphology only):")
    floors_s2m = floors_s2.copy()
    gen_s2m = gen_orig.copy()

    gen_s2m[mask_tokyo_core] = (gen_orig[mask_tokyo_core]
                                * (avg_floors_orig[mask_tokyo_core] / floors_s2m[mask_tokyo_core]))  # no ×1.4
    gen_s2m[mask_secondary] = (gen_orig[mask_secondary]
                               * (avg_floors_orig[mask_secondary] / floors_s2m[mask_secondary]))      # no ×1.5
    gen_s2m[corr_mod] = (gen_orig[corr_mod]
                         * (avg_floors_orig[corr_mod] / floors_s2m[corr_mod]))                         # no ×1.3
    gen_s2m[periph_mod] = (gen_orig[periph_mod]
                           * (avg_floors_orig[periph_mod] / floors_s2m[periph_mod]))                   # no ×1.2
    # For corridor/peripheral meshes with no floor change, gen stays at orig level
    gen_s2m[mask_corridor & ~corr_mod] = gen_orig[mask_corridor & ~corr_mod]
    gen_s2m[mask_peripheral & ~periph_mod] = gen_orig[mask_peripheral & ~periph_mod]

    dem_s2m = dem_s2.copy()
    sr_s2m = sr_s2.copy()
    scenarios["S2m"] = {
        "label": "TOD Polycentric (morphology only)",
        "gen_kwh": gen_s2m, "dem_kwh": dem_s2m,
        "avg_floors": floors_s2m, "sr_ceiling": sr_s2m,
        "n_modified": n_s2_mod,
    }
    print(f"    {n_s2_mod} meshes modified (same floor changes as S2, no PV boost)")
    for label, mask in [("Tokyo Core", mask_tokyo_core), ("Secondary Nodes", mask_secondary),
                         ("Transit Corridors", mask_corridor), ("Peripheral", mask_peripheral)]:
        m = mask.sum()
        if m > 0:
            print(f"    {label:20s}: {m:5d} meshes, "
                  f"gen ×{(gen_s2m[mask]/gen_orig[mask]).mean():.2f} (vs S2: ×{(gen_s2[mask]/gen_orig[mask]).mean():.2f})")

    # ── S3: Solar Community (mild) ──
    # metropolitan_core: >6F → 6F, PV ×1.4
    # regional_city: PV-integrated housing (gen ×1.2, no floor change)
    # rural: unchanged (already optimal)
    print("\n  S3 — Solar Community (mild):")
    floors_s3 = avg_floors_orig.copy()
    gen_s3 = gen_orig.copy()

    mask_s3_metro = (category_3 == "metropolitan_core") & (avg_floors_orig > 6)
    floors_s3[mask_s3_metro] = 6.0
    gen_s3[mask_s3_metro] = (gen_orig[mask_s3_metro]
                             * (avg_floors_orig[mask_s3_metro] / 6.0)
                             * 1.4)

    mask_s3_regional = category_3 == "regional_city"
    gen_s3[mask_s3_regional] = gen_orig[mask_s3_regional] * 1.2  # PV housing boost

    dem_s3 = dem_orig * (floors_s3 / avg_floors_orig)
    dem_s3 = np.maximum(dem_s3, 1.0)
    sr_s3 = 1.0 / np.maximum(floors_s3, 1.0)

    n_s3_metro_mod = mask_s3_metro.sum()
    n_s3_regional_mod = mask_s3_regional.sum()
    scenarios["S3"] = {
        "label": "Solar Community (mild)",
        "gen_kwh": gen_s3, "dem_kwh": dem_s3,
        "avg_floors": floors_s3, "sr_ceiling": sr_s3,
        "n_modified": n_s3_metro_mod + n_s3_regional_mod,
    }
    if n_s3_metro_mod > 0:
        print(f"    metro_core >6F→6F: {n_s3_metro_mod} meshes, "
              f"floors {avg_floors_orig[mask_s3_metro].mean():.1f}→6.0, "
              f"gen ×{(gen_s3[mask_s3_metro]/gen_orig[mask_s3_metro]).mean():.2f}")
    print(f"    regional_city PV housing: {n_s3_regional_mod} meshes, gen ×1.2")

    # ── S3m: Solar Community (Morphology only, no PV boost) ──
    # Same floor changes as S3, but all PV multipliers = 1.0
    # Isolates pure height cap effect
    print("\n  S3m — Solar Community (morphology only):")
    floors_s3m = floors_s3.copy()
    gen_s3m = gen_orig.copy()
    gen_s3m[mask_s3_metro] = (gen_orig[mask_s3_metro]
                              * (avg_floors_orig[mask_s3_metro] / 6.0))  # no ×1.4
    # regional_city: no floor change → gen unchanged
    gen_s3m[mask_s3_regional] = gen_orig[mask_s3_regional]  # no ×1.2
    dem_s3m = dem_s3.copy()
    sr_s3m = sr_s3.copy()
    scenarios["S3m"] = {
        "label": "Solar Community (morphology only)",
        "gen_kwh": gen_s3m, "dem_kwh": dem_s3m,
        "avg_floors": floors_s3m, "sr_ceiling": sr_s3m,
        "n_modified": n_s3_metro_mod,
    }
    if n_s3_metro_mod > 0:
        print(f"    metro_core >6F→6F: {n_s3_metro_mod} meshes, "
              f"gen ×{(gen_s3m[mask_s3_metro]/gen_orig[mask_s3_metro]).mean():.2f} "
              f"(vs S3: ×{(gen_s3[mask_s3_metro]/gen_orig[mask_s3_metro]).mean():.2f})")
    print(f"    regional_city: gen unchanged (no floor change, no PV boost)")

    # ── 2.5. Demand conservation post-processing ──
    # For each scenario, rescale demand to match S0 total while preserving
    # dem ∝ floors relationship. This ensures CPI changes reflect morphology
    # effects, not changes in aggregate demand.
    print("\n[2.5/5] Applying demand conservation (total demand = S0 baseline)...")
    S0_dem_total = scenarios["S0"]["dem_kwh"].sum()
    print(f"  S0 total demand: {S0_dem_total/1e9:.2f} TWh")

    for sn in ["S1", "S1m", "S2", "S2m", "S3", "S3m"]:
        sc = scenarios[sn]
        raw_dem_total = sc["dem_kwh"].sum()
        scale = S0_dem_total / raw_dem_total if raw_dem_total > 0 else 1.0
        sc["dem_kwh"] = sc["dem_kwh"] * scale
        sc["dem_kwh"] = np.maximum(sc["dem_kwh"], 1.0)
        print(f"  {sn}: raw={raw_dem_total/1e9:.1f} TWh → "
              f"conserved={sc['dem_kwh'].sum()/1e9:.1f} TWh (scale={scale:.4f})")

    # ── 3. Summary per scenario × category_3 ──
    print("\n[3/5] Scenario summary by category_3...")
    cats_report = ["metropolitan_core", "regional_city", "rural"]

    summary_rows = []
    for sn, sc in scenarios.items():
        for cat in cats_report:
            mask = category_3 == cat
            if not mask.any():
                continue
            row = {
                "urban_form_scenario": sn,
                "label": sc["label"],
                "category_3": cat,
                "n_meshes": int(mask.sum()),
                "sr_ceiling_mean": float(sc["sr_ceiling"][mask].mean()),
                "sr_ceiling_median": float(np.median(sc["sr_ceiling"][mask])),
                "avg_floors_mean": float(sc["avg_floors"][mask].mean()),
                "gen_total_twh": float(sc["gen_kwh"][mask].sum() / 1e9),
                "dem_total_twh": float(sc["dem_kwh"][mask].sum() / 1e9),
                "net_twh": float((sc["gen_kwh"][mask].sum() - sc["dem_kwh"][mask].sum()) / 1e9),
                "donor_pct": float((sc["gen_kwh"][mask] > sc["dem_kwh"][mask]).mean()),
            }
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "scenario_summary.csv", index=False)

    # ── 4. Save mesh-level geometry ──
    print("\n[4/5] Saving mesh-level geometry...")
    geo_cols = {
        "mesh_code": mesh_codes,
        "category_3": category_3,
        "unit_type": df["unit_type"].values,
        "parent_muni_code": df["parent_muni_code"].values,
        "parent_municipality": df["parent_municipality"].values,
        "urbanization_index": df["urbanization_index"].values,
        "lon": lon, "lat": lat,
    }
    for sn, sc in scenarios.items():
        geo_cols[f"gen_{sn}_kwh"] = sc["gen_kwh"]
        geo_cols[f"dem_{sn}_kwh"] = sc["dem_kwh"]
        geo_cols[f"avg_floors_{sn}"] = sc["avg_floors"]
        geo_cols[f"sr_ceiling_{sn}"] = sc["sr_ceiling"]

    geo_df = pd.DataFrame(geo_cols)
    geo_df.to_parquet(OUT_DIR / "scenario_geometry.parquet", index=False)
    print(f"  scenario_geometry.parquet: {len(geo_df)} meshes × {len(geo_df.columns)} cols")

    # ── 5. Sanity checks ──
    print("\n[5/5] Sanity checks...")
    print(f"\n  System totals:")
    for sn, sc in scenarios.items():
        g = sc["gen_kwh"].sum() / 1e9
        d = sc["dem_kwh"].sum() / 1e9
        net = g - d
        donor_pct = (sc["gen_kwh"] > sc["dem_kwh"]).mean() * 100
        print(f"  {sn} ({sc['label']:30s}): gen={g:.1f} TWh, dem={d:.1f} TWh, "
              f"net={net:+.1f} TWh, donors={donor_pct:.1f}%")

    print(f"\n  Key deltas vs S0 (by category_3, a100 gen/dem):")
    for sn in ["S1", "S1m", "S2", "S2m", "S3", "S3m"]:
        sc = scenarios[sn]
        print(f"  {sn} ({sc['label']}):")
        for cat in cats_report:
            mask = category_3 == cat
            if not mask.any():
                continue
            dg = (sc["gen_kwh"][mask].sum() - scenarios["S0"]["gen_kwh"][mask].sum()) / 1e9
            dd = (sc["dem_kwh"][mask].sum() - scenarios["S0"]["dem_kwh"][mask].sum()) / 1e9
            dsr = sc["sr_ceiling"][mask].mean() - scenarios["S0"]["sr_ceiling"][mask].mean()
            print(f"    {cat:25s}: Δgen={dg:+.1f} TWh, Δdem={dd:+.1f} TWh, Δsr_ceiling={dsr:+.4f}")

    print(f"\n{'='*70}")
    print("Build complete.")
    print(f"{'='*70}")



def run_simulate():
    import pandas as pd
    import numpy as np
    from scipy.sparse import csr_matrix
    from pathlib import Path
    import time
    import sys
    import calendar
    import json
    import glob
    from collections import OrderedDict

    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    PROJ_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJ_ROOT / "data" / "input" / "mesh"
    sys.path.insert(0, str(PROJ_ROOT / "code" / "simulation"))
    from formal_input_contract import load_technical_potential, validate_technical_potential
    STATIC_DATA_DIR = PROJ_ROOT / "data" / "input" / "shapes"  # non-coordinate ref files
    SIMULATION_DIR = PROJ_ROOT / "code" / "simulation"
    if str(SIMULATION_DIR) not in sys.path:
        sys.path.insert(0, str(SIMULATION_DIR))

    from allocation_rules import conditional_donor_normalized

    GEOMETRY_DIR = Path(__file__).resolve().parent / "results"
    OUT_DIR = Path(__file__).resolve().parent / "results"
    CKPT_DIR = OUT_DIR / "checkpoints"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ═══ Parameters ═══
    LAMBDAS_KM = [5.0]  # formal: R=5, λ=5 only (checklist §1.2)
    LAMBDA_INF = "inf"
    ALL_LAMBDAS = LAMBDAS_KM + [LAMBDA_INF]

    ALPHA_LEVELS = OrderedDict([
        ("tp015", 0.15),
        ("tp030", 0.30),
    ])

    # Full year (12 months)
    SIM_MONTHS = list(range(1, 13))

    # ── Scenario dictionary (demand rules — P0-E fix) ──
    # These parameters are frozen; do not change without author decision.
    # demand_total_rule: each scenario's total demand re-scaled to S0=290.20 TWh
    #   (see build_scenarios.py §2.5)
    # demand_redistribution_rule: demand redistributed proportionally to changed floors
    # demand_shape_rule: blend S0 per-mesh dem_shape toward urban/rural reference
    #   based on floor_ratio vs thresholds; blend weight = 0.30
    DEM_BLEND_ALPHA = 0.30         # blend weight toward reference shape
    FLOOR_RATIO_UP = 1.15           # taller threshold (> → blend toward urban)
    FLOOR_RATIO_DOWN = 0.85         # shorter threshold (< → blend toward rural)


    def gini(x):
        x = np.asarray(x, dtype=np.float64)
        if len(x) == 0 or x.sum() == 0:
            return 0.0
        x_sorted = np.sort(x)
        n = len(x)
        return (2 * np.sum(np.arange(1, n + 1) * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n


    def log(msg):
        print(msg, flush=True)


    # ══════════════════════════════════════════════════════════════════════
    log("=" * 70)
    log("Inter-community energy sharing simulation: urban form scenarios (annual, α=15%/30%)")
    log("=" * 70)
    t_total = time.time()

    # ── 1. Load data ──
    log("\n[1/7] Loading data...")

    geo = pd.read_parquet(GEOMETRY_DIR / "scenario_geometry.parquet")
    arv2, formal_potential_manifest = load_technical_potential()
    base_meta = pd.read_parquet(
        DATA_DIR / "mesh_attributes.parquet",
        columns=["mesh_code", "unit_type"]
    )
    arv2 = arv2.merge(base_meta, on="mesh_code", how="inner")
    dem_shapes_all = pd.read_parquet(STATIC_DATA_DIR / "dem_shapes.parquet")
    season = pd.read_parquet(STATIC_DATA_DIR / "season_factors.parquet")
    dist = pd.read_parquet(DATA_DIR / "sparse_distances.parquet")

    season_map = dict(zip(season["month"], season["season_factor"]))

    # Full-year DEM_NORM
    DEM_NORM = sum(calendar.monthrange(2024, m)[1] * season_map[m] for m in SIM_MONTHS)
    log(f"  DEM_NORM (Annual) = {DEM_NORM:.4f}")

    # ── 2. Build unified mesh index ──
    log("\n[2/7] Building mesh index...")

    gen_files = sorted(glob.glob(str(STATIC_DATA_DIR / "gen_shapes" / "month=*.parquet")))
    gs_meshes = set()
    for gf in gen_files:
        gs_part = pd.read_parquet(gf, columns=["mesh_code"])
        gs_meshes.update(gs_part["mesh_code"].unique())

    geo_meshes = set(geo["mesh_code"].values)
    dem_meshes = set(dem_shapes_all["mesh_code"])
    dist_mesh_i = set(int(m) for m in arv2["mesh_code"].values[dist["mesh_idx_i"].values])
    dist_mesh_j = set(int(m) for m in arv2["mesh_code"].values[dist["mesh_idx_j"].values])
    dist_meshes = dist_mesh_i | dist_mesh_j

    common = sorted(geo_meshes & gs_meshes & dem_meshes & dist_meshes)
    n_meshes = len(common)
    formal_contract = validate_technical_potential(
        arv2, formal_potential_manifest, set(common)
    )
    log(f"  FORMAL POTENTIAL CONTRACT: {formal_contract['potential_kwh']/1e9:.6f} TWh")
    mesh_to_idx = {int(m): i for i, m in enumerate(common)}
    mesh_codes = np.array(common, dtype=np.int64)

    log(f"  geo: {len(geo_meshes)}  gen_shapes: {len(gs_meshes)}  "
        f"dem: {len(dem_meshes)}  dist: {len(dist_meshes)}  → common: {n_meshes}")

    # ── 3. Align arrays ──
    log("\n[3/7] Aligning arrays...")

    geo_idx = geo.set_index("mesh_code").loc[common]
    category_3 = geo_idx["category_3"].values
    cats = sorted(set(c for c in category_3 if c != "other"))

    # Scenario gen + dem arrays
    scenario_names = ["S0", "S1", "S1m", "S2", "S2m", "S3", "S3m"]
    scenario_gen_a100 = {}
    scenario_dem = {}
    scenario_floors = {}
    for sn in scenario_names:
        scenario_gen_a100[sn] = geo_idx[f"gen_{sn}_kwh"].values.astype(np.float64)
        scenario_dem[sn] = geo_idx[f"dem_{sn}_kwh"].values.astype(np.float64)
        scenario_floors[sn] = geo_idx[f"avg_floors_{sn}"].values.astype(np.float64)
    floors_s0 = scenario_floors["S0"]

    # Filter zero-demand meshes
    valid = np.ones(n_meshes, dtype=bool)
    for sn in scenario_names:
        valid &= (scenario_dem[sn] > 0)

    if not valid.all():
        n_bad = (~valid).sum()
        log(f"  WARNING: {n_bad} meshes with zero demand — filtering")
        valid_idx = np.where(valid)[0]
        common_valid = [common[i] for i in valid_idx]
        mesh_to_idx = {int(m): i for i, m in enumerate(common_valid)}
        mesh_codes = np.array(common_valid, dtype=np.int64)
        n_meshes = len(mesh_codes)
        geo_idx = geo.set_index("mesh_code").loc[common_valid]
        category_3 = geo_idx["category_3"].values
        for sn in scenario_names:
            scenario_gen_a100[sn] = scenario_gen_a100[sn][valid_idx]
            scenario_dem[sn] = scenario_dem[sn][valid_idx]
            scenario_floors[sn] = scenario_floors[sn][valid_idx]
        floors_s0 = scenario_floors["S0"]

    log(f"  Final mesh count: {n_meshes}")

    # ── 4. Build reference demand shapes + scenario-specific dem_lookup ──
    log("\n[4/7] Building scenario-specific demand shapes...")

    # Load per-mesh dem shapes (Full year (12 months), all day_types)
    dem_filtered = dem_shapes_all[
        dem_shapes_all["mesh_code"].isin(mesh_to_idx) &
        dem_shapes_all["month"].isin(SIM_MONTHS)
    ]

    # Build base dem_lookup: (month, day_type, slot) → array[n_meshes]
    base_dem_lookup = {}
    for (month, dt, slot), group in dem_filtered.groupby(["month", "day_type", "slot"]):
        arr = np.zeros(n_meshes, dtype=np.float64)
        for m, v in zip(group["mesh_code"].values, group["dem_shape"].values):
            if int(m) in mesh_to_idx:
                arr[mesh_to_idx[int(m)]] = v
        base_dem_lookup[(month, dt, slot)] = arr
    log(f"  base dem_lookup: {len(base_dem_lookup)} keys ({len(SIM_MONTHS)} months × 2 day_types × 48 slots)")

    # Build reference profiles from S0 floors + base dem shapes
    mask_urban_ref = (category_3 == "metropolitan_core") & (floors_s0 >= 6)
    mask_rural_ref = (category_3 == "rural") & (floors_s0 <= 2)

    log(f"  urban_ref meshes (metro ≥6F): {mask_urban_ref.sum()}")
    log(f"  rural_ref meshes (rural ≤2F): {mask_rural_ref.sum()}")

    urban_ref = {}
    rural_ref = {}
    for key, arr in base_dem_lookup.items():
        if mask_urban_ref.sum() > 0:
            urban_ref[key] = arr[mask_urban_ref].mean(axis=0)  # scalar
        else:
            urban_ref[key] = 0.0
        if mask_rural_ref.sum() > 0:
            rural_ref[key] = arr[mask_rural_ref].mean(axis=0)
        else:
            rural_ref[key] = 0.0

    # Build scenario-specific dem_lookup
    # For each scenario, determine blend per mesh based on floor ratio
    scenario_dem_lookups = {}

    for sn in scenario_names:
        floor_ratio = np.where(floors_s0 > 0,
                               scenario_floors[sn] / np.maximum(floors_s0, 0.5),
                               1.0)

        # Per-mesh blend weight: positive = toward urban, negative = toward rural
        blend_w = np.zeros(n_meshes, dtype=np.float64)
        blend_w[floor_ratio > FLOOR_RATIO_UP] = DEM_BLEND_ALPHA
        blend_w[floor_ratio < FLOOR_RATIO_DOWN] = -DEM_BLEND_ALPHA

        n_up = (floor_ratio > FLOOR_RATIO_UP).sum()
        n_down = (floor_ratio < FLOOR_RATIO_DOWN).sum()
        n_unchanged = n_meshes - n_up - n_down
        log(f"  {sn}: {n_up} taller (→urban), {n_down} shorter (→rural), "
            f"{n_unchanged} unchanged")

        sn_lookup = {}
        for key, base_arr in base_dem_lookup.items():
            arr = base_arr.copy()
            # Blend toward urban_ref for taller meshes
            if n_up > 0 and urban_ref[key] != 0:
                up_mask = blend_w > 0
                arr[up_mask] = (arr[up_mask] * (1 - DEM_BLEND_ALPHA)
                                + urban_ref[key] * DEM_BLEND_ALPHA)
            # Blend toward rural_ref for shorter meshes
            if n_down > 0 and rural_ref[key] != 0:
                down_mask = blend_w < 0
                arr[down_mask] = (arr[down_mask] * (1 - DEM_BLEND_ALPHA)
                                  + rural_ref[key] * DEM_BLEND_ALPHA)
            # Normalize: ensure sum within each (month, day_type) group is preserved
            # (blend preserves normalization since both ref and base sum to same value)
            sn_lookup[key] = arr

        scenario_dem_lookups[sn] = sn_lookup

    # ── 5. Build weight matrices ──
    log("\n[5/7] Building weight matrices...")

    orig_mesh_list = arv2["mesh_code"].values
    dist_i = dist["mesh_idx_i"].values.astype(np.int32)
    dist_j = dist["mesh_idx_j"].values.astype(np.int32)
    dist_d = dist["distance_km"].values.astype(np.float32)
    dist_mesh_i_map = orig_mesh_list[dist_i]
    dist_mesh_j_map = orig_mesh_list[dist_j]

    keep_i = np.array([int(m) in mesh_to_idx for m in dist_mesh_i_map])
    keep_j = np.array([int(m) in mesh_to_idx for m in dist_mesh_j_map])
    keep = keep_i & keep_j

    di_new = np.array([mesh_to_idx[int(m)] for m in dist_mesh_i_map[keep]], dtype=np.int32)
    dj_new = np.array([mesh_to_idx[int(m)] for m in dist_mesh_j_map[keep]], dtype=np.int32)
    dd_new = dist_d[keep]
    log(f"  Distance pairs (mesh-filtered): {len(dd_new):,}")

    # ── R=5 radius filter (P0-B fix) ──
    RADIUS_KM = 5.0
    radius_mask = dd_new <= RADIUS_KM
    di_r = di_new[radius_mask]
    dj_r = dj_new[radius_mask]
    dd_r = dd_new[radius_mask]
    log(f"  R=5 filter: {len(dd_r):,} pairs kept, "
        f"{len(dd_new)-len(dd_r):,} dropped ({(len(dd_new)-len(dd_r))/max(1,len(dd_new))*100:.1f}%)")
    assert dd_r.max() <= RADIUS_KM + 1e-9, f"STOP: edge > {RADIUS_KM} km found!"

    weight_matrices = {}
    for lam in LAMBDAS_KM:
        t_lam = time.time()
        w_raw = np.exp(-dd_r / lam).astype(np.float32)
        W_raw = csr_matrix((w_raw, (di_r, dj_r)), shape=(n_meshes, n_meshes), dtype=np.float32)
        weight_matrices[lam] = W_raw
        log(f"  λ={lam:.1f}: {W_raw.nnz:,} nnz ({time.time()-t_lam:.1f}s)")

    # ── 6. Define simulation runs ──
    log("\n[6/7] Setting up simulation runs...")

    SIM_RUNS = OrderedDict()

    # S0 × {tp015, tp030}
    for a_label, alpha in ALPHA_LEVELS.items():
        SIM_RUNS[f"S0_{a_label}"] = {
            "urban_form_scenario": "S0",
            "alpha_label": a_label,
            "gen": scenario_gen_a100["S0"] * alpha,
            "dem": scenario_dem["S0"],
            "dem_lookup": scenario_dem_lookups["S0"],
        }

    # S1, S2, S3 × {tp015, tp030} (morphology + PV)
    for sn in ["S1", "S2", "S3"]:
        for a_label, alpha in ALPHA_LEVELS.items():
            SIM_RUNS[f"{sn}_{a_label}"] = {
                "urban_form_scenario": sn,
                "alpha_label": a_label,
                "gen": scenario_gen_a100[sn] * alpha,
                "dem": scenario_dem[sn],
                "dem_lookup": scenario_dem_lookups[sn],
            }

    # S1m, S2m, S3m × {tp015, tp030} (pure morphology, no PV boost)
    # Share dem_lookup with parent scenario (same floors → same demand shapes)
    for sn, parent in [("S1m", "S1"), ("S2m", "S2"), ("S3m", "S3")]:
        for a_label, alpha in ALPHA_LEVELS.items():
            SIM_RUNS[f"{sn}_{a_label}"] = {
                "urban_form_scenario": sn,
                "alpha_label": a_label,
                "gen": scenario_gen_a100[sn] * alpha,
                "dem": scenario_dem[sn],
                "dem_lookup": scenario_dem_lookups[parent],
            }

    log(f"  Total runs: {len(SIM_RUNS)}")
    for name, cfg in SIM_RUNS.items():
        g = cfg["gen"].sum() / 1e9
        d = cfg["dem"].sum() / 1e9
        donors = (cfg["gen"] > cfg["dem"]).mean() * 100
        log(f"  {name:20s}  gen={g:.1f} TWh  dem={d:.1f} TWh  "
            f"gen/dem={g/d:.2f}  donors={donors:.1f}%")

    # ── 7. Main simulation loop ──
    log("\n[7/7] Running simulations (Full year (12 months))...")
    t0 = time.time()

    # Calendar setup (Full year (12 months))
    day_offset = {1: 0, 2: 31, 3: 60, 4: 91, 5: 121, 6: 152,
                  7: 182, 8: 213, 9: 244, 10: 274, 11: 305, 12: 335}
    day_type_lookup = {}
    for m in SIM_MONTHS:
        n_days = calendar.monthrange(2024, m)[1]
        for d in range(1, n_days + 1):
            doy = day_offset[m] + (d - 1)
            wd = calendar.weekday(2024, m, d)
            day_type_lookup[doy] = "we" if wd >= 5 else "wd"

    # Accumulators
    accum = {}
    for name in SIM_RUNS:
        accum[name] = {
            "demand": np.zeros(n_meshes, dtype=np.float64),
            "self_consumed": np.zeros(n_meshes, dtype=np.float64),
            "served": {lam: np.zeros(n_meshes, dtype=np.float64) for lam in ALL_LAMBDAS},
            "surplus_sum": np.zeros(n_meshes, dtype=np.float64),
            "imported": {lam: np.zeros(n_meshes, dtype=np.float64) for lam in ALL_LAMBDAS},
            "total_surplus": {lam: 0.0 for lam in ALL_LAMBDAS},
            "total_imported": {lam: 0.0 for lam in ALL_LAMBDAS},
            "steps": 0, "skipped": 0, "active": 0,
        }

    # Resume from checkpoints
    for name in SIM_RUNS:
        ckpt_path = CKPT_DIR / f"{name}_state.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                ckpt_state = json.load(f)
            completed = set(ckpt_state.get("completed_months", []))
            acc_npz = CKPT_DIR / f"{name}_accum.npz"
            if acc_npz.exists():
                ckpt = np.load(acc_npz)
                a = accum[name]
                a["demand"] = ckpt["demand"]
                a["self_consumed"] = ckpt["self_consumed"]
                a["surplus_sum"] = ckpt["surplus_sum"]
                for lam in ALL_LAMBDAS:
                    lk = f"served_{str(lam).replace('.', '_')}"
                    if lk in ckpt: a["served"][lam] = ckpt[lk]
                    lk2 = f"imported_{str(lam).replace('.', '_')}"
                    if lk2 in ckpt: a["imported"][lam] = ckpt[lk2]
                ckpt.close()
            sur_path = CKPT_DIR / f"{name}_sur.json"
            if sur_path.exists():
                with open(sur_path) as f:
                    sd = json.load(f)
                    a["total_surplus"] = {lam: sd["surplus"].get(str(lam), 0.0) for lam in ALL_LAMBDAS}
                    a["total_imported"] = {lam: sd["imported"].get(str(lam), 0.0) for lam in ALL_LAMBDAS}
            stats_path = CKPT_DIR / f"{name}_stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    st = json.load(f)
                    a["steps"] = st.get("steps", 0)
                    a["skipped"] = st.get("skipped", 0)
                    a["active"] = st.get("active", 0)
            log(f"  [{name}] Loaded checkpoint: months {sorted(completed)}")

    all_zero = np.zeros(n_meshes, dtype=np.float64)

    for month in SIM_MONTHS:
        t_month = time.time()

        # Load gen_shapes for this month
        gen_file = STATIC_DATA_DIR / "gen_shapes" / f"month={month:02d}.parquet"
        gen_month = pd.read_parquet(gen_file)
        gen_month = gen_month[gen_month["mesh_code"].isin(mesh_to_idx)]
        gen_month["mesh_idx"] = gen_month["mesh_code"].map(mesh_to_idx)
        gen_month = gen_month.dropna(subset=["mesh_idx"])
        gen_month["mesh_idx"] = gen_month["mesh_idx"].astype(np.int32)

        gen_by_ds = {}
        for (day, slot), group in gen_month.groupby(["day", "slot"]):
            arr = np.zeros(n_meshes, dtype=np.float64)
            idx = group["mesh_idx"].values
            val = group["gen_shape"].values
            arr[idx] = val
            gen_by_ds[(int(day), int(slot))] = arr

        n_days_in_month = calendar.monthrange(2024, month)[1]

        for run_name, cfg in SIM_RUNS.items():
            a = accum[run_name]

            # Skip completed months
            ckpt_path = CKPT_DIR / f"{run_name}_state.json"
            completed_months = set()
            if ckpt_path.exists():
                with open(ckpt_path) as f:
                    completed_months = set(json.load(f).get("completed_months", []))
            if month in completed_months:
                continue

            run_gen_annual = cfg["gen"]
            run_dem_annual = cfg["dem"]
            dem_lookup = cfg["dem_lookup"]

            for day in range(1, n_days_in_month + 1):
                doy = day_offset[month] + (day - 1)
                gen_doy = day_offset[month] + day
                day_type = day_type_lookup[doy]

                for slot in range(48):
                    a["steps"] += 1

                    gen_shape_arr = gen_by_ds.get((gen_doy, slot), all_zero)
                    dem_key = (month, day_type, slot)
                    dem_shape_arr = dem_lookup.get(dem_key)
                    if dem_shape_arr is None:
                        continue

                    gen = run_gen_annual * gen_shape_arr
                    dem = run_dem_annual * dem_shape_arr / DEM_NORM

                    a["demand"] += dem
                    self_consumed = np.minimum(gen, dem)
                    a["self_consumed"] += self_consumed

                    surplus = np.maximum(gen - dem, 0)
                    deficit = np.maximum(dem - gen, 0)

                    if surplus.sum() <= 0:
                        a["skipped"] += 1
                        for lam in ALL_LAMBDAS:
                            a["served"][lam] += self_consumed
                        continue

                    a["active"] += 1
                    a["surplus_sum"] += surplus

                    # inter-community energy sharing for each λ
                    # ── inter-community energy sharing for each finite λ ──
                    for lam in LAMBDAS_KM:
                        W_raw = weight_matrices[lam]
                        received, imported = conditional_donor_normalized(
                            W_raw, surplus, deficit
                        )

                        served = self_consumed + imported
                        a["served"][lam] += served
                        a["imported"][lam] += imported
                        a["total_surplus"][lam] += surplus.sum()
                        a["total_imported"][lam] += imported.sum()

                    # λ→∞
                    total_surplus = surplus.sum()
                    total_deficit = deficit.sum()
                    if total_deficit > 0 and total_surplus > 0:
                        ratio = min(1.0, total_surplus / total_deficit)
                        imported_inf = deficit * ratio
                    else:
                        imported_inf = np.zeros_like(deficit)
                    served_inf = self_consumed + imported_inf
                    a["served"][LAMBDA_INF] += served_inf
                    a["imported"][LAMBDA_INF] += imported_inf
                    a["total_surplus"][LAMBDA_INF] += surplus.sum()
                    a["total_imported"][LAMBDA_INF] += imported_inf.sum()

            # Save checkpoint
            completed_months.add(month)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ckpt_path, "w") as f:
                json.dump({"completed_months": sorted(completed_months)}, f)
            np.savez_compressed(
                CKPT_DIR / f"{run_name}_accum.npz",
                demand=a["demand"],
                self_consumed=a["self_consumed"],
                surplus_sum=a["surplus_sum"],
                **{f"served_{str(lam).replace('.', '_')}": a["served"][lam] for lam in ALL_LAMBDAS},
                **{f"imported_{str(lam).replace('.', '_')}": a["imported"][lam] for lam in ALL_LAMBDAS},
            )
            with open(CKPT_DIR / f"{run_name}_sur.json", "w") as f:
                json.dump({
                    "surplus": {str(lam): a["total_surplus"][lam] for lam in ALL_LAMBDAS},
                    "imported": {str(lam): a["total_imported"][lam] for lam in ALL_LAMBDAS},
                }, f)
            with open(CKPT_DIR / f"{run_name}_stats.json", "w") as f:
                json.dump({"steps": a["steps"], "skipped": a["skipped"], "active": a["active"]}, f)

        elapsed = time.time() - t_month
        remaining = (len(SIM_MONTHS) - SIM_MONTHS.index(month) - 1) * elapsed / 60
        log(f"  Month {month:2d} done ({elapsed/60:.1f} min, est {remaining:.0f} min remaining)")

    total_sim = time.time() - t0
    log(f"\n  Simulation complete in {total_sim/60:.1f} min")

    # ── 8. Post-processing ──
    log("\n[8/8] Computing summaries...")

    all_rows = []
    summary_rows = []

    for run_name, cfg in SIM_RUNS.items():
        a = accum[run_name]
        if a["active"] == 0:
            log(f"  [{run_name}] SKIP: 0 active steps")
            continue

        acc_dem = a["demand"]
        ssr = np.where(acc_dem > 0, a["self_consumed"] / acc_dem, 0.0)
        donor_mask = a["surplus_sum"] > (acc_dem - a["self_consumed"])
        annual_surplus = a["surplus_sum"]

        for lam in ALL_LAMBDAS:
            lam_str = f"{lam:.1f}" if isinstance(lam, float) else lam
            cssr = np.where(acc_dem > 0, a["served"][lam] / acc_dem, 0.0)
            cpi = cssr - ssr
            sur_val = (a["total_imported"][lam] / a["total_surplus"][lam]
                       if a["total_surplus"][lam] > 0 else 0.0)

            summary_rows.append({
                "run": run_name,
                "urban_form_scenario": cfg["urban_form_scenario"],
                "alpha_label": cfg["alpha_label"],
                "lambda_km": lam if isinstance(lam, float) else np.inf,
                "lambda_label": lam_str,
                "n_meshes": n_meshes,
                "ssr_mean": float(ssr.mean()),
                "cssr_mean": float(cssr.mean()),
                "cpi_mean": float(cpi.mean()),
                "cpi_median": float(np.median(cpi)),
                "cpi_std": float(cpi.std()),
                "cpi_p90": float(np.percentile(cpi, 90)),
                "cpi_gini": float(gini(cpi)),
                "sur": float(sur_val),
                "donor_pct": float(donor_mask.mean()),
                "total_surplus_kwh": float(annual_surplus.sum()),
                "total_imported_kwh": float(a["imported"][lam].sum()),
                "skipped_pct": float(a["skipped"] / max(1, a["steps"])),
            })

            for cat in cats:
                mask = category_3 == cat
                if not mask.any():
                    continue
                summary_rows.append({
                    "run": run_name,
                    "urban_form_scenario": cfg["urban_form_scenario"],
                    "alpha_label": cfg["alpha_label"],
                    "lambda_km": lam if isinstance(lam, float) else np.inf,
                    "lambda_label": lam_str,
                    "category_3": cat,
                    "n_meshes": int(mask.sum()),
                    "ssr_mean": float(ssr[mask].mean()),
                    "cssr_mean": float(cssr[mask].mean()),
                    "cpi_mean": float(cpi[mask].mean()),
                    "cpi_median": float(np.median(cpi[mask])),
                    "cpi_std": float(cpi[mask].std()),
                    "cpi_gini": float(gini(cpi[mask])),
                    "sur": float(sur_val),
                    "donor_pct": float(donor_mask[mask].mean()),
                    "total_surplus_kwh": float(annual_surplus[mask].sum()),
                    "total_imported_kwh": float(a["imported"][lam][mask].sum()),
                })

        # Mesh-level (λ=5 km)
        lam_mesh = 5.0
        cssr_mesh = np.where(acc_dem > 0, a["served"][lam_mesh] / acc_dem, 0.0)
        cpi_mesh = cssr_mesh - ssr
        for i in range(n_meshes):
            all_rows.append({
                "mesh_code": mesh_codes[i],
                "run": run_name,
                "urban_form_scenario": cfg["urban_form_scenario"],
                "alpha_label": cfg["alpha_label"],
                "category_3": category_3[i],
                "ssr": float(ssr[i]),
                "cssr": float(cssr_mesh[i]),
                "cpi": float(cpi_mesh[i]),
                "is_donor": bool(donor_mask[i]),
                "annual_gen_kwh": float(cfg["gen"][i]),
                "annual_dem_kwh": float(cfg["dem"][i]),
                "annual_surplus_kwh": float(annual_surplus[i]),
                "imported_kwh": float(a["imported"][lam_mesh][i]),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(OUT_DIR / "scenario_simulation_summary.parquet", index=False)
    log(f"  [OK] scenario_simulation_summary.parquet: {len(summary_df)} rows")

    mesh_df = pd.DataFrame(all_rows)
    mesh_df.to_parquet(OUT_DIR / "scenario_simulation_mesh.parquet", index=False)
    log(f"  [OK] scenario_simulation_mesh.parquet: {len(mesh_df)} rows")

    # ── 9. Print results ──
    log(f"\n{'='*70}")
    log("KEY RESULTS (λ=5 km, annual)")
    log("=" * 70)

    lam_key = 5.0
    sys_rows = summary_df[
        (summary_df["lambda_km"] == lam_key) &
        (summary_df["category_3"].isna())
    ]

    log(f"\n{'run':20s} {'SSR':>8s} {'CSSR':>8s} {'CPI':>8s} {'CPI_pp':>8s} "
        f"{'SUR':>8s} {'Donor%':>8s}")
    for _, r in sys_rows.iterrows():
        log(f"{r['run']:20s} {r['ssr_mean']:8.4f} {r['cssr_mean']:8.4f} "
            f"{r['cpi_mean']:8.4f} {r['cpi_mean']*100:7.2f}% "
            f"{r['sur']:8.4f} {r['donor_pct']*100:7.1f}%")

    log(f"\nCPI by category_3 (λ=5 km):")
    log(f"{'run':20s} {'metro':>10s} {'regional':>10s} {'rural':>10s}")
    cat_rows = summary_df[
        (summary_df["lambda_km"] == lam_key) &
        (summary_df["category_3"].notna())
    ]
    for run_name in SIM_RUNS:
        sub = cat_rows[cat_rows["run"] == run_name]
        vals = {}
        for cat in ["metropolitan_core", "regional_city", "rural"]:
            r_cat = sub[sub["category_3"] == cat]
            if len(r_cat) > 0:
                vals[cat] = r_cat["cpi_mean"].iloc[0]
        if vals:
            log(f"{run_name:20s} "
                f"{vals.get('metropolitan_core', float('nan'))*100:9.2f}% "
                f"{vals.get('regional_city', float('nan'))*100:9.2f}% "
                f"{vals.get('rural', float('nan'))*100:9.2f}%")

    # Urban form ΔCPI at each technical-potential fraction
    log(f"\n{'='*70}")
    log("URBAN FORM EFFECT (ΔCPI vs S0, same technical-potential fraction)")
    log("=" * 70)
    for a_label in ALPHA_LEVELS:
        log(f"\n── Technical-potential {a_label} ──")
        s0_sub = cat_rows[(cat_rows["run"] == f"S0_{a_label}")]
        for sn in ["S1", "S2", "S3"]:
            sn_sub = cat_rows[(cat_rows["run"] == f"{sn}_{a_label}")]
            parts = []
            for cat in ["metropolitan_core", "regional_city", "rural"]:
                s0_cpi = s0_sub[s0_sub["category_3"] == cat]["cpi_mean"]
                sn_cpi = sn_sub[sn_sub["category_3"] == cat]["cpi_mean"]
                if len(s0_cpi) > 0 and len(sn_cpi) > 0:
                    d = (sn_cpi.iloc[0] - s0_cpi.iloc[0]) * 100
                    parts.append(f"{cat}: {d:+.2f} pp")
            log(f"  {sn}: " + " | ".join(parts))

    total_elapsed = time.time() - t_total
    log(f"\n{'='*70}")
    log(f"Simulation complete. Total elapsed: {total_elapsed/60:.1f} min")
    log(f"{'='*70}")


if __name__ == "__main__":
    run_build()
    run_simulate()

