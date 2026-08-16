# -*- coding: utf-8 -*-
"""Decompositions — distance-band flow decomposition and additive
benefit/Gini/allocation decomposition.
"""





def run_band_decomposition():
    import pandas as pd
    import numpy as np
    from scipy.sparse import csr_matrix
    from pathlib import Path
    import time, sys, calendar, glob

    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    PROJ_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJ_ROOT / "data" / "input" / "mesh"
    STATIC_DATA_DIR = PROJ_ROOT / "data" / "input" / "shapes"  # non-coordinate ref files
    SIM_DIR = PROJ_ROOT / "code" / "simulation"
    if str(SIM_DIR) not in sys.path:
        sys.path.insert(0, str(SIM_DIR))

    OUT_DIR = Path(__file__).resolve().parent / "results" / "controls"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    LAMBDA_KM = 5.0
    ALPHA_LABEL = "tp030"
    ALPHA = 0.30
    CAT_LIST = ["metropolitan_core", "regional_city", "rural"]
    DISTANCE_BANDS = [(0, 1), (1, 5), (5, 10), (10, None)]  # km, None = inf
    BAND_LABELS = ["0-1km", "1-5km", "5-10km", ">10km"]
    N_SLOTS = 48


    def log(msg):
        print(msg, flush=True)


    # ══════════════════════════════════════════════════════════════════════
    log("=" * 70)
    log("Distance-band flow decomposition — per-timestep")
    log(f"  Technical-potential: {ALPHA_LABEL} (α={ALPHA}), λ={LAMBDA_KM}km")
    log("=" * 70)
    t0_total = time.time()

    # ── 1. Load data & build mesh index ──
    log("\n[1/6] Loading data & building mesh index...")
    season = pd.read_parquet(STATIC_DATA_DIR / "season_factors.parquet")
    sf_map = dict(zip(season["month"], season["season_factor"]))
    DEM_NORM = sum(calendar.monthrange(2024, m)[1] * sf_map[m] for m in range(1, 13))
    log(f"  DEM_NORM = {DEM_NORM:.4f}")

    arv2 = pd.read_parquet(PROJ_ROOT / "data" / "input" / "potential" / "technical_potential.parquet")
    dem_all = pd.read_parquet(STATIC_DATA_DIR / "dem_shapes.parquet")
    dist = pd.read_parquet(DATA_DIR / "sparse_distances.parquet")
    base = pd.read_parquet(DATA_DIR / "mesh_attributes.parquet")

    # Find common meshes across all data sources
    base_meshes = set(arv2["mesh_code"])
    gen_meshes = set()
    for gf in sorted(glob.glob(str(STATIC_DATA_DIR / "gen_shapes" / "month=*.parquet"))):
        gs = pd.read_parquet(gf, columns=["mesh_code"])
        gen_meshes.update(gs["mesh_code"].unique())
    dem_meshes = set(dem_all["mesh_code"])
    common = sorted(base_meshes & gen_meshes & dem_meshes)
    mesh_to_idx = {int(m): i for i, m in enumerate(common)}
    n = len(common)
    log(f"  Valid meshes: {n:,}")

    # Align arrays
    arv2_idx = arv2.set_index("mesh_code").loc[common]
    potential = arv2_idx["potential_kwh"].values.astype(np.float64)
    annual_dem = arv2_idx["annual_dem_kwh"].values.astype(np.float64)
    category_3 = arv2_idx["category_3"].values
    cat_idx = {c: category_3 == c for c in CAT_LIST}

    # ── 2. Build distance mapping & band-specific W_raw matrices ──
    log("\n[2/6] Building band-specific weight matrices...")

    orig_mesh_arr = base["mesh_code"].values.astype(np.int64)
    dist_i = dist["mesh_idx_i"].values.astype(np.int32)
    dist_j = dist["mesh_idx_j"].values.astype(np.int32)
    dist_d = dist["distance_km"].values.astype(np.float32)

    dist_mi = orig_mesh_arr[dist_i]
    dist_mj = orig_mesh_arr[dist_j]
    keep_i = np.array([int(m) in mesh_to_idx for m in dist_mi])
    keep_j = np.array([int(m) in mesh_to_idx for m in dist_mj])
    keep = keep_i & keep_j

    di_new = np.array([mesh_to_idx[int(m)] for m in dist_mi[keep]], dtype=np.int32)
    dj_new = np.array([mesh_to_idx[int(m)] for m in dist_mj[keep]], dtype=np.int32)
    dd_new = dist_d[keep]
    n_pairs = len(dd_new)
    log(f"  Distance pairs: {n_pairs:,}")

    # Assign band index to each pair
    band_idx = np.full(n_pairs, -1, dtype=np.int8)
    for b, (lo, hi) in enumerate(DISTANCE_BANDS):
        if hi is None:
            band_idx[dd_new >= lo] = b
        else:
            band_idx[(dd_new >= lo) & (dd_new < hi)] = b
    assert (band_idx >= 0).all(), "Some pairs not assigned to any band!"

    # Build W_raw for λ=5km (full matrix)
    w_all = np.exp(-dd_new / LAMBDA_KM).astype(np.float32)
    W_full = csr_matrix((w_all, (di_new, dj_new)), shape=(n, n), dtype=np.float32)
    log(f"  W_full(λ={LAMBDA_KM}km): {W_full.nnz:,} non-zeros")

    # Build band-specific W_raw matrices
    W_bands = []
    for b in range(len(DISTANCE_BANDS)):
        bmask = band_idx == b
        n_b = bmask.sum()
        if n_b > 0:
            Wb = csr_matrix(
                (w_all[bmask], (di_new[bmask], dj_new[bmask])),
                shape=(n, n), dtype=np.float32
            )
        else:
            Wb = csr_matrix((n, n), dtype=np.float32)
        W_bands.append(Wb)
        log(f"  W_band[{b}] ({BAND_LABELS[b]}): {Wb.nnz:,} non-zeros")

    # Verify: sum of band matrices ≈ W_full
    W_sum = sum(W_bands)
    diff = (W_full - W_sum).data
    max_diff = np.abs(diff).max() if len(diff) > 0 else 0.0
    log(f"  Sanity: max|W_full - sum(W_band)| = {max_diff:.2e}")

    # ── 3. Build demand shape lookup ──
    log("\n[3/6] Building demand shape lookup...")
    dem_mesh_set = set(common)
    dem_filtered = dem_all[dem_all["mesh_code"].isin(dem_mesh_set)]
    dem_lookup = {}
    for (month, dt, slot), group in dem_filtered.groupby(["month", "day_type", "slot"]):
        arr = np.zeros(n, dtype=np.float64)
        idx_list = []
        val_list = []
        for m, v in zip(group["mesh_code"].values, group["dem_shape"].values):
            if int(m) in mesh_to_idx:
                arr[mesh_to_idx[int(m)]] = v
        dem_lookup[(month, dt, slot)] = arr
    log(f"  dem_lookup: {len(dem_lookup)} keys")

    # ── 4. Calendar ──
    day_offset = {1: 0}
    for m in range(2, 13):
        day_offset[m] = day_offset[m-1] + calendar.monthrange(2024, m-1)[1]
    dt_lookup = {}
    for m in range(1, 13):
        for d in range(1, calendar.monthrange(2024, m)[1] + 1):
            doy = day_offset[m] + (d - 1)
            wd = calendar.weekday(2024, m, d)
            dt_lookup[doy] = "we" if wd >= 5 else "wd"

    # ── 5. Per-timestep simulation with band tracking ──
    log("\n[4/6] Running per-timestep inter-community energy sharing with band tracking...")

    # Accumulators
    acc_demand = np.zeros(n, dtype=np.float64)
    acc_sc = np.zeros(n, dtype=np.float64)
    acc_surplus = np.zeros(n, dtype=np.float64)
    acc_imported = np.zeros(n, dtype=np.float64)
    acc_deficit = np.zeros(n, dtype=np.float64)

    # Band accumulators: [band][category_idx] where category_idx 0=system, 1-3=categories
    n_bands = len(DISTANCE_BANDS)
    n_cats = 1 + len(CAT_LIST)  # system + 3 categories
    band_accum = np.zeros((n_bands, n_cats), dtype=np.float64)

    total_timesteps = 0
    active_timesteps = 0
    skipped_timesteps = 0
    all_zero = np.zeros(n, dtype=np.float64)

    for month in range(1, 13):
        t_m = time.time()

        # Load gen_shapes for this month
        gen_file = STATIC_DATA_DIR / "gen_shapes" / f"month={month:02d}.parquet"
        gen_month = pd.read_parquet(gen_file)
        gen_month = gen_month[gen_month["mesh_code"].isin(common)]
        gen_month["mesh_idx"] = gen_month["mesh_code"].map(mesh_to_idx)
        gen_month = gen_month.dropna(subset=["mesh_idx"])
        gen_month["mesh_idx"] = gen_month["mesh_idx"].astype(np.int32)

        gen_by_day_slot = {}
        for (day, slot), group in gen_month.groupby(["day", "slot"]):
            arr = np.zeros(n, dtype=np.float64)
            arr[group["mesh_idx"].values] = group["gen_shape"].values
            gen_by_day_slot[(int(day), int(slot))] = arr
        all_zero = np.zeros(n, dtype=np.float64)

        n_days = calendar.monthrange(2024, month)[1]
        month_active = 0

        for day in range(1, n_days + 1):
            doy = day_offset[month] + (day - 1)
            gen_doy = day_offset[month] + day
            dt = dt_lookup[doy]

            for slot in range(N_SLOTS):
                total_timesteps += 1

                gs_arr = gen_by_day_slot.get((gen_doy, slot), all_zero)
                dem_key = (month, dt, slot)
                if dem_key not in dem_lookup:
                    continue
                dem_base = dem_lookup[dem_key]

                gen = potential * ALPHA * gs_arr
                dem = annual_dem * dem_base / DEM_NORM

                acc_demand += dem
                sc_raw = np.minimum(gen, dem)
                acc_sc += sc_raw
                surplus = np.maximum(gen - dem, 0)
                acc_surplus += surplus
                deficit_t = np.maximum(dem - gen, 0)
                acc_deficit += deficit_t

                sur_sum = surplus.sum()
                if sur_sum <= 0:
                    skipped_timesteps += 1
                    continue

                active_timesteps += 1
                month_active += 1

                # ── conditional_donor_normalized (inlined) ──
                active_receivers = deficit_t > 0
                n_active = active_receivers.sum()

                if n_active == 0:
                    skipped_timesteps += 1
                    continue

                # active_weight = W_full^T @ active_receivers (per-donor)
                active_weight = np.asarray(
                    W_full.T.dot(active_receivers.astype(np.float64))
                ).ravel()

                # scaled_surplus = surplus / active_weight (per-donor)
                scaled_surplus = np.zeros(n, dtype=np.float64)
                np.divide(surplus, active_weight, out=scaled_surplus,
                          where=active_weight > 0)

                # total_received = W_full @ scaled_surplus (per-receiver)
                total_received = np.asarray(
                    W_full.dot(scaled_surplus)
                ).ravel()
                total_received[~active_receivers] = 0.0

                # imported = min(deficit, received)
                imported = np.minimum(deficit_t, total_received)
                acc_imported += imported

                # ── Band decomposition ──
                # band_received[b] = W_band_b @ scaled_surplus
                # Normalize to actual imported amounts (cap at deficit)
                cap_mask = total_received > 0
                cap_ratio = np.ones(n, dtype=np.float64)
                cap_ratio[cap_mask] = imported[cap_mask] / total_received[cap_mask]

                for b in range(n_bands):
                    band_received = np.asarray(
                        W_bands[b].dot(scaled_surplus)
                    ).ravel()
                    band_received[~active_receivers] = 0.0
                    band_imported = band_received * cap_ratio

                    # System total
                    band_accum[b, 0] += band_imported.sum()

                    # Per receiver category
                    for ci, cat in enumerate(CAT_LIST):
                        cat_mask = cat_idx[cat]
                        band_accum[b, 1 + ci] += band_imported[cat_mask].sum()

        elapsed_m = time.time() - t_m
        log(f"  Month {month:2d}: {n_days}d, active steps={month_active}, "
            f"time={elapsed_m:.1f}s")

    # ── 6. Compile results ──
    log("\n[5/6] Compiling results...")

    # Sanity checks
    total_import_accum = acc_imported.sum()
    total_band_flow = band_accum[:, 0].sum()
    log(f"  Total inter-community energy sharing import (from accum): {total_import_accum / 1e9:.4f} TWh")
    log(f"  Total band flow (system):      {total_band_flow / 1e9:.4f} TWh")
    log(f"  Discrepancy: {abs(total_import_accum - total_band_flow) / max(total_import_accum, 1):.2e}")
    log(f"  Active timesteps: {active_timesteps:,} / {total_timesteps:,} "
        f"({active_timesteps/max(total_timesteps,1)*100:.1f}%)")

    rows = []
    # System + per-category
    for ci, cat_label in enumerate(["all"] + CAT_LIST):
        cat_total = band_accum[:, ci].sum()
        for b in range(n_bands):
            band_flow = band_accum[b, ci]
            pct = band_flow / cat_total * 100 if cat_total > 0 else 0.0
            rows.append({
                "lambda_km": LAMBDA_KM,
                "alpha_label": ALPHA_LABEL,
                "category_3": cat_label,
                "distance_band": BAND_LABELS[b],
                "surplus_absorbed_kwh": float(band_flow),
                "share_pct": round(float(pct), 2),
            })
            log(f"  {cat_label:20s} {BAND_LABELS[b]:8s}: "
                f"{band_flow/1e9:.2f} TWh ({pct:.1f}%)")

    band_df = pd.DataFrame(rows)

    # ── Save ──
    log("\n[6/6] Saving...")
    # Archive old static version if it exists
    old_path = OUT_DIR / "band_decomposition.parquet"
    static_path = OUT_DIR / "band_decomposition_static.parquet"
    if old_path.exists():
        import shutil
        shutil.copy2(old_path, static_path)
        log(f"  Archived old static version to: {static_path.name}")

    band_df.to_parquet(old_path, index=False)
    log(f"  Saved per-timestep results: {old_path} ({len(band_df)} rows)")

    # ── Comparison with static (if archived) ──
    if static_path.exists():
        log("\n── Comparison: Per-Timestep vs Static Annual Approximation ──")
        old_df = pd.read_parquet(static_path)
        old_5 = old_df[(old_df["lambda_km"] == 5.0) & (old_df["category_3"] == "all")]
        new_5 = band_df[band_df["category_3"] == "all"]
        log(f"  {'Band':<10s} {'Per-Timestep':>14s} {'Static (old)':>14s} {'Δ':>10s}")
        for b in range(n_bands):
            new_pct = band_accum[b, 0] / total_band_flow * 100
            old_row = old_5[old_5["distance_band"] == BAND_LABELS[b]]
            if len(old_row):
                old_flow = old_row.iloc[0]["surplus_absorbed_kwh"]
                old_total = old_5["surplus_absorbed_kwh"].sum()
                old_pct = old_flow / old_total * 100 if old_total > 0 else 0
                log(f"  {BAND_LABELS[b]:<10s} {new_pct:>13.1f}%  {old_pct:>13.1f}%  "
                    f"{new_pct-old_pct:>+9.1f}pp")

    elapsed = time.time() - t0_total
    log(f"\n{'='*70}")
    log(f"Band decomposition complete. Elapsed: {elapsed/60:.1f} min")
    log(f"{'='*70}")



def run_additive_decomposition():
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
    SIM_DIR = PROJ_ROOT / "data" / "results" / "summary"
    OUT_DIR = Path(__file__).resolve().parent / "results" / "controls"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ALPHA_LABEL = "tp030"
    LAMBDA_KM = 5.0
    LAMBDA_COL = "cpi_5.0"
    CSSR_COL = "cssr_5.0"


    def gini(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[~np.isnan(x)]
        if len(x) == 0 or x.sum() == 0:
            return 0.0
        x_sorted = np.sort(x)
        n = len(x)
        return (2 * np.sum(np.arange(1, n + 1) * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n


    def log(msg):
        print(msg, flush=True)


    log("=" * 70)
    log(f"Additive decomposition (α=30%, λ={LAMBDA_KM} km)")
    log("=" * 70)

    # ── Load ──
    cssr = pd.read_parquet(SIM_DIR / f"cssr_summary_{ALPHA_LABEL}.parquet")
    n = len(cssr)
    cpi_vals = cssr[LAMBDA_COL].values
    ssr_vals = cssr["ssr"].values
    cssr_vals = cssr[CSSR_COL].values
    dem_vals = cssr["annual_dem_kwh"].values
    deficit_vals = cssr["annual_deficit_kwh"].values
    is_receiver = cssr["is_receiver"].values

    # Actual inter-community energy sharing import per mesh (kWh)
    actual_import = cpi_vals * dem_vals
    total_import = actual_import.sum()

    log(f"\n  Meshes: {n}, Total inter-community energy sharing import: {total_import / 1e9:.4f} TWh")

    # ══════════════════════════════════════════════════════════════════
    # T8.1: Benefit Incidence — Actual vs Deficit-Proportional
    # ══════════════════════════════════════════════════════════════════
    log("\n── T8.1 Benefit Incidence: Actual vs Deficit-Proportional Benchmark ──")

    deficit_share = deficit_vals / deficit_vals.sum()
    benchmark_import = deficit_share * total_import

    # SSR decile assignment
    ssr_decile = pd.qcut(ssr_vals, 10, labels=False, duplicates="drop")
    # Normalize to 1-10 range
    n_deciles = ssr_decile.max() + 1
    decile_map = {d: i + 1 for i, d in enumerate(sorted(set(ssr_decile)))}
    ssr_decile_mapped = np.array([decile_map[d] for d in ssr_decile])

    t8_rows = []
    for decile in range(1, 11):
        mask = ssr_decile_mapped == decile
        if mask.sum() == 0:
            continue
        actual_share = actual_import[mask].sum() / total_import
        benchmark_share = benchmark_import[mask].sum() / total_import
        n_meshes = mask.sum()
        mean_ssr = float(ssr_vals[mask].mean())
        mean_deficit = float(deficit_vals[mask].mean())

        t8_rows.append({
            "ssr_decile": decile,
            "n_meshes": n_meshes,
            "mean_ssr": round(mean_ssr, 4),
            "mean_deficit_mwh": round(mean_deficit / 1000, 2),
            "actual_import_share_pct": round(actual_share * 100, 2),
            "benchmark_deficit_proportional_share_pct": round(benchmark_share * 100, 2),
            "excess_over_benchmark_pp": round((actual_share - benchmark_share) * 100, 2),
            "ratio_actual_to_benchmark": round(actual_share / benchmark_share, 3) if benchmark_share > 0 else float('inf'),
        })

    t8_df = pd.DataFrame(t8_rows)
    t8_df.to_csv(OUT_DIR / "benefit_incidence_decomposition.csv", index=False)

    # Summary
    bottom_actual = t8_df[t8_df["ssr_decile"].isin([1, 2])]["actual_import_share_pct"].sum()
    bottom_bench = t8_df[t8_df["ssr_decile"].isin([1, 2])]["benchmark_deficit_proportional_share_pct"].sum()
    top_actual = t8_df[t8_df["ssr_decile"].isin([9, 10])]["actual_import_share_pct"].sum()
    top_bench = t8_df[t8_df["ssr_decile"].isin([9, 10])]["benchmark_deficit_proportional_share_pct"].sum()

    log(f"  Bottom 20% SSR: Actual={bottom_actual:.1f}%, Deficit-proportional={bottom_bench:.1f}%")
    log(f"  Top 20% SSR:    Actual={top_actual:.1f}%, Deficit-proportional={top_bench:.1f}%")
    log(f"  Excess progressivity: bottom gets {bottom_actual - bottom_bench:+.1f}pp more than deficit share")

    # ══════════════════════════════════════════════════════════════════
    # T8.2: Gini Decomposition — Mechanical vs Network
    # ══════════════════════════════════════════════════════════════════
    log("\n── T8.2 Gini Reduction Decomposition ──")

    # Deficit-proportional allocation: import ∝ annual deficit, same total inter-community energy sharing volume
    deficit_prop_import = deficit_share * total_import
    deficit_prop_cssr = np.where(dem_vals > 0,
                                  (ssr_vals * dem_vals + deficit_prop_import) / dem_vals, 0)

    # Network-driven = actual - deficit-proportional
    network_extra_import = actual_import - deficit_prop_import

    g_ssr = gini(ssr_vals)
    g_deficit_prop = gini(deficit_prop_cssr)
    g_actual = gini(cssr_vals)

    gini_total_reduction = g_ssr - g_actual
    gini_mechanical = g_ssr - g_deficit_prop   # from deficit magnitude alone
    gini_network = g_deficit_prop - g_actual    # extra from inter-community energy sharing geography

    mech_pct = gini_mechanical / gini_total_reduction * 100 if gini_total_reduction > 0 else 0
    net_pct = gini_network / gini_total_reduction * 100 if gini_total_reduction > 0 else 0

    log(f"  Gini(SSR)                              = {g_ssr:.4f}")
    log(f"  Gini(SSR + deficit-proportional import) = {g_deficit_prop:.4f}")
    log(f"  Gini(CSSR actual, λ={LAMBDA_KM}km)     = {g_actual:.4f}")
    log(f"  ΔGini_total   = {gini_total_reduction:.4f} (100%)")
    log(f"  ΔGini_mechanical (deficit-driven)        = {gini_mechanical:.4f} ({mech_pct:.1f}%)")
    log(f"  ΔGini_network (inter-community energy sharing geography-driven)     = {gini_network:.4f} ({net_pct:.1f}%)")

    gini_decomp = pd.DataFrame([{
        "gini_ssr": g_ssr,
        "gini_deficit_proportional": g_deficit_prop,
        "gini_cssr_actual": g_actual,
        "delta_gini_total": gini_total_reduction,
        "delta_gini_mechanical_deficit_driven": gini_mechanical,
        "delta_gini_network_geography_driven": gini_network,
        "mechanical_share_pct": round(mech_pct, 1),
        "network_share_pct": round(net_pct, 1),
    }])
    gini_decomp.to_csv(OUT_DIR / "gini_decomposition.csv", index=False)

    # ══════════════════════════════════════════════════════════════════
    # T8.3: Equal-per-receiver benchmark
    # ══════════════════════════════════════════════════════════════════
    log("\n── T8.3 Alternative Allocation Benchmarks ──")
    n_receivers = int(is_receiver.sum())
    if n_receivers > 0:
        equal_per_recv_import = np.where(is_receiver, total_import / n_receivers, 0.0)
        equal_cssr = np.where(dem_vals > 0,
                              (ssr_vals * dem_vals + equal_per_recv_import) / dem_vals, 0)
        g_equal = gini(equal_cssr)

        # Also: surplus-proportional benchmark
        surplus_vals = cssr["annual_surplus_kwh"].values
        surplus_share = np.where(surplus_vals.sum() > 0,
                                 surplus_vals / surplus_vals.sum(), 0)
        surplus_prop_import = surplus_share * total_import
        surplus_prop_cssr = np.where(dem_vals > 0,
                                      (ssr_vals * dem_vals + surplus_prop_import) / dem_vals, 0)
        g_surplus_prop = gini(surplus_prop_cssr)

        log(f"  Equal-per-receiver:   Gini(CSSR) = {g_equal:.4f}, ΔGini = {g_ssr - g_equal:.4f}")
        log(f"  Surplus-proportional:  Gini(CSSR) = {g_surplus_prop:.4f}, ΔGini = {g_ssr - g_surplus_prop:.4f}")
        log(f"  Actual (λ={LAMBDA_KM}km):       Gini(CSSR) = {g_actual:.4f}, ΔGini = {gini_total_reduction:.4f}")

        benchmarks = pd.DataFrame([{
            "benchmark": "equal_per_receiver",
            "gini_cssr": g_equal,
            "delta_gini": g_ssr - g_equal,
            "delta_gini_ratio_to_actual": (g_ssr - g_equal) / gini_total_reduction if gini_total_reduction > 0 else 0,
        }, {
            "benchmark": "surplus_proportional",
            "gini_cssr": g_surplus_prop,
            "delta_gini": g_ssr - g_surplus_prop,
            "delta_gini_ratio_to_actual": (g_ssr - g_surplus_prop) / gini_total_reduction if gini_total_reduction > 0 else 0,
        }, {
            "benchmark": "actual_distance_decay",
            "gini_cssr": g_actual,
            "delta_gini": gini_total_reduction,
            "delta_gini_ratio_to_actual": 1.0,
        }])
        benchmarks.to_csv(OUT_DIR / "allocation_benchmarks.csv", index=False)

    # ══════════════════════════════════════════════════════════════════
    log(f"\nOutputs saved to: {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.csv")):
        log(f"  {f.name}")
    log("=" * 70)


if __name__ == "__main__":
    run_band_decomposition()
    run_additive_decomposition()

