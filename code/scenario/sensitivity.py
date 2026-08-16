# -*- coding: utf-8 -*-
"""Sensitivity analyses — demand response and spatially targeted
technical-potential deployment.
"""





def run_demand_response():
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
    STATIC_DIR = PROJ_ROOT / "data" / "input" / "shapes"               # gen_shapes, dem_shapes, season_factors
    COORD_DIR = PROJ_ROOT / "data" / "input" / "mesh"    # mesh_attributes, sparse_distances
    FORMAL_DIR = PROJ_ROOT / "data" / "input" / "potential"  # technical_potential (正式 potential)
    SIM_DIR = PROJ_ROOT / "code" / "simulation"
    if str(SIM_DIR) not in sys.path:
        sys.path.insert(0, str(SIM_DIR))
    from allocation_rules import conditional_donor_normalized

    GEO_PATH = Path(__file__).resolve().parent / "results" / "scenario_geometry.parquet"
    OUT_DIR = Path(__file__).resolve().parent / "results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Parameters ────────────────────────────────────────────────────
    LAMBDA_PRIMARY = 5.0
    EVENING_SLOTS = list(range(34, 41))   # 17:00-20:00
    MIDDAY_SLOTS = list(range(22, 29))    # 11:00-14:00
    SHIFT_RATIOS = [0.0, 0.10, 0.20, 0.30]
    SHIFT_LABELS = {0.0: "DR_0pct", 0.10: "DR_10pct", 0.20: "DR_20pct", 0.30: "DR_30pct"}
    ENDPOINT_SHIFTS = {0.0, 0.30}
    # To add tp015/tp100: change to ALPHA_LABELS = {"tp030": 0.30, "tp015": 0.15, "tp100": 1.00}
    # and keep ENDPOINT_SHIFTS as is. For now, tp030-only.
    ALPHA_LABELS = {"tp030": 0.30}  # PRIMARY only for initial run
    CAT_LIST = ["metropolitan_core", "regional_city", "rural"]
    N_SLOTS = 48


    def gini(x):
        x = np.asarray(x, dtype=np.float64)
        if len(x) == 0 or x.sum() == 0:
            return 0.0
        x_sorted = np.sort(x)
        n = len(x)
        return (2 * np.sum(np.arange(1, n + 1) * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n


    def build_summary(ssr, cssr, cpi, imported, deficit, surplus, dem_total,
                      gen_eff, sc_total, ci_per_mesh, donor_mask, category_3, extra):
        imp_sum = imported.sum()
        def_sum = deficit.sum()
        sur_sum = surplus.sum()
        row = dict(extra)
        row.update({
            "ssr_mean": float(ssr.mean()), "cssr_mean": float(cssr.mean()),
            "cpi_mean": float(cpi.mean()), "cpi_median": float(np.median(cpi)),
            "sur": float(imp_sum / sur_sum) if sur_sum > 0 else 0.0,
            "deficit_coverage": float(imp_sum / def_sum) if def_sum > 0 else 0.0,
            "transfer_volume_twh": float(imp_sum / 1e9),
            "total_surplus_twh": float(sur_sum / 1e9),
            "total_deficit_twh": float(def_sum / 1e9),
            "total_gen_twh": float(gen_eff.sum() / 1e9),
            "total_dem_twh": float(dem_total.sum() / 1e9),
            "gen_dem_ratio": float(gen_eff.sum() / max(dem_total.sum(), 1)),
            "ci_system": float(sc_total.sum() / max(gen_eff.sum(), 1e-12)),
            "ci_mean": float(ci_per_mesh.mean()),
            "ci_median": float(np.median(ci_per_mesh)),
            "donor_pct": float(donor_mask.mean()),
            "gini_cssr": float(gini(cssr)), "gini_cpi": float(gini(cpi)),
        })
        if category_3 is not None:
            for cat in CAT_LIST:
                mask = category_3 == cat
                if mask.any():
                    imp_c = imported[mask].sum()
                    def_c = deficit[mask].sum()
                    sur_c = surplus[mask].sum()
                    row[f"{cat}_ssr_mean"] = float(ssr[mask].mean())
                    row[f"{cat}_cssr_mean"] = float(cssr[mask].mean())
                    row[f"{cat}_cpi_mean"] = float(cpi[mask].mean())
                    row[f"{cat}_sur"] = float(imp_c / sur_c) if sur_c > 0 else 0.0
                    row[f"{cat}_deficit_coverage"] = float(imp_c / def_c) if def_c > 0 else 0.0
                    row[f"{cat}_transfer_volume_twh"] = float(imp_c / 1e9)
        return row


    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("Demand response — per-timestep inter-community energy sharing")
    print("=" * 70)
    t0_total = time.time()

    # ── 1. Load data & build mesh index ──
    print("\n[1/5] Loading data & building mesh index...")
    season = pd.read_parquet(STATIC_DIR / "season_factors.parquet")
    sf_map = dict(zip(season["month"], season["season_factor"]))
    DEM_NORM = sum(calendar.monthrange(2024, m)[1] * sf_map[m] for m in range(1, 13))
    print(f"  DEM_NORM = {DEM_NORM:.4f}")

    arv2 = pd.read_parquet(FORMAL_DIR / "technical_potential.parquet")
    dem_all = pd.read_parquet(STATIC_DIR / "dem_shapes.parquet")
    dist = pd.read_parquet(COORD_DIR / "sparse_distances.parquet")
    base = pd.read_parquet(COORD_DIR / "mesh_attributes.parquet")
    geo = pd.read_parquet(GEO_PATH)

    base_meshes = set(arv2["mesh_code"])
    gen_meshes = set()
    for gf in sorted(glob.glob(str(STATIC_DIR / "gen_shapes" / "month=*.parquet"))):
        gs = pd.read_parquet(gf, columns=["mesh_code"])
        gen_meshes.update(gs["mesh_code"].unique())
    dem_meshes = set(dem_all["mesh_code"])
    common = sorted(base_meshes & gen_meshes & dem_meshes)
    mesh_to_idx = {m: i for i, m in enumerate(common)}
    n = len(common)
    print(f"  Valid meshes: {n:,}")

    # Align arrays
    arv2_idx = arv2.set_index("mesh_code").loc[common]
    potential = arv2_idx["potential_kwh"].values.astype(np.float64)
    annual_dem = arv2_idx["annual_dem_kwh"].values.astype(np.float64)
    category_3 = arv2_idx["category_3"].values

    # ── 2. Build weight matrix ──
    print("\n[2/5] Building weight matrix...")
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
    W5 = csr_matrix((np.exp(-dd_new / LAMBDA_PRIMARY).astype(np.float32),
                      (di_new, dj_new)), shape=(n, n), dtype=np.float32)
    print(f"  λ={LAMBDA_PRIMARY:.1f}km: {W5.nnz:,} non-zeros ({len(dd_new):,} pairs)")

    # ── 3. Build demand shape lookup ──
    print("\n[3/5] Building demand shape lookup...")
    dem_mesh_set = set(common)
    dem_filtered = dem_all[dem_all["mesh_code"].isin(dem_mesh_set)]
    dem_lookup = {}
    for (month, dt, slot), group in dem_filtered.groupby(["month", "day_type", "slot"]):
        arr = np.zeros(n, dtype=np.float64)
        for m, v in zip(group["mesh_code"].values, group["dem_shape"].values):
            if m in mesh_to_idx:
                arr[mesh_to_idx[m]] = v
        dem_lookup[(month, dt, slot)] = arr
    print(f"  dem_lookup: {len(dem_lookup)} keys (12m × 2dt × 48slots)")

    # Pre-compute evening/midday totals per (month, day_type)
    month_dt_evening = {}
    month_dt_midday = {}
    for month in range(1, 13):
        for dt in ["wd", "we"]:
            eve_sum = np.zeros(n, dtype=np.float64)
            mid_sum = np.zeros(n, dtype=np.float64)
            for slot in EVENING_SLOTS:
                key = (month, dt, slot)
                if key in dem_lookup:
                    eve_sum += dem_lookup[key]
            for slot in MIDDAY_SLOTS:
                key = (month, dt, slot)
                if key in dem_lookup:
                    mid_sum += dem_lookup[key]
            month_dt_evening[(month, dt)] = eve_sum
            month_dt_midday[(month, dt)] = mid_sum

    # ── 4. Temporal simulation ──
    print("\n[4/5] Running per-timestep DR simulation...")

    # Calendar
    day_offset = {1: 0, 2: 31, 3: 60, 4: 91, 5: 121, 6: 152,
                  7: 182, 8: 213, 9: 244, 10: 274, 11: 305, 12: 335}
    dt_lookup = {}
    for m in range(1, 13):
        for d in range(1, calendar.monthrange(2024, m)[1] + 1):
            doy = day_offset[m] + (d - 1)
            wd = calendar.weekday(2024, m, d)
            dt_lookup[doy] = "we" if wd >= 5 else "wd"

    # Technical-potentials to run
    ALPHA_LABELS = {"tp030": 0.30, "tp015": 0.15, "tp100": 1.00}

    # Initialize accumulators
    # Structure: acc[shift_r][ad_label] = {arrays}
    acc = {}
    for sr in SHIFT_RATIOS:
        acc[sr] = {}
        ads = ALPHA_LABELS if sr in ENDPOINT_SHIFTS else {"tp030": 0.30}
        for ad_label, alpha in ads.items():
            acc[sr][ad_label] = {
                "sc": np.zeros(n, dtype=np.float64),
                "surplus": np.zeros(n, dtype=np.float64),
                "demand": np.zeros(n, dtype=np.float64),
                "imp_5": np.zeros(n, dtype=np.float64),
                "imp_inf": np.zeros(n, dtype=np.float64),
                "served_5": np.zeros(n, dtype=np.float64),
                "served_inf": np.zeros(n, dtype=np.float64),
                "active": 0, "skipped": 0,
            }

    total_timesteps = 0

    for month in range(1, 13):
        t_m = time.time()

        # Load gen_shapes for this month
        gen_file = STATIC_DIR / "gen_shapes" / f"month={month:02d}.parquet"
        gen_month = pd.read_parquet(gen_file)
        gen_month = gen_month[gen_month["mesh_code"].isin(common)]
        gen_month["mesh_idx"] = gen_month["mesh_code"].map(mesh_to_idx)
        gen_month = gen_month.dropna(subset=["mesh_idx"])
        gen_month["mesh_idx"] = gen_month["mesh_idx"].astype(np.int32)

        gen_by_day_slot = {}
        for (day, slot), group in gen_month.groupby(["day", "slot"]):
            arr = np.zeros(n, dtype=np.float64)
            idx = group["mesh_idx"].values
            val = group["gen_shape"].values
            arr[idx] = val
            gen_by_day_slot[(int(day), int(slot))] = arr
        all_zero = np.zeros(n, dtype=np.float64)

        n_days = calendar.monthrange(2024, month)[1]
        month_active = 0

        for day in range(1, n_days + 1):
            doy = day_offset[month] + (day - 1)
            gen_doy = day_offset[month] + day          # gen_shapes uses DOY
            dt = dt_lookup[doy]

            for slot in range(N_SLOTS):
                total_timesteps += 1

                gs_arr = gen_by_day_slot.get((gen_doy, slot), all_zero)
                dem_key = (month, dt, slot)
                if dem_key not in dem_lookup:
                    continue
                dem_base = dem_lookup[dem_key]
                eve_total = month_dt_evening.get((month, dt), np.zeros(n))
                mid_total = month_dt_midday.get((month, dt), np.zeros(n))

                for sr in SHIFT_RATIOS:
                    # Shifted demand
                    if sr == 0.0:
                        dem_s = dem_base
                    elif slot in EVENING_SLOTS:
                        dem_s = dem_base * (1.0 - sr)
                    elif slot in MIDDAY_SLOTS:
                        shift_amt = eve_total * sr
                        safe_mid = np.where(mid_total > 0, mid_total, 1.0)
                        dem_s = dem_base + shift_amt * (dem_base / safe_mid)
                    else:
                        dem_s = dem_base

                    ads_to_run = ALPHA_LABELS if sr in ENDPOINT_SHIFTS else {"tp030": 0.30}

                    for ad_label, alpha in ads_to_run.items():
                        a = acc[sr][ad_label]

                        gen = potential * gs_arr * alpha
                        dem = annual_dem * dem_s / DEM_NORM

                        a["demand"] += dem
                        sc = np.minimum(gen, dem)
                        a["sc"] += sc
                        surplus = np.maximum(gen - dem, 0)
                        a["surplus"] += surplus

                        sur_sum = surplus.sum()
                        if sur_sum <= 0:
                            a["skipped"] += 1
                            a["served_5"] += sc
                            a["served_inf"] += sc
                            continue

                        a["active"] += 1
                        month_active += 1

                        deficit = np.maximum(dem - gen, 0)

                        # λ = 5 km
                        _, imp5 = conditional_donor_normalized(W5, surplus, deficit)
                        a["imp_5"] += imp5
                        a["served_5"] += (sc + imp5)

                        # λ = ∞
                        total_s = sur_sum
                        total_d = deficit.sum()
                        ratio = min(1.0, total_s / total_d) if total_d > 0 else 0.0
                        a["imp_inf"] += deficit * ratio
                        a["served_inf"] += (sc + deficit * ratio)

        print(f"  Month {month:2d}: {n_days}d, active steps={month_active}, "
              f"time={time.time()-t_m:.1f}s", flush=True)

    # ── 5. Compile results ──
    print("\n[5/5] Compiling results...")

    all_results = []
    all_mesh_frames = []
    mesh_codes_arr = np.array(common, dtype=np.int64)

    for sr in SHIFT_RATIOS:
        ads_to_run = ALPHA_LABELS if sr in ENDPOINT_SHIFTS else {"tp030": 0.30}
        for ad_label, alpha in ads_to_run.items():
            a = acc[sr][ad_label]
            gen_eff = potential * alpha

            ssr = np.where(a["demand"] > 0, a["sc"] / a["demand"], 0.0)
            ci_per_mesh = np.where(gen_eff > 0, a["sc"] / gen_eff, 0.0)
            donor_mask = gen_eff > a["demand"]

            deficit_annual = np.maximum(a["demand"] - a["sc"], 0)

            # λ = 5 km
            cssr_5 = np.where(a["demand"] > 0, a["served_5"] / a["demand"], 0.0)
            cpi_5 = cssr_5 - ssr
            row_5 = build_summary(ssr, cssr_5, cpi_5, a["imp_5"], deficit_annual,
                                  a["surplus"], a["demand"], gen_eff, a["sc"],
                                  ci_per_mesh, donor_mask, category_3,
                                  {"shift_ratio": sr, "shift_label": SHIFT_LABELS[sr],
                                   "alpha_label": ad_label, "alpha": alpha,
                                   "lambda_km": LAMBDA_PRIMARY, "lambda_label": str(LAMBDA_PRIMARY)})
            all_results.append(row_5)

            # λ = ∞
            cssr_inf = np.where(a["demand"] > 0, a["served_inf"] / a["demand"], 0.0)
            cpi_inf = cssr_inf - ssr
            row_inf = build_summary(ssr, cssr_inf, cpi_inf, a["imp_inf"], deficit_annual,
                                    a["surplus"], a["demand"], gen_eff, a["sc"],
                                    ci_per_mesh, donor_mask, category_3,
                                    {"shift_ratio": sr, "shift_label": SHIFT_LABELS[sr],
                                     "alpha_label": ad_label, "alpha": alpha,
                                     "lambda_km": float("inf"), "lambda_label": "inf"})
            all_results.append(row_inf)

            # Per-mesh (λ=5km)
            all_mesh_frames.append(pd.DataFrame({
                "shift_ratio": sr, "shift_label": SHIFT_LABELS[sr],
                "alpha_label": ad_label,
                "mesh_code": mesh_codes_arr, "category_3": category_3,
                "gen_kwh": gen_eff, "dem_kwh": a["demand"],
                "self_consumed_kwh": a["sc"], "ci": ci_per_mesh,
                "ssr": ssr, "cssr_l5": cssr_5, "cpi_l5": cpi_5,
                "imported_l5": a["imp_5"], "surplus": a["surplus"],
            }))

            print(f"  {SHIFT_LABELS[sr]} {ad_label}: CI={row_5['ci_system']:.4f} "
                  f"SSR={ssr.mean()*100:.1f}% CPI={cpi_5.mean()*100:.2f}pp "
                  f"SUR={row_5['sur']*100:.1f}% inter-community energy sharing={a['imp_5'].sum()/1e9:.2f}TWh "
                  f"active={a['active']} skipped={a['skipped']}")

    # ── Save ──
    print("\n── Saving ──")
    df = pd.DataFrame(all_results)
    mesh_df = pd.concat(all_mesh_frames, ignore_index=True)

    # Fix types for parquet
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
    for col in mesh_df.columns:
        if mesh_df[col].dtype == object:
            mesh_df[col] = mesh_df[col].astype(str)

    df.to_parquet(OUT_DIR / "demand_response.parquet", index=False)
    mesh_df.to_parquet(OUT_DIR / "demand_response_mesh.parquet", index=False)
    print(f"  demand_response.parquet: {len(df)} rows")
    print(f"  demand_response_mesh.parquet: {len(mesh_df)} rows")

    # ── Key results ──
    print("\n" + "=" * 70)
    print("KEY RESULTS: Demand Response → inter-community energy sharing (Per-Timestep)")
    print("=" * 70)

    lam5 = df[df["lambda_km"] == LAMBDA_PRIMARY]
    baseline = lam5[(lam5["shift_ratio"] == 0.0) & (lam5["alpha_label"] == "tp030")]
    if len(baseline):
        b = baseline.iloc[0]
        print(f"\n  Baseline (tp030, no DR): CI={b['ci_system']:.4f} CPI={b['cpi_mean']*100:.2f}pp "
              f"SSR={b['ssr_mean']*100:.1f}% SUR={b['sur']*100:.1f}% "
              f"def_cov={b['deficit_coverage']*100:.1f}% inter-community energy sharing={b['transfer_volume_twh']:.2f}TWh")
        for shift_r in [0.10, 0.20, 0.30]:
            r = lam5[(lam5["shift_ratio"] == shift_r) & (lam5["alpha_label"] == "tp030")]
            if len(r):
                r = r.iloc[0]
                dci = (r["ci_system"] - b["ci_system"]) / b["ci_system"] * 100
                dcpi = (r["cpi_mean"] - b["cpi_mean"]) * 100
                print(f"  DR {shift_r*100:.0f}%: CI={r['ci_system']:.4f} (Δ+{dci:.1f}%) "
                      f"CPI={r['cpi_mean']*100:.2f}pp (Δ{dcpi:+.2f}pp) "
                      f"SUR={r['sur']*100:.1f}% inter-community energy sharing={r['transfer_volume_twh']:.2f}TWh")

    elapsed = time.time() - t0_total
    print(f"\n  Total timesteps: {total_timesteps:,}")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    print(f"{'='*70}")



def run_targeted_deployment():
    import pandas as pd
    import numpy as np
    from scipy.sparse import csr_matrix, diags
    from pathlib import Path
    import time, sys

    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    PROJ_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJ_ROOT / "data" / "input" / "mesh"
    SIM_DIR = PROJ_ROOT / "code" / "simulation"
    if str(SIM_DIR) not in sys.path:
        sys.path.insert(0, str(SIM_DIR))

    from allocation_rules import conditional_donor_normalized

    GEO_PATH = Path(__file__).resolve().parent / "results" / "scenario_geometry.parquet"
    OUT_DIR = Path(__file__).resolve().parent / "results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    LAMBDAS_KM = [1.0, 5.0, 10.0]
    LAMBDA_INF_KEY = "inf"
    LAMBDA_ALL = LAMBDAS_KM + [LAMBDA_INF_KEY]

    UNIFORM_LEVELS = {"tp015": 0.15, "tp030": 0.30}
    TARGETED_RATES = {"metropolitan_core": 0.40, "regional_city": 0.65, "rural": 0.85}
    CAT_LIST = ["metropolitan_core", "regional_city", "rural"]

    # All urban form scenarios (with morphology-only variants)
    ALL_SCENARIOS = ["S0", "S1", "S1m", "S2", "S2m", "S3", "S3m"]


    def gini(x):
        x = np.asarray(x, dtype=np.float64)
        if len(x) == 0 or x.sum() == 0:
            return 0.0
        x_sorted = np.sort(x)
        n = len(x)
        return (2 * np.sum(np.arange(1, n + 1) * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n


    def build_W_raw(di, dj, dd, lam, n):
        """Raw distance-decay weight matrix (NOT column-normalized)."""
        w = np.exp(-dd / lam).astype(np.float32)
        return csr_matrix((w, (di, dj)), shape=(n, n), dtype=np.float32)


    def run_annual_sharing(gen, dem, W_dict):
        """Annual-balance inter-community energy sharing with conditional-donor-normalized allocation.
        Returns per-mesh metrics."""
        surplus = np.maximum(gen - dem, 0)
        deficit = np.maximum(dem - gen, 0)
        self_consumed = np.minimum(gen, dem)
        ssr = np.where(dem > 0, self_consumed / dem, 0.0)

        results = {}
        for lam_key, W_raw in W_dict.items():
            if isinstance(lam_key, str) and lam_key == "inf":
                total_s = surplus.sum()
                total_d = deficit.sum()
                imported = deficit * min(1.0, total_s / total_d) if total_d > 0 else np.zeros_like(deficit)
            else:
                _, imported = conditional_donor_normalized(W_raw, surplus, deficit)

            served = self_consumed + imported
            cssr = np.where(dem > 0, served / dem, 0.0)
            cpi = cssr - ssr
            sur_val = imported.sum() / surplus.sum() if surplus.sum() > 0 else 0.0

            results[lam_key] = {
                "cssr": cssr, "cpi": cpi, "imported": imported,
                "sur": sur_val, "served": served,
            }
        return {"ssr": ssr, "self_consumed": self_consumed,
                "surplus": surplus, "deficit": deficit, **results}


    def compute_flow_matrix(surplus, imported, categories):
        """Approximate inter-category energy flow by surplus-share attribution."""
        n_cats = len(CAT_LIST)
        cat_idx = {c: i for i, c in enumerate(CAT_LIST)}
        flow = np.zeros((n_cats, n_cats))
        surplus_by_cat = {c: surplus[categories == c].sum() for c in CAT_LIST}
        total_surplus = sum(surplus_by_cat.values())
        if total_surplus == 0:
            return flow
        for to_cat in CAT_LIST:
            to_mask = categories == to_cat
            if not to_mask.any():
                continue
            imp = imported[to_mask].sum()
            for from_cat in CAT_LIST:
                if surplus_by_cat[from_cat] > 0:
                    flow[cat_idx[from_cat], cat_idx[to_cat]] += imp * surplus_by_cat[from_cat] / total_surplus
        return flow


    # ══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("Targeted PV technical-potential vs urban form (conditional-donor-normalized inter-community energy sharing)")
    print("=" * 70)
    t0 = time.time()

    # ── 1. Load data ──
    print("\n[1/4] Loading data...")
    geo = pd.read_parquet(GEO_PATH)
    dist = pd.read_parquet(DATA_DIR / "sparse_distances.parquet")
    base = pd.read_parquet(DATA_DIR / "mesh_attributes.parquet")

    # Determine which scenarios actually exist in geometry
    available_scenarios = [s for s in ALL_SCENARIOS if f"gen_{s}_kwh" in geo.columns]
    print(f"  Available scenarios: {available_scenarios}")
    n_mesh = len(geo)
    category_3 = geo["category_3"].values

    # ── 2. Build weight matrices ──
    print("\n[2/4] Building distance-decay weight matrices (W_raw, for conditional-donor-normalized)...")

    geo_mesh_to_idx = {int(m): i for i, m in enumerate(geo["mesh_code"].values)}
    orig_mesh_arr = base["mesh_code"].values.astype(np.int64)
    dist_i = dist["mesh_idx_i"].values.astype(np.int32)
    dist_j = dist["mesh_idx_j"].values.astype(np.int32)
    dist_d = dist["distance_km"].values.astype(np.float32)
    dist_mesh_i = orig_mesh_arr[dist_i]
    dist_mesh_j = orig_mesh_arr[dist_j]

    keep_i = np.array([int(m) in geo_mesh_to_idx for m in dist_mesh_i])
    keep_j = np.array([int(m) in geo_mesh_to_idx for m in dist_mesh_j])
    keep = keep_i & keep_j

    di_new = np.array([geo_mesh_to_idx[int(m)] for m in dist_mesh_i[keep]], dtype=np.int32)
    dj_new = np.array([geo_mesh_to_idx[int(m)] for m in dist_mesh_j[keep]], dtype=np.int32)
    dd_new = dist_d[keep]
    n = len(geo)
    print(f"  Distance pairs: {len(dd_new):,}")

    W = {}
    for lam in LAMBDAS_KM:
        W[lam] = build_W_raw(di_new, dj_new, dd_new, lam, n)
        print(f"  λ={lam:.1f} km: {W[lam].nnz:,} non-zeros (W_raw, donor→receiver)")
    W["inf"] = "inf"  # sentinel

    # ── 3. Run all combinations ──
    print("\n[3/4] Running scenario × technical potential inter-community energy sharing...")

    all_results = []
    all_flows = []
    all_mesh = []

    for scenario in available_scenarios:
        gen_col = f"gen_{scenario}_kwh"
        dem_col = f"dem_{scenario}_kwh"
        base_gen = geo[gen_col].values.astype(np.float64)
        base_dem = geo[dem_col].values.astype(np.float64)
        donor_mask = base_gen > base_dem

        # --- Uniform technical potential ---
        for strat_name, alpha in UNIFORM_LEVELS.items():
            gen = base_gen * alpha
            r = run_annual_sharing(gen, base_dem, W)

            for lam_key in LAMBDA_ALL:
                lam_val = np.inf if isinstance(lam_key, str) else lam_key
                lam_lbl = str(lam_key)
                cpi_arr = r[lam_key]["cpi"]
                cssr_arr = r[lam_key]["cssr"]
                imp_arr = r[lam_key]["imported"]
                deficit_arr = r["deficit"]
                deficit_total = deficit_arr.sum()
                import_total = imp_arr.sum()
                surplus_total = r["surplus"].sum()

                row = {
                    "urban_form_scenario": scenario, "strategy": strat_name,
                    "technical_potential_type": "uniform", "lambda_km": lam_val,
                    "lambda_label": lam_lbl,
                    "uniform_alpha": alpha,
                    "total_gen_twh": gen.sum() / 1e9,
                    "total_dem_twh": base_dem.sum() / 1e9,
                    "gen_dem_ratio": gen.sum() / max(base_dem.sum(), 1),
                    "donor_pct": donor_mask.mean(),
                    # System-level multi-metrics
                    "ssr_mean": float(r["ssr"].mean()),
                    "cssr_mean": float(cssr_arr.mean()),
                    "cpi_mean": float(cpi_arr.mean()),
                    "cpi_median": float(np.median(cpi_arr)),
                    "sur": float(r[lam_key]["sur"]),
                    "deficit_coverage": float(import_total / deficit_total) if deficit_total > 0 else 0.0,
                    "transfer_volume_twh": float(import_total / 1e9),
                    "total_surplus_twh": float(surplus_total / 1e9),
                    "gini_cssr": float(gini(cssr_arr)),
                    "gini_cpi": float(gini(cpi_arr)),
                }
                for cat in CAT_LIST:
                    mask = category_3 == cat
                    if mask.any():
                        imp_cat = imp_arr[mask].sum()
                        def_cat = deficit_arr[mask].sum()
                        row[f"{cat}_ssr_mean"] = float(r["ssr"][mask].mean())
                        row[f"{cat}_cssr_mean"] = float(cssr_arr[mask].mean())
                        row[f"{cat}_cpi_mean"] = float(cpi_arr[mask].mean())
                        row[f"{cat}_sur"] = float(imp_cat / max(r["surplus"][mask].sum(), 1))
                        row[f"{cat}_deficit_coverage"] = float(imp_cat / def_cat) if def_cat > 0 else 0.0
                        row[f"{cat}_donor_pct"] = float(donor_mask[mask].mean())
                        row[f"{cat}_gini_cssr"] = float(gini(cssr_arr[mask]))
                all_results.append(row)

            # Per-mesh (λ=5km only)
            all_mesh.append(pd.DataFrame({
                "urban_form_scenario": scenario, "strategy": strat_name, "technical_potential_type": "uniform",
                "mesh_code": geo["mesh_code"].values,
                "category_3": category_3,
                "gen_kwh": gen, "dem_kwh": base_dem,
                "ssr": r["ssr"],
                "cssr_l5": r[5.0]["cssr"], "cpi_l5": r[5.0]["cpi"],
                "imported_l5": r[5.0]["imported"], "surplus": r["surplus"],
            }))

            # Energy flow (λ=5km only)
            flow_mat = compute_flow_matrix(r["surplus"], r[5.0]["imported"], category_3)
            for fi, from_cat in enumerate(CAT_LIST):
                for ti, to_cat in enumerate(CAT_LIST):
                    all_flows.append({
                        "urban_form_scenario": scenario, "strategy": strat_name,
                        "technical_potential_type": "uniform", "lambda_km": 5.0,
                        "from_category": from_cat, "to_category": to_cat,
                        "energy_twh": flow_mat[fi, ti] / 1e9,
                    })

        # --- Targeted technical potential ---
        targeted_alpha = np.array([TARGETED_RATES.get(c, 0.65) for c in category_3])
        gen_targeted = base_gen * targeted_alpha
        r = run_annual_sharing(gen_targeted, base_dem, W)

        for lam_key in LAMBDA_ALL:
            lam_val = np.inf if isinstance(lam_key, str) else lam_key
            lam_lbl = str(lam_key)
            cpi_arr = r[lam_key]["cpi"]
            cssr_arr = r[lam_key]["cssr"]
            imp_arr = r[lam_key]["imported"]
            deficit_arr = r["deficit"]
            deficit_total = deficit_arr.sum()
            import_total = imp_arr.sum()
            surplus_total = r["surplus"].sum()

            row = {
                "urban_form_scenario": scenario, "strategy": "targeted",
                "technical_potential_type": "targeted", "lambda_km": lam_val,
                "lambda_label": lam_lbl,
                "uniform_alpha": targeted_alpha.mean(),
                "total_gen_twh": gen_targeted.sum() / 1e9,
                "total_dem_twh": base_dem.sum() / 1e9,
                "gen_dem_ratio": gen_targeted.sum() / max(base_dem.sum(), 1),
                "donor_pct": donor_mask.mean(),
                "ssr_mean": float(r["ssr"].mean()),
                "cssr_mean": float(cssr_arr.mean()),
                "cpi_mean": float(cpi_arr.mean()),
                "cpi_median": float(np.median(cpi_arr)),
                "sur": float(r[lam_key]["sur"]),
                "deficit_coverage": float(import_total / deficit_total) if deficit_total > 0 else 0.0,
                "transfer_volume_twh": float(import_total / 1e9),
                "total_surplus_twh": float(surplus_total / 1e9),
                "gini_cssr": float(gini(cssr_arr)),
                "gini_cpi": float(gini(cpi_arr)),
            }
            for cat in CAT_LIST:
                mask = category_3 == cat
                if mask.any():
                    imp_cat = imp_arr[mask].sum()
                    def_cat = deficit_arr[mask].sum()
                    row[f"{cat}_ssr_mean"] = float(r["ssr"][mask].mean())
                    row[f"{cat}_cssr_mean"] = float(cssr_arr[mask].mean())
                    row[f"{cat}_cpi_mean"] = float(cpi_arr[mask].mean())
                    row[f"{cat}_sur"] = float(imp_cat / max(r["surplus"][mask].sum(), 1))
                    row[f"{cat}_deficit_coverage"] = float(imp_cat / def_cat) if def_cat > 0 else 0.0
                    row[f"{cat}_donor_pct"] = float(donor_mask[mask].mean())
                    row[f"{cat}_gini_cssr"] = float(gini(cssr_arr[mask]))
            all_results.append(row)

        all_mesh.append(pd.DataFrame({
            "urban_form_scenario": scenario, "strategy": "targeted", "technical_potential_type": "targeted",
            "mesh_code": geo["mesh_code"].values,
            "category_3": category_3,
            "gen_kwh": gen_targeted, "dem_kwh": base_dem,
            "targeted_alpha": targeted_alpha,
            "ssr": r["ssr"],
            "cssr_l5": r[5.0]["cssr"], "cpi_l5": r[5.0]["cpi"],
            "imported_l5": r[5.0]["imported"], "surplus": r["surplus"],
        }))

        flow_mat = compute_flow_matrix(r["surplus"], r[5.0]["imported"], category_3)
        for fi, from_cat in enumerate(CAT_LIST):
            for ti, to_cat in enumerate(CAT_LIST):
                all_flows.append({
                    "urban_form_scenario": scenario, "strategy": "targeted",
                    "technical_potential_type": "targeted", "lambda_km": 5.0,
                    "from_category": from_cat, "to_category": to_cat,
                    "energy_twh": flow_mat[fi, ti] / 1e9,
                })

        equiv_alpha = targeted_alpha.mean()
        print(f"  {scenario}: targeted gen={gen_targeted.sum()/1e9:.1f} TWh, equiv_α={equiv_alpha:.3f}")

    # ── 4. Compile & save ──
    print("\n[4/4] Saving results...")

    df = pd.DataFrame(all_results)
    flows_df = pd.DataFrame(all_flows)
    mesh_df = pd.concat(all_mesh, ignore_index=True)

    df.to_parquet(OUT_DIR / "targeted_deployment.parquet", index=False)
    flows_df.to_parquet(OUT_DIR / "energy_flows.parquet", index=False)
    mesh_df.to_parquet(OUT_DIR / "targeted_deployment_mesh.parquet", index=False)

    print(f"  targeted_deployment.parquet: {len(df)} rows")
    print(f"  energy_flows.parquet: {len(flows_df)} rows")
    print(f"  targeted_deployment_mesh.parquet: {len(mesh_df)} rows")

    # ── Key results ──
    print("\n" + "=" * 70)
    print("KEY RESULTS (λ=5 km)")
    print("=" * 70)

    lam5 = df[df["lambda_km"] == 5.0]

    # Head-to-head: S2_tp015 vs S0_targeted
    s0_tp015 = lam5[(lam5["urban_form_scenario"] == "S0") & (lam5["strategy"] == "tp015")]
    s2_tp015 = lam5[(lam5["urban_form_scenario"] == "S2") & (lam5["strategy"] == "tp015")]
    s0_targeted = lam5[(lam5["urban_form_scenario"] == "S0") & (lam5["strategy"] == "targeted")]

    if len(s0_tp015) and len(s2_tp015) and len(s0_targeted):
        s0_tp015 = s0_tp015.iloc[0]
        s2_tp015 = s2_tp015.iloc[0]
        s0_targeted = s0_targeted.iloc[0]

        uf_delta = (s2_tp015["cpi_mean"] - s0_tp015["cpi_mean"]) * 100
        as_delta = (s0_targeted["cpi_mean"] - s0_tp015["cpi_mean"]) * 100

        print(f"\n  Technical-potential strategy effect (S0_targeted − S0_tp015): {as_delta:+.2f} pp CPI")
        print(f"  Urban form effect (S2_tp015 − S0_tp015):          {uf_delta:+.2f} pp CPI")
        print(f"  Ratio (urban_form / technical potential_strategy): {uf_delta/as_delta:.2f}×")

        print(f"\n  S2_tp015:     CPI={s2_tp015['cpi_mean']*100:.2f} pp  "
              f"gen={s2_tp015['total_gen_twh']:.1f} TWh  g/d={s2_tp015['gen_dem_ratio']:.2f}  "
              f"def_cov={s2_tp015['deficit_coverage']*100:.1f}%")
        print(f"  S0_targeted: CPI={s0_targeted['cpi_mean']*100:.2f} pp  "
              f"gen={s0_targeted['total_gen_twh']:.1f} TWh  g/d={s0_targeted['gen_dem_ratio']:.2f}  "
              f"def_cov={s0_targeted['deficit_coverage']*100:.1f}%")

    # Category breakdown
    print("\n── CPI by Category (λ=5 km) ──")
    for sc in ["S0", "S2"]:
        for st in ["tp015", "tp030", "targeted"]:
            row = lam5[(lam5["urban_form_scenario"] == sc) & (lam5["strategy"] == st)]
            if len(row):
                r = row.iloc[0]
                cats_str = " | ".join(
                    [f"{c}={r[f'{c}_cpi_mean']*100:.2f}pp" for c in CAT_LIST]
                )
                print(f"  {sc}_{st:10s}: {cats_str}")

    # Urban form ΔCPI
    print("\n── Urban Form ΔCPI (S2 − S0) by Category (λ=5 km) ──")
    for st in ["tp015", "tp030", "targeted"]:
        s0_r = lam5[(lam5["urban_form_scenario"] == "S0") & (lam5["strategy"] == st)]
        s2_r = lam5[(lam5["urban_form_scenario"] == "S2") & (lam5["strategy"] == st)]
        if len(s0_r) and len(s2_r):
            s0_r, s2_r = s0_r.iloc[0], s2_r.iloc[0]
            deltas = " | ".join(
                [f"{c}: {(s2_r[f'{c}_cpi_mean']-s0_r[f'{c}_cpi_mean'])*100:+.2f} pp" for c in CAT_LIST]
            )
            print(f"  {st}: {deltas}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Targeted technical potential v2 complete. Elapsed: {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_demand_response()
    run_targeted_deployment()

