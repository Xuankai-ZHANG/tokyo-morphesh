# -*- coding: utf-8 -*-
"""
inter-community energy sharing Metrics v1 — System-level and distributional metrics
=========================================================
Computes from existing CSSR summary outputs (no re-simulation needed).

Output structure:
  1. metrics_system.parquet — per scenario × λ system-level table
  2. metrics_benefit_incidence.parquet — SSR decile × inter-community energy sharing benefit allocation

Metrics computed:
  Layer 1 (Circulation):
    - transfer_volume_twh: total energy transferred via inter-community energy sharing
    - transfer_penetration: Σ imported / Σ demand
    - donor_export_rate: Σ exported / Σ surplus (= SUR, re-framed)
    - receiver_coverage: mean(imported / deficit) for receivers

  Layer 2 (Distribution):
    - sigma_convergence: σ(log CSSR) / σ(log SSR)  (< 1 = equalizing)
    - gini_ssr, gini_cssr, gini_reduction: ΔG = G(SSR) - G(CSSR)
    - gini_reduction_ratio: ΔG / G(SSR)

  Layer 3 (Incidence):
    - Benefit incidence by SSR decile: share of total inter-community energy sharing flow to each decile
    - Progressivity check: bottom 20% SSR share vs top 20%

"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ── Paths ──────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJ_ROOT / "data" / "results" / "summary"
DATA_DIR = PROJ_ROOT / "data" / "input" / "mesh"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_LABELS = ["0.1", "0.2", "0.5", "1.0", "2.0", "5.0", "10.0", "20.0", "inf"]
SCENARIOS = ["tp010", "tp015", "tp020", "tp030", "tp040", "tp050", "tp100"]


# ── Utilities ───────────────────────────────────────────────────────────

def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n


def theil_l(x: np.ndarray) -> float:
    """Theil L (mean log deviation)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[(~np.isnan(x)) & (x > 0)]
    if len(x) == 0:
        return 0.0
    mu = x.mean()
    return np.mean(np.log(mu / x))


def load_trajectory() -> pd.DataFrame:
    path = RESULT_DIR / "trajectory_summary.parquet"
    if not path.exists():
        raise FileNotFoundError(f"trajectory_summary.parquet not found at {path}")
    return pd.read_parquet(path)


def load_cssr_summary(scenario: str) -> pd.DataFrame:
    path = RESULT_DIR / f"cssr_summary_{scenario}.parquet"
    return pd.read_parquet(path)


# ── Layer 1: System-level Circulation Metrics ───────────────────────────

def compute_system_metrics() -> pd.DataFrame:
    """Compute system-level metrics per scenario × λ.

    All metrics computed from per-mesh CSSR summary data (no dependency on
    trajectory_summary, which has incomplete scenario coverage).
    imported_i ≈ cpi_i × annual_dem_i  (CPI is in pp of demand)
    """
    rows = []

    for scenario in SCENARIOS:
        try:
            cssr_df = load_cssr_summary(scenario)
        except FileNotFoundError:
            print(f"  [SKIP] {scenario}: CSSR summary not found")
            continue

        dem_total = cssr_df["annual_dem_kwh"].sum()
        surplus_total = cssr_df["annual_surplus_kwh"].sum()
        annual_dem = cssr_df["annual_dem_kwh"].values

        for lam in LAMBDA_LABELS:
            cpi_col = f"cpi_{lam}"
            cssr_col = f"cssr_{lam}"
            if cpi_col not in cssr_df.columns:
                continue

            cpi = cssr_df[cpi_col].values
            cssr = cssr_df[cssr_col].values
            ssr = cssr_df["ssr"].values

            # Compute inter-community energy sharing volume directly from per-mesh CPI × demand
            imported_per_mesh = cpi * annual_dem
            imported_total = imported_per_mesh.sum()
            sur_val = imported_total / surplus_total if surplus_total > 0 else 0.0

            # Receiver coverage: mean(imported / deficit)
            is_recv = cssr_df["is_receiver"].values
            deficit = cssr_df["annual_deficit_kwh"].values
            recv_coverage = (
                np.mean(imported_per_mesh[is_recv] / np.maximum(deficit[is_recv], 1.0))
                if is_recv.sum() > 0 else 0.0
            )

            # σ-convergence
            log_ssr = np.log(np.maximum(ssr, 1e-6))
            log_cssr = np.log(np.maximum(cssr, 1e-6))
            sigma_ssr = np.std(log_ssr)
            sigma_cssr = np.std(log_cssr)
            sigma_conv = sigma_cssr / sigma_ssr if sigma_ssr > 0 else np.nan

            # Gini
            g_ssr = gini(ssr)
            g_cssr = gini(cssr)
            g_reduction = g_ssr - g_cssr
            g_reduction_ratio = g_reduction / g_ssr if g_ssr > 0 else 0.0

            # Theil
            t_ssr = theil_l(ssr)
            t_cssr = theil_l(cssr)

            # inter-community energy sharing penetration
            transfer_penetration = imported_total / dem_total if dem_total > 0 else 0.0

            alpha = cssr_df["alpha"].iloc[0] if "alpha" in cssr_df.columns else np.nan

            rows.append({
                "alpha_label": scenario,
                "alpha": alpha,
                "lambda_label": lam,
                # Circulation
                "transfer_volume_twh": imported_total / 1e9,
                "transfer_penetration": transfer_penetration,
                "sur": sur_val,
                "receiver_coverage": recv_coverage,
                # Equalization
                "sigma_ssr": sigma_ssr,
                "sigma_cssr": sigma_cssr,
                "sigma_convergence": sigma_conv,
                "gini_ssr": g_ssr,
                "gini_cssr": g_cssr,
                "gini_reduction": g_reduction,
                "gini_reduction_ratio": g_reduction_ratio,
                "theil_ssr": t_ssr,
                "theil_cssr": t_cssr,
                # System totals
                "total_demand_twh": dem_total / 1e9,
                "total_surplus_twh": surplus_total / 1e9,
                "total_imported_twh": imported_total / 1e9,
                # By-product: shares
                "ssr_mean": float(np.mean(ssr)),
                "cssr_mean": float(np.mean(cssr)),
                "cpi_mean": float(np.mean(cpi)),
                "donor_fraction": float(cssr_df["is_donor"].mean()),
            })

    return pd.DataFrame(rows)


# ── Layer 2: Benefit Incidence by SSR Decile ────────────────────────────

def compute_benefit_incidence() -> pd.DataFrame:
    """For each scenario × λ, compute inter-community energy sharing benefit allocation by SSR decile."""
    rows = []

    for scenario in SCENARIOS:
        try:
            cssr_df = load_cssr_summary(scenario)
        except FileNotFoundError:
            continue

        # Assign SSR deciles (1 = lowest SSR, 10 = highest)
        cssr_df["ssr_decile"] = pd.qcut(
            cssr_df["ssr"], 10, labels=False, duplicates="drop"
        ) + 1

        for lam in LAMBDA_LABELS:
            cpi_col = f"cpi_{lam}"
            if cpi_col not in cssr_df.columns:
                continue

            total_imported_all = (
                cssr_df[cpi_col] * cssr_df["annual_dem_kwh"]
            ).sum()

            if total_imported_all <= 0:
                continue

            for decile in range(1, 11):
                mask = cssr_df["ssr_decile"] == decile
                if mask.sum() == 0:
                    continue

                sub = cssr_df[mask]
                share_of_total_flow = (
                    (sub[cpi_col] * sub["annual_dem_kwh"]).sum()
                    / total_imported_all
                )
                rows.append({
                    "alpha_label": scenario,
                    "lambda_label": lam,
                    "ssr_decile": decile,
                    "n_meshes": int(mask.sum()),
                    "mean_ssr": float(sub["ssr"].mean()),
                    "mean_cpi": float(sub[cpi_col].mean()),
                    "total_imported_twh": float(
                        (sub[cpi_col] * sub["annual_dem_kwh"]).sum() / 1e9
                    ),
                    "share_of_total_transfer_flow": float(share_of_total_flow),
                })

    return pd.DataFrame(rows)


# ── Layer 3: Category-level Cross-tab ───────────────────────────────────

def compute_category_flows() -> pd.DataFrame:
    """Donor/Receiver cross-tab by category_3 for each scenario."""
    rows = []

    for scenario in SCENARIOS:
        try:
            cssr_df = load_cssr_summary(scenario)
        except FileNotFoundError:
            continue

        for lam in LAMBDA_LABELS:
            cpi_col = f"cpi_{lam}"
            if cpi_col not in cssr_df.columns:
                continue

            for cat in ["metropolitan_core", "regional_city", "rural"]:
                mask = cssr_df["category_3"] == cat
                if mask.sum() == 0:
                    continue
                sub = cssr_df[mask]

                surplus_cat = sub["annual_surplus_kwh"].sum()
                deficit_cat = sub["annual_deficit_kwh"].sum()
                imported_cat = (sub[cpi_col] * sub["annual_dem_kwh"]).sum()

                total_surplus_all = cssr_df["annual_surplus_kwh"].sum()
                total_deficit_all = cssr_df["annual_deficit_kwh"].sum()

                rows.append({
                    "alpha_label": scenario,
                    "lambda_label": lam,
                    "category_3": cat,
                    "n_meshes": int(mask.sum()),
                    "n_donors": int(sub["is_donor"].sum()),
                    "n_receivers": int(sub["is_receiver"].sum()),
                    "surplus_twh": surplus_cat / 1e9,
                    "deficit_twh": deficit_cat / 1e9,
                    "imported_twh": imported_cat / 1e9,
                    "surplus_share": surplus_cat / total_surplus_all if total_surplus_all > 0 else 0.0,
                    "deficit_share": deficit_cat / total_deficit_all if total_deficit_all > 0 else 0.0,
                })

    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("inter-community energy sharing Metrics v1: System + Distribution")
    print("=" * 65)

    # ── System metrics ──
    print("\n[1/3] Computing system-level metrics...")
    sys_df = compute_system_metrics()
    out = RESULT_DIR / "metrics_system.parquet"
    sys_df.to_parquet(out, index=False)
    print(f"  -> {out.name}: {sys_df.shape[0]} rows × {sys_df.shape[1]} cols")

    # Quick summary (tp030, λ=5.0)
    highlight = sys_df[(sys_df["alpha_label"] == "tp030") & (sys_df["lambda_label"] == "5.0")]
    if len(highlight) > 0:
        h = highlight.iloc[0]
        print(f"\n  tp030 λ=5.0 km snapshot:")
        print(f"    inter-community energy sharing Volume:     {h['transfer_volume_twh']:.2f} TWh")
        print(f"    inter-community energy sharing Penetration: {h['transfer_penetration']*100:.2f}%")
        print(f"    σ-convergence:   {h['sigma_convergence']:.4f}  (<1 = equalizing)")
        print(f"    Gini reduction:  {h['gini_reduction']:.4f}  ({h['gini_reduction_ratio']*100:.1f}% of pre-inter-community energy sharing Gini)")
        print(f"    Receiver coverage: {h['receiver_coverage']*100:.2f}%")

    # ── Benefit incidence ──
    print("\n[2/3] Computing SSR-decile benefit incidence...")
    bi_df = compute_benefit_incidence()
    out = RESULT_DIR / "metrics_benefit_incidence.parquet"
    bi_df.to_parquet(out, index=False)
    print(f"  -> {out.name}: {bi_df.shape[0]} rows × {bi_df.shape[1]} cols")

    # Progressivity check (tp030, λ=5.0)
    bi_tp030 = bi_df[(bi_df["alpha_label"] == "tp030") & (bi_df["lambda_label"] == "5.0")]
    if len(bi_tp030) > 0:
        bottom20 = bi_tp030[bi_tp030["ssr_decile"].isin([1, 2])]["share_of_total_transfer_flow"].sum()
        top20 = bi_tp030[bi_tp030["ssr_decile"].isin([9, 10])]["share_of_total_transfer_flow"].sum()
        print(f"    Bottom 20% SSR (decile 1-2) gets: {bottom20*100:.1f}% of inter-community energy sharing flow")
        print(f"    Top 20% SSR (decile 9-10) gets:   {top20*100:.1f}% of inter-community energy sharing flow")
        print(f"    Progressivity ratio: {bottom20/top20:.2f}×  (>1 = pro-poor)")

    # ── Category flows ──
    print("\n[3/3] Computing category flow cross-tab...")
    cat_df = compute_category_flows()
    out = RESULT_DIR / "metrics_category_flows.parquet"
    cat_df.to_parquet(out, index=False)
    print(f"  -> {out.name}: {cat_df.shape[0]} rows × {cat_df.shape[1]} cols")

    # Rural→Metro asymmetry (tp030, λ=5.0)
    cat_tp030 = cat_df[(cat_df["alpha_label"] == "tp030") & (cat_df["lambda_label"] == "5.0")]
    if len(cat_tp030) > 0:
        print(f"\n  tp030 λ=5.0 km category flow:")
        for _, row in cat_tp030.iterrows():
            print(f"    {row['category_3']:20s}: {row['surplus_share']*100:5.1f}% surplus, "
                  f"{row['deficit_share']*100:5.1f}% deficit, "
                  f"{row['imported_twh']:.3f} TWh imported, "
                  f"{row['n_donors']:,}/{row['n_receivers']:,} donor/recv")

    # ── Cross-tier comparison ──
    print(f"\n{'='*65}")
    print("Cross-Tier Comparison (λ=5.0 km)")
    print(f"{'='*65}")
    for tier_scenario, tier_label in [
        ("tp010", "tp010 (10% PV)"),
        ("tp030", "tp030 (30% PV)"),
        ("tp050", "tp050 (50% PV)"),
        ("tp100", "tp100 (100% PV)"),
    ]:
        row = sys_df[(sys_df["alpha_label"] == tier_scenario) & (sys_df["lambda_label"] == "5.0")]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"\n  {tier_label}:")
            print(f"    SSR mean: {r['ssr_mean']:.4f}, CSSR mean: {r['cssr_mean']:.4f}")
            print(f"    CPI mean: {r['cpi_mean']:.4f} pp, inter-community energy sharing Volume: {r['transfer_volume_twh']:.3f} TWh")
            print(f"    inter-community energy sharing Penetration: {r['transfer_penetration']*100:.3f}%, SUR: {r['sur']*100:.2f}%")
            print(f"    σ-convergence: {r['sigma_convergence']:.4f}, Gini reduction: {r['gini_reduction_ratio']*100:.1f}%")
            print(f"    Receiver coverage: {r['receiver_coverage']*100:.2f}%, Donor fraction: {r['donor_fraction']*100:.1f}%")

    print(f"\nDone. All metrics saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
