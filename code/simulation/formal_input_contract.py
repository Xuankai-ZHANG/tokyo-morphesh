"""Single authoritative technical-potential input contract for formal downstream runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POTENTIAL_DIR = PROJECT_ROOT / "data" / "input" / "potential"
TECHNICAL_POTENTIAL_PATH = POTENTIAL_DIR / "technical_potential.parquet"
MANIFEST_PATH = POTENTIAL_DIR / "manifest.json"
LOCKED_DENSITY_KWH_M2 = 200.0


def validate_technical_potential(
    tp: pd.DataFrame, manifest: dict, final_mesh_codes: set[int]
) -> dict:
    density = float(manifest.get("target_roof_energy_density_kwh_m2_yr", np.nan))
    if not np.isclose(density, LOCKED_DENSITY_KWH_M2, rtol=0, atol=1e-12):
        raise ValueError(f"formal potential density must be locked at 200, got {density}")
    if tp["mesh_code"].duplicated().any():
        raise ValueError("formal technical-potential input contains duplicate mesh_code")
    selected = tp[tp["mesh_code"].isin(final_mesh_codes)]
    expected_rows = int(manifest["final_support_rows"])
    if len(selected) != expected_rows or len(selected) != len(final_mesh_codes):
        raise ValueError(
            f"formal support mismatch: selected={len(selected)}, "
            f"codes={len(final_mesh_codes)}, manifest={expected_rows}"
        )
    total_kwh = float(selected["potential_kwh"].sum())
    expected_kwh = float(manifest["final_support_potential_twh"]) * 1e9
    if not np.isclose(total_kwh, expected_kwh, rtol=1e-10, atol=1e-3):
        raise ValueError(
            f"formal potential total mismatch: data={total_kwh}, manifest={expected_kwh}"
        )
    return {"rows": len(selected), "potential_kwh": total_kwh}


def load_technical_potential() -> tuple[pd.DataFrame, dict]:
    tp = pd.read_parquet(TECHNICAL_POTENTIAL_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    density = float(manifest.get("target_roof_energy_density_kwh_m2_yr", np.nan))
    if not np.isclose(density, LOCKED_DENSITY_KWH_M2, rtol=0, atol=1e-12):
        raise ValueError(f"formal potential density must be locked at 200, got {density}")
    return tp, manifest
