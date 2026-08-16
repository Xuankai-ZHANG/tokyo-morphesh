"""Reassemble chunked cssr_daily tables (split for GitHub single-file limits).

Usage:
    python scripts/reassemble_daily.py                # reassemble all scenarios
    python scripts/reassemble_daily.py tp030          # one scenario
    python scripts/reassemble_daily.py --verify # reassemble + verify

Each cssr_daily_tpNNN.parquet was row-split into N parts; this concatenates
them back into data/results/daily/cssr_daily_tpNNN.parquet.
"""
from pathlib import Path
import pandas as pd
import argparse

DAILY = Path(__file__).resolve().parents[1] / "data" / "results" / "daily"


def reassemble(scenario: str, verify: bool = False) -> Path:
    parts = sorted(DAILY.glob(f"cssr_daily_{scenario}_part*of*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no parts found for {scenario}")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    out = DAILY / f"cssr_daily_{scenario}.parquet"
    df.to_parquet(out, index=False)
    if verify:
        n_expected = 20_833
        assert len(df) == n_expected, f"expected {n_expected} rows, got {len(df)}"
        assert len(df.columns) == 733, f"expected 733 cols, got {len(df.columns)}"
        print(f"  verified: {len(df)} rows x {len(df.columns)} cols")
    print(f"  -> {out.name} ({out.stat().st_size/1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", help="scenario labels (e.g. tp030); default = all")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    scenarios = args.scenarios or ["tp010", "tp015", "tp020", "tp030", "tp040", "tp050", "tp100"]
    for s in scenarios:
        print(f"Reassembling {s}...")
        reassemble(s, verify=args.verify)
