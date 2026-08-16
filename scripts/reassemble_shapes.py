"""Reassemble chunked generation/demand shapes before running the simulation.

The 30-min shapes (gen_shapes by month, dem_shapes) were row-split into ~30MB
parts for GitHub. This reconstructs the originals that the engine expects:

    data/input/shapes/gen_shapes/month=01..12.parquet
    data/input/shapes/dem_shapes.parquet

Usage:
    python scripts/reassemble_shapes.py          # all shapes
"""
from pathlib import Path
import pandas as pd

SHAPES = Path(__file__).resolve().parents[1] / "data" / "input" / "shapes"


def reassemble(search_dir: Path, prefix: str, out: Path):
    parts = sorted(search_dir.glob(f"{prefix}_part*of*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no parts matching {prefix}_part*of* in {search_dir}")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  -> {out.relative_to(SHAPES)} ({len(df):,} rows, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    for mm in range(1, 13):
        reassemble(SHAPES / "gen_shapes", f"gen_shapes_month{mm:02d}",
                   SHAPES / "gen_shapes" / f"month={mm:02d}.parquet")
    reassemble(SHAPES, "dem_shapes", SHAPES / "dem_shapes.parquet")
    print("Done. Shapes reassembled; the simulation engine can now run.")
