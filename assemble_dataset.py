"""Join public Crystoria tables, or select a legacy internal-format valid view."""

import argparse
from pathlib import Path

import polars as pl


def public_table(data, geometry="none"):
    data = Path(data)
    frame = pl.scan_parquet(data / "labels/phonons.parquet")
    for relative in (
        "labels/thermodynamics.parquet",
        "labels/relaxation.parquet",
        "provenance/identity.parquet",
    ):
        other = pl.scan_parquet(data / relative)
        extra = [
            name for name in other.collect_schema().names() if name not in frame.collect_schema()
        ]
        frame = frame.join(other.select("record_id", *extra), on="record_id", validate="1:1")
    for name in (
        ("source", "relaxed")
        if geometry == "both"
        else (("source",) if geometry == "source" else ())
    ):
        structures = pl.scan_parquet(data / "structures" / name / "*.parquet").select(
            "record_id", pl.col("structure_json").alias(f"{name}_structure_json")
        )
        frame = frame.join(structures, on="record_id", validate="1:1")
    return frame


def main():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Public 01-data directory")
    source.add_argument(
        "--records", nargs="+", help="Legacy internal shards; not required for public-data use"
    )
    p.add_argument("--geometry", choices=("none", "source", "both"), default="none")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    frame = (
        public_table(args.data, args.geometry)
        if args.data
        else pl.scan_parquet(args.records).filter(pl.col("phonon_label_valid_for_stability"))
    )
    count = frame.select(pl.len()).collect().item()
    if count != 2649640:
        raise ValueError(f"Population mismatch: {count}")
    frame.sink_parquet(args.output, compression="zstd")
    print(count)


if __name__ == "__main__":
    main()
