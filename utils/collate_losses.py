from pathlib import Path
import argparse
import json
import polars as pl


def load_losses(runs_dir: str | Path) -> pl.DataFrame:
    runs_dir = Path(runs_dir)

    rows = []

    for path in runs_dir.rglob("losses.json"):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        test_losses = data.get("test_losses", {})

        row = {
            "directory": path.parent.name,
            "path": str(path.parent),
            **test_losses,
        }

        rows.append(row)

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)

    if "einstein_loss" in df.columns:
        df = df.sort("einstein_loss")

    return df


def summarise_losses(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame()

    numeric_cols = [
        name
        for name, dtype in df.schema.items()
        if dtype.is_numeric()
    ]

    if not numeric_cols:
        return pl.DataFrame()

    summaries = []

    for col in numeric_cols:
        x = pl.col(col).cast(pl.Float64)

        summary = df.select(
            pl.lit(col).alias("loss"),
            x.min().alias("min"),
            x.max().alias("max"),
            x.std().alias("std"),
            x.mean().alias("average"),
        )

        summaries.append(summary)

    return pl.concat(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect losses.json files from a runs directory and summarise loss statistics."
    )

    parser.add_argument(
        "runs_dir",
        type=Path,
        help="Directory containing run subdirectories with losses.json files.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("losses_summary.csv"),
        help="Path to save the summary CSV.",
    )

    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Optional path to save the raw collected losses CSV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_losses(args.runs_dir)

    if df.is_empty():
        print("No losses.json files found.")
        return

    print("Raw losses:")
    print(df)

    summary_df = summarise_losses(df)

    print("\nSummary:")
    print(summary_df)

    summary_df.write_csv(args.output)
    print(f"Saved summary to {args.output}")

    if args.raw_output is not None:
        df.write_csv(args.raw_output)
        print(f"Saved raw losses to {args.raw_output}")


if __name__ == "__main__":
    main()
