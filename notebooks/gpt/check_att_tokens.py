# check_att_tokens_v2.py
from __future__ import annotations
import polars as pl
import polars.selectors as cs
from pathlib import Path

CORE = {"person_id", "date_col"}

BASE_CANDIDATES = [
    Path("data/destiny"),
    Path("/mnt/dwork/users/xc2/data/destiny"),
]

REFERENCE_NAME = "ind.parquet"
ATT_FILES = [
    "att_birthday.parquet",
    "att_calendar.parquet",
    "att_month.parquet",
    "att_season.parquet",
    "att_year.parquet",
    "att_calendar.parquet"
]

FRACTIONAL_SAMPLE_ROWS = 200_000


def find_base() -> Path:
    for b in BASE_CANDIDATES:
        if b.exists():
            return b
    raise FileNotFoundError(f"None of the bases exist: {BASE_CANDIDATES}")

def scan_any(path: Path) -> pl.LazyFrame:
    if path.is_file() and path.suffix == ".parquet":
        return pl.scan_parquet(str(path))
    if path.is_dir():
        return pl.scan_parquet(str(path / "*.parquet"))
    maybe = path.with_suffix(".parquet")
    if maybe.exists():
        return pl.scan_parquet(str(maybe))
    return pl.scan_parquet(str(path / "*.parquet"))

def dtype_str(dt: pl.DataType) -> str:
    try: return str(dt)
    except Exception: return repr(dt)

def summarize_schema(lf: pl.LazyFrame) -> dict[str, pl.DataType]:
    # Avoid PerformanceWarning by not touching lf.schema directly
    return dict(lf.collect_schema())

def core_presence(schema: dict[str, pl.DataType]) -> tuple[set[str], set[str]]:
    present = CORE & set(schema.keys())
    missing = CORE - set(schema.keys())
    return present, missing

NUM_TYPES = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64}

def get_numeric_cols(schema: dict[str, pl.DataType], exclude: set[str]) -> list[str]:
    return [c for c, t in schema.items() if c not in exclude and t in NUM_TYPES]

def count_nulls_nans(lf: pl.LazyFrame, schema: dict[str, pl.DataType]) -> tuple[pl.DataFrame, pl.DataFrame]:
    cols = list(schema.keys())
    float_cols = [c for c, t in schema.items() if t in (pl.Float32, pl.Float64)]

    nulls = lf.select([pl.col(c).is_null().sum().alias(c) for c in cols]).collect()
    nans = lf.select([pl.col(c).is_nan().sum().alias(c) for c in float_cols]).collect() if float_cols else pl.DataFrame({})
    return nulls, nans

def count_fractionals(lf: pl.LazyFrame, float_cols: list[str], sample_rows: int) -> pl.DataFrame:
    if not float_cols:
        return pl.DataFrame({})
    df = lf.select(float_cols).limit(sample_rows).collect()
    exprs = []
    for c in float_cols:
        exprs.append(
            pl.when(pl.col(c).is_null() | pl.col(c).is_nan())
              .then(False)
              .otherwise((pl.col(c) % 1) != 0)
              .sum()
              .alias(c)
        )
    return df.select(exprs)

def compare_to_reference(att_schema: dict[str, pl.DataType], ref_schema: dict[str, pl.DataType]) -> dict:
    att_cols, ref_cols = set(att_schema), set(ref_schema)
    inter = att_cols & ref_cols

    dtype_mismatches = {
        c: (dtype_str(att_schema[c]), dtype_str(ref_schema[c]))
        for c in sorted(inter) if dtype_str(att_schema[c]) != dtype_str(ref_schema[c])
    }
    only_in_att = sorted(att_cols - ref_cols)
    only_in_ref = sorted(ref_cols - att_cols)
    att_numeric = get_numeric_cols(att_schema, exclude=CORE)
    ref_numeric = get_numeric_cols(ref_schema, exclude=CORE)
    return {
        "dtype_mismatches": dtype_mismatches,
        "only_in_att": only_in_att,
        "only_in_ref": only_in_ref,
        "att_numeric_event_cols": att_numeric,
        "ref_numeric_event_cols": ref_numeric,
    }

def pretty_int(x) -> str:
    try: return f"{int(x):,}"
    except Exception: return str(x)

def main():
    pl.Config.set_tbl_cols(200)
    pl.Config.set_tbl_rows(30)

    base = find_base()
    print(f"Using base: {base}\n")

    ref_path = base / REFERENCE_NAME
    lf_ref = scan_any(ref_path)
    ref_schema = summarize_schema(lf_ref)

    print("Reference file:", ref_path)
    print("Ref columns:", len(ref_schema))
    print("Ref numeric (excluding CORE):", len(get_numeric_cols(ref_schema, exclude=CORE)))
    print()

    for name in ATT_FILES:
        path = base / name
        print("=" * 80)
        print(f"ATT file: {path}")
        if not path.exists() and not path.with_suffix("").exists():
            print("  ! File not found (skipping)")
            continue

        lf = scan_any(path)
        schema = summarize_schema(lf)
        nrows = lf.select(pl.len()).collect()[0, 0]

        present, missing = core_presence(schema)

        nulls, nans = count_nulls_nans(lf, schema)
        # ✅ function form for horizontal sum (works in Polars ≥1.0)
        total_nulls = int(nulls.select(pl.sum_horizontal(pl.all())).item()) if nulls.width else 0
        total_nans  = int(nans.select(pl.sum_horizontal(pl.all())).item())  if nans.width  else 0

        float_cols = [c for c, t in schema.items() if t in (pl.Float32, pl.Float64)]
        fracs = count_fractionals(lf, float_cols, FRACTIONAL_SAMPLE_ROWS)
        total_fracs = int(fracs.select(pl.sum_horizontal(pl.all())).item()) if fracs.width else 0

        suspicious = {
            c: dtype_str(t) for c, t in schema.items()
            if t not in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                         pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                         pl.Float32, pl.Float64,
                         pl.Utf8, pl.Boolean, pl.Date, pl.Datetime, pl.Categorical)
        }

        cmp = compare_to_reference(schema, ref_schema)
        att_numeric = cmp["att_numeric_event_cols"]
        ref_numeric = cmp["ref_numeric_event_cols"]

        print(f"Rows: {pretty_int(nrows)} | Columns: {len(schema)}")
        print(f"CORE present: {sorted(present)} | CORE missing: {sorted(missing)}")
        print(f"Total NULLs: {pretty_int(total_nulls)} | Total NaNs (floats only): {pretty_int(total_nans)}")
        if float_cols:
            print(f"Float cols: {len(float_cols)} | Fractional values (sample {FRACTIONAL_SAMPLE_ROWS:,} rows): {pretty_int(total_fracs)}")
        else:
            print("Float cols: 0")
        if suspicious:
            print("Suspicious dtypes (could break event packing):")
            for c, t in suspicious.items():
                print(f"  - {c}: {t}")

        if nulls.width:
            top_nulls = (
                nulls.transpose(include_header=True, header_name="column", column_names=["count"])
                     .sort("count", descending=True)
                     .filter(pl.col("count") > 0)
                     .head(8)
            )
            if top_nulls.height:
                print("Top NULL columns:")
                for row in top_nulls.iter_rows(named=True):
                    print(f"  - {row['column']}: {pretty_int(row['count'])}")

        if nans.width:
            top_nans = (
                nans.transpose(include_header=True, header_name="column", column_names=["count"])
                    .sort("count", descending=True)
                    .filter(pl.col("count") > 0)
                    .head(8)
            )
            if top_nans.height:
                print("Top NaN columns (floats):")
                for row in top_nans.iter_rows(named=True):
                    print(f"  - {row['column']}: {pretty_int(row['count'])}")

        if cmp["dtype_mismatches"]:
            print("Dtype mismatches vs reference (shared columns):")
            for c, (t_att, t_ref) in cmp["dtype_mismatches"].items():
                print(f"  - {c}: ATT={t_att} | REF={t_ref}")

        if cmp["only_in_att"]:
            print("Columns only in ATT:")
            msg = ", ".join(cmp["only_in_att"][:20])
            print("  ", msg, ("..." if len(cmp["only_in_att"]) > 20 else ""))

        if cmp["only_in_ref"]:
            print("Columns only in reference:")
            msg = ", ".join(cmp["only_in_ref"][:20])
            print("  ", msg, ("..." if len(cmp["only_in_ref"]) > 20 else ""))

        print(f"Numeric event columns (excluding CORE): ATT={len(att_numeric)} | REF={len(ref_numeric)}")
        if len(att_numeric) != len(ref_numeric):
            print("  ! Different numeric width; if you pack fixed-length events, pin your event_cols explicitly.")
        print()

if __name__ == "__main__":
    main()
