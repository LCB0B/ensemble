# build_time_sources.py
import polars as pl
from pathlib import Path
import gc

from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TextColumn

TOKEN_PREFIX = "TIM"

# ---- token helpers ----------------------------------------------------------

def month_token(e):
    return pl.concat_str([
        pl.lit(f"{TOKEN_PREFIX}_MONTH_"),
        e.dt.month().cast(pl.Utf8).str.zfill(2)
    ])

def season_token(e):
    m = e.dt.month()
    return (
        pl.when(m.is_in([12, 1, 2])).then(pl.lit(f"{TOKEN_PREFIX}_SEASON_WINTER"))
         .when(m.is_in([3, 4, 5])).then(pl.lit(f"{TOKEN_PREFIX}_SEASON_SPRING"))
         .when(m.is_in([6, 7, 8])).then(pl.lit(f"{TOKEN_PREFIX}_SEASON_SUMMER"))
         .otherwise(pl.lit(f"{TOKEN_PREFIX}_SEASON_AUTUMN"))
    )

def explode_ranges(df, start_col, end_col, every, token_expr):
    # avoids exploding inverted ranges
    return (
        df.with_columns(
            date_col = pl.when(pl.col(end_col) >= pl.col(start_col))
                         .then(pl.date_ranges(pl.col(start_col), pl.col(end_col), every))
                         .otherwise(pl.lit([]))
        )
        .explode("date_col")
        .with_columns(time_token = token_expr(pl.col("date_col")))
        .select(
            pl.col("person_id").cast(pl.Int64),
            pl.col("date_col").cast(pl.Date),
            pl.col("time_token").cast(pl.Utf8),
        )
    )


def build_year_source(lifelines: pl.DataFrame) -> pl.DataFrame:
    return (
        lifelines
        .with_columns(
            start = pl.datetime(pl.col("birthday").dt.year() + 1, 1, 1),
            end   = pl.col("event_final_date").dt.truncate("1y")
        )
        .pipe(explode_ranges, "start", "end", "1y",
              lambda e: pl.format(f"{TOKEN_PREFIX}_YEAR_{{}}", e.dt.year()))
        .sort(["person_id", "date_col"])
    )

def build_birthday_source(lifelines: pl.DataFrame, max_age: int = 110) -> pl.DataFrame:
    df = lifelines.with_columns(
        start = pl.col("birthday").dt.offset_by("1y"),
        end   = pl.col("event_final_date"),
        dates = pl.date_ranges(pl.col("birthday").dt.offset_by("1y"),
                               pl.col("event_final_date"), "1y")
    ).with_columns(
        ages = pl.int_ranges(1, pl.col("dates").list.len() + 1, 1)
    )
    return (
        df.explode(["dates", "ages"])
          .filter(pl.col("ages") <= max_age)
          .select(
              pl.col("person_id").cast(pl.Int64).alias("person_id"),
              pl.col("dates").cast(pl.Date).alias("date_col"),
              pl.format(f"{TOKEN_PREFIX}_AGE_{{}}", pl.col("ages")).alias("time_token"),
          )
          .with_columns(pl.col("time_token").cast(pl.Utf8))
          .sort(["person_id", "date_col"])
    )


def _iter_person_chunks(df: pl.DataFrame, persons_per_chunk: int):
    n = df.height
    i = 0
    while i < n:
        n_take = min(persons_per_chunk, n - i)
        yield i // persons_per_chunk, df.slice(i, n_take)
        i += n_take


def _write_stream_parquet(df_iter, outfile: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import gc

    outfile.parent.mkdir(parents=True, exist_ok=True)
    if outfile.exists():
        outfile.unlink()

    # Use LargeString so both String and LargeString inputs can be cast safely.
    schema_arrow = pa.schema([
        pa.field("person_id", pa.int64()),
        pa.field("date_col",  pa.date32()),
        pa.field("time_token", pa.large_string()),
    ])

    writer = None
    total_rows = 0

    for chunk_idx, df in df_iter:
        tbl = df.to_arrow()

        # Cast whole table to the unified schema (String -> LargeString is OK)
        if tbl.schema != schema_arrow:
            tbl = tbl.cast(schema_arrow)

        if writer is None:
            writer = pq.ParquetWriter(
                where=str(outfile),
                schema=schema_arrow,
                compression="zstd",
                version="2.6",
                use_dictionary=True,
            )

        writer.write_table(tbl)  # one row group per chunk
        total_rows += tbl.num_rows
        del df, tbl
        gc.collect()

    if writer is not None:
        writer.close()

    return total_rows

def write_month_single(lifelines: pl.DataFrame, outdir: Path, persons_per_chunk: int, prog: Progress):
    base = lifelines.select(
        pl.col("person_id"),
        start = pl.col("birthday").dt.truncate("1mo"),
        end   = pl.col("event_final_date").dt.truncate("1mo"),
    )
    def gen():
        for chunk_idx, chunk in _iter_person_chunks(base, persons_per_chunk):
            yield chunk_idx, explode_ranges(chunk, "start", "end", "1mo", month_token)

    outfile = outdir / "att_month.parquet"
    task = prog.add_task("month: stream → parquet", total=None)
    n_rows = _write_stream_parquet(gen(), outfile)
    prog.console.log(f"[month] wrote {n_rows:,} rows -> {outfile}")
    prog.remove_task(task)


def write_season_single(lifelines: pl.DataFrame, outdir: Path, persons_per_chunk: int, every: str, prog: Progress):
    base = lifelines.select(
        pl.col("person_id"),
        start = pl.col("birthday").dt.truncate(every),
        end   = pl.col("event_final_date").dt.truncate(every),
    )
    def gen():
        for _, chunk in _iter_person_chunks(base, persons_per_chunk):
            yield _, explode_ranges(chunk, "start", "end", every, season_token)

    outfile = outdir / "att_season.parquet"
    task = prog.add_task("season: stream → parquet", total=None)
    n_rows = _write_stream_parquet(gen(), outfile)
    prog.console.log(f"[season] wrote {n_rows:,} rows -> {outfile}")
    prog.remove_task(task)


def write_all(lifelines_path: Path, outdir: Path,
              persons_per_chunk: int = 5_000,
              season_every: str = "3mo"):
    outdir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(), BarColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as prog:
        t_read = prog.add_task("read lifelines", total=1)
        lifelines = pl.read_parquet(
            lifelines_path, columns=["person_id","birthday","event_final_date"]
        ).sort("event_final_date")
        prog.advance(t_read)

        # small ones
        def _build_and_write_single(name: str, fn):
            task = prog.add_task(f"{name}: build → write", total=2)
            df = fn(lifelines)
            prog.advance(task, 1)
            outfile = outdir / f"att_{name}.parquet"
            df.write_parquet(outfile, compression="zstd", compression_level=3)
            prog.advance(task, 1)
            prog.console.log(f"[{name}] rows={df.height:,} -> {outfile}")
            del df; gc.collect()

        _build_and_write_single("year", build_year_source)
        _build_and_write_single("birthday", build_birthday_source)

        # big ones (single file via streaming)
        write_month_single(lifelines, outdir, persons_per_chunk, prog=prog)
        write_season_single(lifelines, outdir, persons_per_chunk, every=season_every, prog=prog)


if __name__ == "__main__":
    from src.paths import FPATH
    write_all(
        FPATH.NETWORK_DATA / "destiny" / "cohort" / "lifelines.parquet",
        FPATH.NETWORK_DATA / "destiny",
        persons_per_chunk=500_000,   # adjust based on RAM
        season_every="3mo",        
    )
