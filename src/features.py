""" File for creating various features such as age, abspos, segment, etc."""

from typing import List, Iterator, Tuple, Dict, Any
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import polars as pl
import polars.selectors as cs

from src.tokenize import tokenize
from src.utils import calculate_abspos
from src.log_data import log_sequence_length_comparison

from itertools import islice

from datetime import datetime

HOURS_PER_YEAR = 365.25 * 24.0

def iter_time_token_batches(
    birthdate_map: dict[int, datetime],
    vocab: dict[str,int],
    end_year: int = 2020,
    max_age: int = 110,
    batch_size: int = 10_000,
) -> Iterator[pl.DataFrame]:
    """
    Yields DataFrames of synthetic time‐token rows in batches of `batch_size` persons.
    """
    origin = datetime(2020,1,1)
    unk_id = vocab.get("[UNK]", 0)
    items_iter = iter(birthdate_map.items())

    while True:
        batch = list(islice(items_iter, batch_size))
        if not batch:
            break

        rows = []
        for pid, bdate in batch:
            # precompute birth_abs once
            delta = bdate - origin
            birth_abs = delta.days * 24.0 + delta.seconds / 3600.0
            byear     = bdate.year
            # YEAR tokens: at Jan 1 of each year [byear+1 .. end_year]
            for y in range(byear+1, end_year+1):
                token  = vocab.get(f"YEAR_{y}", unk_id)
                # abspos = (y - 2020) * HOURS_PER_YEAR
                ap     = (y - 2020) * HOURS_PER_YEAR
                rows.append((pid, [token], float(y-byear), ap))
            # AGE tokens: at each birthday from age 0..min(max_age, end_year-byear)
            age_limit = min(max_age, end_year - byear)
            for a in range(age_limit+1):
                token = vocab.get(f"AGE_{a}", unk_id)
                ap    = birth_abs + a * HOURS_PER_YEAR
                rows.append((pid, [token], float(a), ap))

        # build & yield in one shot; downstream code will sort by abspos if needed
        yield pl.DataFrame(
            rows,
            orient="row",
            schema=["person_id","event","age","abspos"],
        )


def create_cls_source(birthdates: pl.DataFrame) -> pl.DataFrame:
    """Adds cls tokens to birthdates"""
    return birthdates.with_columns(cls_col=pl.lit("[CLS]"))


def add_sep_tokens(sep_token: int = 2):
    """Adds seperator tokens to the end of each (TOKENIZED) event"""
    return pl.col("event").list.concat(sep_token)


def create_abspos(
    df: pl.DataFrame, origin_point=pl.datetime(2020, 1, 1, time_unit="ns")
):
    """Creates the abspos columns by doing 'date_col' - origin_point"""
    return df.with_columns(
        abspos=calculate_abspos(pl.col("date_col"), origin_point=origin_point)
    )


def create_year_tokens(df: pl.DataFrame, vocab: dict) -> pl.DataFrame:
    """Creates year token IDs from date_col for time_tokens encoding"""
    return df.with_columns(
        year_token=pl.col("date_col").dt.year().map_elements(
            lambda year: vocab.get(f"YEAR_{year}", vocab["[UNK]"]),
            return_dtype=pl.Int64
        )
    )


def create_age_tokens(df: pl.DataFrame, birthdates: pl.DataFrame, vocab: dict) -> pl.DataFrame:
    """Creates age token IDs by joining birthdates and calculating age tokens"""
    return (
        df.join(birthdates, on="person_id", how="inner")
        .with_columns(
            age_years=((pl.col("date_col") - pl.col("birthday")).dt.total_days() / 365.25).floor()
        )
        .with_columns(
            age_token=pl.col("age_years").clip(0, 100).map_elements(
                lambda age: vocab.get(f"AGE_{int(age)}", vocab["[UNK]"]),
                return_dtype=pl.Int64
            )
        )
        .drop("birthday", "age_years")
    )



def create_ages(df: pl.DataFrame, birthdates: pl.DataFrame):
    """Creates ages by joining birthdates and subtracting from date_col"""
    return (
        df.join(birthdates, on="person_id", how="inner")
        .with_columns(
            age=((pl.col("date_col") - pl.col("birthday")).dt.total_days() / 365.25)
        )
        .drop("birthday")
    )


def _cleanup_completed_persons(current_batch_pids: set, last_year: dict, last_age: dict) -> None:
    """
    Clean up state for person_ids not in current batch to manage memory.
    Assumes person_ids are processed roughly in order.
    """
    # Keep only person_ids from recent batches
    pids_to_remove = []
    for pid in last_year.keys():
        if pid not in current_batch_pids:
            pids_to_remove.append(pid)

    # Remove a portion of old person_ids (conservative cleanup)
    removal_count = min(len(pids_to_remove) // 2, 10000)  # Remove up to 10k at a time
    for pid in pids_to_remove[:removal_count]:
        last_year.pop(pid, None)
        last_age.pop(pid, None)



def create_tokenized_events(
    sources: List[ds.Dataset],
    vocab: dict,
    birthdates: pl.DataFrame,
    dir_path: Path,
    sep_token: bool,
    fill_nulls=False,
    time_encoding: str = "time2vec",
    log_dir: Path = None,
) -> ds.Dataset:
    """
    Tokenizes and creates events with features, saving it to dir_path / 'tokenized.parquet'

    Parameters:
        sources: The list of Datasets to be processed and tokenized
        vocab: The vocabulary used for tokenization
        birthdates: DataFrame with person_id and birthday columns
        dir_path: The "name" of this data version
        sep_token: Whether to add separator tokens
        fill_nulls: Whether to fill null values with [UNK] token
        time_encoding: "time2vec" (default) for continuous time or "time_tokens" for discrete tokens
        log_dir: Optional directory for logging sequence length statistics

    Assumptions:
        Sources to have ID='person_id' and timestamp='date_col'
    """
    file_path = dir_path / "tokenized.parquet"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Schema 
    schema = pa.schema([
        pa.field("person_id", pa.int64()),
        pa.field("event", pa.large_list(pa.int64())),
        pa.field("age", pa.float64()),
        pa.field("abspos", pa.float64()),
    ])
   
    writer = pq.ParquetWriter(file_path, schema=schema)

    # Initialize logging variables for sequence length tracking
    original_lengths = []
    new_lengths = []

    for source in tqdm(sources, "Tokenizing sources"):
        for source_batch in source.to_batches():
            # Sometimes there aren't any rows due to filtering of batches
            if source_batch.num_rows > 0:  # TODO: Move this check to writing
                chunk_df = pl.from_arrow(source_batch)
                # Tokenize string columns
                chunk_df = tokenize(chunk_df, vocab)

                # Select event columns (by negation of non-event columns)
                event_columns = ~cs.by_name("person_id", "date_col")

                if fill_nulls:
                    chunk_df = chunk_df.with_columns(
                        event_columns.fill_null(vocab["[UNK]"])
                    )

                # Convert to dataframe of (person_id, date_col, event)
                chunk_df = (
                    chunk_df.with_columns(
                        pl.Series("event", chunk_df.select(event_columns).to_numpy())
                    )
                    .select(
                        "person_id", "date_col", "event"
                    )  # Only select needed columns
                    .cast(
                        {
                            "event": pl.List(pl.Int64),
                        }
                    )  # Convert numpy NaNs to Null
                )
                # Drop Nulls from event list (done here so it's element-wise)
                if not fill_nulls:  # Can safely skip if nulls are filled
                    chunk_df = chunk_df.with_columns(
                        pl.col("event").list.drop_nulls()
                    ).filter(pl.col("event").list.len() > 0)

                # Track original sequence lengths for logging
                if log_dir:
                    batch_original_lengths = chunk_df.select(pl.col("event").list.len()).to_series().to_list()
                    original_lengths.extend(batch_original_lengths)

         
                # Original Time2Vec mode: create age and abspos features
                chunk_df = create_ages(chunk_df, birthdates)
                chunk_df = create_abspos(chunk_df)
                chunk_df = chunk_df.drop("date_col")

                # Add separator tokens if requested
                if sep_token:
                    chunk_df = chunk_df.with_columns(add_sep_tokens(vocab["[SEP]"]))

                # Track new sequence lengths for logging
                if log_dir:
                    batch_new_lengths = chunk_df.select(pl.col("event").list.len()).to_series().to_list()
                    new_lengths.extend(batch_new_lengths)

                writer.write_table(chunk_df.to_arrow())

    if time_encoding == "time_tokens":
        print(f"[{datetime.now().strftime('%H:%M:%S')}] insert time tokens")
        birthdate_map = {row["person_id"]: row["birthday"] for row in birthdates.to_dicts()}

        # Calculate total batches for progress tracking
        batch_size = 1_000
        total_people = len(birthdate_map)
        total_batches = (total_people + batch_size - 1) // batch_size  # Ceiling division

        print(f"Processing {total_people:,} people in {total_batches:,} batches of {batch_size:,}")

        for sched_df in tqdm(
            iter_time_token_batches(
                birthdate_map=birthdate_map,
                vocab=vocab,
                end_year=2020,
                max_age=100,
                batch_size=batch_size,
            ),
            total=total_batches,
            desc="Time token batches"
        ):
            writer.write_table(sched_df.to_arrow())
        print(f"[{datetime.now().strftime('%H:%M:%S')}] done time tokens")

    writer.close()

    return ds.dataset(file_path)
