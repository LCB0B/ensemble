""" File for creating various features such as age, abspos, segment, etc."""

from typing import List
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import polars as pl
import polars.selectors as cs

from src.tokenize import tokenize
from src.utils import calculate_abspos
from src.log_data import log_time_token_insertion_stats, log_sequence_length_comparison


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


def insert_time_tokens_conditionally(df: pl.DataFrame) -> pl.DataFrame:
    """
    Inserts time tokens using precomputed position vectors.
    Extremely fast implementation that precomputes all temporal boundaries.
    """
    if len(df) == 0:
        return df.select(["person_id", "event"])

    # Precompute temporal boundary positions for maximum speed
    return (
        df
        .with_row_index("row_idx")
        .with_columns([
            # Compute boundaries in one pass using row comparison
            (
                (pl.col("year_token") != pl.col("year_token").shift(1).over("person_id")) |
                (pl.int_range(pl.len()).over("person_id") == 0)
            ).alias("needs_year"),
            (
                (pl.col("age_token") != pl.col("age_token").shift(1).over("person_id")) |
                (pl.int_range(pl.len()).over("person_id") == 0)
            ).alias("needs_age"),
        ])
        .with_columns([
            # Build time tokens list in single operation using case logic
            pl.when(pl.col("needs_year") & pl.col("needs_age"))
            .then(pl.concat_list([pl.col("year_token"), pl.col("age_token")]))
            .when(pl.col("needs_year"))
            .then(pl.col("year_token").cast(pl.List(pl.Int64)))
            .when(pl.col("needs_age"))
            .then(pl.col("age_token").cast(pl.List(pl.Int64)))
            .otherwise(pl.lit([]).cast(pl.List(pl.Int64)))
            .alias("time_prefix")
        ])
        .with_columns([
            # Single list concatenation
            pl.col("time_prefix").list.concat(pl.col("event")).alias("event")
        ])
        .select(["person_id", "event"])
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

    # Schema depends on time encoding mode
    if time_encoding == "time2vec":
        schema = pa.schema([
            pa.field("person_id", pa.int64()),
            pa.field("event", pa.large_list(pa.int64())),
            pa.field("age", pa.float64()),
            pa.field("abspos", pa.float64()),
        ])
    else:  # time_tokens
        schema = pa.schema([
            pa.field("person_id", pa.int64()),
            pa.field("event", pa.large_list(pa.int64())),
        ])
    writer = pq.ParquetWriter(file_path, schema=schema)

    # Initialize logging variables
    total_events_processed = 0
    total_tokens_added = 0
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

                # Process based on time encoding mode
                if time_encoding == "time2vec":
                    # Original Time2Vec mode: create age and abspos features
                    chunk_df = create_ages(chunk_df, birthdates)
                    chunk_df = create_abspos(chunk_df)
                    chunk_df = chunk_df.drop("date_col")
                else:
                    # Time tokens mode: insert time tokens conditionally when temporal boundaries change
                    chunk_df = create_year_tokens(chunk_df, vocab)
                    chunk_df = create_age_tokens(chunk_df, birthdates, vocab)

                    # Store original event lengths before time token insertion
                    if log_dir:
                        pre_insertion_lengths = chunk_df.select(pl.col("event").list.len()).to_series().to_list()

                    chunk_df = insert_time_tokens_conditionally(chunk_df)

                    # Track time token insertion stats - calculate actual tokens added
                    if log_dir:
                        post_insertion_lengths = chunk_df.select(pl.col("event").list.len()).to_series().to_list()
                        tokens_added_this_batch = sum(post - pre for pre, post in zip(pre_insertion_lengths, post_insertion_lengths))
                        batch_events = len(chunk_df)
                        total_events_processed += batch_events
                        total_tokens_added += tokens_added_this_batch

                # Add separator tokens if requested
                if sep_token:
                    chunk_df = chunk_df.with_columns(add_sep_tokens(vocab["[SEP]"]))

                # Track new sequence lengths for logging
                if log_dir:
                    batch_new_lengths = chunk_df.select(pl.col("event").list.len()).to_series().to_list()
                    new_lengths.extend(batch_new_lengths)

                writer.write_table(chunk_df.to_arrow())
    writer.close()

    # Log statistics if logging is enabled
    if log_dir:
        if time_encoding == "time_tokens":
            # Calculate detailed time token statistics
            average_tokens_per_sequence = total_tokens_added / total_events_processed if total_events_processed > 0 else 0

            # Calculate sequence length increases
            if original_lengths and new_lengths:
                length_increases = [new - orig for orig, new in zip(original_lengths, new_lengths)]
                avg_length_increase = sum(length_increases) / len(length_increases) if length_increases else 0

                # Enhanced logging with per-sequence metrics
                enhanced_stats = {
                    "events_processed": total_events_processed,
                    "tokens_added": total_tokens_added,
                    "average_tokens_per_sequence": average_tokens_per_sequence,
                    "average_length_increase_per_sequence": avg_length_increase,
                    "total_sequences": len(original_lengths),
                    "length_increase_details": {
                        "min_increase": min(length_increases) if length_increases else 0,
                        "max_increase": max(length_increases) if length_increases else 0,
                        "median_increase": sorted(length_increases)[len(length_increases)//2] if length_increases else 0
                    }
                }

                from src.log_data import log_processing_phase
                log_processing_phase(
                    phase_name="time_token_insertion_detailed",
                    stats=enhanced_stats,
                    log_dir=log_dir
                )

            log_time_token_insertion_stats(
                events_processed=total_events_processed,
                tokens_added=total_tokens_added,
                log_dir=log_dir,
                batch_id="tokenized_events"
            )

        # Log sequence length comparison
        if original_lengths and new_lengths:
            log_sequence_length_comparison(
                original_lengths=original_lengths,
                new_lengths=new_lengths,
                log_dir=log_dir,
                phase="tokenized_events"
            )

    return ds.dataset(file_path)
