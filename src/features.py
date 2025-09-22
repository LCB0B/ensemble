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
from src.log_data import log_sequence_length_comparison


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


def insert_time_tokens_per_batch(df: pl.DataFrame, vocab: dict) -> pl.DataFrame:
    """
    DISABLE time token insertion during tokenization phase.

    The issue: Time tokens were being inserted at the event level, causing
    duplicate tokens for every data source within the same time period.

    Solution: Skip time token insertion here. Time tokens should be inserted
    later during sequence assembly when events are properly grouped by time.
    """
    # For now, skip time token insertion entirely but keep expected columns
    return df.select(["person_id", "event", "age", "abspos", "date_col"])


def insert_time_tokens_globally(df: pl.DataFrame, vocab: dict) -> pl.DataFrame:
    """
    Insert time tokens at global temporal boundaries across all events.

    This function processes the complete chronologically-ordered dataset and inserts
    YEAR_XXXX and AGE_YY tokens only at true temporal boundaries (new years and birthdays).

    Args:
        df: DataFrame with columns [person_id, event, age, abspos, date_col]
        vocab: Vocabulary dictionary containing time token mappings

    Returns:
        DataFrame with time tokens inserted and temporal columns dropped: [person_id, event]
    """
    if len(df) == 0:
        return df.select(["person_id", "event"])

    # Sort by person and chronological order to ensure proper temporal sequencing
    df = df.sort(["person_id", "abspos"])

    # Calculate discrete temporal values for boundary detection
    df = df.with_columns([
        # Calculate discrete year from abspos (hours since 2020-01-01)
        (2020 + (pl.col("abspos") / (24 * 365.25)).floor()).alias("discrete_year"),
        # Calculate discrete age for boundary detection
        pl.col("age").floor().alias("discrete_age"),
    ])

    # Detect temporal boundaries within each person's timeline
    df = df.with_columns([
        # Previous values for comparison (null for first event per person)
        pl.col("discrete_year").shift(1).over("person_id").alias("prev_year"),
        pl.col("discrete_age").shift(1).over("person_id").alias("prev_age"),
    ])

    # Identify boundary conditions
    df = df.with_columns([
        # Year boundary: first event for person OR year changed
        (pl.col("prev_year").is_null() | (pl.col("discrete_year") != pl.col("prev_year"))).alias("year_boundary"),
        # Age boundary: first event for person OR age changed
        (pl.col("prev_age").is_null() | (pl.col("discrete_age") != pl.col("prev_age"))).alias("age_boundary"),
    ])

    # Create time tokens
    df = df.with_columns([
        # Year token
        pl.col("discrete_year").map_elements(
            lambda year: vocab.get(f"YEAR_{int(year)}", vocab.get("[UNK]", 0)),
            return_dtype=pl.Int64
        ).alias("year_token"),
        # Age token (clip to 0-100 range)
        pl.col("discrete_age").clip(0, 100).map_elements(
            lambda age: vocab.get(f"AGE_{int(age)}", vocab.get("[UNK]", 0)),
            return_dtype=pl.Int64
        ).alias("age_token"),
    ])

    # Build time token lists for insertion
    df = df.with_columns([
        pl.when(pl.col("year_boundary") & pl.col("age_boundary"))
        .then(pl.concat_list([pl.col("year_token"), pl.col("age_token")]))
        .when(pl.col("year_boundary"))
        .then(pl.col("year_token").cast(pl.List(pl.Int64)))
        .when(pl.col("age_boundary"))
        .then(pl.col("age_token").cast(pl.List(pl.Int64)))
        .otherwise(pl.lit([]).cast(pl.List(pl.Int64)))
        .alias("time_tokens")
    ])

    # Insert time tokens at the beginning of events that have boundaries
    df = df.with_columns([
        pl.when(pl.col("year_boundary") | pl.col("age_boundary"))
        .then(pl.col("time_tokens").list.concat(pl.col("event")))
        .otherwise(pl.col("event"))
        .alias("event")
    ])

    # Return clean schema: only person_id and event (drop all temporal metadata)
    return df.select(["person_id", "event"])


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

    # Schema is the same for both modes after processing
    if time_encoding == "time2vec":
        schema = pa.schema([
            pa.field("person_id", pa.int64()),
            pa.field("event", pa.large_list(pa.int64())),
            pa.field("age", pa.float64()),
            pa.field("abspos", pa.float64()),
        ])
    else:  # time_tokens - include temporal metadata during processing (cleaned up post-loop)
        schema = pa.schema([
            pa.field("person_id", pa.int64()),
            pa.field("date_col", pa.timestamp("us")),
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

                # Process based on time encoding mode
                if time_encoding == "time2vec":
                    # Original Time2Vec mode: create age and abspos features
                    chunk_df = create_ages(chunk_df, birthdates)
                    chunk_df = create_abspos(chunk_df)
                    chunk_df = chunk_df.drop("date_col")
                else:
                    # Time tokens mode: process batches like time2vec but preserve temporal metadata
                    # Sort by person_id and date_col to ensure chronological order
                    chunk_df = chunk_df.sort(["person_id", "date_col"])
                    # Use same temporal preprocessing as time2vec
                    chunk_df = create_ages(chunk_df, birthdates)
                    chunk_df = create_abspos(chunk_df)

                    # Normalize timestamp precision to microseconds for schema consistency
                    # Handle both date and datetime/timestamp types
                    date_col_dtype = chunk_df["date_col"].dtype
                    if date_col_dtype == pl.Date:
                        chunk_df = chunk_df.with_columns(
                            pl.col("date_col").cast(pl.Datetime("us"))
                        )
                    else:
                        chunk_df = chunk_df.with_columns(
                            pl.col("date_col").dt.cast_time_unit("us")
                        )

                    # Keep temporal metadata for post-loop global time token insertion
                    # Don't drop date_col, age, abspos yet - will be dropped after global insertion

                # Add separator tokens if requested
                if sep_token:
                    chunk_df = chunk_df.with_columns(add_sep_tokens(vocab["[SEP]"]))

                # Track new sequence lengths for logging
                if log_dir:
                    batch_new_lengths = chunk_df.select(pl.col("event").list.len()).to_series().to_list()
                    new_lengths.extend(batch_new_lengths)

                writer.write_table(chunk_df.to_arrow())
    writer.close()

    # Post-loop processing: Global time token insertion for time_tokens mode
    if time_encoding == "time_tokens":
        try:
            print("Applying global time token insertion...")

            # Read the complete dataset back for global processing
            complete_df = pl.read_parquet(file_path)
            print(f"Read {len(complete_df)} rows from parquet for global processing")

            if len(complete_df) > 0:
                # Apply global time token insertion
                print("Inserting time tokens globally...")
                complete_df = insert_time_tokens_globally(complete_df, vocab)

                # Rewrite the parquet file with time tokens and clean schema
                print("Writing cleaned parquet file...")
                complete_df.write_parquet(file_path)

                print(f"Global time token insertion complete. Clean schema: {complete_df.columns}")
            else:
                print("Warning: No data to process for global time token insertion!")
        except Exception as e:
            print(f"Error during global time token insertion: {e}")
            import traceback
            traceback.print_exc()
            raise

    # Log statistics if logging is enabled
    if log_dir:
        # Note: For time_tokens mode, detailed stats are now handled by global insertion
        # Original logging variables are not populated since we use post-loop processing

        # Log sequence length comparison
        if original_lengths and new_lengths:
            log_sequence_length_comparison(
                original_lengths=original_lengths,
                new_lengths=new_lengths,
                log_dir=log_dir,
                phase="tokenized_events"
            )

    return ds.dataset(file_path)
