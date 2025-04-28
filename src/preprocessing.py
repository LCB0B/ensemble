import polars as pl
from pathlib import Path
from src.utils import filter_parquet_by_person_ids_to_dataset
from src.paths import FPATH, check_and_copy_file, copy_file_or_dir
from src.chunking import write_dataset_to_parquet_in_batches
from typing import Tuple, List, Optional
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc
import math


def bin_column(col_name: str, dataset: ds.Dataset, bins: int) -> Path:
    """
    Bins the values of a specified column in a dataset into quantiles, writes the binned result to a parquet file,
    and returns the file path.

    Args:
        col_name (str): The name of the column to bin.
        dataset (pl.DataFrame): The input dataset containing the column.
        bins (int): The number of bins to create. Defaults to 100.

    Returns:
        Path: The path to the saved parquet file.
    """
    # Convert column to Polars DataFrame
    col = pl.from_arrow(dataset.to_table(columns=[col_name]))

    # Generate cutoff points for binning
    cutoffs: List[float] = [i / bins for i in range(1, bins)]

    print(f"{col_name}: Counting unique values...")
    # Count unique values
    n_unique = col.select(pl.col(col_name).n_unique()).item()

    # Bin the column into quantiles and assign labels
    binned_col = (
        (pl.col(col_name).rank(method="dense") / n_unique)
        .cut(cutoffs, labels=[f"{col_name}_Q{i}" for i in range(1, bins + 1)])
        .cast(pl.Utf8)  # Cast from Categorical to String
        .alias(f"binned_{col_name}")
    )

    # Define the output path
    output_path = FPATH.TEMP_FILES / f"binned_{col_name}_temp.parquet"

    # Write the binned column to a parquet file
    print(f"{col_name}: Binning...")
    col.select(binned_col).write_parquet(output_path)

    return output_path


def truncate_columns(
    df: pl.DataFrame, truncations_specs: list[Tuple[str, int]]
) -> pl.DataFrame:
    """
    Truncate values in specified columns to the given cutoff lengths.

    Args:
        df (pl.DataFrame): The input DataFrame.
        truncations_specscols (list[Tuple[str, int]]): List of tuples of colname-truncation length to process.

    Returns:
        pl.DataFrame: The modified DataFrame with truncated columns.
    """
    for col, cutoff in truncations_specs:
        df = df.with_columns(pl.col(col).cast(pl.Utf8).str.slice(0, cutoff).alias(col))
    return df


def prefix_col_name(df: pl.LazyFrame, cols: list[str]) -> pl.LazyFrame:
    """
    Iterate through specified columns in a LazyFrame and replace values.
    Null values are replaced with f'{col}_null'. Non-null values are prefixed
    with the column name.

    Args:
        df (pl.LazyFrame): The input LazyFrame.
        cols (list[str]): List of column names to process.

    Returns:
        pl.LazyFrame: The modified LazyFrame.
    """
    for col in cols:
        # Replace null values and prefix non-null values
        df = df.with_columns(pl.concat_str([pl.lit(col + "_"), pl.col(col)]).alias(col))
    return df


def prefix_col_name_keep_nulls(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """
    Iterate through specified columns in a DataFrame and prefix non-null values
    with the column name. Null values are left unchanged.

    Args:
        df (pl.DataFrame): The input DataFrame.
        cols (list[str]): List of column names to process.

    Returns:
        pl.DataFrame: The modified DataFrame with prefixed non-null values.
    """
    for col in cols:
        df = df.with_columns(
            pl.when(pl.col(col).is_not_null())
            .then(pl.lit(f"{col}_") + pl.col(col).cast(pl.Utf8))
            .otherwise(pl.col(col))  # Keep null as null
            .alias(col)
        )
    return df


def concat_datasets(
    datasets: List[ds.Dataset],
    output_file_path: Path,
    truncation_specs: List[Tuple[str, int]],
    cols_prefix: List[str],
    cols_binned: List[str],
    cols_unchanged: List[str],
    date_col: str,
    chunk_size: int = 1_000_000,
) -> None:
    """
    Reads chunks from multiple Parquet files using dataset.take in chunks, applies truncation and prefixing only on the first chunk of the
    original file, performs horizontal concatenation, and saves the result to a common Parquet file using ParquetWriter.
    Only specific columns (prefixed, binned, unchanged) are kept in the final output.

    Args:
        datasets (ds.Dataset): List of Datasets to process and concat.
        output_file_path (Path): Path to the output Parquet file.
        truncation_specs (List[Tuple[str, int]]): Columns to truncate and their respective lengths.
        cols_prefix (List[str]): Columns to prefix non-null values.
        cols_binned (List[str]): Columns to be binned and kept in the output.
        cols_unchanged (List[str]): Columns to keep unchanged in the output.
        date_col (str): Column to be renamed to 'date_col'.
        chunk_size (int): Number of rows to read per chunk. Default 100,000.
    """
    individual_total_rows = [dataset.count_rows() for dataset in datasets]
    total_rows = individual_total_rows[0]
    assert all(
        total == total_rows for total in individual_total_rows
    ), "All datasets must have the same number of rows."

    indices = list(range(0, total_rows, chunk_size))

    # Create a list of binned column names
    binned_cols_names = [f"binned_{bin_specs[0]}" for bin_specs in cols_binned]

    # Define the columns to keep, excluding unbinned columns that match a binned column
    _cols_to_keep = (
        ["person_id", "date_col"] + cols_prefix + binned_cols_names + cols_unchanged
    )
    cols_which_were_binned = [bin_specs[0] for bin_specs in cols_binned]
    cols_to_keep = [col for col in _cols_to_keep if col not in cols_which_were_binned]

    # while True:
    #     chunks = []
    writer = None
    for start_idx in tqdm(indices, "Concatenating datasets"):
        chunks = []
        for i, dataset in enumerate(datasets):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = dataset.take(indices=list(range(start_idx, end_idx)))
            chunk = pl.from_arrow(chunk)

            if i == 0:
                if truncation_specs:
                    chunk = truncate_columns(chunk, truncation_specs)
                if cols_prefix:
                    chunk = prefix_col_name_keep_nulls(chunk, cols_prefix)
                chunk = chunk.rename({date_col: "date_col"})

            chunks.append(chunk)

        if len(chunks) != len(datasets):
            print(
                f"Didn't get the expected amount of chunks, got {len(chunks)}, expected {len(datasets)}"
            )
            break

        # Concatenate the chunks horizontally
        concatenated_chunk = pl.concat(chunks, how="horizontal")

        concatenated_chunk = concatenated_chunk.select(cols_to_keep)

        table = concatenated_chunk.to_arrow()

        if writer is None:
            writer = pq.ParquetWriter(output_file_path, schema=table.schema)

        # Write the concatenated chunk to the output file
        writer.write_table(table)

    if writer:
        writer.close()

    FPATH.copy_to_opposite_drive(output_file_path)


def write_person_id_birthday_filtered_dataset(
    file_path: str,
    person_df: pl.DataFrame,
    output_path: Path,
    date_col: str,
    batch_size: int = 10_000_000,
) -> ds.Dataset:
    """
    Writes a PyArrow Dataset to Parquet in batches after merging with a Polars dataframe
    and applying date-based filtering.

    Args:
       file_path (str): The path to the Parquet file.
        person_df (pl.DataFrame): Polars dataframe containing `person_id` and `birthday` columns.
        output_path (Path): Path to write the filtered Parquet file.
        date_col (str): The name of the date column in the Parquet dataset to filter by.
        batch_size (int): Number of rows per batch for processing.

    Returns:
        ds.Dataset: A new PyArrow Dataset pointing to the filtered Parquet data.
    """
    # Step 1: Convert the Polars dataframe to a PyArrow table
    person_table = person_df.to_arrow()
    input_dataset = ds.dataset(file_path, format="parquet")

    # Step 2: Prepare the Parquet writer
    with pq.ParquetWriter(output_path, input_dataset.schema) as writer:

        # Step 3: Process the dataset in batches (done using take to avoid small batches)
        n_rows = input_dataset.count_rows()
        n_batches = math.ceil(n_rows / batch_size)
        for i in tqdm(range(0, n_batches), "Filtering batches"):

            lowest_idx = i * batch_size
            highest_idx = (i + 1) * batch_size
            if highest_idx > n_rows:
                highest_idx = n_rows

            batch_table = input_dataset.take(indices=range(lowest_idx, highest_idx))

            # Step 4: Perform an in-memory join of the current batch with the birthdays (person_table)
            merged_table = batch_table.join(
                person_table, keys="person_id", join_type="inner"
            )

            # Step 5: Apply the date filter (where date_col >= birthday)
            filtered_table = merged_table.filter(
                pc.greater_equal(merged_table[date_col], merged_table["_birthday"])
            )

            # Step 6: Ensure the table schema matches the original schema by selecting the original columns
            filtered_table = filtered_table.select(input_dataset.schema.names)

            # Step 7: Write the filtered table in batches
            if len(filtered_table) > 0:
                writer.write_table(filtered_table)

    # Return the new dataset pointing to the output path
    return ds.dataset(output_path, format="parquet")


def process_dataset(
    dump_path: Path,
    person_subsample: List[int],
    truncation_specs: List[Tuple[str, int]],
    cols_to_prefix: List[str],
    binning_specs: List[Tuple[str, int]],
    cols_unchanged: List[str],
    date_col_name: str,
    output_file_path: Path,
    df_remove_pre_date_events: Optional[pl.DataFrame] = None,
) -> None:
    """
    Processes the dataset by filtering, binning, and concatenating datasets.

    Args:
        dump_path (Path): Path to the original dataset dump.
        person_subsample (List[int]): List of person IDs to filter.
        fname_prefix (str): Prefix for the output filenames.
        fname (str): Base name for the output file.
        truncation_specs (List[Tuple[str, int]]): Specifications for truncating columns.
        cols_to_prefix (List[str]): Columns to prefix.
        binning_specs (List[Tuple[str, int]]): Columns and bin sizes.
        cols_unchanged (List[str]): Columns to leave unchanged.
        date_col_name (str): Name of the date column.
        output_file_path (Path): Output file path.
        df_remove_pre_date_events (Optional[pl.DataFrame]): Dataframe with person_id and cutoff column _birthday
    """
    copy_file_or_dir(dump_path)
    # Step 1: Filter
    # NOTE: Faster to filter and dump once, rather than filter in each downstream operation (including each chunk in concat_datasets)
    if df_remove_pre_date_events is not None:
        filtered_dataset = write_person_id_birthday_filtered_dataset(
            dump_path,
            df_remove_pre_date_events,
            FPATH.TEMP_FILES / "process_dataset.parquet",
            date_col_name,
        )
    else:
        _filtered_dataset = filter_parquet_by_person_ids_to_dataset(
            dump_path, person_subsample
        )
        filtered_dataset = write_dataset_to_parquet_in_batches(
            _filtered_dataset, FPATH.TEMP_FILES / "process_dataset.parquet"
        )

    # Step 2: Bin the 'wage' and 'length' columns and get their paths
    binned_paths = []
    for col, n_bins in binning_specs:
        binned_path = bin_column(col, filtered_dataset, n_bins)
        binned_paths.append(binned_path)

    # Step 3: Concat datasets
    # # Step 3 also does other processing
    datasets = [filtered_dataset] + [
        ds.dataset(file_path, format="parquet") for file_path in binned_paths
    ]

    concat_datasets(
        datasets,
        output_file_path=output_file_path,
        truncation_specs=truncation_specs,
        cols_prefix=cols_to_prefix,
        cols_binned=binning_specs,
        date_col=date_col_name,
        cols_unchanged=cols_unchanged,
    )


# Helper function to process column configurations
def extract_specs(columns: dict):
    """Extract truncation, binning, prefix, and unchanged specs from the columns configuration.

    Args:
        columns (dict): A dictionary of column names and their preprocessing configurations.

    Returns:
        tuple: A tuple containing truncation_specs (list), binning_specs (list), cols_to_prefix (list), and cols_unchanged (list).
    """
    truncation_specs = []
    binning_specs = []
    cols_to_prefix = []
    cols_unchanged = []

    for col, specs in columns.items():
        if "truncate" in specs:
            truncation_specs.append((col, specs["truncate"]))
        if "bin" in specs:
            binning_specs.append((col, specs["bin"]))
        if specs.get("prefix", False):
            cols_to_prefix.append(col)
        if specs.get("unchanged", False):
            cols_unchanged.append(col)

    return truncation_specs, binning_specs, cols_to_prefix, cols_unchanged


def replace_zero_with_null(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    Replace values with null in specified columns based on their data type:
    - For string columns: replace '0' and '00' with null.
    - For numerical columns: replace 0 with null.

    Args:
        df (pl.DataFrame): Input Polars DataFrame.
        columns (list[str]): List of column names to apply replacements on.

    Returns:
        pl.DataFrame: Updated DataFrame with specified replacements.
    """
    for col in columns:
        # Check if the column is of numeric type (integer or float)
        if pl.Int64 == df.schema[col] or pl.Float64 == df.schema[col]:
            df = df.with_columns(
                pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col)).alias(col)
            )
        # Check if the column is of string type
        elif pl.Utf8 == df.schema[col]:
            df = df.with_columns(
                pl.when(pl.col(col).is_in(["0", "00"]))
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
    return df


def remove_prefix_from_cols(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Removes a specified prefix from all columns that start with it and replaces
    the old columns with the renamed ones in the dataframe.

    Args:
        df (pl.DataFrame): The input Polars dataframe.
        prefix (str): The prefix to remove from column names.

    Returns:
        pl.DataFrame: DataFrame with specified prefix removed from applicable column names.
    """
    return df.rename(
        {col: col[len(prefix) :] for col in df.columns if col.startswith(prefix)}
    )
