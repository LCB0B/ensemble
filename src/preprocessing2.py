import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

import polars as pl
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

from src.chunking import write_dataset_to_parquet_in_batches
from src.paths import FPATH, check_and_copy_file, copy_file_or_dir
from src.utils import filter_parquet_by_person_ids_to_dataset


def unique_preserve_order(lst: List[Any]) -> List[Any]:
    """
    Return unique elements in a list while preserving their original order.

    Args:
        lst (List[Any]): A list of elements.

    Returns:
        List[Any]: A list of unique elements in the order they appear.
    """
    seen = set()
    unique_lst = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            unique_lst.append(item)
    return unique_lst


def bin_column(
    col_name: str, dataset: ds.Dataset, bins: int, temp_path: Path = FPATH.TEMP_FILES
) -> Path:
    """
    Bins the values of a specified column in a dataset into quantiles, writes the binned result to a parquet file,
    and returns the file path.

    Args:
        col_name (str): The name of the column to bin.
        dataset (ds.Dataset): The input dataset containing the column.
        bins (int): The number of bins to create. Defaults to 100.

    Returns:
        Path: The path to the saved parquet file.
    """
    # Convert column to Polars DataFrame
    col = pl.from_arrow(dataset.to_table(columns=[col_name]))

    # Generate cutoff points for binning
    cutoffs = [i / bins for i in range(1, bins)]

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
    output_path = temp_path / f"binned_{col_name}_temp.parquet"

    # Write the binned column to a parquet file
    print(f"{col_name}: Binning...")
    col.select(binned_col).write_parquet(output_path)

    return output_path


def bin_column_grouped_by(
    col_name: str,
    group_col: str,
    dataset: ds.Dataset,
    bins: int,
    temp_path: Path = FPATH.TEMP_FILES,
) -> Path:
    """
    Bins the values of a specified column into quantiles within each group, preserving the original row order,
    writes the binned result to a parquet file, and returns the file path.

    Args:
        col_name (str): The name of the column to bin.
        group_col (str): The column name to group by.
        dataset (ds.Dataset): The input dataset containing the columns.
        bins (int): The number of bins to create.

    Returns:
        Path: The path to the saved parquet file.
    """
    # Read required columns from dataset and convert to Polars DataFrame
    df = pl.from_arrow(dataset.to_table(columns=[col_name, group_col]))
    df = df.with_columns(pl.col(group_col).dt.year().alias("group_col")).drop(group_col)

    # Add an index to preserve original row order
    df = df.with_row_index("order")

    # Define cutoff points and labels
    cutoffs = [i / bins for i in range(1, bins)]
    labels = [f"{col_name}_Q{i}" for i in range(1, bins + 1)]

    # Compute rank and unique count per group, then calculate bin ratio and apply cut.
    df = df.with_columns(
        [
            pl.col(col_name).rank(method="dense").over("group_col").alias("rank"),
            pl.col(col_name).n_unique().over("group_col").alias("n_unique"),
        ]
    ).drop(["group_col", col_name])

    df = df.with_columns(
        (
            (pl.col("rank") / pl.col("n_unique"))
            .cut(cutoffs, labels=labels)
            .cast(pl.Utf8)
        ).alias(f"binned_grouped_{col_name}")
    )

    # Restore original order and select only the binned column
    df = df.sort("order")
    result_df = df.select(f"binned_grouped_{col_name}")

    # Define the output path using a temporary directory
    output_path = temp_path / f"binned_grouped_{col_name}.parquet"
    result_df.write_parquet(output_path)

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


def prefix_value_keep_nulls(
    df: pl.DataFrame, cols_value_list: list[(str, str)]
) -> pl.DataFrame:
    """
    Iterate through specified columns in a DataFrame and prefix non-null values
    with the given value name. Null values are left unchanged.

    Args:
        df (pl.DataFrame): The input DataFrame.
        cols (: list[(str, str)]): List of tuples with column names and values.

    Returns:
        pl.DataFrame: The modified DataFrame with prefixed non-null values.
    """
    for col, value in cols_value_list:
        df = df.with_columns(
            pl.when(pl.col(col).is_not_null())
            .then(pl.lit(f"{value}_") + pl.col(col).cast(pl.Utf8))
            .otherwise(pl.col(col))  # Keep null as null
            .alias(col)
        )
    return df


def concat_datasets(
    datasets: List[ds.Dataset],
    output_file_path: Path,
    truncation_specs: List[Tuple[str, int]],
    cols_to_prefix_colname: List[str],
    cols_to_prefix_value: List[Tuple[str, str]],
    cols_overall_binned: List[str],
    cols_yearly_binned: List[str],
    cols_unchanged: List[str],
    date_col: str,
    chunk_size: int = 200_000_000,
    overall_group_name: Optional[str] = None,
) -> None:
    """
    Reads chunks from multiple Parquet files using dataset.take in chunks, applies truncation and prefixing only on the first chunk of the
    original file, performs horizontal concatenation, and saves the result to a common Parquet file using ParquetWriter.
    Only specific columns (prefixed, binned, unchanged) are kept in the final output.

    Args:
        datasets (ds.Dataset): List of Datasets to process and concat.
        output_file_path (Path): Path to the output Parquet file.
        truncation_specs (List[Tuple[str, int]]): Columns to truncate and their respective lengths.
        cols_to_prefix_colname (List[str]): Columns to prefix non-null values with column name.
        cols_to_prefix_value (List[Tuple[str, str]): Columns to prefix non-null values with value.
        cols_binned (List[str]): Columns to be binned and kept in the output.
        cols_unchanged (List[str]): Columns to keep unchanged in the output.
        date_col (str): Column to be renamed to 'date_col'.
        chunk_size (int): Number of rows to read per chunk. Default 100,000,000.
        overall_group_name (Optional(str)): Overall group name to prefix onto all values.
    """
    individual_total_rows = [dataset.count_rows() for dataset in datasets]
    total_rows = individual_total_rows[0]
    assert all(
        total == total_rows for total in individual_total_rows
    ), "All datasets must have the same number of rows."

    indices = list(range(0, total_rows, chunk_size))

    # Create a list of binned column names
    binned_cols_names = [f"binned_{bin_name}" for bin_name in cols_overall_binned] + [
        f"binned_grouped_{bin_name}" for bin_name in cols_yearly_binned
    ]

    cols_which_were_binned = cols_overall_binned + cols_yearly_binned

    # Define the columns to keep
    _cols_to_keep = unique_preserve_order(
        ["person_id", "date_col"]
        + cols_to_prefix_colname
        + [col_val_tuple[0] for col_val_tuple in cols_to_prefix_value]
        + binned_cols_names
        + cols_unchanged
    )

    # Remove column which have been binned
    cols_to_keep = [col for col in _cols_to_keep if col not in cols_which_were_binned]

    writer = None
    for start_idx in tqdm(indices, "Writing to file"):
        chunks = []
        for i, dataset in enumerate(datasets):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = dataset.take(indices=list(range(start_idx, end_idx)))
            chunk = pl.from_arrow(chunk)

            # First dataset is full dataset, remaining are binned columns
            if i == 0:
                if truncation_specs:
                    chunk = truncate_columns(chunk, truncation_specs)
                if cols_to_prefix_colname:
                    chunk = prefix_col_name_keep_nulls(chunk, cols_to_prefix_colname)
                if cols_to_prefix_value:
                    chunk = prefix_value_keep_nulls(chunk, cols_to_prefix_value)

                chunk = chunk.rename({date_col: "date_col"})

            # Prefix all values with possible overall group name
            if overall_group_name:
                overall_group_name_list = [
                    (col, overall_group_name)
                    for col in chunk.columns
                    if col not in ("person_id", "date_col")
                ]
                chunk = prefix_value_keep_nulls(chunk, overall_group_name_list)

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


from pathlib import Path
from typing import Union

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# def write_person_id_birthday_filtered_dataset(
#     file_path: Union[str, Path],
#     person_df: pl.DataFrame,
#     output_path: Path,
#     date_col: str,
#     batch_size: int = 100_000_000,
# ) -> ds.Dataset:
#     """
#     Stream‐reads a Parquet dataset in batches, but first globally filters to:
#       1) person_id ∈ your lookup table
#     Then does the in‐memory join+exact birthday filter and writes out Parquet.

#     Args:
#         file_path (Union[str, Path]): Input Parquet file or directory.
#         person_df (pl.DataFrame): Polars DF with `person_id` and `birthday`.
#         output_path (Path): Where to write the filtered Parquet.
#         date_col (str): Name of the date column in the dataset.
#         batch_size (int): Rows per Arrow‐scan batch.

#     Returns:
#         ds.Dataset: A PyArrow Dataset pointing at the output Parquet.
#     """
#     # 1) Prepare lookup structures
#     person_tbl = person_df.to_arrow()
#     person_ids = person_df["person_id"].unique().to_list()

#     # 2) Build dataset + scanner
#     dataset = ds.dataset(file_path, format="parquet")
#     schema = dataset.schema

#     scanner = dataset.scanner(
#         batch_size=batch_size,  # Internal fragmentation will dramatically reduce this batch size, but eh
#         filter=ds.field("person_id").isin(person_ids),
#     )

#     # 3) Stream, join, precise‐filter, write with progress bar
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     with pq.ParquetWriter(output_path, schema) as writer:
#         for rb in tqdm(
#             scanner.to_batches(),
#             desc="Filtering to remove pre-birth events",
#         ):
#             tbl = pa.Table.from_batches([rb])
#             joined = tbl.join(person_tbl, keys="person_id", join_type="inner")
#             mask = pc.greater_equal(joined[date_col], joined["_birthday"])
#             filtered = joined.filter(mask)
#             if filtered.num_rows:
#                 writer.write_table(filtered.select(schema.names))

#     # 4) Return the new dataset
#     return ds.dataset(output_path, format="parquet")


def write_person_id_birthday_filtered_dataset(
    file_path: str,
    person_df: pl.DataFrame,
    output_path: Path,
    date_col: str,
    batch_size: int = 100_000_000,
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
    n_rows = input_dataset.count_rows()
    n_batches = math.ceil(n_rows / batch_size)
    output_path.parent.mkdir(exist_ok=True)
    # Step 2: Prepare the Parquet writer
    with pq.ParquetWriter(output_path, input_dataset.schema) as writer:

        # Step 3: Process the dataset in batches (done using take to avoid small batches)
        for i in tqdm(range(0, n_batches), "Filtering to exclude previous events"):

            lowest_idx = i * batch_size
            highest_idx = (i + 1) * batch_size
            if highest_idx > n_rows:
                highest_idx = n_rows

            batch_table = input_dataset.take(indices=range(lowest_idx, highest_idx))

            # Step 4: Perform an in-memory join of the current batch with the birthdays (person_table)
            batch_table = batch_table.join(
                person_table, keys="person_id", join_type="inner"
            )

            # Step 5: Apply the date filter (where date_col >= birthday)
            batch_table = batch_table.filter(
                pc.greater_equal(batch_table[date_col], batch_table["_birthday"])
            )

            # Step 6: Ensure the table schema matches the original schema by selecting the original columns
            batch_table = batch_table.select(input_dataset.schema.names)

            # Step 7: Write the filtered table in batches
            if len(batch_table) > 0:
                writer.write_table(batch_table)

    # Return the new dataset pointing to the output path
    return ds.dataset(output_path, format="parquet")


def process_dataset(
    dump_path: Path,
    person_subsample: List[int],
    truncation_specs: List[Tuple[str, int]],
    cols_to_prefix_colname: List[str],
    cols_to_prefix_value: List[Tuple[str, str]],
    overall_binning_specs: List[Tuple[str, int]],
    yearly_binning_specs: List[Tuple[str, str, int]],
    cols_unchanged: List[str],
    date_col_name: str,
    output_file_path: Path,
    df_remove_pre_date_events: Optional[pl.DataFrame] = None,
    overall_group_name: Optional[str] = None,
    chunk_size: Optional[int] = 200_000_000,
    copy: Optional[bool] = True,
    temp_files: Optional[Path] = FPATH.swap_drives(FPATH.TEMP_FILES),
) -> None:
    """
    Processes the dataset by filtering, binning, and concatenating datasets.

    Args:
        dump_path (Path): (Path) Path to the original dataset dump.
        person_subsample (List[int]): List of person IDs to filter.
        truncation_specs (List[Tuple[str, int]]):  Specifications for truncating columns.
        cols_to_prefix_colname (List[str]):  Columns to prefix with their name.
        cols_to_prefix_value (List[Tuple[str, str]]): Columns to prefix with a fixed value.
        binning_specs (List[Tuple[str, int]]): Columns and the bin amount.
        binning_group_by (List[Tuple[str, str, int]]): List of tuples with (column_to_bin, n_bins, grouping_col). Grouping_col should be the date col, as it extracts the year from this
        cols_unchanged (List[str]): Columns to leave unchanged.
        date_col_name (str):  Name of the date column.
        output_file_path (Path): Path for the final output file.
        df_remove_pre_date_events (Optional[pl.DataFrame]): DataFrame used for filtering.
        overall_group_name (Optional[str]): Overall group name to prefix onto all values.
        chunk_size (Optional[int]): Chunk size when processing data in memory.
        copy (Optional[bool]): To copy or not the files in the other drive
    """
    if copy:
        copy_file_or_dir(dump_path)
    # check_and_copy_file(dump_path)
    # Step 1: Filter
    # NOTE: Faster to filter and dump once, rather than filter in each downstream operation (including each chunk in concat_datasets)
    if df_remove_pre_date_events is not None:
        pre_date_events_pnrs = set(
            df_remove_pre_date_events.select("person_id").unique().to_series().to_list()
        )
        assert pre_date_events_pnrs >= set(
            person_subsample
        ), "pre_date_events is not a superset of person_subsample, so person_subsample is not your actual sample"
        df_remove_pre_date_events = df_remove_pre_date_events.filter(
            pl.col("person_id").is_in(person_subsample)
        )
        filtered_dataset = write_person_id_birthday_filtered_dataset(
            dump_path,
            df_remove_pre_date_events,
            temp_files / "process_dataset.parquet",
            date_col_name,
            batch_size=chunk_size,
        )
    else:
        _filtered_dataset = filter_parquet_by_person_ids_to_dataset(
            dump_path, person_subsample
        )
        filtered_dataset = write_dataset_to_parquet_in_batches(
            _filtered_dataset, temp_files / "process_dataset.parquet"
        )

    # Step 2: Bin columns and get their paths
    binned_paths = []
    binned_overall_cols = []
    binned_yearly_cols = []
    for col, n_bins, group_col in yearly_binning_specs:
        print(f"Yearly binning: {col}")
        binned_path = bin_column_grouped_by(
            col, group_col, filtered_dataset, n_bins, temp_path=temp_files
        )
        binned_paths.append(binned_path)
        binned_yearly_cols.append(col)
    for col, n_bins in overall_binning_specs:
        print(f"Overall binning: {col}")
        binned_path = bin_column(col, filtered_dataset, n_bins, temp_path=temp_files)
        binned_paths.append(binned_path)
        binned_overall_cols.append(col)
    # Step 3: Concat datasets
    # # Step 3 also does other processing
    datasets = [filtered_dataset] + [
        ds.dataset(file_path, format="parquet") for file_path in binned_paths
    ]

    concat_datasets(
        datasets,
        output_file_path=output_file_path,
        truncation_specs=truncation_specs,
        cols_to_prefix_colname=cols_to_prefix_colname,
        cols_to_prefix_value=cols_to_prefix_value,
        cols_yearly_binned=binned_yearly_cols,
        cols_overall_binned=binned_overall_cols,
        date_col=date_col_name,
        cols_unchanged=cols_unchanged,
        chunk_size=chunk_size,
        overall_group_name=overall_group_name,
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
    overall_binning_specs = []
    yearly_binning_specs = []
    cols_to_prefix_colname = []
    cols_to_prefix_value = []
    cols_unchanged = []

    for col, specs in columns.items():
        if "truncate" in specs:
            truncation_specs.append((col, specs["truncate"]))
        if "overall_bin" in specs:
            overall_binning_specs.append((col, specs["overall_bin"]))
        if "yearly_bin" in specs:
            yearly_binning_specs.append(
                (col, specs["yearly_bin"][0], specs["yearly_bin"][1])
            )
        if specs.get("prefix_colname", False):
            cols_to_prefix_colname.append(col)
        if specs.get("prefix_value", False):
            cols_to_prefix_value.append((col, specs["prefix_value"]))
        if specs.get("unchanged", False):
            cols_unchanged.append(col)

    return (
        truncation_specs,
        overall_binning_specs,
        yearly_binning_specs,
        cols_to_prefix_colname,
        cols_to_prefix_value,
        cols_unchanged,
    )


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


def bin_numeric_column(
    df: pl.DataFrame, col: str, cutoffs: List[float]
) -> pl.DataFrame:
    """
    Bin a numeric column into string labels based on provided cutoffs.

    For each adjacent pair of cutoffs, values greater than the lower cutoff
    and less than or equal to the upper cutoff are assigned a label
    "bin_{lower}_{upper}". Values above the last cutoff are labeled as
    "over_{last_cutoff}". Values not falling into any bin are labeled as "unbinned".

    Args:
        df (pl.DataFrame): The Polars DataFrame.
        col (str): The name of the column to bin.
        cutoffs (List[float]): A sorted list of cutoff values.

    Returns:
        pl.DataFrame: The DataFrame with an added column "{col}_binned" containing bin labels.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not cutoffs:
        raise ValueError("Cutoffs list cannot be empty.")

    expr = None
    # Build the when-then chain for adjacent cutoff pairs.
    for i in range(len(cutoffs) - 1):
        lower, upper = cutoffs[i], cutoffs[i + 1]
        condition = (pl.col(col) > lower) & (pl.col(col) <= upper)
        if lower + 1 == upper:
            label = f"manual_bin_{upper}"
        else:
            label = f"manual_bin_{lower}_{upper}"
        if expr is None:
            expr = pl.when(condition).then(pl.lit(label))
        else:
            expr = expr.when(condition).then(pl.lit(label))

    # Condition for values above the last cutoff.
    expr = (
        expr.when(pl.col(col) > cutoffs[-1])
        .then(pl.lit(f"over_{cutoffs[-1]}"))
        .otherwise(pl.lit(None, pl.Utf8))
    )

    return df.with_columns(expr.alias(f"manual_bin_{col}"))


def normalize_by_year(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    Normalize specified columns within each year group using z-score.

    Args:
        df (pl.DataFrame): Input DataFrame containing a 'year' column.
        columns (list[str]): List of column names (str) to normalize.
    """
    return (
        df.group_by("year")
        .agg(
            [
                *[pl.col(c).mean().alias(f"{c}_mean") for c in columns],
                *[pl.col(c).std().alias(f"{c}_std") for c in columns],
            ]
        )
        .join(df, on="year")
        .with_columns(
            [
                ((pl.col(c) - pl.col(f"{c}_mean")) / pl.col(f"{c}_std")).alias(
                    f"{c}_norm"
                )
                for c in columns
            ]
        )
        .select(df.columns + [f"{c}_norm" for c in columns])
    )
