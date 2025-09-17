""" Functions responsible for creating in-memory chunks of big parquet files """

from pathlib import Path
from typing import List, Union

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

from src.earlylife.src.utils import get_pnrs


def yield_chunks(sources: Union[ds.Dataset, List[ds.Dataset]], chunk_size=100_000):
    """Yields chunks (pl.DataFrame) from sources (ds.Dataset)"""
    pnrs = get_pnrs(sources)

    for i in tqdm(range(0, len(pnrs), chunk_size), "Creating Dataset"):
        chunk_pnrs = pnrs[i : i + chunk_size]
        if isinstance(sources, ds.Dataset):
            yield get_chunk(sources, chunk_pnrs)
        elif isinstance(sources, list):
            yield [get_chunk(source, chunk_pnrs) for source in sources]
        else:
            raise ValueError(
                f"Wrong typing {type(sources)}, only ds.Dataset or List[ds.Dataset]"
            )


def yield_chunks_given_pnrs(
    sources: Union[ds.Dataset, List[ds.Dataset]], pnrs: List, chunk_size=500_000
):
    for i in tqdm(
        range(0, len(pnrs), chunk_size), f"Yielding chunks of {chunk_size:_} pnrs"
    ):
        chunk_pnrs = pnrs[i : i + chunk_size]
        if isinstance(sources, ds.Dataset):
            yield get_chunk_given_pnrs(sources, chunk_pnrs)
        elif isinstance(sources, list):
            yield [get_chunk_given_pnrs(source, chunk_pnrs) for source in sources]
        else:
            raise ValueError(
                f"Wrong typing {type(sources)}, only ds.Dataset or List[ds.Dataset]"
            )


def get_chunk(source: ds.Dataset, chunk_pnrs: List) -> pl.DataFrame:
    """Subsets source into a table using chunk_pnrs and converts to pl.DataFrame"""
    table = source.to_table(filter=pc.is_in(pc.field("person_id"), chunk_pnrs))
    return pl.from_arrow(table)


def get_chunk_given_pnrs(source: ds.Dataset, chunk_pnrs: List) -> pl.DataFrame:
    table = source.to_table(
        filter=pc.is_in(pc.field("person_id"), pa.array(chunk_pnrs))
    )
    return pl.from_arrow(table)


def read_and_concatenate_parquet_files(
    file_paths: List[str], chunk_size: int, output_file_path: Path
) -> None:
    """
    Reads chunks of K rows from multiple Parquet files, performs horizontal concatenation,
    and saves these to a common Parquet file using ParquetWriter.

    Args:
        file_paths (List[str]): List of paths to the Parquet files to read.
        chunk_size (int): Number of rows to read per chunk.
        output_file_path (Path): Path to the output Parquet file.
    """
    # Create a ParquetWriter for the output file
    writer = None

    # Create Parquet readers for all input files
    readers = [pq.ParquetFile(file_path) for file_path in file_paths]

    batch_iterators = [reader.iter_batches(batch_size=chunk_size) for reader in readers]
    i = 0
    while True:
        # Read the next batch from each file
        print(f"Writing chunk {i}")
        chunks = []

        for batch_iterator in batch_iterators:
            try:
                batch = next(batch_iterator)
                chunk = pl.from_arrow(batch)
                chunks.append(chunk)
            except StopIteration:
                # Exit the loop if any file has no more chunks
                break

        if len(chunks) != len(file_paths):
            # Exit the loop if any file has no more chunks
            break

        # Concatenate the chunks horizontally
        concatenated_chunk = pl.concat(chunks, how="horizontal")
        # Convert to pyarrow table
        table = concatenated_chunk.to_arrow()

        if writer is None:
            writer = pq.ParquetWriter(output_file_path, schema=table.schema)

        # Write the concatenated chunk to the output file
        writer.write_table(table)
        i += 1

    if writer:
        writer.close()


def write_dataset_to_parquet_in_batches(
    dataset: ds.Dataset, output_path: Path, batch_size: int = 10_000_000
) -> ds.Dataset:
    """
    Write a PyArrow Dataset to Parquet in batches and return a new Dataset pointing to the output path.
    Faster and less memory than ds.write_dataset()

    Args:
        dataset (ds.Dataset): PyArrow Dataset to write.
        output_path (Path): Path to write the Parquet file.
        batch_size (int): Number of rows per batch.

    Returns:
        ds.Dataset: A new PyArrow Dataset pointing to the output path.
    """
    with pq.ParquetWriter(output_path, dataset.schema) as writer:
        for batch in dataset.to_batches(batch_size=batch_size):
            writer.write_table(pa.Table.from_batches([batch]))

    return ds.dataset(output_path, format="parquet")
