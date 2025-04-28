"""" Functions to dump data """

from pathlib import Path
from typing import Optional
import pandas as pd
import polars as pl
import oracledb
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm
from sqlalchemy import create_engine, text as sql_text

from src.paths import FPATH


def execute_query_to_single_parquet_file(
    database: str,
    sql_script: str,
    filepath: Path,
    schema: Optional[str] = None,
    chunk_size: int = 50_000_000,
) -> None:
    """
    Executes a SQL query on a specified database and saves the results in multiple Parquet files,
    copying them to a K drive. This function does not return a DataFrame.

    Args:
        database (str): The name of the database to connect to.
        sql_script (str): The SQL script to execute.
        fname (str): The base folder name where Parquet files will be stored.
        schema (Optional[str]): The schema to use within the database. Defaults to None.
        chunk_size (int): The number of rows per chunk for splitting the SQL output.
            Defaults to 50,000,000.
    """
    oracledb.init_oracle_client()

    # Create connection string and engine
    if schema:
        connection_string = f"oracle+oracledb://[{schema}]@{database}"
    else:
        connection_string = f"oracle+oracledb://{database}"
    engine = create_engine(connection_string)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    # save to parquet in chunks
    with engine.connect() as connection:
        for df in tqdm(pd.read_sql(sql_script, con=connection, chunksize=chunk_size)):
            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(filepath, schema=table.schema)
            writer.write_table(table)

    writer.close()

    FPATH.copy_to_opposite_drive(filepath)


def save_and_copy_lazyframe(lazyframe: pl.LazyFrame, sink_path: Path) -> None:
    """
    Save a Polars LazyFrame to a parquet file and copy it to a new location.

    Args:
        lazyframe (pl.LazyFrame): The Polars LazyFrame to save.
        sink_path (Path): The pathfile to save the parquet file to.
    """
    # Save the LazyFrame to a parquet file
    lazyframe.sink_parquet(sink_path)

    # Copy the file, replacing the destination if it exists
    FPATH.copy_to_opposite_drive(sink_path)


def execute_query(database: str, sql_script: str, schema: str = None) -> pd.DataFrame:
    """
    Connects to an Oracle database and executes a given SQL query to return a DataFrame.

    Args:
    - database (str): The database name.
    - sql_script (str): The SQL query to be executed.
    - schema (str, optional): The schema name. Default is None.

    Returns:
    - pd.DataFrame: A DataFrame containing the query results.
    """
    oracledb.init_oracle_client()
    if schema:
        connection_string = f"oracle+oracledb://[{schema}]@{database}"
    else:
        connection_string = f"oracle+oracledb://{database}"
    engine = create_engine(connection_string)

    with engine.connect() as connection:
        df = pl.from_pandas(pd.read_sql_query(sql_text(sql_script), connection))

    return df
