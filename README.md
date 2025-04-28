# Project2vec


## Dumps
The first step is to dump the relevant SQL tables to parquet, as data extraction can take quite awhile. This is only done **once**, so it is worth doing

Dumping happens in a notebook, one for each table, so that any needed work can take place

**Defintion:** A dump is a parquet file from an SQL table, where relevant columns are extracted and renamed

**Location:**: PROJECT_PATH/data/dumps

**Example:** 
```
fname: Name of the dumped file
database: Database to read from
SQL_SCRIPT: Script to execute

from src.dumping import execute_query_to_single_parquet_file
from src.paths import FPATH

dump_fpath=FPATH.DUMP_DIR / {fname}_DUMP.parquet
execute_query_to_single_parquet_file(database, SQL_SCRIPT, filepath=dump_fpath, chunk_size=50_000_000)
```

## Post-processing
As the dumps represent a **raw** presentation of the SQL tables, you might want to apply some post-processing, such as binning, prefixing and truncating.

A post-processed filename should be the fname of the dump followed by "_VERSION", where VERSION is project-specific and a name for the post-processing procedure (e.g. "life2vec")
**Definition:** A post-processed file is a DUMP that has been processed (e.g. binned, prefixed or truncated)

**Location:** PROJECT_PATH/data

## Data module
The data module is implemented through Lightning and combines all the dumps into a single stream, creating EVENTS for each person. The data module is used directly in the Trainer and uses LMDB as the Dataset backend

**Definition:** The final version of the data that is converted to sequences

**Location:** PROJECT_PATH/data/DATA_MODULE_NAME

### EVENT
An Event consists of a PERSON_ID, to identify the individual, a DATE, to identifcy when it happened, and INFORMATION, where information can be one or more columns of info.
```
PERSON_ID: The ID of the individual
DATE_COL: The datetime of the event
INFORMATION: One or more columns with info about the event
```

