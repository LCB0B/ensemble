import pandas as pd
import polars as pl
from tqdm import tqdm

from src.earlylife.src.collate_utils import censor_person
from src.earlylife.src.dataset import FinetuneLMDBDataset, LMDBDataset
from src.earlylife.src.paths import FPATH
from src.earlylife.src.utils import calculate_abspos

ONE_YEAR_ABSPOS = 365.25 * 24
# %%
""" CHANGE THESE VARIABLES"""
OUTCOME_FNAME = "econ_vuln"
DIR_PATH = "vulnerable_full_lmdb"
SAMPLE_FOLDER = "earlylife_samples"
NEGATIVE_CENSOR = 0
# ^REMEMBER TO CHANGE ABOVE

# %%
outcome_path = FPATH.NETWORK_DATA / SAMPLE_FOLDER / f"{OUTCOME_FNAME}_targets.parquet"
lmdb_path = FPATH.NETWORK_DATA / DIR_PATH / "dataset.lmdb"
lengths_path = FPATH.NETWORK_DATA / DIR_PATH / "lengths.parquet"
if outcome_path is not None:
    outcomes = pl.read_parquet(outcome_path)
    outcomes_dict = (
        outcomes.with_columns(
            calculate_abspos(pl.col("censor")),
        )
        .to_pandas()
        .to_dict(orient="list")
    )
    dataset = FinetuneLMDBDataset(None, lmdb_path, outcomes_dict)
    lengths_path = lengths_path.with_name(
        lengths_path.stem + "_" + outcome_path.stem + lengths_path.suffix
    )
else:
    dataset = LMDBDataset(None, lmdb_path)
dataset._init_db()

# %%
lengths = {"person_id": [], "censor": [], "length": []}
for i in tqdm(range(len(dataset))):
    item = dataset[i]
    person = item["data"]
    pnr = dataset.index_to_pnr(i)
    if "outcome_info" in item:
        person = censor_person(
            person,
            item["outcome_info"],
            background=1,
            negative_censor=NEGATIVE_CENSOR,
        )
    plen = sum([len(event) for event in person["event"]])
    lengths["person_id"].append(int(pnr))
    lengths["censor"].append(item["outcome_info"]["censor"])
    lengths["length"].append(plen)


df = pd.DataFrame(lengths)

# %%


df.to_parquet(lengths_path)
FPATH.copy_to_opposite_drive(lengths_path)
