import os

os.environ["POLARS_MAX_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT
os.environ["RAYON_NUM_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT


# %%
print("hello")

# %%


# calculate_finetune_sequence_lengths(
#     outcome_fname="income",
#     dir_path="socmob_family_income9",
#     sample_folder="socmob_sample",
#     outcome_suffix="_targets_transformer",
# )

# %%
# outcome_fname = "income"
# dir_path = "socmob_family_income9"
# sample_folder = "socmob_sample"
# outcome_suffix = "_targets_transformer"
# # Paths
# outcome_path = FPATH.DATA / sample_folder / f"{outcome_fname}{outcome_suffix}.parquet"
# lmdb_path = FPATH.DATA / dir_path / "dataset.lmdb"
# lengths_path = FPATH.DATA / dir_path / "lengths.parquet"
# # adjust lengths filename to include outcome name
# lengths_path = lengths_path.with_name(
#     lengths_path.stem + "_" + outcome_path.stem + lengths_path.suffix
# )
# # TODO: Is this sufficient, or do we need some sort of check of network drive?
# # iirc created on local io and then treated as part of dir_path, so will be copied back and forth with teardowns and check_and_copy_file_or_dir
# if not lengths_path.exists():
#     outcomes = pl.read_parquet(outcome_path)
#     outcomes_dict = (
#         outcomes.with_columns(calculate_abspos(pl.col("censor")))
#         .to_pandas()
#         .to_dict(orient="list")
#     )
#     dataset = FinetuneLMDBDataset(None, lmdb_path, outcomes_dict)
#     dataset._init_db()
#     # Calculate lengths
#     lengths = {"person_id": [], "censor": [], "length": []}
#     for idx in tqdm(range(len(dataset)), desc="Computing lengths"):
#         item = dataset[idx]
#         person_events = item["data"]
#         if "outcome_info" in item:
#             person_events = censor_person(
#                 person_events,
#                 item["outcome_info"],
#                 background=1,
#             )
#         total_len = sum(len(ev) for ev in person_events["event"])
#         lengths["person_id"].append(int(dataset.index_to_pnr(idx)))
#         lengths["censor"].append(item.get("outcome_info", {}).get("censor"))
#         lengths["length"].append(total_len)
#     # Save
#     df = pd.DataFrame(lengths)
#     # df.to_parquet(lengths_path)
# %%
import pickle

import pandas as pd
import polars as pl
from tqdm import tqdm

from src.collate_utils import censor_person
from src.dataset import FinetuneLMDBDataset
from src.paths import FPATH
from src.utils import calculate_abspos

with open(FPATH.NETWORK_DATA / "dump_dict.pkl", "rb") as f:
    batch = pickle.load(f)

# %%

batch

# %%

any(batch["event"][1:] != batch["target"][:-1])

# %%

batch["event"].shape

# %%

batch["event"]

# %%


batch["target"][:-1][batch["event"][1:] != batch["target"][:-1]]
# %%

batch["target"][:-1][batch["event"][1:] != batch["target"][:-1]]

# %%


batch["target"].shape

# %%

(batch["event"][1:] != batch["target"][:-1]).sum()

# %%

data_
# %%
data_loaded.keys()

# %%

parents = data_loaded.pop("parents", None)
# %%
df = pl.DataFrame(data_loaded)
# %%

df.with_columns(pl.Series(parents).alias("parents"))
# %%
pl.read_parquet(
    FPATH.NETWORK_DATA / "socmob_sample" / "income_targets_transformer.parquet"
)
# %%
data_loaded.keys()
# %%
len(parents)
# %%
df.shape
# %%
parents
# %%

# %%

df.with_columns(
    pl.Series([x if x is not None else [] for x in parents])
    .map_elements(lambda x: None if len(x) == 0 else x)
    .alias("parents")
)
# %%
import torch

# %%
preds = torch.load(
    r"K:\project2vec\project2vec\data\socmob_sample_preds\test_family_income\test_family_income_test.pt"
)

# %%
concatenated = {}
n_preds_per_person = int(len(preds["predictions"]) / len(preds["person_id"]))
for key in preds.keys():
    if isinstance(preds[key][0], torch.Tensor):
        concatenated[key] = torch.cat(preds[key]).view(n_preds_per_person, -1)
    else:
        concatenated[key] = preds[key]

# %%

len(preds["targets"])
# %%
preds

# %%
len(preds["targets"][0])
# %%
preds["targets"]
# %%


# %%

import pickle

import pandas as pd
import polars as pl
from tqdm import tqdm

from src.collate_utils import censor_person
from src.dataset import FinetuneLMDBDataset
from src.paths import FPATH
from src.utils import calculate_abspos

# %%
df = pl.read_parquet(
    FPATH.NETWORK_DATA / "socmob_sample" / "pretrain_outcomes_transformer.parquet"
)
# %%
df

# %%
match_relations = ["Mother"]
df.filter(
    pl.col("parents")
    .list.eval(pl.element().struct.field("relation_details").is_in(match_relations))
    .list.any()
)

# %%

df.filter(pl.col("person_id") == 2272542)["parents"][0][0]
# %%
[{SOME_INT, "Father"}, {SOMEOTHER_INT, "Mother"}]
