import pandas as pd
import polars as pl
from tqdm import tqdm

from src.earlylife.src.collate_utils import censor_person
from src.earlylife.src.dataset import FinetuneLMDBDataset
from src.earlylife.src.paths import FPATH
from src.earlylife.src.utils import calculate_abspos


def calculate_finetune_sequence_lengths(
    outcome_fname: str,
    dir_path: str,
    sample_folder: str,
):
    """
    Generate a Parquet file of event-sequence lengths per person, applying optional censoring.

    Args:
        outcome_fname (str): Base name (without extension) for the outcome targets Parquet.
        dir_path (str): Subdirectory containing the LMDB dataset.
        sample_folder (str): Subfolder where the outcome file lives.
    """
    # Paths
    outcome_path = FPATH.DATA / sample_folder / f"{outcome_fname}_targets.parquet"
    lmdb_path = FPATH.DATA / dir_path / "dataset.lmdb"
    lengths_path = FPATH.DATA / dir_path / "lengths.parquet"
    # adjust lengths filename to include outcome name
    lengths_path = lengths_path.with_name(
        lengths_path.stem + "_" + outcome_path.stem + lengths_path.suffix
    )

    # TODO: Is this sufficient, or do we need some sort of check of network drive?
    # iirc created on local io and then treated as part of dir_path, so will be copied back and forth with teardowns and check_and_copy_file_or_dir
    if not lengths_path.exists():

        outcomes = pl.read_parquet(outcome_path)
        outcomes_dict = (
            outcomes.with_columns(calculate_abspos(pl.col("censor")))
            .to_pandas()
            .to_dict(orient="list")
        )
        dataset = FinetuneLMDBDataset(None, lmdb_path, outcomes_dict)
        dataset._init_db()

        # Calculate lengths
        lengths = {"person_id": [], "censor": [], "length": []}
        for idx in tqdm(range(len(dataset)), desc="Computing lengths"):
            item = dataset[idx]
            person_events = item["data"]
            if "outcome_info" in item:
                person_events = censor_person(
                    person_events,
                    item["outcome_info"],
                    background=1,
                )
            total_len = sum(len(ev) for ev in person_events["event"])
            lengths["person_id"].append(int(dataset.index_to_pnr(idx)))
            lengths["censor"].append(item.get("outcome_info", {}).get("censor"))
            lengths["length"].append(total_len)

        # Save
        df = pd.DataFrame(lengths)
        df.to_parquet(lengths_path)
        # TODO: Teardown should handle this, I guess?
        # FPATH.copy_to_opposite_drive(lengths_path)
