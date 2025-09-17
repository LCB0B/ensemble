import os

os.environ["POLARS_MAX_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT
os.environ["RAYON_NUM_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT


from pathlib import PosixPath

import hydra  # noqa: E402
import polars as pl
import pyarrow.dataset as ds
import torch
import torch._dynamo.compiled_autograd
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, open_dict  # noqa: E402

from src.calculate_lengths import calculate_finetune_sequence_lengths
from src.datamodule2_Magnus import FamilyRegressionFinetuneLifeLDM
from src.encoder_nano_socmob import FamilyRegressionFinetuneNanoEncoder
from src.loggers import RetryOrSkipTensorBoardLogger
from src.paths import (
    FPATH,
    check_and_copy_file_or_dir,
    copy_file_or_dir,
    get_newest_path,
    get_wandb_runid,
)
from src.prediction_writer import (
    SaveSimpleInfoRegression,  # noqa: E402
)


@hydra.main(
    config_path=(FPATH.CONFIGS / "socmob").as_posix(),
    config_name="hparams_family_inference.yaml",
    version_base=None,
)
def main(hparams: DictConfig):
    # %%
    torch.serialization.add_safe_globals([PosixPath])
    torch._dynamo.config.cache_size_limit = 16
    # # Load hparams
    # with open(
    #     FPATH.CONFIGS / "socmob" / "hparams_family_finetune.yaml", "r", encoding="utf-8"
    # ) as stream:
    #     hparams = yaml.safe_load(stream)
    seed_everything(73)
    # Set training variables
    torch.set_float32_matmul_precision(hparams.float32_matmul_precision)

    # Where to save
    dir_path = FPATH.DATA / hparams.dir_path

    # ----------- Outcome info -----------

    sample_folder = FPATH.NETWORK_DATA / hparams.sample_folder
    sample_folder.mkdir(exist_ok=True)

    outcome = hparams.outcome
    print(f"Prediction target: {outcome}")

    # Sample splits
    train_person_ids = (
        pl.read_parquet(sample_folder / f"{outcome}_train_pids.parquet")
        .with_columns(pl.col("person_id").cast(str))["person_id"]
        .to_list()
    )
    val_person_ids = (
        pl.read_parquet(sample_folder / f"{outcome}_val_pids.parquet")
        .with_columns(pl.col("person_id").cast(str))["person_id"]
        .to_list()
    )

    test_person_ids = (
        pl.read_parquet(sample_folder / f"{outcome}_test_pids.parquet")
        .with_columns(pl.col("person_id").cast(str))["person_id"]
        .to_list()
    )
    # ----------- Input info -----------
    data_path = FPATH.DATA / hparams.source_dir
    data_path.mkdir(exist_ok=True)

    source_paths = [
        (data_path / path).with_suffix(".parquet") for path in hparams.sources
    ]
    background_path = data_path / "background.parquet"

    for s in source_paths + [background_path]:
        if hparams.force_copy_of_sources:
            copy_file_or_dir(s)
        elif hparams.use_network_io_for_sources:
            background_path = FPATH.swap_drives(background_path)
            source_paths = [FPATH.swap_drives(path) for path in source_paths]
        else:
            check_and_copy_file_or_dir(s)
    background_path = (
        FPATH.NETWORK_DATA / hparams.source_dir / hparams.background
    ).with_suffix(".parquet")
    outcomes_path = (
        FPATH.NETWORK_DATA
        / hparams.sample_folder
        / f"{hparams.outcome}_targets_transformer"
    ).with_suffix(".parquet")

    sources = [ds.dataset(s, format="parquet") for s in source_paths]
    background = pl.read_parquet(background_path)
    outcomes = pl.read_parquet(outcomes_path)

    dm = FamilyRegressionFinetuneLifeLDM(
        dir_path=dir_path,
        sources=sources,
        outcomes=outcomes,
        background=background,
        subset_background=hparams.subset_background,
        n_tokens=hparams.n_tokens,
        lengths=f"{hparams.lengths}_{hparams.outcome}_targets_transformer",
        train_person_ids=train_person_ids,
        val_person_ids=val_person_ids,
        test_person_ids=test_person_ids,
        inference_type=hparams.inference_type,
        num_workers=hparams.num_workers,
        max_seq_len=hparams.max_seq_len,
        prediction_windows=hparams.prediction_windows,
        source_dir=hparams.source_dir,
        feature_set=hparams.feature_set,
    )
    dm.prepare_data()  # TODO: Ideally we should not call this and let Lightning call it (and get the dm info somewhere else)

    calculate_finetune_sequence_lengths(
        outcome_fname=hparams.outcome,
        dir_path=hparams.dir_path,
        sample_folder=hparams.sample_folder,
        outcome_suffix="_targets_transformer",
    )

    # get vocab size
    with open_dict(hparams):
        hparams.vocab_size = len(dm.pipeline.vocab)

    checkpoint_folder = (
        FPATH.CHECKPOINTS_TRANSFORMER / hparams.checkpoint_experiment_name
    )
    checkpoint_name = (
        get_newest_path(checkpoint_folder)
        if str(hparams.checkpoint_run_name).lower() == "latest"
        else str(hparams.checkpoint_run_name)
    )
    print(
        f"Loading checkpoint: {checkpoint_name} from experiment {hparams.checkpoint_experiment_name}"
    )
    checkpoint_path = checkpoint_folder / checkpoint_name / "best.ckpt"
    model = FamilyRegressionFinetuneNanoEncoder.load_from_checkpoint(
        checkpoint_path, strict=False
    )

    pred_writer = SaveSimpleInfoRegression(
        fname=(
            FPATH.NETWORK_DATA
            / hparams.inference_data_folder
            / hparams.experiment_name
            / f"{hparams.checkpoint_run_name}_{hparams.inference_type}"
        ).with_suffix(".pt")
    )

    trainer = Trainer(
        max_epochs=hparams["max_epochs"],
        accelerator="auto",
        devices=1,
        callbacks=[pred_writer],
        strategy="auto",
        deterministic=False,
        precision=hparams["precision"],
        log_every_n_steps=10,
    )

    trainer.predict(model, datamodule=dm, return_predictions=False)


if __name__ == "__main__":
    main()
