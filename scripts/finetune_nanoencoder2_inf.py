import os
from pathlib import PosixPath
import torch._dynamo.compiled_autograd
import yaml
import torch
import pyarrow.dataset as ds
import polars as pl

from src.datamodule2 import PredictFinetuneLifeLDM
from src.encoder_nano_risk import PredictionFinetuneNanoEncoder
from src.paths import FPATH, check_and_copy_file_or_dir, get_wandb_runid
from lightning.pytorch import Trainer
from src.loggers import RetryOrSkipTensorBoardLogger
from lightning.pytorch import seed_everything
from src.prediction_writer import SaveSimpleInfo


if __name__ == "__main__":
    torch.serialization.add_safe_globals([PosixPath])
    torch._dynamo.config.cache_size_limit = 16
    # Load hparams
    with open(
        FPATH.CONFIGS / "hparams_risk_finetune.yaml", "r", encoding="utf-8"
    ) as stream:
        hparams = yaml.safe_load(stream)
    hparams["experiment_name"] = hparams["experiment_name"] + "_inf"
    run_id = f"{get_wandb_runid(FPATH.TB_LOGS / hparams['experiment_name'])}-{hparams['outcome']}"
    print(f"Experiment: {hparams['experiment_name']} / {run_id}")

    seed_everything(73)
    # Set training variables
    torch.set_float32_matmul_precision(hparams["float32_matmul_precision"])
    N_DEVICES = (
        len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if torch.cuda.is_available()
        else 1
    )

    #### Data ####
    source_paths = [
        (FPATH.DATA / hparams["source_dir"] / path).with_suffix(".parquet")
        for path in hparams["sources"]
    ]
    background_path = (
        FPATH.DATA / hparams["source_dir"] / hparams["background"]
    ).with_suffix(".parquet")
    outcomes_path = (
        FPATH.DATA / hparams["source_dir"] / hparams["outcome"]
    ).with_suffix(".parquet")

    for s in source_paths + [background_path, outcomes_path]:
        check_and_copy_file_or_dir(s)

    sources = [ds.dataset(s, format="parquet") for s in source_paths]
    background = pl.read_parquet(background_path)
    outcomes = pl.read_parquet(outcomes_path)

    dm = PredictFinetuneLifeLDM(
        dir_path=FPATH.DATA / hparams["dir_path"],
        sources=sources,
        outcomes=outcomes,
        background=background,
        cls_token=hparams["cls_token"],
        sep_token=hparams["sep_token"],
        subset_background=hparams["subset_background"],
        n_tokens=hparams["n_tokens"],
        lengths=hparams["lengths"],
        num_workers=hparams["num_workers"],
        max_seq_len=hparams["max_seq_len"],
        prediction_windows=hparams["prediction_windows"],
        source_dir=hparams["source_dir"],
    )
    dm.prepare_data()  # TODO: Ideally we should not call this and let Lightning call it (and get the dm info somewhere else)

    # get vocab size
    hparams["vocab_size"] = len(dm.pipeline.vocab)

    # Load checkpoint if fine-tuning
    model = PredictionFinetuneNanoEncoder(**hparams)

    # Trainer setup
    logger = RetryOrSkipTensorBoardLogger(
        save_dir=FPATH.TB_LOGS / hparams["experiment_name"],
        name="",
        version=run_id,
        default_hp_metric=False,
    )
    pred_writer = SaveSimpleInfo(
        fname=(
            FPATH.NETWORK_DATA / "preds" / hparams["experiment_name"] / run_id
        ).with_suffix(".pt")
    )

    callbacks = [pred_writer]

    checkpoint_path = FPATH.CHECKPOINTS_TRANSFORMER / hparams["checkpoint"]

    model = model.load_from_checkpoint(checkpoint_path)
    model.eval()

    trainer = Trainer(
        max_epochs=hparams["max_epochs"],
        accelerator="auto",
        devices=N_DEVICES,
        callbacks=callbacks,
        logger=logger,
        strategy="auto",
        deterministic=False,
        precision=hparams["precision"],
        log_every_n_steps=1000,
    )

    # Train
    trainer.predict(model, datamodule=dm, return_predictions=False)
