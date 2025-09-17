import os

os.environ["POLARS_MAX_THREADS"] = "8"
os.environ["RAYON_NUM_THREADS"] = "8"

import warnings  # noqa: E402
from datetime import datetime  # noqa: E402

import hydra  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import torch  # noqa: E402
from lightning.pytorch import Trainer, seed_everything  # noqa: E402
from lightning.pytorch.callbacks import (  # noqa: E402
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import DictConfig, open_dict  # noqa: E402

from src.earlylife.src.datamodule import PretrainLifeLightningDataModule  # noqa: E402
from src.earlylife.src.encoder_nano import PretrainNanoEncoder  # noqa: E402
from src.earlylife.src.loggers import RetryOrSkipTensorBoardLogger  # noqa: E402
from src.earlylife.src.paths import (  # noqa: E402
    FPATH,
    check_and_copy_file_or_dir,
    copy_file_or_dir,
    get_wandb_runid,
)

warnings.filterwarnings(
    "ignore",
    message="Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly.",
    category=UserWarning,
    module="torch.nn.modules.activation",
)
warnings.filterwarnings(
    "ignore",
    message=".*You are using `torch.load` with `weights_only=False`.*",
    module="lightning_fabric.utilities.cloud_io",
)

import random
from typing import Any, Optional

from torch.utils.tensorboard import SummaryWriter


class IntermittentWriter:
    """
    Wraps a real SummaryWriter and randomly raises OSError
    on add_scalar and flush() to simulate flaky storage.
    """

    def __init__(self, writer: SummaryWriter, fail_rate: float = 0.2) -> None:
        """
        Args:
            writer (SummaryWriter): The underlying writer to proxy.
            fail_rate (float): Probability [0.0,1.0] of raising OSError.
        """
        self._writer = writer
        self.fail_rate = fail_rate

    def add_scalar(
        self, tag: str, scalar_value: float, global_step: Optional[int] = None
    ) -> None:
        """
        Forward to the real writer, or randomly fail.

        Args:
            tag (str): Metric name.
            scalar_value (float): Metric value.
            global_step (Optional[int]): Global step.
        """
        if random.random() < self.fail_rate:
            raise OSError("🔌 simulated add_scalar hiccup")
        self._writer.add_scalar(tag, scalar_value, global_step)

    def flush(self) -> None:
        """
        Forward to the real writer, or randomly fail on flush.

        Raises:
            OSError: If simulated failure occurs.
        """
        if random.random() < self.fail_rate:
            raise OSError("🔌 simulated flush hiccup")
        self._writer.flush()

    def close(self) -> None:
        """Forward close() so file handles get released."""
        self._writer.close()

    def __getattr__(self, name: str) -> Any:
        """Proxy any other calls/attributes to the real writer."""
        return getattr(self._writer, name)


import time

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter


class RetryEverythingSummaryWriter:
    """
    Proxy around SummaryWriter that retries every method call on Exception.

    Args:
        writer (SummaryWriter): underlying real writer.
        max_retries (int): how many attempts before giving up.
        retry_delay (float): seconds to wait between retries.
    """

    def __init__(
        self,
        writer: SummaryWriter,
        max_retries: int = 10,
        retry_delay: float = 1.0,
    ) -> None:
        self._writer = writer
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def __getattr__(self, name: str) -> Any:
        """
        Wrap any callable attribute of the real writer in retry logic.
        Non-callable attrs pass through unchanged.
        """
        orig = getattr(self._writer, name)
        if not callable(orig):
            return orig  # e.g. `log_dir`

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            for i in range(1, self._max_retries + 1):
                try:
                    return orig(*args, **kwargs)
                except Exception as e:
                    if i < self._max_retries:
                        print(
                            f"[TBRetry] `{name}` failed {i}/{self._max_retries}: {e!r}, retrying…"
                        )
                        time.sleep(self._retry_delay)
                        continue
                    print(
                        f"[TBRetry] `{name}` giving up after {i}/{self._max_retries}: {e!r}"
                    )
                    return None  # swallow on final failure

        return wrapped


class RetryEverythingTensorBoardLogger(TensorBoardLogger):
    """
    A TensorBoardLogger whose `.experiment` is wrapped to retry **all**
    SummaryWriter operations up to `max_retries` times.

    Args:
        *args: passed to TensorBoardLogger (save_dir, name, version, etc.).
        max_retries (int): how many retry attempts on any Exception.
        retry_delay (float): seconds between retries.
        **kwargs: forwarded to SummaryWriter.
    """

    def __init__(
        self,
        *args: Any,
        max_retries: int = 10,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        # force instantiation of the real SummaryWriter
        real_writer: SummaryWriter = self.experiment
        # replace it with our retrying proxy
        self._experiment = RetryEverythingSummaryWriter(
            real_writer, max_retries=max_retries, retry_delay=retry_delay
        )


# Convert DictConfig to regular dict and extract scalars only
def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Recursively flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@hydra.main(
    config_path=FPATH.CONFIGS.as_posix(),
    config_name="hparams_pretrain.yaml",
    version_base=None,
)
def main(hparams: DictConfig):
    if hparams.run_id is not None:
        run_id = hparams.run_id
    else:
        run_id = get_wandb_runid(FPATH.TB_LOGS / hparams.experiment_name)

    seed_everything(73)
    print(f"Experiment: {hparams.experiment_name}")
    # Set training variables
    torch.set_float32_matmul_precision(hparams.float32_matmul_precision)
    n_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    # Where to save
    dir_path = FPATH.DATA / hparams["dir_path"]

    # ----------- Sample info -----------

    # Sample splits
    sample_folder = FPATH.DATA / hparams.sample_folder
    sample_folder.mkdir(exist_ok=True)

    train_pids_file = sample_folder / "pretrain_train_pids.parquet"
    copy_file_or_dir(train_pids_file)

    val_pids_file = sample_folder / "pretrain_val_pids.parquet"
    copy_file_or_dir(val_pids_file)

    train_person_ids = (
        pl.read_parquet(train_pids_file)
        .with_columns(pl.col("person_id").cast(str))["person_id"]
        .to_list()
    )
    val_person_ids = (
        pl.read_parquet(val_pids_file)
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

    background = pl.read_parquet(background_path)

    # CENSORING
    print(f"Filtering sources to be before {hparams.cutoff_year}")
    cutoff_date = datetime(hparams.cutoff_year, 1, 1)
    pretrain_cutoff = pa.scalar(cutoff_date, type=pa.timestamp("ns"))
    filtered_sources = [
        ds.dataset(filepath, format="parquet").filter(
            pc.less(ds.field("date_col"), pretrain_cutoff)
        )
        for filepath in source_paths
    ]

    print(type(hparams.max_seq_len))
    # ---------- Training setup ----------
    dm = PretrainLifeLightningDataModule(
        dir_path=dir_path,
        sources=filtered_sources,
        background=background,
        cls_token=hparams.include_cls,
        sep_token=hparams.include_sep,
        train_person_ids=train_person_ids,
        val_person_ids=val_person_ids,
        segment=hparams.include_segment,
        n_tokens=hparams.n_tokens,
        num_workers=hparams.num_workers,
        max_seq_len=hparams.max_seq_len,
        cutoff=hparams.token_freq_cutoff,
        source_dir=hparams.source_dir,
        lengths=hparams.lengths,
    )

    dm.prepare_data()  # TODO: Ideally we should not call this and let Lightning call it (and get the dm info somewhere else)

    # get vocab size
    with open_dict(hparams):
        hparams.vocab_size = len(dm.pipeline.vocab)

    model = PretrainNanoEncoder(**hparams)

    # ---------- Callback setup ----------
    logger = RetryOrSkipTensorBoardLogger(
        save_dir=FPATH.TB_LOGS / hparams.experiment_name,
        name="",
        version=run_id,
        default_hp_metric=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=FPATH.CHECKPOINTS_TRANSFORMER / hparams.experiment_name / run_id,
        filename="best",
        monitor="val/loss",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [lr_monitor, checkpoint_callback]

    ckpt_path = None

    # ---------- Load failed model ----------
    if hparams.load_failed_model:
        ckpt_path = (
            FPATH.CHECKPOINTS_TRANSFORMER
            / hparams.failed_experiment_name
            / hparams.failed_run_name
            / "last.ckpt"
        )
        print("Loading failed model")

    # TEST LOGGING FAILURE
    logger = RetryEverythingTensorBoardLogger(
        save_dir=FPATH.TB_LOGS / "logs_test_logging",
        name="demo",
        version="v0",
        max_retries=10,
        retry_delay=0.25,
        default_hp_metric=False,
        # flush_secs=0,
        # max_queue=1,
    )

    # first grab the real writer
    real = logger.experiment
    # wrap it in your intermittent failure simulator
    intermittent = IntermittentWriter(real, fail_rate=0.5)
    # then wrap *that* in the retrying proxy
    wrapped = RetryEverythingSummaryWriter(
        intermittent, max_retries=10, retry_delay=0.25
    )
    # finally replace
    logger._experiment = wrapped

    # ---------- Train ----------
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="auto",
        devices=n_devices,
        callbacks=callbacks,
        logger=logger,
        strategy="auto",
        deterministic=False,
        precision=hparams.precision,
        log_every_n_steps=25,
        fast_dev_run=hparams.fast_dev_run,
        val_check_interval=100,
        limit_train_batches=1000,
        limit_val_batches=100,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
