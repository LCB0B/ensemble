import os

os.environ["POLARS_MAX_THREADS"] = "4"  # MUST BE BEFORE POLARS IMPORT
os.environ["RAYON_NUM_THREADS"] = "4"  # MUST BE BEFORE POLARS IMPORT

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import yaml  # noqa: E402
import torch  # noqa: E402s
import warnings  # noqa: E402
import polars as pl  # noqa: E402
from src.datamodule4 import PretrainLifeLightningDataModule, LifeLightningDataModule  # noqa: E402
from src.encoder_nano_risk import PretrainNanoEncoder, CausalEncoder  # noqa: E402
from src.paths import FPATH, check_and_copy_file_or_dir, get_wandb_runid  # noqa: E402
from lightning.pytorch import Trainer  # noqa: E402
from lightning.pytorch.strategies import DDPStrategy  # noqa: E402
from lightning.pytorch.callbacks import (  # noqa: E402
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger  # noqa: E402
from lightning.pytorch import seed_everything  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402fire
import json

# This is an erroneous warning, the mask is indeed already bool
warnings.filterwarnings(
    "ignore",
    message="Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly.",
    category=UserWarning,
    module="torch.nn.modules.activation",
)

LOAD_MODEL = False

# Load hparams
with open(FPATH.CONFIGS / "hparams_pretrain2.yaml", "r") as stream:
    hparams = yaml.safe_load(stream)
run_id = get_wandb_runid(FPATH.TB_LOGS / hparams["experiment_name"])

seed_everything(73)
print(f"Experiment: {hparams['experiment_name']}")

# Set training variables
torch.set_float32_matmul_precision(hparams["float32_matmul_precision"])
n_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
# assert (
#     n_devices == 4
# ), "Not training on all four GPUs. If this is intended, comment out this line. To enable all GPUs, run 'export CUDA_VISIBLE_DEVICES=0,1,2,3' from terminal"

#### Data ####

# TODO: Give data through config?
dir_path = FPATH.DATA / hparams["dir_path"]




print(dir_path)
# TODO: Move all to config
dm = LifeLightningDataModule(
    dir_path=dir_path,
    lmdb_path=dir_path / "dataset.lmdb",
    vocab_path=dir_path / "vocab.json",
    pnr_to_idx_path=dir_path / "pnr_to_database_idx.json",
    background_length = 0,
    cls_token=True,  # hparams["include_cls"],
    sep_token=False,  # hparams["include_sep"],
    segment=False,  # hparams["include_segment"],
    batch_size=hparams["batch_size"],
    num_workers=hparams["num_workers"],
    max_seq_len=hparams["max_seq_len"],
)

dm.dataset = dm._load_dataset(path=dir_path / "dataset.lmdb")
    
dm.dataset.split(0.8)  # train, val, test

with open(dir_path / "pnr_to_database_idx.json", "r", encoding="utf-8") as json_file:
    dm.dataset.pnr_to_database_idx = json.load(json_file)

# get vocab size
hparams["vocab_size"] = len(dm.vocab)
print(f'Vocabulary size { len(dm.vocab)}')
train_dataloader_length = int(dm.dataset.__len__()*0.8)

# The iterations are spread out over the devices, hence the division by devices
hparams["steps_per_epoch"] = train_dataloader_length / n_devices
hparams["optimizer_max_iters"] = hparams["max_epochs"] * hparams["steps_per_epoch"]
hparams["swiglu"] = True
#model = CausalEncoder(**hparams)
model = PretrainNanoEncoder(**hparams)

if hparams["compile"]:
    model = torch.compile(model)
    print("Model has been compiled")

# Trainer setup

logger = TensorBoardLogger(
    save_dir=FPATH.TB_LOGS / hparams["experiment_name"],
    name="",
    version=run_id,
    default_hp_metric=False,
)
# Generate and print the TensorBoard command
log_dir = FPATH.TB_LOGS / hparams["experiment_name"]
print(f"Run the following command to start TensorBoard:\ntensorboard --logdir {log_dir}")

lr_monitor = LearningRateMonitor(logging_interval="step")

if LOAD_MODEL:
    checkpoint_dir = FPATH.CHECKPOINTS / hparams["checkpoint_run_name"] / run_id
else:
    checkpoint_dir = FPATH.CHECKPOINTS / hparams["experiment_name"] / run_id

checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="best",
    monitor="val/loss",
    save_top_k=1,
    save_last=True,
)
# early_stopping_callback = EarlyStopping("val/acc", patience=3)

callbacks = [lr_monitor, checkpoint_callback]  # , early_stopping_callback]

# profiler = PyTorchProfiler(
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./performance_profiling'),
#     sort_by_key="self_cuda_memory_usage",
#     schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, skip_first=10),
#     profile_memory=True,
# )

strategy = DDPStrategy(find_unused_parameters=False)

trainer = Trainer(
    max_epochs=hparams["max_epochs"],
    accelerator="auto",
    devices=n_devices,
    callbacks=callbacks,
    #logger=logger,
    strategy="auto",  # strategy,
    deterministic=True,
    precision=hparams["precision"],
    log_every_n_steps=500,
    # profiler=profiler,
    # fast_dev_run=50,
    # limit_val_batches=0
)

# Train
trainer.fit(model, datamodule=dm)


