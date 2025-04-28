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


print("Running prepare_data()...")
dm.prepare_data() # Should just print messages

print("Running setup(stage='fit')...")
# Use 'fit' for train_dataloader or 'validate' for val_dataloader
# The error happened during sanity check, which uses validation loader
dm.setup(stage='validate')

print("Getting val_dataloader()...")
# Change to dm.train_dataloader() if you want to debug the training loader
dataloader = dm.val_dataloader()

if dataloader is None:
    print("ERROR: Dataloader creation failed (maybe dataset is empty?).")
else:
    print(f"Dataloader created. Type: {type(dataloader)}")
    print(f"Collate function being used: {dataloader.collate_fn}")
    print("\nAttempting to get the first batch...")
    print("NOTE: This step will execute the collate_fn. If the error is inside")
    print("      the collate_fn, it will likely occur now.")

    try:
        # Get an iterator and fetch the first batch
        data_iter = iter(dataloader)
        first_batch = next(data_iter) # This is where the collate_fn runs

        print("\n--- Successfully obtained the first batch! ---")
        print(f"Batch type: {type(first_batch)}")

        if isinstance(first_batch, dict):
            print("Batch keys and tensor shapes/types:")
            for key, value in first_batch.items():
                if torch.is_tensor(value):
                    print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  - {key}: type={type(value)}")
        else:
            # Handle cases where collate_fn might return something else (e.g., list)
            print(f"Batch content type: {type(first_batch)}")
            print(f"Batch content preview: {str(first_batch)[:200]}...") # Print preview

        # --- Debugging ---
        print("\n ---> You can now inspect 'first_batch' <---")
        print(" ---> Set a breakpoint here using pdb.set_trace() to explore interactively <---")
        # pdb.set_trace() # Uncomment this line to drop into the debugger

    except StopIteration:
        print("\nERROR: Dataloader is empty, cannot get a batch.")
    except (RuntimeError, KeyError, ValueError, IndexError) as e:
        # Catch errors likely originating from collate_fn
        print(f"\n--- ERROR occurred while getting the first batch (likely in collate_fn): ---")
        print(f"Error type: {type(e)}")
        print(f"Error message: {e}")
        print("\nSuggestion: Add `import pdb; pdb.set_trace()` inside your specific")
        print(f"collate function ({dataloader.collate_fn.__class__.__name__} in src/collate_fn2.py)")
        print("just before the line that seems to cause the error, then run this script again.")
        # Re-raising might be useful for full traceback in some environments
        # raise e
    except Exception as e:
        # Catch any other unexpected errors
        print(f"\n--- An UNEXPECTED ERROR occurred: ---")
        print(f"Error type: {type(e)}")
        print(f"Error message: {e}")
        raise e # Re-raise unexpected errors

print("\nDebugging script finished.")