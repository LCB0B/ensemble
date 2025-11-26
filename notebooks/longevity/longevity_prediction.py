import sys
import torch
import polars as pl
from pathlib import Path
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from flash_attn.bert_padding import unpad_input, pad_input

# Add project root
sys.path.append(".") 

from src.datamodule2 import PredictFinetuneLifeLDM
from src.encoder_nano_gpt import PredictionFinetuneNanoEncoder
from src.paths import FPATH

# --- CONFIGURATION ---
OUTPUT_DIR = Path("data/prediction_longevity")
PRED_TIMES = [40, 50, 60, 70]  # Ages we want predictions for

RUNS = [
    {
        "name": "Background Model",
        "ckpt_path": "checkpoints/transformer/longevity_finetune/005_nasty_orca-destiny_dataset_att_yb_background-TIM_longevity_80-ckpt_021_majestic_chameleon-noembed-destiny_dataset_att_yb_background/last.ckpt",
        "config_name": "hparams_gpt_finetune_longevity_bg.yaml",
        "output_filename": "predictions_background.parquet"
    },
    {
        "name": "Full Data Model",
        "ckpt_path": "checkpoints/transformer/gpt/018_charismatic_centaur-destiny_dataset_att_yb-TIM_longevity_80-ckpt_008_resilient_mustang-noembed-destiny_dataset_att_yb/last.ckpt",
        "config_name": "hparams_gpt_finetune_longevity.yaml",
        "output_filename": "predictions_full.parquet"
    }
]

# --- PATCHED MODEL ---
class PatchedNanoEncoder(PredictionFinetuneNanoEncoder):
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # 1. PATCH: Ensure attn_mask exists
        if "attn_mask" not in batch:
            if "event" in batch:
                batch["attn_mask"] = (batch["event"] != 0)

        # 2. PATCH: Handle Flash Attention Unpadding
        if "cu_seqlens" not in batch:
            event_unpadded, indices, cu_seqlens, max_seqlen = unpad_input(
                batch["event"].unsqueeze(-1), 
                batch["attn_mask"]
            )[:4]
            
            batch["event"] = event_unpadded.squeeze(-1)
            batch["cu_seqlens"] = cu_seqlens
            batch["max_seqlen_in_batch"] = max_seqlen
            batch["indices"] = indices

        # 3. Forward Pass
        x = self.forward(batch, repad=False)
        
        # 4. Decode to Logits
        logits = self.decoder_finetune(x)
        
        # 5. Repad to (B, T, Windows)
        B = len(batch["cu_seqlens"]) - 1
        T = batch["max_seqlen_in_batch"]
        logits_padded = pad_input(logits, batch["indices"], B, T)
        
        # 6. Extract Probabilities for Specific Ages
        probs = torch.sigmoid(logits_padded) # (B, T, 1)
        age_tensor = batch["age"]            # (B, T)
        
        # Output tensor: (B, Num_Pred_Times)
        batch_preds = torch.full((B, len(PRED_TIMES)), float('nan'), device=self.device)
        
        for i, target_age in enumerate(PRED_TIMES):
            for b in range(B):
                # Find first index where age >= target_age
                mask = age_tensor[b] >= target_age
                if mask.any():
                    idx = torch.where(mask)[0][0]
                    # Grab prediction (Window 0)
                    batch_preds[b, i] = probs[b, idx, 0]
        
        return batch_preds

# --- WRITER CALLBACK ---
class ParquetWriter(BasePredictionWriter):
    def __init__(self, output_dir, output_filename):
        # CRITICAL FIX: Set write_interval to "batch_and_epoch" so that
        # write_on_batch_end is actually called to collect the results.
        super().__init__(write_interval="batch_and_epoch")
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.results = []

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        # prediction is (B, 4)
        preds = prediction.cpu().numpy()
        person_ids = batch["person_id"]
        
        batch_df = pl.DataFrame({
            "person_id": person_ids,
            "prob_age_40": preds[:, 0],
            "prob_age_50": preds[:, 1],
            "prob_age_60": preds[:, 2],
            "prob_age_70": preds[:, 3],
        })
        self.results.append(batch_df)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        if not self.results:
            print("WARNING: No results collected! (Did batch hooks fire?)")
            return

        print(f"Concatenating {len(self.results)} batches...")
        final_df = pl.concat(self.results)
        final_df = final_df.sort("person_id")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / self.output_filename
        
        print(f"Saving predictions to: {out_path}")
        final_df.write_parquet(out_path)
        self.results = []

def run_inference(run_config):
    print(f"\n=== Processing: {run_config['name']} ===")
    
    ckpt_path = Path(run_config["ckpt_path"])
    config_path = FPATH.CONFIGS / "gpt" / run_config["config_name"]
    
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    hparams = OmegaConf.load(config_path)
    
    # Setup DataModule
    source_paths = [(FPATH.DATA / hparams.source_dir / p).with_suffix(".parquet") for p in hparams.sources]
    background_path = (FPATH.DATA / hparams.source_dir / hparams.background).with_suffix(".parquet")
    outcomes_path = (FPATH.DATA / hparams.source_dir / hparams.outcome).with_suffix(".parquet")
    cohort_paths = {k: (FPATH.DATA / hparams.source_dir / c).with_suffix(".parquet") for k, c in hparams.cohorts.items()}
    
    sources = [ds.dataset(s, format="parquet") for s in source_paths]
    background = pl.read_parquet(background_path)
    outcomes = pl.read_parquet(outcomes_path)
    cohorts = {k: pl.read_parquet(p, columns=["person_id"]) for k, p in cohort_paths.items()}

    dm = PredictFinetuneLifeLDM(
        dir_path=FPATH.DATA / hparams.dir_path,
        sources=sources,
        cohorts=cohorts,
        outcomes=outcomes,
        background=background,
        subset_background=hparams.subset_background,
        n_tokens=hparams.n_tokens,
        lengths=hparams.lengths,
        num_workers=0, # Set to 0 to fix fork warnings
        max_seq_len=hparams.max_seq_len,
        prediction_windows=hparams.prediction_windows,
        source_dir=hparams.source_dir,
        sep_token=hparams.sep_token,
    )
    dm.prepare_data()
    dm.setup()
    hparams.vocab_size = len(dm.pipeline.vocab)

    print(f"Loading weights from: {ckpt_path.name}")
    model = PatchedNanoEncoder.load_from_checkpoint(
        ckpt_path,
        strict=False,
        **hparams
    )

    writer_callback = ParquetWriter(
        output_dir=OUTPUT_DIR, 
        output_filename=run_config["output_filename"]
    )
    
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        precision=hparams.precision, 
        logger=False,
        callbacks=[writer_callback],
        inference_mode=True,
    )

    trainer.predict(model, dataloaders=[dm.val_dataloader()])
    print(f"Finished {run_config['name']}")

def main():
    for run_config in RUNS:
        try:
            run_inference(run_config)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed during {run_config['name']}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()