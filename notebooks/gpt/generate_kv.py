# scripts/gpt/generate_from_pretrain_run.py
# minimal generation using the pretrain datamodule and the exact logged hparams

import os
import yaml
import torch
import pyarrow.dataset as ds
import polars as pl
from pathlib import PosixPath

from lightning.pytorch import seed_everything

from src.datamodule2 import PretrainLDM
from src.encoder_nano_gpt import GenerativeNanoEncoder
from src.paths import FPATH, check_and_copy_file_or_dir
from src.utils import print_main


# config from the specific run you mentioned
RUN_DIR = FPATH.TB_LOGS / "gpt" / "003_courteous_dragon-noembed-destiny_dataset_SEP"
HPARAMS_PATH = RUN_DIR / "hparams.yaml"
CKPT_PATH = FPATH.CHECKPOINTS_TRANSFORMER / "gpt" / "003_courteous_dragon-noembed-destiny_dataset_SEP" / "best.ckpt"

# small generation knobs
N_PEOPLE = 10
N_SAMPLES_PER_PERSON = 50
START_AFTER_N_TOKENS = 512
MAX_NEW_TOKENS = 2048
TEMPERATURE = 1.0
TOP_P = 0.9
TOP_K = None
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = None
USE_CACHE = False
PREALLOCATE_CACHE = False
MIXED_PRECISION = True
PROFILE = False


def _repeat_batch(batch, repeat: int):
    if repeat <= 1:
        return batch
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.repeat_interleave(repeat, dim=0)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    torch.serialization.add_safe_globals([PosixPath])
    with open(HPARAMS_PATH, "r", encoding="utf-8") as f:
        hparams = yaml.safe_load(f)

    seed_everything(73)
    torch.set_float32_matmul_precision(hparams["float32_matmul_precision"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data paths from hparams
    source_paths = [
        (FPATH.DATA / hparams["source_dir"] / path).with_suffix(".parquet")
        for path in hparams["sources"]
    ]
    background_path = (FPATH.DATA / hparams["source_dir"] / hparams["background"]).with_suffix(".parquet")
    cohort_paths = {
        key: (FPATH.DATA / hparams["source_dir"] / cohort).with_suffix(".parquet")
        for key, cohort in hparams["cohorts"].items()
    }

    for p in source_paths + [background_path] + list(cohort_paths.values()):
        check_and_copy_file_or_dir(p, verbosity=2)

    sources = [ds.dataset(p, format="parquet") for p in source_paths]
    background = pl.read_parquet(background_path)
    cohorts = {k: pl.read_parquet(path, columns=["person_id"]) for k, path in cohort_paths.items()}

    # same datamodule as pretrain
    dm = PretrainLDM(
        dir_path=FPATH.DATA / hparams["dir_path"],
        sources=sources,
        background=background,
        cohorts=cohorts,
        subset_background=hparams["subset_background"],
        n_tokens=hparams["n_tokens"],
        lengths=hparams["lengths"],
        num_workers=hparams["num_workers"],
        max_seq_len=hparams["max_seq_len"],
        source_dir=hparams["source_dir"],
    )
    dm.prepare_data()

    # warmup estimation same as pretrain (optional)
    if "warmup_epochs" in hparams:
        dm.setup()
        sampler = dm.get_sampler(dm.train_dataset)
        avg_batch_size = hparams["n_tokens"] / (sum(sampler.lengths) / len(sampler.lengths))
        n_steps = len(dm.train_dataset) / avg_batch_size
        hparams["warmup_steps"] = int(n_steps * hparams["warmup_epochs"])

    # vocab and ignore_tokens same as pretrain
    hparams["vocab_size"] = len(dm.pipeline.vocab)


    # model with generation class
    model = GenerativeNanoEncoder(**hparams)
    model._mixed_precision = bool(MIXED_PRECISION)
    # load pretrain checkpoint
    state = torch.load(CKPT_PATH, map_location="cpu")
    state = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state, strict=False)

    #so that we dont need autocast
    model.prepare_for_inference(device="cuda", dtype="fp16", warmup=True)
    print(f'model {model}')


    print_main(f"loaded state (missing: {len(missing)}, unexpected: {len(unexpected)})")

    model.eval().to(device)

    # take one batch from train
    if dm.train_dataset is None:
        dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    print(f'batch : {batch}')

    # move to device and keep only a few people
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)

    bsz = batch["event"].size(0)
    take = min(N_PEOPLE, bsz)
    for k, v in batch.items():
        if torch.is_tensor(v) and v.ndim >= 1 and v.size(0) == bsz:
            batch[k] = v[:take]

    # repeat per person for multiple samples

    import time

    # same args used for the baseline
    gen_args = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        pad_token_id=PAD_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        n_samples_per_person=N_SAMPLES_PER_PERSON,
        start_after_n_tokens=START_AFTER_N_TOKENS,
        return_logits=False,
    )

    # baseline (no kv-cache)
    torch.cuda.synchronize(); t0 = time.time()
    out_base = model.generate(batch.copy(), **gen_args)
    torch.cuda.synchronize(); t1 = time.time()

    # kv-cache
    torch.cuda.synchronize(); t2 = time.time()
    out_kv = model.generate_kv(batch.copy(), **gen_args)
    torch.cuda.synchronize(); t3 = time.time()

    print(f"no-cache time: {(t1 - t0)*1000:.1f} ms")
    print(f"kv-cache  time: {(t3 - t2)*1000:.1f} ms")

    # # quick sanity: print one sample from each
    # inv_vocab = {v: k for k, v in dm.pipeline.vocab.items()}
    # def decode(ids, pad_id=0):
    #     toks = []
    #     for x in ids:
    #         i = int(x)
    #         if i == pad_id: continue
    #         toks.append(inv_vocab.get(i, f"<unk:{i}>"))
    #     return toks

    # for i in [1,2,4]:
    #     print("\n--- baseline ---")
    #     print("prompt:", " ".join(decode(out_base["prompt"][i].tolist(), PAD_TOKEN_ID)))
    #     print("\n")
    #     print("generated:", " ".join(decode(out_base["generated"][i].tolist(), PAD_TOKEN_ID)))

    #     print("\n--- kv-cache ---")
    #     print("prompt:", " ".join(decode(out_kv["prompt"][i].tolist(), PAD_TOKEN_ID)))
    #     print("\n")
    #     print("generated:", " ".join(decode(out_kv["generated"][i].tolist(), PAD_TOKEN_ID)))


