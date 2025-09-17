"""
Lightweight, lazy data access helpers.

Goal: avoid loading full dataset contents; only map person_id -> index (cheap),
then pull just the required LMDB entries.

Assumptions about LMDBDataset (adapt if different):
- Attributes / methods (any subset):
    env              (lmdb.Environment) OR internal open on first access
    observations     (dict with key 'person_id' listing ids in order)
    __len__()
    __getitem__(i)   (fallback decode path)
    _format_key(i) or key_format pattern
    _deserialize / _decode / deserialize(raw) returning dict(sample)
Each sample dict should contain at least:
    {
      "person_id": <str|int>,
      "event": 1D LongTensor (sequence),
      ...
    }

Provided utilities:
- load_dataset(...)
- PersonIndex (build once)
- fetch_by_indices(dataset, indices, field, ...)
- fetch_by_person_ids(dataset, person_ids, ...)
- stream_person_ids(dataset, person_ids, batch_size, ...)
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence, Dict, Any, Union, List, Optional, Callable, Tuple
import torch

try:
    from src.dataset import LMDBDataset  # adjust path if needed
except ImportError:
    LMDBDataset = None


# ---------------- Index (person_id -> dataset row) ----------------
class PersonIndex:
    """
    Immutable mapping wrapper.
    """
    def __init__(self, pid_to_idx: Dict[str, int], ids: Optional[List[str]] = None):
        self.pid_to_idx = pid_to_idx
        self.ids = ids if ids is not None else list(pid_to_idx.keys())

    @classmethod
    def build(cls, dataset, person_key: str = "person_id") -> "PersonIndex":
        mapping: Dict[str, int] = {}
        obs = getattr(dataset, "observations", None)
        if isinstance(obs, dict) and person_key in obs:
            for i, pid in enumerate(obs[person_key]):
                mapping[str(pid)] = i
        else:
            # Brute force (only reads metadata if dataset[i] is lazy)
            for i in range(len(dataset)):
                try:
                    rec = dataset[i]
                except Exception:
                    break
                pid = rec.get(person_key)
                if pid is not None:
                    mapping[str(pid)] = i
        return cls(mapping)

    def ids_to_indices(self, person_ids: Sequence[Union[str, int]]) -> List[int]:
        return [self.pid_to_idx[str(pid)] for pid in person_ids if str(pid) in self.pid_to_idx]

    def has(self, person_id: Union[str, int]) -> bool:
        return str(person_id) in self.pid_to_idx


# ---------------- Dataset loader ----------------
def load_dataset(dataset_dir: Union[str, Path],
                 lmdb_name: str = "dataset.lmdb",
                 **dataset_kwargs):
    """
    Creates LMDBDataset instance (no full materialization).
    """
    if LMDBDataset is None:
        raise ImportError("Adjust import path for LMDBDataset.")
    ds = LMDBDataset(None, Path(dataset_dir) / lmdb_name, **dataset_kwargs)
    if hasattr(ds, "_init_db"):
        ds._init_db()
    return ds


# ---------------- Internal key / decode helpers ----------------
def _format_key(dataset, idx: int) -> bytes:
    if hasattr(dataset, "_format_key"):
        return dataset._format_key(idx)  # type: ignore
    fmt = getattr(dataset, "key_format", "%08d")
    if isinstance(fmt, bytes):
        return fmt % idx
    return (fmt % idx).encode()


def _decode_sample(dataset, raw: bytes):
    for attr in ("_deserialize", "_decode", "deserialize"):
        fn = getattr(dataset, attr, None)
        if callable(fn):
            return fn(raw)
    import pickle
    return pickle.loads(raw)


# ---------------- Core fetch by integer indices ----------------
def fetch_by_indices(dataset,
                     indices: Sequence[int],
                     field: str = "event",
                     pad: bool = False,
                     pad_value: int = 0,
                     max_len: Optional[int] = None,
                     decode_hook: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                     chunk_size: int = 1024
                     ) -> Tuple[List[int], Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:
    """
    Retrieve a tensor field for the specified dataset indices.

    Returns:
      kept_indices,
      list[tensor] OR padded tensor [N,L],
      lengths tensor [N] (if pad=True else None)
    """
    if not indices:
        empty_list: List[torch.Tensor] = []
        if pad:
            return [], torch.empty(0, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        return [], empty_list, None

    env = getattr(dataset, "env", None)
    use_direct = env is not None
    varlen: List[torch.Tensor] = []

    if use_direct:
        # Single read txn (fast). chunk_size kept for future if streaming is needed.
        with env.begin(write=False) as txn:
            for i in indices:
                raw = txn.get(_format_key(dataset, i))
                if raw is None:
                    continue
                sample = _decode_sample(dataset, raw)
                if decode_hook:
                    sample = decode_hook(sample)
                if field not in sample:
                    continue
                t = sample[field]
                if not torch.is_tensor(t):
                    raise TypeError(f"Field {field} not tensor at row {i}")
                varlen.append(t.view(-1))
    else:
        for i in indices:
            sample = dataset[i]
            if decode_hook:
                sample = decode_hook(sample)
            if field not in sample:
                continue
            t = sample[field]
            if not torch.is_tensor(t):
                raise TypeError(f"Field {field} not tensor at row {i}")
            varlen.append(t.view(-1))

    kept_indices = indices[:len(varlen)]

    if not pad:
        return kept_indices, varlen, None

    lengths = torch.tensor([t.numel() for t in varlen], dtype=torch.long)
    L = int(lengths.max().item()) if lengths.numel() else 0
    if max_len is not None:
        L = min(L, max_len)
    padded = torch.full((len(varlen), L), pad_value,
                        dtype=varlen[0].dtype if varlen else torch.long)
    for i, t in enumerate(varlen):
        l = min(t.numel(), L)
        padded[i, :l] = t[:l]
    return kept_indices, padded, lengths


# ---------------- Fetch by person_ids ----------------
def fetch_by_person_ids(dataset,
                        person_ids: Sequence[Union[str, int]],
                        person_index: Optional[PersonIndex] = None,
                        field: str = "event",
                        pad: bool = False,
                        pad_value: int = 0,
                        max_len: Optional[int] = None,
                        **fetch_kwargs):
    """
    Map person_ids -> indices, then fetch.

    Returns:
      kept_person_ids,
      list[tensor] OR padded tensor,
      lengths (if pad)
    """
    if person_index is None:
        person_index = PersonIndex.build(dataset)

    idxs: List[int] = []
    kept_pids: List[str] = []
    for pid in person_ids:
        key = str(pid)
        if key in person_index.pid_to_idx:
            idxs.append(person_index.pid_to_idx[key])
            kept_pids.append(key)

    kept_indices, data, lengths = fetch_by_indices(
        dataset,
        idxs,
        field=field,
        pad=pad,
        pad_value=pad_value,
        max_len=max_len,
        **fetch_kwargs
    )
    # kept_indices align with kept_pids
    return kept_pids, data, lengths


# ---------------- Streaming over a subset ----------------
def stream_person_ids(dataset,
                      person_ids: Sequence[Union[str, int]],
                      batch_size: int = 512,
                      field: str = "event",
                      pad: bool = False,
                      person_index: Optional[PersonIndex] = None,
                      **fetch_kwargs):
    """
    Yield successive batches over the requested subset only.
    """
    if person_index is None:
        person_index = PersonIndex.build(dataset)
    for start in range(0, len(person_ids), batch_size):
        sub = person_ids[start:start + batch_size]
        yield fetch_by_person_ids(
            dataset,
            sub,
            person_index=person_index,
            field=field,
            pad=pad,
            **fetch_kwargs
        )


# ---------------- Example (commented) ----------------
# ds = load_dataset("data/destiny_dataset")
# pindex = PersonIndex.build(ds)
# some_ids = ["person_000001","person_000010","person_000123"]
# kept, varlen, _ = fetch_by_person_ids(ds, some_ids, person_index=pindex, pad=False)
# kept, padded, lengths = fetch_by_person_ids(ds, some_ids, person_index=pindex, pad=True)
# for ids_chunk, seqs_chunk, lens in stream_person_ids(ds, some_ids, batch_size=2, pad=True):
#     pass

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Quick retrieval demo (prints shapes).")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--person_ids", required=True,
                    help="JSON list file or comma-separated person_ids")
    ap.add_argument("--field", default="event")
    ap.add_argument("--pad", action="store_true")
    ap.add_argument("--max_len", type=int, default=None)
    args = ap.parse_args()

    if Path(args.person_ids).is_file():
        person_ids = json.load(open(args.person_ids))
    else:
        person_ids = [p.strip() for p in args.person_ids.split(",") if p.strip()]

    ds = load_dataset(args.dataset_dir)
    pindex = PersonIndex.build(ds)
    kept, data, lengths = fetch_by_person_ids(
        ds,
        person_ids,
        person_index=pindex,
        field=args.field,
        pad=args.pad,
        max_len=args.max_len
    )
    if args.pad:
        print(f"Fetched {len(kept)} persons padded tensor shape={data.shape} lengths_shape={lengths.shape}")
    else:
        # Show up to first 5 sequence lengths
        example_lens = [t.numel() for t in data[:5]]
        print(f"Fetched {len(kept)} persons (variable lengths) example_len={example_lens}")