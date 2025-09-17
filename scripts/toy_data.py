import os
import json
import shutil
import lmdb
import random
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Union, Tuple

import polars as pl
import numpy as np
from tqdm import tqdm

import math, hmac, hashlib, getpass
import lzma, msgpack

IntLike = Union[int, np.integer]

import pyarrow as pa
import pyarrow.parquet as pq



class ToyDatasetCreator:
    def __init__(self, 
                 original_data_path: str, 
                 toy_data_path: str,
                 sample_ratio: float = 0.01,
                 max_samples: Optional[int] = None,
                 seed: int = 42):
        """
        Create a toy dataset maintaining the same structure as the original.
        """
        self.original_data_path = Path(original_data_path)
        self.toy_data_path = Path(toy_data_path)
        self.sample_ratio = sample_ratio
        self.max_samples = max_samples
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Paths for original dataset
        self.original_destiny = self.original_data_path / "destiny"
        self.original_destiny_dataset = self.original_data_path / "destiny_dataset"
        
        # Paths for toy dataset  
        self.toy_destiny = self.toy_data_path / "destiny"
        self.toy_destiny_dataset = self.toy_data_path / "destiny_dataset"
        
        # Stats only (we do NOT persist a mapping dict for operation)
        self.token_id_mapping: Dict[int, int] = {}
        self.sampled_indices: List[int] = []

        # LMDB key encoder (set by probing)
        self._key_encoder = None       # type: Optional[callable]
        self._encoder_desc = "unset"
        self._ascii_width = None
        self._valid_idx_set = None

        # Token-ID mapping key (prompted once; None => skip mapping)
        self._mapping_key: Optional[bytes] = None

        # Treat these dict keys as token sequences to scramble.
        # Include "event" because your dataloader uses it as tokens.
        self._TOKEN_FIELDS = {"input_ids", "token_ids", "tokens", "labels", "event"}
        
    def create_toy_dataset(self):
        print("🚀 Creating toy dataset...")
        self._create_directory_structure()
        self._copy_static_files()
        self._sample_consistent_indices()
        self._write_sampled_parquets_all_lazy()


        self._prompt_for_mapping_key_once()

        # Probe the original LMDB to detect actual key encoding (fast, small peek)
        self._probe_key_encoder_from_db()

        self._process_lmdb_dataset()
        self._update_pnr_mapping()
        self._update_lengths()
        self._verify_toy_lmdb()

        print(f"✅ Toy dataset created at: {self.toy_data_path}")
        print(f"📊 Token ID mapping (runtime) touched {len(self.token_id_mapping)} unique tokens")
        print(f"🎯 Used {len(self.sampled_indices)} consistent indices across all files")
        
    def _create_directory_structure(self):
        print("📁 Creating directory structure...")
        self.toy_data_path.mkdir(parents=True, exist_ok=True)
        self.toy_destiny.mkdir(parents=True, exist_ok=True)
        self.toy_destiny_dataset.mkdir(parents=True, exist_ok=True)
        (self.toy_destiny_dataset / "dataset.lmdb").mkdir(parents=True, exist_ok=True)
        
    def _copy_static_files(self):
        """Copy only non-parquet static files."""
        print("📋 Copying static files...")
        for src, dst in [
            (self.original_destiny_dataset / "vocab.json",   self.toy_destiny_dataset / "vocab.json"),
            (self.original_destiny_dataset / "pipeline.pt",  self.toy_destiny_dataset / "pipeline.pt"),
        ]:
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        print("✅ Static files copied")
    

    def _write_sampled_background(self):
        """
        Create a filtered background.parquet containing only rows whose person_id
        appear in the sampled subset. Ensures person_id is Int64 (i64).
        """
        print("🧹 Creating sampled background.parquet...")

        orig_bg = self.original_destiny / "background.parquet"
        out_bg  = self.toy_destiny / "background.parquet"

        # Load full PNR->idx mapping and build idx->PNR
        with open(self._pnr_mapping_path_used, "r", encoding="utf-8") as f:
            pnr_to_idx = json.load(f)
        idx_to_pnr = {v: k for k, v in pnr_to_idx.items()}

        # Get sampled PNRs (as int64)
        sampled_pnrs_i64 = []
        for idx in self.sampled_indices:
            p = idx_to_pnr.get(idx)
            if p is None:
                continue
            try:
                sampled_pnrs_i64.append(int(p))
            except Exception:
                # if a PNR key is not numeric, skip it
                pass

        if not sampled_pnrs_i64:
            print("⚠️  No numeric sampled PNRs found; copying original background as-is.")
            shutil.copy2(orig_bg, out_bg)
            return

        # Use lazy scan to avoid loading full background; enforce Int64 dtype
        lf = pl.scan_parquet(str(orig_bg)).with_columns(
            pl.col("person_id").cast(pl.Int64)
        ).filter(
            pl.col("person_id").is_in(sampled_pnrs_i64)
        )

        df = lf.collect()
        # ensure final dtype is Int64
        if "person_id" in df.columns:
            df = df.with_columns(pl.col("person_id").cast(pl.Int64))

        df.write_parquet(out_bg)
        print(f"✅ Sampled background written: {len(df)} rows → {out_bg}")


    def _load_pnr_mapping(self) -> Dict[str, int]:
        preferred = self.original_destiny_dataset / "pnr_to_database_idx.json"
        fallback = self.original_destiny_dataset / "pnr_to_idx.json"
        path = preferred if preferred.exists() else fallback
        if not path.exists():
            raise FileNotFoundError("Could not find PNR mapping file (pnr_to_database_idx.json or pnr_to_idx.json).")
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        print(f"📄 Loaded PNR mapping from: {path.name} ({len(mapping)} entries)")
        self._pnr_mapping_path_used = path
        return mapping

    def _sample_consistent_indices(self):
        print("🎯 Sampling consistent indices (from PNR mapping)...")
        pnr_map = self._load_pnr_mapping()
        all_indices = list(pnr_map.values())
        total_records = len(all_indices)
        print(f"📊 PNR mapping lists {total_records} database indices")

        if self.max_samples is not None:
            num_samples = min(self.max_samples, int(total_records * self.sample_ratio))
        else:
            num_samples = int(total_records * self.sample_ratio)
        num_samples = max(1, num_samples)
        print(f"🎯 Sampling {num_samples} records ({num_samples/total_records*100:.2f}%)")

        self.sampled_indices = sorted(random.sample(all_indices, num_samples))
        print(f"✅ Sampled {len(self.sampled_indices)} consistent indices")

        self._ascii_width = len(str(max(all_indices))) if all_indices else 1
        self._valid_idx_set = set(all_indices)

    # ---------------------------
    # Key probing (fast, small peek)
    # ---------------------------
    def _try_parse_key(self, key: bytes):
        """Yield (pattern_name, int_value, extra) candidates parsed from key bytes."""
        # pure ASCII digits
        if key and all(48 <= b <= 57 for b in key):
            yield ("ascii", int(key.decode("ascii")), {"width": len(key)})

        # NUL + ASCII digits
        if len(key) > 1 and key[0] == 0x00 and all(48 <= b <= 57 for b in key[1:]):
            yield ("nulled_ascii", int(key[1:].decode("ascii")), {"width": len(key) - 1})

        # 4/8 byte binary, BE/LE
        if len(key) in (4, 8):
            yield (f"be{len(key)}", int.from_bytes(key, "big", signed=False), {})
            yield (f"le{len(key)}", int.from_bytes(key, "little", signed=False), {})

    def _probe_key_encoder_from_db(self, max_probe: int = 1000):
        """Peek at up to max_probe keys and pick the encoding that maps to known indices most often."""
        lmdb_path = self.original_destiny_dataset / "dataset.lmdb"
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
        scores: Dict[str, int] = {}
        widths: Dict[str, Dict[int, int]] = {}

        with env.begin() as txn:
            cursor = txn.cursor()
            seen = 0
            valid = self._valid_idx_set or set()
            for k, _ in cursor:
                for name, val, extra in self._try_parse_key(k):
                    if val in valid:
                        scores[name] = scores.get(name, 0) + 1
                        if name in ("ascii", "nulled_ascii"):
                            w = extra.get("width", 0)
                            widths.setdefault(name, {})
                            widths[name][w] = widths[name].get(w, 0) + 1
                seen += 1
                if seen >= max_probe:
                    break

        env.close()

        if not scores:
            print("⚠️  Key probe inconclusive; will try per-index detection.")
            self._key_encoder = None
            self._encoder_desc = "unknown (per-index detect)"
            return

        best = max(scores.items(), key=lambda kv: kv[1])[0]
        desc = best
        if best == "ascii":
            wc = widths.get("ascii", {})
            if wc:
                width = max(wc.items(), key=lambda kv: kv[1])[0]
                self._key_encoder = lambda i, w=width: (f"{i:0{w}d}" if len(str(i)) < w else str(i)).encode("ascii")
                desc += f" (zero-padded ASCII width≈{width})"
            else:
                self._key_encoder = lambda i: str(i).encode("ascii")
                desc += " (ASCII)"
        elif best == "nulled_ascii":
            wc = widths.get("nulled_ascii", {})
            if wc:
                width = max(wc.items(), key=lambda kv: kv[1])[0]
                self._key_encoder = lambda i, w=width: b"\x00" + (f"{i:0{w}d}" if len(str(i)) < w else str(i)).encode("ascii")
                desc += f" (NUL + zero-padded ASCII width≈{width})"
            else:
                self._key_encoder = lambda i: b"\x00" + str(i).encode("ascii")
                desc += " (NUL + ASCII)"
        elif best == "be8":
            self._key_encoder = lambda i: i.to_bytes(8, "big", signed=False); desc += " (8B BE)"
        elif best == "le8":
            self._key_encoder = lambda i: i.to_bytes(8, "little", signed=False); desc += " (8B LE)"
        elif best == "be4":
            self._key_encoder = lambda i: i.to_bytes(4, "big", signed=False); desc += " (4B BE)"
        elif best == "le4":
            self._key_encoder = lambda i: i.to_bytes(4, "little", signed=False); desc += " (4B LE)"
        else:
            self._key_encoder = None

        self._encoder_desc = desc
        print(f"🔑 Detected LMDB key encoding: {self._encoder_desc}")

    # ---------------------------
    # Per-index fallback encoding tries
    # ---------------------------
    def _encode_key_candidates(self, idx: int) -> List[bytes]:
        c: List[bytes] = []
        # 8/4 byte integers
        try: c.append(idx.to_bytes(8, "big", signed=False))
        except OverflowError: pass
        if idx < (1<<32): c.append(idx.to_bytes(4, "big", signed=False))
        try: c.append(idx.to_bytes(8, "little", signed=False))
        except OverflowError: pass
        if idx < (1<<32): c.append(idx.to_bytes(4, "little", signed=False))
        # ASCII forms
        s = str(idx).encode("ascii")
        c.append(s)
        # try some reasonable zero-pads and nulled-forms
        for w in (len(s), len(s)+1, 8, 10, 12):
            if w > len(s):
                zp = f"{idx:0{w}d}".encode("ascii")
                c.append(zp)
                c.append(b"\x00" + zp)
        c.append(b"\x00" + s)
        # de-dup
        seen, out = set(), []
        for b in c:
            if b not in seen:
                seen.add(b); out.append(b)
        return out

    def _get_record_by_index(self, txn, idx: int) -> Optional[bytes]:
        # Prefer probed global encoder, if any
        if self._key_encoder is not None:
            v = txn.get(self._key_encoder(idx))
            if v is not None:
                return v
        # Fallback: try candidates for this single index
        for k in self._encode_key_candidates(idx):
            v = txn.get(k)
            if v is not None:
                return v
        return None

    # ---------------------------
    # Exact value codec (matches writer)
    # ---------------------------
    @staticmethod
    def _decode_value(raw: bytes):
        # Matches LMDBDataset.decode (lzma + msgpack)
        return msgpack.unpackb(lzma.decompress(raw), raw=False)

    @staticmethod
    def _encode_value(obj) -> bytes:
        # Matches LMDBDataset.encode (msgpack + lzma)
        return lzma.compress(msgpack.packb(obj, use_bin_type=True))

    # -----------------------
    # Token-ID mapping config
    # -----------------------
    def _prompt_for_mapping_key_once(self):
        entered = 'jqd88ac'
        self._mapping_key = entered.encode("utf-8") if entered else None
       

    # -----------------------
    # LMDB processing
    # -----------------------
    def _process_lmdb_dataset(self):
        print("🗄️  Processing LMDB dataset...")

        original_lmdb_path = self.original_destiny_dataset / "dataset.lmdb"
        toy_lmdb_path = self.toy_destiny_dataset / "dataset.lmdb"
        
        original_env = lmdb.open(str(original_lmdb_path), readonly=True, lock=False, readahead=False)
        toy_env = lmdb.open(str(toy_lmdb_path), map_size=1024**3)  # 1GB
        
        not_found = 0
        decode_fail = 0
        encode_fail = 0
        processed = 0

        with original_env.begin() as original_txn, toy_env.begin(write=True) as toy_txn:
            for new_idx, original_idx in enumerate(tqdm(self.sampled_indices, desc="Writing toy LMDB", unit="rec")):
                raw = self._get_record_by_index(original_txn, original_idx)
                if raw is None:
                    not_found += 1
                    continue

                # If no scrambling: copy raw bytes as-is (fast path)
                if self._mapping_key is None:
                    new_key = self._encode_ascii_key(new_idx)
                    toy_txn.put(new_key, raw, overwrite=True)
                    processed += 1
                    continue

                # Else: decode → map tokens → re-encode
                try:
                    data = self._decode_value(raw)
                except Exception:
                    decode_fail += 1
                    continue

                try:
                    mapped_data = self._map_token_ids(
                        data,
                        key=self._mapping_key,
                        vocab_size=None,          # infer from data
                        preserve_ids={0},         # keep PAD=0 (attn_mask = event != 0)
                        rounds=12,
                        tweak=b"toy-dataset"
                    )
                except Exception:
                    mapped_data = data

                try:
                    new_raw = self._encode_value(mapped_data)
                    new_key = self._encode_ascii_key(new_idx)
                    toy_txn.put(new_key, new_raw, overwrite=True)
                    processed += 1
                except Exception:
                    encode_fail += 1
                    continue
        
        original_env.close()
        toy_env.close()
        
        print(f"✅ LMDB dataset processed: {processed} records saved")
        if not_found:
            print(f"⚠️  Skipped {not_found} sampled records not found in LMDB")
        if decode_fail:
            print(f"⚠️  Skipped {decode_fail} records due to decode errors")
        if encode_fail:
            print(f"⚠️  Skipped {encode_fail} records due to encode errors")

    @staticmethod
    def _encode_ascii_key(idx: int) -> bytes:
        # New toy DB keys are plain ASCII digits (matches your reader expectation)
        return str(idx).encode("utf-8")

    # -----------------------
    # Feistel / FPE helpers
    # -----------------------
    @staticmethod
    def _hmac_int(key: bytes, data: bytes) -> int:
        return int.from_bytes(hmac.new(key, data, hashlib.sha256).digest(), "big")

    @staticmethod
    def _feistel_encrypt(x: int, key: bytes, w_bits: int, rounds: int, tweak: bytes) -> int:
        L_bits = w_bits // 2
        R_bits = w_bits - L_bits
        L_mask = (1 << L_bits) - 1
        L = (x >> R_bits) & L_mask
        R = x & ((1 << R_bits) - 1)
        for r in range(rounds):
            R_bytes = R.to_bytes(max(1, (R_bits + 7)//8), "big")
            dom = w_bits.to_bytes(2,"big") + r.to_bytes(2,"big") + tweak
            F = ToyDatasetCreator._hmac_int(key, dom + R_bytes) & L_mask
            L, R = R, (L ^ F) & L_mask
        return ((L << R_bits) | R) & ((1 << w_bits) - 1)

    @staticmethod
    def _feistel_decrypt(x: int, key: bytes, w_bits: int, rounds: int, tweak: bytes) -> int:
        L_bits = w_bits // 2
        R_bits = w_bits - L_bits
        L_mask = (1 << L_bits) - 1
        L = (x >> R_bits) & L_mask
        R = x & ((1 << R_bits) - 1)
        for r in reversed(range(rounds)):
            newR = L
            R_bytes = newR.to_bytes(max(1, (R_bits + 7)//8), "big")
            dom = w_bits.to_bytes(2,"big") + r.to_bytes(2,"big") + tweak
            F = ToyDatasetCreator._hmac_int(key, dom + R_bytes) & L_mask
            newL = (R ^ F) & L_mask
            L, R = newL, newR
        return ((L << R_bits) | R) & ((1 << w_bits) - 1)

    @staticmethod
    def _infer_vocab_size_from_data(data) -> int:
        max_id = -1
        if isinstance(data, (int, np.integer)):
            max_id = int(data)
        elif isinstance(data, (list, tuple)):
            for x in data:
                max_id = max(max_id, ToyDatasetCreator._infer_vocab_size_from_data(x) - 1)
        elif isinstance(data, dict):
            for v in data.values():
                max_id = max(max_id, ToyDatasetCreator._infer_vocab_size_from_data(v) - 1)
        elif isinstance(data, np.ndarray):
            if data.size:
                max_id = max(max_id, int(np.max(data)))
        return max_id + 1 if max_id >= 0 else 0

    # ------------------------------------------------
    # Mapping funcs (NO saved dict; prompts once)
    # ------------------------------------------------
    def _get_mapped_token_id(self,
                             original_id: IntLike,
                             *,
                             key: Optional[Union[str, bytes]] = None,
                             vocab_size: Optional[int] = None,
                             preserve_ids: Optional[Iterable[int]] = None,
                             rounds: int = 12,
                             tweak: Union[str, bytes] = b"") -> int:
        if key is None:
            key = getpass.getpass("Enter mapping key: ")
        if isinstance(key, str):
            key = key.encode("utf-8")
        if isinstance(tweak, str):
            tweak = tweak.encode("utf-8")

        N = int(vocab_size if vocab_size is not None else getattr(self, "vocab_size", None) or 0)
        if N <= 0:
            raise ValueError("vocab_size must be provided or available on self.vocab_size")
        i = int(original_id)
        if not (0 <= i < N):
            raise ValueError(f"Token ID {i} out of range 0..{N-1}")

        preserve = set(preserve_ids or [])
        if i in preserve:
            return i

        w_bits = max(1, math.ceil(math.log2(N)))
        y = i
        while True:
            y = ToyDatasetCreator._feistel_encrypt(y, key, w_bits=w_bits, rounds=rounds, tweak=tweak)
            if (0 <= y < N) and (y not in preserve):
                if i not in self.token_id_mapping:
                    self.token_id_mapping[i] = y  # stats only
                return y

    def _map_token_ids(self,
                       data,
                       *,
                       key: Optional[Union[str, bytes]] = None,
                       vocab_size: Optional[int] = None,
                       preserve_ids: Optional[Iterable[int]] = None,
                       rounds: int = 12,
                       tweak: Union[str, bytes] = b""):
        if vocab_size is None:
            vocab_size = getattr(self, "vocab_size", None)
            if vocab_size is None:
                vocab_size = ToyDatasetCreator._infer_vocab_size_from_data(data)
        if vocab_size <= 0:
            raise ValueError("Could not determine vocab_size; please pass it explicitly.")

        def map_one(x):
            return self._get_mapped_token_id(
                x, key=key, vocab_size=vocab_size,
                preserve_ids=preserve_ids, rounds=rounds, tweak=tweak
            )

        if isinstance(data, (int, np.integer)):
            return map_one(int(data))

        if isinstance(data, (list, tuple)):
            mapped = [self._map_token_ids(elem, key=key, vocab_size=vocab_size,
                                          preserve_ids=preserve_ids, rounds=rounds, tweak=tweak)
                      for elem in data]
            return type(data)(mapped)

        if isinstance(data, np.ndarray):
            if data.dtype.kind in ("i", "u"):
                flat = data.ravel()
                out = np.array([map_one(int(x)) for x in flat], dtype=data.dtype)
                return out.reshape(data.shape)
            else:
                return data

        if isinstance(data, dict):
            mapped = {}
            for k, v in data.items():
                if k in self._TOKEN_FIELDS:
                    mapped[k] = self._map_token_ids(v, key=key, vocab_size=vocab_size,
                                                    preserve_ids=preserve_ids, rounds=rounds, tweak=tweak)
                else:
                    if isinstance(v, (list, tuple, dict, np.ndarray, int, np.integer)):
                        mapped[k] = self._map_token_ids(
                            v, key=key, vocab_size=vocab_size,
                            preserve_ids=preserve_ids, rounds=rounds, tweak=tweak
                        )
                    else:
                        mapped[k] = v
            return mapped

        return data

    # -----------------------
    # PNR and lengths updates
    # -----------------------
    def _update_pnr_mapping(self):
        print("🗂️  Updating PNR mapping...")
        original_pnr_path = self._pnr_mapping_path_used
        with open(original_pnr_path, 'r', encoding="utf-8") as f:
            original_mapping = json.load(f)
        print(f"📊 Original PNR mapping has {len(original_mapping)} entries")

        idx_to_pnr = {v: k for k, v in original_mapping.items()}
        toy_mapping = {}
        missing = 0
        for new_idx, original_idx in enumerate(tqdm(self.sampled_indices, desc="Updating PNR", unit="idx")):
            pnr = idx_to_pnr.get(original_idx)
            if pnr is None:
                missing += 1
                continue
            toy_mapping[pnr] = new_idx
        
        toy_pnr_path = self.toy_destiny_dataset / original_pnr_path.name
        with open(toy_pnr_path, 'w', encoding="utf-8") as f:
            json.dump(toy_mapping, f, indent=2)
        
        print(f"✅ PNR mapping updated: {len(toy_mapping)} entries")
        if missing:
            print(f"⚠️  {missing} sampled indices were not present in original PNR mapping (skipped)")

    def _update_lengths(self):
        print("📏 Updating lengths...")
        original_lengths_path = self.original_destiny_dataset / "lengths.parquet"
        toy_lengths_path = self.toy_destiny_dataset / "lengths.parquet"
        original_lengths = pl.read_parquet(original_lengths_path)
        print(f"📊 Original lengths has {len(original_lengths)} entries")
        
        # The writer created lengths in the same sequential db order (no explicit index column).
        original_with_idx = original_lengths.with_row_index('index')
        toy_lengths = original_with_idx.filter(pl.col('index').is_in(self.sampled_indices))

        # Reorder to match toy db new indices 0..k-1
        order_map = {orig_idx: pos for pos, orig_idx in enumerate(self.sampled_indices)}
        toy_lengths = toy_lengths.with_columns(
            pl.col('index').map_elements(lambda x: order_map.get(x, -1), return_dtype=pl.Int64).alias('_new_index')
        ).filter(pl.col('_new_index') >= 0).sort('_new_index').drop(['_new_index', 'index'])

        # Assign fresh sequential index 0..k-1
        toy_lengths = toy_lengths.with_row_index("index")

        toy_lengths.write_parquet(toy_lengths_path)
        print(f"✅ Lengths updated: {len(toy_lengths)} entries")

    # -----------------------
    # Verification
    # -----------------------
    def _verify_toy_lmdb(self):
        path = self.toy_destiny_dataset / "dataset.lmdb"
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            st = txn.stat()
            print(f"🔍 Toy LMDB stats: {st}")
            c = txn.cursor()
            for i, (k, v) in enumerate(c):
                print(f"  sample key[{i}]={k!r} len={len(k)}")
                try:
                    _ = self._decode_value(v)
                    print("  ✓ decoded one sample")
                except Exception as e:
                    print(f"  ⚠ decode error on sample: {e}")
                if i >= 2:
                    break
        env.close()


    def _collect_sampled_pnrs_i64(self) -> list[int]:
        """Build Int64 person_ids for the sampled indices using pnr_to_database_idx.json."""
        with open(self._pnr_mapping_path_used, "r", encoding="utf-8") as f:
            pnr_to_idx = json.load(f)
        idx_to_pnr = {v: k for k, v in pnr_to_idx.items()}
        out: list[int] = []
        for idx in self.sampled_indices:
            p = idx_to_pnr.get(idx)
            if p is None:
                continue
            try:
                out.append(int(p))
            except Exception:
                pass
        return out


    
    def _filter_parquet_folder_lazy(
        self,
        src_root: Path,
        dst_root: Path,
        sampled_pnrs_i64: list[int],
        chunk_rows: int = 1_000_000,   # tune if needed
    ):
        """
        Filter every Parquet under src_root to dst_root using Polars.
        - Fast path: streaming filter + sink_parquet (no big memory).
        - Fallback: chunked slices with Polars, appending to Parquet via PyArrow.
        Ensures person_id is Int64. Files without person_id are copied as-is.
        """
        dst_root.mkdir(parents=True, exist_ok=True)

        # Make membership list distinct & Int64
        sampled_pnrs_i64 = list({int(x) for x in sampled_pnrs_i64})

        # Enable streaming globally during this phase (best effort)
        try:
            prev_streaming = pl.Config.get_streaming()
            pl.Config.set_streaming(True)
        except Exception:
            prev_streaming = None

        files = sorted(p for p in src_root.rglob("*.parquet") if p.is_file())
        for src in tqdm(files, desc=f"Filtering {src_root.name}/*.parquet", unit="file"):
            rel = src.relative_to(src_root)
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            # read schema *without* materializing
            try:
                schema = pl.scan_parquet(str(src)).collect_schema()
            except Exception:
                shutil.copy2(src, dst)
                continue

            if "person_id" not in schema:
                shutil.copy2(src, dst)
                continue

            # ---- FAST PATH: streaming sink with simple filter (no joins)
            lf = (
                pl.scan_parquet(str(src))
                .with_columns(pl.col("person_id").cast(pl.Int64))
                .filter(pl.col("person_id").is_in(sampled_pnrs_i64))
            )
            try:
                # if supported, this streams rows straight to Parquet and stays low-mem
                lf.sink_parquet(str(dst), compression="zstd")
                continue
            except Exception:
                # fall back to chunked slices below
                pass

            # ---- FALLBACK: Polars chunked slices + append to Parquet
            # compute total rows cheaply from Parquet metadata
            try:
                pf = pq.ParquetFile(str(src))
                total_rows = pf.metadata.num_rows
            except Exception:
                # if metadata fails, copy raw (last resort)
                shutil.copy2(src, dst)
                continue

            # we’ll append row groups with a ParquetWriter once we have a first non-empty chunk
            writer = None
            wrote_any = False

            # iterate over slices of the file
            for offset in range(0, total_rows, chunk_rows):
                lf_chunk = (
                    pl.scan_parquet(str(src))
                    .slice(offset, chunk_rows)
                    .with_columns(pl.col("person_id").cast(pl.Int64))
                    .filter(pl.col("person_id").is_in(sampled_pnrs_i64))
                )

                # collect this slice *streaming*; may be empty
                try:
                    df = lf_chunk.collect(streaming=True)
                except Exception:
                    # if something odd happens, skip this chunk
                    continue

                if df.height == 0:
                    continue

                # Lazily create writer on first non-empty chunk, honoring output schema (person_id i64)
                if writer is None:
                    tbl = df.to_arrow()
                    writer = pq.ParquetWriter(
                        str(dst),
                        tbl.schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                else:
                    tbl = df.to_arrow()

                writer.write_table(tbl)
                wrote_any = True

            if writer is not None:
                writer.close()

            # If no rows matched at all, create an empty file with correct schema
            if not wrote_any:
                # ensure person_id is int64 in schema
                # build minimal schema from input schema with cast for person_id
                fields = []
                for name, dtype in schema.items():
                    if name == "person_id":
                        fields.append(pa.field("person_id", pa.int64()))
                    else:
                        # map Polars dtype to Arrow dtype via an empty series conversion
                        fields.append(pa.field(name, pl.Series([], dtype=dtype).to_arrow().type))
                empty_tbl = pa.table({f.name: pa.array([], type=f.type) for f in fields})
                pq.write_table(empty_tbl, str(dst), compression="zstd", use_dictionary=True)

        # restore prior streaming setting
        if prev_streaming is not None:
            try:
                pl.Config.set_streaming(prev_streaming)
            except Exception:
                pass



    def _write_sampled_parquets_all_lazy(self):
        """
        Filter/copy ALL .parquet files in destiny/ and destiny_dataset/ by sampled person_id via Polars Lazy.
        Ensures person_id is Int64 (i64) on outputs.
        """
        print("🧹 Filtering all Parquet files by sampled person_id (Polars Lazy)…")
        sampled = self._collect_sampled_pnrs_i64()
        if not sampled:
            print("⚠️  No numeric sampled PNRs; copying parquets as-is.")
            for src_root, dst_root in [
                (self.original_destiny, self.toy_destiny),
                (self.original_destiny_dataset, self.toy_destiny_dataset),
            ]:
                for src in src_root.rglob("*.parquet"):
                    rel = src.relative_to(src_root)
                    dst = dst_root / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            return

        self._filter_parquet_folder_lazy(self.original_destiny, self.toy_destiny, sampled)
        self._filter_parquet_folder_lazy(self.original_destiny_dataset, self.toy_destiny_dataset, sampled)
        print("✅ All Parquet files filtered/copied (Lazy).")




def create_toy_dataset_simple(original_data_path: str, 
                              toy_data_path: str,
                              sample_ratio: float = 0.01,
                              max_samples: Optional[int] = None,
                              seed: int = 42):
    creator = ToyDatasetCreator(
        original_data_path=original_data_path,
        toy_data_path=toy_data_path,
        sample_ratio=sample_ratio,
        max_samples=max_samples,
        seed=seed
    )
    
    creator.create_toy_dataset()
    
    stats = {
        'original_path': str(creator.original_data_path),
        'toy_path': str(creator.toy_data_path),
        'sample_ratio': creator.sample_ratio,
        'max_samples': creator.max_samples,
        'seed': creator.seed,
        'unique_tokens_mapped': len(creator.token_id_mapping),
    }
    try:
        original_lmdb = creator.original_destiny_dataset / "dataset.lmdb" / "data.mdb"
        toy_lmdb = creator.toy_destiny_dataset / "dataset.lmdb" / "data.mdb"
        if original_lmdb.exists() and toy_lmdb.exists():
            stats['original_lmdb_size_mb'] = original_lmdb.stat().st_size / (1024**2)
            stats['toy_lmdb_size_mb'] = toy_lmdb.stat().st_size / (1024**2)
            stats['size_reduction_ratio'] = stats['toy_lmdb_size_mb'] / stats['original_lmdb_size_mb']
    except Exception as e:
        print(f"⚠️  Could not get file size statistics: {e}")

    print("\n📈 Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return creator


# Example usage
if __name__ == "__main__":
    creator = create_toy_dataset_simple(
        original_data_path="./data",
        toy_data_path="./toy_data", 
        sample_ratio=0.01,   # 1% of data
        max_samples=100,    # cap
        seed=42
    )
    
    print("\n🎉 Toy dataset creation complete!")
    print(f"📁 Original structure maintained in: ./toy_data")
