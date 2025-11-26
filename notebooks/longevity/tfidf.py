import sys
import json
import torch
import random
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from scipy.stats import chi2_contingency, fisher_exact

# Add project root
sys.path.append(".")

from src.paths import FPATH
from src.dataset import LMDBDataset

# --- CONFIGURATION ---
PRED_DIR = Path("data/prediction_longevity")
FIGURES_DIR = Path("figures/longevity")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BG_PREDS = PRED_DIR / "predictions_background.parquet"
FULL_PREDS = PRED_DIR / "predictions_full.parquet"
CONFIG_PATH = FPATH.CONFIGS / "gpt" / "hparams_gpt_finetune_longevity.yaml"

MAPPING_PATH = Path("mappings/MASTER_CATEGORY_MAPPINGS.csv")
if not MAPPING_PATH.exists():
    MAPPING_PATH = Path("MASTER_CATEGORY_MAPPINGS.csv")

METADATA_PATH = Path("/mnt/dwork/users/xc2/data/destiny/background.parquet")

# Analysis Settings
ANALYSIS_AGE = 60
TOP_K_TARGET = 5000     
N_BACKGROUND = 100000    

def load_description_map(csv_path):
    print(f"[1/8] Loading descriptions from {csv_path}...")
    if not Path(csv_path).exists(): return {}
    try:
        df = pl.read_csv(csv_path, columns=["code", "description"]).drop_nulls()
        codes = df["code"].str.to_lowercase().to_list()
        descs = df["description"].to_list()
        return dict(zip(codes, descs))
    except Exception: return {}

def load_vocab_map(hparams):
    print(f"[2/8] Loading vocabulary...")
    vocab_path = FPATH.DATA / hparams.dir_path / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path, 'r') as f: vocab = json.load(f)
    else:
        pipeline_path = FPATH.DATA / hparams.dir_path / "pipeline.pt"
        pipeline = torch.load(pipeline_path, weights_only=False)
        vocab = pipeline.vocab
    return {v: k for k, v in vocab.items()}

def get_valid_control_pool(age):
    print(f"[3/8] Loading valid survivor pool for Age {age}...")
    col = f"prob_age_{age}"
    if not BG_PREDS.exists(): raise FileNotFoundError(f"Prediction file not found: {BG_PREDS}")
    df = pl.read_parquet(BG_PREDS).select(["person_id", col])
    valid_ids = df.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan())["person_id"]
    print(f"   Found {len(valid_ids)} valid survivors.")
    return set(valid_ids.to_list())

def get_target_ids(mode="gap_high"):
    print(f"\n[Selection] Finding Top {TOP_K_TARGET} Individuals (Mode: {mode.upper()})...")
    col = f"prob_age_{ANALYSIS_AGE}"
    bg = pl.read_parquet(BG_PREDS).select(pl.col("person_id"), pl.col(col).alias("p_bg"))
    full = pl.read_parquet(FULL_PREDS).select(pl.col("person_id"), pl.col(col).alias("p_full"))
    
    df = bg.join(full, on="person_id", how="inner")
    
    if "gap" in mode:
        df = df.with_columns((pl.col("p_full") - pl.col("p_bg")).alias("score"))
    else:
        df = df.with_columns(pl.col("p_full").alias("score"))
    
    df = df.filter(pl.col("score").is_not_nan() & pl.col("score").is_not_null())
    
    if "low" in mode:
        df = df.sort("score", descending=False)
    else:
        df = df.sort("score", descending=True)
        
    if len(df) == 0: raise ValueError(f"No valid individuals found for mode {mode}")

    print(f"   Max Score: {df['score'].max():.4f}")
    top_k = df.head(TOP_K_TARGET)
    print(f"   Min Score (Top {len(top_k)}): {top_k['score'].min():.4f}")
    
    return top_k["person_id"].to_list()

def get_matched_controls(target_ids, valid_pool_ids, n_samples=20000, mode=""):
    """
    Stratified Sampling with Replacement + Plotting.
    """
    print(f"[Sampling] Creating Stratified Control Group (N={n_samples})...")
    
    lf = pl.scan_parquet(METADATA_PATH)
    cols = lf.collect_schema().names()
    target_col = "birthyear" if "birthyear" in cols else "event"
    
    bg_df = (
        lf.select(["person_id", target_col])
        .with_columns(pl.col(target_col).str.extract(r"(\d{4})", 1).cast(pl.Int32).alias("year"))
        .collect()
    )
    
    target_people = bg_df.filter(pl.col("person_id").is_in(target_ids))
    target_dist = target_people["year"].value_counts().sort("year")
    total_targets = len(target_people)
    multiplier = n_samples / total_targets
    
    control_ids = []
    
    # Data for Plotting
    plot_data_target = dict(zip(target_dist["year"], target_dist["count"]))
    plot_data_control = {}

    for row in target_dist.iter_rows(named=True):
        year = row["year"]
        count_in_target = row["count"]
        required = int(np.ceil(count_in_target * multiplier))
        
        # Filter pool: Same Year + Not Target + Valid Survivor
        pool = bg_df.filter(
            (pl.col("year") == year) & 
            (~pl.col("person_id").is_in(target_ids)) &
            (pl.col("person_id").is_in(valid_pool_ids))
        )
        
        available = len(pool)
        
        # --- FIX: SAMPLING WITH REPLACEMENT IF SCARCE ---
        if available == 0:
            print(f"   [WARNING] No valid controls found for Year {year}!")
            selected = []
        elif available < required:
            # Resample to fill the quota
            selected = pool["person_id"].sample(required, with_replacement=True, seed=42).to_list()
        else:
            # Standard sample
            selected = pool["person_id"].sample(required, seed=42).to_list()
        
        control_ids.extend(selected)
        plot_data_control[year] = len(selected)

    # --- PLOT AGE MATCHING ---
    years = sorted(plot_data_target.keys())
    target_counts = [plot_data_target[y] for y in years]
    target_props = np.array(target_counts) / sum(target_counts)
    
    control_counts = [plot_data_control.get(y, 0) for y in years]
    control_total = sum(control_counts) if sum(control_counts) > 0 else 1
    control_props = np.array(control_counts) / control_total

    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(years))
    
    plt.bar(x - width/2, target_props, width, label='Target Group', color='#d62728', alpha=0.7)
    plt.bar(x + width/2, control_props, width, label='Control Group', color='#1f77b4', alpha=0.7)
    
    plt.xticks(x[::2], years[::2], rotation=45) 
    plt.xlabel("Birth Year")
    plt.ylabel("Proportion of Group")
    plt.title(f"Birth Year Distribution Matching ({mode.upper()})")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"age_matching_{mode}.png")
    print(f"   Saved matching plot: {FIGURES_DIR / f'age_matching_{mode}.png'}")
    # -------------------------

    random.shuffle(control_ids)
    return control_ids[:n_samples]

def fetch_sequences(dataset, person_ids):
    sequences = []
    db_indices = [dataset.pnr_to_db_idx(p) for p in person_ids]
    db_indices = [i for i in db_indices if i is not None]
    if not db_indices: return []

    truncated_count = 0
    with dataset.env.begin() as txn:
        with txn.cursor() as cur:
            keys = [dataset.encode_key(i) for i in db_indices]
            results = cur.getmulti(keys)
            for _, val_bytes in results:
                record = dataset.decode(val_bytes)
                events = record["event"]
                if "age" in record and len(record["age"]) == len(events):
                    mask = [a < ANALYSIS_AGE for a in record["age"]]
                    events = [e for e, m in zip(events, mask) if m]
                    truncated_count += 1
                sequences.append(events)
    
    print(f"   Fetched {len(sequences)} sequences (Truncated {truncated_count} based on Age < {ANALYSIS_AGE})")
    return sequences

def run_tfidf(target_seqs, control_seqs, id2token, desc_map, suffix):
    print(f"   Running TF-IDF ({suffix})...")
    
    def decode(seq_list):
        docs = []
        for seq in seq_list:
            flat = [item for sublist in seq for item in (sublist if isinstance(sublist, list) else [sublist])]
            words = [id2token.get(int(t), str(t)) for t in flat]
            docs.append(" ".join(words))
        return docs

    target_docs = decode(target_seqs)
    control_docs = decode(control_seqs)
    
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b", 
        min_df=5, 
        stop_words=['PAD', 'CLS', 'SEP', 'UNK', 'MASK', 'None']
    )
    vectorizer.fit(target_docs + control_docs)
    
    tfidf_matrix_target = vectorizer.transform(target_docs)
    tfidf_matrix_control = vectorizer.transform(control_docs)
    
    sum_scores = np.asarray(tfidf_matrix_target.sum(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = np.argsort(sum_scores)[::-1]
    
    # Data Collection for CSV
    csv_data = []
    
    print(f"\n{'RANK':<4} | {'TOKEN':<35} | {'SCORE':<6} | {'P-VALUE':<10} | {'TEST':<8} | {'DESCRIPTION'}")
    print("-" * 130)
    
    for i in range(min(50, len(feature_names))):
        idx = sorted_indices[i]
        token = feature_names[idx]
        score = sum_scores[idx]
        
        count_target = tfidf_matrix_target[:, idx].nnz
        count_control = tfidf_matrix_control[:, idx].nnz
        n_target, n_control = len(target_docs), len(control_docs)
        
        contingency = [[count_target, n_target - count_target], 
                       [count_control, n_control - count_control]]
        
        if min(contingency[0] + contingency[1]) < 5:
            test_name, p_val = "Fisher", fisher_exact(contingency)[1]
        else:
            test_name, p_val = "Chi2", chi2_contingency(contingency)[1]

        p_str = "<0.001" if p_val < 0.001 else f"{p_val:.4f}"
        desc = desc_map.get(token.lower(), "")
        if not desc:
            if "birthyear" in token.lower(): desc = f"Birth Year {token.split('_')[-1]}"
            elif "tim_year" in token.lower(): desc = f"Calendar Year {token.split('_')[-1]}"
        
        # Print
        print(f"{i+1:<4} | {token:<35} | {score:<6.1f} | {p_str:<10} | {test_name:<8} | {desc[:40]}..")
        
        csv_data.append({
            "rank": i+1,
            "token": token,
            "tfidf_score": score,
            "p_value": p_val,
            "test_type": test_name,
            "target_prevalence": count_target / n_target,
            "control_prevalence": count_control / n_control,
            "description": desc
        })

    pl.DataFrame(csv_data).write_csv(FIGURES_DIR / f"tfidf_results_{suffix}.csv")
    print(f"   Saved CSV: {FIGURES_DIR / f'tfidf_results_{suffix}.csv'}")

    tokens, scores = zip(*[(d['token'], d['tfidf_score']) for d in csv_data[:25]])
    plt.figure(figsize=(12, 8))
    plt.barh(tokens[::-1], scores[::-1], color='#2ca02c')
    plt.xlabel("Cumulative TF-IDF Score")
    plt.title(f"Distinguishing Tokens ({suffix.upper()})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"tfidf_{suffix}.png")

def run_intensity(target_seqs, control_seqs, id2token, desc_map, suffix):
    print(f"   Running Intensity Analysis ({suffix})...")

    def get_intensity(sequences):
        if not sequences: return {}
        total_people = len(sequences)
        counts = Counter()
        for seq in sequences:
            flat = [item for sublist in seq for item in (sublist if isinstance(sublist, list) else [sublist])]
            counts.update(flat)
        return {id2token.get(int(k), str(k)): v/total_people for k, v in counts.items()}

    target_int = get_intensity(target_seqs)
    control_int = get_intensity(control_seqs)

    sorted_target = sorted(target_int.items(), key=lambda x: x[1], reverse=True)[:20]
    tokens, target_values = zip(*sorted_target)
    control_values = [control_int.get(t, 0) for t in tokens]

    labels = []
    for t in tokens:
        d = desc_map.get(t.lower(), "")
        if not d:
            if "birthyear" in t.lower(): d = f"Birth Year {t.split('_')[-1]}"
            elif "tim_year" in t.lower(): d = f"Calendar Year {t.split('_')[-1]}"
        labels.append(f"{t} ({d[:20]}..)" if d else t)

    y = np.arange(len(tokens))
    height = 0.35

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(y - height/2, target_values, height, label='Target Group', color='#d62728', alpha=0.8)
    ax.barh(y + height/2, control_values, height, label='Matched Control', color='#1f77b4', alpha=0.8)

    ax.set_xlabel('Mean Occurrences per Person')
    ax.set_title(f"Event Intensity Comparison ({suffix.upper()})")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.legend()
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"intensity_comparison_{suffix}.png")

def run_frequency(target_seqs, control_seqs, id2token, desc_map, suffix):
    print(f"   Running Frequency Analysis ({suffix})...")

    def get_prevalence(sequences):
        if not sequences: return {}
        counts = Counter()
        for seq in sequences:
            flat = [item for sublist in seq for item in (sublist if isinstance(sublist, list) else [sublist])]
            counts.update(set(flat)) # Unique per person
        return {id2token.get(int(k), str(k)): v/len(sequences) for k, v in counts.items()}

    target_freqs = get_prevalence(target_seqs)
    
    sorted_items = sorted(target_freqs.items(), key=lambda x: x[1], reverse=True)[:25]
    tokens, values = zip(*sorted_items)
    
    labels = []
    for t in tokens:
        d = desc_map.get(t.lower(), "")
        if not d:
            if "birthyear" in t.lower(): d = f"Birth Year {t.split('_')[-1]}"
            elif "tim_year" in t.lower(): d = f"Calendar Year {t.split('_')[-1]}"
        labels.append(f"{t} ({d[:20]}..)" if d else t)

    plt.figure(figsize=(12, 10))
    plt.barh(labels[::-1], values[::-1], color='#d62728', alpha=0.8)
    plt.xlabel("Prevalence")
    plt.title(f"Most Common Events ({suffix.upper()})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"frequency_{suffix}.png")

def process_group(dataset, id2token, desc_map, valid_pool, mode):
    try:
        target_ids = get_target_ids(mode=mode)
        control_ids = get_matched_controls(target_ids, valid_pool, n_samples=N_BACKGROUND, mode=mode)
        
        print(f"   Fetching sequences...")
        target_seqs = fetch_sequences(dataset, target_ids)
        control_seqs = fetch_sequences(dataset, control_ids)
        
        run_tfidf(target_seqs, control_seqs, id2token, desc_map, suffix=mode)
        run_frequency(target_seqs, control_seqs, id2token, desc_map, suffix=mode)
        run_intensity(target_seqs, control_seqs, id2token, desc_map, suffix=mode)
        
    except Exception as e:
        print(f"   Error in {mode}: {e}")
        import traceback
        traceback.print_exc()

def main():
    if not CONFIG_PATH.exists(): return
    hparams = OmegaConf.load(CONFIG_PATH)
    id2token = load_vocab_map(hparams)
    desc_map = load_description_map(MAPPING_PATH)
    
    lmdb_path = FPATH.DATA / hparams.dir_path / "dataset.lmdb"
    if not lmdb_path.exists(): return
    
    dataset = LMDBDataset(data=None, lmdb_path=lmdb_path)
    dataset._init_db()
    
    # 1. Pre-load the pool of people valid at ANALYSIS_AGE
    valid_pool = get_valid_control_pool(ANALYSIS_AGE)
    
    # 2. Run Analyses
    #modes = ["prob_high", "prob_low", "gap_high", "gap_low"]
    modes = ["prob_high", "prob_low"]
    for m in modes:
        process_group(dataset, id2token, desc_map, valid_pool, mode=m)

if __name__ == "__main__":
    main()