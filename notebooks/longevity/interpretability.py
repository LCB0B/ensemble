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
MAPPING_PATH = Path("MASTER_CATEGORY_MAPPINGS.csv") 

ANALYSIS_AGE = 60
TOP_K_TARGET = 1000     # Analyze top 1000 high-gap individuals
N_BACKGROUND = 20000    # Sample size for the matched background group

def load_description_map(csv_path):
    """
    Loads the mapping file and returns a dictionary: { 'code_lower': 'Description' }
    """
    print(f"\n[1/6] Loading mapping file from {csv_path}...")
    if not Path(csv_path).exists():
        print(f"   Warning: Mapping file not found at {csv_path}")
        return {}

    try:
        # Load Code and Description
        df = pl.read_csv(csv_path, columns=["code", "description"])
        
        # Drop nulls
        df = df.drop_nulls(subset=["code"])
        
        # LOWERCASE the code column to ensure matching works
        codes = df["code"].str.to_lowercase().to_list()
        descs = df["description"].to_list()
        
        result = dict(zip(codes, descs))
        print(f"   Success! Loaded {len(result)} descriptions.")
        print(f"   Sample: 'lab_akm_type_akm_independent' -> '{result.get('lab_akm_type_akm_independent', 'N/A')}'")
        return result
    except Exception as e:
        print(f"   Error loading mapping file: {e}")
        return {}

def load_vocab_map(hparams):
    """Reverses the dictionary: {0: '[PAD]', 1: '[CLS]', ...}"""
    print(f"\n[2/6] Loading vocabulary...")
    vocab_path = FPATH.DATA / hparams.dir_path / "vocab.json"
    
    if vocab_path.exists():
        print(f"   Loading full vocab from {vocab_path}...")
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    else:
        print("   WARNING: Full vocab file not found. Using internal snippet/pipeline.")
        pipeline_path = FPATH.DATA / hparams.dir_path / "pipeline.pt"
        pipeline = torch.load(pipeline_path, weights_only=False)
        vocab = pipeline.vocab

    # REVERSE THE DICT
    id2token = {v: k for k, v in vocab.items()}
    print(f"   Vocab size: {len(id2token)} tokens.")
    return id2token

def get_target_ids():
    """Finds the people where Full Model >> Background Model"""
    print(f"\n[3/6] Identifying Target Group (High Gain at Age {ANALYSIS_AGE})...")
    col_name = f"prob_age_{ANALYSIS_AGE}"
    
    if not BG_PREDS.exists() or not FULL_PREDS.exists():
        raise FileNotFoundError("Prediction files missing. Run longevity_prediction.py first.")

    bg = pl.read_parquet(BG_PREDS).select(pl.col("person_id"), pl.col(col_name).alias("p_bg"))
    full = pl.read_parquet(FULL_PREDS).select(pl.col("person_id"), pl.col(col_name).alias("p_full"))
    
    # Merge
    df = bg.join(full, on="person_id", how="inner")
    
    # Calculate Gap & Filter
    df = df.drop_nulls()
    df = df.with_columns((pl.col("p_full") - pl.col("p_bg")).alias("gap"))
    df = df.filter(pl.col("gap") > 0).sort("gap", descending=True)
    
    # --- PRINT STATS ---
    print(f"   Total individuals with positive gap: {len(df)}")
    
    # Create a subset for the top K to avoid indexing errors
    top_k_df = df.head(TOP_K_TARGET)
    
    print(f"   Gap Statistics for Top {TOP_K_TARGET}:")
    print(f"     Max Gain: {top_k_df['gap'].max():.4f}")
    print(f"     Min Gain (in top k): {top_k_df['gap'].min():.4f}")
    print(f"     Mean Gain: {top_k_df['gap'].mean():.4f}")
    
    targets = top_k_df["person_id"].to_list()
    return targets

def get_matched_background_ids(target_ids, hparams, n_samples=2000):
    """
    Samples background individuals who share the same BIRTH YEAR as the target group.
    """
    print("\n[4/6] Creating Matched Background Sample...")
    
    # 1. Locate Background File
    bg_filename = hparams.get("background", "background")
    bg_path = (FPATH.DATA / hparams.source_dir / bg_filename).with_suffix(".parquet")
    
    if not bg_path.exists():
        bg_path = Path('/mnt/dwork/users/xc2/data/destiny/background.parquet')
        
    print(f"   Reading background demographics from: {bg_path}")
    
    # 2. Load Data (Lazy scan)
    lf = pl.scan_parquet(bg_path)
    
    # Check for correct column name
    available_cols = lf.collect_schema().names()
    target_col = "birthyear"
    if target_col not in available_cols:
        print(f"   Columns found: {available_cols}")
        raise ValueError(f"Could not find '{target_col}' in {bg_path}")

    # 3. Load only necessary columns
    lf = lf.select(["person_id", target_col])
    bg_df = lf.collect()
    
    # 4. Identify Target Birth Cohorts
    target_df = bg_df.filter(pl.col("person_id").is_in(target_ids))
    
    if target_df.is_empty():
        print("   WARNING: None of the target IDs were found in the background file. Cannot match cohorts.")
        return []
    
    # --- PRINT COHORT STATS ---
    cohort_counts = target_df[target_col].value_counts().sort(target_col)
    print("   Target Group Birth Year Distribution (Top 5):")
    for row in cohort_counts.head(5).iter_rows():
        print(f"     {row[0]}: {row[1]} people")

    # Get unique birth tokens
    target_cohorts = target_df[target_col].unique().to_list()
    
    # 5. Filter Background
    matched_pool = bg_df.filter(
        (pl.col(target_col).is_in(target_cohorts)) & 
        (~pl.col("person_id").is_in(target_ids))
    )
    
    print(f"   Found {len(matched_pool)} matching peers in the background pool.")
    
    # 6. Sample
    if len(matched_pool) > n_samples:
        sampled_ids = matched_pool["person_id"].sample(n_samples, seed=42).to_list()
    else:
        print("   Warning: Pool smaller than requested N. Taking all available.")
        sampled_ids = matched_pool["person_id"].to_list()
        
    return sampled_ids

def fetch_sequences(dataset, person_ids):
    """Fetches sequences for a list of IDs using the LMDB"""
    sequences = []
    db_indices = [dataset.pnr_to_db_idx(p) for p in person_ids]
    db_indices = [i for i in db_indices if i is not None]
    
    if not db_indices:
        print("   Warning: No sequences found in LMDB for provided IDs.")
        return []

    # Bulk fetch
    with dataset.env.begin() as txn:
        with txn.cursor() as cur:
            keys = [dataset.encode_key(i) for i in db_indices]
            results = cur.getmulti(keys)
            
            for _, val_bytes in results:
                record = dataset.decode(val_bytes)
                sequences.append(record["event"]) 
                
    # --- PRINT SEQ STATS ---
    if sequences:
        lens = [len(s) for s in sequences]
        print(f"   Fetched {len(sequences)} sequences.")
        print(f"   Avg Length: {np.mean(lens):.1f} tokens | Max: {np.max(lens)} | Min: {np.min(lens)}")
        
        # Print sample raw tokens
        print(f"   Sample Raw Tokens (First 5): {sequences[0][:5]}")
        
    return sequences

def run_tfidf_comparison(target_seqs, background_seqs, id2token, desc_map=None):
    print(f"\n[6/6] Running TF-IDF (Target: {len(target_seqs)} vs Matched Background: {len(background_seqs)})")
    
    if not target_seqs or not background_seqs:
        print("   Error: Empty sequence lists. Cannot run TF-IDF.")
        return []

    # 1. Flatten and Decode Helper
    def decode(seq_list, name=""):
        docs = []
        for seq in seq_list:
            flat_seq = []
            for item in seq:
                if isinstance(item, list):
                    flat_seq.extend(item)
                else:
                    flat_seq.append(item)
            
            words = []
            for t in flat_seq:
                try:
                    t_key = int(t)
                    words.append(id2token.get(t_key, str(t_key)))
                except (ValueError, TypeError):
                    words.append(str(t))
            
            docs.append(" ".join(words))
        
        if docs:
            print(f"   Sample Decoded Document ({name}): '{docs[0][:100]}...'")
        return docs
    
    # 2. Convert Sequences to Documents
    target_docs = decode(target_seqs, "Target")
    background_docs = decode(background_seqs, "Background")
    
    # 3. Initialize Vectorizer
    print("   Fitting Vectorizer...")
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b", 
        min_df=5, 
        stop_words=['PAD', 'CLS', 'SEP', 'UNK', 'MASK', 'None', 'null']
    )
    
    # Fit on combined data
    all_docs = background_docs + target_docs
    vectorizer.fit(all_docs) 
    
    # 4. Transform Target Group
    tfidf_matrix = vectorizer.transform(target_docs)
    
    # 5. Aggregate Scores
    sum_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = np.argsort(sum_scores)[::-1]
    
    print(f"\n{'RANK':<5} | {'TOKEN':<40} | {'SCORE':<6} | {'DESCRIPTION'}")
    print("-" * 120)
    
    top_features = []
    for i in range(min(50, len(feature_names))): 
        idx = sorted_indices[i]
        token = feature_names[idx]
        score = sum_scores[idx]
        
        # --- LOOKUP LOGIC ---
        description = ""
        if desc_map:
            description = desc_map.get(token.lower(), "")
            
        # Fallback formatting
        if not description:
            if "birthyear" in token.lower():
                parts = token.split('_')
                if parts[-1].isdigit():
                     description = f"Birth Year {parts[-1]}"
            elif "tim_year" in token.lower():
                parts = token.split('_')
                if parts[-1].isdigit():
                    description = f"Calendar Year {parts[-1]}"

        top_features.append((token, score))
        
        # Truncate desc
        desc_print = (description[:60] + '..') if len(description) > 60 else description
        print(f"{i+1:<5} | {token:<40} | {score:<6.1f} | {desc_print}")

    return top_features

def main():
    if not CONFIG_PATH.exists():
        print(f"[ERROR] Config not found: {CONFIG_PATH}")
        return

    hparams = OmegaConf.load(CONFIG_PATH)
    
    # 1. Load Maps
    id2token = load_vocab_map(hparams)
    desc_map = load_description_map(MAPPING_PATH)
    
    # 2. Initialize Dataset
    lmdb_path = FPATH.DATA / hparams.dir_path / "dataset.lmdb"
    if not lmdb_path.exists():
        print(f"[ERROR] LMDB not found: {lmdb_path}")
        return

    dataset = LMDBDataset(data=None, lmdb_path=lmdb_path)
    dataset._init_db()
    
    # 3. Get Groups
    try:
        target_ids = get_target_ids()
        matched_bg_ids = get_matched_background_ids(target_ids, hparams, n_samples=N_BACKGROUND)
    except Exception as e:
        print(f"Error during group selection: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Fetch Sequences
    print("\n[5/6] Fetching Sequences from LMDB...")
    print("   Fetching Target Sequences...")
    target_seqs = fetch_sequences(dataset, target_ids)
    
    print("   Fetching Matched Background Sequences...")
    background_seqs = fetch_sequences(dataset, matched_bg_ids)
    
    # 5. Compare
    top_results = run_tfidf_comparison(target_seqs, background_seqs, id2token, desc_map)
    
    # 6. Plot
    if top_results:
        tokens, scores = zip(*top_results[:20])
        plt.figure(figsize=(12, 10))
        plt.barh(tokens[::-1], scores[::-1], color='#2ca02c')
        plt.xlabel("Cumulative TF-IDF Score")
        plt.title(f"Events Distinguishing High-Longevity (Age {ANALYSIS_AGE})\n(Matched for Birth Year)")
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        out_file = FIGURES_DIR / "tfidf_distinguishing_features_matched_verbose.png"
        plt.savefig(out_file)
        print(f"\nPlot saved to {out_file}")

if __name__ == "__main__":
    main()