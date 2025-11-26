import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import sys

# Add project root for path imports
sys.path.append(".") 
from src.paths import FPATH

# --- CONFIGURATION ---
PRED_DIR = Path("data/prediction_longevity")
BG_PATH = PRED_DIR / "predictions_background.parquet"
FULL_PATH = PRED_DIR / "predictions_full.parquet"

OUTCOME_PATH = FPATH.DATA / "destiny" / "outcomes_longevity" / "TIM_longevity_80.parquet"

FIGURES_DIR = Path("figures/longevity_metrics")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_AGES = [40, 50, 60, 70]

def despine(ax):
    """Helper to remove top and right spines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def load_targets():
    """Loads the ground truth labels from the outcome file."""
    print(f"Loading targets from: {OUTCOME_PATH}")
    if not OUTCOME_PATH.exists():
        raise FileNotFoundError(f"Outcome file not found at {OUTCOME_PATH}")
        
    df = pl.read_parquet(OUTCOME_PATH, columns=["person_id", "outcome"])
    
    df = df.with_columns(
        target = pl.col("outcome").is_not_null().cast(pl.Int8)
    ).select(["person_id", "target"]).unique()
    
    return df

def load_and_process(path, targets_df, model_name):
    """Loads 'wide' prediction files and joins with targets."""
    print(f"\nLoading {model_name} from {path}...")
    if not path.exists():
        print(f"WARNING: File not found: {path}")
        return None

    preds_df = pl.read_parquet(path)
    
    # Join with Targets
    df = preds_df.join(targets_df, on="person_id", how="inner")
    print(f"  - Merged with targets: {len(df)} rows")

    results = {}
    
    # We also store the raw dataframe for the scatter plot later
    results['raw_df'] = df 
    
    for age in TARGET_AGES:
        col_name = f"prob_age_{age}"
        
        if col_name not in df.columns:
            results[age] = None
            continue
            
        subset = df.filter(
            pl.col(col_name).is_not_null() & 
            pl.col(col_name).is_not_nan()
        )
        
        if len(subset) == 0:
            results[age] = None
            continue
            
        preds = subset[col_name].to_numpy()
        targets = subset["target"].to_numpy()
        
        if len(np.unique(targets)) < 2:
            results[age] = None
            continue

        fpr, tpr, _ = roc_curve(targets, preds)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(targets, preds)
        pr_auc = auc(recall, precision)
        
        prevalence = targets.sum() / len(targets)
        
        results[age] = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "prevalence": prevalence,
            "fpr": fpr, "tpr": tpr,
            "precision": precision, "recall": recall,
            "count": len(targets)
        }
        print(f"  - Age {age}: AUROC={roc_auc:.3f}, PRAUC={pr_auc:.3f} (N={len(targets)})")
        
    return results

def plot_bar_comparison(bg_results, full_results):
    """Plots Grouped Bar Charts with Random Baseline."""
    metrics = [("AUROC", "roc_auc"), ("PRAUC", "pr_auc")]
    
    for title, key in metrics:
        ages = [f"Age {a}" for a in TARGET_AGES]
        bg_vals = []
        full_vals = []
        random_vals = []
        
        for age in TARGET_AGES:
            bg_v = bg_results[age][key] if bg_results.get(age) else 0
            full_v = full_results[age][key] if full_results.get(age) else 0
            bg_vals.append(bg_v)
            full_vals.append(full_v)
            
            if key == "roc_auc":
                random_vals.append(0.5)
            elif key == "pr_auc":
                if full_results.get(age):
                    random_vals.append(full_results[age]['prevalence'])
                elif bg_results.get(age):
                    random_vals.append(bg_results[age]['prevalence'])
                else:
                    random_vals.append(0.0)

        x = np.arange(len(ages))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects3 = ax.bar(x - width, random_vals, width, label='Random', color='black', alpha=0.2, hatch='//')
        rects1 = ax.bar(x, bg_vals, width, label='Background', color='#888888')
        rects2 = ax.bar(x + width, full_vals, width, label='Full Data', color='#D55E00')
        
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Age Group')
        ax.set_xticks(x)
        ax.set_xticklabels(ages)
        ax.legend(loc='upper left') # Moved to Upper Left to avoid covering low bars
        ax.set_ylim(0, 1.05)
        
        despine(ax)
        
        for rects in [rects1, rects2, rects3]:
            for rect in rects:
                h = rect.get_height()
                if h > 0:
                    ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
                                
        save_path = FIGURES_DIR / f"barplot_{title}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

def plot_curves_per_age(bg_results, full_results):
    for age in TARGET_AGES:
        bg = bg_results.get(age)
        full = full_results.get(age)
        if not bg and not full: continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # ROC
        if bg: ax1.plot(bg['fpr'], bg['tpr'], label=f'BG (AUC={bg["roc_auc"]:.3f})', color='#888888')
        if full: ax1.plot(full['fpr'], full['tpr'], label=f'Full (AUC={full["roc_auc"]:.3f})', color='#D55E00')
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_title(f'ROC Curve - Age {age}')
        ax1.legend()
        despine(ax1)
        
        # PR
        if bg: ax2.plot(bg['recall'], bg['precision'], label=f'BG (AUC={bg["pr_auc"]:.3f})', color='#888888')
        if full: ax2.plot(full['recall'], full['precision'], label=f'Full (AUC={full["pr_auc"]:.3f})', color='#D55E00')
        
        prev = full['prevalence'] if full else (bg['prevalence'] if bg else 0)
        ax2.axhline(y=prev, color='k', linestyle='--', label=f'Prev={prev:.2f}')
        
        ax2.set_title(f'PR Curve - Age {age}')
        ax2.legend()
        despine(ax2)
        
        save_path = FIGURES_DIR / f"curves_age_{age}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

def plot_calibration_scatter_age60(bg_results, full_results):
    """
    Scatter plot of raw predictions for Age 60.
    X-axis: Background Model Probability
    Y-axis: Full Model Probability
    Color: Target (Red=Death, Green=Survival) - optional, or just density.
    """
    age = 60
    col_name = f"prob_age_{age}"
    
    # Extract raw dataframes
    bg_df = bg_results.get('raw_df')
    full_df = full_results.get('raw_df')
    
    if bg_df is None or full_df is None:
        print("Cannot plot calibration scatter: Raw dataframes missing.")
        return

    # Filter for Age 60 valid predictions
    # We need to join them to get paired predictions for the SAME person
    # Rename columns to avoid clash
    bg_slim = bg_df.select(["person_id", col_name]).rename({col_name: "prob_bg"})
    full_slim = full_df.select(["person_id", col_name, "target"]).rename({col_name: "prob_full"})
    
    merged = bg_slim.join(full_slim, on="person_id", how="inner").drop_nulls()
    
    if len(merged) == 0:
        print("No overlapping predictions for Age 60 scatter plot.")
        return
        
    # Subsample if too large (e.g. > 10k points) to keep file size manageable
    if len(merged) > 20000:
        print(f"Subsampling scatter plot from {len(merged)} to 20,000 points...")
        merged = merged.sample(20000, seed=42)
        
    x = merged["prob_bg"].to_numpy()
    y = merged["prob_full"].to_numpy()
    
    plt.figure(figsize=(8, 8))
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Agreement")
    
    # Scatter
    plt.scatter(x, y, s=1, alpha=0.1, color='purple', label='Predictions')
    
    plt.xlabel(f"Background Model Probability (Age {age})")
    plt.ylabel(f"Full Model Probability (Age {age})")
    plt.title(f"Prediction Correlation Scatter (Age {age})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Optional: Add correlation coefficient
    corr = np.corrcoef(x, y)[0, 1]
    plt.text(0.05, 0.95, f"Pearson r = {corr:.3f}", transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top')
    
    despine(plt.gca())
    
    save_path = FIGURES_DIR / "calibration_scatter_age60.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

def main():
    try:
        targets_df = load_targets()
    except FileNotFoundError as e:
        print(e)
        return

    bg_res = load_and_process(BG_PATH, targets_df, "Background")
    full_res = load_and_process(FULL_PATH, targets_df, "Full Data")
    
    if not bg_res and not full_res:
        print("No data processed. Exiting.")
        return

    print("\nGenerating Plots...")
    plot_bar_comparison(bg_res, full_res)
    plot_curves_per_age(bg_res, full_res)
    
    # New Scatter Plot
    plot_calibration_scatter_age60(bg_res, full_res)
    
    print("\nDone.")

if __name__ == "__main__":
    main()