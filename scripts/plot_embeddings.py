import os
import json
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # For color mapping
import matplotlib.cm as cm # Import colormaps
from pathlib import Path # Use pathlib for path handling
import re
from math import ceil, sqrt # Needed for grid size calculation

# --- Dimensionality Reduction ---
# Choose one by uncommenting or setting REDUCTION_METHOD below
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
import umap.umap_ as umap
try:
    import pacmap
except ImportError:
    print("Warning: pacmap not installed. PaCMAP reduction method will not be available.")
    pacmap = None # Set to None if not available
# --- End Dimensionality Reduction ---


# Ensure these imports point to the correct model definition file and paths helper
from src.encoder_nano_risk import PretrainNanoEncoder, CausalEncoder
from src.paths import FPATH

# --- Configuration ---
# !!! MUST SET THESE ACCORDING TO YOUR TRAINING RUN !!!
SCALE = False
EXPERIMENT_NAME = "stable_pretrain"
#RUN_NAME = "087_rational_whale"  # From hparams_pretrain2.yaml
RUN_NAME = "8192"
CHECKPOINT_FILENAME = "last.ckpt"    # Or 'last.ckpt'
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Paths derived from your setup
HPARAMS_FILE = FPATH.TB_LOGS / EXPERIMENT_NAME / RUN_NAME /  "hparams.yaml"
# --- Use the specific vocab path you provided ---
# --- Checkpoint directory structure based on your training script ---
CHECKPOINT_DIR_BASE = FPATH.CHECKPOINTS # Adjust if your checkpoint base path differs

# Plotting options
# --- MODIFIED: Add 'pacmap' as an option ---
REDUCTION_METHOD = 'umap' # Change to 'umap', 'pacmap', 'tsne', 'pca' (ensure imports match)
N_COMPONENTS = 2         # Plot in 2D
PLOT_SUBSET = None # 20000       # Max number of points to process with reducer (None for all)
ANNOTATE_POINTS = 0   # <--- INCREASED ANNOTATIONS FOR TESTING SPATIAL SAMPLING
# --- MODIFIED: Ensure output filename updates automatically based on REDUCTION_METHOD ---
OUTPUT_FILENAME = f"figures/embedding_plot_{EXPERIMENT_NAME}_{RUN_NAME}_{REDUCTION_METHOD}_spatial_annot_{ANNOTATE_POINTS}.png"

# --- NEW: Select which token types to plot ---
# Set to None to plot all types, or a list of strings like ["ICD10", "MED", "Labor"]
TOKEN_TYPES_TO_PLOT =["Health_ICD10", "Health_Prescription",
                      "Health_Doctor", "Health General", "Labor",
                      "Demographic",'Social','Education',
                      'Labor_Income'
                      ]
# --- End NEW Configuration ---

# --- End Configuration ---

# --- Helper function to categorize tokens based on prefixes (Updated Order) ---
def get_token_type(token):
    if not isinstance(token, str):
        return "Non-String" # Handle potential non-string items if vocab is corrupted
    if token.startswith("[") and token.endswith("]"): return "Special"
    # Specific Health Categories First
    if token.startswith("HEA_ICD10"): return "Health_ICD10"        # Specific Diagnosis
    if token.startswith("HEA_atc"): return "Health_Prescription"            # Specific Medication (ATC codes)
    if token.startswith("HEA_ydelsesnumme"): return "Health_Doctor"       # Specific Doctor/Specialty
    if token.startswith("HEA_spe2"): return "Health_Doctor"       # Specific Doctor/Specialty
    # Income (more specific LAB type)
    if (token.startswith("LAB_")) and (re.search(r'_Q\d{1,3}', token)): return 'Labor_Income'
    # Other Broad Categories
    if token.startswith("DEM_"): return "Demographic"
    if token.startswith("LAB_"): return "Labor"           # General Labor (excluding Income matched above)
    if token.startswith("EDU_"): return "Education"
    if token.startswith("SOC_"): return "Social"
    # General Health Catch-all (if not matched above)
    if token.startswith("HEA_"): return "Health General"
    # Fallback
    return "Other"
# --- End Helper Function ---

# --- Define a semantic color map with shading for subcategories ---
# (Color map definition remains the same as your previous version)
category_colors = {
    # Health Group (using shades of Blue/Green-Blue)
    "Health_ICD10": cm.Blues(0.9),         # Bright Blue
    "Health_Prescription": cm.Blues(0.7),           # Medium Blue
    "Health_Doctor": cm.Blues(0.5),           # Darker Blue
    "Health General": cm.Blues(0.3), # Greenish-Blue (distinct but related)

    # Labor (using shades of Orange/Brown/Yellow)
    "Labor": cm.Oranges(0.5),         # Medium Orange
    "Labor_Income": cm.Oranges(0.7),        # Lighter Orange (sub-category of Labor)

    #Education
    "Education": cm.Greens(0.7),    

    #Social
    "Social": cm.Reds(0.6),       # Yellow/Gold (

    # Demographic Group (using shades of Purple)
    "Demographic": cm.spring(0.3),    # Medium Purple

    # Special/Other Group (using shades of Grey)
    "Special": cm.Greys(0.8),          # Light Grey
    "Other": cm.Greys(0.5),            # Darker Grey
    "Non-String": "#000000",          # Black for errors/unexpected
}
# --- End Color Map ---


# --- Main Script ---
if __name__ == "__main__":

    # --- Sections 1-6 remain the same (Loading Hparams, Checkpoint, Model, Embeddings, Vocab, Data Prep) ---
    # 1. Load Hyperparameters
    print(f"Loading hyperparameters from: {HPARAMS_FILE}")
    if not HPARAMS_FILE.exists():
        raise FileNotFoundError(f"Hparams file not found at {HPARAMS_FILE}")
    with open(HPARAMS_FILE, "r") as stream:
        hparams = yaml.safe_load(stream)

    try:
        data_dir_path = hparams['data']['dir_path']
        VOCAB_PATH = FPATH.DATA / 'life_test_compiled'  / 'vocab.json'
    except KeyError:
        print("Warning: 'data.dir_path' not found in hparams.yaml. Attempting fallback.")
        dir_path_fallback = hparams.get('dir_path', 'default_data_dir')
        VOCAB_PATH = FPATH.DATA / dir_path_fallback / 'vocab.json'
        print(f"Using fallback vocab path based on '{dir_path_fallback}'")

    hparams_vocab_path_str = hparams.get('data', {}).get('vocab_path')

    if hparams_vocab_path_str:
        VOCAB_PATH_USED = Path(hparams_vocab_path_str)
        print(f"Using specific vocab path from hparams data.vocab_path: {VOCAB_PATH_USED}")
    else:
        VOCAB_PATH_USED = VOCAB_PATH
        print(f"Using vocab path derived from hparams dir_path: {VOCAB_PATH_USED}")

    if not VOCAB_PATH_USED.exists():
         raise FileNotFoundError(f"Vocabulary file not found at {VOCAB_PATH_USED}")

    if 'vocab_size' not in hparams:
         hparams['vocab_size'] = 50000

    # 2. Construct Checkpoint Path
    checkpoint_path = CHECKPOINT_DIR_BASE / EXPERIMENT_NAME / RUN_NAME  / CHECKPOINT_FILENAME
    print(f"Looking for checkpoint at: {checkpoint_path}")
    if not checkpoint_path.exists():
        alt_checkpoint_dir = Path('lightning_logs') / EXPERIMENT_NAME / RUN_NAME / 'checkpoints'
        alt_checkpoint_path = alt_checkpoint_dir / CHECKPOINT_FILENAME
        if alt_checkpoint_path.exists():
             print(f"Checkpoint not found at primary path, using alternative: {alt_checkpoint_path}")
             checkpoint_path = alt_checkpoint_path
        else:
             tb_log_ckpt_path = FPATH.TB_LOGS / EXPERIMENT_NAME / RUN_NAME / 'checkpoints' / CHECKPOINT_FILENAME
             if tb_log_ckpt_path.exists():
                 print(f"Checkpoint not found at primary/alt path, using path relative to TB_LOGS: {tb_log_ckpt_path}")
                 checkpoint_path = tb_log_ckpt_path
             else:
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}, {alt_checkpoint_path}, or {tb_log_ckpt_path}. Please verify paths and structure.")

    # 3. Load the Model
    print("Loading model from checkpoint...")
    try:
        model = CausalEncoder.load_from_checkpoint(
            checkpoint_path, map_location='cpu', strict=False, **hparams
        )
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}\nModel Hyperparameters (from file): {hparams}")
        raise

    # 4. Access Embedding Weights
    try:
        if hasattr(model, 'embedding'): embedding_layer = model.embedding
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'embedding'): embedding_layer = model.encoder.embedding
        else: raise AttributeError("Could not find embedding layer.")
        embedding_weights = embedding_layer.weight.detach().cpu().numpy()
        actual_vocab_size, embedding_dim = embedding_weights.shape
        print(f"Embedding weights shape: {(actual_vocab_size, embedding_dim)}")
    except AttributeError as e:
        raise AttributeError(f"Could not find embedding layer: {e}. Check model definition.")

    # 5. Load Vocabulary Mapping
    print(f"Loading vocabulary from: {VOCAB_PATH_USED}")
    with open(VOCAB_PATH_USED, 'r', encoding='utf-8') as f: vocab = json.load(f)
    idx_to_token = {int(idx): token for token, idx in vocab.items()}
    print(f"Vocabulary size from file: {len(vocab)}")
    if actual_vocab_size != len(vocab):
        print(f"🚨 Warning: Model embedding vocab size ({actual_vocab_size}) != vocab.json size ({len(vocab)}).")
        for i in range(actual_vocab_size):
            if i not in idx_to_token: idx_to_token[i] = f"__MODEL_ONLY_{i}__"

    # 6. Prepare Full Data for Reduction
    if PLOT_SUBSET is None or PLOT_SUBSET >= actual_vocab_size:
        indices_to_reduce = list(range(actual_vocab_size))
        print(f"Processing all {actual_vocab_size} tokens for {REDUCTION_METHOD.upper()}.")
    else:
        indices_to_reduce = np.random.choice(actual_vocab_size, min(PLOT_SUBSET, actual_vocab_size), replace=False).tolist()
        print(f"Processing random subset of {len(indices_to_reduce)} tokens for {REDUCTION_METHOD.upper()}.")
    embeddings_to_reduce = embedding_weights[indices_to_reduce, :]
    token_labels_full_subset = [idx_to_token.get(i, f"__ERR_{i}__") for i in indices_to_reduce]
    token_types_full_subset = [get_token_type(label) for label in token_labels_full_subset]

    # 7. Dimensionality Reduction
    print(f"Performing dimensionality reduction using {REDUCTION_METHOD.upper()}...")
    if REDUCTION_METHOD.lower() == 'umap':
        reducer = umap.UMAP(
            n_components=N_COMPONENTS, n_neighbors=30, min_dist=0.1, metric='cosine', verbose=True
        )
        embeddings_reduced = reducer.fit_transform(embeddings_to_reduce)
    elif REDUCTION_METHOD.lower() == 'pacmap':
        if pacmap is None:
            raise ImportError("PaCMAP method selected, but pacmap library is not installed.")
        embeddings_np = np.ascontiguousarray(embeddings_to_reduce).astype(np.float64)
        reducer = pacmap.PaCMAP(
            n_components=N_COMPONENTS, n_neighbors=20, MN_ratio=0.5, FP_ratio=2.0,num_iters=700, verbose=True
        )
        embeddings_reduced = reducer.fit_transform(embeddings_np, init="pca")
    # Add elif blocks here for 'tsne', 'pca' if needed
    else:
        raise ValueError(f"Unsupported REDUCTION_METHOD: {REDUCTION_METHOD}.")
    print(f"Reduced embeddings shape: {embeddings_reduced.shape}")

    # 8. Filter Data *After* Reduction
    if TOKEN_TYPES_TO_PLOT is not None:
        print(f"Filtering plot to include only: {TOKEN_TYPES_TO_PLOT}")
        filter_mask = [token_type in TOKEN_TYPES_TO_PLOT for token_type in token_types_full_subset]
        if not any(filter_mask):
             print("Warning: Filtering resulted in zero points to plot.")
             exit()
        embeddings_reduced_filtered = embeddings_reduced[filter_mask, :]
        token_labels_filtered = [label for label, keep in zip(token_labels_full_subset, filter_mask) if keep]
        token_types_filtered = [ttype for ttype, keep in zip(token_types_full_subset, filter_mask) if keep]
        print(f"Plotting {len(token_labels_filtered)} points after filtering.")
    else:
        print("Plotting all token types processed by reducer.")
        embeddings_reduced_filtered = embeddings_reduced
        token_labels_filtered = token_labels_full_subset
        token_types_filtered = token_types_full_subset


    # 9. Plotting with Matplotlib
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(15, 15))

    num_points_to_plot = embeddings_reduced_filtered.shape[0]
    if num_points_to_plot == 0:
        print("No points to plot after filtering. Skipping plot generation.")
        # Optionally exit or just save an empty plot
        plt.close(fig) # Close the empty figure
        exit()


    unique_types_to_plot = sorted(list(set(token_types_filtered)))

    # Assign fallback colors if needed (same logic as before)
    fallback_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    color_idx = 0
    assigned_fallback_colors = set(category_colors.values())
    for typ in unique_types_to_plot:
        if typ not in category_colors:
            print(f"Warning: Token type '{typ}' not in predefined category_colors. Assigning fallback color.")
            while fallback_colors[color_idx % len(fallback_colors)] in assigned_fallback_colors: color_idx += 1
            chosen_color = fallback_colors[color_idx % len(fallback_colors)]; category_colors[typ] = chosen_color
            assigned_fallback_colors.add(chosen_color); color_idx += 1

    # Plot points by type (same logic as before)
    for typ in unique_types_to_plot:
        type_indices_in_filtered = [idx for idx, t in enumerate(token_types_filtered) if t == typ]
        if not type_indices_in_filtered: continue
        ax.scatter(
            embeddings_reduced_filtered[type_indices_in_filtered, 0],
            embeddings_reduced_filtered[type_indices_in_filtered, 1],
            color=category_colors[typ], label=typ, alpha=0.7, s=1, edgecolors='none',
        )

    # --- MODIFIED: Spatial Annotation Sampling ---
    indices_to_annotate = [] # Reset list for annotation indices
    if ANNOTATE_POINTS > 0 and num_points_to_plot > 0:
        print(f"Selecting {ANNOTATE_POINTS} points for spatial annotation...")

        # Determine grid boundaries
        x_min, x_max = np.min(embeddings_reduced_filtered[:, 0]), np.max(embeddings_reduced_filtered[:, 0])
        y_min, y_max = np.min(embeddings_reduced_filtered[:, 1]), np.max(embeddings_reduced_filtered[:, 1])

        # Adjust boundaries slightly to avoid edge issues with digitize
        x_min -= 1e-6
        x_max += 1e-6
        y_min -= 1e-6
        y_max += 1e-6

        # Determine grid size (aim for slightly more cells than points)
        # Heuristic: sqrt(N * factor) where factor adjusts density
        grid_size = max(2, int(ceil(sqrt(ANNOTATE_POINTS * 1.5))))
        print(f"Using a {grid_size}x{grid_size} grid.")

        # Define grid bins
        x_bins = np.linspace(x_min, x_max, grid_size + 1)
        y_bins = np.linspace(y_min, y_max, grid_size + 1)

        # Assign points to grid cells
        # Store { (row, col): [list of point indices in this cell] }
        cells = {}

        # Digitize x and y coordinates separately
        x_coords = embeddings_reduced_filtered[:, 0]
        y_coords = embeddings_reduced_filtered[:, 1]

        # Get 1-based bin indices for each dimension
        x_bin_indices = np.digitize(x_coords, bins=x_bins)
        y_bin_indices = np.digitize(y_coords, bins=y_bins)

        # Populate the cells dictionary using 0-based indices
        for i in range(num_points_to_plot):
            # Subtract 1 for 0-based cell index. Clamp to handle potential edge cases.
            col_idx = min(max(0, x_bin_indices[i] - 1), grid_size - 1)
            row_idx = min(max(0, y_bin_indices[i] - 1), grid_size - 1)
            cell_coord = (row_idx, col_idx) # Use (row, col) convention

            if cell_coord not in cells:
                cells[cell_coord] = []
            cells[cell_coord].append(i) # Store the original index within the filtered list
        # Sample one point from distinct cells
        non_empty_cells = list(cells.keys())
        np.random.shuffle(non_empty_cells) # Shuffle cells to pick randomly

        print(f"Found {len(non_empty_cells)} non-empty grid cells.")
        target_annotations = min(ANNOTATE_POINTS, len(non_empty_cells)) # Can't annotate more than available cells

        for i in range(target_annotations):
            cell_coord = non_empty_cells[i]
            # Pick a random point index from the list in this cell
            chosen_point_index = np.random.choice(cells[cell_coord])
            indices_to_annotate.append(chosen_point_index)

        print(f"Selected {len(indices_to_annotate)} points via grid sampling.")

        # Annotate the selected points
        for i in indices_to_annotate:
            ax.text(embeddings_reduced_filtered[i, 0], embeddings_reduced_filtered[i, 1],
                    token_labels_filtered[i][4:14],color = category_colors[get_token_type(token_labels_filtered[i])], fontsize=8, alpha=0.9, zorder=5, # Ensure text is on top
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.5, ec='none')) # Slightly more opaque background
    # --- End MODIFIED Annotation Section ---

    # --- Plot Styling and Saving (Mostly same as before) ---
    if SCALE:
        q = 0.99
        if embeddings_reduced_filtered.shape[0] > 0:
            try: # Add try-except for quantile calculation robustness
                xmin = np.quantile(embeddings_reduced_filtered[:,0],1-q)
                xmax = np.quantile(embeddings_reduced_filtered[:,0],q)
                ymin = np.quantile(embeddings_reduced_filtered[:,1],1-q)
                ymax = np.quantile(embeddings_reduced_filtered[:,1],q)
                if np.isfinite(xmin) and np.isfinite(xmax) and xmin < xmax: ax.set_xlim([xmin,xmax])
                if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax: ax.set_ylim([ymin,ymax])
            except Exception as e:
                print(f"Warning: Could not calculate quantiles for scaling - {e}")
        else:
            print("Warning: Cannot scale axes, no points to plot after filtering.")

    ax.set_aspect('equal')
    ax.set_axis_off()

    legend = ax.legend(title='Token Type', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=16, markerscale=5,frameon=False)
    if legend: legend.get_title().set_fontsize('13')

    fig.tight_layout(rect=[0, 0, 0.85, 1])

    Path(OUTPUT_FILENAME).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(OUTPUT_FILENAME, dpi=500, bbox_inches='tight')
    print(f"Plot saved to {OUTPUT_FILENAME}")
    # plt.show()