#!/usr/bin/env python3
"""
Create sequence embeddings colored by demographic variables.
"""

import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import umap
import pandas as pd
from typing import Dict, List, Any, Optional
import argparse
import json
import pickle

# Set up paths
import sys
sys.path.append('.')
from src.datamodule4 import LifeLightningDataModule
from src.encoder_nano_risk import CausalEncoder
from src.paths import FPATH


def load_model_and_data(checkpoint_path: Path, hparams_path: Path, data_dir: Path, device: str = "cuda"):
    """Load model and data for embedding analysis."""
    
    # Load hyperparameters
    with open(hparams_path, "r") as f:
        hparams = yaml.safe_load(f)
    
    print(f"Loading model from: {checkpoint_path}")
    model = CausalEncoder.load_from_checkpoint(
        checkpoint_path,
        map_location=torch.device(device),
        strict=False,
        **hparams
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Set up data module
    print(f"Setting up data module with: {data_dir}")
    
    # Set up paths for pre-compiled data
    lmdb_path = data_dir / "dataset.lmdb"
    vocab_path = data_dir / "vocab.json"
    pnr_to_idx_path = data_dir / "pnr_to_database_idx.json"
    
    datamodule = LifeLightningDataModule(
        dir_path=data_dir,
        lmdb_path=lmdb_path,
        vocab_path=vocab_path,
        pnr_to_idx_path=pnr_to_idx_path,
        background=None,
        cls_token=hparams.get('include_cls', False),
        sep_token=hparams.get('include_sep', True),
        segment=hparams.get('include_segment', True),
        batch_size=32,  # Larger batch for efficiency
        num_workers=4,
        max_seq_len=hparams.get('max_seq_len', 2048),
        cutoff=hparams.get('token_freq_cutoff', 100)
    )
    
    datamodule.setup('predict')
    dataloader = datamodule.predict_dataloader()
    
    # Load vocabulary for token decoding
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    return model, dataloader, datamodule, vocab


def extract_embeddings_with_demographics(model, dataloader, datamodule, vocab, device: str, max_batches: int = 100):
    """Extract embeddings and demographic information."""
    
    # Create reverse vocabulary mapping
    id_to_token = {v: k for k, v in vocab.items()}
    
    embeddings_data = {
        'embeddings': [],
        'sequences': [],
        'demographics': {
            'gender': [],
            'birth_year': [],
            'kommune': [],
            'socio13': [],
            'income_percentile': []
        }
    }
    
    print(f"Extracting embeddings and demographics from {max_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            print(f"Processing batch {batch_idx + 1}/{max_batches}")
            
            # Process batch
            try:
                batch = datamodule.transfer_batch_to_device(batch, device, 0)
                batch = datamodule.on_after_batch_transfer(batch, 0)
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
            
            try:
                # Get attention-weighted embeddings
                attention_embeddings, _ = model.get_sequence_embedding_attention_pool(batch)
                embeddings_data['embeddings'].append(attention_embeddings.cpu())
                
                # Extract sequences for demographic analysis
                sequences = batch['event'].cpu()
                embeddings_data['sequences'].append(sequences)
                
                # Extract demographics for each sequence in the batch
                for seq_idx in range(sequences.shape[0]):
                    seq = sequences[seq_idx]
                    demographics = extract_demographics_from_sequence(seq, id_to_token)
                    
                    embeddings_data['demographics']['gender'].append(demographics['gender'])
                    embeddings_data['demographics']['birth_year'].append(demographics['birth_year'])
                    embeddings_data['demographics']['kommune'].append(demographics['kommune'])
                    embeddings_data['demographics']['socio13'].append(demographics['socio13'])
                    embeddings_data['demographics']['income_percentile'].append(demographics['income_percentile'])
                        
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Concatenate all embeddings
    if embeddings_data['embeddings']:
        embeddings_data['embeddings'] = torch.cat(embeddings_data['embeddings'], dim=0)
        embeddings_data['sequences'] = torch.cat(embeddings_data['sequences'], dim=0)
    
    print(f"Extracted embeddings for {len(embeddings_data['demographics']['gender'])} sequences")
    return embeddings_data


def extract_demographics_from_sequence(sequence, id_to_token):
    """Extract demographic information from a sequence."""
    demographics = {
        'gender': None,
        'birth_year': None,
        'kommune': None,
        'socio13': None,
        'income_percentile': None
    }
    
    # Convert sequence to tokens
    tokens = []
    for token_id in sequence:
        if token_id.item() != 0:  # Skip padding
            token = id_to_token.get(token_id.item(), f"UNK_{token_id.item()}")
            tokens.append(token)
    
    # Extract demographics
    for token in tokens:
        # Gender: DEM_female_X
        if token.startswith('DEM_female_'):
            try:
                gender_code = int(token.split('_')[-1])
                demographics['gender'] = 'female' if gender_code == 1 else 'male'
            except (ValueError, IndexError):
                pass
        
        # Birth year: DEM_birthyear_YYYY
        elif token.startswith('DEM_birthyear_'):
            try:
                year = int(token.split('_')[-1])
                demographics['birth_year'] = year
            except (ValueError, IndexError):
                pass
        
        # Kommune: DEM_kom_XXX
        elif token.startswith('DEM_kom_'):
            try:
                kommune = token.split('_')[-1]
                demographics['kommune'] = kommune
            except IndexError:
                pass
        
        # Socio13: LAB_socio13_XXX (take the last occurrence)
        elif token.startswith('LAB_socio13_'):
            try:
                socio_code = int(token.split('_')[-1])
                demographics['socio13'] = socio_code
            except (ValueError, IndexError):
                pass
        
        # Income percentile: LAB_perindkialt_13_QXX (take the last occurrence)
        elif token.startswith('LAB_perindkialt_13_Q'):
            try:
                percentile = int(token.split('_Q')[-1])
                demographics['income_percentile'] = percentile
            except (ValueError, IndexError):
                pass
    
    return demographics


def save_embeddings_data(embeddings_data: Dict, filepath: Path):
    """Save embeddings data to disk."""
    print(f"Saving embeddings to: {filepath}")
    
    # Convert tensors to numpy for pickling
    save_data = {
        'embeddings': embeddings_data['embeddings'].numpy(),
        'sequences': embeddings_data['sequences'].numpy(),
        'demographics': embeddings_data['demographics']
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"✓ Embeddings saved successfully")


def load_embeddings_data(filepath: Path) -> Dict:
    """Load embeddings data from disk."""
    print(f"Loading embeddings from: {filepath}")
    
    with open(filepath, 'rb') as f:
        save_data = pickle.load(f)
    
    # Convert back to tensors
    embeddings_data = {
        'embeddings': torch.from_numpy(save_data['embeddings']),
        'sequences': torch.from_numpy(save_data['sequences']),
        'demographics': save_data['demographics']
    }
    
    print(f"✓ Embeddings loaded successfully ({len(embeddings_data['demographics']['gender'])} sequences)")
    return embeddings_data


def create_demographic_plots(embeddings_data, output_dir: Path):
    """Create UMAP and t-SNE plots colored by demographics."""
    
    output_dir.mkdir(exist_ok=True)
    embeddings = embeddings_data['embeddings'].numpy()
    
    print("Computing UMAP projection...")
    umap_reducer = umap.UMAP(
        n_neighbors=5, 
        min_dist=0.1, 
        n_components=2, 
        random_state=42,
        metric='cosine'
    )
    embeddings_umap = umap_reducer.fit_transform(embeddings)
    
    print("Computing t-SNE projection...")
    tsne_reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
    embeddings_tsne = tsne_reducer.fit_transform(embeddings)
    
    # Create plots for each demographic variable
    demographics = embeddings_data['demographics']
    
    # 1. Gender plots
    print("Creating gender plots...")
    create_gender_plots(embeddings_umap, embeddings_tsne, demographics['gender'], output_dir)
    
    # 2. Birth year plots
    print("Creating birth year plots...")
    create_birth_year_plots(embeddings_umap, embeddings_tsne, demographics['birth_year'], output_dir)
    
    # 3. Kommune plots
    print("Creating kommune plots...")
    create_kommune_plots(embeddings_umap, embeddings_tsne, demographics['kommune'], output_dir)
    
    # 4. Socio13 plots
    print("Creating socio13 plots...")
    create_socio13_plots(embeddings_umap, embeddings_tsne, demographics['socio13'], output_dir)
    
    # 5. Income percentile plots
    print("Creating income percentile plots...")
    create_income_percentile_plots(embeddings_umap, embeddings_tsne, demographics['income_percentile'], output_dir)


def create_clean_plot(x, y, colors, title_suffix, output_dir, cmap=None, categorical=False):
    """Create clean minimal plots for both UMAP and t-SNE."""
    
    for projection, proj_data in [("umap", (x[0], x[1])), ("tsne", (y[0], y[1]))]:
        plt.figure(figsize=(12, 12), facecolor='white')
        
        if categorical:
            # For categorical data, use discrete colors
            unique_vals = list(set([c for c in colors if c is not None]))
            color_map = plt.cm.get_cmap(cmap if cmap else 'tab10', len(unique_vals))
            
            for i, val in enumerate(unique_vals):
                mask = np.array([c == val for c in colors])
                if np.any(mask):
                    plt.scatter(
                        proj_data[0][mask], 
                        proj_data[1][mask], 
                        c=[color_map(i)], 
                        alpha=0.7,
                        s=5,
                        edgecolors='none',
                        label=str(val)
                    )
            
            # Add legend for categorical data with larger markers
            plt.legend(
                bbox_to_anchor=(1.02, 1), 
                loc='upper left',
                frameon=True,
                fancybox=False,
                shadow=False,
                framealpha=0.9,
                edgecolor='black',
                facecolor='white',
                markerscale=2.0  # Make legend markers larger
            )
        else:
            # For continuous data, use colormap
            # Filter out None values
            valid_mask = np.array([c is not None for c in colors])
            valid_colors = [c for c in colors if c is not None]
            
            if len(valid_colors) > 0:
                scatter = plt.scatter(
                    proj_data[0][valid_mask], 
                    proj_data[1][valid_mask], 
                    c=valid_colors, 
                    cmap=cmap if cmap else 'viridis',
                    alpha=0.7,
                    s=5,
                    edgecolors='none'
                )
                
                # Add colorbar for continuous data
                cbar = plt.colorbar(
                    scatter, 
                    shrink=0.8,
                    aspect=20,
                    pad=0.02
                )
                cbar.outline.set_edgecolor('black')
                cbar.outline.set_linewidth(1)
        
        # Add mini border around the plot
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        
        # Remove ticks but keep spines
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gca().set_aspect('equal')
        
        # Save with legend/colorbar included
        plt.savefig(
            output_dir / f"embeddings_{projection}_{title_suffix}.png", 
            dpi=300, 
            bbox_inches='tight', 
            pad_inches=0.1,
            facecolor='white',
            edgecolor='none'
        )
        plt.close()


def create_gender_plots(embeddings_umap, embeddings_tsne, gender_data, output_dir):
    """Create gender-colored plots (purple for female, orange for male)."""
    
    for projection, proj_data in [("umap", embeddings_umap), ("tsne", embeddings_tsne)]:
        plt.figure(figsize=(12, 12), facecolor='white')
        
        # Plot each gender separately for clean coloring with labels
        gender_colors = [('Female', '#8E44AD'), ('Male', '#E67E22'), ('Unknown', '#95A5A6')]
        gender_map = {'female': 'Female', 'male': 'Male', None: 'Unknown'}
        
        for display_name, color in gender_colors:
            # Map display name back to data values
            gender_key = None
            for k, v in gender_map.items():
                if v == display_name:
                    gender_key = k
                    break
            
            mask = np.array([g == gender_key for g in gender_data])
            if np.any(mask):
                plt.scatter(
                    proj_data[mask, 0], 
                    proj_data[mask, 1], 
                    c=color,
                    alpha=0.7,
                    s=5,
                    edgecolors='none',
                    label=display_name
                )
        
        # Add legend with larger markers
        plt.legend(
            bbox_to_anchor=(1.02, 1), 
            loc='upper left',
            frameon=True,
            fancybox=False,
            shadow=False,
            framealpha=0.9,
            edgecolor='black',
            facecolor='white',
            markerscale=2.0  # Make legend markers larger
        )
        
        # Add mini border around the plot
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        
        # Remove ticks but keep spines
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gca().set_aspect('equal')
        
        plt.savefig(
            output_dir / f"embeddings_{projection}_gender.png", 
            dpi=300, 
            bbox_inches='tight', 
            pad_inches=0.1,
            facecolor='white',
            edgecolor='none'
        )
        plt.close()


def create_birth_year_plots(embeddings_umap, embeddings_tsne, birth_year_data, output_dir):
    """Create birth year colored plots using viridis."""
    create_clean_plot(
        (embeddings_umap[:, 0], embeddings_umap[:, 1]),
        (embeddings_tsne[:, 0], embeddings_tsne[:, 1]),
        birth_year_data,
        "birth_year",
        output_dir,
        cmap='viridis'
    )


def create_kommune_plots(embeddings_umap, embeddings_tsne, kommune_data, output_dir):
    """Create kommune colored plots grouped by first digit as categorical."""
    # Extract first digit from kommune codes and treat as categorical
    def get_kommune_first_digit(kommune_code):
        if kommune_code is None:
            return None
        try:
            # Convert to string and take first character, then back to int
            first_digit = int(str(kommune_code)[0])
            return f"Region {first_digit}"
        except (ValueError, IndexError):
            return "Unknown"
    
    # Group by first digit
    grouped_kommune_data = [get_kommune_first_digit(k) for k in kommune_data]
    
    create_clean_plot(
        (embeddings_umap[:, 0], embeddings_umap[:, 1]),
        (embeddings_tsne[:, 0], embeddings_tsne[:, 1]),
        grouped_kommune_data,
        "kommune",
        output_dir,
        cmap='tab10',
        categorical=True
    )


def create_socio13_plots(embeddings_umap, embeddings_tsne, socio13_data, output_dir):
    """Create socio13 colored plots with meaningful labels."""
    
    # Socio13 code descriptions (original)
    socio13_labels = {
        0: 'Not in AKM',
        110: 'Self-employed',
        111: 'Self-employed, 10+ employees',
        112: 'Self-employed, 5-9 employees', 
        113: 'Self-employed, 1-4 employees',
        114: 'Self-employed, no employees',
        120: 'Employee spouse',
        131: 'Management work',
        132: 'Highest level skills',
        133: 'Intermediate skills',
        134: 'Basic level skills',
        135: 'Other employees',
        139: 'Employee, undisclosed',
        210: 'Unemployed (>6 months)',
        220: 'Sickness/training benefits',
        310: 'In education',
        321: 'Early retirees',
        322: 'State pensioners',
        323: 'Retirement employee',
        330: 'Cash recipient',
        410: 'Others',
        420: 'Children <15 years'
    }
    
    # Create merged categories for cleaner visualization
    def merge_socio13_categories(code):
        if code is None:
            return None
        # Merge all self-employed categories (110, 111, 112, 113, 114) and employee spouse (120)
        if code in [110, 111, 112, 113, 114, 120]:
            return 'Self-employed/Spouse'
        # Keep other categories as individual groups
        elif code == 0:
            return 'Not in AKM'
        elif code == 131:
            return 'Management'
        elif code == 132:
            return 'High skills'
        elif code == 133:
            return 'Intermediate skills'
        elif code == 134:
            return 'Basic skills'
        elif code == 135:
            return 'Other employees'
        elif code == 139:
            return 'Employee (undisclosed)'
        elif code == 210:
            return 'Unemployed'
        elif code == 220:
            return 'Sickness/Training'
        elif code == 310:
            return 'In education'
        elif code == 321:
            return 'Early retirees'
        elif code == 322:
            return 'State pensioners'
        elif code == 323:
            return 'Retirement employee'
        elif code == 330:
            return 'Cash recipient'
        elif code == 410:
            return 'Others'
        elif code == 420:
            return 'Children <15'
        else:
            return f'Unknown ({code})'
    
    # Convert to merged categories
    merged_socio13_data = [merge_socio13_categories(code) for code in socio13_data]
    
    # Create custom socio13 plots with grouped colors
    create_socio13_custom_plots(embeddings_umap, embeddings_tsne, merged_socio13_data, socio13_data, output_dir)
    
    # Create summary of both original and merged socio13 distribution
    socio_counts = {}
    merged_counts = {}
    
    for i, socio in enumerate(socio13_data):
        if socio is not None:
            socio_counts[socio] = socio_counts.get(socio, 0) + 1
            merged_cat = merged_socio13_data[i]
            merged_counts[merged_cat] = merged_counts.get(merged_cat, 0) + 1
    
    # Save socio13 summary
    summary_lines = [
        "Socio13 Distribution Summary:",
        "=" * 40,
        "",
        "MERGED CATEGORIES (used in plots):",
        ""
    ]
    
    for cat, count in sorted(merged_counts.items(), key=lambda x: x[1], reverse=True):
        summary_lines.append(f"{cat}: {count} people")
    
    summary_lines.extend([
        "",
        "ORIGINAL DETAILED CATEGORIES:",
        ""
    ])
    
    for code, count in sorted(socio_counts.items()):
        label = socio13_labels.get(code, f"Unknown code {code}")
        summary_lines.append(f"{code}: {label} - {count} people")
    
    with open(output_dir / "socio13_summary.txt", 'w') as f:
        f.write('\n'.join(summary_lines))


def create_socio13_custom_plots(embeddings_umap, embeddings_tsne, merged_socio13_data, original_socio13_data, output_dir):
    """Create socio13 plots with colors grouped by first two digits of original codes."""
    
    # Define more distinct color groups with variations within each group
    color_schemes = {
        0: ['#2C3E50'],                                    # Dark blue-gray - Not in AKM
        11: ['#E74C3C', '#C0392B', '#A93226'],            # Red family - Self-employed (110-120)
        13: ['#3498DB', '#2980B9', '#1F618D', '#5DADE2', '#85C1E9'],  # Blue family - Employees (131-139)
        21: ['#F39C12', '#E67E22', '#D35400'],            # Orange family - Unemployed/Benefits (210-220)
        31: ['#27AE60', '#229954'],                       # Green family - Education (310)
        32: ['#8E44AD', '#9B59B6', '#7D3C98'],           # Purple family - Retirement (321-323, 330)
        41: ['#95A5A6', '#7F8C8D']                        # Gray family - Others (410-420)
    }
    
    # Create mapping from merged categories to colors and group info
    category_info = {}
    
    # First pass: identify all categories and their groups
    for merged_cat, orig_code in zip(merged_socio13_data, original_socio13_data):
        if merged_cat is not None and merged_cat not in category_info:
            if orig_code == 0:
                group = 0
            elif orig_code and 110 <= orig_code <= 120:
                group = 11
            elif orig_code and 131 <= orig_code <= 139:
                group = 13
            elif orig_code and 210 <= orig_code <= 220:
                group = 21
            elif orig_code == 310:  # Education only
                group = 31
            elif orig_code and orig_code in [321, 322, 323, 330]:  # Retirement/Cash recipients
                group = 32
            elif orig_code and 410 <= orig_code <= 420:
                group = 41
            else:
                group = 99  # Unknown
            
            category_info[merged_cat] = {
                'group': group,
                'orig_code': orig_code
            }
    
    # Sort categories by group and assign colors
    sorted_categories = sorted(category_info.keys(), 
                              key=lambda cat: (category_info[cat]['group'], 
                                             category_info[cat]['orig_code'] or 0))
    
    category_colors = {}
    group_color_indices = {0: 0, 11: 0, 13: 0, 21: 0, 31: 0, 32: 0, 41: 0, 99: 0}
    
    for category in sorted_categories:
        group = category_info[category]['group']
        if group in color_schemes:
            colors = color_schemes[group]
            color_idx = group_color_indices[group] % len(colors)
            category_colors[category] = colors[color_idx]
            group_color_indices[group] += 1
        else:
            category_colors[category] = '#95A5A6'  # Gray for unknown
    
    for projection, proj_data in [("umap", embeddings_umap), ("tsne", embeddings_tsne)]:
        plt.figure(figsize=(14, 12), facecolor='white')  # Slightly wider for legend
        
        # Plot each category in the sorted order for consistent legend
        handles = []
        labels = []
        
        for category in sorted_categories:
            if category in category_colors:
                mask = np.array([cat == category for cat in merged_socio13_data])
                if np.any(mask):
                    scatter = plt.scatter(
                        proj_data[mask, 0], 
                        proj_data[mask, 1], 
                        c=category_colors[category],
                        alpha=0.7,
                        s=5,
                        edgecolors='none',
                        label=category
                    )
                    handles.append(scatter)
                    labels.append(category)
        
        # Add legend with larger markers, ordered by groups
        plt.legend(
            handles, labels,
            bbox_to_anchor=(1.02, 1), 
            loc='upper left',
            frameon=True,
            fancybox=False,
            shadow=False,
            framealpha=0.9,
            edgecolor='black',
            facecolor='white',
            markerscale=2.0  # Make legend markers larger
        )
        
        # Add mini border around the plot
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        
        # Remove ticks but keep spines
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gca().set_aspect('equal')
        
        plt.savefig(
            output_dir / f"embeddings_{projection}_socio13.png", 
            dpi=300, 
            bbox_inches='tight', 
            pad_inches=0.1,
            facecolor='white',
            edgecolor='none'
        )
        plt.close()


def create_income_percentile_plots(embeddings_umap, embeddings_tsne, income_percentile_data, output_dir):
    """Create income percentile colored plots using plasma colormap."""
    create_clean_plot(
        (embeddings_umap[:, 0], embeddings_umap[:, 1]),
        (embeddings_tsne[:, 0], embeddings_tsne[:, 1]),
        income_percentile_data,
        "income_percentile",
        output_dir,
        cmap='plasma'
    )
    
    # Create summary of income percentile distribution
    income_counts = {}
    valid_percentiles = []
    
    for percentile in income_percentile_data:
        if percentile is not None:
            income_counts[percentile] = income_counts.get(percentile, 0) + 1
            valid_percentiles.append(percentile)
    
    # Save income percentile summary
    summary_lines = [
        "Income Percentile Distribution Summary:",
        "=" * 45,
        "",
        f"Total people with income data: {len(valid_percentiles)}",
        f"Total people without income data: {income_percentile_data.count(None)}",
        ""
    ]
    
    if valid_percentiles:
        summary_lines.extend([
            f"Income percentile range: Q{min(valid_percentiles)} - Q{max(valid_percentiles)}",
            f"Mean percentile: Q{np.mean(valid_percentiles):.1f}",
            f"Median percentile: Q{np.median(valid_percentiles):.1f}",
            "",
            "Distribution by percentile:",
            ""
        ])
        
        # Show distribution in bins
        percentile_bins = {
            "Q1-Q20 (Bottom 20%)": [p for p in valid_percentiles if 1 <= p <= 20],
            "Q21-Q40 (Lower-middle 20%)": [p for p in valid_percentiles if 21 <= p <= 40], 
            "Q41-Q60 (Middle 20%)": [p for p in valid_percentiles if 41 <= p <= 60],
            "Q61-Q80 (Upper-middle 20%)": [p for p in valid_percentiles if 61 <= p <= 80],
            "Q81-Q100 (Top 20%)": [p for p in valid_percentiles if 81 <= p <= 100]
        }
        
        for bin_name, bin_values in percentile_bins.items():
            summary_lines.append(f"{bin_name}: {len(bin_values)} people ({len(bin_values)/len(valid_percentiles)*100:.1f}%)")
        
        summary_lines.extend([
            "",
            "Detailed percentile counts:",
            ""
        ])
        
        for percentile, count in sorted(income_counts.items()):
            summary_lines.append(f"Q{percentile}: {count} people")
    
    with open(output_dir / "income_percentile_summary.txt", 'w') as f:
        f.write('\n'.join(summary_lines))


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Create demographic-colored sequence embeddings")
    parser.add_argument("--experiment_name", type=str, default="stable_pretrain")
    parser.add_argument("--experiment_subdir", type=str, default="8192")
    parser.add_argument("--data_dir", type=str, default="life_all_compiled")
    parser.add_argument("--max_batches", type=int, default=750)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_embeddings", action="store_true", help="Save extracted embeddings for future use")
    parser.add_argument("--load_embeddings", action="store_true", help="Load pre-computed embeddings instead of extracting")
    parser.add_argument("--embeddings_dir", type=str, default=None, help="Directory to save/load embeddings")
    
    args = parser.parse_args()
    
    # Set up paths
    hparams_path = FPATH.TB_LOGS / args.experiment_name / args.experiment_subdir / 'hparams.yaml'
    checkpoint_path = FPATH.CHECKPOINTS / args.experiment_name / args.experiment_subdir / 'last.ckpt'
    data_dir = FPATH.DATA / args.data_dir
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = FPATH.GENERATED / "demographic_embeddings"
    
    # Set up embeddings directory (default to same as output)
    if args.embeddings_dir:
        embeddings_dir = Path(args.embeddings_dir)
    else:
        embeddings_dir = FPATH.GENERATED / "embeddings_cache"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DEMOGRAPHIC SEQUENCE EMBEDDING ANALYSIS")
    print("=" * 60)
    print(f"Model: {checkpoint_path}")
    print(f"Data: {data_dir}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"Embeddings cache: {embeddings_dir}")
    print("=" * 60)
    
    # Generate embeddings file name based on experiment and batch count
    embeddings_filename = f"embeddings_{args.experiment_name}_{args.experiment_subdir}_{args.max_batches}batches.pkl"
    embeddings_filepath = embeddings_dir / embeddings_filename
    
    # Load or extract embeddings
    if args.load_embeddings and embeddings_filepath.exists():
        print("Loading pre-computed embeddings...")
        embeddings_data = load_embeddings_data(embeddings_filepath)
    else:
        if args.load_embeddings:
            print(f"Warning: Embeddings file not found at {embeddings_filepath}")
            print("Proceeding with extraction...")
        
        # Load model and data
        model, dataloader, datamodule, vocab = load_model_and_data(
            checkpoint_path, hparams_path, data_dir, args.device
        )
        
        # Extract embeddings with demographics
        embeddings_data = extract_embeddings_with_demographics(
            model, dataloader, datamodule, vocab, args.device, args.max_batches
        )
        
        # Save embeddings if requested
        if args.save_embeddings:
            save_embeddings_data(embeddings_data, embeddings_filepath)
    
    # Create demographic plots
    create_demographic_plots(embeddings_data, output_dir)
    
    # Generate summary report
    gender_counts = {g: embeddings_data['demographics']['gender'].count(g) 
                    for g in ['male', 'female', None]}
    birth_years = [y for y in embeddings_data['demographics']['birth_year'] if y is not None]
    income_percentiles = [p for p in embeddings_data['demographics']['income_percentile'] if p is not None]
    
    report = [
        "DEMOGRAPHIC ANALYSIS SUMMARY",
        "=" * 40,
        "",
        f"Total sequences analyzed: {len(embeddings_data['demographics']['gender'])}",
        f"Embeddings file: {embeddings_filename}",
        "",
        "Gender distribution:",
        f"  Male: {gender_counts['male']}",
        f"  Female: {gender_counts['female']}",
        f"  Unknown: {gender_counts[None]}",
        "",
        "Birth year range:",
        f"  Min: {min(birth_years) if birth_years else 'N/A'}",
        f"  Max: {max(birth_years) if birth_years else 'N/A'}",
        f"  Count: {len(birth_years)}",
        "",
        "Income percentile range:",
        f"  Min: Q{min(income_percentiles) if income_percentiles else 'N/A'}",
        f"  Max: Q{max(income_percentiles) if income_percentiles else 'N/A'}",
        f"  Count: {len(income_percentiles)}",
        "",
        "Generated plots:",
        "  - embeddings_umap_gender.png",
        "  - embeddings_tsne_gender.png", 
        "  - embeddings_umap_birth_year.png",
        "  - embeddings_tsne_birth_year.png",
        "  - embeddings_umap_kommune.png",
        "  - embeddings_tsne_kommune.png",
        "  - embeddings_umap_socio13.png",
        "  - embeddings_tsne_socio13.png",
        "  - embeddings_umap_income_percentile.png",
        "  - embeddings_tsne_income_percentile.png",
        "",
        "Usage examples:",
        "  # Extract and save embeddings:",
        f"  python scripts/demographic_embeddings.py --save_embeddings --max_batches {args.max_batches}",
        "",
        "  # Load saved embeddings and create new plots:",
        f"  python scripts/demographic_embeddings.py --load_embeddings --max_batches {args.max_batches}",
        "",
        "  # Use custom embeddings directory:",
        "  python scripts/demographic_embeddings.py --embeddings_dir /path/to/cache",
        ""
    ]
    
    report_text = '\n'.join(report)
    print(report_text)
    
    with open(output_dir / "demographic_analysis_report.txt", 'w') as f:
        f.write(report_text)
    
    print(f"\nDemographic analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()