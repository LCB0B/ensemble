#!/usr/bin/env python
# filepath: /home/louibo/ensemble/scripts/interactive_embeddings.py

import os
import json
import torch
import yaml
import numpy as np
import re
import matplotlib.cm as cm
from pathlib import Path
from typing import List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# For dimensionality reduction
import umap.umap_ as umap
try:
    import pacmap
except ImportError:
    print("Warning: pacmap not installed. PaCMAP reduction method will not be available.")
    pacmap = None

# Import your model classes and path helpers
from src.encoder_nano_risk import PretrainNanoEncoder, CausalEncoder
from src.paths import FPATH

# Define token type helper function
def get_token_type(token):
    """Categorizes tokens based on prefixes"""
    if not isinstance(token, str):
        return "Non-String"
    if token.startswith("[") and token.endswith("]"): return "Special"
    # Health categories
    if token.startswith("HEA_ICD10"): return "Health_ICD10"
    if token.startswith("HEA_atc"): return "Health_Prescription"
    if token.startswith("HEA_ydelsesnumme"): return "Health_Doctor"
    if token.startswith("HEA_spe2"): return "Health_Doctor"
    # Income (more specific LAB type)
    if (token.startswith("LAB_")) and (re.search(r'_Q\d{1,3}', token)): return 'Labor_Income'
    # Other categories
    if token.startswith("DEM_"): return "Demographic"
    if token.startswith("LAB_"): return "Labor"
    if token.startswith("EDU_"): return "Education"
    if token.startswith("SOC_"): return "Social"
    # General Health catch-all
    if token.startswith("HEA_"): return "Health General"
    # Fallback
    return "Other"

# Convert matplotlib colormap colors to hex for Plotly
def cm_to_hex(color_func, value):
    """Convert matplotlib colormap function output to hex color string"""
    rgba = color_func(value)
    # If rgba is a tuple/list of values between 0-1
    if hasattr(rgba, '__len__') and len(rgba) >= 3:
        # Convert RGB values (0-1) to hex string
        if len(rgba) >= 3:
            r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
            return f'#{r:02x}{g:02x}{b:02x}'
    # Fallback
    return '#7f7f7f'  # Default gray

# Define category colors using matplotlib colormap functions to match original
category_colors = {
    # Health Group (using shades of Blues)
    "Health_ICD10": cm_to_hex(cm.Blues, 0.9),       # Bright Blue
    "Health_Prescription": cm_to_hex(cm.Blues, 0.7), # Medium Blue
    "Health_Doctor": cm_to_hex(cm.Blues, 0.5),       # Darker Blue
    "Health General": cm_to_hex(cm.Blues, 0.3),      # Greenish-Blue
    
    # Labor (using shades of Orange)
    "Labor": cm_to_hex(cm.Oranges, 0.5),            # Medium Orange
    "Labor_Income": cm_to_hex(cm.Oranges, 0.7),     # Lighter Orange
    
    # Education
    "Education": cm_to_hex(cm.Greens, 0.7),
    
    # Social
    "Social": cm_to_hex(cm.Reds, 0.6),
    
    # Demographic Group
    "Demographic": cm_to_hex(cm.spring, 0.3),       # Purple
    
    # Special/Other Group
    "Special": cm_to_hex(cm.Greys, 0.8),            # Light Grey
    "Other": cm_to_hex(cm.Greys, 0.5),              # Darker Grey
    "Non-String": "#000000",                        # Black
}

def plot_interactive_embeddings(
    experiment_name: str = "stable_pretrain",
    run_name: str = "8192",
    checkpoint_filename: str = "last.ckpt",
    reduction_method: str = 'umap',
    n_components: int = 2,
    plot_subset: Optional[int] = None,
    token_types_to_plot: Optional[List[str]] = None,
    output_filename: Optional[str] = None,
    random_init: bool = True,  # Use random initialization if checkpoint loading fails
    initial_point_size: int = 4  # Default point size
):
    """
    Creates an interactive HTML plot of token embeddings using Plotly.
    
    Args:
        experiment_name: Name of the experiment folder
        run_name: Name of the run/subfolder
        checkpoint_filename: Filename of checkpoint to load
        reduction_method: Method for dimensionality reduction ('umap', 'pacmap')
        n_components: Number of dimensions to reduce to (2 or 3)
        plot_subset: Maximum number of points to plot (None for all)
        token_types_to_plot: List of token types to include (None for all)
        output_filename: Output HTML filename (auto-generated if None)
        random_init: Whether to use random initialization if checkpoint loading fails
        initial_point_size: Initial size of points in the plot
    """
    # 1. Setup paths and configuration
    hparams_file = FPATH.TB_LOGS / experiment_name / run_name / "hparams.yaml"
    print(f"Loading hyperparameters from: {hparams_file}")
    if not hparams_file.exists():
        raise FileNotFoundError(f"Hparams file not found at {hparams_file}")
        
    with open(hparams_file, "r") as stream:
        hparams = yaml.safe_load(stream)
    
    # Handle directory paths for vocabulary
    try:
        data_dir_path = hparams['data']['dir_path']
        vocab_path = FPATH.DATA / data_dir_path / 'vocab.json'
    except KeyError:
        print("Warning: 'data.dir_path' not found in hparams. Using fallback.")
        dir_path_fallback = hparams.get('dir_path', 'life_test_compiled')
        vocab_path = FPATH.DATA / dir_path_fallback / 'vocab.json'
        print(f"Using fallback vocab path based on '{dir_path_fallback}'")
    
    hparams_vocab_path_str = hparams.get('data', {}).get('vocab_path')
    if hparams_vocab_path_str:
        vocab_path_used = Path(hparams_vocab_path_str)
        print(f"Using vocab path from hparams: {vocab_path_used}")
    else:
        vocab_path_used = vocab_path
        print(f"Using derived vocab path: {vocab_path_used}")
        
    if not vocab_path_used.exists():
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path_used}")
    
    if 'vocab_size' not in hparams:
        hparams['vocab_size'] = 50000
    
    # 2. Model loading
    checkpoint_path = FPATH.CHECKPOINTS / experiment_name / run_name / checkpoint_filename
    print(f"Looking for checkpoint at: {checkpoint_path}")
    
    # Try alternative locations if primary not found
    if not checkpoint_path.exists():
        alt_paths = [
            Path('lightning_logs') / experiment_name / run_name / 'checkpoints' / checkpoint_filename,
            FPATH.TB_LOGS / experiment_name / run_name / 'checkpoints' / checkpoint_filename
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                print(f"Using alternative checkpoint path: {alt_path}")
                checkpoint_path = alt_path
                break
    
    # 3. Load model and extract embeddings
    try:
        print("Loading model from checkpoint...")
        model = CausalEncoder.load_from_checkpoint(
            checkpoint_path, map_location='cpu', strict=False, **hparams
        )
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        if not random_init:
            print(f"Error loading model: {e}")
            raise
        print(f"Error loading model: {e}\nUsing random initialization instead.")
        model = CausalEncoder(**hparams)
        model.eval()
        print("Model initialized with random weights.")
    
    # 4. Extract embedding weights
    try:
        if hasattr(model, 'embedding'):
            embedding_layer = model.embedding
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'embedding'):
            embedding_layer = model.encoder.embedding
        else:
            raise AttributeError("Could not find embedding layer")
            
        embedding_weights = embedding_layer.weight.detach().cpu().numpy()
        actual_vocab_size, embedding_dim = embedding_weights.shape
        print(f"Embedding weights shape: {(actual_vocab_size, embedding_dim)}")
    except AttributeError as e:
        raise AttributeError(f"Could not access embedding weights: {e}")
    
    # 5. Load vocabulary mapping
    print(f"Loading vocabulary from: {vocab_path_used}")
    with open(vocab_path_used, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    idx_to_token = {int(idx): token for token, idx in vocab.items()}
    print(f"Vocabulary size from file: {len(vocab)}")
    
    if actual_vocab_size != len(vocab):
        print(f"Warning: Model embedding size ({actual_vocab_size}) ≠ vocab.json size ({len(vocab)})")
        for i in range(actual_vocab_size):
            if i not in idx_to_token:
                idx_to_token[i] = f"__MODEL_ONLY_{i}__"
    
    # 6. Prepare data for dimensionality reduction
    if plot_subset is None or plot_subset >= actual_vocab_size:
        indices_to_reduce = list(range(actual_vocab_size))
        print(f"Processing all {actual_vocab_size} tokens for {reduction_method.upper()}")
    else:
        indices_to_reduce = np.random.choice(actual_vocab_size, min(plot_subset, actual_vocab_size), replace=False).tolist()
        print(f"Processing random subset of {len(indices_to_reduce)} tokens")
    
    embeddings_to_reduce = embedding_weights[indices_to_reduce, :]
    token_labels = [idx_to_token.get(i, f"__ERR_{i}__") for i in indices_to_reduce]
    token_types = [get_token_type(label) for label in token_labels]
    
    # 7. Dimensionality reduction
    print(f"Performing {reduction_method.upper()} dimensionality reduction...")
    if reduction_method.lower() == 'umap':
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=30,
            min_dist=0.1,
            metric='cosine',
            verbose=True
        )
        embeddings_reduced = reducer.fit_transform(embeddings_to_reduce)
    elif reduction_method.lower() == 'pacmap':
        if pacmap is None:
            raise ImportError("PaCMAP method selected but pacmap is not installed")
        embeddings_np = np.ascontiguousarray(embeddings_to_reduce).astype(np.float64)
        reducer = pacmap.PaCMAP(
            n_components=n_components,
            n_neighbors=20,
            MN_ratio=0.5,
            FP_ratio=2.0,
            num_iters=700,
            verbose=True
        )
        embeddings_reduced = reducer.fit_transform(embeddings_np, init="pca")
    else:
        raise ValueError(f"Unsupported reduction method: {reduction_method}")
    
    print(f"Reduced embeddings shape: {embeddings_reduced.shape}")
    
    # 8. Filter by token type if requested
    if token_types_to_plot is not None:
        print(f"Filtering to include only: {token_types_to_plot}")
        filter_mask = [token_type in token_types_to_plot for token_type in token_types]
        if not any(filter_mask):
            print("Warning: Filtering resulted in zero points to plot.")
            return
            
        embeddings_reduced = embeddings_reduced[filter_mask]
        token_labels = [label for label, keep in zip(token_labels, filter_mask) if keep]
        token_types = [ttype for ttype, keep in zip(token_types, filter_mask) if keep]
        print(f"Plotting {len(token_labels)} points after filtering.")
    
    # 9. Create Plotly figure
    print("Creating interactive plot...")
    
    # Prepare hover text with more detail
    hover_texts = [f"<b>Token:</b> {label}<br><b>Type:</b> {typ}" for label, typ in zip(token_labels, token_types)]
    
    # Create a main plot area for scatter plot
    fig = make_subplots(rows=1, cols=1)
    
    # Add traces for each token type
    unique_types = sorted(set(token_types))
    for typ in unique_types:
        # Get indices for this type
        indices = [i for i, t in enumerate(token_types) if t == typ]
        if not indices:
            continue
            
        # Create scatter trace for this type
        fig.add_trace(
            go.Scattergl(
                x=embeddings_reduced[indices, 0],
                y=embeddings_reduced[indices, 1],
                mode='markers',
                marker=dict(
                    size=initial_point_size,
                    color=category_colors.get(typ, "#7f7f7f"),
                    opacity=0.7,
                    line=dict(width=0)
                ),
                text=[hover_texts[i] for i in indices],
                hoverinfo='text',
                name=typ,
                legendgroup=typ
            )
        )
    
    # Update layout with equal aspect ratio and other settings
    fig.update_layout(
        title=f"Token Embeddings - {experiment_name}/{run_name}",
        legend_title="Token Type",
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            itemsizing='constant',  # Make legend markers constant size
            itemwidth=30,          # Wider legend items
            font=dict(size=14),    # Larger legend text
            tracegroupgap=5        # Space between legend groups
        ),
        margin=dict(l=20, r=20, t=60, b=60),
        hovermode="closest",
        template="plotly_white",
        # Equal aspect ratio
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        ),
        # Point size control with HTML - positioned at the bottom
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(
                        label="Reset View",
                        method="relayout",
                        args=[{"xaxis.autorange": True, "yaxis.autorange": True}]
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.11,
                xanchor="left",
                y=-0.15,  # Position below the plot
                yanchor="top"
            )
        ]
    )
    
    # Store trace mappings for index lookup
    # Maps each trace's points back to original indices
    trace_to_indices = {}
    trace_idx = 0
    for typ in unique_types:
        indices = [i for i, t in enumerate(token_types) if t == typ]
        if indices:  # Only store if we have points
            trace_to_indices[trace_idx] = indices
            trace_idx += 1
    
    # Add custom JavaScript for interactive point size control and nearest neighbors
    js_code = """
    <script>
    // Add event listeners after Plotly is loaded
    window.onload = function() {
        // Create size control slider
        var sizeSlider = document.createElement('input');
        sizeSlider.type = 'range';
        sizeSlider.min = '0.5';
        sizeSlider.max = '20';
        sizeSlider.value = '""" + str(initial_point_size) + """';
        sizeSlider.step = '0.5';
        sizeSlider.style = 'width: 200px; margin: 0 10px;';
        
        // Create label for slider
        var sizeLabel = document.createElement('label');
        sizeLabel.innerHTML = 'Point Size: ';
        sizeLabel.style = 'font-family: Arial; margin-right: 5px;';
        
        // Create value display
        var sizeValue = document.createElement('span');
        sizeValue.innerHTML = sizeSlider.value;
        sizeValue.style = 'font-family: Arial; margin-left: 5px; min-width: 30px; display: inline-block;';
        
        // Create container div - positioned below the legend
        var container = document.createElement('div');
        container.style = 'position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); z-index: 1000; background: white; padding: 5px; border-radius: 3px; box-shadow: 0 0 5px rgba(0,0,0,0.2);';
        
        // Add elements to container
        container.appendChild(sizeLabel);
        container.appendChild(sizeSlider);
        container.appendChild(sizeValue);
        
        // Add container to document body
        document.body.appendChild(container);
        
        // Add event listener to update point size
        sizeSlider.addEventListener('input', function() {
            var gd = document.querySelector('.js-plotly-plot');
            var plotData = gd._fullData;
            
            sizeValue.innerHTML = this.value;
            
            for (var i = 0; i < plotData.length; i++) {
                Plotly.restyle(gd, {'marker.size': parseFloat(this.value)}, [i]);
            }
        });
        
        // Create nearest neighbors container
        var neighborsContainer = document.createElement('div');
        neighborsContainer.id = 'neighbors-container';
        neighborsContainer.style = 'position: absolute; top: 450px; right: 10px; width: 250px; max-height: 300px; overflow-y: auto; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.15); display: none; z-index: 1000;';
        
        // Add header to neighbors container
        var neighborsHeader = document.createElement('h3');
        neighborsHeader.innerHTML = 'Nearest Neighbors';
        neighborsHeader.style = 'margin-top: 0; margin-bottom: 8px; font-size: 14px; font-family: Arial; text-align: center;';
        neighborsContainer.appendChild(neighborsHeader);
        
        // Create list for neighbors
        var neighborsList = document.createElement('ul');
        neighborsList.id = 'neighbors-list';
        neighborsList.style = 'list-style-type: none; padding: 0; margin: 0; font-family: Arial; font-size: 12px;';
        neighborsContainer.appendChild(neighborsList);
        
        // Add to document
        document.body.appendChild(neighborsContainer);
        
        // Get plot data to find nearest neighbors
        var gd = document.querySelector('.js-plotly-plot');
        
        // Load token data from embedded JSON
        var tokenData = """ + json.dumps({
            'token_labels': token_labels,
            'token_types': token_types,
            'coords': embeddings_reduced.tolist()
        }) + """;
        
        // Load trace-to-indices mapping
        var traceToIndices = """ + json.dumps(trace_to_indices) + """;
        
        // Function to calculate Euclidean distance
        function euclideanDistance(p1, p2) {
            return Math.sqrt(Math.pow(p1[0] - p2[0], 2) + Math.pow(p1[1] - p2[1], 2));
        }
        
        // Function to find nearest neighbors
        function findNearestNeighbors(pointIndex, n = 10) {
            var targetPoint = tokenData.coords[pointIndex];
            var distances = [];
            
            // Calculate distances to all other points
            for (var i = 0; i < tokenData.coords.length; i++) {
                if (i !== pointIndex) {
                    distances.push({
                        index: i,
                        distance: euclideanDistance(targetPoint, tokenData.coords[i]),
                        label: tokenData.token_labels[i],
                        type: tokenData.token_types[i]
                    });
                }
            }
            
            // Sort by distance
            distances.sort(function(a, b) {
                return a.distance - b.distance;
            });
            
            // Return top N
            return distances.slice(0, n);
        }
        
        // Add click event to plot
        gd.on('plotly_click', function(data) {
            var point = data.points[0];
            var curveNumber = point.curveNumber;
            var pointNumber = point.pointNumber;
            
            // Use the trace-to-indices mapping to get the correct global index
            var globalIndex = traceToIndices[curveNumber][pointNumber];
            
            if (globalIndex >= 0) {
                // Get clicked token
                var clickedToken = tokenData.token_labels[globalIndex];
                var clickedType = tokenData.token_types[globalIndex];
                
                // Find nearest neighbors
                var neighbors = findNearestNeighbors(globalIndex);
                
                // Update neighbors list
                var neighborsList = document.getElementById('neighbors-list');
                neighborsList.innerHTML = '<li style="padding: 5px; background: #f0f0f0; margin-bottom: 5px; border-radius: 3px;"><b>Selected:</b> ' + clickedToken + ' (' + clickedType + ')</li>';
                
                for (var i = 0; i < neighbors.length; i++) {
                    var neighbor = neighbors[i];
                    var typeColor = getColorForType(neighbor.type);
                    neighborsList.innerHTML += '<li style="padding: 3px; border-left: 4px solid ' + typeColor + '; margin-bottom: 3px;">' + 
                        (i+1) + '. ' + neighbor.label + 
                        '<span style="float: right; font-size: 10px; color: #777;">' + neighbor.distance.toFixed(2) + '</span></li>';
                }
                
                // Show the container
                document.getElementById('neighbors-container').style.display = 'block';
            }
        });
        
        // Function to get color for token type
        function getColorForType(type) {
            var colors = """ + json.dumps(category_colors) + """;
            return colors[type] || "#7f7f7f";
        }
    }
    </script>
    """
    
    # 10. Save as interactive HTML
    if output_filename is None:
        output_filename = f"figures/interactive_embeddings_{experiment_name}_{run_name}_{reduction_method}.html"
    
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get the HTML string
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=True, include_mathjax='cdn')
    
    # Insert our custom JavaScript before the closing </body> tag
    modified_html = html_str.replace('</body>', f'{js_code}</body>')
    
    # Write the HTML to file
    with open(output_path, 'w') as f:
        f.write(modified_html)
    
    print(f"Interactive plot saved to {output_path}")
    
    return fig  # Return the figure for display in notebooks

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create interactive embedding plot")
    parser.add_argument("--experiment", type=str, default="stable_pretrain", help="Experiment name")
    parser.add_argument("--run", type=str, default="8192", help="Run name/folder")
    parser.add_argument("--checkpoint", type=str, default="last.ckpt", help="Checkpoint filename")
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "pacmap"], 
                        help="Dimensionality reduction method")
    parser.add_argument("--dimensions", type=int, default=2, choices=[2, 3], 
                        help="Output dimensions (2D or 3D)")
    parser.add_argument("--subset", type=int, default=None, 
                        help="Number of tokens to plot (None for all)")
    parser.add_argument("--output", type=str, default=None, 
                        help="Output filename (HTML)")
    parser.add_argument("--random", action="store_true", 
                        help="Use random initialization if checkpoint loading fails")
    parser.add_argument("--point-size", type=int, default=4,
                        help="Initial size of points in the plot")
    
    args = parser.parse_args()
    
    # Default token types to plot
    token_types = [
        "Health_ICD10", "Health_Prescription", "Health_Doctor", 
        "Health General", "Labor", "Demographic", 
        "Social", "Education", "Labor_Income"
    ]
    
    plot_interactive_embeddings(
        experiment_name=args.experiment,
        run_name=args.run,
        checkpoint_filename=args.checkpoint,
        reduction_method=args.method,
        n_components=args.dimensions,
        plot_subset=args.subset,
        token_types_to_plot=token_types,
        output_filename=args.output,
        random_init=args.random,
        initial_point_size=args.point_size
    )