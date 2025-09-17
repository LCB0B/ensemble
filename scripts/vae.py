import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import lmdb
import json
import time
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Import paths
from src.paths import FPATH

# ----- LMDB Loading and Caching Functions -----
def load_embeddings_from_lmdb(lmdb_path: str, sample_size: int = None, random_seed: int = 42):
    """
    Load embeddings from LMDB format
    
    Args:
        lmdb_path: Path to LMDB database
        sample_size: Number of embeddings to sample (None for all)
        random_seed: Random seed for sampling
    
    Returns:
        numpy array of embeddings
    """
    lmdb_path = Path(lmdb_path)
    if not lmdb_path.exists():
        raise FileNotFoundError(f"LMDB database not found: {lmdb_path}")
    
    print(f"Loading embeddings from LMDB: {lmdb_path}")
    
    env = lmdb.open(str(lmdb_path), readonly=True, max_dbs=3)
    embeddings_db = env.open_db(b'embeddings')
    config_db = env.open_db(b'config')
    
    # Load config to get dimensions and total count
    with env.begin() as txn:
        config_bytes = txn.get(b'config', db=config_db)
        if config_bytes:
            config = json.loads(config_bytes.decode('utf-8'))
        else:
            config = {}
    
    total_embeddings = config.get('total_embeddings', 0)
    embedding_dim = config.get('embedding_dim', 0)
    
    print(f"Total embeddings in LMDB: {total_embeddings:,}")
    print(f"Embedding dimensions: {embedding_dim}")
    
    # Determine how many to load
    if sample_size is None or sample_size >= total_embeddings:
        load_size = total_embeddings
        indices = range(total_embeddings)
        print(f"Loading all {load_size:,} embeddings")
    else:
        load_size = sample_size
        np.random.seed(random_seed)
        indices = np.random.choice(total_embeddings, size=load_size, replace=False)
        indices = sorted(indices)  # Sort for better LMDB access pattern
        print(f"Sampling {load_size:,} embeddings from {total_embeddings:,}")
    
    # Load embeddings
    embeddings = np.zeros((load_size, embedding_dim), dtype=np.float32)
    
    with env.begin() as txn:
        for i, idx in enumerate(indices):
            key = str(idx).encode('utf-8')
            value = txn.get(key, db=embeddings_db)
            if value:
                embedding = np.frombuffer(value, dtype=np.float32)
                embeddings[i] = embedding
            
            if load_size > 10000:  # Only show progress for large loads
                step = max(1, len(indices) // 10)
                if (i + 1) % step == 0:
                    print(f"   Loaded {i+1:,}/{load_size:,} embeddings")
    
    env.close()
    print(f"Successfully loaded {embeddings.shape[0]:,} embeddings")
    
    return embeddings


def copy_lmdb_to_local_cache(source_lmdb_path: Path, cache_lmdb_path: Path, sample_size: int = None, random_seed: int = 42):
    """Copy LMDB data to local cache, with optional sampling"""
    
    print(f"Copying LMDB from {source_lmdb_path} to {cache_lmdb_path}")
    
    # Open source
    source_env = lmdb.open(str(source_lmdb_path), readonly=True, max_dbs=3)
    source_embeddings_db = source_env.open_db(b'embeddings')
    source_config_db = source_env.open_db(b'config')
    
    # Get source info
    with source_env.begin() as txn:
        config_bytes = txn.get(b'config', db=source_config_db)
        if config_bytes:
            source_config = json.loads(config_bytes.decode('utf-8'))
        else:
            source_config = {}
    
    total_embeddings = source_config.get('total_embeddings', 0)
    embedding_dim = source_config.get('embedding_dim', 0)
    
    # Determine what to copy
    if sample_size is None or sample_size >= total_embeddings:
        indices = list(range(total_embeddings))
        final_size = total_embeddings
        print(f"Copying all {total_embeddings:,} embeddings")
    else:
        np.random.seed(random_seed)
        indices = sorted(np.random.choice(total_embeddings, size=sample_size, replace=False))
        final_size = sample_size
        print(f"Copying {sample_size:,} sampled embeddings from {total_embeddings:,}")
    
    # Create cache LMDB
    cache_lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_lmdb_path.exists():
        import shutil
        shutil.rmtree(cache_lmdb_path)
    
    # Calculate map size
    map_size = final_size * embedding_dim * 4 * 2  # float32 * safety factor
    cache_env = lmdb.open(str(cache_lmdb_path), map_size=map_size, max_dbs=3)
    cache_embeddings_db = cache_env.open_db(b'embeddings')
    cache_config_db = cache_env.open_db(b'config')
    
    # Copy embeddings
    with source_env.begin() as source_txn, cache_env.begin(write=True) as cache_txn:
        for i, orig_idx in enumerate(indices):
            source_key = str(orig_idx).encode('utf-8')
            cache_key = str(i).encode('utf-8')  # Sequential keys in cache
            
            value = source_txn.get(source_key, db=source_embeddings_db)
            if value:
                cache_txn.put(cache_key, value, db=cache_embeddings_db)
            
            if (i + 1) % 10000 == 0:
                print(f"   Copied {i+1:,}/{final_size:,} embeddings")
        
        # Copy/update config
        cache_config = source_config.copy()
        cache_config.update({
            'total_embeddings': final_size,
            'cached_from': str(source_lmdb_path),
            'cached_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'sample_size': sample_size,
            'random_seed': random_seed if sample_size else None
        })
        
        config_bytes = json.dumps(cache_config).encode('utf-8')
        cache_txn.put(b'config', config_bytes, db=cache_config_db)
    
    source_env.close()
    cache_env.close()
    
    print(f"LMDB cache created: {final_size:,} embeddings at {cache_lmdb_path}")
    return cache_lmdb_path


def get_or_create_local_lmdb_cache(model_name: str, sample_size: int = None, random_seed: int = 42, force_refresh: bool = False):
    """
    Get or create local LMDB cache from source LMDB
    
    Returns:
        Path to local LMDB cache
    """
    # Source LMDB path (on project drive)
    source_lmdb_path = FPATH.FPATH_PROJECT / "embeddings" / model_name / "embeddings.lmdb"
    
    # Local cache directory (on data drive for faster access)
    cache_dir = FPATH.DATA / "embeddings" / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache LMDB name
    if sample_size is None:
        cache_lmdb_path = cache_dir / "embeddings_full.lmdb"
        cache_info_file = cache_dir / "cache_info_full.json"
    else:
        cache_lmdb_path = cache_dir / f"embeddings_sample_{sample_size}_seed_{random_seed}.lmdb"
        cache_info_file = cache_dir / f"cache_info_sample_{sample_size}_seed_{random_seed}.json"
    
    # Check if cache exists and is valid
    if cache_lmdb_path.exists() and cache_info_file.exists() and not force_refresh:
        print(f"Local LMDB cache found: {cache_lmdb_path}")
        
        try:
            with open(cache_info_file, 'r') as f:
                cache_info = json.load(f)
            
            # Check if source LMDB has been modified
            if source_lmdb_path.exists():
                source_mtime = source_lmdb_path.stat().st_mtime
                if cache_info.get('source_mtime', 0) >= source_mtime:
                    print("Cache is up to date, using local LMDB cache")
                    return cache_lmdb_path
                else:
                    print("Source LMDB is newer, refreshing cache...")
            else:
                print("Source LMDB no longer exists, but using existing cache...")
                return cache_lmdb_path
                
        except (json.JSONDecodeError, FileNotFoundError):
            print("Cache info corrupted, refreshing cache...")
    
    # Check if LMDB source exists
    if not source_lmdb_path.exists():
        raise FileNotFoundError(
            f"LMDB database not found: {source_lmdb_path}\n"
            f"Please run the embedding extraction script first to create the LMDB dataset."
        )
    
    # Create cache from source
    print(f"Creating local LMDB cache from: {source_lmdb_path}")
    copy_lmdb_to_local_cache(source_lmdb_path, cache_lmdb_path, sample_size, random_seed)
    
    # Save cache info
    cache_info = {
        'source_path': str(source_lmdb_path),
        'source_type': 'lmdb',
        'source_mtime': source_lmdb_path.stat().st_mtime,
        'cached_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'sample_size': sample_size,
        'random_seed': random_seed,
        'cache_lmdb_path': str(cache_lmdb_path)
    }
    
    with open(cache_info_file, 'w') as f:
        json.dump(cache_info, f, indent=2)
    
    return cache_lmdb_path


def load_embeddings_from_local_lmdb(cache_lmdb_path: Path, preload_to_memory: bool = True):
    """Load embeddings from local LMDB cache"""
    
    print(f"Loading embeddings from local LMDB cache: {cache_lmdb_path}")
    
    if preload_to_memory:
        # Load all to memory for fastest access
        return load_embeddings_from_lmdb(str(cache_lmdb_path))
    else:
        # Return LMDB path for memory-mapped access
        return str(cache_lmdb_path)


class LMDBDataset(torch.utils.data.Dataset):
    """Dataset that loads from LMDB on-demand (for memory-mapped access)"""
    
    def __init__(self, lmdb_path: str):
        self.lmdb_path = lmdb_path
        
        # Get dataset size
        env = lmdb.open(lmdb_path, readonly=True, max_dbs=3)
        config_db = env.open_db(b'config')
        
        with env.begin() as txn:
            config_bytes = txn.get(b'config', db=config_db)
            if config_bytes:
                config = json.loads(config_bytes.decode('utf-8'))
                self.length = config.get('total_embeddings', 0)
                self.embedding_dim = config.get('embedding_dim', 0)
            else:
                raise ValueError("No config found in LMDB")
        
        env.close()
        
        # Keep one environment per worker (thread-safe)
        self.env = None
        self.embeddings_db = None
        
    def _init_db(self):
        """Initialize DB connection (called once per worker)"""
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, max_dbs=3)
            self.embeddings_db = self.env.open_db(b'embeddings')
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        self._init_db()
        
        with self.env.begin() as txn:
            key = str(idx).encode('utf-8')
            value = txn.get(key, db=self.embeddings_db)
            
            if value is None:
                raise IndexError(f"Embedding {idx} not found")
            
            embedding = np.frombuffer(value, dtype=np.float32)
            return torch.from_numpy(embedding).float()
    
    def __del__(self):
        if self.env:
            self.env.close()


# ----- Standard PCA and UMAP plotting -----
def load_and_plot_embeddings(embeddings: np.ndarray, figures_dir: str = "figures"):
    os.makedirs(figures_dir, exist_ok=True)
    
    # PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(pcs[:, 0], pcs[:, 1], s=1, alpha=0.6, edgecolor='none')
    plt.title("PCA of Embeddings")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/pca_embeddings.png", dpi=300)
    plt.close()

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    ums = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(ums[:, 0], ums[:, 1], s=1, alpha=0.6, edgecolor='none')
    plt.title("UMAP of Embeddings")
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/umap_embeddings.png", dpi=300)
    plt.close()

    print(f"PCA and UMAP plots saved to {figures_dir}/pca_embeddings.png and {figures_dir}/umap_embeddings.png")

# ----- Simple VAE -----
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2, hidden_dims=[512, 256, 128]):
        """
        Simple VAE with flexible architecture
        
        Args:
            input_dim: Input dimension (embedding size)
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions for encoder/decoder
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Build Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build Decoder
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        prev_dim = latent_dim
        for h_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2)
            ])
            prev_dim = h_dim
        
        # Final layer
        decoder_layers.append(nn.Linear(hidden_dims_reversed[-1], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent code to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z
    
    def loss_function(self, x, x_recon, mu, logvar, beta=1.0):
        """
        Compute VAE loss
        
        Args:
            x: Original input
            x_recon: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            beta: Weight for KL divergence (beta-VAE)
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def sample(self, num_samples, device):
        """Generate new samples from the latent space"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


# ----- Advanced VAE Variants for Structured Embeddings -----
class PlanarFlow(nn.Module):
    """Planar normalizing flow layer"""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))
        self.scale = nn.Parameter(torch.randn(dim))
        self.bias = nn.Parameter(torch.randn(1))
        
    def forward(self, z):
        # Planar flow transformation
        activation = torch.tanh(torch.sum(self.weight * z, dim=1, keepdim=True) + self.bias)
        return z + self.scale.unsqueeze(0) * activation
    
    def log_det_jacobian(self, z):
        # Log determinant of Jacobian for planar flow
        activation = torch.tanh(torch.sum(self.weight * z, dim=1, keepdim=True) + self.bias)
        psi = (1 - activation**2) * self.weight.unsqueeze(0)
        return torch.log(torch.abs(1 + torch.sum(psi * self.scale.unsqueeze(0), dim=1)) + 1e-8)


class FlowVAE(VAE):
    """VAE with normalizing flow prior for more complex latent structure"""
    
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[1024, 512, 256, 128], n_flows=4):
        super().__init__(input_dim, latent_dim, hidden_dims)
        
        # Create flow layers
        self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(n_flows)])
        self.n_flows = n_flows
        
    def flow_forward(self, z0):
        """Transform z0 through normalizing flows"""
        zk = z0
        log_det_jacobian = 0
        
        for flow in self.flows:
            zk = flow(zk)
            log_det_jacobian += flow.log_det_jacobian(zk)
            
        return zk, log_det_jacobian
    
    def loss_function(self, x, x_recon, mu, logvar, beta=0.1):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # Sample from encoder
        z0 = self.reparameterize(mu, logvar)
        
        # Transform through flows
        zk, log_det_jacobian = self.flow_forward(z0)
        
        # KL with flow
        # log q(z0) - log p(zk) + log_det_jacobian
        log_q_z0 = -0.5 * torch.sum(logvar + (z0 - mu)**2 / torch.exp(logvar), dim=1)
        log_p_zk = -0.5 * torch.sum(zk**2, dim=1)  # Standard normal prior
        
        kl_loss = torch.sum(log_q_z0 - log_p_zk - log_det_jacobian)
        
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss
    
    def sample(self, num_samples, device):
        """Sample from flow prior"""
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(num_samples, self.latent_dim).to(device)
            # Transform through flows  
            zk, _ = self.flow_forward(z)
            return self.decode(zk)


class MixtureOfGaussiansVAE(VAE):
    """VAE with Mixture of Gaussians prior - encourages clustering"""
    
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[1024, 512, 256, 128], 
                 n_components=10, component_weight=1.0):
        super().__init__(input_dim, latent_dim, hidden_dims)
        
        self.n_components = n_components
        self.component_weight = component_weight
        
        # Learnable mixture parameters
        self.mixture_logits = nn.Parameter(torch.randn(n_components))  # Component weights
        self.mixture_means = nn.Parameter(torch.randn(n_components, latent_dim))  # Component means
        self.mixture_logvars = nn.Parameter(torch.zeros(n_components, latent_dim))  # Component log variances
        
    def mixture_prior_log_prob(self, z):
        """Compute log probability under mixture of Gaussians prior"""
        # z: [batch_size, latent_dim]
        # Expand for broadcasting
        z_expanded = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
        means_expanded = self.mixture_means.unsqueeze(0)  # [1, n_components, latent_dim]
        logvars_expanded = self.mixture_logvars.unsqueeze(0)  # [1, n_components, latent_dim]
        
        # Compute log probability for each component
        log_probs = -0.5 * torch.sum(
            logvars_expanded + (z_expanded - means_expanded)**2 / torch.exp(logvars_expanded),
            dim=2
        )  # [batch_size, n_components]
        
        # Add mixture weights
        mixture_weights = F.log_softmax(self.mixture_logits, dim=0)  # [n_components]
        log_probs = log_probs + mixture_weights.unsqueeze(0)  # [batch_size, n_components]
        
        # Log sum exp over components
        return torch.logsumexp(log_probs, dim=1)  # [batch_size]
    
    def loss_function(self, x, x_recon, mu, logvar, beta=0.1):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        
        # Encoder log probability (recognition model)
        log_q_z = -0.5 * torch.sum(logvar + (z - mu)**2 / torch.exp(logvar), dim=1)
        
        # Prior log probability (mixture of Gaussians)
        log_p_z = self.mixture_prior_log_prob(z)
        
        # KL divergence
        kl_loss = torch.sum(log_q_z - log_p_z)
        
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss
    
    def sample(self, num_samples, device):
        """Sample from mixture of Gaussians prior"""
        with torch.no_grad():
            # Sample component indices
            mixture_weights = F.softmax(self.mixture_logits, dim=0)
            component_indices = torch.multinomial(mixture_weights, num_samples, replacement=True)
            
            # Sample from selected components
            selected_means = self.mixture_means[component_indices]  # [num_samples, latent_dim]
            selected_logvars = self.mixture_logvars[component_indices]  # [num_samples, latent_dim]
            
            # Sample from Gaussians
            eps = torch.randn_like(selected_means)
            z = selected_means + eps * torch.exp(0.5 * selected_logvars)
            
            return self.decode(z.to(device))
    
    def get_cluster_assignments(self, z):
        """Get soft cluster assignments for latent codes"""
        with torch.no_grad():
            z_expanded = z.unsqueeze(1)
            means_expanded = self.mixture_means.unsqueeze(0)
            logvars_expanded = self.mixture_logvars.unsqueeze(0)
            
            # Compute log probabilities
            log_probs = -0.5 * torch.sum(
                logvars_expanded + (z_expanded - means_expanded)**2 / torch.exp(logvars_expanded),
                dim=2
            )
            
            # Add mixture weights and normalize
            mixture_weights = F.log_softmax(self.mixture_logits, dim=0)
            log_probs = log_probs + mixture_weights.unsqueeze(0)
            
            return F.softmax(log_probs, dim=1)


class VampPriorVAE(VAE):
    """VAE with VampPrior - uses learned pseudo-inputs as prior"""
    
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[1024, 512, 256, 128], 
                 n_pseudo_inputs=500):
        super().__init__(input_dim, latent_dim, hidden_dims)
        
        self.n_pseudo_inputs = n_pseudo_inputs
        
        # Learnable pseudo-inputs
        self.pseudo_inputs = nn.Parameter(torch.randn(n_pseudo_inputs, input_dim))
        
    def vamp_prior_log_prob(self, z):
        """Compute log probability under VampPrior"""
        # Encode pseudo-inputs to get mixture components
        pseudo_mu, pseudo_logvar = self.encode(self.pseudo_inputs)
        
        # Expand z for broadcasting: [batch_size, 1, latent_dim]
        z_expanded = z.unsqueeze(1)
        
        # Expand pseudo parameters: [1, n_pseudo_inputs, latent_dim]
        pseudo_mu_expanded = pseudo_mu.unsqueeze(0)
        pseudo_logvar_expanded = pseudo_logvar.unsqueeze(0)
        
        # Compute log probabilities for each pseudo-input component
        log_probs = -0.5 * torch.sum(
            pseudo_logvar_expanded + 
            (z_expanded - pseudo_mu_expanded)**2 / torch.exp(pseudo_logvar_expanded),
            dim=2
        )  # [batch_size, n_pseudo_inputs]
        
        # Uniform mixture weights (1/K for each component)
        log_probs = log_probs - np.log(self.n_pseudo_inputs)
        
        # Log sum exp over pseudo-inputs
        return torch.logsumexp(log_probs, dim=1)
    
    def loss_function(self, x, x_recon, mu, logvar, beta=0.1):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        
        # Encoder log probability
        log_q_z = -0.5 * torch.sum(logvar + (z - mu)**2 / torch.exp(logvar), dim=1)
        
        # VampPrior log probability
        log_p_z = self.vamp_prior_log_prob(z)
        
        # KL divergence
        kl_loss = torch.sum(log_q_z - log_p_z)
        
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss
    
    def sample(self, num_samples, device):
        """Sample from VampPrior"""
        with torch.no_grad():
            # Randomly select pseudo-inputs
            indices = torch.randint(0, self.n_pseudo_inputs, (num_samples,))
            selected_pseudo = self.pseudo_inputs[indices].to(device)
            
            # Encode selected pseudo-inputs
            pseudo_mu, pseudo_logvar = self.encode(selected_pseudo)
            
            # Sample from the corresponding Gaussians
            eps = torch.randn_like(pseudo_mu)
            z = pseudo_mu + eps * torch.exp(0.5 * pseudo_logvar)
            
            return self.decode(z)


# ----- Training functions for advanced VAEs -----
def train_advanced_vae(model, dataloader, optimizer, device, epoch, beta=0.1):
    """Training function that works with all VAE variants"""
    model.train()
    train_loss = 0
    train_recon = 0
    train_kl = 0
    
    for batch_idx, (data,) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass (works for all VAE variants)
        if isinstance(model, (FlowVAE, MixtureOfGaussiansVAE, VampPriorVAE)):
            x_recon, mu, logvar, z = model(data)
            loss, recon_loss, kl_loss = model.loss_function(data, x_recon, mu, logvar, beta)
        else:
            # Standard VAE
            x_recon, mu, logvar, z = model(data)
            loss, recon_loss, kl_loss = model.loss_function(data, x_recon, mu, logvar, beta)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_recon += recon_loss.item()
        train_kl += kl_loss.item()
        
    avg_loss = train_loss / len(dataloader.dataset)
    avg_recon = train_recon / len(dataloader.dataset)
    avg_kl = train_kl / len(dataloader.dataset)
    
    return avg_loss, avg_recon, avg_kl


def evaluate_advanced_vae(model, dataloader, device, beta=0.1):
    """Evaluation function that works with all VAE variants"""
    model.eval()
    test_loss = 0
    test_recon = 0
    test_kl = 0
    
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)
            
            if isinstance(model, (FlowVAE, MixtureOfGaussiansVAE, VampPriorVAE)):
                x_recon, mu, logvar, z = model(data)
                loss, recon_loss, kl_loss = model.loss_function(data, x_recon, mu, logvar, beta)
            else:
                x_recon, mu, logvar, z = model(data)
                loss, recon_loss, kl_loss = model.loss_function(data, x_recon, mu, logvar, beta)
            
            test_loss += loss.item()
            test_recon += recon_loss.item()
            test_kl += kl_loss.item()
    
    avg_loss = test_loss / len(dataloader.dataset)
    avg_recon = test_recon / len(dataloader.dataset)
    avg_kl = test_kl / len(dataloader.dataset)
    
    return avg_loss, avg_recon, avg_kl
class ClusterVAE(VAE):
    """VAE with clustering objective to create less smooth embeddings"""
    
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[1024, 512, 256, 128], n_clusters=10):
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.n_clusters = n_clusters
        
        # Learnable cluster centers
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))
        
    def cluster_loss(self, z, temperature=1.0):
        """Encourage clustering in latent space"""
        # Distance to each cluster center
        distances = torch.cdist(z, self.cluster_centers.unsqueeze(0)).squeeze(0)  # [batch, n_clusters]
        
        # Soft assignment (closer = higher probability)
        assignments = F.softmax(-distances / temperature, dim=1)
        
        # Entropy penalty (encourage confident assignments)
        entropy = -torch.sum(assignments * torch.log(assignments + 1e-10), dim=1).mean()
        
        return entropy * 0.1  # Weight the clustering loss
    
    def loss_function(self, x, x_recon, mu, logvar, beta=0.1):
        # Standard VAE loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Add clustering loss
        z = self.reparameterize(mu, logvar)
        cluster_loss = self.cluster_loss(z)
        
        total_loss = recon_loss + beta * kl_loss + cluster_loss
        
        return total_loss, recon_loss, kl_loss


class WassersteinAE(nn.Module):
    """Wasserstein Auto-Encoder - creates more structured latent space"""
    
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[1024, 512, 256, 128]):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Build Encoder (same as VAE but no logvar)
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)  # Add dropout for regularization
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_encode = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build Decoder
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        prev_dim = latent_dim
        for h_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims_reversed[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_encode(h)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def mmd_loss(self, z, reg_weight=100.0):
        """Maximum Mean Discrepancy loss - encourages z to match prior"""
        # Sample from prior
        prior_z = torch.randn_like(z)
        
        # MMD between z and prior_z
        def compute_kernel(x, y):
            x_size = x.size(0)
            y_size = y.size(0)
            dim = x.size(1)
            
            tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
            tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)
            
            return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim)
        
        x_kernel = compute_kernel(z, z)
        y_kernel = compute_kernel(prior_z, prior_z)
        xy_kernel = compute_kernel(z, prior_z)
        
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return reg_weight * mmd
    
    def loss_function(self, x, x_recon, z):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        mmd_loss = self.mmd_loss(z)
        
        total_loss = recon_loss + mmd_loss
        return total_loss, recon_loss, mmd_loss
    
    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    def __init__(self, input_dim, latent_dim=2, hidden_dims=[512, 256, 128]):
        """
        Simple VAE with flexible architecture
        
        Args:
            input_dim: Input dimension (embedding size)
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions for encoder/decoder
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Build Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build Decoder
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        prev_dim = latent_dim
        for h_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2)
            ])
            prev_dim = h_dim
        
        # Final layer
        decoder_layers.append(nn.Linear(hidden_dims_reversed[-1], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent code to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z
    
    def loss_function(self, x, x_recon, mu, logvar, beta=1.0):
        """
        Compute VAE loss
        
        Args:
            x: Original input
            x_recon: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            beta: Weight for KL divergence (beta-VAE)
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def sample(self, num_samples, device):
        """Generate new samples from the latent space"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples

# ----- Training function -----
def train_vae(model, dataloader, optimizer, device, epoch, beta=1.0):
    model.train()
    train_loss = 0
    train_recon = 0
    train_kl = 0
    
    for batch_idx, (data,) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        x_recon, mu, logvar, z = model(data)
        loss, recon_loss, kl_loss = model.loss_function(data, x_recon, mu, logvar, beta)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_recon += recon_loss.item()
        train_kl += kl_loss.item()
        
    avg_loss = train_loss / len(dataloader.dataset)
    avg_recon = train_recon / len(dataloader.dataset)
    avg_kl = train_kl / len(dataloader.dataset)
    
    return avg_loss, avg_recon, avg_kl

# ----- Evaluation function -----
def evaluate_vae(model, dataloader, device, beta=1.0):
    model.eval()
    test_loss = 0
    test_recon = 0
    test_kl = 0
    
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)
            x_recon, mu, logvar, z = model(data)
            loss, recon_loss, kl_loss = model.loss_function(data, x_recon, mu, logvar, beta)
            
            test_loss += loss.item()
            test_recon += recon_loss.item()
            test_kl += kl_loss.item()
    
    avg_loss = test_loss / len(dataloader.dataset)
    avg_recon = test_recon / len(dataloader.dataset)
    avg_kl = test_kl / len(dataloader.dataset)
    
    return avg_loss, avg_recon, avg_kl

# ----- Get latent embeddings -----
def get_latent_embeddings(model, dataloader, device):
    model.eval()
    latents = []
    
    with torch.no_grad():
        for data, in dataloader:
            data = data.to(device)
            mu, _ = model.encode(data)
            latents.append(mu.cpu())
    
    return torch.cat(latents, dim=0).numpy()

# ----- Main processing -----
def main(model_name: str,
         sample_size: int = None,
         figures_dir: str = None,
         use_vae: bool = False,
         vae_params: dict = None,
         batch_size: int = 256,
         epochs: int = 100,
         lr: float = 1e-3,
         beta: float = 1.0,
         seed: int = 42,
         force_refresh_cache: bool = False,
         num_workers: int = 4,
         preload_data: bool = True,
         preload_model: bool = False,
         config_suffix: str = ""):  # NEW: Add config suffix parameter:
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Set up multiprocessing
    import torch.multiprocessing as mp
    if num_workers > 0:
        mp.set_sharing_strategy('file_system')  # Better for large tensors
    
    # Set up figures directory using FPATH with config suffix
    if figures_dir is None:
        base_figures_dir = str(FPATH.FIGURES / model_name)
        if config_suffix:
            figures_dir = f"{base_figures_dir}_{config_suffix}"
        else:
            figures_dir = base_figures_dir
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"Processing model: {model_name}")
    print(f"Source LMDB: {FPATH.FPATH_PROJECT / 'embeddings' / model_name / 'embeddings.lmdb'}")
    print(f"Local cache: {FPATH.DATA / 'embeddings' / model_name}")
    print(f"Figures: {FPATH.FIGURES / model_name}")
    
    # Get or create local LMDB cache
    cache_lmdb_path = get_or_create_local_lmdb_cache(
        model_name=model_name,
        sample_size=sample_size,
        random_seed=seed,
        force_refresh=force_refresh_cache
    )
    
    # Load embeddings based on strategy
    if preload_data:
        print("Preloading all data to memory for fastest DataLoader...")
        embeddings = load_embeddings_from_local_lmdb(cache_lmdb_path, preload_to_memory=True)
        dataset = TensorDataset(torch.from_numpy(embeddings).float())
        print(f"Preloaded {embeddings.shape[0]:,} embeddings to memory")
    else:
        print("Using LMDB dataset for memory-mapped loading...")
        dataset = LMDBDataset(str(cache_lmdb_path))
        embeddings = None  # We'll get the shape from the dataset
        
        # Get embedding info from dataset
        env = lmdb.open(str(cache_lmdb_path), readonly=True, max_dbs=3)
        config_db = env.open_db(b'config')
        with env.begin() as txn:
            config_bytes = txn.get(b'config', db=config_db)
            config = json.loads(config_bytes.decode('utf-8'))
        env.close()
        
        embedding_dim = config['embedding_dim']
        total_embeddings = config['total_embeddings']
        print(f"Using LMDB dataset: {total_embeddings:,} embeddings, {embedding_dim} dimensions")
    
    input_dim = embeddings.shape[1] if embeddings is not None else embedding_dim
    total_size = embeddings.shape[0] if embeddings is not None else total_embeddings
    
    print(f"Dataset shape: ({total_size:,}, {input_dim})")
    if embeddings is not None:
        print(f"Memory usage: ~{embeddings.nbytes / 1024**3:.1f} GB")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    if not use_vae:
        # For PCA/UMAP, we need the actual embeddings array
        if embeddings is None:
            print("Loading embeddings for PCA/UMAP visualization...")
            embeddings = load_embeddings_from_local_lmdb(cache_lmdb_path, preload_to_memory=True)
        load_and_plot_embeddings(embeddings, figures_dir)
    else:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup VAE - choose from multiple variants
        vae_params = vae_params or {}
        vae_type = vae_params.get('type', 'standard')  # standard, flow, mixture, vamp
        
        if vae_type == 'flow':
            model = FlowVAE(
                input_dim=input_dim,
                latent_dim=vae_params.get('latent_dim', 8),
                hidden_dims=vae_params.get('hidden_dims', [1024, 512, 256, 128]),
                n_flows=vae_params.get('n_flows', 4)
            ).to(device)
            print("Using FlowVAE with normalizing flows")
            
        elif vae_type == 'mixture':
            model = MixtureOfGaussiansVAE(
                input_dim=input_dim,
                latent_dim=vae_params.get('latent_dim', 8),
                hidden_dims=vae_params.get('hidden_dims', [1024, 512, 256, 128]),
                n_components=vae_params.get('n_components', 10)
            ).to(device)
            print(f"Using MixtureOfGaussiansVAE with {vae_params.get('n_components', 10)} components")
            
        elif vae_type == 'vamp':
            model = VampPriorVAE(
                input_dim=input_dim,
                latent_dim=vae_params.get('latent_dim', 8),
                hidden_dims=vae_params.get('hidden_dims', [1024, 512, 256, 128]),
                n_pseudo_inputs=vae_params.get('n_pseudo_inputs', 500)
            ).to(device)
            print(f"Using VampPriorVAE with {vae_params.get('n_pseudo_inputs', 500)} pseudo-inputs")
            
        else:
            # Standard VAE
            model = VAE(
                input_dim=input_dim,
                latent_dim=vae_params.get('latent_dim', 2),
                hidden_dims=vae_params.get('hidden_dims', [512, 256, 128])
            ).to(device)
            print("Using standard VAE")
            
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"VAE type: {vae_type}")
        print(f"Architecture: {vae_params}")
        
        # Split into train and validation
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Setup data loaders with optimized settings
        persistent_workers = num_workers > 0
        pin_memory = torch.cuda.is_available() and num_workers <= 2  # Only for small num_workers
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if num_workers > 0 else 2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if num_workers > 0 else 2
        )
        full_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=min(num_workers, 2),  # Use fewer workers for inference
            pin_memory=pin_memory,
            persistent_workers=min(num_workers, 2) > 0
        )
        
        print(f"DataLoader settings:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Num workers: {num_workers}")
        print(f"  - Pin memory: {pin_memory}")
        print(f"  - Persistent workers: {persistent_workers}")
        print(f"  - Batches per epoch: train={len(train_loader)}, val={len(val_loader)}")
        print(f"  - Expected GPU memory per batch: ~{batch_size * input_dim * 4 / 1024**2:.1f} MB")
        
        # Monitor GPU usage if available
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB total")
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")

        if not preload_model:
            # Setup optimizer
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
            
            # Training loop with tqdm
            best_val_loss = float('inf')
            train_losses = []
            val_losses = []
            
            print(f"Starting VAE training...")
            
            try:
                with tqdm(range(1, epochs + 1), desc="Training VAE", total=epochs) as pbar:
                    for epoch in pbar:
                        # Train - use advanced training function
                        train_loss, train_recon, train_kl = train_advanced_vae(model, train_loader, optimizer, device, epoch, beta)
                        
                        # Validate - use advanced evaluation function
                        val_loss, val_recon, val_kl = evaluate_advanced_vae(model, val_loader, device, beta)
                        
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        
                        # Learning rate scheduling
                        scheduler.step(val_loss)
                        
                        # Save best model with config suffix
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            model_save_path = f"{figures_dir}/best_vae_model.pth"
                            torch.save(model.state_dict(), model_save_path)
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'TrLoss': f'{train_loss:.3f}',
                            'TrRecon': f'{train_recon:.3f}',
                            'TrKL': f'{train_kl:.3f}',
                            'ValLoss': f'{val_loss:.3f}',
                            'ValRecon': f'{val_recon:.3f}',
                            'ValKL': f'{val_kl:.3f}',
                            'LR': f'{optimizer.param_groups[0]["lr"]:.1e}',
                            'Best': f'{best_val_loss:.3f}'
                        })
                        
            except KeyboardInterrupt:
                print(f"\nTraining interrupted by user")
                print("Continuing with best saved model...")
            
            # Plot training curves
            plt.figure(figsize=(10, 10))
            plt.plot(train_losses, label='Train')
            plt.plot(val_losses, label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('VAE Training Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{figures_dir}/vae_training.png", dpi=300)
            plt.close()

        # Load best model
        if (Path(figures_dir) / "best_vae_model.pth").exists():
            model.load_state_dict(torch.load(f"{figures_dir}/best_vae_model.pth"))
            print(f"Loaded best model from: {figures_dir}/best_vae_model.pth")
        
        # Get latent embeddings
        print("Extracting latent embeddings...")
        latent_embeddings = get_latent_embeddings(model, full_loader, device)
        
        # Visualize latent space
        print(f"Latent space shape: {latent_embeddings.shape}")
        
        if latent_embeddings.shape[1] == 2:
            # Direct 2D visualization
            plt.figure(figsize=(10, 10))
            plt.scatter(latent_embeddings[:, 0], latent_embeddings[:, 1], s=0.1, alpha=0.6, edgecolor='none')
            plt.title("VAE 2D Latent Space")
            plt.xlabel("Z1")
            plt.ylabel("Z2")
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{figures_dir}/vae_embedding.png", dpi=300)
            plt.close()
            
        elif latent_embeddings.shape[1] > 2:
            # Higher dimensional latent space - create multiple visualizations
            
            # 1. First two dimensions
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.scatter(latent_embeddings[:, 0], latent_embeddings[:, 1], s=0.1, alpha=0.6, edgecolor='none')
            plt.title("VAE Latent Space (Z1 vs Z2)")
            plt.xlabel("Z1")
            plt.ylabel("Z2")
            plt.grid(True, alpha=0.3)
            
            # 2. PCA of latent space
            plt.subplot(1, 3, 2)
            from sklearn.decomposition import PCA
            pca_latent = PCA(n_components=2)
            latent_pca = pca_latent.fit_transform(latent_embeddings)
            plt.scatter(latent_pca[:, 0], latent_pca[:, 1], s=0.1, alpha=0.6, edgecolor='none')
            plt.title("PCA of VAE Latent Space")
            plt.xlabel(f"PC1 ({pca_latent.explained_variance_ratio_[0]:.2%} var)")
            plt.ylabel(f"PC2 ({pca_latent.explained_variance_ratio_[1]:.2%} var)")
            plt.grid(True, alpha=0.3)
            
            # 3. UMAP of latent space (if not too large)
            plt.subplot(1, 3, 3)
            if len(latent_embeddings) <= 100000:  # UMAP can be slow on very large datasets
                import umap
                umap_latent = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                latent_umap = umap_latent.fit_transform(latent_embeddings)
                plt.scatter(latent_umap[:, 0], latent_umap[:, 1], s=0.1, alpha=0.6, edgecolor='none')
                plt.title("UMAP of VAE Latent Space")
                plt.xlabel("UMAP1")
                plt.ylabel("UMAP2")
                plt.grid(True, alpha=0.3)
            else:
                # For very large datasets, just show variance per dimension
                latent_vars = np.var(latent_embeddings, axis=0)
                plt.bar(range(len(latent_vars)), latent_vars)
                plt.title("Variance per Latent Dimension")
                plt.xlabel("Latent Dimension")
                plt.ylabel("Variance")
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{figures_dir}/vae_embedding.png", dpi=300)
            plt.close()
            
            # Additional visualization: Latent dimension statistics
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            latent_means = np.mean(latent_embeddings, axis=0)
            latent_stds = np.std(latent_embeddings, axis=0)
            x_pos = np.arange(len(latent_means))
            plt.bar(x_pos, latent_means, yerr=latent_stds, alpha=0.7, capsize=3)
            plt.title("Mean ± Std per Latent Dimension")
            plt.xlabel("Latent Dimension")
            plt.ylabel("Value")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.hist(latent_embeddings.flatten(), bins=50, alpha=0.7, density=True)
            plt.title("Overall Latent Value Distribution")
            plt.xlabel("Latent Values")
            plt.ylabel("Density")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{figures_dir}/vae_latent_analysis.png", dpi=300)
            plt.close()
            
            print(f"Latent space analysis:")
            print(f"  - Dimensions: {latent_embeddings.shape[1]}")
            print(f"  - Mean activation per dim: {latent_means}")
            print(f"  - Std per dim: {latent_stds}")
            print(f"  - Most active dims: {np.argsort(latent_stds)[-3:][::-1]}")  # Top 3 most varying
            
            # Special visualization for Mixture of Gaussians VAE
            if isinstance(model, MixtureOfGaussiansVAE):
                print(f"\nMixture of Gaussians Analysis:")
                
                # Get cluster assignments
                z_sample = torch.from_numpy(latent_embeddings[:10000]).float().to(device)  # Sample for speed
                cluster_probs = model.get_cluster_assignments(z_sample).cpu().numpy()
                cluster_assignments = np.argmax(cluster_probs, axis=1)
                
                # Plot colored by cluster assignment
                plt.figure(figsize=(15, 5))
                
                if latent_embeddings.shape[1] >= 2:
                    plt.subplot(1, 3, 1)
                    scatter = plt.scatter(latent_embeddings[:10000, 0], latent_embeddings[:10000, 1], 
                                        c=cluster_assignments, s=0.5, alpha=0.6, cmap='tab10')
                    plt.title(f"Cluster Assignments (n={model.n_components})")
                    plt.xlabel("Z1")
                    plt.ylabel("Z2")
                    plt.colorbar(scatter)
                    
                    # Plot mixture centers
                    centers = model.mixture_means.detach().cpu().numpy()
                    if centers.shape[1] >= 2:
                        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='X', 
                                  edgecolor='black', linewidth=2, label='Mixture Centers')
                        plt.legend()
                
                plt.subplot(1, 3, 2)
                # Cluster size histogram
                unique, counts = np.unique(cluster_assignments, return_counts=True)
                plt.bar(unique, counts, alpha=0.7)
                plt.title("Cluster Sizes")
                plt.xlabel("Cluster ID")
                plt.ylabel("Number of Points")
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 3, 3)
                # Mixture weights
                mixture_weights = F.softmax(model.mixture_logits, dim=0).detach().cpu().numpy()
                plt.bar(range(len(mixture_weights)), mixture_weights, alpha=0.7)
                plt.title("Learned Mixture Weights")
                plt.xlabel("Component")
                plt.ylabel("Weight")
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{figures_dir}/mixture_analysis.png", dpi=300)
                plt.close()
                
                print(f"  - Active clusters: {len(unique)}/{model.n_components}")
                print(f"  - Largest cluster: {np.max(counts)} points")
                print(f"  - Smallest cluster: {np.min(counts)} points")
                print(f"  - Mixture weights: {mixture_weights[:5]}...")  # Show first 5
        
        print(f"Visualization complete. Plots saved to {figures_dir}")
        
        return model, latent_embeddings


def clean_cache(model_name: str = None):
    """Clean local LMDB embedding cache"""
    if model_name:
        cache_dir = FPATH.DATA / "embeddings" / model_name
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print(f"Cleaned LMDB cache for model: {model_name}")
    else:
        cache_root = FPATH.DATA / "embeddings"
        if cache_root.exists():
            import shutil
            shutil.rmtree(cache_root)
            print("Cleaned all LMDB embedding caches")


if __name__ == "__main__":
    # Configuration - Update these for your model
    MODEL_NAME = "021_muddy_cobra-pretrain-lr0.0003"
    SAMPLE_SIZE = int(1e8)  # Use large number to get all embeddings
    
    print(f"Processing model: {MODEL_NAME}")
    print(f"Source LMDB: {FPATH.FPATH_PROJECT / 'embeddings' / MODEL_NAME / 'embeddings.lmdb'}")
    print(f"Local cache: {FPATH.DATA / 'embeddings' / MODEL_NAME}")
    print(f"Figures: {FPATH.FIGURES / MODEL_NAME}")
    
    # Check if LMDB source exists
    lmdb_path = FPATH.FPATH_PROJECT / 'embeddings' / MODEL_NAME / 'embeddings.lmdb'
    if not lmdb_path.exists():
        print(f"\nERROR: LMDB database not found: {lmdb_path}")
        print("Please run the embedding extraction script first to create the LMDB dataset.")
        exit(1)
    else:
        print(f"✓ LMDB source found: {lmdb_path}")
    
    # Global settings for all experiments
    TRAIN_MODELS = True  # Set to False to just load existing models and visualize
    BATCH_SIZE = 4096
    EPOCHS = 300
    LEARNING_RATE = 1e-3
    NUM_WORKERS = 8
    
    print(f"\nExperiment Settings:")
    print(f"  - Train models: {'Yes' if TRAIN_MODELS else 'No (load existing)'}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Num workers: {NUM_WORKERS}")
    
    
    # Define all configurations to test
    configurations = [
        {
            "name": "standard_vae",
            "params": {
                "type": "standard",
                "latent_dim": 2,
                "hidden_dims": [1024, 512, 256, 128],
            },
            "beta": 0.1,
            "description": "Standard VAE baseline"
        },
        {
            "name": "standard_vae_low_beta",
            "params": {
                "type": "standard", 
                "latent_dim":2,
                "hidden_dims": [1024, 512, 256, 128],
            },
            "beta": 0.01,
            "description": "Standard VAE with very low beta"
        },
        {
            "name": "mixture_vae_10comp",
            "params": {
                "type": "mixture",
                "latent_dim": 2,
                "hidden_dims": [1024, 512, 256, 128],
                "n_components": 10,
            },
            "beta": 0.05,
            "description": "Mixture VAE with 10 components"
        },
        {
            "name": "mixture_vae_20comp", 
            "params": {
                "type": "mixture",
                "latent_dim": 2,
                "hidden_dims": [1024, 512, 256, 128],
                "n_components": 20,
            },
            "beta": 0.05,
            "description": "Mixture VAE with 20 components"
        },
        {
            "name": "mixture_vae_50comp",
            "params": {
                "type": "mixture",
                "latent_dim": 2,
                "hidden_dims": [1024, 512, 256, 128], 
                "n_components": 50,
            },
            "beta": 0.05,
            "description": "Mixture VAE with 50 components"
        },
        {
            "name": "flow_vae_4flows",
            "params": {
                "type": "flow",
                "latent_dim": 2,
                "hidden_dims": [1024, 512, 256, 128],
                "n_flows": 4,
            },
            "beta": 0.1,
            "description": "Flow VAE with 4 normalizing flows"
        },
        {
            "name": "flow_vae_8flows",
            "params": {
                "type": "flow",
                "latent_dim": 2,
                "hidden_dims": [1024, 512, 256, 128],
                "n_flows": 8,
            },
            "beta": 0.1,
            "description": "Flow VAE with 8 normalizing flows"
        },
        {
            "name": "vamp_vae_500pseudo",
            "params": {
                "type": "vamp",
                "latent_dim": 2,
                "hidden_dims": [1024, 512, 256, 128],
                "n_pseudo_inputs": 500,
            },
            "beta": 0.1,
            "description": "VampPrior VAE with 500 pseudo-inputs"
        },
        {
            "name": "vamp_vae_1000pseudo",
            "params": {
                "type": "vamp",
                "latent_dim": 2,
                "hidden_dims": [1024, 512, 256, 128],
                "n_pseudo_inputs": 1000,
            },
            "beta": 0.1,
            "description": "VampPrior VAE with 1000 pseudo-inputs"
        },
    ]
    print(f"\nRunning systematic comparison of {len(configurations)} VAE variants...")

    # Store results for comparison
    results = {}
    
    # Loop through all configurations
    for i, config in enumerate(configurations):
        config_name = config["name"]
        config_params = config["params"]
        config_beta = config["beta"]
        config_desc = config["description"]
        
        print(f"\n{'='*80}")
        print(f"RUNNING CONFIGURATION {i+1}/{len(configurations)}: {config_name.upper()}")
        print(f"Description: {config_desc}")
        print(f"Parameters: {config_params}")
        print(f"Beta: {config_beta}")
        print(f"Expected output: {FPATH.FIGURES / MODEL_NAME}_{config_name}")
        print(f"{'='*80}")
        
        try:
            # Run the experiment
            model, latent_embeddings = main(
                model_name=MODEL_NAME,
                sample_size=SAMPLE_SIZE,
                use_vae=True,
                vae_params=config_params,
                batch_size=4096,
                epochs=300,
                lr=1e-3,
                beta=config_beta,
                seed=42,
                force_refresh_cache=False,
                num_workers=8,
                preload_data=True,
                preload_model=False,  # Set to False if you want to train each model
                config_suffix=config_name  # This will create separate folders/models
            )
            
            # Store results
            results[config_name] = {
                'model': model,
                'latent_embeddings': latent_embeddings,
                'config': config,
                'latent_shape': latent_embeddings.shape,
                'latent_std': np.std(latent_embeddings, axis=0),
                'latent_mean': np.mean(latent_embeddings, axis=0)
            }
            
            print(f"✓ SUCCESS: {config_name} completed")
            print(f"   Latent shape: {latent_embeddings.shape}")
            print(f"   Output directory: {FPATH.FIGURES / MODEL_NAME}_{config_name}")
            
        except Exception as e:
            print(f"✗ ERROR: {config_name} failed with error: {str(e)}")
            results[config_name] = {'error': str(e), 'config': config}
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    successful_configs = [name for name, result in results.items() if 'error' not in result]
    failed_configs = [name for name, result in results.items() if 'error' in result]
    
    print(f"Successful configurations: {len(successful_configs)}/{len(configurations)}")
    for name in successful_configs:
        shape = results[name]['latent_shape']
        print(f"   - {name}: {shape}")
        
    if failed_configs:
        print(f"\n Failed configurations: {len(failed_configs)}")
        for name in failed_configs:
            print(f"   - {name}: {results[name]['error']}")
    
    print(f"\nResults saved in: {FPATH.FIGURES / MODEL_NAME}_<config_name>")
    print(f" Models saved as: best_vae_model.pth in each config directory")
    
    print("\n All VAE variant experiments completed!")
    
    # Uncomment to clean cache if needed
    # clean_cache(MODEL_NAME)