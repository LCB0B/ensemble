import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, Beta, kl_divergence
import torch.nn.functional as F

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

# ----- Dirichlet-Process VAE -----
class DPVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2, hidden_dim=128, trunc_K=20, dp_alpha=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.trunc_K = trunc_K
        self.dp_alpha = dp_alpha

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim * trunc_K)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim * trunc_K)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Variational sticks for DP
        self.var_a = nn.Parameter(torch.ones(trunc_K - 1))
        self.var_b = nn.Parameter(torch.ones(trunc_K - 1))
        self.prior_a = torch.ones(trunc_K - 1)
        self.prior_b = torch.ones(trunc_K - 1) * dp_alpha

    def _stick_weights(self, a, b):
        v = Beta(a, b).rsample()
        remaining = torch.cumprod(1 - v + 1e-10, dim=0)
        pis = torch.cat([v[:1], v[1:] * remaining[:-1], remaining[-1:]], dim=0)
        return pis

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu_all = self.fc_mu(h).view(-1, self.trunc_K, self.latent_dim)
        logvar_all = self.fc_logvar(h).view(-1, self.trunc_K, self.latent_dim)
        return mu_all, logvar_all

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_mix):
        return self.decoder(z_mix)

    def forward(self, x):
        mu_all, logvar_all = self.encode(x)
        z = self.reparameterize(mu_all, logvar_all)          # [batch, K, D]
        pis = self._stick_weights(self.var_a, self.var_b)     # [K]
        pis_expand = pis.unsqueeze(0).unsqueeze(-1)           # [1, K, 1]
        z_mix = (pis_expand * z).sum(dim=1)                   # [batch, D]
        x_hat = self.decode(z_mix)
        return x_hat, mu_all, logvar_all, pis, z_mix

    def loss(self, x, x_hat, mu_all, logvar_all, pis):
        recon = F.mse_loss(x_hat, x, reduction='sum')
        # KL z|x to N(0,I)
        q_z = Normal(mu_all, 0.5 * logvar_all.exp())
        p_z = Normal(torch.zeros_like(mu_all), torch.ones_like(logvar_all))
        kl_z = kl_divergence(q_z, p_z).sum()
        # KL sticks
        q_v = Beta(self.var_a, self.var_b)
        p_v = Beta(self.prior_a.to(self.var_a.device), self.prior_b.to(self.var_b.device))
        kl_v = kl_divergence(q_v, p_v).sum()
        return recon + kl_z + kl_v

# ----- Main processing -----
def main(embeddings_path: str, figures_dir: str = "figures",
         use_dpv: bool = False, dp_params: dict = None,
         batch_size: int = 64, epochs: int = 50, lr: float = 1e-3):
    os.makedirs(figures_dir, exist_ok=True)
    embeddings = np.load(embeddings_path)
    input_dim = embeddings.shape[1]

    if not use_dpv:
        load_and_plot_embeddings(embeddings, figures_dir)
    else:
        # Setup DPVAE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dp_params = dp_params or {}
        model = DPVAE(input_dim=input_dim, **dp_params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        dataset = TensorDataset(torch.from_numpy(embeddings).float())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train
        model.train()
        for epoch in range(1, epochs+1):
            total_loss = 0
            for (x_batch,) in loader:
                x_batch = x_batch.to(device)
                x_hat, mu_all, logvar_all, pis, _ = model(x_batch)
                loss = model.loss(x_batch, x_hat, mu_all, logvar_all, pis)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataset):.4f}")

        # Inference: get mixed z
        model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(embeddings).float().to(device)
            _, mu_all, logvar_all, pis, z_mix = model(x_tensor)
            z = z_mix.cpu().numpy()

        # Plot DPVAE latent
        plt.figure(figsize=(8, 6))
        plt.scatter(z[:, 0], z[:, 1], s=1, alpha=0.6, edgecolor='none')
        plt.title("DPVAE 2D Latent Space")
        plt.xlabel("Z1"); plt.ylabel("Z2")
        plt.tight_layout()
        dpv_fig = os.path.join(figures_dir, "dpvae_latent.png")
        plt.savefig(dpv_fig, dpi=300)
        plt.close()
        print(f"DPVAE latent space plot saved to {dpv_fig}")

if __name__ == "__main__":
    embeddings_path = r'/home/xc2/kdrev/22SSI/ensemble/lcbob/project2vec/embeddings/021_muddy_cobra-pretrain-lr0.0003/embeddings.npy'
    embeddings = np.load(embeddings_path)
    np.random.seed(42)
    n_samples = int(1e5)
    idx = np.random.choice(embeddings.shape[0], size=n_samples, replace=False)
    embeddings_sample = embeddings[idx]
    load_and_plot_embeddings(embeddings_sample)
    # main(
    #     embeddings_path=embedding_path,
    #     figures_dir="figures",
    #     use_dpv=True,  # set False to run PCA+UMAP instead
    #     dp_params={"latent_dim": 2, "hidden_dim": 128, "trunc_K": 20, "dp_alpha": 1.0},
    #     batch_size=64,
    #     epochs=50,
    #     lr=1e-3
    # )

