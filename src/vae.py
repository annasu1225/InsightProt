import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # Learn a global log-variance for Gaussian reconstruction (diagonal, shared across dims)
        self.recon_logvar = nn.Parameter(torch.tensor(0.0))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar

    def elbo_loss(self, x, beta: float = 1.0):
        xhat, mu, logvar = self.forward(x)

        # Gaussian negative log-likelihood (up to constant)
        # NLL = 0.5 * sum( (x-xhat)^2 / sigma^2 + log sigma^2 )
        sigma2 = torch.exp(self.recon_logvar).clamp_min(1e-6)
        recon = 0.5 * (((x - xhat) ** 2) / sigma2 + self.recon_logvar).sum(dim=1).mean()

        # KL(q(z|x) || N(0,I))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        loss = recon + beta * kl
        return loss, recon.detach(), kl.detach()

    @torch.no_grad()
    def nll_per_sample(self, x) -> torch.Tensor:
        """
        Approximate per-sample negative log-likelihood proxy using reconstruction NLL at posterior mean.
        """
        mu, logvar = self.encode(x)
        z = mu
        xhat = self.decode(z)
        sigma2 = torch.exp(self.recon_logvar).clamp_min(1e-6)
        nll = 0.5 * (((x - xhat) ** 2) / sigma2 + self.recon_logvar).sum(dim=1)
        return nll
    
