import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, max_length, dropout_rate=0.2):
        """
        vocab_size   : Number of unique characters.
        embedding_dim: Embedding dimension.
        hidden_dim   : Hidden layer size.
        latent_dim   : Latent space dimension.
        max_length   : Maximum sequence length.
        dropout_rate : Dropout probability.
        """
        super(VAE, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_length = max_length

        # Encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder_fc = nn.Linear(max_length * embedding_dim, hidden_dim)
        self.encoder_bn = nn.BatchNorm1d(hidden_dim)
        self.encoder_dropout = nn.Dropout(dropout_rate)
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_bn = nn.BatchNorm1d(hidden_dim)
        self.decoder_dropout = nn.Dropout(dropout_rate)
        self.decoder_fc2 = nn.Linear(hidden_dim, max_length * vocab_size)

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def encode(self, x):
        # x shape: (batch, max_length)
        x_emb = self.embedding(x)  # (batch, max_length, embedding_dim)
        x_flat = x_emb.view(x.size(0), -1)  # (batch, max_length * embedding_dim)
        h = F.relu(self.encoder_fc(x_flat))
        h = self.encoder_bn(h)
        h = self.encoder_dropout(h)
        z_mean = self.encoder_mean(h)
        z_log_var = self.encoder_log_var(h)
        
        if self.training:
            z = self.reparameterize(z_mean, z_log_var)
        else:
            z = z_mean # During eval, I do not want the model to return a random sample from the distribution
            
        return z, z_mean, z_log_var

    def decode(self, z):
        h_dec = F.relu(self.decoder_fc1(z))
        h_dec = self.decoder_bn(h_dec)
        h_dec = self.decoder_dropout(h_dec)
        out = self.decoder_fc2(h_dec)  # (batch, max_length * vocab_size)
        out = out.view(-1, self.max_length, self.vocab_size)
        return out

    def forward(self, x):
        z, z_mean, z_log_var = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z_mean, z_log_var