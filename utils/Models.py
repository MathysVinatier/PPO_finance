import torch 
from torch import nn

# Select device: GPU if available, otherwise CPU
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# -----------------------------
# Multi-Layer Perceptron (MLP)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 128)):
        """
        input_dim: dimension of input features
        output_dim: dimension of output features
        hidden_dims: tuple of hidden layer sizes
        """
        super().__init__()
        layers = []
        last_dim = input_dim
        # Build hidden layers with ReLU activations
        for h in hidden_dims:
            layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        # Final linear layer (no activation)
        layers += [nn.Linear(last_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through all layers
        return self.net(x)


# -----------------------------
# FeedForward Network
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=10):
        """
        A simple feedforward network with two MLPs and a ReLU in between.
        """
        super().__init__()
        self.flatten = nn.Flatten()  # Flatten input if it has extra dimensions
        self.linear_relu_stack = nn.Sequential(
            MLP(input_dim=input_dim, output_dim=hidden_dim),
            nn.ReLU(),
            MLP(input_dim=hidden_dim, output_dim=output_dim)
        )

    def forward(self, x):
        # Flatten and pass through the stack
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# -----------------------------
# Stock Embedder
# -----------------------------
class StockEmbedder(nn.Module):
    def __init__(self, num_features, d_model)->None:
        """
        Projects raw stock features (OHLCV, etc.) into a d_model dimensional embedding space.
        """
        super().__init__()
        self.linear = nn.Linear(num_features, d_model)

    def forward(self, x)->torch.Tensor:
        return self.linear(x)


# -----------------------------
# Add & Norm Layer
# -----------------------------
class AddNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6)->None:
        """
        Implements a residual connection followed by layer normalization.
        """
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x, sublayer_output)->torch.Tensor:
        # Residual connection + LayerNorm
        return self.norm(x + sublayer_output)

# -----------------------------
# Time2Vec
# -----------------------------
class Time2Vec(nn.Module):
    def __init__(self, kernel_size: int):
        """
        kernel_size: number of time features to generate
        """
        super().__init__()
        self.freq = nn.Linear(1, kernel_size - 1)
        self.phase = nn.Linear(1, kernel_size - 1)
        self.linear = nn.Linear(1, 1)

    def forward(self, t):
        """
        t: tensor of shape (batch, seq_len, 1)
        """
        linear_term = self.linear(t)
        periodic_term = torch.sin(self.freq(t) + self.phase(t))
        return torch.cat([linear_term, periodic_term], dim=-1)  # (batch, seq_len, kernel_size)


# -----------------------------
# Transformer Encoder for Stock Data
# -----------------------------
class PPO_Transformer(nn.Module):
    def __init__(self, num_features, d_model=64, num_heads=8, num_classes=2, time_dim=8):
        """
        num_features: number of input stock features per timestep
        d_model: embedding dimension for transformer
        num_heads: number of attention heads
        num_classes: number of output classes (buy/sell)
        time_dim: size of time2vec encoding
        """
        super().__init__()

        # --- Time encoding ---
        self.time2vec = Time2Vec(time_dim)

        # --- Embedding layer to project raw + time features ---
        self.Embedded = StockEmbedder(num_features + time_dim, d_model)

        # --- Multi-Head Attention ---
        self.MHAttention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )

        # --- Add & Norm after attention ---
        self.AddNorm_MHAttention = AddNorm(d_model=d_model)

        # --- Feed-Forward network ---
        self.FeedForward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        # --- Add & Norm after feed-forward ---
        self.AddNorm_FeedForward = AddNorm(d_model=d_model)

        # --- Final classification layer ---
        self.fc = nn.Linear(d_model, num_classes)

        # --- Output probability activation (for buy/sell probs) ---
        self.prob_layer = nn.Softmax()

    def forward(self, x):
        """
        x: (batch, seq_len, num_features)
        """
        batch_size, seq_len, _ = x.shape

        # Generate continuous time indices for Time2Vec
        t = torch.arange(seq_len, device=x.device).float().unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
        t_encoded = self.time2vec(t)

        # Concatenate time features with original market features
        x = torch.cat([x, t_encoded], dim=-1)

        # Embed the combined features
        x = self.Embedded(x)

        # Multi-Head Attention block
        attn_output, _ = self.MHAttention(x, x, x)
        x = self.AddNorm_MHAttention(x, attn_output)

        # Feed-forward block
        ff_output = self.FeedForward(x)
        x = self.AddNorm_FeedForward(x, ff_output)

        # Pool across sequence dimension (mean pooling)
        pooled = x.mean(dim=1)

        # Classification head
        logits = self.fc(pooled)

        # Convert to probability (range [0, 1])
        probs = self.prob_layer(logits)

        return probs  # shape: (batch, 2)

# -----------------------------
# Example usage for PPO_Transformer
# -----------------------------
if __name__ == "__main__":

    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from Environment import TradingEnv, DataLoader

    # -----------------------------
    # Setup
    # -----------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    df = DataLoader().read("data/General/TSLA_2019_2024.csv")

    # Select and normalize features
    features = df[["Close", "High", "Low", "Open", "Volume"]].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = torch.tensor(features, dtype=torch.float32, device=DEVICE)

    # -----------------------------
    # Create sliding window sequences
    # -----------------------------
    seq_len = 7
    X_seq = []
    for i in range(len(features) - seq_len):
        X_seq.append(features[i:i + seq_len])  # shape: (seq_len, num_features)

    X_seq = torch.stack(X_seq)  # (num_samples, seq_len, num_features)
    print(f"X_seq shape: {X_seq.shape}")  # e.g., (N, 10, 5)

    # -----------------------------
    # Initialize PPO Transformer Model
    # -----------------------------
    model = PPO_Transformer(
        num_features=X_seq.shape[2],    # input features per timestep
        d_model=64,        # transformer embedding size
        num_heads=8,       # attention heads
        num_classes=2,     # buy/sell
        time_dim=8         # time2vec dimension
    ).to(DEVICE)

    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # -----------------------------
    # Forward Pass Example
    # -----------------------------
    # Take a small batch of random sequences
    X_batch = X_seq[:7].to(DEVICE)  # shape: (batch, seq_len, num_features)

    # Forward pass through the model
    probs = model(X_batch)

    print(f"Output shape: {probs.shape}")  # (batch, 2)
    print("Predicted probabilities (buy/sell):")
    print(probs)

    # -----------------------------
    # Example: PPO-style action sampling
    # -----------------------------
    dist = torch.distributions.Categorical(probs)
    actions = dist.sample()

    # probs shape: (batch, 2) -> [p_buy, p_sell]
    threshold = 0.65

    # Find the most likely action (0=buy, 1=sell)
    max_probs, max_actions = torch.max(probs, dim=1)

    # Initialize all actions as "hold" (0)
    actions = torch.zeros_like(max_actions)  # default hold (can use 0 for hold if you prefer)
    hold_label = 0
    buy_label = 1
    sell_label = 2

    # Apply threshold rule
    actions = torch.where(max_probs > threshold, max_actions + 1, hold_label * torch.ones_like(max_actions))

    print("Probabilities :\n", probs)
    print("Max probs : ", max_probs)
    print("Selected actions : ", actions)
