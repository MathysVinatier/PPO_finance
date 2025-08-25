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
# Transformer Encoder for Stock Data
# -----------------------------
class Encoder_Transformer(nn.Module):
    def __init__(self, num_features, d_model=64, num_heads=8, num_classes=2):
        """
        num_features: number of input stock features per timestep
        d_model: embedding dimension for transformer
        num_heads: number of attention heads
        num_classes: output classes (for classification)
        """
        super().__init__()

        # Embedding layer to project raw stock features
        self.Embedded = StockEmbedder(num_features, d_model)

        # Multi-Head Attention
        self.MHAttention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )

        # Add & Norm after attention
        self.AddNorm_MHAttention = AddNorm(d_model=d_model)

        # Feed-Forward network
        self.FeedForward = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )

        # Add & Norm after feed-forward
        self.AddNorm_FeedForward = AddNorm(d_model=d_model)

        # Final classification layer
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, num_features)
        x = self.Embedded(x)  # Embed stock features

        # Multi-Head Attention
        attn_output, _ = self.MHAttention(x, x, x)
        # Add & Norm after attention
        x = self.AddNorm_MHAttention(x, attn_output)

        # Feed-Forward network
        ff_output = self.FeedForward(x)
        # Add & Norm after feed-forward
        x = self.AddNorm_FeedForward(x, ff_output)

        # Pool across the sequence dimension (average pooling)
        pooled = x.mean(dim=1)

        # Final classification logits
        logits = self.fc(pooled)
        return logits


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Initialize model
    model = Encoder_Transformer(num_features=512, d_model=512, num_heads=8, num_classes=3).to(DEVICE)
    print(model)

    # Fake input: batch of 2 sequences, 10 timesteps, 512 features
    X = torch.rand(2, 10, 512, device=DEVICE)

    # Forward pass
    logits = model(X)

    # Convert logits to probabilities and predicted class
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)

    print("Logits :", logits)
    print("Predicted class :", y_pred)
