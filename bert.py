import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert (
            hidden_size % num_heads == 0
        ), "Hidden size must be divisible by number of heads"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(
            hidden_size, hidden_size * 3
        )  # Combined Q, K, V projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V in a single operation
        qkv = self.qkv_proj(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        Q, K, V = qkv.permute(2, 0, 3, 1, 4)  # Split and rearrange dimensions

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            scores.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        out = (
            torch.matmul(attention, V).transpose(1, 2).reshape(batch_size, seq_len, -1)
        )

        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        x = self.norm1(x + self.dropout(self.attention(x, attention_mask)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class BERTClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        ff_dim,
        num_classes=2,
        max_len=512,
        dropout=0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_size))

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_size, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids) + self.pos_embedding[:, : input_ids.size(1)]

        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        return self.fc(self.dropout(x[:, 0, :]))  # CLS token representation
