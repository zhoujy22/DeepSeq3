import torch.nn as nn
import torch

"""Pooling Transformer model for graph embeddings."""
class PoolingTransformer(nn.Module):
    def __init__(self, dim_hidden, num_heads=4, num_layers=1):
        super(PoolingTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden,
            nhead=num_heads,
            dim_feedforward=dim_hidden * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.query_token = nn.Parameter(torch.zeros(1, 1, dim_hidden))  # learnable [CLS] token
        nn.init.trunc_normal_(self.query_token, std=0.02)

    def forward(self, x):
        """
        x: [N, dim_hidden] (node embeddings)
        return: [1, dim_hidden] (graph embedding)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, dim_hidden]
        query = self.query_token.expand(x.size(0), -1, -1)  # [N, 1, dim_hidden]
        x = torch.cat([query, x], dim=1)  # [1, N+1, dim_hidden]
        x = self.encoder(x)  # [1, N+1, dim_hidden]
        return x[:, 0, :]    # [1, dim_hidden] — 取 query token