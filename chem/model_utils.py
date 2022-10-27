from embed import SymbolicEmbeddingsGumbel, SymbolicEmbeddingsVQ
import torch.nn as nn

class SymbolicEmbeddingBlock(nn.Module):
    def __init__(self, n_symbols, pattern_length, symbol_dim, num_radial, act, mode='concat'):
        super().__init__()
        self.act = act

        self.emb = SymbolicEmbeddingsVQ(128, n_symbols, pattern_length, symbol_dim, mode='concat')
        hidden_channels = self.emb.dim
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.init_weights()
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, rbf, i, j):
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))