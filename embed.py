import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from math import sqrt

from utils import knn

class SymbolicEmbeddingsGumbel(nn.Module):
    def __init__(self, n_categories, n_symbols, pattern_length, symbol_dim, mode='concat'):
        super().__init__()
        self.n_categories = n_categories
        self.n_symbols = n_symbols
        self.pattern_length = pattern_length
        self.symbol_dim = symbol_dim
        self.mode = mode

        if self.mode == 'concat':
            self.dim = symbol_dim * pattern_length
        elif self.mode == 'split':
            self.dim = symbol_dim
        else:
            raise NotImplementedError("concat or split")

        self.tau = nn.Parameter(torch.Tensor([1.]))

        self.symbols = nn.Parameter(torch.empty(n_symbols, symbol_dim))
        self.pattern_map = nn.Parameter(torch.empty(n_categories, pattern_length, n_symbols))

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.symbols.uniform_(-sqrt(3), sqrt(3))
            self.pattern_map.uniform_(-sqrt(3), sqrt(3))    #this may need a look

    def reinitialize(self, indices):
        with torch.no_grad():
            init.uniform_(self.pattern_map[indices], -sqrt(3), sqrt(3)) #this may need a look

    def forward(self, inputs):
        energies = self.pattern_map[inputs]
        weights = F.gumbel_softmax(energies, hard=True, tau=self.tau)

        embeds = torch.matmul(weights, self.symbols)

        if self.mode == 'concat':
            embeds = embeds.view(*inputs.size(), -1)
        elif self.mode == 'split':
            embeds = embeds.view(*inputs.size()[:-1], -1, self.symbol_dim)
        else:
            raise NotImplementedError("concat or split")

        return embeds


class SymbolicEmbeddingsVQ(nn.Module):
    def __init__(self, n_categories, n_symbols, pattern_length, symbol_dim, mode='concat'):
        super().__init__()
        self.n_categories = n_categories
        self.n_symbols = n_symbols
        self.pattern_length = pattern_length
        self.symbol_dim = symbol_dim
        self.mode = mode

        self.tau = nn.Parameter(torch.Tensor([1.]))

        self.symbols = nn.Parameter(torch.empty(n_symbols, symbol_dim))
        self.latents = nn.Parameter(torch.empty(n_categories, pattern_length, symbol_dim))

        self.register_buffer("pattern", torch.ones(n_categories, pattern_length, dtype=torch.long) *-1)
        self.register_buffer("symbol_loss_buffer", torch.tensor([0]))

        self.init_weights()

        #self.update_pattern()

    def init_weights(self):
        with torch.no_grad():
            self.symbols.uniform_(-sqrt(3), sqrt(3))
            self.latents.uniform_(-sqrt(3), sqrt(3))    #this may need a look

    def reinitialize(self, indices):
        with torch.no_grad():
            init.normal_(self.latents[indices])

    def update_pattern(self, indices=None):
        if indices is not None:
            self.pattern[indices] = knn(self.latents[indices], self.symbols.unsqueeze(0), 1).squeeze(-1)
        else:
            self.pattern = knn(self.latents, self.symbols.unsqueeze(0), 1).squeeze(-1)
        self.symbol_loss_buffer = torch.tensor([0])

    def forward(self, inputs):
        self.update_pattern(indices=inputs.unique())

        latent_embeds = self.latents[inputs]
        discrete_embeds = self.symbols[self.pattern[inputs]]
        embeds = discrete_embeds.detach() + latent_embeds - latent_embeds.detach()

        if self.training:
            self.symbol_loss_buffer += (discrete_embeds - latent_embeds.detach()).pow(2).sum(dim=-1).mean()

        if self.mode == 'concat':
            embeds = embeds.view(*inputs.size(), -1)
        elif self.mode == 'split':
            embeds = embeds.view(*inputs.size()[:-1], -1, self.symbol_dim)
        else:
            raise NotImplementedError("concat or split")

        return embeds