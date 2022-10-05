import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class SymbolicEmbeddingsGumbel(nn.Module):
    def __init__(self, n_categories, n_symbols, pattern_length, symbol_dim, mode='concat'):
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

    def init_weights(self):
        self.symbols.weight.uniform_(-sqrt(3), sqrt(3))
        self.symbols.pattern_map.uniform_(-sqrt(3), sqrt(3))    #this may need a look

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
        self.n_categories = n_categories
        self.n_symbols = n_symbols
        self.pattern_length = pattern_length
        self.symbol_dim = symbol_dim
        self.mode = mode

        self.tau = nn.Parameter(torch.Tensor([1.]))

        self.symbols = nn.Parameter(torch.empty(n_symbols, symbol_dim))
        self.latents = nn.Parameter(torch.empty(n_categories, pattern_length, symbol_dim))

        init.normal_(self.symbols.weight)
        init.normal_(self.latents.weight)

    def reinitialize(self, indices):
        with torch.no_grad():
            init.normal_(self.latents[indices])

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