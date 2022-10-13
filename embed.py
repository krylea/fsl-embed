import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from math import sqrt

from utils import knn


class SymbolicEmbeddings(nn.Module):
    def __init__(self, n_categories, n_symbols, pattern_length, symbol_dim, mode='concat', augment_dim=-1):
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

        self.symbols = nn.Parameter(torch.empty(n_symbols, symbol_dim))
        self.augments = None
        if augment_dim > 0:
            self.augments = nn.Parameter(torch.empty(n_categories, augment_dim))
            self.dim += augment_dim

        self.build_pattern(n_categories, n_symbols, pattern_length, symbol_dim)

        self.init_weights()

    def build_pattern(self, n_categories, n_symbols, pattern_length, symbol_dim, **kwargs):
        pass

    def init_weights():
        pass

    def _get_symbols(inputs):
        pass

    def forward(self, inputs):
        embeds = self._get_symbols(inputs)  # bs x l x p x d_s

        if self.mode == 'concat':
            embeds = embeds.view(*inputs.size(), -1)    # bs x l x d
            if self.augments is not None:
                augments = self.augments[inputs]
                embeds = torch.cat([embeds, augments], dim=-1)
        elif self.mode == 'split':
            embeds = embeds.view(*inputs.size()[:-1], -1, self.symbol_dim)
        else:
            raise NotImplementedError("concat or split")

        return embeds




class SymbolicEmbeddingsGumbel(SymbolicEmbeddings):
    def __init__(self, n_categories, n_symbols, pattern_length, symbol_dim, mode='concat', augment_dim=-1, hard=True):
        super().__init__(n_categories, n_symbols, pattern_length, symbol_dim, mode, augment_dim)
        self.hard = hard
        self.tau = nn.Parameter(torch.Tensor([1.]))

    def build_pattern(self, n_categories, n_symbols, pattern_length, symbol_dim, **kwargs):
        self.pattern_map = nn.Parameter(torch.empty(n_categories, pattern_length, n_symbols))

    def init_weights(self):
        with torch.no_grad():
            self.symbols.uniform_(-sqrt(3), sqrt(3))
            self.pattern_map.uniform_(-sqrt(3), sqrt(3))    #this may need a look

    def _get_symbols(self, inputs):
        energies = self.pattern_map[inputs]
        weights = F.gumbel_softmax(energies, hard=self.hard, tau=self.tau)
        embeds = torch.matmul(weights, self.symbols)
        return embeds


class SymbolicEmbeddingsVQ(SymbolicEmbeddings):
    def __init__(self, n_categories, n_symbols, pattern_length, symbol_dim, mode='concat', augment_dim=-1, beta=1.):
        super().__init__(n_categories, n_symbols, pattern_length, symbol_dim, mode, augment_dim)
        self.beta = beta
        #self.update_pattern()

    def build_pattern(self, n_categories, n_symbols, pattern_length, symbol_dim, **kwargs):
        self.latents = nn.Parameter(torch.empty(n_categories, pattern_length, symbol_dim))
        self.register_buffer("pattern", torch.ones(n_categories, pattern_length, dtype=torch.long) *-1)
        self.register_buffer("symbol_loss_buffer", torch.zeros([]))
        self.register_buffer("latent_loss_buffer", torch.zeros([]))

    def init_weights(self):
        with torch.no_grad():
            self.symbols.uniform_(-sqrt(3), sqrt(3))
            self.latents.uniform_(-sqrt(3), sqrt(3))    #this may need a look

    def update_pattern(self, indices=None):
        if indices is not None:
            self.pattern[indices] = knn(self.latents[indices], self.symbols.unsqueeze(0), 1).squeeze(-1)
        else:
            self.pattern = knn(self.latents, self.symbols.unsqueeze(0), 1).squeeze(-1)
        self.symbol_loss_buffer = torch.zeros([]).to(self.symbol_loss_buffer.device)
        self.latent_loss_buffer = torch.zeros([]).to(self.latent_loss_buffer.device)

    def _get_symbols(self, inputs):
        self.update_pattern(indices=inputs.unique())

        latent_embeds = self.latents[inputs]
        discrete_embeds = self.symbols[self.pattern[inputs]]
        embeds = discrete_embeds.detach() + latent_embeds - latent_embeds.detach()

        if self.training:
            self.symbol_loss_buffer += (discrete_embeds - latent_embeds.detach()).pow(2).mean()
            self.latent_loss_buffer += self.beta * (discrete_embeds.detach() - latent_embeds).pow(2).mean()

        return embeds