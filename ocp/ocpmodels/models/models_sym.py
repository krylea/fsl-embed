
from chem.model_utils import SymbolicEmbeddingBlock
from ocpmodels.common.registry import registry
from ocpmodels.models import DimeNetPlusPlus, DimeNetPlusPlusWrap
from torch_geometric.nn.acts import swish

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

@registry.register_model("dimenetplusplus-sym")
class DimeNetPlusPlusSymbolic(DimeNetPlusPlusWrap):
    def __init__(self,
        n_symbols, 
        pattern_length, 
        symbol_dim,
        num_atoms,
        bond_feat_dim,  # not used
        num_targets,
        use_pbc=True,
        regress_forces=True,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        otf_graph=False,
        cutoff=10.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=swish,
        mode='concat'
    ):
        super().__init__(
            num_atoms,
            bond_feat_dim,  # not used
            num_targets,
            use_pbc=use_pbc,
            regress_forces=regress_forces,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            otf_graph=otf_graph,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers
        )
        self.emb = SymbolicEmbeddingBlock(n_symbols, pattern_length, hidden_channels // pattern_length, num_radial, act, mode=mode)