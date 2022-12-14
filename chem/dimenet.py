
from chem.model_utils import SymbolicEmbeddingBlock
from ocpmodels.common.registry import registry
from ocpmodels.models import DimeNetPlusPlus, DimeNetPlusPlusWrap
from torch_geometric.nn.acts import swish

@registry.register_model("dimenetplusplus-sym")
class DimeNetPlusPlusSymbolic(DimeNetPlusPlusWrap):
    def __init__(self,
        n_symbols, 
        pattern_length, 
        symbol_dim
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
            hidden_channels=pattern_length * symbol_dim,
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
        self.emb = SymbolicEmbeddingBlock(n_symbols, pattern_length, symbol_dim, num_radial, act, mode=mode)