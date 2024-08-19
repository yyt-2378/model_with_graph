from transformers.configuration_utils import PretrainedConfig
import torch


class GraphAttentionTransformerOC20Config(PretrainedConfig):
    model_type = 'graph_attention_transformer_oc20'

    def __init__(self,
                 irreps_node_embedding='256x0e+128x1e',
                 num_layers=6,
                 irreps_node_attr='1x0e',
                 use_node_attr=False,
                 irreps_sh='1x0e+1x1e',
                 max_radius=5.0,
                 number_of_basis=128,
                 fc_neurons=[64, 64],
                 use_atom_edge_attr=False,
                 irreps_atom_edge_attr='1x0e',
                 irreps_feature='512x0e',
                 irreps_head='32x0e+16x1e',
                 num_heads=8,
                 irreps_pre_attn='256x0e+128x1e',
                 rescale_degree=False,
                 nonlinear_message=True,
                 irreps_mlp_mid='768x0e+384x1e',
                 norm_layer='layer',
                 alpha_drop=0.2,
                 proj_drop=0.0,
                 out_drop=0.0,
                 drop_path_rate=0.0,
                 otf_graph=True,
                 use_pbc=True,
                 max_neighbors=500,
                 checkpoint_path='/mnt/petrelfs/yangyaotian/project/equiformer/oc20_weights/checkpoints/2022-04-18-03-14-08/best_checkpoint.pt',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.irreps_node_embedding = irreps_node_embedding
        self.num_layers = num_layers
        self.irreps_node_attr = irreps_node_attr
        self.use_node_attr = use_node_attr
        self.irreps_sh = irreps_sh
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.fc_neurons = fc_neurons
        self.use_atom_edge_attr = use_atom_edge_attr
        self.irreps_atom_edge_attr = irreps_atom_edge_attr
        self.irreps_feature = irreps_feature
        self.irreps_head = irreps_head
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = irreps_mlp_mid
        self.norm_layer = norm_layer
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc
        self.max_neighbors = max_neighbors
        # todo: check how to merge model weights
        self.checkpoint_path = checkpoint_path
        self.device = device
