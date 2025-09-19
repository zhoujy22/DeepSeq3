import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
import torch_geometric as tg
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
from torch_geometric.utils import add_self_loops, degree
#from torch_scatter import scatter_add
from torch_geometric.nn.glob import *
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class AggConv(MessagePassing):
    '''
    Modified based on GCNConv implementation in PyG.
    This version includes proper degree normalization, making it a true GCN.
    '''
    def __init__(self, in_channels, out_channels=None, wea=False, mlp=None, reverse=False):
        # We still use 'add' aggregation, but now we apply normalization in the message() method.
        super().__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')
        
        if out_channels is None:
            out_channels = in_channels
        assert (in_channels > 0) and (out_channels > 0), 'The dimension for the AggConv should be larger than 0.'

        self.wea = wea
        
        # A single linear layer for message transformation, as in GCN.
        self.msg = nn.Linear(in_channels, out_channels) if mlp is None else mlp

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        # 1. Add self-loops to the graph to include node's own features
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # 2. Compute normalization. This is the key GCN step.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 3. Propagate messages. The normalization factor will be passed to the message() function.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(self, x_j, norm, edge_attr=None):
        # x_j has shape [E, in_channels]
        # norm has shape [E]

        # Apply the linear transformation to the message.
        msg_out = self.msg(x_j)

        # Apply normalization to the message before aggregation.
        # This is the core GCN formula.
        return norm.view(-1, 1) * msg_out

    def update(self, aggr_out):
        return aggr_out