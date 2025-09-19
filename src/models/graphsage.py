import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops
class SAGEAggregator(MessagePassing):
    """
    带边特征的 GraphSAGE 均值聚合器 (PyG 实现)。
    节点更新公式：
    h_i' = W_self * x_i + W_neigh * mean_{j∈N(i)}( concat(x_j, e_{ji}) )
    """
    def __init__(self, in_channels, out_channels, edge_dim=None, dropout=0.5):
        super(SAGEAggregator, self).__init__(aggr='mean')

        # 如果有边特征，邻居线性层的输入 = 节点特征 + 边特征
        neigh_in = in_channels + (edge_dim if edge_dim is not None else 0)
        self.lin_neigh = nn.Linear(neigh_in, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        num_nodes = x.size(0)

        

        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        msg = x_j
        return self.lin_neigh(msg)

    def update(self, aggr_out, x):
        # 加上自身特征的映射
        out = aggr_out + self.lin_self(x.clone())
        return self.dropout(out)