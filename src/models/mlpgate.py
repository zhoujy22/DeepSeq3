from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, Optional

import torch
import math
from torch import nn
from utils.dag_utils import subgraph, custom_backward_subgraph
from utils.utils import generate_hs_init
import torch.nn.functional as F
from .mlp import MLP
from .mlp_aggr import MlpAggr
from .gat_conv import AGNNConv
from .gcn_conv import AggConv
from .graphsage import SAGEAggregator
from .deepset_conv import DeepSetConv
from .gated_sum_conv import GatedSumConv
from .aggnmlp import AttnMLP
from .tfmlp import TFMLP
from .PoolTransformer import PoolingTransformer
import torch_geometric.nn as gnn
from torch_geometric.nn.attention import PerformerAttention
from .SGFormer import SGFormer
from .DiffFormer import DIFFormer_v2
from torch.nn import LSTM, GRU
import sys 
_aggr_function_factory = {
    'mlp': MlpAggr,             # MLP, similar as NeuroSAT  
    'attnmlp': AttnMLP,         # MLP with attention
    'tfmlp': TFMLP,             # MLP with transformer
    'aggnconv': AGNNConv,       # DeepGate with attention
    'conv_sum': AggConv,        # GCN
    'deepset': DeepSetConv,     # DeepSet, similar as NeuroSAT  
    'GraphSAGE': SAGEAggregator, # GraphSAGE
} 

_update_function_factory = {
    'lstm': LSTM,
    'gru': GRU,
}

def dec_to_bin_array(num, N):
    """
    把十进制数转成长度为 N 的二进制数组
    :param num: 十进制数
    :param N: 数组长度 (寄存器个数)
    :return: [0,1,...] 的数组
    """
    return [int(x) for x in format(num, f'0{N}b')]

class MLPGate(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits.
    '''
    def __init__(self, args):
        super(MLPGate, self).__init__()
        
        self.args = args

         # configuration
        self.num_rounds = args.num_rounds
        self.device = args.device
        self.predict_diff = args.predict_diff
        self.intermediate_supervision = args.intermediate_supervision
        self.reverse = args.reverse
        self.custom_backward = args.custom_backward
        self.use_edge_attr = args.use_edge_attr
        self.mask = args.mask

        # dimensions
        self.num_aggr = args.num_aggr
        self.dim_node_feature = args.dim_node_feature
        self.dim_hidden = args.dim_hidden
        self.dim_mlp = args.dim_mlp
        self.dim_pred = args.dim_pred
        self.num_fc = args.num_fc
        self.wx_update = args.wx_update
        self.wx_mlp = args.wx_mlp
        self.dim_edge_feature = args.dim_edge_feature

        # Network 
        if args.aggr_function == 'mlp' or args.aggr_function == 'attnmlp':
            self.aggr_and_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*1, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
            self.aggr_and_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
            self.aggr_and_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*3, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
            self.aggr_not_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*1, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
            self.aggr_not_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
            self.aggr_not_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*3, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
            self.aggr_ff_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*1, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
            
        else:
            self.aggr_and_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*1, self.dim_hidden)
            self.aggr_and_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, self.dim_hidden)
            self.aggr_and_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*3, self.dim_hidden)
            self.aggr_not_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*1, self.dim_hidden)
            self.aggr_not_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, self.dim_hidden)
            self.aggr_not_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*3, self.dim_hidden)
            self.aggr_ff_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*1, self.dim_hidden)
            
        self.update_and_strc = GRU(self.dim_hidden, self.dim_hidden)
        self.update_and_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_and_seq = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_strc = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_seq = GRU(self.dim_hidden, self.dim_hidden)
        self.update_ff_strc = GRU(self.dim_hidden, self.dim_hidden)
        # Linear transformation for FF
        

        if self.args.lt_ff:
            self.linear_transform_ff_func = MLP(self.dim_hidden, args.dim_hidden, args.dim_hidden, num_layer=2)
            self.linear_transform_ff_seq = MLP(self.dim_hidden, args.dim_hidden, args.dim_hidden, num_layer=2)

        else:
            if args.aggr_function == 'mlp' or args.aggr_function == 'attnmlp':
                self.aggr_ff_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
                self.aggr_ff_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*3, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
            else:
                self.aggr_ff_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, args.dim_hidden)
                self.aggr_ff_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*3, args.dim_hidden)
        
            self.update_ff_func = GRU(self.dim_hidden, self.dim_hidden)
            self.update_ff_seq = GRU(self.dim_hidden, self.dim_hidden)

        # Readout 
        self.readout_prob0 = MLP(self.dim_hidden, args.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        self.readout_prob1 = MLP(self.dim_hidden, args.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
     
        self.readout_trans_prob = MLP(self.dim_hidden, args.dim_mlp, 2, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        

        # consider the embedding for the LSTM/GRU model initialized by non-zeros
        self.one = torch.ones(1).to(self.device)
        # self.hs_emd_int = nn.Linear(1, self.dim_hidden)
        self.hf_emd_int = nn.Linear(1, self.dim_hidden)
        self.hs_emd_int = nn.Linear(1, self.dim_hidden)
        self.hseq_emd_int = nn.Linear(1, self.dim_hidden)
        self.one.requires_grad = False
        sys.setrecursionlimit(100000)

        
    def forward(self, G):
   
        num_nodes = G.num_nodes
        num_layers_f = max(G.forward_level).item() + 1
        num_layers_b = max(G.backward_level).item() + 1
        max_sim=1
        
        # initialize the function hidden state
        hf_init = self.hf_emd_int(self.one).view(1, -1) # (1 x 1 x dim_hidden)
        hf_init = hf_init.repeat(num_nodes, 1) # (1 x num_nodes x dim_hidden)
        # inferring against a particular workload
        hf_init = self.imply_mask(G, hf_init, self.dim_hidden) 

        # initialize the function hidden state
        hs_init = self.hs_emd_int(self.one).view(1, -1) # (1 x 1 x dim_hidden)
        hs_init = hs_init.repeat(num_nodes, 1) # (1 x num_nodes x dim_hidden)
        # inferring against a particular workload
        hs_init = self.imply_mask(G, hs_init, self.dim_hidden) 

        # initialize the sequential hidden state
        hseq_init = self.hseq_emd_int(self.one).view(1, -1) # (1 x 1 x dim_hidden)
        hseq_init = hseq_init.repeat(num_nodes, 1) # (1 x num_nodes x dim_hidden)
        # inferring against a particular workload
        hseq_init = self.imply_seq_mask(G, hseq_init, self.dim_hidden) 

        preds = self._gru_forward(G, hs_init, hf_init,hseq_init, num_layers_f, num_layers_b)     # [N, D]
        return preds, max_sim
            
    def _gru_forward(self, G, hs_init, hf_init, hseq_init, num_layers_f, num_layers_b, h_true=None, h_false=None):
        G = G.to(self.device)
        
       
        x, edge_index= G.x, G.edge_index
    
        edge_attr = G.edge_attr

        hs = hs_init.to(self.device)
        hf = hf_init.to(self.device)

        #node_state_seq_not = torch.cat([hf, hseq], dim=-1)
  
        node_state = torch.cat([hs, hf], dim=-1)
     
        and_mask = G.gate.squeeze(1) == 1
        not_mask = G.gate.squeeze(1) == 2
        ff_mask  = G.gate.squeeze(1) == 3
        
        
        for _ in range(self.num_rounds):
            for level in range(0, num_layers_f):
                # forward layer
                layer_mask = G.forward_level == level

                # AND Gate
                l_and_node = G.forward_index[layer_mask & and_mask]
                if l_and_node.size(0) > 0:
                    and_edge_index, and_edge_attr = subgraph(l_and_node, edge_index, edge_attr, dim=1)
                    # Update structure hidden state
                    msg = self.aggr_and_strc(hs, and_edge_index,and_edge_attr)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hs_and = torch.index_select(hs, dim=0, index=l_and_node)
                    _, hs_and = self.update_and_strc(and_msg.unsqueeze(0), hs_and.unsqueeze(0))
                    hs = hs.clone()
                    hs[l_and_node, :] = hs_and.squeeze(0)

                    # Update function hidden state
                    msg = self.aggr_and_func(node_state, and_edge_index, and_edge_attr)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hf_and = torch.index_select(hf, dim=0, index=l_and_node)
                    _, hf_and = self.update_and_func(and_msg.unsqueeze(0), hf_and.unsqueeze(0))
                    hf = hf.clone()
                    hf[l_and_node, :] = hf_and.squeeze(0)

                        # Update sequential hidden state
                    # msg = self.aggr_and_seq(node_state_seq, and_edge_index, and_edge_attr) #function or structure or sequential?
                    # and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    # hseq_and = torch.index_select(hseq, dim=0, index=l_and_node)
                    
                    # _, hseq_and = self.update_and_seq(and_msg.unsqueeze(0), hseq_and.unsqueeze(0))
                    # #hseq = hseq.clone()
                    # hseq[l_and_node, :] = hseq_and.squeeze(0)

                # NOT Gate
                l_not_node = G.forward_index[layer_mask & not_mask]
                if l_not_node.size(0) > 0:
                    not_edge_index, not_edge_attr = subgraph(l_not_node, edge_index, edge_attr, dim=1)
                    # Update structure hidden state
                    msg = self.aggr_not_strc(hs, not_edge_index, not_edge_attr)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hs_not = torch.index_select(hs, dim=0, index=l_not_node)
                    _, hs_not = self.update_not_strc(not_msg.unsqueeze(0), hs_not.unsqueeze(0))
                    hs = hs.clone()
                    hs[l_not_node, :] = hs_not.squeeze(0)

                    # Update function hidden state
                    msg = self.aggr_not_func(node_state, not_edge_index, not_edge_attr)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hf_not = torch.index_select(hf, dim=0, index=l_not_node)
                    _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
                    hf = hf.clone()
                    hf[l_not_node, :] = hf_not.squeeze(0)

                    # Update sequential hidden state
                    # msg = self.aggr_not_seq(node_state_seq, not_edge_index, not_edge_attr)
                    # not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    # hseq_not = torch.index_select(hseq, dim=0, index=l_not_node)
                    # _, hseq_not = self.update_not_seq(not_msg.unsqueeze(0), hseq_not.unsqueeze(0))
                    # #hseq = hseq.clone()
                    # hseq[l_not_node, :] = hseq_not.squeeze(0)

                #FF Gate
                l_ff_fanin_node = G.forward_index[layer_mask & ff_mask]   
                if len(l_ff_fanin_node) > 0:    
                    ff_edge_index, ff_edge_attr = subgraph(l_ff_fanin_node, edge_index, edge_attr, dim=1)
                    # Update structure hidden state
                    msg = self.aggr_ff_strc(hs, ff_edge_index, ff_edge_attr)
                    ff_msg = torch.index_select(msg, dim=0, index=l_ff_fanin_node)
                    hs_ff = torch.index_select(hs, dim=0, index=l_ff_fanin_node)
                    _, hs_ff = self.update_ff_strc(ff_msg.unsqueeze(0), hs_ff.unsqueeze(0))
                    hs = hs.clone()
                    hs[l_ff_fanin_node, :] = hs_ff.squeeze(0)

                    # Update function hidden state
                    msg = self.aggr_ff_func(node_state, ff_edge_index, ff_edge_attr)
                    ff_msg = torch.index_select(msg, dim=0, index=l_ff_fanin_node)
                    hf_ff = torch.index_select(hf, dim=0, index=l_ff_fanin_node)
                    _, hf_ff = self.update_ff_func(ff_msg.unsqueeze(0), hf_ff.unsqueeze(0))
                    hf = hf.clone()
                    hf[l_ff_fanin_node, :] = hf_ff.squeeze(0)

                    # Update sequential hidden state
                    # msg = self.aggr_ff_seq(node_state_seq, ff_edge_index, ff_edge_attr)
                    # ff_msg = torch.index_select(msg, dim=0, index=l_ff_fanin_node)
                    # hseq_ff = torch.index_select(hseq, dim=0, index=l_ff_fanin_node)
                    # _, hseq_ff = self.update_ff_seq(ff_msg.unsqueeze(0), hseq_ff.unsqueeze(0))
                    # #hseq = hseq.clone()
                    # hseq[l_ff_fanin_node, :] = hseq_ff.squeeze(0)
                        
                            
                # Update node state
                node_state = torch.cat([hs, hf], dim=-1)


        node_embedding = node_state.squeeze(0)

        hs = node_embedding[:, :self.dim_hidden]
        hf = node_embedding[:, self.dim_hidden:]

        # Readout
        y_prob1 = self.readout_prob0(hf)
        y_prob0 = self.readout_prob1(hs)
        #return hs, hf
        return hs, hf,  y_prob1, y_prob0 #is_rc, 
    

    # workload initialization at PIs at function embedding
    def imply_mask(self, G, h, dim_hidden):

        # true_mask = (G.mask != -1.0).unsqueeze(0).to(self.device)
        # normal_mask = (G.mask == -1.0).unsqueeze(0).to(self.device)

        true_mask = (G.mask != -1).to(self.device)
        normal_mask = (G.mask == -1).to(self.device)

        #h_true = G.mask.unsqueeze(0).to(self.device)
        h_true = G.mask.to(self.device)
        h_true = h_true.expand(-1, dim_hidden).to(self.device)
      
        h_mask = h * normal_mask + h_true * true_mask
    
        return h_mask
    
    def imply_seq_mask(self, G, h, dim_hidden):

        # true_mask = (G.mask != -1.0).unsqueeze(0).to(self.device)
        # normal_mask = (G.mask == -1.0).unsqueeze(0).to(self.device)

        true_mask = (G.mask != -1).to(self.device)
        normal_mask = (G.mask == -1).to(self.device)
      
        h_true_1 = G.seq_mask[:,0].to(self.device)
        h_true_1 = h_true_1.reshape([len(h_true_1), 1])

        h_true_2 = G.seq_mask[:,1].to(self.device)
        h_true_2 = h_true_2.reshape([len(h_true_2), 1])

        size = int(dim_hidden/2)
        h_true_1 = h_true_1.expand(-1, size).to(self.device)
        h_true_2 = h_true_2.expand(-1, size).to(self.device)

        h_true = torch.cat([h_true_1, h_true_2], dim=-1)

        h_mask = h * normal_mask + h_true * true_mask
   
        return h_mask
    
class MLPGate2(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits.
    '''
    def __init__(self, args):
        super(MLPGate2, self).__init__()
        self.args = args
        # configuration
        self.num_rounds = args.num_rounds
        self.device = args.device
        self.predict_diff = args.predict_diff
        self.intermediate_supervision = args.intermediate_supervision
        self.reverse = args.reverse
        self.custom_backward = args.custom_backward
        self.use_edge_attr = args.use_edge_attr
        self.mask = args.mask

        # dimensions
        self.num_aggr = args.num_aggr
        self.dim_node_feature = args.dim_node_feature
        self.dim_hidden = args.dim_hidden
        self.dim_mlp = args.dim_mlp
        self.dim_pred = args.dim_pred
        self.num_fc = args.num_fc
        self.wx_update = args.wx_update
        self.wx_mlp = args.wx_mlp
        self.dim_edge_feature = args.dim_edge_feature

        # -------------------------------------Network define begin-------------------------------------
        self.pooling_transformer_query = PoolingTransformer(dim_hidden=self.dim_hidden)
        self.pooling_transformer_input = PoolingTransformer(dim_hidden=self.dim_hidden)
        self.pooling_transformer_query2 = PoolingTransformer(dim_hidden=self.dim_hidden)
        self.pooling_transformer_input2 = PoolingTransformer(dim_hidden=self.dim_hidden)
        self.forward_aggr1 = TFMLP(in_channels = self.dim_hidden*2, ouput_channels = self.dim_hidden*2, edge_attr_dim = 16) 
        self.backward_aggr1 = TFMLP(in_channels = self.dim_hidden*2, ouput_channels = self.dim_hidden*2, edge_attr_dim = 16, reverse = True)
    
        self.forward_update1 = GRU(self.dim_hidden*2, args.dim_hidden*2)
        self.backward_update1 = GRU(self.dim_hidden*2, args.dim_hidden*2)
        self.forward_aggr0 = TFMLP(in_channels = self.dim_hidden*2, ouput_channels = self.dim_hidden*2, edge_attr_dim = 16) 
        self.backward_aggr0 = TFMLP(in_channels = self.dim_hidden*2, ouput_channels = self.dim_hidden*2, edge_attr_dim = 16, reverse = True)
    
        self.forward_update0 = GRU(self.dim_hidden*2, args.dim_hidden*2)
        self.backward_update0 = GRU(self.dim_hidden*2, args.dim_hidden*2)
        # -------------------------------------Network define endin-------------------------------------

        # -------------------------------------Readout begin-------------------------------------
        self.readout = MLP(self.dim_hidden*2, args.dim_mlp , 1, num_layer=3, p_drop=0.20, act_layer='relu', norm_layer="layernorm")
        self.readout2 = MLP(self.dim_hidden*2, args.dim_mlp , 1, num_layer=3, p_drop=0.20, act_layer='relu', norm_layer="layernorm")
        # -------------------------------------Readout endin-------------------------------------
        self.cell_emd_initializer0 = nn.Linear(self.dim_hidden, self.dim_hidden*2)
        self.cell_emd_initializer1 = nn.Linear(self.dim_hidden, self.dim_hidden*2)
        
        
    def forward(self, G, finite_list_full, trans_matrics):
        cir_name = G.name
        S = finite_list_full[cir_name[0]][0].shape[0]
        # preds = torch.zeros(S, S).to(self.device)
        labels = torch.zeros(S, S).to(self.device)
        gt = torch.zeros(S, S).to(self.device)
        h0, h1, h0_trans, h1_trans= self._gru_forward(G)     # [N, D]
        N, D = h0.shape
        state_list = []
        for idx in range(S):
            state_bits = dec_to_bin_array(idx, N)   # 返回长度=N的0/1数组或tensor
            state_bits = torch.tensor(dec_to_bin_array(idx, N), dtype=torch.float32)
            state_list.append(state_bits)
        state_list = torch.stack(state_list, dim=0).to(self.device)  # [S, N]
        masks = (state_list == 0)                 # [S, N], bool
        selected = torch.where(
            masks.unsqueeze(-1),                   # [S, N, 1]
            h0.unsqueeze(0),                       # [1, N, D] -> broadcast
            h1.unsqueeze(0)                        # [1, N, D] -> broadcast
        )
        graph_embeds_query = self.pooling_transformer_query(selected) # [S, D]
        masks_input = (state_list == 0)  # [S, N]
        selected_input = torch.where(
            masks_input.unsqueeze(-1),
            h0.unsqueeze(0),
            h1.unsqueeze(0)
        )  # [S, N, D]
        graph_embeds_input = self.pooling_transformer_input(selected_input)  # [S, D]
        # logits = self.readout(graph_embeds_query).squeeze(-1)  # [S]
        # logits = torch.sigmoid(logits)  # [S]
        graph_embeds = torch.cat([
            graph_embeds_query.unsqueeze(0).expand(S, S, D),  # broadcast query
            graph_embeds_input.unsqueeze(1).expand(S, S, D)   # broadcast input
        ], dim=-1)  # [S, S, 2D]

        graph_embeds_flat = graph_embeds.view(S * S, -1)
        logits_flat = self.readout(graph_embeds_flat).squeeze(-1)  # [S*S]
        logits = logits_flat.view(S, S)  # [S, S]
        labels = (finite_list_full[cir_name[0]].to(self.device) > 0).float()  # [S, S]
        #labels = (finite_list_full[cir_name[0]][0] > 0).float().to(self.device)
        
        masks = (state_list == 0)                 # [S, N], bool
        selected = torch.where(
            masks.unsqueeze(-1),                   # [S, N, 1]
            h0_trans.unsqueeze(0),                       # [1, N, D] -> broadcast
            h1_trans.unsqueeze(0)                        # [1, N, D] -> broadcast
        )
        graph_embeds_query = self.pooling_transformer_query2(selected) # [S, D]
        masks_input = (state_list == 0)  # [S, N]
        selected_input = torch.where(
            masks_input.unsqueeze(-1),
            h0_trans.unsqueeze(0),
            h1_trans.unsqueeze(0)
        )  # [S, N, D]
        graph_embeds_input = self.pooling_transformer_input2(selected_input)  # [S, D]

        graph_embeds = torch.cat([
            graph_embeds_query.unsqueeze(0).expand(S, S, D),  # broadcast query
            graph_embeds_input.unsqueeze(1).expand(S, S, D)   # broadcast input
        ], dim=-1)  # [S, S, 2D]

        graph_embeds_flat = graph_embeds.view(S * S, -1)
        probs_flat = self.readout2(graph_embeds_flat).squeeze(-1)  # [S*S]
        probs = probs_flat.view(S, S)  # [S, S]
        probs = nn.functional.log_softmax(probs, dim=1)  # [S, S]
        gt = trans_matrics[cir_name[0]].to(self.device)
        return logits, labels, probs, gt
    def _gru_forward(self, G):
        G = G.to(self.device)
        x, edge_index= G.x, G.edge_index
        
        edge_attr = G.edge_attr
        forward_index = torch.LongTensor(range(G.forward_index.shape[0]))
        h_init1 = G.h_init1.to(self.device)
        h_init0 = G.h_init0.to(self.device)
        h0 = self.cell_emd_initializer0(h_init1)
        h1 = self.cell_emd_initializer1(h_init0)
        for _ in range(self.num_rounds):
            target_node = forward_index
            if target_node.size(0) > 0:
                # -------------------------get current edge---------------------- todo
                target_edge_index, target_edge_attr = edge_index, edge_attr 
                # ------------Update hidden state
                msg = self.forward_aggr0(h0, target_edge_index, target_edge_attr.float())
                target_node = target_node.to(msg.device)
                target_msg = torch.index_select(msg, dim=0, index=target_node)
                h_target = torch.index_select(h0, dim=0, index=target_node)
                _, h_target = self.forward_update0(target_msg.unsqueeze(0), h_target.unsqueeze(0))
                h0[target_node, :] = h_target.squeeze(0)

                # -------------------------get current edge----------------------
                target_edge_index, target_edge_attr = edge_index, edge_attr 
                # ------------Update hidden state
                msg = self.backward_aggr0(h0, target_edge_index, target_edge_attr.float())
                target_msg = torch.index_select(msg, dim=0, index=target_node)
                h_target = torch.index_select(h0, dim=0, index=target_node)
                _, h_target = self.backward_update0(target_msg.unsqueeze(0), h_target.unsqueeze(0))
                h0[target_node, :] = h_target.squeeze(0)
            if target_node.size(0) > 0:
                # -------------------------get current edge----------------------
                target_edge_index, target_edge_attr = edge_index, edge_attr 
                # ------------Update hidden state
                msg = self.forward_aggr1(h1, target_edge_index, target_edge_attr.float())
                target_node = target_node.to(msg.device)
                target_msg = torch.index_select(msg, dim=0, index=target_node)
                h_target = torch.index_select(h1, dim=0, index=target_node)
                _, h_target = self.forward_update1(target_msg.unsqueeze(0), h_target.unsqueeze(0))
                h1[target_node, :] = h_target.squeeze(0)

                # -------------------------get current edge----------------------
                target_edge_index, target_edge_attr = edge_index, edge_attr 
                # ------------Update hidden state
                msg = self.backward_aggr1(h1, target_edge_index, target_edge_attr.float())
                target_msg = torch.index_select(msg, dim=0, index=target_node)
                h_target = torch.index_select(h1, dim=0, index=target_node)
                _, h_target = self.backward_update1(target_msg.unsqueeze(0), h_target.unsqueeze(0))
                h1[target_node, :] = h_target.squeeze(0)
        h0_finite = h0[:, :self.dim_hidden]
        h1_finite = h1[:, :self.dim_hidden]
        h0_trans = h0[:, self.dim_hidden:]
        h1_trans = h1[:, self.dim_hidden:]
        return h0_finite, h1_finite, h0_trans, h1_trans

def find_nodes_in_cycle(cyclic_ff, node, edge_index, visited, one_path, nodes_in_path, first_flag=True):
    #print("calling dfs")
    if cyclic_ff == int(node) and not first_flag:
        nodes_in_path = nodes_in_path + one_path
        #print("returning from dfs")
        return nodes_in_path
    
    src = edge_index[0] == node
    fanout_nodes = edge_index[1][src] #all fanouts of node
    
    for f in fanout_nodes:
        if not visited[f]:
            visited[f] = True
            one_path[f] = True
            nodes_in_path = find_nodes_in_cycle(cyclic_ff,f,edge_index,visited,one_path,nodes_in_path,first_flag=False)
            one_path[f] = False
    
    #print("existing: search finished")
    return nodes_in_path
    
class CustomGNN(nn.Module):
    def __init__(self, args, num_rounds=1, dim_hidden=128, aggr='transformer'):
        super(CustomGNN, self).__init__()

        self.num_rounds = num_rounds
        self.dim_hidden = dim_hidden
        self.aggr = aggr
        self.args = args

        # Network for AND gate
        self.aggr_and = self.get_and_aggr(aggr)
        self.aggr_not = self.get_and_aggr(aggr)

        # self.hf_init = nn.Embedding(4, self.dim_hidden)
        self.hf_init = MLP(10, self.dim_hidden, self.dim_hidden, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        self.hs_init = MLP(10, self.dim_hidden, self.dim_hidden, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        self.level_embedding = nn.Embedding(300, self.dim_hidden) # assume max 300 layers

        self.final_mlp = MLP(2*self.dim_hidden, self.dim_hidden, self.dim_hidden, 3)

        self.readout_prob = MLP(self.dim_hidden, 32, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        self.readout_trans = MLP(self.dim_hidden, 32, 2, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        
        self.pos_enc = nn.Embedding(10, 128)
        
    def get_and_aggr(self, aggr):

        tf_layer = nn.TransformerEncoderLayer(d_model=self.dim_hidden, nhead=8, dim_feedforward=self.dim_hidden*2, batch_first=True)
        return nn.TransformerEncoder(tf_layer, num_layers=2)
        
 
    @property
    def last_shared_layer(self): # for aggr=transformer
        if self.aggr != 'transformer':
            raise AttributeError("only aggr='transformer' can access last_shared_layer")
        return self.aggr_and.layers[-1].linear2
    
        
    def forward(self, G, input_pattern=None):
        device = next(self.parameters()).device
        max_num_layers = torch.max(G.forward_level).item() + 1
        min_num_layers = 1

        #gate_type: 0: PI, 1: AND, 2: NOT
        if self.aggr == 'transformer':
            hlogic = self.hf_init(G.x.float().to(device))
        else:
            hlogic = torch.zeros(G.num_nodes, self.dim_hidden).to(G.y_prob1.device)
        
        if self.aggr == 'transformer':
            htrans = self.hs_init(G.x.float().to(device))
        else:
            htrans = torch.zeros(G.num_nodes, self.dim_hidden).to(G.prob.device)

        hlogic[G.forward_level==0] = G.y_prob1[G.forward_level==0].repeat(1,self.dim_hidden)
        htrans[G.forward_level==0] = G.y_trans_prob[G.forward_level==0].mean(dim=1, keepdim=True).repeat(1, self.dim_hidden)


        for _ in range(self.num_rounds):
            # h_new = h.clone()
            for level in range(min_num_layers, max_num_layers):
                h_new = hlogic.clone()
                l_and_node = G.forward_index[G.forward_level == level]

                if l_and_node.size(0) > 0:
                    and_mask  = G.forward_level[G.edge_index[1]] == level
                    and_edge_index = G.edge_index[:,and_mask]

                    and_tgt_node_idx = and_edge_index[1]
                    and_src_node_idx = and_edge_index[0]

                    unique_tgt_nodes, inverse_indices = torch.unique(and_tgt_node_idx, return_inverse=True)
                    max_length = torch.bincount(inverse_indices).max().item()

                    # Create index matrix and padding mask
                    index_matrix = torch.full((len(unique_tgt_nodes), max_length), -1, dtype=torch.long, device=hlogic.device)
                    padding_mask = torch.ones((len(unique_tgt_nodes), max_length), dtype=torch.bool, device=hlogic.device)

                    # Use scatter to fill index_matrix and padding_mask
                    src_counts = torch.bincount(inverse_indices)
                    src_offsets = torch.cumsum(src_counts, dim=0) - src_counts
                    src_positions = torch.arange(and_src_node_idx.size(0), device=hlogic.device) - src_offsets[inverse_indices]

                    index_matrix[inverse_indices, src_positions] = and_src_node_idx
                    padding_mask[inverse_indices, src_positions] = False

                    # Create padded sequences
                    padded_sequences = torch.cat([hlogic[unique_tgt_nodes].unsqueeze(1),hlogic[index_matrix]],dim=1)
                    padding_mask = torch.cat([torch.zeros((len(unique_tgt_nodes), 1), dtype=torch.bool, device=hlogic.device), padding_mask], dim=1)

                    # add positional encoding  
                    pos = torch.arange(0, padded_sequences.shape[1], device=hlogic.device).unsqueeze(0).repeat(padded_sequences.shape[0], 1)
                    aggr = self.aggr_and(padded_sequences+ self.pos_enc(pos), src_key_padding_mask = padding_mask)[:,0]

                    # aggr = self.aggr_and(padded_sequences, src_key_padding_mask = padding_mask)[:,0]
                   
                    h_new[unique_tgt_nodes, :] = aggr

                hlogic = h_new
            for level in range(min_num_layers, max_num_layers):
                h_new = htrans.clone()
                l_and_node = G.forward_index[G.forward_level == level]

                if l_and_node.size(0) > 0:
                    and_mask  = G.forward_level[G.edge_index[1]] == level
                    and_edge_index = G.edge_index[:,and_mask]

                    and_tgt_node_idx = and_edge_index[1]
                    and_src_node_idx = and_edge_index[0]

                    unique_tgt_nodes, inverse_indices = torch.unique(and_tgt_node_idx, return_inverse=True)
                    max_length = torch.bincount(inverse_indices).max().item()

                    # Create index matrix and padding mask
                    index_matrix = torch.full((len(unique_tgt_nodes), max_length), -1, dtype=torch.long, device=htrans.device)
                    padding_mask = torch.ones((len(unique_tgt_nodes), max_length), dtype=torch.bool, device=htrans.device)

                    # Use scatter to fill index_matrix and padding_mask
                    src_counts = torch.bincount(inverse_indices)
                    src_offsets = torch.cumsum(src_counts, dim=0) - src_counts
                    src_positions = torch.arange(and_src_node_idx.size(0), device=htrans.device) - src_offsets[inverse_indices]

                    index_matrix[inverse_indices, src_positions] = and_src_node_idx
                    padding_mask[inverse_indices, src_positions] = False

                    # Create padded sequences
                    padded_sequences = torch.cat([htrans[unique_tgt_nodes].unsqueeze(1),htrans[index_matrix]],dim=1)
                    padding_mask = torch.cat([torch.zeros((len(unique_tgt_nodes), 1), dtype=torch.bool, device=htrans.device), padding_mask], dim=1)

                    # add positional encoding  
                    pos = torch.arange(0, padded_sequences.shape[1], device=htrans.device).unsqueeze(0).repeat(padded_sequences.shape[0], 1)
                    aggr = self.aggr_not(padded_sequences+ self.pos_enc(pos), src_key_padding_mask = padding_mask)[:,0]

                    # aggr = self.aggr_and(padded_sequences, src_key_padding_mask = padding_mask)[:,0]
                   
                    h_new[unique_tgt_nodes, :] = aggr

                htrans = h_new
            del h_new
        prob = self.readout_prob(hlogic)
        trans = self.readout_trans(htrans)
        hseq = torch.zeros_like(htrans)
        y_prob0 = torch.zeros_like(prob)
        return (hlogic, htrans, hseq, prob, y_prob0, trans), 1
        
    
    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()
        
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)

class MLPGate3(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits.
    '''
    def __init__(self, args):
        super(MLPGate3, self).__init__()
        
        self.args = args

         # configuration
        self.num_rounds = args.num_rounds
        self.device = args.device
        self.predict_diff = args.predict_diff
        self.intermediate_supervision = args.intermediate_supervision
        self.reverse = args.reverse
        self.custom_backward = args.custom_backward
        self.use_edge_attr = args.use_edge_attr
        self.mask = args.mask

        # dimensions
        self.num_aggr = args.num_aggr
        self.dim_node_feature = args.dim_node_feature
        self.dim_hidden = args.dim_hidden
        self.dim_mlp = args.dim_mlp
        self.dim_pred = args.dim_pred
        self.num_fc = args.num_fc
        self.wx_update = args.wx_update
        self.wx_mlp = args.wx_mlp
        self.dim_edge_feature = args.dim_edge_feature

        # Network 
        if args.aggr_function == 'mlp' or args.aggr_function == 'attnmlp':
            self.aggr_and_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, args.dim_mlp, self.dim_hidden*2, num_layer=3, act_layer='relu')
            self.aggr_and_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*4, args.dim_mlp, self.dim_hidden*2, num_layer=3, act_layer='relu')
            self.aggr_and_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*6, args.dim_mlp, self.dim_hidden*2, num_layer=3, act_layer='relu')
            self.aggr_not_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, args.dim_mlp, self.dim_hidden*2, num_layer=3, act_layer='relu')
            self.aggr_not_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*4, args.dim_mlp, self.dim_hidden*2, num_layer=3, act_layer='relu')
            self.aggr_not_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*6, args.dim_mlp, self.dim_hidden*2, num_layer=3, act_layer='relu')
            self.aggr_ff_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, args.dim_mlp, self.dim_hidden*2, num_layer=3, act_layer='relu')
            
        else:
            self.aggr_and_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, self.dim_hidden*2)
            self.aggr_and_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*4, self.dim_hidden*2)
            self.aggr_and_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*6, self.dim_hidden*2)
            self.aggr_not_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, self.dim_hidden*2)
            self.aggr_not_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*4, self.dim_hidden*2)
            self.aggr_not_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*6, self.dim_hidden*2)
            self.aggr_ff_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, self.dim_hidden*2)
            
        self.update_and_strc = GRU(self.dim_hidden*2, self.dim_hidden*2)
        self.update_and_func = GRU(self.dim_hidden*2, self.dim_hidden*2)
        self.update_and_seq = GRU(self.dim_hidden*2, self.dim_hidden*2)
        self.update_not_strc = GRU(self.dim_hidden*2, self.dim_hidden*2)
        self.update_not_func = GRU(self.dim_hidden*2, self.dim_hidden*2)
        self.update_not_seq = GRU(self.dim_hidden*2, self.dim_hidden*2)
        self.update_ff_strc = GRU(self.dim_hidden*2, self.dim_hidden*2)
        # Linear transformation for FF
        

        if self.args.lt_ff:
            self.linear_transform_ff_func = MLP(self.dim_hidden*2, args.dim_hidden, args.dim_hidden, num_layer=2)
            self.linear_transform_ff_seq = MLP(self.dim_hidden*2, args.dim_hidden, args.dim_hidden, num_layer=2)

        else:
            if args.aggr_function == 'mlp' or args.aggr_function == 'attnmlp':
                self.aggr_ff_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*4, args.dim_mlp, self.dim_hidden*2, num_layer=3, act_layer='relu')
                self.aggr_ff_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*6, args.dim_mlp, self.dim_hidden*2, num_layer=3, act_layer='relu')
            else:
                self.aggr_ff_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*4, args.dim_hidden*2)
                self.aggr_ff_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*6, args.dim_hidden*2)
        
            self.update_ff_func = GRU(self.dim_hidden*2, self.dim_hidden*2)
            self.update_ff_seq = GRU(self.dim_hidden*2, self.dim_hidden*2)
        self.pooling_transformer_query = PoolingTransformer(dim_hidden=self.dim_hidden)
        self.pooling_transformer_input = PoolingTransformer(dim_hidden=self.dim_hidden)
        self.pooling_transformer_query2 = PoolingTransformer(dim_hidden=self.dim_hidden)
        self.pooling_transformer_input2 = PoolingTransformer(dim_hidden=self.dim_hidden)
        # # Readout 
        self.readout = MLP(self.dim_hidden*2, args.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        self.readout2 = MLP(self.dim_hidden*2, args.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
     
        # self.readout_trans_prob = MLP(self.dim_hidden, args.dim_mlp, 2, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        

        # consider the embedding for the LSTM/GRU model initialized by non-zeros
        self.one = torch.ones(1).to(self.device)
        # self.hs_emd_int = nn.Linear(1, self.dim_hidden)
        self.hf_emd_int = nn.Linear(1, self.dim_hidden*2)
        self.hseq_emd_int = nn.Linear(1, self.dim_hidden*2)
        self.one.requires_grad = False
        sys.setrecursionlimit(100000)

        
    def forward(self, G, finite_list_full, trans_matrics):
   
        num_nodes = G.num_nodes
        num_layers_f = max(G.forward_level).item() + 1
        num_layers_b = max(G.backward_level).item() + 1
        
        # initialize the structure hidden state
        if self.args.disable_encode: 
            hs_init = torch.zeros(num_nodes, self.dim_hidden*2)
            max_sim = 0
            hs_init = hs_init.to(self.device)
        else:
            hs_init = torch.zeros(num_nodes, self.dim_hidden*2)
            hs_init, max_sim = generate_hs_init(G, hs_init, self.dim_hidden*2)
            hs_init = hs_init.to(self.device)
        
        # initialize the function hidden state
        hf_init = self.hf_emd_int(self.one).view(1, -1) # (1 x 1 x dim_hidden)
        hf_init = hf_init.repeat(num_nodes, 1) # (1 x num_nodes x dim_hidden)
        # inferring against a particular workload
        hf_init = self.imply_mask(G, hf_init, self.dim_hidden*2) 


        # initialize the sequential hidden state
        hseq_init = self.hseq_emd_int(self.one).view(1, -1) # (1 x 1 x dim_hidden)
        hseq_init = hseq_init.repeat(num_nodes, 1) # (1 x num_nodes x dim_hidden)
        # inferring against a particular workload
        hseq_init = self.imply_seq_mask(G, hseq_init, self.dim_hidden*2) 

        cir_name = G.name
        S = finite_list_full[cir_name[0]][0].shape[0]
        # preds = torch.zeros(S, S).to(self.device)
        labels = torch.zeros(S, S).to(self.device)
        gt = torch.zeros(S, S).to(self.device)
        h0, h1, h0_trans, h1_trans = self._gru_forward(G, hs_init, hf_init,hseq_init, num_layers_f, num_layers_b)     # [N, D]
        target1 = torch.tensor([0,0,0,1,0,0,0,0,0,0], dtype=torch.float).to(self.device)
        target2 = torch.tensor([0,0,0,0,1,0,0,0,0,0], dtype=torch.float).to(self.device)
        mask =( (G.x == target1) | (G.x == target2)).all(dim=1)
        h0 = h0[mask]
        h1 = h1[mask]
        h0_trans = h0_trans[mask]
        h1_trans = h1_trans[mask]
        N, D = h0.shape

        state_list = []
        for idx in range(S):
            state_bits = dec_to_bin_array(idx, N)   # 返回长度=N的0/1数组或tensor
            state_bits = torch.tensor(dec_to_bin_array(idx, N), dtype=torch.float32)
            state_list.append(state_bits)
        state_list = torch.stack(state_list, dim=0).to(self.device)  # [S, N]
        masks = (state_list == 0)                 # [S, N], bool
        selected = torch.where(
            masks.unsqueeze(-1),                   # [S, N, 1]
            h0.unsqueeze(0),                       # [1, N, D] -> broadcast
            h1.unsqueeze(0)                        # [1, N, D] -> broadcast
        )
        graph_embeds_query = self.pooling_transformer_query(selected) # [S, D]
        masks_input = (state_list == 0)  # [S, N]
        selected_input = torch.where(
            masks_input.unsqueeze(-1),
            h0.unsqueeze(0),
            h1.unsqueeze(0)
        )  # [S, N, D]
        graph_embeds_input = self.pooling_transformer_input(selected_input)  # [S, D]
        # logits = self.readout(graph_embeds_query).squeeze(-1)  # [S]
        # logits = torch.sigmoid(logits)  # [S]
        graph_embeds = torch.cat([
            graph_embeds_query.unsqueeze(0).expand(S, S, D),  # broadcast query
            graph_embeds_input.unsqueeze(1).expand(S, S, D)   # broadcast input
        ], dim=-1)  # [S, S, 2D]

        graph_embeds_flat = graph_embeds.view(S * S, -1)
        logits_flat = self.readout(graph_embeds_flat).squeeze(-1)  # [S*S]
        logits = logits_flat.view(S, S)  # [S, S]
        labels = (finite_list_full[cir_name[0]].to(self.device) > 0).float()  # [S, S]
        #labels = (finite_list_full[cir_name[0]][0] > 0).float().to(self.device)
        
        masks = (state_list == 0)                 # [S, N], bool
        selected = torch.where(
            masks.unsqueeze(-1),                   # [S, N, 1]
            h0_trans.unsqueeze(0),                       # [1, N, D] -> broadcast
            h1_trans.unsqueeze(0)                        # [1, N, D] -> broadcast
        )
        graph_embeds_query = self.pooling_transformer_query2(selected) # [S, D]
        masks_input = (state_list == 0)  # [S, N]
        selected_input = torch.where(
            masks_input.unsqueeze(-1),
            h0_trans.unsqueeze(0),
            h1_trans.unsqueeze(0)
        )  # [S, N, D]
        graph_embeds_input = self.pooling_transformer_input2(selected_input)  # [S, D]

        graph_embeds = torch.cat([
            graph_embeds_query.unsqueeze(0).expand(S, S, D),  # broadcast query
            graph_embeds_input.unsqueeze(1).expand(S, S, D)   # broadcast input
        ], dim=-1)  # [S, S, 2D]

        graph_embeds_flat = graph_embeds.view(S * S, -1)
        probs_flat = self.readout2(graph_embeds_flat).squeeze(-1)  # [S*S]
        probs = probs_flat.view(S, S)  # [S, S]
        gt = trans_matrics[cir_name[0]].to(self.device)
        return logits, labels, probs, gt
    
    def _gru_forward(self, G, hs_init, hf_init, hseq_init, num_layers_f, num_layers_b, h_true=None, h_false=None):
        G = G.to(self.device)
        
       
        x, edge_index= G.x, G.edge_index
    
        edge_attr = G.edge_attr

        hs = hs_init.to(self.device)
        hf = hf_init.to(self.device)
        hseq = hseq_init.to(self.device)

        #node_state_seq_not = torch.cat([hf, hseq], dim=-1)
  
        node_state = torch.cat([hs, hf], dim=-1)
        node_state_seq = torch.cat([hs, hf, hseq], dim=-1)
     
        and_mask = G.gate.squeeze(1) == 1
        not_mask = G.gate.squeeze(1) == 2
        ff_mask  = G.gate.squeeze(1) == 3
        
        
        for _ in range(self.num_rounds):
            for level in range(0, num_layers_f):
                # forward layer
                layer_mask = G.forward_level == level

                # AND Gate
                l_and_node = G.forward_index[layer_mask & and_mask]
                if l_and_node.size(0) > 0:
                    and_edge_index, and_edge_attr = subgraph(l_and_node, edge_index, edge_attr, dim=1)
                    # Update structure hidden state
                    msg = self.aggr_and_strc(hs, and_edge_index,and_edge_attr)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hs_and = torch.index_select(hs, dim=0, index=l_and_node)
                    _, hs_and = self.update_and_strc(and_msg.unsqueeze(0), hs_and.unsqueeze(0))
                    hs[l_and_node, :] = hs_and.squeeze(0)

                    # Update function hidden state
                    msg = self.aggr_and_func(node_state, and_edge_index, and_edge_attr)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hf_and = torch.index_select(hf, dim=0, index=l_and_node)
                    _, hf_and = self.update_and_func(and_msg.unsqueeze(0), hf_and.unsqueeze(0))
                    hf[l_and_node, :] = hf_and.squeeze(0)

                     # Update sequential hidden state
                    msg = self.aggr_and_seq(node_state_seq, and_edge_index, and_edge_attr) #function or structure or sequential?
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hseq_and = torch.index_select(hseq, dim=0, index=l_and_node)
                    _, hseq_and = self.update_and_seq(and_msg.unsqueeze(0), hseq_and.unsqueeze(0))
                    hseq[l_and_node, :] = hseq_and.squeeze(0)

                # NOT Gate
                l_not_node = G.forward_index[layer_mask & not_mask]
                if l_not_node.size(0) > 0:
                    not_edge_index, not_edge_attr = subgraph(l_not_node, edge_index, edge_attr, dim=1)
                    # Update structure hidden state
                    msg = self.aggr_not_strc(hs, not_edge_index, not_edge_attr)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hs_not = torch.index_select(hs, dim=0, index=l_not_node)
                    _, hs_not = self.update_not_strc(not_msg.unsqueeze(0), hs_not.unsqueeze(0))
                    hs[l_not_node, :] = hs_not.squeeze(0)

                    # Update function hidden state
                    msg = self.aggr_not_func(node_state, not_edge_index, not_edge_attr)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hf_not = torch.index_select(hf, dim=0, index=l_not_node)
                    _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
                    hf[l_not_node, :] = hf_not.squeeze(0)

                    # Update sequential hidden state
                    msg = self.aggr_not_seq(node_state_seq, not_edge_index, not_edge_attr)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hseq_not = torch.index_select(hseq, dim=0, index=l_not_node)
                    _, hseq_not = self.update_not_seq(not_msg.unsqueeze(0), hseq_not.unsqueeze(0))
                    hseq[l_not_node, :] = hseq_not.squeeze(0)

                #FF Gate
                l_ff_fanin_node = G.forward_index[layer_mask & ff_mask]   
                if len(l_ff_fanin_node) > 0:    
                    ff_edge_index, ff_edge_attr = subgraph(l_ff_fanin_node, edge_index, edge_attr, dim=1)
                    # Update structure hidden state
                    msg = self.aggr_ff_strc(hs, ff_edge_index, ff_edge_attr)
                    ff_msg = torch.index_select(msg, dim=0, index=l_ff_fanin_node)
                    hs_ff = torch.index_select(hs, dim=0, index=l_ff_fanin_node)
                    _, hs_ff = self.update_ff_strc(ff_msg.unsqueeze(0), hs_ff.unsqueeze(0))
                    hs[l_ff_fanin_node, :] = hs_ff.squeeze(0)

                    # Update function hidden state
                    msg = self.aggr_ff_func(node_state, ff_edge_index, ff_edge_attr)
                    ff_msg = torch.index_select(msg, dim=0, index=l_ff_fanin_node)
                    hf_ff = torch.index_select(hf, dim=0, index=l_ff_fanin_node)
                    _, hf_ff = self.update_ff_func(ff_msg.unsqueeze(0), hf_ff.unsqueeze(0))
                    hf[l_ff_fanin_node, :] = hf_ff.squeeze(0)

                    # Update sequential hidden state
                    msg = self.aggr_ff_seq(node_state_seq, ff_edge_index, ff_edge_attr)
                    ff_msg = torch.index_select(msg, dim=0, index=l_ff_fanin_node)
                    hseq_ff = torch.index_select(hseq, dim=0, index=l_ff_fanin_node)
                    _, hseq_ff = self.update_ff_seq(ff_msg.unsqueeze(0), hseq_ff.unsqueeze(0))
                    hseq[l_ff_fanin_node, :] = hseq_ff.squeeze(0)
                    
                         
                # Update node state
                node_state = torch.cat([hs, hf], dim=-1)

                # Update sequential node state
                node_state_seq = torch.cat([hs, hf, hseq], dim=-1)

        node_embedding = node_state.squeeze(0)
        node_embedding_seq = node_state_seq.squeeze(0)

        hs = node_embedding[:, :self.dim_hidden*2]
        hf = node_embedding[:, self.dim_hidden*2:]
        hseq = node_embedding_seq[:, self.dim_hidden*2 + self.dim_hidden*2:]

        h0 = hs[:, :self.dim_hidden]
        h0_trans = hs[:, self.dim_hidden:] 
        h1 = hf[:, :self.dim_hidden] 
        h1_trans = hf[:, self.dim_hidden:]

        return h0, h1, h0_trans, h1_trans
    def imply_mask(self, G, h, dim_hidden):

        # true_mask = (G.mask != -1.0).unsqueeze(0).to(self.device)
        # normal_mask = (G.mask == -1.0).unsqueeze(0).to(self.device)

        true_mask = (G.mask != -1).to(self.device)
        normal_mask = (G.mask == -1).to(self.device)

        #h_true = G.mask.unsqueeze(0).to(self.device)
        h_true = G.mask.to(self.device)
        h_true = h_true.expand(-1, dim_hidden).to(self.device)
      
        h_mask = h * normal_mask + h_true * true_mask
    
        return h_mask
    
    def imply_seq_mask(self, G, h, dim_hidden):

        # true_mask = (G.mask != -1.0).unsqueeze(0).to(self.device)
        # normal_mask = (G.mask == -1.0).unsqueeze(0).to(self.device)

        true_mask = (G.mask != -1).to(self.device)
        normal_mask = (G.mask == -1).to(self.device)
      
        h_true_1 = G.seq_mask[:,0].to(self.device)
        h_true_1 = h_true_1.reshape([len(h_true_1), 1])

        h_true_2 = G.seq_mask[:,1].to(self.device)
        h_true_2 = h_true_2.reshape([len(h_true_2), 1])

        size = int(dim_hidden/2)
        h_true_1 = h_true_1.expand(-1, size).to(self.device)
        h_true_2 = h_true_2.expand(-1, size).to(self.device)

        h_true = torch.cat([h_true_1, h_true_2], dim=-1)

        h_mask = h * normal_mask + h_true * true_mask
   
        return h_mask

class Baseline(nn.Module):
    def __init__(self,args,
                num_layers=9,
                in_channels=10,
                dim_hidden=128,
                model_name='GraphSAGE'):
        super(Baseline, self).__init__()
        self.device = args.device
        self.dim_hidden = dim_hidden
        self.model_name = model_name
        if model_name == 'GCN':
            self.model = gnn.GCN(in_channels,dim_hidden, num_layers, out_channels=dim_hidden)
        elif model_name == 'GAT':
            self.model = gnn.GAT(in_channels,dim_hidden, num_layers, out_channels=dim_hidden)  
        elif model_name == 'GraphSAGE':
            self.model = gnn.GraphSAGE(in_channels,dim_hidden, num_layers, out_channels=dim_hidden)
        elif model_name == 'GIN':
            self.model = gnn.GIN(in_channels,dim_hidden, num_layers, out_channels=dim_hidden)
        elif model_name == 'SGFormer':
            self.model = SGFormer(in_channels=in_channels, hidden_channels=dim_hidden, out_channels=dim_hidden,trans_num_layers=6)
        elif model_name == 'DIFFormer':
            self.model = DIFFormer_v2(in_channels=in_channels, hidden_channels=dim_hidden, out_channels=dim_hidden)
        self.readout_prob0 = MLP(self.dim_hidden, 32, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        self.readout_prob1 = MLP(self.dim_hidden, 32, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
     
    def forward(self, G):
        G = G.to(self.device)
        if self.model_name == 'DIFFormer':
            h = self.model(G)
        elif self.model_name == 'SGFormer':
            h = self.model(G.x, G.edge_index, G.batch)
        else: 
            h = self.model(G.x, G.edge_index)
        prob0 = self.readout_prob0(h)
        prob1 = self.readout_prob1(h)
        return (h, h, prob0, prob1), 1

class GPS(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any]):
        super().__init__()

        self.node_emb = nn.Linear(10, channels - pe_dim)
        self.pe_lin = nn.Linear(10, pe_dim)
        self.pe_norm = nn.BatchNorm1d(10)
        self.edge_emb = nn.Linear(16, channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            nnlocal = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.Linear(channels, channels),
            )
            conv = gnn.GPSConv(channels, gnn.GINEConv(nnlocal), heads=4,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp1 = MLP(channels, 128, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        self.mlp0 = MLP(channels, 128, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        self.mlptrans = MLP(channels, 128, 2, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, G):
        x, pe, edge_index, edge_attr, batch = G.x, G.pe, G.edge_index, G.edge_attr, G.batch
        x_pe = self.pe_norm(pe)
        x_indices = x.squeeze(-1)
        x = torch.cat((self.node_emb(x_indices), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, batch,edge_attr=edge_attr)
        prob1 = self.mlp1(x)
        prob0 = self.mlp0(x)
        trans = self.mlptrans(x)
        return (x,x,x,prob1,prob0,trans),1
class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1



def get_mlp_gate(args):
    return MLPGate(args)
def get_mlp_gate_stage2(args):
    return MLPGate2(args)
def get_mlp_gate_stage3(args):
    return MLPGate3(args)
def get_baseline(args):
    return Baseline(args, model_name=args.modelname)
def get_graphgps(args):
    return GPS(channels=128, pe_dim=20, num_layers=6, attn_type='multihead', attn_kwargs={'dropout':0.2})
# # Update FF
                    # ff_edge_index, ff_edge_attr = subgraph(l_ff_fanin_node, ff_fanin_edge_index, dim=1)

                    # # Update FF's structure hidden state
                    # msg = self.aggr_ff_strc(hs, ff_edge_index)
                    # ff_msg = torch.index_select(msg, dim=0, index=l_ff_fanin_node)
                    # hs_ff = torch.index_select(hs, dim=0, index=l_ff_fanin_node)
                    # _, hs_ff = self.update_ff_strc(ff_msg.unsqueeze(0), hs_ff.unsqueeze(0))
                    # hs[l_ff_fanin_node, :] = hs_ff.squeeze(0)

                    
                    # if self.args.lt_ff:

                    #     # Update function
                    #     ff_fanin_state = torch.index_select(hf, dim=0, index=ff_edge_index[0].long())
                    #     hf[ff_edge_index[1], :] = self.linear_transform_ff_func(ff_fanin_state)

                    #     # Update sequential
                    #     ff_fanin_state = torch.index_select(hseq, dim=0, index=ff_edge_index[0].long())
                    #     hseq[ff_edge_index[1], :] = self.linear_transform_ff_seq(ff_fanin_state)

                    # else:
                    #     # Update function hidden state
                    #     msg = self.aggr_ff_func(hf, ff_edge_index, ff_edge_attr)
                    #     ff_msg = torch.index_select(msg, dim=0, index=l_ff_fanin_node)
                    #     hf_ff = torch.index_select(hf, dim=0, index=l_ff_fanin_node)
                    #     _, hf_ff = self.update_ff_func(ff_msg.unsqueeze(0), hf_ff.unsqueeze(0))
                    #     hf[l_ff_fanin_node, :] = hf_ff.squeeze(0)

                    #     # Update sequential hidden state
                    #     msg = self.aggr_ff_seq(node_state_seq, ff_edge_index, ff_edge_attr)
                    #     ff_msg = torch.index_select(msg, dim=0, index=l_ff_fanin_node)
                    #     hseq_ff = torch.index_select(hseq, dim=0, index=l_ff_fanin_node)
                    #     _, hseq_ff = self.update_ff_seq(ff_msg.unsqueeze(0), hseq_ff.unsqueeze(0))
                    #     hseq[l_ff_fanin_node, :] = hseq_ff.squeeze(0)

                    # nodes_in_path = torch.tensor([False] * len(x))

                    # #Find cyclic subgraph 
                    # # during inference, this code will execute 
                    # # all nodes in all cycles of this circuits are collected during data preperation
                    # if hasattr(G, 'nodes_in_cycles'):
                    #     for node in ff_fanin_edge_index[1]:       
                    #         if cyclic_FFs_nodes[node]:
                    #             nodes_in_path += G.nodes_in_cycles[node,:] 

                    # else:   # Find cyclic subgraph ; for training
                    #     one_path = torch.tensor([False] * len(x)) 
                    #     for node in ff_edge_index[1]:
                    #         visited = [False] * len(x)
                    #         if cyclic_FFs_nodes[node]:
                    #             nodes_in_path += find_nodes_in_cycle(int(node), int(node), original_edge_index, visited, one_path, nodes_in_path)
                    
                    # nodes_in_path = nodes_in_path.to(self.device)
                    
                    # # Update subgraph 
                    # for c_level in range(0, num_layers_f):
                    #     c_layer_mask = G.forward_level == c_level
                    #     c_layer_mask &= nodes_in_path
                        
                    #     # AND Gate
                    #     c_l_and_node = G.forward_index[c_layer_mask & and_mask]
                    #     if c_l_and_node.size(0) > 0:
                    #         c_and_edge_index, c_and_edge_attr = subgraph(c_l_and_node, edge_index, edge_attr, dim=1)
                    #         # Update structure hidden state
                    #         c_msg = self.aggr_and_strc(hs, c_and_edge_index, c_and_edge_attr)
                    #         c_and_msg = torch.index_select(c_msg, dim=0, index=c_l_and_node)
                    #         c_hs_and = torch.index_select(hs, dim=0, index=c_l_and_node)
                    #         _, c_hs_and = self.update_and_strc(c_and_msg.unsqueeze(0), c_hs_and.unsqueeze(0))
                    #         hs[c_l_and_node, :] = c_hs_and.squeeze(0)

                    #         # Update function hidden state
                    #         c_msg = self.aggr_and_func(node_state, c_and_edge_index, c_and_edge_attr)
                    #         c_and_msg = torch.index_select(c_msg, dim=0, index=c_l_and_node)
                    #         c_hf_and = torch.index_select(hf, dim=0, index=c_l_and_node)
                    #         _, c_hf_and = self.update_and_func(c_and_msg.unsqueeze(0), c_hf_and.unsqueeze(0))
                    #         hf[c_l_and_node, :] = c_hf_and.squeeze(0)

                    #         # Update sequential hidden state
                    #         c_msg = self.aggr_and_seq(node_state_seq, c_and_edge_index, c_and_edge_attr)
                    #         c_and_msg = torch.index_select(c_msg, dim=0, index=c_l_and_node)
                    #         c_hseq_and = torch.index_select(hseq, dim=0, index=c_l_and_node)
                    #         _, c_hseq_and = self.update_and_seq(c_and_msg.unsqueeze(0), c_hseq_and.unsqueeze(0))
                    #         hseq[c_l_and_node, :] = c_hseq_and.squeeze(0)
                    
                    #     # NOT Gate
                    #     c_l_not_node = G.forward_index[c_layer_mask & not_mask]
                    #     if c_l_not_node.size(0) > 0:
                    #         c_not_edge_index, c_not_edge_attr = subgraph(c_l_not_node, edge_index, edge_attr, dim=1)
                    #         # Update structure hidden state
                    #         c_msg = self.aggr_not_strc(hs, c_not_edge_index, c_not_edge_attr)
                    #         c_not_msg = torch.index_select(c_msg, dim=0, index=c_l_not_node)
                    #         c_hs_not = torch.index_select(hs, dim=0, index=c_l_not_node)
                    #         _, c_hs_not = self.update_not_strc(c_not_msg.unsqueeze(0), c_hs_not.unsqueeze(0))
                    #         hs[c_l_not_node, :] = c_hs_not.squeeze(0)
                            
                    #         # Update function hidden state
                    #         c_msg = self.aggr_not_func(hf, c_not_edge_index, c_not_edge_attr)
                    #         c_not_msg = torch.index_select(c_msg, dim=0, index=c_l_not_node)
                    #         c_hf_not = torch.index_select(hf, dim=0, index=c_l_not_node)
                    #         _, c_hf_not = self.update_not_func(c_not_msg.unsqueeze(0), c_hf_not.unsqueeze(0))
                    #         hf[c_l_not_node, :] = c_hf_not.squeeze(0)
                            
                    #         # Update sequential hidden state
                    #         c_msg = self.aggr_not_seq(node_state_seq, c_not_edge_index, c_not_edge_attr)
                    #         c_not_msg = torch.index_select(c_msg, dim=0, index=c_l_not_node)
                    #         c_hseq_not = torch.index_select(hseq, dim=0, index=c_l_not_node)
                    #         _, c_hseq_not = self.update_not_seq(c_not_msg.unsqueeze(0), c_hseq_not.unsqueeze(0))
                    #         hseq[c_l_not_node, :] = c_hseq_not.squeeze(0)
                            
                    #     # Last Update FF
                    #     c_l_ff_fanin_node = G.forward_index[c_layer_mask & ff_fanin_mask]
                    #     if len(c_l_ff_fanin_node) > 0:           
                    #         # Update FF
                    #         c_ff_edge_index, c_ff_edge_attr = subgraph(c_l_ff_fanin_node, ff_fanin_edge_index, dim=0)


                    #         # Update FF's structure hidden state
                    #         msg = self.aggr_ff_strc(hs, c_ff_edge_index)
                    #         c_ff_msg = torch.index_select(msg, dim=0, index=c_l_ff_fanin_node)
                    #         c_hs_ff = torch.index_select(hs, dim=0, index=c_l_ff_fanin_node)
                    #         _, c_hs_ff = self.update_ff_strc(c_ff_msg.unsqueeze(0), c_hs_ff.unsqueeze(0))
                    #         hs[c_l_ff_fanin_node, :] = c_hs_ff.squeeze(0)

                    #         if self.args.lt_ff:
                    #             # Update Function
                    #             c_ff_fanin_state = torch.index_select(hf, dim=0, index=c_ff_edge_index[0].long())
                    #             hf[c_ff_edge_index[1], :] = self.linear_transform_ff_func(c_ff_fanin_state)

                    #             # Update Sequential
                    #             c_ff_fanin_state = torch.index_select(hseq, dim=0, index=c_ff_edge_index[0].long())
                    #             hseq[c_ff_edge_index[1], :] = self.linear_transform_ff_seq(c_ff_fanin_state)
                            
                    #         else:
                    #             # Update function hidden state
                    #             msg = self.aggr_ff_func(hf, c_ff_edge_index, c_ff_edge_attr)
                    #             c_ff_msg = torch.index_select(msg, dim=0, index=c_l_ff_fanin_node)
                    #             c_hf_ff = torch.index_select(hf, dim=0, index=c_l_ff_fanin_node)
                    #             _, c_hf_ff = self.update_ff_func(c_ff_msg.unsqueeze(0), c_hf_ff.unsqueeze(0))
                    #             hf[c_l_ff_fanin_node, :] = c_hf_ff.squeeze(0)

                    #             # Update sequential hidden state
                    #             msg = self.aggr_ff_seq(node_state_seq, c_ff_edge_index, c_ff_edge_attr)
                    #             c_ff_msg = torch.index_select(msg, dim=0, index=c_l_ff_fanin_node)
                    #             c_hseq_ff = torch.index_select(hseq, dim=0, index=c_l_ff_fanin_node)
                    #             _, c_hseq_ff = self.update_ff_seq(c_ff_msg.unsqueeze(0), c_hseq_ff.unsqueeze(0))
                    #             hseq[c_l_ff_fanin_node, :] = c_hseq_ff.squeeze(0)