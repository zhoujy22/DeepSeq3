from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from utils.dag_utils import subgraph, custom_backward_subgraph
from utils.utils import generate_hs_init

from .mlp import MLP
from .mlp_aggr import MlpAggr
from .gat_conv import AGNNConv
from .gcn_conv import AggConv
from .deepset_conv import DeepSetConv
from .gated_sum_conv import GatedSumConv
from .aggnmlp import AttnMLP
from .tfmlp import TFMLP

from torch.nn import LSTM, GRU
import sys 
_aggr_function_factory = {
    'mlp': MlpAggr,             # MLP, similar as NeuroSAT  
    'attnmlp': AttnMLP,         # MLP with attention
    'tfmlp': TFMLP,             # MLP with transformer
    'aggnconv': AGNNConv,       # DeepGate with attention
    'conv_sum': AggConv,        # GCN
    'deepset': DeepSetConv,     # DeepSet, similar as NeuroSAT  
} 

_update_function_factory = {
    'lstm': LSTM,
    'gru': GRU,
}


class MLPGate_DG2(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits.
    '''
    def __init__(self, args):
        super(MLPGate_DG2, self).__init__()
        
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
            self.aggr_and_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
            self.aggr_not_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*1, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
            self.aggr_not_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*1, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
            self.aggr_not_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
            self.aggr_ff_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*1, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
            
        else:
            self.aggr_and_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*1, self.dim_hidden)
            self.aggr_and_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, self.dim_hidden)
            self.aggr_and_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*2, self.dim_hidden)
            self.aggr_not_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*1, self.dim_hidden)
            self.aggr_not_func = _aggr_function_factory[args.aggr_function](self.dim_hidden*1, self.dim_hidden)
            self.aggr_not_seq = _aggr_function_factory[args.aggr_function](self.dim_hidden*1, self.dim_hidden)
            self.aggr_ff_strc = _aggr_function_factory[args.aggr_function](self.dim_hidden*1, self.dim_hidden)
            
        self.update_and_strc = GRU(self.dim_hidden, self.dim_hidden)
        self.update_and_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_and_seq = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_strc = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_seq = GRU(self.dim_hidden, self.dim_hidden)
        self.update_ff_strc = GRU(self.dim_hidden, self.dim_hidden)
        # Linear transformation for FF
        
        self.linear_transform_ff_func = MLP(self.dim_hidden, args.dim_hidden, args.dim_hidden, num_layer=2)
        self.linear_transform_ff_seq = MLP(self.dim_hidden, args.dim_hidden, args.dim_hidden, num_layer=2)

        # Readout 
        self.readout_prob = MLP(self.dim_hidden, args.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        self.readout_rc = MLP(self.dim_hidden * 2, args.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', sigmoid=True)
        self.readout_trans_prob = MLP(self.dim_hidden, args.dim_mlp, 2, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')

        # consider the embedding for the LSTM/GRU model initialized by non-zeros
        self.one = torch.ones(1).to(self.device)
        # self.hs_emd_int = nn.Linear(1, self.dim_hidden)
        self.hf_emd_int = nn.Linear(1, self.dim_hidden)
        self.hseq_emd_int = nn.Linear(1, self.dim_hidden)
        self.one.requires_grad = False
        sys.setrecursionlimit(100000)
        
        # Load Pretrained DG2
        checkpoint = torch.load(self.args.dg2_path, map_location=lambda storage, loc: storage)
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
        

    def forward(self, G):
        num_nodes = G.num_nodes
        num_layers_f = max(G.forward_level).item() + 1
        num_layers_b = max(G.backward_level).item() + 1
        
        # initialize the structure hidden state
        if self.args.disable_encode: 
            hs_init = torch.zeros(num_nodes, self.dim_hidden)
            max_sim = 0
            hs_init = hs_init.to(self.device)
        else:
            hs_init = torch.zeros(num_nodes, self.dim_hidden)
            hs_init, max_sim = generate_hs_init(G, hs_init, self.dim_hidden)
            hs_init = hs_init.to(self.device)
        
        # initialize the function hidden state
        hf_init = self.hf_emd_int(self.one).view(1, -1) # (1 x 1 x dim_hidden)
        hf_init = hf_init.repeat(num_nodes, 1) # (1 x num_nodes x dim_hidden)
        # inferring against a particular workload
        hf_init = self.imply_mask(G, hf_init, self.dim_hidden) 


        # initialize the sequential hidden state
        hseq_init = self.hseq_emd_int(self.one).view(1, -1) # (1 x 1 x dim_hidden)
        hseq_init = hseq_init.repeat(num_nodes, 1) # (1 x num_nodes x dim_hidden)
        # inferring against a particular workload
        hseq_init = self.imply_seq_mask(G, hseq_init, self.dim_hidden) 

        preds = self._gru_forward(G, hs_init, hf_init,hseq_init, num_layers_f, num_layers_b)
        
        return preds, max_sim
            
    def _gru_forward(self, G, hs_init, hf_init, hseq_init, num_layers_f, num_layers_b, h_true=None, h_false=None):
        G = G.to(self.device)
        
       
        x, edge_index, ff_fanin_edge_index, original_edge_index, cyclic_FFs_nodes = G.x, G.edge_index, G.ff_fanin_edge_index , G.original_edge_index, G.cyclic_FFs_nodes, 
    
        edge_attr = G.edge_attr if self.use_edge_attr else None

        hs = hs_init.to(self.device)
        hf = hf_init.to(self.device)
        hseq = hseq_init.to(self.device)
  
        node_state = torch.cat([hs, hf], dim=-1)
        node_state_seq = torch.cat([hs, hseq], dim=-1)
     
        and_mask = G.gate.squeeze(1) == 1
        not_mask = G.gate.squeeze(1) == 2
        ff_mask  = G.gate.squeeze(1) == 3
        
        ff_fanin_mask = torch.zeros(G.forward_index.shape).to(self.device)
        ff_fanin_mask[ff_fanin_edge_index[0]] = 1
        ff_fanin_mask = ff_fanin_mask.bool()
  
        for _ in range(self.num_rounds):
            for level in range(1, num_layers_f):
                # forward layer
                layer_mask = G.forward_level == level

                # AND Gate
                l_and_node = G.forward_index[layer_mask & and_mask]
                if l_and_node.size(0) > 0:
                    and_edge_index, and_edge_attr = subgraph(l_and_node, edge_index, edge_attr, dim=1)
                    # Update structure hidden state
                    msg = self.aggr_and_strc(hs, and_edge_index, and_edge_attr)
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
                    msg = self.aggr_not_func(hf, not_edge_index, not_edge_attr)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hf_not = torch.index_select(hf, dim=0, index=l_not_node)
                    _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
                    hf[l_not_node, :] = hf_not.squeeze(0)

                    # Update sequential hidden state
                    msg = self.aggr_not_seq(hseq, not_edge_index, not_edge_attr)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hseq_not = torch.index_select(hseq, dim=0, index=l_not_node)
                    _, hseq_not = self.update_not_seq(not_msg.unsqueeze(0), hseq_not.unsqueeze(0))
                    hseq[l_not_node, :] = hseq_not.squeeze(0)

                l_ff_fanin_node = G.forward_index[layer_mask & ff_fanin_mask]    
                if len(l_ff_fanin_node) > 0:           
                    # Update FF
                    ff_edge_index, ff_edge_attr = subgraph(l_ff_fanin_node, ff_fanin_edge_index, dim=0)
                   
                    ff_fanin_state = torch.index_select(hf, dim=0, index=ff_edge_index[0].long())
                    hf[ff_edge_index[1], :] = self.linear_transform_ff_func(ff_fanin_state)
                    ff_fanin_state = torch.index_select(hseq, dim=0, index=ff_edge_index[0].long())
                    hseq[ff_edge_index[1], :] = self.linear_transform_ff_seq(ff_fanin_state)
            
                    
                    nodes_in_path = torch.tensor([False] * len(x))

                    #Find cyclic subgraph 
                    # during inference, this code will execute 
                    # all nodes in all cycles of this circuits are collected during data preperation
                    if hasattr(G, 'nodes_in_cycles'):
                        for node in ff_fanin_edge_index[1]:       
                            if cyclic_FFs_nodes[node]:
                                nodes_in_path += G.nodes_in_cycles[node,:] 

                    else:   # Find cyclic subgraph ; for training
                        one_path = torch.tensor([False] * len(x)) 
                        for node in ff_edge_index[1]:
                            visited = [False] * len(x)
                            if cyclic_FFs_nodes[node]:
                                nodes_in_path += find_nodes_in_cycle(int(node), int(node), original_edge_index, visited, one_path, nodes_in_path)
                    
                    nodes_in_path = nodes_in_path.to(self.device)
                    
                    # Update subgraph 
                    for c_level in range(1, num_layers_f):
                        c_layer_mask = G.forward_level == c_level
                        c_layer_mask &= nodes_in_path
                        
                        # AND Gate
                        c_l_and_node = G.forward_index[c_layer_mask & and_mask]
                        if c_l_and_node.size(0) > 0:
                            c_and_edge_index, c_and_edge_attr = subgraph(c_l_and_node, edge_index, edge_attr, dim=1)
                            # Update structure hidden state
                            c_msg = self.aggr_and_strc(hs, c_and_edge_index, c_and_edge_attr)
                            c_and_msg = torch.index_select(c_msg, dim=0, index=c_l_and_node)
                            c_hs_and = torch.index_select(hs, dim=0, index=c_l_and_node)
                            _, c_hs_and = self.update_and_strc(c_and_msg.unsqueeze(0), c_hs_and.unsqueeze(0))
                            hs[c_l_and_node, :] = c_hs_and.squeeze(0)

                            # Update function hidden state
                            c_msg = self.aggr_and_func(node_state, c_and_edge_index, c_and_edge_attr)
                            c_and_msg = torch.index_select(c_msg, dim=0, index=c_l_and_node)
                            c_hf_and = torch.index_select(hf, dim=0, index=c_l_and_node)
                            _, c_hf_and = self.update_and_func(c_and_msg.unsqueeze(0), c_hf_and.unsqueeze(0))
                            hf[c_l_and_node, :] = c_hf_and.squeeze(0)

                            # Update sequential hidden state
                            c_msg = self.aggr_and_seq(node_state_seq, c_and_edge_index, c_and_edge_attr)
                            c_and_msg = torch.index_select(c_msg, dim=0, index=c_l_and_node)
                            c_hseq_and = torch.index_select(hseq, dim=0, index=c_l_and_node)
                            _, c_hseq_and = self.update_and_seq(c_and_msg.unsqueeze(0), c_hseq_and.unsqueeze(0))
                            hseq[c_l_and_node, :] = c_hseq_and.squeeze(0)
                    
                        # NOT Gate
                        c_l_not_node = G.forward_index[c_layer_mask & not_mask]
                        if c_l_not_node.size(0) > 0:
                            c_not_edge_index, c_not_edge_attr = subgraph(c_l_not_node, edge_index, edge_attr, dim=1)
                            # Update structure hidden state
                            c_msg = self.aggr_not_strc(hs, c_not_edge_index, c_not_edge_attr)
                            c_not_msg = torch.index_select(c_msg, dim=0, index=c_l_not_node)
                            c_hs_not = torch.index_select(hs, dim=0, index=c_l_not_node)
                            _, c_hs_not = self.update_not_strc(c_not_msg.unsqueeze(0), c_hs_not.unsqueeze(0))
                            hs[c_l_not_node, :] = c_hs_not.squeeze(0)
                            
                            # Update function hidden state
                            c_msg = self.aggr_not_func(hf, c_not_edge_index, c_not_edge_attr)
                            c_not_msg = torch.index_select(c_msg, dim=0, index=c_l_not_node)
                            c_hf_not = torch.index_select(hf, dim=0, index=c_l_not_node)
                            _, c_hf_not = self.update_not_func(c_not_msg.unsqueeze(0), c_hf_not.unsqueeze(0))
                            hf[c_l_not_node, :] = c_hf_not.squeeze(0)
                            
                            # Update sequential hidden state
                            c_msg = self.aggr_not_seq(hseq, c_not_edge_index, c_not_edge_attr)
                            c_not_msg = torch.index_select(c_msg, dim=0, index=c_l_not_node)
                            c_hseq_not = torch.index_select(hseq, dim=0, index=c_l_not_node)
                            _, c_hseq_not = self.update_not_seq(c_not_msg.unsqueeze(0), c_hseq_not.unsqueeze(0))
                            hseq[c_l_not_node, :] = c_hseq_not.squeeze(0)
                            
                        # Last Update FF
                        c_l_ff_fanin_node = G.forward_index[c_layer_mask & ff_fanin_mask]
                        if len(c_l_ff_fanin_node) > 0:           
                            # Update FF
                            c_ff_edge_index, c_ff_edge_attr = subgraph(c_l_ff_fanin_node, ff_fanin_edge_index, dim=0)
                            c_ff_fanin_state = torch.index_select(hf, dim=0, index=c_ff_edge_index[0].long())
                            hf[c_ff_edge_index[1], :] = self.linear_transform_ff_func(c_ff_fanin_state)
                            c_ff_fanin_state = torch.index_select(hseq, dim=0, index=c_ff_edge_index[0].long())
                            hseq[c_ff_edge_index[1], :] = self.linear_transform_ff_seq(c_ff_fanin_state)
                            
                # Update node state
                node_state = torch.cat([hs, hf], dim=-1)

                # Update sequential node state
                node_state_seq = torch.cat([hs, hseq], dim=-1)

     
        #FFs edges
        src = ff_fanin_edge_index[0]
        dst = ff_fanin_edge_index[1]

        # Update FF's structure hidden state
        msg = self.aggr_ff_strc(hs, ff_fanin_edge_index)
        ff_msg = torch.index_select(msg, dim=0, index=dst.long())
        hs_ff = torch.index_select(hs, dim=0, index=dst.long())
        _, hs_ff = self.update_ff_strc(ff_msg.unsqueeze(0), hs_ff.unsqueeze(0))
        hs[dst.long(), :] = hs_ff.squeeze(0)
        node_state[dst.long(), :self.dim_hidden] = hs_ff.squeeze(0)
        node_state_seq[dst.long(), :self.dim_hidden] = hs_ff.squeeze(0)

        """
       
        ## FF function embeddings
        #linear transformation of representation at FF to induce clock delay
        ff_fanin_state =  node_state.index_select(0, src.long())
        node_state[dst.long(), self.dim_hidden:] = self.linear_transform_ff_func(ff_fanin_state[:, self.dim_hidden:])
    
        ## FF sequential embeddings
        #linear transformation of representation at FF to induce clock delay
        ff_fanin_state =  node_state_seq.index_select(0, src.long())
        node_state_seq[dst.long(), self.dim_hidden:] = self.linear_transform_ff_seq(ff_fanin_state[:, self.dim_hidden:])
        
        """

        node_embedding = node_state.squeeze(0)
        node_embedding_seq = node_state_seq.squeeze(0)

        hs = node_embedding[:, :self.dim_hidden]
        hf = node_embedding[:, self.dim_hidden:]
        hseq = node_embedding_seq[:, self.dim_hidden:]

        # Readout
        prob = self.readout_prob(hf)
        trans_prob = self.readout_trans_prob(hseq)
        
        rc_emb = torch.cat([hs[G.rc_pair_index[0]], hs[G.rc_pair_index[1]]], dim=1)
        is_rc = self.readout_rc(rc_emb)

       # return hs, hf, hseq, prob, trans_prob, is_rc, 
        return hs, hf, hseq, prob, trans_prob, is_rc, 
    

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
    

# def find_nodes_in_cycle(cyclic_ff, node, edge_index,visited, nodes_in_path, flag):
#     if cyclic_ff == int(node) and not flag:
#         return True, nodes_in_path
    
#     src = edge_index[0]==node
#     fanout_nodes = edge_index[1][src] #all fanouts of node

#     for f in fanout_nodes:
#         if not visited[f]:
#             visited[f] = True
#             res, nodes_in_path = find_nodes_in_cycle(cyclic_ff,f,edge_index,visited,nodes_in_path,flag=False)
#             if res:
#                 nodes_in_path.append(int(f))
#                 return res, nodes_in_path
#             else:
#                 not res, nodes_in_path
#     return False, nodes_in_path

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
    

def get_mlp_gate_dg2(args):
    return MLPGate_DG2(args)