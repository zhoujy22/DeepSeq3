import subprocess
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import remove_isolated_nodes
from torch_geometric.utils.convert import to_networkx

from .ordered_data import OrderedData,OrderedData2
from utils.dag_utils import return_order_info, map_node_ids
from utils.data_utils import construct_node_feature, add_skip_connection, add_edge_attr, one_hot

from utils.sat_utils import recursion_generation, one_hot_gate_type, write_dimacs_to
from utils.circuit_utils import  separate_ff_fanin_fanout, aig_simulation, get_cyclic_ff


def circuit_parse_pyg(x, edge_index, y, use_edge_attr=False, reconv_skip_connection=False, logic_diff_embedding="positional", predict_diff=False, diff_multiplier=10, no_node_cop=False, node_reconv=False, un_directed=False, num_gate_types=9, dim_edge_feature=32, logic_implication=False, mask=False):
    '''
    A function to parse the circuits and labels stored in `.npz` format to `Pytorch Geometric` Data.
    Optional, will add the skip connection, and the edge attributes into the graphs if specified.
    ...
    Parameters:
        x : numpy array with shape of [num_nodes, 9], the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 9, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level; 3rd - C1; 4th - C0; 5th - Obs; 6th - fan-out, 7th - boolean recovengence, 8th - index of the source node (-1 for non recovengence), 9th - masked or not and the masked value (-1, no masked). 
        edge_index : numpy array with shape of [num_edges, 2], thhe connectivity matrix.
        use_edge_attr : bool, whether to use the edge attributes.
        reconv_skip_connection: bool, whether to add the skip connection between source nodes and reconvergence nodes.
        logic_diff_embedding: str, the way to encode the discrete logic level.
        predict_diff : bool, whether to predict the difference between the simulated ground-truth probability and C1.
        diff_multiplier : int, the multiplier for the difference between the simulated ground-truth probability and C1.
        node_cop : bool, whether to use the C1 values as the node features.
        node_reconv : bool, whether to use the reconvergence info as the node features.
        dim_edge_feature : int, the dimension of node features.
        logic_implication: bool, whether to use the logic implication as the node feature or not.
        mask: bool, whether to use the masking of node embedding or not.
    Return:
        graph : torch_geometric.data.Data, the constructed pyG data.
    '''
    x_torch = construct_node_feature(x, no_node_cop, node_reconv, num_gate_types)
    y_torch = torch.tensor(y, dtype=torch.float)


    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = add_edge_attr(len(edge_index), dim_edge_feature, 1)
    if reconv_skip_connection:
        edge_index, edge_attr = add_skip_connection(x, edge_index, edge_attr, dim_edge_feature)
    
    if predict_diff:
        y_torch = (y_torch - torch.tensor(x[:, 3], dtype=torch.float).unsqueeze(1)) * diff_multiplier
    edge_index = edge_index.t().contiguous()
    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x_torch.size(0))

    if use_edge_attr:
        raise 'Unsupport edge attr'
    else:
        graph = OrderedData(x=x_torch, edge_index=edge_index, forward_level=forward_level, forward_index=forward_index, 
                            backward_level=backward_level, backward_index=backward_index)
        graph.y = y_torch.reshape([len(x_torch), 1])
        graph.use_edge_attr = False

    # add reconvegence info
    graph.rec = torch.tensor(x[:, 3:4], dtype=torch.float)
    graph.rec_src = torch.tensor(x[:, 4:5], dtype=torch.float)
    # add gt info
    graph.gt = torch.tensor(y, dtype=torch.float)
    # add indices for gate types
    graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)

    if un_directed:
        graph = ToUndirected()(graph)
    return graph

def circuitsat_parse_pyg(iclauses, n_vars, n_clauses, exp_depth):
    '''
    A function to parse the cnf to circuit to `Pytorch Geometric` Data.
    Input:
        iclause: clauses list
        n_vars: number of variables
        n_clauses: number of clauses
        exp_depth: CNF expansion depth
    Return:
        x: one_hot encoding of [PI, AND, OR, NOT]
        edge_index: edge connection pairs: each pair [x, y] from x to y
    '''
    x = []
    edge_index = []

    # PI and inv_PI
    x.append([])    # 0 is reserved
    inv2idx = {}
    for var_idx in range(1, n_vars+1, 1):
        x.append(one_hot_gate_type('PI'))
    has_inv = [0] * (n_vars+1)
    for clause in iclauses:
        for ele in clause:
            if ele < 0:
                has_inv[abs(ele)] = 1
    for var_idx in range(1, n_vars+1, 1):
        if has_inv[var_idx]:
            inv2idx[var_idx] = len(x)
            x.append(one_hot_gate_type('NOT'))
            edge_index.append([var_idx, inv2idx[var_idx]])

    # PO
    po_idx = len(x)
    x.append(one_hot_gate_type('OR'))
    iclauses_tmp = []
    for clause in iclauses:
        iclauses_tmp.append(clause.copy())
    recursion_generation(iclauses_tmp, po_idx, 0, exp_depth,
                         n_vars, x, edge_index, inv2idx)

    # Remove the reserved 0
    x = x[1: ]
    for edge in edge_index:
        edge[0] -= 1
        edge[1] -= 1
    
    # build the graph
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor([1])

    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x.size(0))
    
    graph = OrderedData(x=x, edge_index=edge_index, forward_level=forward_level, forward_index=forward_index, 
    backward_level=backward_level, backward_index=backward_index)

    # graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, n_vars=n_vars, n_clauses=n_clauses)
    graph.y, graph.n_vars, graph.n_clauses = y, n_vars, n_clauses

    return graph

def aig_parse_pyg(iclauses, n_vars, n_clauses, prob_loss):
    '''
    A function to parse the cnf to aig, then to circuit to `Pytorch Geometric` Data.
    Input:
        iclause: clauses list
        n_vars: number of variables
        n_clauses: number of clauses
    Return:
        x: one_hot encoding of [PI, AND, NOT]
        edge_index: edge connection pairs: each pair [x, y] from x to y
    For AIG, the nodes can be categorized as the Literal node, internal AND nodes, internal NOT node. The type values for each kind of nodes are as follows:
        * Literal input node: 0;
        * Internal AND nodes: 1;
        * Internal NOT nodes: 2;
    '''
    # step 1: store dimacs format
    dimacs_tmp = '/tmp/sat.dimacs'
    write_dimacs_to(n_vars, iclauses, dimacs_tmp)
    # step 2: dimacs to aig
    aig_tmp = '/tmp/sat.aig'
    subprocess.call(["./external/aiger/cnf2aig/cnf2aig", dimacs_tmp, aig_tmp])
    # step 3: aig to abc opimized aig
    aig_abc_tmp = '/tmp/aig_abc.aig'
    subprocess.call(["./external/abc/abc", "-c", "r %s; b; ps; b; rw -l; rw -lz; b; rw -lz; b; ps; cec; w %s" % (aig_tmp, aig_abc_tmp)])
    # step 4: aig to aag
    aag_abc_tmp = '/tmp/aig_abc.aag'
    subprocess.call(["./external/aiger/aiger/aigtoaig", aig_abc_tmp, aag_abc_tmp])
    # step 4: read aag
    with open(aag_abc_tmp, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(" ")
    assert header[0] == 'aag', 'The header of AIG file is wrong.'
    # “M”, “I”, “L”, “O”, “A” separated by spaces.
    n_variables = eval(header[1])
    n_inputs = eval(header[2])
    n_outputs = eval(header[4])
    n_and = eval(header[5])
    assert n_outputs == 1, 'The AIG has multiple outputs.'
    assert n_variables == (n_inputs + n_and), 'There are unused AND gates.'
    assert n_variables != n_inputs, '# variable equals to # inputs'
    # Construct AIG graph
    x = []
    edge_index = []
    # node_labels = []
    not_dict = {}
    
    # Add Literal node
    for i in range(n_inputs):
        x += [one_hot(0, 3)]
        # node_labels += [0]

    # Add AND node
    for i in range(n_inputs+1, n_inputs+1+n_and):
        x += [one_hot(1, 3)]
        # node_labels += [1]


    # sanity-check
    for (i, line) in enumerate(lines[1:1+n_inputs]):
        literal = line.strip().split(" ")
        assert len(literal) == 1, 'The literal of input should be single.'
        assert int(literal[0]) == 2 * (i + 1), 'The value of a input literal should be the index of variables mutiplying by two.'

    literal = lines[1+n_inputs].strip().split(" ")[0]
    assert int(literal) == (n_variables * 2) or int(literal) == (n_variables * 2) + 1, 'The value of the output literal shoud be (n_variables * 2)'
    sign_final = int(literal) % 2
    index_final_and = int(literal) // 2 - 1

    for (i, line) in enumerate(lines[2+n_inputs: 2+n_inputs+n_and]):
        literals = line.strip().split(" ")
        assert len(literals) == 3, 'invalidate the definition of two-input AND gate.'
        assert int(literals[0]) == 2 * (i + 1 + n_inputs)
    var_def = lines[2+n_variables].strip().split(" ")[0]

    assert var_def == 'i0', 'The definition of variables is wrong.'
    # finish sanity-check

    # Add edge
    for (i, line) in enumerate(lines[2+n_inputs: 2+n_inputs+n_and]):
        line = line.strip().split(" ")
        # assert len(line) == 3, 'The length of AND lines should be 3.'
        output_idx = int(line[0]) // 2 - 1
        # assert (int(line[0]) % 2) == 0, 'There is inverter sign in output literal.'

        # 1. First edge
        input1_idx = int(line[1]) // 2 - 1
        sign1_idx = int(line[1]) % 2
        # If there's a NOT node
        if sign1_idx == 1:
            if input1_idx in not_dict.keys():
                not_idx = not_dict[input1_idx]
            else:
                x += [one_hot(2, 3)]
                # node_labels += [2]
                not_idx = len(x) - 1
                not_dict[input1_idx] = not_idx
                edge_index += [[input1_idx, not_idx]]
            edge_index += [[not_idx, output_idx]]
        else:
            edge_index += [[input1_idx, output_idx]]


        # 2. Second edge
        input2_idx = int(line[2]) // 2 - 1
        sign2_idx = int(line[2]) % 2
        # If there's a NOT node
        if sign2_idx == 1:
            if input2_idx in not_dict.keys():
                not_idx = not_dict[input2_idx]
            else:
                x += [one_hot(2, 3)]
                # node_labels += [2]
                not_idx = len(x) - 1
                not_dict[input2_idx] = not_idx
                edge_index += [[input2_idx, not_idx]]
            edge_index += [[not_idx, output_idx]]
        else:
            edge_index += [[input2_idx, output_idx]]
    
    
    if sign_final == 1:
        x += [one_hot(2, 3)]
        # node_labels += [2]
        not_idx = len(x) - 1
        edge_index += [[index_final_and, not_idx]]
    
    # simulation
    if prob_loss:
        y_prob = aig_simulation(x, edge_index)
        y_prob = torch.tensor(y_prob, dtype=torch.float)

    x = torch.cat(x, dim=0).float()
    edge_index = torch.tensor(edge_index).t().contiguous()
    
    y = torch.tensor([1])

    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x.size(0))
    
    graph = OrderedData(x=x, edge_index=edge_index, forward_level=forward_level, forward_index=forward_index, 
    backward_level=backward_level, backward_index=backward_index)
    
    # Draw graph
    # nx_graph = to_networkx(graph)

    # import matplotlib.pyplot as plt
    # plt.figure(1,figsize=(14,12)) 
    # nx.draw(nx_graph, cmap=plt.get_cmap('Set1'), node_color = np.array(node_labels), node_size=75,linewidths=6)
    # plt.show()

    graph.y, graph.n_vars, graph.n_clauses = y, n_vars, n_clauses
    graph.y_prob = y_prob

    return graph

def parse_pyg_mlpgate(x, edge_index, y_trans_prob, y_prob1, y_prob0, tt_pair_index,  tt_diff, rc_pair_index, 
                      is_rc, ff_pair_index,is_ff_equ,ff_sim, use_edge_attr=False, reconv_skip_connection=False, 
                      no_node_cop=False, node_reconv=False, un_directed=False,num_gate_types=9, dim_edge_feature=32,
                      logic_implication=False, mask=False,  gate_to_index=[],test_data=False):

   
    x_torch = construct_node_feature(x, no_node_cop, node_reconv, num_gate_types)
 
    
    tt_pair_index = torch.tensor(tt_pair_index, dtype=torch.long)    
    tt_pair_index = tt_pair_index.t().contiguous()                         # shape[num_tt_pairs, 2]

    rc_pair_index = torch.tensor(rc_pair_index, dtype=torch.long)
    rc_pair_index = rc_pair_index.t().contiguous()

    tt_diff = torch.tensor(tt_diff)
    is_rc = torch.tensor(is_rc, dtype=torch.float32).unsqueeze(1)


    ff_pair_index = torch.tensor(ff_pair_index, dtype=torch.long)
    ff_pair_index = ff_pair_index.t().contiguous()
    ff_sim = torch.tensor(ff_sim)

    #edge_index, ff_fanin_edge_index, ff_fanout_edge_index = separate_ff_fanin(x, edge_index, gate_to_index, )
    #ff_fanout_edge_index = torch.tensor(ff_fanout_edge_index, dtype=torch.long)
    #ff_fanout_edge_index = ff_fanout_edge_index.t().contiguous()

    #removing cycle for making ordered data object
    # edge_index_without_ff_fanin, ff_fanin_edge_index, edge_index_without_ff_fout, ff_fanout_edge_data = separate_ff_fanin_fanout(x, edge_index, gate_to_index, ) 
    # ff_fanin_edge_index = torch.tensor(ff_fanin_edge_index, dtype=torch.long)
    # ff_fanin_edge_index = ff_fanin_edge_index.t().contiguous()

    # ff_fanout_edge_data = torch.tensor(ff_fanout_edge_data, dtype=torch.long)
    # ff_fanout_edge_data = ff_fanout_edge_data.t().contiguous()

    # if len(ff_fanin_edge_index.shape) == 1 or len(ff_fanout_edge_data.shape) == 1:
    #    return False, []
    

    # edge_index_without_ff_fout = torch.tensor(edge_index_without_ff_fout, dtype=torch.long)
    # edge_index_without_ff_fout = edge_index_without_ff_fout.t().contiguous()

    
    # edge_index_without_ff_fanin = torch.tensor(edge_index_without_ff_fanin, dtype=torch.long)
    
    edge_attr = add_edge_attr(len(edge_index), dim_edge_feature, 1)
    edge_index = torch.tensor(edge_index, dtype=torch.long) 
    edge_index_t = edge_index.t().contiguous()
    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index_t, x_torch.size(0))

    
    if use_edge_attr:
        raise 'Unsupport edge attr'
    else:
        graph = OrderedData(x=x_torch, edge_index=edge_index_t, edge_attr=edge_attr,
                            rc_pair_index=rc_pair_index, is_rc=is_rc,
                            tt_pair_index=tt_pair_index, tt_diff=tt_diff, 
                            ff_pair_index=ff_pair_index,
                            is_ff_equ= is_ff_equ,
                            ff_sim= ff_sim,
                            forward_level=forward_level, forward_index=forward_index, 
                            backward_level=backward_level, backward_index=backward_index,
                            )
        graph.use_edge_attr = False
 
    # add reconvegence info
    # graph.rec = torch.tensor(x[:, 3:4], dtype=torch.float)
    # graph.rec_src = torch.tensor(x[:, 4:5], dtype=torch.float)
    # add gt info
    # add indices for gate types
  
    graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)
   
    graph.y_prob1 = torch.tensor(y_prob1).reshape((len(x), 1))
    graph.y_prob0 = torch.tensor(y_prob0).reshape((len(x), 1))
    graph.y_trans_prob = torch.tensor(y_trans_prob, dtype=torch.float)

    graph.mask = torch.tensor(graph.y_prob1, dtype=torch.float)
    print(gate_to_index)
    graph.mask[(graph.gate != gate_to_index['INPUT']) & (graph.gate != gate_to_index['PDFF'])] = -1.0

    graph.seq_mask = graph.y_trans_prob

    init_mask = torch.empty(len(x),1).fill_(0)
    init_mask[(graph.gate != gate_to_index['INPUT']) & (graph.gate != gate_to_index['PDFF'])] = 1
    
    graph.seq_mask = init_mask * graph.seq_mask
   
    init_mask[(graph.gate == gate_to_index['INPUT']) & (graph.gate == gate_to_index['PDFF'])] = -1
    graph.seq_mask = graph.seq_mask + init_mask
    
    #print(torch.sum(graph.gate == gate_to_index['DFF']))

    if un_directed:
        graph = ToUndirected()(graph)
    return True, graph

def parse_pyg_mlpgate2(x, edge_index, y_trans_prob, y_prob,h_init1,h_init0,
                      use_edge_attr=False, reconv_skip_connection=False, 
                      no_node_cop=False, node_reconv=False, un_directed=False,num_gate_types=2, dim_edge_feature=32,
                      logic_implication=False, mask=False,  gate_to_index=[],test_data=False):

   
    x_torch = construct_node_feature(x, no_node_cop, node_reconv, num_gate_types)

    h_init1 = torch.tensor(h_init1, dtype=torch.float32)
    h_init0 = torch.tensor(h_init0, dtype=torch.float32)
    
    edge_attr = add_edge_attr(len(edge_index), dim_edge_feature, 1)
    edge_index = torch.tensor(edge_index, dtype=torch.long) 
    edge_index_t = edge_index.t().contiguous()
    forward_level = torch.arange(x_torch.size(0))
    forward_index = torch.arange(x_torch.size(0))
    backward_level = torch.arange(x_torch.size(0))
    backward_index = torch.arange(x_torch.size(0))
    
    if use_edge_attr:
        raise 'Unsupport edge attr'
    else:
        graph = OrderedData2(x=x_torch, edge_index=edge_index_t, edge_attr=edge_attr,
                            h_init1=h_init1, h_init0=h_init0,
                            forward_level=forward_level, forward_index=forward_index, 
                            backward_level=backward_level, backward_index=backward_index,
                            )
 
    graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)
   
    graph.y_prob = torch.tensor(y_prob).reshape((len(x), 1))
    graph.y_trans_prob = torch.tensor(y_trans_prob, dtype=torch.float)

    print("Done Ordering")
    

    if un_directed:
        graph = ToUndirected()(graph)
    return True, graph
def parse_pyg_mlpgate3(x, edge_index, y_trans_prob, y_prob,
                      use_edge_attr=False, reconv_skip_connection=False, 
                      no_node_cop=False, node_reconv=False, un_directed=False,num_gate_types=2, dim_edge_feature=32,
                      logic_implication=False, mask=False,  gate_to_index=[],test_data=False):

   
    x_torch = construct_node_feature(x, no_node_cop, node_reconv, num_gate_types)

    
    edge_attr = add_edge_attr(len(edge_index), dim_edge_feature, 1)
    edge_index = torch.tensor(edge_index, dtype=torch.long) 
    edge_index_t = edge_index.t().contiguous()
    forward_level = torch.arange(x_torch.size(0))
    forward_index = torch.arange(x_torch.size(0))
    backward_level = torch.arange(x_torch.size(0))
    backward_index = torch.arange(x_torch.size(0))
    
    if use_edge_attr:
        raise 'Unsupport edge attr'
    else:
        graph = OrderedData2(x=x_torch, edge_index=edge_index_t, edge_attr=edge_attr,
                            forward_level=forward_level, forward_index=forward_index, 
                            backward_level=backward_level, backward_index=backward_index,
                            )
 
    graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)
    
    graph.y_prob = torch.tensor(y_prob).reshape((len(x), 1))
    graph.y_trans_prob = torch.tensor(y_trans_prob, dtype=torch.float)
    graph.mask = torch.tensor(graph.y_prob, dtype=torch.float)

    graph.mask[(graph.gate != gate_to_index['INPUT']) & (graph.gate != gate_to_index['PDFF'])] = -1.0

    graph.seq_mask = graph.y_trans_prob

    init_mask = torch.empty(len(x),1).fill_(0)
    init_mask[(graph.gate != gate_to_index['INPUT']) & (graph.gate != gate_to_index['PDFF'])] = 1
    
    graph.seq_mask = init_mask * graph.seq_mask
   
    init_mask[(graph.gate != gate_to_index['INPUT']) & (graph.gate != gate_to_index['PDFF'])] = -1
    graph.seq_mask = graph.seq_mask + init_mask
    print("Done Ordering")
    

    if un_directed:
        graph = ToUndirected()(graph)
    return True, graph
"""
def parse_pyg_mlpgate_sat(x, edge_index, y_trans_prob, y_prob, tt_pair_index,  tt_diff, 
    use_edge_attr=False, reconv_skip_connection=False, no_node_cop=False, node_reconv=False, un_directed=False,
      num_gate_types=9, dim_edge_feature=32, logic_implication=False, mask=False,  gate_to_index=[],test_data=False):
   
    x_torch = construct_node_feature(x, no_node_cop, node_reconv, num_gate_types)
 
    
    tt_pair_index = torch.tensor(tt_pair_index, dtype=torch.long)
    tt_pair_index = tt_pair_index.t().contiguous()


    tt_diff = torch.tensor(tt_diff)
    
    #edge_index, ff_fanin_edge_index, ff_fanout_edge_index = separate_ff_fanin(x, edge_index, gate_to_index, )
    #ff_fanout_edge_index = torch.tensor(ff_fanout_edge_index, dtype=torch.long)
    #ff_fanout_edge_index = ff_fanout_edge_index.t().contiguous()

    #removing cycle for making ordered data object
    edge_index_without_ff_fanin, ff_fanin_edge_index, edge_index_without_ff_fout, ff_fanout_edge_data = separate_ff_fanin_fanout(x, edge_index, gate_to_index, ) 
    ff_fanin_edge_index = torch.tensor(ff_fanin_edge_index, dtype=torch.long)
    ff_fanin_edge_index = ff_fanin_edge_index.t().contiguous()

    ff_fanout_edge_data = torch.tensor(ff_fanout_edge_data, dtype=torch.long)
    ff_fanout_edge_data = ff_fanout_edge_data.t().contiguous()

    if len(ff_fanin_edge_index.shape) == 1 or len(ff_fanout_edge_data.shape) == 1:
       return False, []
    

    edge_index_without_ff_fout = torch.tensor(edge_index_without_ff_fout, dtype=torch.long)
    edge_index_without_ff_fout = edge_index_without_ff_fout.t().contiguous()

    
    edge_index_without_ff_fanin = torch.tensor(edge_index_without_ff_fanin, dtype=torch.long)

    edge_attr = add_edge_attr(len(edge_index_without_ff_fanin), dim_edge_feature, 1)
    if reconv_skip_connection:
        edge_index_without_ff_fanin, edge_attr = add_skip_connection(x, edge_index_without_ff_fanin, edge_attr, dim_edge_feature)
    
    edge_index_without_ff_fanin = edge_index_without_ff_fanin.t().contiguous()


    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index_without_ff_fanin, x_torch.size(0))

    # for cycles
    #no_cycle, cyclic_FFs_nodes, edges_in_cyclic_FFs = get_ff_cycles(x, edge_index, gate_to_index)
    #no_cycle, cyclic_FFs_nodes = get_cyclic_ff(x, edge_index, gate_to_index)
    #print(no_cycle)
    #cyclic_FFs_nodes = torch.tensor(cyclic_FFs_nodes, dtype=torch.long)
    cyclic_FFs_nodes = []
    no_cycle = 0

    original_edge_index = torch.tensor(edge_index, dtype=torch.long)
    original_edge_index = original_edge_index.t().contiguous()



    #Find cyclic subgraph 
    if test_data:
        nodes_in_cycles = torch.ones(len(x), len(x))
        nodes_in_cycles  = nodes_in_cycles.fill_(False)
        
        one_path = torch.tensor([False] * len(x))
        cycle_path = torch.tensor([False] * len(x))
        for node in ff_fanin_edge_index[1]:         
            visited = [False] * len(x)
            cycle_path = torch.tensor([False] * len(x))
            if cyclic_FFs_nodes[node]:
                cycle_path = find_nodes_in_cycle(int(node), int(node), original_edge_index, visited, one_path, cycle_path)
                nodes_in_cycles[node,:] += cycle_path

    if use_edge_attr:
        raise 'Unsupport edge attr'
    else:
        graph = OrderedData(x=x_torch, edge_index=edge_index_without_ff_fanin, 
                            ff_fanin_edge_index=ff_fanin_edge_index, #ff_fanout_edge_index=ff_fanout_edge_index,
                           
                            tt_pair_index=tt_pair_index, tt_diff=tt_diff, 
                            
                            no_cycle= no_cycle,
                            cyclic_FFs_nodes =  cyclic_FFs_nodes, 
                            #edges_in_cyclic_FFs = edges_in_cyclic_FFs,
                            original_edge_index =  original_edge_index,
                            forward_level=forward_level, forward_index=forward_index, 
                            backward_level=backward_level, backward_index=backward_index,
                            )
        graph.use_edge_attr = False
 
    # add reconvegence info
    # graph.rec = torch.tensor(x[:, 3:4], dtype=torch.float)
    # graph.rec_src = torch.tensor(x[:, 4:5], dtype=torch.float)
    # add gt info
    # add indices for gate types
    if test_data:
        graph.nodes_in_cycles = nodes_in_cycles
    
    # Stone add: 

  
    graph.ff_fanin_edge_index = ff_fanin_edge_index
    graph.cyclic_FFs_nodes = cyclic_FFs_nodes
  
    graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)
   
    graph.prob = torch.tensor(y_prob).reshape((len(x), 1))
    graph.y_trans_prob = torch.tensor(y_trans_prob, dtype=torch.float)

    graph.mask = torch.tensor(graph.prob, dtype=torch.float)
  
    graph.mask[graph.gate != gate_to_index['PI']] = -1.0

    graph.seq_mask   = graph.y_trans_prob

    init_mask = torch.empty(len(x),1).fill_(0)
    init_mask[graph.gate == gate_to_index['PI']] = 1
    
    graph.seq_mask = init_mask * graph.seq_mask
   
    init_mask[graph.gate != gate_to_index['PI']] = -1
    graph.seq_mask = graph.seq_mask + init_mask
    
    #print(torch.sum(graph.gate == gate_to_index['DFF']))

    if un_directed:
        graph = ToUndirected()(graph)
    return True, graph



def parse_pyg_mlpgate_simple(x, edge_index, y_trans_prob, y_prob,
    use_edge_attr=False, reconv_skip_connection=False, no_node_cop=False, node_reconv=False, un_directed=False,
      num_gate_types=9, dim_edge_feature=32, logic_implication=False, mask=False,  gate_to_index=[]):
   
    x_torch = construct_node_feature(x, no_node_cop, node_reconv, num_gate_types)

    
    
    #edge_index, ff_fanin_edge_index, ff_fanout_edge_index = separate_ff_fanin(x, edge_index, gate_to_index, )
    #ff_fanout_edge_index = torch.tensor(ff_fanout_edge_index, dtype=torch.long)
    #ff_fanout_edge_index = ff_fanout_edge_index.t().contiguous()

    #removing cycle for making ordered data object
    edge_index_without_ff_fanin, ff_fanin_edge_index, edge_index_without_ff_fout, ff_fanout_edge_data = separate_ff_fanin_fanout(x, edge_index, gate_to_index, ) 
    ff_fanin_edge_index = torch.tensor(ff_fanin_edge_index, dtype=torch.long)
    ff_fanin_edge_index = ff_fanin_edge_index.t().contiguous()

    ff_fanout_edge_data = torch.tensor(ff_fanout_edge_data, dtype=torch.long)
    ff_fanout_edge_data = ff_fanout_edge_data.t().contiguous()

    if len(ff_fanin_edge_index.shape) == 1 or len(ff_fanout_edge_data.shape) == 1:
       return False, []
    

    edge_index_without_ff_fout = torch.tensor(edge_index_without_ff_fout, dtype=torch.long)
    edge_index_without_ff_fout = edge_index_without_ff_fout.t().contiguous()

    
    edge_index_without_ff_fanin = torch.tensor(edge_index_without_ff_fanin, dtype=torch.long)

    edge_attr = add_edge_attr(len(edge_index_without_ff_fanin), dim_edge_feature, 1)
    if reconv_skip_connection:
        edge_index_without_ff_fanin, edge_attr = add_skip_connection(x, edge_index_without_ff_fanin, edge_attr, dim_edge_feature)
    
    edge_index_without_ff_fanin = edge_index_without_ff_fanin.t().contiguous()


    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index_without_ff_fanin, x_torch.size(0))

    # for cycles
    #no_cycle, cyclic_FFs_nodes, edges_in_cyclic_FFs = get_ff_cycles(x, edge_index, gate_to_index)
    #no_cycle, cyclic_FFs_nodes = get_cyclic_ff(x, edge_index, gate_to_index)

    #edges_in_cyclic_FFs = torch.tensor(edges_in_cyclic_FFs, dtype=torch.long)
    #edges_in_cyclic_FFs = edges_in_cyclic_FFs.t().contiguous()
    
    #cyclic_FFs_nodes = torch.tensor(cyclic_FFs_nodes, dtype=torch.long)
    
    
    #original_edge_index = torch.tensor(edge_index, dtype=torch.long)
    #original_edge_index = original_edge_index.t().contiguous()

    if use_edge_attr:
        raise 'Unsupport edge attr'
    else:
        graph = OrderedData(x=x_torch, edge_index=edge_index_without_ff_fanin, 
                            ff_fanin_edge_index=ff_fanin_edge_index, #ff_fanout_edge_index=ff_fanout_edge_index,
                            #rc_pair_index=rc_pair_index, is_rc=is_rc,
                            #tt_pair_index=tt_pair_index, tt_diff=tt_diff, 
                            #diff_pair_index = diff_pair_index,
                            #is_trans_diff = is_trans_diff,
                            #trans_state_diff = trans_state_diff,
                            #no_cycle= no_cycle,
                            #cyclic_FFs_nodes =  cyclic_FFs_nodes, 
                            #edges_in_cyclic_FFs = edges_in_cyclic_FFs,
                            #original_edge_index =  original_edge_index,
                            forward_level=forward_level, forward_index=forward_index, 
                            backward_level=backward_level, backward_index=backward_index)
        graph.use_edge_attr = False
 
    # add reconvegence info
    # graph.rec = torch.tensor(x[:, 3:4], dtype=torch.float)
    # graph.rec_src = torch.tensor(x[:, 4:5], dtype=torch.float)
    # add gt info
    # add indices for gate types
    
    # Stone add: 
  
    graph.ff_fanin_edge_index = ff_fanin_edge_index
  
    graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)
   
    graph.prob = torch.tensor(y_prob).reshape((len(x), 1))
    graph.y_trans_prob = torch.tensor(y_trans_prob, dtype=torch.float)

    graph.mask = torch.tensor(graph.prob, dtype=torch.float)
  
    graph.mask[graph.gate != gate_to_index['PI']] = -1.0

    graph.seq_mask   = graph.y_trans_prob

    init_mask = torch.empty(len(x),1).fill_(0)
    init_mask[graph.gate == gate_to_index['PI']] = 1
    
    graph.seq_mask = init_mask * graph.seq_mask
   
    init_mask[graph.gate != gate_to_index['PI']] = -1
    graph.seq_mask = graph.seq_mask + init_mask
 
   
    if un_directed:
        graph = ToUndirected()(graph)
    return True, graph


def parse_pyg_mlpgate_lec(x, edge_index, gnn_rounds, y_trans_prob, y_prob, tt_pair_index,  tt_diff, rc_pair_index, is_rc, diff_pair_index, is_trans_diff, trans_state_diff,
              ff_pair_index,is_ff_equ,ff_sim,            
    use_edge_attr=False, reconv_skip_connection=False, no_node_cop=False, node_reconv=False, un_directed=False,
      num_gate_types=9, dim_edge_feature=32, logic_implication=False, mask=False,  gate_to_index=[],test_data=False):
   
    x_torch = construct_node_feature(x, no_node_cop, node_reconv, num_gate_types)

  
    tt_pair_index = torch.tensor(tt_pair_index, dtype=torch.long)
    tt_pair_index = tt_pair_index.t().contiguous()
    rc_pair_index = torch.tensor(rc_pair_index, dtype=torch.long)
    rc_pair_index = rc_pair_index.t().contiguous()
    tt_diff = torch.tensor(tt_diff)
    is_rc = torch.tensor(is_rc, dtype=torch.float32).unsqueeze(1)
    

    diff_pair_index = torch.tensor(diff_pair_index, dtype=torch.long)
    diff_pair_index = diff_pair_index.t().contiguous()
    is_trans_diff = torch.tensor(is_trans_diff)
    trans_state_diff = torch.tensor(trans_state_diff)
    

    #ff_pair_index,is_ff_equ,ff_sim,
    #print(is_ff_equ)
    
    ff_pair_index = torch.tensor(ff_pair_index, dtype=torch.long)
    ff_pair_index = ff_pair_index.t().contiguous()
    ff_sim = torch.tensor(ff_sim)
    
 
    #edge_index, ff_fanin_edge_index, ff_fanout_edge_index = separate_ff_fanin(x, edge_index, gate_to_index, )
    #ff_fanout_edge_index = torch.tensor(ff_fanout_edge_index, dtype=torch.long)
    #ff_fanout_edge_index = ff_fanout_edge_index.t().contiguous()

    #removing cycle for making ordered data object
    edge_index_without_ff_fanin, ff_fanin_edge_index, edge_index_without_ff_fout, ff_fanout_edge_data = separate_ff_fanin_fanout(x, edge_index, gate_to_index, ) 
    ff_fanin_edge_index = torch.tensor(ff_fanin_edge_index, dtype=torch.long)
    ff_fanin_edge_index = ff_fanin_edge_index.t().contiguous()

    ff_fanout_edge_data = torch.tensor(ff_fanout_edge_data, dtype=torch.long)
    ff_fanout_edge_data = ff_fanout_edge_data.t().contiguous()

    if len(ff_fanin_edge_index.shape) == 1 or len(ff_fanout_edge_data.shape) == 1:
       return False, []
    

    edge_index_without_ff_fout = torch.tensor(edge_index_without_ff_fout, dtype=torch.long)
    edge_index_without_ff_fout = edge_index_without_ff_fout.t().contiguous()

    
    edge_index_without_ff_fanin = torch.tensor(edge_index_without_ff_fanin, dtype=torch.long)

    edge_attr = add_edge_attr(len(edge_index_without_ff_fanin), dim_edge_feature, 1)
    if reconv_skip_connection:
        edge_index_without_ff_fanin, edge_attr = add_skip_connection(x, edge_index_without_ff_fanin, edge_attr, dim_edge_feature)
    
    edge_index_without_ff_fanin = edge_index_without_ff_fanin.t().contiguous()


    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index_without_ff_fanin, x_torch.size(0))

    # for cycles
    #no_cycle, cyclic_FFs_nodes, edges_in_cyclic_FFs = get_ff_cycles(x, edge_index, gate_to_index)
    no_cycle, cyclic_FFs_nodes = get_cyclic_ff(x, edge_index, gate_to_index)
    cyclic_FFs_nodes = torch.tensor(cyclic_FFs_nodes, dtype=torch.long)
    
    original_edge_index = torch.tensor(edge_index, dtype=torch.long)
    original_edge_index = original_edge_index.t().contiguous()



    #Find cyclic subgraph 
    if test_data:
        nodes_in_cycles = torch.ones(len(x), len(x))
        nodes_in_cycles  = nodes_in_cycles.fill_(False)
        
        one_path = torch.tensor([False] * len(x))
        cycle_path = torch.tensor([False] * len(x))
        for node in ff_fanin_edge_index[1]:         
            visited = [False] * len(x)
            cycle_path = torch.tensor([False] * len(x))
            if cyclic_FFs_nodes[node]:
                cycle_path = find_nodes_in_cycle(int(node), int(node), original_edge_index, visited, one_path, cycle_path)
                nodes_in_cycles[node,:] += cycle_path

    if use_edge_attr:
        raise 'Unsupport edge attr'
    else:
        graph = OrderedData(x=x_torch, edge_index=edge_index_without_ff_fanin, 
                            ff_fanin_edge_index=ff_fanin_edge_index, #ff_fanout_edge_index=ff_fanout_edge_index,
                            rc_pair_index=rc_pair_index, is_rc=is_rc,
                            tt_pair_index=tt_pair_index, tt_diff=tt_diff, 
                            diff_pair_index = diff_pair_index,
                            is_trans_diff = is_trans_diff,
                            trans_state_diff = trans_state_diff,
                            ff_pair_index = ff_pair_index,
                            no_cycle= no_cycle,
                            cyclic_FFs_nodes =  cyclic_FFs_nodes, 
                            #edges_in_cyclic_FFs = edges_in_cyclic_FFs,
                            original_edge_index =  original_edge_index,
                            forward_level=forward_level, forward_index=forward_index, 
                            backward_level=backward_level, backward_index=backward_index,
                            )
        graph.use_edge_attr = False
 
    # add reconvegence info
    # graph.rec = torch.tensor(x[:, 3:4], dtype=torch.float)
    # graph.rec_src = torch.tensor(x[:, 4:5], dtype=torch.float)
    # add gt info
    # add indices for gate types
    if test_data:
        graph.nodes_in_cycles = nodes_in_cycles
    
    # Stone add: 

    #graph.diff_pair_index = diff_pair_index
    #graph.is_trans_diff = is_trans_diff
    #graph.trans_state_diff = trans_state_diff
    graph.ff_fanin_edge_index = ff_fanin_edge_index
    graph.cyclic_FFs_nodes = cyclic_FFs_nodes
    graph.ff_pair_index = ff_pair_index
    graph.ff_sim = ff_sim
  
    graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)
   
    graph.prob = torch.tensor(y_prob).reshape((len(x), 1))
    graph.y_trans_prob = torch.tensor(y_trans_prob, dtype=torch.float)

    graph.mask = torch.tensor(graph.prob, dtype=torch.float)
  
    graph.mask[graph.gate != gate_to_index['PI']] = -1.0

    graph.seq_mask   = graph.y_trans_prob

    init_mask = torch.empty(len(x),1).fill_(0)
    init_mask[graph.gate == gate_to_index['PI']] = 1
    
    graph.seq_mask = init_mask * graph.seq_mask
   
    init_mask[graph.gate != gate_to_index['PI']] = -1
    graph.seq_mask = graph.seq_mask + init_mask
    
    
    if un_directed:
        graph = ToUndirected()(graph)
    return True, graph
"""

def parse_pyg_mlpgate_finetune_pe(x, edge_index, y_trans_prob, y_prob, 
    use_edge_attr=False, reconv_skip_connection=False, no_node_cop=False, node_reconv=False, un_directed=False,
      num_gate_types=9, dim_edge_feature=32, logic_implication=False, mask=False,  gate_to_index=[],first = False, test_data=False):
   
    x_torch = construct_node_feature(x, no_node_cop, node_reconv, num_gate_types)

    
   
    edge_index_without_ff_fanin, ff_fanin_edge_index, edge_index_without_ff_fout, ff_fanout_edge_data = separate_ff_fanin_fanout(x, edge_index, gate_to_index, ) 
    ff_fanin_edge_index = torch.tensor(ff_fanin_edge_index, dtype=torch.long)
    ff_fanin_edge_index = ff_fanin_edge_index.t().contiguous()

    ff_fanout_edge_data = torch.tensor(ff_fanout_edge_data, dtype=torch.long)
    ff_fanout_edge_data = ff_fanout_edge_data.t().contiguous()

    if len(ff_fanin_edge_index.shape) == 1 or len(ff_fanout_edge_data.shape) == 1:
       return False, []
    

    edge_index_without_ff_fout = torch.tensor(edge_index_without_ff_fout, dtype=torch.long)
    edge_index_without_ff_fout = edge_index_without_ff_fout.t().contiguous()

    
    edge_index_without_ff_fanin = torch.tensor(edge_index_without_ff_fanin, dtype=torch.long)

    edge_attr = add_edge_attr(len(edge_index_without_ff_fanin), dim_edge_feature, 1)
    if reconv_skip_connection:
        edge_index_without_ff_fanin, edge_attr = add_skip_connection(x, edge_index_without_ff_fanin, edge_attr, dim_edge_feature)
    
    edge_index_without_ff_fanin = edge_index_without_ff_fanin.t().contiguous()


    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index_without_ff_fanin, x_torch.size(0))

    # for cycles
    #no_cycle, cyclic_FFs_nodes, edges_in_cyclic_FFs = get_ff_cycles(x, edge_index, gate_to_index)
    no_cycle, cyclic_FFs_nodes = get_cyclic_ff(x, edge_index, gate_to_index)
    cyclic_FFs_nodes = torch.tensor(cyclic_FFs_nodes, dtype=torch.long)
    
    original_edge_index = torch.tensor(edge_index, dtype=torch.long)
    original_edge_index = original_edge_index.t().contiguous()


    #Find cyclic subgraph 
    #if test_data and first:
    if test_data:
        nodes_in_cycles = torch.ones(len(x), len(x))
        nodes_in_cycles  = nodes_in_cycles.fill_(False)
        
        one_path = torch.tensor([False] * len(x))
        cycle_path = torch.tensor([False] * len(x))
        for node in ff_fanin_edge_index[1]:         
            visited = [False] * len(x)
            cycle_path = torch.tensor([False] * len(x))
            if cyclic_FFs_nodes[node]:
                cycle_path = find_nodes_in_cycle(int(node), int(node), original_edge_index, visited, one_path, cycle_path)
                nodes_in_cycles[node,:] += cycle_path

    if use_edge_attr:
        raise 'Unsupport edge attr'
    else:
        graph = OrderedData(x=x_torch, edge_index=edge_index_without_ff_fanin, 
                            ff_fanin_edge_index=ff_fanin_edge_index, #ff_fanout_edge_index=ff_fanout_edge_index,
                          
                            no_cycle= no_cycle,
                            cyclic_FFs_nodes =  cyclic_FFs_nodes, 
                            #edges_in_cyclic_FFs = edges_in_cyclic_FFs,
                            original_edge_index =  original_edge_index,
                            forward_level=forward_level, forward_index=forward_index, 
                            backward_level=backward_level, backward_index=backward_index,
                            )
        graph.use_edge_attr = False
 
    # add reconvegence info
    # graph.rec = torch.tensor(x[:, 3:4], dtype=torch.float)
    # graph.rec_src = torch.tensor(x[:, 4:5], dtype=torch.float)
    # add gt info
    # add indices for gate types
    #if test_data and first:
    if test_data:
        graph.nodes_in_cycles = nodes_in_cycles
    
    # Stone add: 

    graph.ff_fanin_edge_index = ff_fanin_edge_index
    graph.cyclic_FFs_nodes = cyclic_FFs_nodes
  
    graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)
   
    graph.prob = torch.tensor(y_prob).reshape((len(x), 1))
    graph.y_trans_prob = torch.tensor(y_trans_prob, dtype=torch.float)

    graph.mask = torch.tensor(graph.prob, dtype=torch.float)
  
    graph.mask[graph.gate != gate_to_index['PI']] = -1.0

    graph.seq_mask   = graph.y_trans_prob

    init_mask = torch.empty(len(x),1).fill_(0)
    init_mask[graph.gate == gate_to_index['PI']] = 1
    
    graph.seq_mask = init_mask * graph.seq_mask
   
    init_mask[graph.gate != gate_to_index['PI']] = -1
    graph.seq_mask = graph.seq_mask + init_mask
    
    
    if un_directed:
        graph = ToUndirected()(graph)
    return True, graph


def find_nodes_in_cycle(cyclic_ff, node, edge_index, visited, one_path,nodes_in_path, first_flag=True):
    #print("calling dfs")
    if cyclic_ff == int(node) and not first_flag:
      
        nodes_in_path = nodes_in_path + one_path
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