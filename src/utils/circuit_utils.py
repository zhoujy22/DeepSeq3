'''
Utility functions for circuit: including random pattern generation, logic simulator, \
    reconvergence identification, 
'''
import torch
from numpy.random import randint
import copy
from collections import Counter
from utils.graph_utils import detect_cycles
import random
import numpy as np
def dec2list(num, no_PIs):
    res = []
    bin_num = bin(num)[2:].zfill(no_PIs)
    for ele in bin_num:
        res.append(int(ele))
    return res

def read_file(file_name):
    f = open(file_name, "r")
    data = f.readlines()
    return data

def random_pattern_generator(no_PIs):
    vector = [0] * no_PIs
    p = np.random.rand()
    vector = np.random.choice(a=[0, 1], p=[p, 1-p], size=no_PIs)
    return vector


def logic(gate_type, signals, gate_to_index):
    if 'AND' in gate_to_index.keys() and gate_type == gate_to_index['AND']:  # AND
        for s in signals:
            if s == 0:
                return 0
        return 1

    elif 'NAND' in gate_to_index.keys() and gate_type == gate_to_index['NAND']:  # NAND
        for s in signals:
            if s == 0:
                return 1
        return 0

    elif 'OR' in gate_to_index.keys() and gate_type == gate_to_index['OR']:  # OR
        for s in signals:
            if s == 1:
                return 1
        return 0

    elif 'NOR' in gate_to_index.keys() and gate_type == gate_to_index['NOR']:  # NOR
        for s in signals:
            if s == 1:
                return 0
        return 1

    elif 'NOT' in gate_to_index.keys() and gate_type == gate_to_index['NOT']:  # NOT
        for s in signals:
            if s == 1:
                return 0
            else:
                return 1

    elif 'XOR' in gate_to_index.keys() and gate_type == gate_to_index['XOR']:  # XOR
        z_count = 0
        o_count = 0
        for s in signals:
            if s == 0:
                z_count = z_count + 1
            elif s == 1:
                o_count = o_count + 1
        if z_count == len(signals) or o_count == len(signals):
            return 0
        return 1
    else:
        for s in signals:
            return s
def prob_logic(gate_type, signals):
    '''
    Function to calculate Controlability values, i.e. C1 and C0 for the given node.
    
    ...
    Parameters:
        gate_type: int, the integer index for the target node.
        signals : list(float), the values for the fan-in signals
    Return:
        zero: float, C0
        one: flaot, C1
    '''
    one = 0.0
    zero = 0.0

    if gate_type == 1:  # AND
        mul = 1.0
        for s in signals:
            mul = mul * s[1]
        one = mul
        zero = 1.0 - mul

    elif gate_type == 2:  # NAND
        mul = 1.0
        for s in signals:
            mul = mul * s[1]
        zero = mul
        one = 1.0 - mul

    elif gate_type == 3:  # OR
        mul = 1.0
        for s in signals:
            mul = mul * s[0]
        zero = mul
        one = 1.0 - mul

    elif gate_type == 4:  # NOR
        mul = 1.0
        for s in signals:
            mul = mul * s[0]
        one = mul
        zero = 1.0 - mul

    elif gate_type == 5:  # NOT
        for s in signals:
            one = s[0]
            zero = s[1]

    elif gate_type == 6:  # XOR
        mul0 = 1.0
        mul1 = 1.0
        for s in signals:
            mul0 = mul0 * s[0]
        for s in signals:
            mul1 = mul1 * s[1]

        zero = mul0 + mul1
        one = 1.0 - zero

    return zero, one


# TODO: correct observability logic
def obs_prob(x, r, y, input_signals):
    if x[r][1] == 1 or x[r][1] == 2:
        obs = y[r]
        for s in input_signals:
            for s1 in input_signals:
                if s != s1:
                    obs = obs * x[s1][3]
            if obs < y[s] or y[s] == -1:
                y[s] = obs

    elif x[r][1] == 3 or x[r][1] == 4:
        obs = y[r]
        for s in input_signals:
            for s1 in input_signals:
                if s != s1:
                    obs = obs * x[s1][4]
            if obs < y[s] or y[s] == -1:
                y[s] = obs

    elif x[r][1] == 5:
        obs = y[r]
        for s in input_signals:
            if obs < y[s] or y[s] == -1:
                y[s] = obs

    elif x[r][1] == 6:
        if len(input_signals) != 2:
            print('Not support non 2-input XOR Gate')
            raise
        # computing for a node
        obs = y[r]
        s = input_signals[1]
        if x[s][3] > x[s][4]:
            obs = obs * x[s][3]
        else:
            obs = obs * x[s][4]
        y[input_signals[0]] = obs

        # computing for b node
        obs = y[r]
        s = input_signals[0]
        if x[s][3] > x[s][4]:
            obs = obs * x[s][3]
        else:
            obs = obs * x[s][4]
        y[input_signals[1]] = obs

    return y



def simulator(x_data, PI_indexes, level_list, fanin_list, num_patterns):
    '''
       Logic simulator
       ...
       Parameters:
           x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level; 3rd - C1, 4th - C0, 5th - Obs.
           level_list: logic levels
           fanin_list: the fanin node indexes for each node
           fanout_list: the fanout node indexes for each node
       Return:
           y_data : simualtion result
       '''
    y = [0] * len(x_data)
    y1 = [0] * len(x_data)
    pattern_count = 0
    no_of_patterns = min(num_patterns, 10 * pow(2, len(PI_indexes)))
    print('No of Patterns: {:}'.format(no_of_patterns))

    print('[INFO] Begin simulation')
    while pattern_count < no_of_patterns:
        input_vector = random_pattern_generator(len(PI_indexes))

        j = 0
        for i in PI_indexes:
            y[i] = input_vector[j]
            j = j + 1

        for level in range(1, len(level_list), 1):
            for node_idx in level_list[level]:
                source_signals = []
                for pre_idx in fanin_list[node_idx]:
                    source_signals.append(y[pre_idx])
                if len(source_signals) > 0:
                    gate_type = x_data[node_idx][1]
                    y[node_idx] = logic(gate_type, source_signals)
                    if y[node_idx] == 1:
                        y1[node_idx] = y1[node_idx] + 1

        pattern_count = pattern_count + 1
        if pattern_count % 10000 == 0:
            print("pattern count = {:}k".format(int(pattern_count / 1000)))

    for i, _ in enumerate(y1):
        y1[i] = [y1[i] / pattern_count]

    for i in PI_indexes:
        y1[i] = [0.5]

    return y1



def get_gate_type(line, gate_to_index):
    '''
    Function to get the interger index of the gate type.
    ...
    Parameters:
        line : str, the single line in the bench file.
        gate_to_index: dict, the mapping from the gate name to the integer index
    Return:
        vector_row : int, the integer index for the gate. Currently consider 7 gate types.
    '''
    vector_row = -1
    for (gate_name, index) in gate_to_index.items():
        if gate_name  in line:
            vector_row = index

    if vector_row == -1:
        raise KeyError('[ERROR] Find unsupported gate')

    return vector_row


def add_node_index(data):
    '''
    A pre-processing function to handle with the `.bench` format files.
    Will add the node index before the line, and also calculate the total number of nodes.
    Modified .
    ...
    Parameters:
        data : list(str), the lines read out from a bench file
    Return:
        data : list(str), the updated lines for a circuit
        node_index: int, the number of the circuits, not considering `OUTPUT` lines.
        index_map: dict(int:int), the mapping from the original node name to the updated node index.
    '''
    node_index = 0
    x_data = {}  # the node feature matrix, will be updated
    index_map = {}
    output_index = []
    for i, val in enumerate(data):
        # node level and index  for PI
        if val.startswith('#'):
            continue
        if "INPUT" in val:
            node_name = val.split("(")[1].split(")")[0].strip()
            index_map[node_name] = str(node_index)
            x_data[node_index] = str(node_index) + ":" + val[:-1] #+ ";0"
            node_index += 1
        if "OUTPUT" in val:
            node_name = "O" + val.split("(")[1].split(")")[0].strip()
            index_map[node_name] = str(node_index)
            x_data[node_index] = str(node_index) + ":" + "OUTPUT(" + node_name + ")" #+ ";0"
            node_index += 1
        # index for gate nodes
        if ("= NAND" in val) or ("= NOR" in val) or ("= AND" in val) or ("= OR" in val) or (
                "= NOT" in val) or ("= XOR" in val):
            node_name = val.split("=")[0].strip()
            index_map[node_name] = str(node_index)
            x_data[node_index] = str(node_index) + ":" + val[:-1] #data[i] = node_index:node_name = cell_type(connect_pins)
            node_index += 1
        elif "DFF" in val:
            node_name = val.split("=")[0].strip()
            index_map[node_name] = str(node_index)
            x_data[node_index] = str(node_index) + ":" +val[:-1]
            node_index += 1
            pnode_name = 'P' + val.split("=")[0].strip()
            index_map[pnode_name] = str(node_index)
            x_data[node_index] = str(node_index) + ":" + "PDFF(" + pnode_name + ')'  # add the PDFF node
            node_index += 1
        elif "BUFF" in val:
            node_name = val.split("=")[0].strip()
            index_map[node_name] = str(node_index)
            x_data[node_index] = str(node_index) + ":" +val[:-1]
            node_index += 1
        elif "vdd" in val:
            node_name = val.split("=")[0]
            index_map[node_name] = str(node_index)
            x_data[node_index] = str(node_index) + ":" + val[:-1] #+ ";0"
            node_index += 1
    return x_data, node_index, index_map, output_index

def new_node(name2idx, x_data, node_name, gate_type):
    x_data.append([node_name, gate_type])
    name2idx[node_name] = len(name2idx)

def feature_generation(data, gate_to_index):
    '''
        A pre-processing function to handle with the modified `.bench` format files.
        Will generate the necessary attributes, adjacency matrix, edge connectivity matrix, etc.
            fixed bug: the key word of gates should be 'OR(' instead of 'OR',
            because variable name may be 'MEMORY' has 'OR'
        ...
        Parameters:
            data : list(str), the lines read out from a bench file (after modified by `add_node_index`)
            gate_to_index: dict(str:int), the mapping from the gate name to the gate index.
        Return:
            x_data: list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level.
            edge_index_data: list(list(int)), the connectivity matrix wiht shape of [num_edges, 2]
            level_list: logic level [max_level + 1, xx]
            fanin_list: the fanin node indexes for each node
            fanout_list: the fanout node indexes for each node
    '''
    name2idx = {}
    node_cnt = 0
    x_data = []
    edge_index_data = []

    for line in data:
        if 'INPUT(' in line:
            node_name = line.split("(")[-1].split(')')[0]
            new_node(name2idx, x_data, node_name, get_gate_type('INPUT', gate_to_index))
        elif 'AND(' in line or 'NAND(' in line or 'OR(' in line or 'NOR(' in line \
                or 'NOT(' in line or 'XOR(' in line:
            node_name = line.split(':')[-1].split('=')[0].replace(' ', '')
            gate_type = line.split('=')[-1].split('(')[0].replace(' ', '')
            new_node(name2idx, x_data, node_name, get_gate_type(gate_type, gate_to_index))

    for line_idx, line in enumerate(data):
        if 'AND(' in line or 'NAND(' in line or 'OR(' in line or 'NOR(' in line \
                or 'NOT(' in line or 'XOR(' in line:
            node_name = line.split(':')[-1].split('=')[0].replace(' ', '')
            gate_type = line.split('=')[-1].split('(')[0].replace(' ', '')
            src_list = line.split('(')[-1].split(')')[0].replace(' ', '').split(',')
            dst_idx = name2idx[node_name]
            for src_node in src_list:
                src_node_idx = name2idx[src_node]
                edge_index_data.append([src_node_idx, dst_idx])

    fanout_list = []
    fanin_list = []
    bfs_q = []
    x_data_level = [-1] * len(x_data)
    max_level = 0
    for idx, x_data_info in enumerate(x_data):
        fanout_list.append([])
        fanin_list.append([])
        if x_data_info[1] == 0:
            bfs_q.append(idx)
            x_data_level[idx] = 0
    for edge in edge_index_data:
        fanout_list[edge[0]].append(edge[1])
        fanin_list[edge[1]].append(edge[0])
    while len(bfs_q) > 0:
        idx = bfs_q[-1]
        bfs_q.pop()
        tmp_level = x_data_level[idx] + 1
        for next_node in fanout_list[idx]:
            if x_data_level[next_node] < tmp_level:
                x_data_level[next_node] = tmp_level
                bfs_q.insert(0, next_node)
                if x_data_level[next_node] > max_level:
                    max_level = x_data_level[next_node]
    level_list = []
    for level in range(max_level+1):
        level_list.append([])
    if -1 in x_data_level:
        print('Wrong')
        raise
    else:
        if max_level == 0:
            level_list = [[]]
        else:
            for idx in range(len(x_data)):
                x_data[idx].append(x_data_level[idx])
                level_list[x_data_level[idx]].append(idx)
    return x_data, edge_index_data, level_list, fanin_list, fanout_list

def rename_node(x_data):
    '''
    Convert the data[0] (node name : str) to the index (node index: int)
    ---
    Parameters:
        x_data: list(list(xx)), the node feature matrix
    Return:
        x_data: list(list(xx)), the node feature matrix
    '''
    idx2name = {}
    for idx, x_data_info in enumerate(x_data):
        idx2name[idx] = x_data_info[0]
        x_data_info[0] = idx
    return x_data, idx2name

def circuit_extraction(x_data, adj, circuit_depth, num_nodes, sub_circuit_size=25):
    '''
    Function to extract several subcircuits from the original circuit.
    ...
    Parameters:
        x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level.
        adj : list(list(int)), the adjacency matrix, adj[i][j] = {e(j, i) is in E} 
        circuit_depth : int, the logic depth of the circuit
        num_nodes : int, the total number of nodes in the circuit
        sub_circuit_size: int, the maximum size of the sub-circuits
    Return:
        sub_circuits_x_data : 
        sub_circuits_edges : 
        matrices : 
        
    '''
    adjs = []
    sub_circuits_x_data = []
    sub_circuits_edges = []
    sub_circuits_PIs = []
    sub_circuits_PIs = []

    iterations = 0
    # the current minmium level for the sub-circuit
    min_circuit_level = 0
    # the current maximum level for the sub-circuit
    max_circuit_level = sub_circuit_size

    # init level list
    level_lst = [[] for _ in range(circuit_depth)]

    # level_lis[i] contains the indices for nodes under this logic level
    for idx, node_data in enumerate(x_data):
        level_lst[node_data[2]].append(idx)

    # init predecessor list
    pre_lst = [[] for _ in range(num_nodes)]

    for col_idx, col in enumerate(adj):
        for row_idx, ele in enumerate(col):
            if ele == 1:
                pre_lst[col_idx].append(row_idx)

    while max_circuit_level <= circuit_depth:

        sub_x_data, sub_edges, sub_PIs = generate_sub_circuit(x_data, min_circuit_level, max_circuit_level - 1, level_lst, pre_lst)

        # adj_sub = [ [0] *  len(sub_x_data) ] * len(sub_x_data)
        adj_sub = [[0 for _ in range(len(sub_x_data))] for _ in range(len(sub_x_data))]
        for edge_data in sub_edges:
            adj_sub[edge_data[1]][edge_data[0]] = 1

        adjs.append(adj_sub)

        sub_circuits_x_data.append(sub_x_data)
        sub_circuits_edges.append(sub_edges)
        sub_circuits_PIs.append(sub_PIs)

        min_circuit_level = max_circuit_level
        max_circuit_level += sub_circuit_size

        if (max_circuit_level > circuit_depth > min_circuit_level) and (min_circuit_level != (circuit_depth - 1)):

            sub_x_data, sub_edges, sub_PIs = generate_sub_circuit(x_data, min_circuit_level, max_circuit_level - 1,
                                                                  level_lst, pre_lst)

            # adj_sub = [[0] * len(sub_x_data)] * len(sub_x_data)
            adj_sub = [[0 for x in range(sub_x_data)] for y in range(sub_x_data)]
            for edge_data in sub_edges:
                adj_sub[edge_data[1]][edge_data[0]] = 1

            adjs.append(adj_sub)

            sub_circuits_x_data.append(sub_x_data)
            sub_circuits_edges.append(sub_edges)
            sub_circuits_PIs.append(sub_PIs)
    return sub_circuits_x_data, sub_circuits_edges, adjs, sub_circuits_PIs


def generate_sub_circuit(x_data, min_circuit_level, max_circuit_level, level_lst, pre_lst):
    '''
    Function to extract a sub-circuit from the original circuit using the logic level information.
    ...
    Parameters:
        x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level.
        min_circuit_level : int, the current minmium level for the sub-circuit
        max_circuit_level: int, the maximum size of the sub-circuits
        level_lst : list(list(int)), level_lis[i] contains the indices for nodes under this logic level
        pre_lst : list(list(int)), pre_lst[i] contains the indices for predecessor nodes for the i-th node.
    Return:
        sub_x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level.
        sub_edge : list(list(int)), the connectivity matrix wiht shape of [num_edges, 2]
        sub_pi_indexes : list(int), the index for the primary inputs.
    '''
    sub_x_data = []
    sub_pi_indexes = []
    # the list that contains node indices for the extracted logic range.
    sub_node = []
    sub_edge = []
    x_data_tmp = copy.deepcopy(x_data)

    # Picking all nodes in desired depth
    for level in range(min_circuit_level, max_circuit_level + 1):
        if level < len(level_lst):
            for node in level_lst[level]:
                sub_node.append(node)

    # Update logic level
    for idx in sub_node:
        x_data_tmp[idx][2] = x_data_tmp[idx][2] - (min_circuit_level)

    # Separate PI and Gate
    PIs = []
    Gates = []
    for idx in sub_node:
        if x_data_tmp[idx][2] == 0:
            x_data_tmp[idx][1] = 0
            PIs.append(idx)
        else:
            Gates.append(idx)

    # Search subcircuit edge
    for idx in Gates:
        for pre_idx in pre_lst[idx]:
            sub_edge.append([pre_idx, idx])
            # Insert new PI. mli: consider the corner cases that there are some internal nodes connected to the predecessors that are located in the level less than min_circuit_level
            if x_data[pre_idx][2] < min_circuit_level:
                x_data_tmp[pre_idx][1] = 0
                x_data_tmp[pre_idx][2] = 0
                PIs.append(pre_idx)
                sub_node.append(pre_idx)

    # Ignore the no edge node
    node_mask = [0] * len(x_data)
    for edge in sub_edge:
        node_mask[edge[0]] = 1
        node_mask[edge[1]] = 1

    # Map to subcircuit index
    sub_node = list(set(sub_node))
    sub_node = sorted(sub_node, key=lambda x: x_data[x][2])
    sub_cnt = 0
    ori2sub_map = {}  # Original index map to subcircuit
    for node_idx in sub_node:
        if node_mask[node_idx] == 1:
            sub_x_data.append(x_data_tmp[node_idx].copy())
            ori2sub_map[node_idx] = sub_cnt
            sub_cnt += 1
    for edge_idx, edge in enumerate(sub_edge):
        sub_edge[edge_idx] = [ori2sub_map[edge[0]], ori2sub_map[edge[1]]]
    for pi_idx in PIs:
        if node_mask[pi_idx] == 1:
            sub_pi_indexes.append(ori2sub_map[pi_idx])

    return sub_x_data, sub_edge, sub_pi_indexes


def generate_prob_cont(x_data, PI_indexes, level_list, fanin_list):
    '''
    Function to calculate Controlability values, i.e. C1 and C0 for the nodes.
    ...
    Parameters:
        x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level.
        PI_indexes : list(int), the indices for the primary inputs
        level_list: logic levels
        fanin_list: the fanin node indexes for each node
    Return:
        x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level; 3rd - C1; 4th - C0.
    '''
    y = [0] * len(x_data)

    for i in PI_indexes:
        y[i] = [0.5, 0.5]

    for level in range(1, len(level_list), 1):
        for idx in level_list[level]:
            source_node = fanin_list[idx]
            source_signals = []
            for node in source_node:
                source_signals.append(y[node])
            if len(source_signals) > 0:
                zero, one = prob_logic(x_data[idx][1], source_signals)
                y[idx] = [zero, one]

    for i, prob in enumerate(y):
        x_data[i].append(prob[1])
        x_data[i].append(prob[0])

    return x_data


def generate_prob_obs(x_data, level_list, fanin_list, fanout_list):
    '''
        Function to calculate Observability values, i.e. CO.
        ...
        Parameters:
            x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level.
            level_list: logic levels
            fanin_list: the fanin node indexes for each node
            fanout_list: the fanout node indexes for each node
        Return:
            x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level; 3rd - C1; 4th - C0; 5th - CO.
        '''
    # Array node into level_list

    y = [-1] * len(x_data)

    POs_indexes = []
    for idx, nxt in enumerate(fanout_list):
        if len(nxt) == 0:
            POs_indexes.append(idx)
            y[idx] = 1

    for level in range(len(level_list) - 1, -1, -1):
        for idx in level_list[level]:
            source_signals = fanin_list[idx]
            if len(source_signals) > 0:
                y = obs_prob(x_data, idx, y, source_signals)

    for i, val in enumerate(y):
        x_data[i].append(val)

    return x_data


def dfs_reconvergent_circuit(node_idx, vis, dst_idx, fanout_list, result, x_data):
    if node_idx == dst_idx:
        result += vis
        return
    for nxt_idx in fanout_list[node_idx]:
        if x_data[nxt_idx][2] <= x_data[dst_idx][2]:
            vis.append(nxt_idx)
            dfs_reconvergent_circuit(nxt_idx, vis, dst_idx, fanout_list, result, x_data)
            vis = vis[:-1]
    return result


def identify_reconvergence(x_data, level_list, fanin_list, fanout_list):
    '''
    Function to identify the reconvergence nodes in the given circuit.
    The algorithm is done under the principle that we only consider the minimum reconvergence structure.
    ...
    Parameters:
        x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level; 3rd - C1, 4th - C0, 5th - Obs.
        level_list: logic levels
        fanin_list: the fanin node indexes for each node
        fanout_list: the fanout node indexes for each node
    Return:
        x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level; 3rd - C1; 4th - C0; 5th - Obs; 6th - fan-out, 7th - boolean recovengence, 8th - index of the source node (-1 for non recovengence).
        rc_lst: list(int), the index for the reconvergence nodes
    '''
    for idx, node in enumerate(x_data):
        if len(fanout_list[idx]) > 1:
            x_data[idx].append(1)
        else:
            x_data[idx].append(0)

        # fanout list (FOL)
    FOL = []
    fanout_num = []
    is_del = []
    # RC (same as reconvergence_nodes)
    rc_lst = []
    max_level = 0
    for x_data_info in x_data:
        if x_data_info[2] > max_level:
            max_level = x_data_info[2]
        FOL.append([])
    for idx, x_data_info in enumerate(x_data):
        fanout_num.append(len(fanout_list[idx]))
        is_del.append(False)

    for level in range(max_level + 1):
        if level == 0:
            for idx in level_list[0]:
                x_data[idx].append(0)
                x_data[idx].append(-1)
                if x_data[idx][6]:
                    FOL[idx].append(idx)
        else:
            for idx in level_list[level]:
                FOL_tmp = []
                FOL_del_dup = []
                save_mem_list = []
                for pre_idx in fanin_list[idx]:
                    if is_del[pre_idx]:
                        print('[ERROR] This node FOL has been deleted to save memory')
                        raise
                    FOL_tmp += FOL[pre_idx]
                    fanout_num[pre_idx] -= 1
                    if fanout_num[pre_idx] == 0:
                        save_mem_list.append(pre_idx)
                for save_mem_idx in save_mem_list:
                    FOL[save_mem_idx].clear()
                    is_del[save_mem_idx] = True
                FOL_cnt_dist = Counter(FOL_tmp)
                source_node_idx = 0
                source_node_level = -1
                is_rc = False
                for dist_idx in FOL_cnt_dist:
                    FOL_del_dup.append(dist_idx)
                    if FOL_cnt_dist[dist_idx] > 1:
                        is_rc = True
                        if x_data[dist_idx][2] > source_node_level:
                            source_node_level = x_data[dist_idx][2]
                            source_node_idx = dist_idx
                if is_rc:
                    x_data[idx].append(1)
                    x_data[idx].append(source_node_idx)
                    rc_lst.append(idx)
                else:
                    x_data[idx].append(0)
                    x_data[idx].append(-1)

                FOL[idx] = FOL_del_dup
                if x_data[idx][6]:
                    FOL[idx].append(idx)
    del (FOL)

    # for node in range(len(x_data)):
    #     x_data[node].append(0)
    # for rc_idx in rc_lst:
    #     x_data[rc_idx][-1] = 1

    return x_data, rc_lst


def backward_search(node_idx, fanin_list, x_data, min_level):
    if x_data[node_idx][2] <= min_level:
        return []
    result = []
    for pre_node in fanin_list[node_idx]:
        if pre_node not in result:
            l = [pre_node]
            res = backward_search(pre_node, fanin_list, x_data, min_level)
            result = result + l + list(set(res))
        else:
            l = [pre_node]
            result = result + l
    return result


def check_reconvergence(x_data, edge_list):
    pre_lst = []
    for node in x_data:
        pre_lst.append([])
    for edge in edge_list:
        pre_lst[edge[1]].append(edge[0])

    for idx, node in enumerate(x_data):
        if node[-2] == 1:
            source_level = x_data[node[-1]][2]
            vis_list = backward_search(idx, pre_lst, x_data, source_level)
            vis_cnt_dist = Counter(vis_list)
            find_source = False
            for dist_idx in vis_cnt_dist:
                if vis_cnt_dist[dist_idx] > 1:
                    find_source = True
                    if x_data[dist_idx][2] > source_level:
                        print("[ERROR] Not the nearest source")
                        raise
            if not find_source:
                print("[ERROR] No source node find")
                raise




def circuit_statistics(circuit_name, x_data, edge_list):
    print('================== Statistics INFO ==================')
    print('Circuit Name: {}'.format(circuit_name))
    print('Number of Nodes: {}'.format(len(x_data)))
    gate_type_cnt = [0] * 10
    gate_type = []
    for x_data_info in x_data:
        gate_type_cnt[x_data_info[1]] += 1
    for k in range(10):
        if gate_type_cnt[k] > 0:
            gate_type.append(k)
    print('Number of Gate Types: {}'.format(len(gate_type)))
    print('Gate: ', gate_type)

    # gate level difference
    level_diff = []
    for node_idx, node_info in enumerate(x_data):
        if node_info[-2] == 1:
            level_diff.append([node_idx, node_info[-1], x_data[node_idx][2] - x_data[node_info[-1]][2]])
    level_diff = sorted(level_diff, key=lambda x: x[-1])
    if level_diff == []:
        print('No reconvergent node')
    else:
        print('Max level = {:}, from {} to {}'.format(level_diff[-1][2],
                                                      x_data[level_diff[-1][0]][0], x_data[level_diff[-1][1]][0]))
        print('Min level = {:}, from {} to {}'.format(level_diff[0][2],
                                                      x_data[level_diff[0][0]][0], x_data[level_diff[0][1]][0]))

    # reconvergent area
    fanout_list = []
    rc_cnt = 0
    for idx, node_info in enumerate(x_data):
        fanout_list.append([])
        if node_info[-2] == 1:
            rc_cnt += 1
    for edge in edge_list:
        fanout_list[edge[0]].append(edge[1])
    rc_gates = []
    for node_idx, node_info in enumerate(x_data):
        if node_info[-2] == 1:
            src_idx = node_info[-1]
            dst_idx = node_idx
            rc_gates += dfs_reconvergent_circuit(src_idx, [src_idx], dst_idx, fanout_list, [], x_data)
    rc_gates_merged = list(set(rc_gates))
    print('Reconvergent nodes: {:}/{:} = {:}'.format(rc_cnt, len(x_data),
                                                     rc_cnt / len(x_data)))
    print('Reconvergent area: {:}/{:} = {:}'.format(len(rc_gates_merged), len(x_data),
                                                    len(rc_gates_merged) / len(x_data)))


def check_difference(dataset):
    diff = 0
    tot = 0
    for g in dataset:
        diff += torch.sum(torch.abs((g.c1 - g.gt)))
        tot += g.c1.size(0)
    print('Average difference between C1 and GT is: ', (diff/tot).item())
    diff = 0
    tot = 0
    for g in dataset:
        diff += torch.sum(torch.abs((g.c1 - g.gt)) * g.rec)
        tot += torch.sum(g.rec)
    print('Average difference between C1 and GT (reconvergent nodes) is: ', (diff/tot).item())
    diff = 0
    tot = 0
    for g in dataset:
        diff += torch.sum(torch.abs((g.c1 - g.gt)) * (1- g.rec))
        tot += torch.sum(1 - g.rec)
    print('Average difference between C1 and GT (non-reconvergent nodes) is: ', (diff/tot).item())


def aig_simulation(x_data, edge_index_data, num_patterns=15000):
    fanout_list = []
    fanin_list = []
    bfs_q = []
    x_data_level = [-1] * len(x_data)
    max_level = 0
    for idx, x_data_info in enumerate(x_data):
        fanout_list.append([])
        fanin_list.append([])
    for edge in edge_index_data:
        fanout_list[edge[0]].append(edge[1])
        fanin_list[edge[1]].append(edge[0])

    PI_indexes = []
    for idx, ele in enumerate(fanin_list):
        if len(ele) == 0:
            PI_indexes.append(idx)
            x_data_level[idx] = 0
            bfs_q.append(idx)

    while len(bfs_q) > 0:
        idx = bfs_q[-1]
        bfs_q.pop()
        tmp_level = x_data_level[idx] + 1
        for next_node in fanout_list[idx]:
            if x_data_level[next_node] < tmp_level:
                x_data_level[next_node] = tmp_level
                bfs_q.insert(0, next_node)
                if x_data_level[next_node] > max_level:
                    max_level = x_data_level[next_node]
    level_list = []
    for level in range(max_level+1):
        level_list.append([])
    for idx, ele in enumerate(x_data_level):
        level_list[ele].append(idx)

    ######################
    # Simulation
    ######################
    y = [0] * len(x_data)
    y1 = [0] * len(x_data)
    pattern_count = 0
    no_of_patterns = min(num_patterns, 10 * pow(2, len(PI_indexes)))
    print('No of Patterns: {:}'.format(no_of_patterns))
    print('[INFO] Begin simulation')
    while pattern_count < no_of_patterns:
        input_vector = random_pattern_generator(len(PI_indexes))
        j = 0
        for i in PI_indexes:
            y[i] = input_vector[j]
            j = j + 1
        for level in range(1, len(level_list), 1):
            for node_idx in level_list[level]:
                source_signals = []
                for pre_idx in fanin_list[node_idx]:
                    source_signals.append(y[pre_idx])
                if len(source_signals) > 0:
                    if int(x_data[node_idx][0][1]) == 1:
                        gate_type = 1
                    elif int(x_data[node_idx][0][2]) == 1:
                        gate_type = 5
                    else:
                        raise("This is PI")
                    y[node_idx] = logic(gate_type, source_signals)
                    if y[node_idx] == 1:
                        y1[node_idx] = y1[node_idx] + 1

        pattern_count = pattern_count + 1
        if pattern_count % 10000 == 0:
            print("pattern count = {:}k".format(int(pattern_count / 1000)))

    for i, _ in enumerate(y1):
        y1[i] = [y1[i] / pattern_count]

    for i in PI_indexes:
        y1[i] = [0.5]

    return y1

def parse_xdata(x_data, edge_data):
    fanout_list = []
    fanin_list = []
    level_list = []
    max_level = 0
    for x_data_info in x_data:
        fanout_list.append([])
        fanin_list.append([])
        if int(x_data_info[2]) > max_level:
            max_level = int(x_data_info[2])
    for level in range(max_level + 1):
        level_list.append([])
        
    # Fanin, Fanout
    for edge in edge_data:
        fanout_list[edge[0]].append(edge[1])
        fanin_list[edge[1]].append(edge[0])
    
    # Level
    for idx, x_data_info in enumerate(x_data):
        level_list[int(x_data_info[2])].append(idx)
        
    # x_data
    new_x_data = []
    new_edge_data = []
    for x_data_info in x_data:
        new_x_data.append(x_data_info.tolist())
    for edge in edge_data:
        #new_edge_data.append(edge.tolist())
        new_edge_data.append(edge)
    
    return new_x_data, new_edge_data, fanin_list, fanout_list, level_list

def get_ff_connection(x_data, fanin_list, fanout_list, level_list): 
    fpi_list = []
    ff_fanin_list = []
    ff_fanout_list = []
    
    for idx, x_data_info in enumerate(x_data):
        fpi_list.append([])
        ff_fanin_list.append([])
        ff_fanout_list.append([])
        if x_data_info[1] == 0 or x_data_info[1] == 3:
            fpi_list[idx].append(idx)
    
    for level in range(len(level_list)):
        for idx in level_list[level]:
            # Update fanout_idx
            for fanin_idx in fanin_list[idx]:
                fpi_list[idx] += fpi_list[fanin_idx]
            fpi_list[idx] = list(set(fpi_list[idx]))
                
    for idx, x_data_info in enumerate(x_data):
        if x_data[idx][1] == 3:
            comb_idx = fanin_list[idx][0]
            for ff_idx in fpi_list[comb_idx]:
                ff_fanin_list[idx].append(ff_idx)
                ff_fanout_list[ff_idx].append(idx)
                
    # Detect PO
    po_ff_list = []
    for idx, x_data_info in enumerate(x_data):
        if len(fanin_list[idx]) > 0 and len(fanout_list[idx]) == 0:
            po_ff_list += fpi_list[idx]
    po_ff_list = list(set(po_ff_list))        
    
    return ff_fanin_list, ff_fanout_list


def get_level(x_data, fanin_list, fanout_list):
    bfs_q = []
    x_level = [-1] * len(x_data)
    max_level = 0
    for idx, x_data_info in enumerate(x_data):
        if len(fanout_list[idx]) == 0 and len(fanin_list[idx]) != 0:
            bfs_q.append(idx)
            x_level[idx] = 0
    while len(bfs_q) > 0:
        idx = bfs_q[-1]
        bfs_q.pop()
        tmp_level = x_level[idx] + 1
        for next_node in fanin_list[idx]:
            if x_level[next_node] < tmp_level:
                x_level[next_node] = tmp_level
                bfs_q.insert(0, next_node)
                if x_level[next_node] > max_level:
                    max_level = x_level[next_node]
    level_list = []
    for level in range(max_level+1):
        level_list.append([])
        
    if max_level == 0:
        level_list = [[]]
    else:
        for idx in range(len(x_data)):
            level_list[x_level[idx]].append(idx)
    return level_list

def get_fanin_fanout(x_data, edge_index):
    fanout_list = []
    fanin_list = []
    for idx, x_data_info in enumerate(x_data):
        fanout_list.append([])
        fanin_list.append([])
    for edge in edge_index:
        fanout_list[edge[0]].append(edge[1])
        fanin_list[edge[1]].append(edge[0])
    return fanin_list, fanout_list


def feature_gen_connect(data, gate_to_index, output_index):
    '''
        A pre-processing function to handle with the modified `.bench` format files.
        Will generate the necessary attributes, adjacency matrix, edge connectivity matrix, etc.
            fixed bug: the key word of gates should be 'OR(' instead of 'OR',
            because variable name may be 'MEMORY' has 'OR'
        ...
        Parameters:
            data : list(str), the lines read out from a bench file (after modified by `add_node_index`)
            gate_to_index: dict(str:int), the mapping from the gate name to the gate index.
        Return:
            x_data: list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level.
            edge_index_data: list(list(int)), the connectivity matrix wiht shape of [num_edges, 2]
            level_list: logic level [max_level + 1, xx]
            fanin_list: the fanin node indexes for each node
            fanout_list: the fanout node indexes for each node
    '''
    name2idx = {}
    node_cnt = 0
    x_data = []
    edge_index_data = []
    for line in data.values():
        if 'INPUT(' in line:
            node_name = line.split("(")[-1].split(')')[0].strip()
            new_node(name2idx, x_data, node_name, get_gate_type('INPUT', gate_to_index))
        elif 'OUTPUT' in line:
            node_name = line.split("(")[-1].split(')')[0].strip()
            new_node(name2idx, x_data, node_name, get_gate_type('OUTPUT', gate_to_index))
        elif 'PDFF(' in line:
            node_name = line.split("(")[-1].split(')')[0].strip()
            new_node(name2idx, x_data, node_name, get_gate_type('PDFF', gate_to_index))
        elif 'DFF(' in line:
            node_name = line.split(':')[-1].split('=')[0].replace(' ', '').strip()
            new_node(name2idx, x_data, node_name, get_gate_type('DFF', gate_to_index))
        elif 'AND(' in line or 'NAND(' in line or 'OR(' in line or 'NOR(' in line \
                or 'NOT(' in line or 'XOR(' in line or 'BUFF(' in line:
            node_name = line.split(':')[-1].split('=')[0].replace(' ', '').strip()
            gate_type = line.split('=')[-1].split('(')[0].replace(' ', '').strip()
            new_node(name2idx, x_data, node_name, get_gate_type(gate_type, gate_to_index))
        elif 'vdd' in line:
            node_name = line.split(':')[-1].split('=')[0].strip()
            new_node(name2idx, x_data, node_name, get_gate_type('const',gate_to_index))
    for line_idx, line in enumerate(data.values()):
        if 'AND(' in line or 'NAND(' in line or 'OR(' in line or 'NOR(' in line \
                or 'NOT(' in line or 'XOR(' in line or 'BUFF(' in line or ' DFF(' in line :
            node_name = line.split(':')[-1].split('=')[0].replace(' ', '').strip()
            gate_type = line.split('=')[-1].split('(')[0].replace(' ', '').strip()
            dst_idx = name2idx[node_name]
            
            src_list = line.split('(')[-1].split(')')[0].replace(' ', '').split(',')
            
            for src_node in src_list:
                if x_data[name2idx[src_node]][1] == gate_to_index['DFF']:
                    src_node = 'P' + src_node
                src_node_idx = name2idx[src_node]
                edge_index_data.append([src_node_idx, dst_idx])
        elif 'OUTPUT' in line:
            node_name = line.split('(')[-1].split(')')[0]
            dst_idx = name2idx[node_name]
            src_list = node_name[1:]
            src_node_idx = name2idx[src_list]
            edge_index_data.append([src_node_idx, dst_idx])
    return x_data, edge_index_data

def feature_gen_level(x_data, fanout_list, gate_to_index={'GND': 999, 'VDD': 999}):
    bfs_q = []
    x_data_level = [-1] * len(x_data)
    x_data_level_raw = [-1] * len(x_data)
    max_level = 0
    for idx, x_data_info in enumerate(x_data):
        if x_data_info[1] == 0 or x_data_info[1] == 4 or x_data_info[1]==8:
            bfs_q.append(idx)
            x_data_level[idx] = 0
    while len(bfs_q) > 0:
        idx = bfs_q[-1]
        bfs_q.pop()
        tmp_level = x_data_level[idx] + 1
        for next_node in fanout_list[idx]:
            if x_data_level[next_node] < tmp_level:
                x_data_level[next_node] = tmp_level
                bfs_q.insert(0, next_node)
                if x_data_level[next_node] > max_level:
                    max_level = x_data_level[next_node]
    level_list = []
    for level in range(max_level+1):
        level_list.append([])
    for idx, x_data_info in enumerate(x_data):
        if x_data_info[1] == 8:
            x_data_level[idx] = 0
        else:
            if x_data_level[idx] == -1:
                print(x_data_info)
                print('[ERROR] Find unconnected node')
                raise
    if max_level == 0:
        level_list = [[]]
    else:
        for idx in range(len(x_data)):
            level_list[x_data_level[idx]].append(idx)
            x_data[idx].append(x_data_level[idx])


    return x_data, level_list

def add_dff(data, edge_data, fanin_list, fanout_list, num_nodes):
    
    new_edge_data = []
    for edge in edge_data:
        new_edge_data.append(edge)
    
    for idx, x_data_info in enumerate(data):
        if fanout_list[idx] == [] and fanin_list[idx] != [] and x_data_info[1] != 3 and x_data_info[1] != 5:
            fanout_list[idx].append(num_nodes)
            new_edge_data.append([idx, num_nodes])
            fanin_list.append([])
            fanout_list.append([])
            fanin_list[num_nodes].append(idx)
            data.append([f'DN{idx}', 3])
            num_nodes += 1
    return data, new_edge_data, fanin_list, fanout_list, num_nodes

def parse_bench(file, gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2, 'DFF': 3, 'PDFF': 4}, MAX_LENGTH=-1):
    data = read_file(file)
    data, num_nodes, _, output_index = add_node_index(data)  #data[i] = node_index:cell_type
    if MAX_LENGTH > 0 and num_nodes > MAX_LENGTH:
        return [], [], [], [], []
    data, edge_data = feature_gen_connect(data, gate_to_index, output_index) 

    fanin_list, fanout_list = get_fanin_fanout(data, edge_data)# data[i] = [node_name, cell_index, logic_level]
    data, edge_data, fanin_list, fanout_list, num_nodes = add_dff(data, edge_data, fanin_list, fanout_list, num_nodes)
    data, level_list = feature_gen_level(data, fanout_list, gate_to_index) 
    data, idx2name = rename_node(data) # data[i] = [node_idx, cell_index, logic_level]
    return data, edge_data, fanin_list, fanout_list, level_list, idx2name


def simulator_truth_table(x_data, PI_indexes, level_list, fanin_list, gate_to_index):
    no_of_patterns = int(pow(2, len(PI_indexes)))
    truth_table = []
    for idx in range(len(x_data)):
        truth_table.append([])

    for pattern_idx in range(no_of_patterns):
        input_vector = dec2list(pattern_idx, len(PI_indexes))
        state = [-1] * len(x_data)

        for k, pi_idx in enumerate(PI_indexes):
            state[pi_idx] = input_vector[k]

        for level in range(1, len(level_list), 1):
            for node_idx in level_list[level]:
                source_signals = []
                for pre_idx in fanin_list[node_idx]:
                    source_signals.append(state[pre_idx])
                if len(source_signals) > 0:
                    gate_type = x_data[node_idx][1]
                    res = logic(gate_type, source_signals, gate_to_index)
                    state[node_idx] = res

        for idx in range(len(x_data)):
            truth_table[idx].append(state[idx])
    
    return truth_table

def simulator_truth_table_random(x_data, PI_indexes, level_list, fanin_list, gate_to_index, num_patterns=15000):
    truth_table = []
    for idx in range(len(x_data)):
        truth_table.append([])

    for pattern_idx in range(num_patterns):
        input_vector = random_pattern_generator(len(PI_indexes))
        state = [-1] * len(x_data)
        for k, pi_idx in enumerate(PI_indexes):
            state[pi_idx] = input_vector[k]

        for level in range(1, len(level_list), 1):
            for node_idx in level_list[level]:
                source_signals = []
                for pre_idx in fanin_list[node_idx]:
                    source_signals.append(state[pre_idx])
                if len(source_signals) > 0:
                    gate_type = x_data[node_idx][1]
                    res = logic(gate_type, source_signals, gate_to_index)
                    state[node_idx] = res
        for idx in range(len(x_data)):
            truth_table[idx].append(state[idx])
    
    return truth_table
def bin_array_to_dec(bin_list):
    """
    把二进制数组转成十进制数
    :param bin_list: [0,1,...] 的数组
    :return: 十进制整数
    """
    return int("".join(str(x) for x in bin_list), 2)
def dec_to_bin_array(num, N):
    """
    把十进制数转成长度为 N 的二进制数组
    :param num: 十进制数
    :param N: 数组长度 (寄存器个数)
    :return: [0,1,...] 的数组
    """
    return [int(x) for x in format(num, f'0{N}b')]
def simulator_finite_state_machine(x_data, PI_indexes, level_list, fanin_list, gate_to_index, idx2name, number, num_patterns=300, trace_length=10):
    truth_table = torch.zeros(number, number)
    for s in range(number):
        temp = dec_to_bin_array(s, number)
        for pattern_idx in range(num_patterns):
            state = [-1] * len(x_data)
            k = 0
            for idx, x_data_info in enumerate(x_data):
                    if x_data_info[1] == gate_to_index['DFF']:
                        state[idx] = temp[k]
                        k += 1
            for index in range(trace_length):
                input_vector = random_pattern_generator(len(PI_indexes))
                for k, pi_idx in enumerate(PI_indexes):
                    state[pi_idx] = input_vector[k]
                
                tmp_level = 1
                visited = PI_indexes.copy()
                for x in x_data:
                    if x[1] == gate_to_index['DFF']:
                        visited.append(x[0])
                num_visited = len(visited)
                #print(level_list)
                while num_visited < len(x_data):
                    count = 0
                    for level in range(tmp_level, len(level_list)):
                        for node_idx in level_list[level]:
                            if node_idx in visited and x_data[node_idx][1] != gate_to_index['DFF']:
                                #print(node_idx,count,level,num_visited,len(x_data))
                                continue
                            source_signals = []
                            flag = False
                            fanin = fanin_list[node_idx]
                            for pre_idx in fanin:
                                if state[pre_idx] == -1:                      
                                    flag = True
                                    break
                                source_signals.append(state[pre_idx]) 
                            if flag:
                                continue
                            count += 1
                            num_visited += 1
                            if len(source_signals) > 0:
                                gate_type = x_data[node_idx][1]
                                res = logic(gate_type, source_signals, gate_to_index)
                                state[node_idx] = res
                                if x_data[node_idx][1] != gate_to_index['DFF']:
                                    #print(state[node_idx])
                                    visited.append(node_idx)
                                
                    if count == 0:
                        tmp_level = 0
                        for node_idx in level_list[tmp_level]:
                            if x_data[node_idx][1] == 4:
                                node_name = idx2name[x_data[node_idx][0]].strip('P')
                                for idx, raw in enumerate(x_data):
                                    if node_name == idx2name[raw[0]] and idx in visited:
                                        state[node_idx] = state[idx]
                                        visited.append(node_idx)
                        num_visited = len(visited)
                        
                    tmp_level += 1  
                result = []
                for idx in range(len(x_data)):
                    if x_data[idx][1] == gate_to_index['DFF']:
                        result.append(state[idx])
                current_state = bin_array_to_dec(result)
                truth_table[s, current_state] += 1
            #print(truth_table[pattern_idx])
    truth_table = truth_table / (num_patterns * trace_length)

    return truth_table
def simulator_sequence_state(x_data, PI_indexes, level_list, fanin_list, gate_to_index, idx2name, num_patterns=10):
    truth_table = []
    for idx in range(num_patterns):
        truth_table.append([[] for _ in range(2)])

    for pattern_idx in range(num_patterns):
        state = [-1] * len(x_data)
        
        for idx, x_data_info in enumerate(x_data):
            if x_data_info[1] == gate_to_index['DFF']:
                state[idx] = 0
        for rounds in range(2):
            input_vector = random_pattern_generator(len(PI_indexes))
            for k, pi_idx in enumerate(PI_indexes):
                state[pi_idx] = input_vector[k]
            tmp_level = 1
            visited = PI_indexes.copy()
            for x in x_data:
                if x[1] == gate_to_index['DFF']:
                    visited.append(x[0])
            num_visited = len(visited)
            #print(level_list)
            while num_visited < len(x_data):
                count = 0
                for level in range(tmp_level, len(level_list)):
                    for node_idx in level_list[level]:
                        if node_idx in visited and x_data[node_idx][1] != gate_to_index['DFF']:
                            #print(node_idx,count,level,num_visited,len(x_data))
                            continue
                        source_signals = []
                        flag = False
                        fanin = fanin_list[node_idx]
                        for pre_idx in fanin:
                            if state[pre_idx] == -1:                      
                                flag = True
                                break
                            source_signals.append(state[pre_idx]) 
                        if flag:
                            continue
                        count += 1
                        num_visited += 1
                        if len(source_signals) > 0:
                            gate_type = x_data[node_idx][1]
                            res = logic(gate_type, source_signals, gate_to_index)
                            state[node_idx] = res
                            if x_data[node_idx][1] != gate_to_index['DFF']:
                                #print(state[node_idx])
                                visited.append(node_idx)
                            
                if count == 0:
                    tmp_level = 0
                    for node_idx in level_list[tmp_level]:
                        if x_data[node_idx][1] == 4:
                            node_name = idx2name[x_data[node_idx][0]].strip('P')
                            for idx, raw in enumerate(x_data):
                                if node_name == idx2name[raw[0]] and idx in visited:
                                    state[node_idx] = state[idx]
                                    visited.append(node_idx)
                    num_visited = len(visited)
                    
                tmp_level += 1  

            for idx in range(len(x_data)):
                if x_data[idx][1] == gate_to_index['INPUT']:
                    truth_table[pattern_idx][rounds].append(state[idx])
            for idx in range(len(x_data)):
                if x_data[idx][1] == gate_to_index['DFF']:
                    truth_table[pattern_idx][rounds].append(state[idx])
         
        
        #print(truth_table[pattern_idx])
    
    return truth_table

def separate_ff_fanin_fanout(x_data, edge_index, gate_to_index):
  
    no_fanin_edge_data = []
    no_fanout_edge_data = []
    ff_fanin_edge_data = []
    ff_fanout_edge_data = []

    for edge in edge_index:
        if x_data[edge[1]][1] == gate_to_index['DFF']:
            ff_fanin_edge_data.append([edge[0], edge[1]])
        else:
            no_fanin_edge_data.append([edge[0], edge[1]])
        
        if x_data[edge[0]][1] == gate_to_index['DFF']:
            ff_fanout_edge_data.append([edge[0], edge[1]])
        else:
            no_fanout_edge_data.append([edge[0], edge[1]])

    return no_fanin_edge_data, ff_fanin_edge_data, no_fanout_edge_data, ff_fanout_edge_data


def get_cyclic_ff(x_data, edge_index, gate_to_index):

   
        # Parse circuit 
        x_data, edge_index, fanin_list, fanout_list, level_list = parse_xdata(x_data, edge_index)
        max_level = len(level_list) + 1
        PI_indexes = []
        for idx, x_data_info in enumerate(x_data):
            if int(x_data_info[1]) == gate_to_index['PI']:
                PI_indexes.append(idx)
                
        # Get RTL Graph
        ff_fanin_list, ff_fanout_list =get_ff_connection(x_data, fanin_list, fanout_list, level_list)
                
    
        # Detect Cycle
        no_cycle, cyclic_FFs_nodes = detect_cycles(x_data, ff_fanout_list, edge_index)

       # print('Max level: {:}, No cycle: {:}'.format(max_level, no_cycle))
    
        return no_cycle, cyclic_FFs_nodes
