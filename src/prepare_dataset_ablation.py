'''
Parse the AIG (in bench format) and truth table for each nodes
16-11-2022
Note: 
    gate_to_index = {'PI': 0, 'AND': 1, 'NOT': 2}
    x_data: 0 - Name, 1 - gate type, 2 - level, 3 - is RC, 4 - RC source node 
'''
from itertools import combinations
import argparse
import glob
import os
import sys
import platform
import time
import numpy as np
from collections import Counter
import torch
import copy
import json
import utils.circuit_utils as circuit_utils
import utils.utils as utils
import fsm_simulator
# aig_folder = './rawaig/'
NO_PATTERNS = 15000

gate_to_index = {'INPUT': 0, 'AND': 1, 'NOT': 2, 'DFF': 3, 'PDFF': 4, 'NAND': 5, 'NOR': 6, 'OR': 7, 'XOR': 8, 'BUF': 9}
MIN_LEVEL = 3
MIN_PI_SIZE = 4
MAX_INCLUDE = 1.5
MAX_PROB_GAP = 0.05
MAX_LEVEL_GAP = 5

MIDDLE_DIST_IGNORE = [0.2, 0.8]

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default='ablation')
    parser.add_argument('--start_idx', default=0, type=int)
    parser.add_argument('--end_idx', default=20000, type=int)
    parser.add_argument('--aig_folder', default='./raw_data/experimentB')

    args = parser.parse_args()
    return args

def estimate_transition_prob_from_tt(tt):
    """
    从真值表估算每个节点的翻转概率（0→1 和 1→0）

    参数：
        tt (List[List[int]]): 每个节点在所有输入组合下的输出值（真值表）

    返回：
        t_01 (List[float]): 每个节点从0翻转为1的概率
        t_10 (List[float]): 每个节点从1翻转为0的概率
    """
    t_01 = []
    t_10 = []

    for node_tt in tt:
        num_01 = 0
        num_10 = 0
        total_flips = 0

        for i in range(len(node_tt) - 1):
            a = node_tt[i]
            b = node_tt[i + 1]
            if a != b:
                total_flips += 1
                if a == 0 and b == 1:
                    num_01 += 1
                elif a == 1 and b == 0:
                    num_10 += 1

        if total_flips == 0:
            t_01.append(0.0)
            t_10.append(0.0)
        else:
            t_01.append(num_01 / total_flips)
            t_10.append(num_10 / total_flips)

    return t_01, t_10

def gen_tt_pair(x_data, fanin_list, fanout_list, level_list, tt_prob):
    tt_len = len(tt[0])
    pi_cone_list = []
    for idx in range(len(x_data)):
        pi_cone_list.append([])

    # Get pre fanout
    for level in range(len(level_list)):
        if level == 0:
            for idx in level_list[level]:
                pi_cone_list[idx].append(idx)
        else:
            for idx in level_list[level]:
                for fanin_idx in fanin_list[idx]:
                    pi_cone_list[idx] += pi_cone_list[fanin_idx]
                pre_dist = Counter(pi_cone_list[idx])
                pi_cone_list[idx] = list(pre_dist.keys())

    # Pair
    tt_pair_index = []
    tt_dis = []
    min_tt_dis = []
    for i in range(len(x_data)):
        if x_data[i][2] < MIN_LEVEL or len(pi_cone_list[i]) < MIN_PI_SIZE:
            continue
        for j in range(i+1, len(x_data), 1):
            if x_data[j][2] < MIN_LEVEL or len(pi_cone_list[j]) < MIN_PI_SIZE:
                continue
            # Cond. 2: probability
            if abs(tt_prob[i] - tt_prob[j]) > MAX_PROB_GAP:
                continue
            # Cond. 1: Level
            if abs(x_data[i][2] - x_data[j][2]) > MAX_LEVEL_GAP:
                continue

            # Cond. 5: Include
            if pi_cone_list[i] != pi_cone_list[j]:
                continue

            distance = np.array(tt[i]) - np.array(tt[j])
            distance_value = np.linalg.norm(distance, ord=1) / tt_len

            # Cond. 4: Extreme distance
            if distance_value > MIDDLE_DIST_IGNORE[0] and distance_value < MIDDLE_DIST_IGNORE[1]:
                continue
            
            tt_pair_index.append([i, j])
            tt_dis.append(distance_value)
            distance_e = (1-np.array(tt[i])) - np.array(tt[j])
            min_distance = min(np.linalg.norm(distance, ord=1), np.linalg.norm(distance_e, ord=1))
            min_tt_dis.append(min_distance / tt_len)

    return tt_pair_index, tt_dis, min_tt_dis

def dfs_PI(start_node, edge_index, x_data, gate_to_index):
    from collections import defaultdict

    # 构建邻接表（反向边）
    adj = defaultdict(list)
    src_nodes = []  
    dst_nodes = []
    for edge in edge_index:
        src_nodes.append(edge[0])  
        dst_nodes.append(edge[1])
    for i in range(len(src_nodes)):
        adj[int(dst_nodes[i])].append(int(src_nodes[i]))

    visited = set()
    result_nodes = set()

    def visit(u):
        if u in visited:
            return
        visited.add(u)

        gate_type = int(x_data[u][1])  
        #print(u, gate_type)
        if gate_type == gate_to_index['PDFF']:
            result_nodes.add(u)

        for v in adj[u]:
            visit(v)

    visit(start_node)

    return result_nodes

if __name__ == '__main__':
    graphs = {}
    labels = {}
    args = get_parse_args()
    output_folder = './data/{}'.format(args.exp_id)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    tot_circuit = 0
    cir_idx = 0
    tot_nodes = 0
    tot_pairs = 0
    name_list = []
    num_table = {}
    
    print('[INFO] Read bench from: ', args.aig_folder)
    for bench_filename in glob.glob(os.path.join(args.aig_folder, '*.bench')):
        tot_circuit += 1
        name_list.append(bench_filename)
    for bench_filename in name_list[args.start_idx: min(args.end_idx, len(name_list))]:
        print(bench_filename)
        circuit_name = bench_filename.split('/')[-1].split('.')[0]

        x_data, edge_index, fanin_list, fanout_list, level_list, idx2name = circuit_utils.parse_bench(bench_filename, gate_to_index)
        x_data_dff = []
        edge_index_data = []
        for idx, x in enumerate(x_data):
            if x[1] == gate_to_index['DFF']:
                x_data_dff.append(copy.deepcopy(x))
                start_node = idx
                reachable_nodes = dfs_PI(start_node, edge_index, x_data, gate_to_index)
                for node in reachable_nodes:
                    if x_data[node][1] == gate_to_index['PDFF']:
                        pnode_name = idx2name[node]
                        node_name = pnode_name.strip('P')
                        if node_name in idx2name.values():
                            node = list(idx2name.values()).index(node_name)
                    edge_index_data.append([node, start_node])
                
        # PI
        PI_index = []
        PPI_PI_index = level_list[0]
        for x in x_data:
            if x[1] == gate_to_index['INPUT']:
                PI_index.append(x[0])
        # Simulation 
        start_time = time.time()
        if len(PPI_PI_index) < 13:
            tt = circuit_utils.simulator_truth_table(x_data, PPI_PI_index, level_list, fanin_list, gate_to_index)
        else:
            tt = circuit_utils.simulator_truth_table_random(x_data, PPI_PI_index, level_list, fanin_list, gate_to_index, NO_PATTERNS)
        y = [0] * len(x_data) # 逻辑1概率
        index = 0
        for idx in range(len(x_data)):
            y[idx] = np.sum(tt[idx]) / len(tt[idx])
        number = pow(2, len(x_data_dff))
        x_data_py = [(int(a), int(b)) for a, b, _ in x_data]
        trans_matric = fsm_simulator.reachability_closure(
                                                            x_data=x_data_py,
                                                            PI_indexes=[int(x) for x in PI_index],
                                                            level_list=[[int(y) for y in level] for level in level_list],
                                                            fanin_list=[[int(y) for y in fanin] for fanin in fanin_list],
                                                            gate_to_index={str(k): int(v) for k,v in gate_to_index.items()},
                                                            idx2name={int(k): str(v) for k,v in idx2name.items()},
                                                            number=int(len(x_data_dff))
                                                        )
        finite_list = fsm_simulator.simulate_one_step_matrix(
                                                            x_data=x_data_py,
                                                            PI_indexes=[int(x) for x in PI_index],
                                                            level_list=[[int(y) for y in level] for level in level_list],
                                                            fanin_list=[[int(y) for y in fanin] for fanin in fanin_list],
                                                            gate_to_index={str(k): int(v) for k,v in gate_to_index.items()},
                                                            idx2name={int(k): str(v) for k,v in idx2name.items()},
                                                            number=int(len(x_data_dff))
                                                        )
        end_time = time.time()
        tt_dff = [0] * len(x_data_dff)
        index = 0
        for idx in range(len(x_data)):
            if x_data[idx][1] == gate_to_index['DFF']:
                tt_dff[index] = tt[idx]
                index += 1
        t_01, t_10 = estimate_transition_prob_from_tt(tt)


        # Save 
        x_data = utils.rename_node(x_data)
        graphs[circuit_name] = {'x': np.array(x_data).astype('float32'), "edge_index": np.array(edge_index)}
        labels[circuit_name] = {
            'prob_logic1': np.array([y]).astype('float32'),
            'tt_01': np.array(t_01).astype('float32'),'tt_10': np.array(t_10).astype('float32'),
            'finite_list':np.array(finite_list).astype('float32'),
            'trans_matric':np.array(trans_matric).astype('float32'),
        }
        tot_nodes += len(x_data_dff)
        print('Save: {}, # DFF: {:} , time: {:.2f} s ({:} / {:})'.format(
            circuit_name, len(x_data_dff), end_time - start_time, cir_idx, args.end_idx - args.start_idx
        ))

        if cir_idx != 0 and cir_idx % 1000 == 0:
            output_filename_circuit = os.path.join(output_folder, 'tmp_{:}_graphs.npz'.format(cir_idx))
            output_filename_labels = os.path.join(output_folder, 'tmp_{:}_labels.npz'.format(cir_idx))
            np.savez_compressed(output_filename_circuit, circuits=graphs)
            np.savez_compressed(output_filename_labels, labels=labels)
        cir_idx += 1
    
    output_filename_circuit = os.path.join(output_folder, 'graphs.npz')
    output_filename_labels = os.path.join(output_folder, 'labels.npz')
    print('# Graphs: {:}, # Nodes: {:}'.format(len(graphs), tot_nodes))
    print('Total pairs: ', tot_pairs)
    np.savez_compressed(output_filename_circuit, circuits=graphs)
    np.savez_compressed(output_filename_labels, labels=labels)
    print(output_filename_circuit)
