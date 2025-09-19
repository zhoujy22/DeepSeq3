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
import torch
from collections import Counter

import utils.circuit_utils as circuit_utils
import utils.utils as utils

# aig_folder = './rawaig/'
NO_PATTERNS = 15000

gate_to_index = {'INPUT': 0, 'AND': 1, 'NOT': 2, 'DFF': 3, 'PDFF': 4, 'OUTPUT': 5, 'NOR': 6, 'OR': 7, 'const': 8, 'BUF': 9}
MIN_LEVEL = 3
MIN_PI_SIZE = 4
MAX_INCLUDE = 1.5
MAX_PROB_GAP = 0.05
MAX_LEVEL_GAP = 5

MIDDLE_DIST_IGNORE = [0.2, 0.8]

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default='trainV2')
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
            t_01.append(num_01 / (len(node_tt) - 1))
            t_10.append(num_10 / (len(node_tt) - 1))

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
    print('[INFO] Read bench from: ', args.aig_folder)
    for bench_filename in glob.glob(os.path.join(args.aig_folder, '*.bench')):
        tot_circuit += 1
        name_list.append(bench_filename)
    for bench_filename in name_list[args.start_idx: min(args.end_idx, len(name_list))]:
        print(bench_filename)
        circuit_name = bench_filename.split('/')[-1].split('.')[0]

        x_data, edge_index, fanin_list, fanout_list, level_list, idx2name = circuit_utils.parse_bench(bench_filename, gate_to_index)
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
            print("random simulate")
        y = [0] * len(x_data) # 逻辑1概率
        for idx in range(len(x_data)):
            y[idx] = np.sum(tt[idx]) / len(tt[idx])
            
        # Pair
        tt_pair_index, tt_dis, min_tt_dis = gen_tt_pair(x_data, fanin_list, fanout_list, level_list, y)
        end_time = time.time()
        t_01, t_10 = estimate_transition_prob_from_tt(tt)
        # get ff truth table
        ff_indexes = [i for i, x in enumerate(x_data) if x[1] == gate_to_index['DFF']]
        ff_tt = [tt[idx] for idx in ff_indexes]  # 每个 FF 的 truth table 行向量
        ff_tt = np.array(ff_tt)
        # calculate FF similarity
        ff_pair_index = []
        ff_sim = []
        is_ff_equ = []
        sim_threshold = 0.05
        for i, j in combinations(range(len(ff_indexes)), 2):
            vec_i = np.array(ff_tt[i])
            vec_j = np.array(ff_tt[j])
            diff = np.abs(vec_i - vec_j)
            sim_score = 1 - np.sum(diff) / len(vec_i)
            ff_pair_index.append([ff_indexes[i], ff_indexes[j]])
            ff_sim.append(sim_score)
            is_ff_equ.append(int(sim_score > (1 - sim_threshold)))
        
        # Save 
        x_data = utils.rename_node(x_data)
        graphs[circuit_name] = {'x': np.array(x_data).astype('float32'), "edge_index": np.array(edge_index)}
        labels[circuit_name] = {
            'tt_pair_index': np.array(tt_pair_index), 'tt_dis': np.array(tt_dis).astype('float32'), 
            'prob_logic1': np.array(y).astype('float32'), 
            'prob_logic0': 1 - np.array(y).astype('float32'),
            'min_tt_dis': np.array(min_tt_dis).astype('float32'), 
            'ff_pair_index': np.array(ff_pair_index), 'ff_sim': np.array(ff_sim).astype('float32'),
            'is_ff_equ': np.array(is_ff_equ).astype('int8'),
            't_01': np.array(t_01).astype('float32'), 't_10': np.array(t_10).astype('float32'),
        }
        tot_nodes += len(x_data)
        tot_pairs += len(tt_dis)
        print('Save: {}, # PI: {:}, Tot Pairs: {:.1f}k, time: {:.2f} s ({:} / {:})'.format(
            circuit_name, len(PI_index), tot_pairs/1000, end_time - start_time, cir_idx, args.end_idx - args.start_idx
        ))

        
        cir_idx += 1

    output_filename_circuit = os.path.join(output_folder, 'graphs.npz')
    output_filename_labels = os.path.join(output_folder, 'labels.npz')
    print('# Graphs: {:}, # Nodes: {:}'.format(len(graphs), tot_nodes))
    print('Total pairs: ', tot_pairs)
    np.savez_compressed(output_filename_circuit, circuits=graphs)
    np.savez_compressed(output_filename_labels, labels=labels)
    print(output_filename_circuit)
