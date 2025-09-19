from typing import Optional, Callable, List
import os.path as osp
import numpy as np
import torch
import shutil
import os
import copy
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import AddRandomWalkPE
from utils.data_utils import read_npz_file
from .load_data import parse_pyg_mlpgate, parse_pyg_mlpgate2, parse_pyg_mlpgate3
import sys
class MLPGateDataset(InMemoryDataset):
    r"""
    A variety of circuit graph datasets, *e.g.*, open-sourced benchmarks,
    random circuits.

    Args:
        root (string): Root directory where the dataset should be saved.
        args (object): The arguments specified by the main program.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root, args, transform=None, pre_transform=None, pre_filter=None):
        self.name = 'AIG'
        self.args = args
        sys.setrecursionlimit(100000)

        assert (transform == None) and (pre_transform == None) and (pre_filter == None), "Cannot accept the transform, pre_transfrom and pre_filter args now."

        # Reload
        inmemory_dir = os.path.join(args.data_dir, 'inmemory')
        if args.reload_dataset and os.path.exists(inmemory_dir):
            shutil.rmtree(inmemory_dir)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
         

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        if self.args.small_train:
            name = 'inmemory_small'
        else:
            name = 'inmemory'
        if self.args.no_rc:
            name += '_norc'
        return osp.join(self.root, name)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.args.circuit_file, self.args.label_file]

    @property
    def processed_file_names(self) -> str:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        tot_pairs = 0
        circuits = read_npz_file(self.args.circuit_file, self.args.data_dir)['circuits'].item()
        labels = read_npz_file(self.args.label_file, self.args.data_dir)['labels'].item()
   
       
        if self.args.small_train:
            subset = 10

        for cir_idx, cir_name in enumerate(circuits):
            

            print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100))
       
            x = circuits[cir_name]["x"]
            edge_index = circuits[cir_name]["edge_index"]

            #gnn_rounds = circuits[cir_name]["gnn_rounds"]

            # logic prob
            y_prob1 = torch.tensor(labels[cir_name]['prob_logic1'])
            y_prob0 = torch.tensor(labels[cir_name]['prob_logic0'])
            

             # trans prob
            y_01 = torch.tensor(labels[cir_name]["t_01"]).reshape([len(x), 1])
            y_10 = torch.tensor(labels[cir_name]["t_10"]).reshape([len(x), 1])
            

            y_trans_prob = torch.cat([y_01, y_10], dim=1)
            if torch.isnan(y_trans_prob).any():
                print("Error: y_trans_prob contains NaN values!")
                exit(0)
            if self.args.no_rc:
                rc_pair_index = [[0, 1]]
                is_rc = [0]
            else:
                rc_pair_index = labels[cir_name]['rc_pair_index']
                is_rc = labels[cir_name]['is_rc']
        
            tt_len = len(labels[cir_name]['tt_dis'])
            tt_pair_index = labels[cir_name]['tt_pair_index']
            tt_diff = labels[cir_name]['tt_dis']
            tt_pair_index = tt_pair_index.reshape([tt_len, 2])
            tt_diff = tt_diff.reshape(tt_len)
           
            
            ff_len = len(labels[cir_name]['ff_sim'])
            ff_pair_index = labels[cir_name]['ff_pair_index']
            is_ff_equ = labels[cir_name]['is_ff_equ']
            ff_sim = labels[cir_name]['ff_sim']
            ff_pair_index = torch.tensor(ff_pair_index).reshape(ff_len,2)

            ff_sim = torch.tensor(ff_sim).reshape(ff_len)

            if len(rc_pair_index) == 0 or len(ff_pair_index) == 0 or len(tt_pair_index) == 0:
                print('No tt,ff or rc pairs: ', cir_name)
                continue

            tot_pairs += (len(tt_diff)/len(x))
            
            if len(tt_pair_index) == 0:
                print('No tt,ff or rc pairs: ', cir_name)
                continue
            status, graph = parse_pyg_mlpgate(
                x, edge_index,y_trans_prob, y_prob1, y_prob0, tt_pair_index,  tt_diff, rc_pair_index, is_rc, ff_pair_index, is_ff_equ, ff_sim,
                self.args.use_edge_attr, self.args.reconv_skip_connection, self.args.no_node_cop,
                self.args.node_reconv, self.args.un_directed, self.args.num_gate_types,
                self.args.dim_edge_feature, self.args.logic_implication, self.args.mask,self.args.gate_to_index, self.args.test_data)
            
            if not status:
                continue
            print(graph)
            print(graph.keys())
            data_list.append(graph)
            
            if self.args.small_train and cir_idx > subset:
                break
            print(len(data_list))
           
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
        print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))
       
    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

class MLPGateDataset2(InMemoryDataset):

    def __init__(self, root, args, transform=None, pre_transform=None, pre_filter=None):
        self.name = 'SNG'
        self.args = args
        sys.setrecursionlimit(100000)

        assert (transform == None) and (pre_transform == None) and (pre_filter == None), "Cannot accept the transform, pre_transfrom and pre_filter args now."

        # Reload
        inmemory_dir = os.path.join(args.data_dir, 'inmemory')
        if args.reload_dataset and os.path.exists(inmemory_dir):
            shutil.rmtree(inmemory_dir)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        finite_list_path = osp.join(self.processed_dir, 'finite_list.pt')
        trans_matric_path = osp.join(self.processed_dir, 'trans_matric.pt')
        if os.path.exists(finite_list_path):
            self.finite_list_list = torch.load(finite_list_path)
        else:
            self.finite_list_list = None    
        
        if os.path.exists(trans_matric_path):
            self.trans = torch.load(trans_matric_path)
        else:
            self.trans = None

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        if self.args.small_train:
            name = 'inmemory_small'
        else:
            name = 'inmemory'
        if self.args.no_rc:
            name += '_norc'
        return osp.join(self.root, name)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.args.circuit_file, self.args.label_file]

    @property
    def processed_file_names(self) -> str:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        probs_list = {}
        tot_pairs = 0
        finite_list_dict = {}
        trans_matric_dict = {}
        circuits = read_npz_file(self.args.circuit_file, self.args.data_dir)['circuits'].item()
        labels = read_npz_file(self.args.label_file, self.args.data_dir)['labels'].item()
   
       
        if self.args.small_train:
            subset = 10
        h_init_dict = torch.load("/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/exp/stage1_embedding.pt")
        print(len(h_init_dict))
        for cir_idx, cir_name in enumerate(circuits):
            print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100))
            x = circuits[cir_name]["x"]
            edge_index = circuits[cir_name]["edge_index"]
            node_ids = x[:, 0].astype(int)  # 已升序
            edge_index_remap = np.searchsorted(node_ids, edge_index.astype(int))
            x_remap = x.copy()
            x_remap[:, 0] = np.arange(len(x))
            old2new = {old: new for new, old in enumerate(node_ids)}
            #gnn_rounds = circuits[cir_name]["gnn_rounds"]

            # logic prob
            y_prob = labels[cir_name]['prob_logic1']

             # trans prob
            y_01 = torch.tensor(labels[cir_name]["tt_01"]).reshape([len(x), 1])
            y_10 = torch.tensor(labels[cir_name]["tt_10"]).reshape([len(x), 1])

            y_trans_prob = torch.cat([y_01, y_10], dim=1)
          

            finite_list = labels[cir_name]['finite_list']
            trans_matric = labels[cir_name]['trans_matric']
            print(finite_list)
            if cir_name in h_init_dict:
                h_init_data = h_init_dict[cir_name]  # list of dicts
                h_init_tensor = torch.zeros((x_remap.shape[0], self.args.dim_hidden), dtype=torch.float32)
                h_init_tensor0 = torch.zeros((x_remap.shape[0], self.args.dim_hidden), dtype=torch.float32)

                for entry in h_init_data:
                    old_idx = entry["cell_idx"]
                    if old_idx not in old2new:
                        continue  # 有些 cell_idx 可能未出现在图中
                    new_idx = old2new[old_idx]
                    embedding = torch.tensor(entry["embedding_h1"], dtype=torch.float32)
                    h_init_tensor[new_idx] = embedding
                    h_init_tensor0[new_idx] = torch.tensor(entry["embedding_h0"], dtype=torch.float32)
                h_init1 = h_init_tensor
                h_init0 = h_init_tensor0
            else:
                print(f"[Warning] No h_init found for circuit: {cir_name}")
                continue
            status, graph = parse_pyg_mlpgate2(
                x_remap, edge_index_remap,y_trans_prob, y_prob, h_init1, h_init0,
                self.args.use_edge_attr, self.args.reconv_skip_connection, self.args.no_node_cop,
                self.args.node_reconv, self.args.un_directed, self.args.num_gate_types,
                self.args.dim_edge_feature, self.args.logic_implication, self.args.mask,self.args.gate_to_index, self.args.test_data)
            print("Done!")
            graph.name = cir_name
            
            print(graph)
            if graph.x.shape[0] != graph.h_init1.shape[0]:
                print(f"[Warning] Mismatch in node count: {graph.x.shape[0]} vs {graph.h_init.shape[0]} for circuit {cir_name}")
                continue

            data_list.append(graph)
            # low_thr = getattr(self.args, "low_thr", 0.2)
            # low_keep_rate = getattr(self.args, "low_keep_rate", 0.01)
            # max_keep = getattr(self.args, "max_keep", None)
            # seed = getattr(self.args, "seed", 42)

            finite_np = np.asarray(finite_list)  # [N, N]
            trans_matric_np = np.asarray(trans_matric)
            # probs_np = np.asarray(probs)         # [S]

            # S = probs_np.shape[0]
            # rng = np.random.default_rng(seed)

            # low_mask = probs_np < low_thr               # [S]
            # high_mask = ~low_mask                       # [S]
            # rand_keep = rng.random(S) < low_keep_rate
            # keep_mask = high_mask | (low_mask & rand_keep)
            # keep_idx = np.nonzero(keep_mask)[0]

            # if keep_idx.size == 0:
            #     keep_idx = np.array([int(np.argmax(probs_np))])  # 至少留一个最高概率的

           

            # finite_np = finite_np[keep_idx]
            # probs_np = probs_np[keep_idx]
            ### <<< 下采样结束
            finite_list = torch.tensor(finite_np, dtype=torch.float32)
            trans_matric = torch.tensor(trans_matric_np, dtype=torch.float32)
            # prob_t = torch.tensor(probs_np, dtype=torch.float32)
            finite_list_dict[cir_name] = finite_list
            trans_matric_dict[cir_name] = trans_matric
            # probs_list[cir_name]=prob_t
            if self.args.small_train and cir_idx > subset:
                break
            print(len(data_list))

            print(graph.keys())
        data_list = [g for g in data_list if g.edge_index.numel() > 0]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(finite_list_dict, osp.join(self.processed_dir, 'finite_list.pt'))
        torch.save(trans_matric_dict, osp.join(self.processed_dir, 'trans_matric.pt'))
        # torch.save(probs_list, osp.join(self.processed_dir, 'probs_list.pt'))
        print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
        print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))
       
    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

class MLPGateDataset3(InMemoryDataset):

    def __init__(self, root, args, transform=None, pre_transform=None, pre_filter=None):
        self.name = 'SNG'
        self.args = args
        sys.setrecursionlimit(100000)

        assert (transform == None) and (pre_transform == None) and (pre_filter == None), "Cannot accept the transform, pre_transfrom and pre_filter args now."

        # Reload
        inmemory_dir = os.path.join(args.data_dir, 'inmemory')
        if args.reload_dataset and os.path.exists(inmemory_dir):
            shutil.rmtree(inmemory_dir)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        finite_list_path = osp.join(self.processed_dir, 'finite_list.pt')
        if os.path.exists(finite_list_path):
            self.finite_list_list = torch.load(finite_list_path)
        else:
            self.finite_list_list = None    
        trans_matric_path = osp.join(self.processed_dir, 'trans_matric.pt')
        if os.path.exists(trans_matric_path):
            self.trans = torch.load(trans_matric_path)
        else:
            self.trans = None

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        if self.args.small_train:
            name = 'inmemory_small'
        else:
            name = 'inmemory'
        if self.args.no_rc:
            name += '_norc'
        return osp.join(self.root, name)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.args.circuit_file, self.args.label_file]

    @property
    def processed_file_names(self) -> str:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        probs_list = {}
        tot_pairs = 0
        finite_list_dict = {}
        trans_matric_dict = {}
        circuits = read_npz_file(self.args.circuit_file, self.args.data_dir)['circuits'].item()
        labels = read_npz_file(self.args.label_file, self.args.data_dir)['labels'].item()
   
       
        if self.args.small_train:
            subset = 10
        
        for cir_idx, cir_name in enumerate(circuits):
            print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100))
            x = circuits[cir_name]["x"]
            edge_index = circuits[cir_name]["edge_index"]
            # node_ids = x[:, 0].astype(int)  # 已升序
            # edge_index_remap = np.searchsorted(node_ids, edge_index.astype(int))
            # x_remap = x.copy()
            # x_remap[:, 0] = np.arange(len(x))
            # old2new = {old: new for new, old in enumerate(node_ids)}
            #gnn_rounds = circuits[cir_name]["gnn_rounds"]

            # logic prob
            y_prob = labels[cir_name]['prob_logic1']

             # trans prob
            y_01 = torch.tensor(labels[cir_name]["tt_01"]).reshape([len(x), 1])
            y_10 = torch.tensor(labels[cir_name]["tt_10"]).reshape([len(x), 1])

            y_trans_prob = torch.cat([y_01, y_10], dim=1)
          

            finite_list = labels[cir_name]['finite_list']
            trans_matric = labels[cir_name]['trans_matric']
            

            status, graph = parse_pyg_mlpgate3(
                x, edge_index,y_trans_prob, y_prob, 
                self.args.use_edge_attr, self.args.reconv_skip_connection, self.args.no_node_cop,
                self.args.node_reconv, self.args.un_directed, self.args.num_gate_types,
                self.args.dim_edge_feature, self.args.logic_implication, self.args.mask,self.args.gate_to_index, self.args.test_data)
            print("Done!")
            graph.name = cir_name
            
            print(graph)
            

            data_list.append(graph)
            # low_thr = getattr(self.args, "low_thr", 0.2)
            # low_keep_rate = getattr(self.args, "low_keep_rate", 0.01)
            # max_keep = getattr(self.args, "max_keep", None)
            # seed = getattr(self.args, "seed", 42)

            finite_np = np.asarray(finite_list)  # [N, N]
            trans_matric_np = np.asarray(trans_matric)
            # probs_np = np.asarray(probs)         # [S]

            # S = probs_np.shape[0]
            # rng = np.random.default_rng(seed)

            # low_mask = probs_np < low_thr               # [S]
            # high_mask = ~low_mask                       # [S]
            # rand_keep = rng.random(S) < low_keep_rate
            # keep_mask = high_mask | (low_mask & rand_keep)
            # keep_idx = np.nonzero(keep_mask)[0]

            # if keep_idx.size == 0:
            #     keep_idx = np.array([int(np.argmax(probs_np))])  # 至少留一个最高概率的

           

            # finite_np = finite_np[keep_idx]
            # probs_np = probs_np[keep_idx]
            ### <<< 下采样结束
            finite_list = torch.tensor(finite_np, dtype=torch.float32)
            trans_matric = torch.tensor(trans_matric_np, dtype=torch.float32)

            # prob_t = torch.tensor(probs_np, dtype=torch.float32)
            finite_list_dict[cir_name] = finite_list
            trans_matric_dict[cir_name] = finite_list
            # probs_list[cir_name]=prob_t
            if self.args.small_train and cir_idx > subset:
                break
            print(len(data_list))

            print(graph.keys())
        data_list = [g for g in data_list if g.edge_index.numel() > 0]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(finite_list_dict, osp.join(self.processed_dir, 'finite_list.pt'))
        torch.save(trans_matric_dict, osp.join(self.processed_dir, 'trans_matric.pt'))
        # torch.save(probs_list, osp.join(self.processed_dir, 'probs_list.pt'))
        print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
        print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))
       
    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

class GraphGPSdataset(InMemoryDataset):
    r"""
    A variety of circuit graph datasets, *e.g.*, open-sourced benchmarks,
    random circuits.

    Args:
        root (string): Root directory where the dataset should be saved.
        args (object): The arguments specified by the main program.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root, args, transform=None, pre_transform=None, pre_filter=None):
        self.name = 'AIG'
        self.args = args
        sys.setrecursionlimit(100000)

        assert (transform == None) and (pre_transform == None) and (pre_filter == None), "Cannot accept the transform, pre_transfrom and pre_filter args now."

        # Reload
        inmemory_dir = os.path.join(args.data_dir, 'inmemory')
        if args.reload_dataset and os.path.exists(inmemory_dir):
            shutil.rmtree(inmemory_dir)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
         

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        if self.args.small_train:
            name = 'inmemory_small'
        else:
            name = 'inmemory'
        if self.args.no_rc:
            name += '_norc'
        return osp.join(self.root, name)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.args.circuit_file, self.args.label_file]

    @property
    def processed_file_names(self) -> str:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        tot_pairs = 0
        circuits = read_npz_file(self.args.circuit_file, self.args.data_dir)['circuits'].item()
        labels = read_npz_file(self.args.label_file, self.args.data_dir)['labels'].item()
   
       
        if self.args.small_train:
            subset = 10

        for cir_idx, cir_name in enumerate(circuits):
            

            print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100))
       
            x = circuits[cir_name]["x"]
            edge_index = circuits[cir_name]["edge_index"]

            #gnn_rounds = circuits[cir_name]["gnn_rounds"]

            # logic prob
            y_prob1 = torch.tensor(labels[cir_name]['prob_logic1'])
            y_prob0 = torch.tensor(labels[cir_name]['prob_logic0'])
            

             # trans prob
            y_01 = torch.tensor(labels[cir_name]["t_01"]).reshape([len(x), 1])
            y_10 = torch.tensor(labels[cir_name]["t_10"]).reshape([len(x), 1])
            

            y_trans_prob = torch.cat([y_01, y_10], dim=1)
            if torch.isnan(y_trans_prob).any():
                print("Error: y_trans_prob contains NaN values!")
                exit(0)
            if self.args.no_rc:
                rc_pair_index = [[0, 1]]
                is_rc = [0]
            else:
                rc_pair_index = labels[cir_name]['rc_pair_index']
                is_rc = labels[cir_name]['is_rc']
        
            tt_len = len(labels[cir_name]['tt_dis'])
            tt_pair_index = labels[cir_name]['tt_pair_index']
            tt_diff = labels[cir_name]['tt_dis']
            tt_pair_index = tt_pair_index.reshape([tt_len, 2])
            tt_diff = tt_diff.reshape(tt_len)
           
            
            ff_len = len(labels[cir_name]['ff_sim'])
            ff_pair_index = labels[cir_name]['ff_pair_index']
            is_ff_equ = labels[cir_name]['is_ff_equ']
            ff_sim = labels[cir_name]['ff_sim']
            ff_pair_index = torch.tensor(ff_pair_index).reshape(ff_len,2)

            ff_sim = torch.tensor(ff_sim).reshape(ff_len)

            if len(rc_pair_index) == 0 or len(ff_pair_index) == 0 or len(tt_pair_index) == 0:
                print('No tt,ff or rc pairs: ', cir_name)
                continue

            tot_pairs += (len(tt_diff)/len(x))
            
            if len(tt_pair_index) == 0:
                print('No tt,ff or rc pairs: ', cir_name)
                continue
            status, graph = parse_pyg_mlpgate(
                x, edge_index,y_trans_prob, y_prob1, y_prob0, tt_pair_index,  tt_diff, rc_pair_index, is_rc, ff_pair_index, is_ff_equ, ff_sim,
                self.args.use_edge_attr, self.args.reconv_skip_connection, self.args.no_node_cop,
                self.args.node_reconv, self.args.un_directed, self.args.num_gate_types,
                self.args.dim_edge_feature, self.args.logic_implication, self.args.mask,self.args.gate_to_index, self.args.test_data)
            
            if not status:
                continue
            # transform = AddRandomWalkPE(walk_length=self.args.randomwalk, attr_name='pe')
            # graph = transform(graph)
            graph.name = cir_name
            print(graph)
            print(graph.keys())
            data_list.append(graph)
            
            if self.args.small_train and cir_idx > subset:
                break
            print(len(data_list))
           
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
        print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))
       
    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
