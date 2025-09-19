from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import torch
from torch_geometric import data
from torch_geometric.loader import DataLoader, DataListLoader

from config import get_parse_args
from models.model import create_model, load_model, save_model
from utils.logger import Logger
from utils.random_seed import set_seed
from utils.circuit_utils import check_difference
from trains.train_factory import train_factory
from datasets.mig_dataset import MIGDataset
from datasets.mlpgate_dataset import MLPGateDataset


import torch
import shutil
import os
import copy
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from utils.data_utils import read_npz_file
from datasets.load_data import parse_pyg_mlpgate

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main(args):
    print('==> Using settings {}'.format(args))

    #################
    # Device 
    #################
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
    args.device = torch.device('cuda:0' if args.gpus[0] >= 0 else 'cpu')
    args.world_size = 1
    args.rank = 0  # global rank
    if args.device != 'cpu' and len(args.gpus) > 1:
        args.distributed = len(args.gpus)
    else:
        args.distributed = False
    if args.distributed:
        if 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.getenv('LOCAL_RANK'))
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print('Training in distributed mode. Device {}, Process {:}, total {:}.'.format(
            args.device, args.rank, args.world_size
        ))
    else:
        print('Training in single device: ', args.device)
    if args.local_rank == 0:
        logger = Logger(args, args.local_rank)
        load_model_path = os.path.join(args.save_dir, 'model_last.pth')
        if args.resume and not os.path.exists(load_model_path):
            if args.pretrained_path == '':
                raise "No pretrained model (.pth) found"
            else:
                shutil.copy(args.pretrained_path, load_model_path)
                print('Copy pth from: ', args.pretrained_path)
        
    #################
    # Dataset
    #################
    """
    if args.local_rank == 0:
        print('==> Loading dataset from: ', args.data_dir)
    dataset = MLPGateDataset(args.data_dir, args)
    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]
    data_len = len(dataset)
    if args.local_rank == 0:
        print("Size: ", len(dataset))
        print('Splitting the dataset into training and validation sets..')
    training_cutoff = int(data_len * args.trainval_split)
    if args.local_rank == 0:
        print('# training circuits: ', training_cutoff)
        print('# validation circuits: ', data_len - training_cutoff)
    """

    #################
    # Model
    #################
    model = create_model(args)
    if args.local_rank == 0:
        print('==> Creating model...')
        print(model)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    if args.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, args.load_model, optimizer, args.resume, args.lr, args.lr_step, args.local_rank, args.device)

    Trainer =  train_factory[args.arch]
    trainer = Trainer(args, model, optimizer)
    trainer.set_device(args.device, args.local_rank, args.gpus)
    
    circuits = read_npz_file(args.circuit_file, args.data_dir)['circuits'].item()
    labels = read_npz_file(args.label_file, args.data_dir)['labels'].item()
        
    if args.small_train:
        subset = 100

    for cir_idx, cir_name in enumerate(circuits):
        if cir_idx == 0:
                continue
        print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100))
           
        x = circuits[cir_name]["x"]
        edge_index = circuits[cir_name]["edge_index"]
        gnn_rounds = circuits[cir_name]["gnn_rounds"]
            # logic prob
        y_prob = labels[cir_name]['y']

             # trans prob
        y_01 = torch.tensor(labels[cir_name]["t_01"]).reshape([len(x), 1])
        y_10 = torch.tensor(labels[cir_name]["t_10"]).reshape([len(x), 1])

        y_trans_prob = torch.cat([y_01, y_10], dim=1)
          
        if args.no_rc:
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
            
        diff_pair_index = labels[cir_name]['diff_pair_index']
        is_trans_diff = labels[cir_name]['is_trans_diff']
        trans_state_diff = labels[cir_name]['trans_state_diff']
            
        tot_pairs = 0
        if len(rc_pair_index) == 0 or len(diff_pair_index) == 0:
                print('No tt,ff or rc pairs: ', cir_name)
                continue

        tot_pairs += (len(tt_diff)/len(x))
            # check the gate types
            # assert (x[:, 1].max() == (len(self.args.gate_to_index)) - 1), 'The gate types are not consistent.'
          
        data_list = []
        status, graph = parse_pyg_mlpgate(
                x, edge_index, gnn_rounds, y_trans_prob, y_prob, tt_pair_index,  tt_diff, rc_pair_index, is_rc, diff_pair_index, is_trans_diff, trans_state_diff,
                args.use_edge_attr, args.reconv_skip_connection, args.no_node_cop,
                args.node_reconv, args.un_directed, args.num_gate_types,
                args.dim_edge_feature, args.logic_implication, args.mask,args.gate_to_index,test_data= True
            )
        if not status:
                continue
        graph.name = cir_name
        data_list.append(graph)
       
        train_dataset = []
        val_dataset = []

        data_len = len(data_list)
        training_cutoff = int(data_len * args.trainval_split)

        train_dataset = data_list[:training_cutoff]
        val_dataset = data_list[training_cutoff:]

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                sampler=val_sampler)
      
        if args.val_only:
            log_dict_val, _ = trainer.val(0, val_loader,args.local_rank)
        
        return

        if args.local_rank == 0:
            print('==> Starting training...')
        best = 1e10
        for epoch in range(start_epoch + 1, args.num_epochs + 1):
            mark = epoch if args.save_all else 'last'
            train_loader.sampler.set_epoch(epoch)
            log_dict_train, _ = trainer.train(epoch, train_loader, args.local_rank)
            if args.local_rank == 0:
                logger.write('epoch: {} |'.format(epoch), args.local_rank)
                for k, v in log_dict_train.items():
                    logger.scalar_summary('train_{}'.format(k), v, epoch, args.local_rank)
                    logger.write('{} {:8f} | '.format(k, v), args.local_rank)
                if args.save_intervals > 0 and epoch % args.save_intervals == 0:
                    save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(mark)),
                            epoch, model, optimizer)
            with torch.no_grad():
                val_loader.sampler.set_epoch(0)
                log_dict_val, _ = trainer.val(epoch, val_loader, args.local_rank)

            if args.local_rank == 0:
                for k, v in log_dict_val.items():
                    logger.scalar_summary('val_{}'.format(k), v, epoch, args.local_rank)
                    logger.write('{} {:8f} | '.format(k, v), args.local_rank)
                if log_dict_val[args.metric] < best:
                    best = log_dict_val[args.metric]
                    save_model(os.path.join(args.save_dir, 'model_best.pth'),
                            epoch, model)
                else:
                    save_model(os.path.join(args.save_dir, 'model_last.pth'),
                            epoch, model, optimizer)
            if args.local_rank == 0:
                logger.write('\n', args.local_rank)
            if epoch in args.lr_step:
                if args.local_rank == 0:
                    save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(epoch)),
                            epoch, model, optimizer)
                #lr = args.lr * (0.1 ** (args.lr_step.index(epoch) + 1))
                lr = args.lr 
                if args.local_rank == 0:
                    #print('Drop LR to', lr)
                    print('Maintain LR to', lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        if args.local_rank == 0:
            logger.close()


if __name__ == '__main__':
 
    args = get_parse_args()

    set_seed(args)

    main(args)
