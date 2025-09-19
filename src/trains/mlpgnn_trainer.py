from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import DataParallel
from progress.bar import Bar
from utils.utils import AverageMeter, zero_normalization, get_function_acc, save_embedding_json
from torch import distributed as dist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils.utils import SoftLabelCrossEntropy
from sklearn.metrics import f1_score, precision_score, recall_score
_loss_factory = {
    # Regression
    'l1': nn.L1Loss,
    'sl1': nn.SmoothL1Loss,
    'l2': nn.MSELoss,
    # Classification
    'bce': nn.BCEWithLogitsLoss,
    'CrossEntropy':nn.KLDivLoss
}

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, reg_loss, cls_loss, gpus, device):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.reg_loss = reg_loss
        self.cls_loss = cls_loss
        self.gpus = gpus
        self.device = device
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, stage):
      
        preds , max_sim  =  self.model(batch)
        pattern = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device=batch.x.device)
        mask = torch.any(batch.x != pattern, dim=1)
        hs, hf, y_prob1, y_prob0 = preds
        
        # Task 1: Logic Probability Prediction
        prob_loss1 = self.reg_loss(y_prob1[mask].to(self.device), batch.y_prob1[mask].to(self.device))
        prob_loss0 = self.reg_loss(y_prob0[mask].to(self.device), batch.y_prob0[mask].to(self.device))
        # Task 2: Trans Probability Prediction
        #trans_loss = self.reg_loss(trans_prob[mask].to(self.device), batch.y_trans_prob[mask].to(self.device))
        # Task 3: Structural Prediction
        #rc_loss = torch.tensor(0.0)
        
        
        # # Task 4: Function Prediction
        # node_a = hf[batch.tt_pair_index[0]]
        # node_b = hf[batch.tt_pair_index[1]]
        # emb_dis = 1 - torch.cosine_similarity(node_a, node_b, eps=1e-8)

        # func_loss = self.reg_loss(emb_dis.to(self.device), batch.tt_diff.to(self.device))
    
        # if stage == 2:
        #     # Task 5: Sequential Behavior Prediction
        #     node_a = hseq[batch.ff_pair_index[0]]
        #     node_b = hseq[batch.ff_pair_index[1]]
        #     emb_dis = torch.cosine_similarity(node_a, node_b, eps=1e-8)
        #     seq_loss = self.reg_loss(emb_dis.to(self.device), batch.ff_sim[0].to(self.device))
        # else:
        #     seq_loss = torch.tensor(0.0)
        
        loss_stats = {'LProb1': prob_loss1,  'LProb0': prob_loss0}
    
        return hs, hf, y_prob1, y_prob0, loss_stats

class ModelWithLoss3(torch.nn.Module):
    def __init__(self, model, reg_loss, cls_loss, gpus, device):
        super(ModelWithLoss3, self).__init__()
        self.model = model
        self.reg_loss = reg_loss
        self.cls_loss = cls_loss
        self.gpus = gpus
        self.device = device
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, stage, finite_list_full, trans_matrics):
        
        logits, labels, preds, ground_true =  self.model(batch, finite_list_full, trans_matrics)
        clsloss = self.cls_loss(logits.to(self.device), labels.to(self.device))
        regloss = self.reg_loss(preds.to(self.device), ground_true.to(self.device))
        logits = self.sigmoid(logits)
        pred = {"pred_label": logits, "ground_true_label": labels,"pred_prob": preds, "ground_true": ground_true}
        loss_stats = {'cls_loss': clsloss, "reg_loss":regloss}  # for logging
        return pred, loss_stats

class ModelWithLoss2(torch.nn.Module):
    def __init__(self, model, reg_loss, cls_loss, gpus, device):
        super(ModelWithLoss2, self).__init__()
        self.model = model
        if reg_loss == nn.KLDivLoss:
            self.reg_loss = reg_loss(reduction='batchmean')
        else:   
            self.reg_loss = reg_loss
        if cls_loss == nn.BCEWithLogitsLoss:
            self.cls_loss = cls_loss(pos_weight=torch.tensor([3.0], device=device))
        else:
            self.cls_loss = cls_loss
        self.gpus = gpus
        self.device = device
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, stage, finite_list_full, trans_matrics):
        
        logits, labels, preds, ground_true =  self.model(batch, finite_list_full, trans_matrics)
        clsloss = self.cls_loss(logits.to(self.device), labels.to(self.device))
        regloss = self.reg_loss(preds.to(self.device), ground_true.to(self.device))
        preds = torch.exp(preds)
        
        pred = {"pred_label": logits, "ground_true_label": labels,"pred_prob": preds, "ground_true": ground_true}
        loss_stats = {'cls_loss': clsloss, "reg_loss":regloss}  # for logging
        return pred, loss_stats
        

class MLPGNNTrainer(object):
    def __init__(
            self, args, model, optimizer=None):
        self.args = args
        self.optimizer = optimizer
        self.loss_stats, self.reg_loss, self.cls_loss = self._get_losses(args.reg_loss, args.cls_loss)
        self.reg_loss = self.reg_loss.to(self.args.device)
        self.cls_loss = self.cls_loss.to(self.args.device)
        self.model_with_loss = ModelWithLoss(model, self.reg_loss, self.cls_loss, args.gpus, args.device)
    
    def set_weight(self, w_prob, w_rc, w_func, w_trans, w_seq):
        self.args.prob_weight = w_prob
        self.args.rc_weight = w_rc
        self.args.func_weight = w_func
        self.args.trans_weight = w_trans
        self.args.seq_weight = w_seq
        
    def set_device(self, device, local_rank, gpus):
        if len(gpus)> 1:
            self.model_with_loss = self.model_with_loss.to(device)
            self.model_with_loss = nn.parallel.DistributedDataParallel(self.model_with_loss,
                                                                       device_ids=[local_rank], 
                                                                       find_unused_parameters=True)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, dataset, local_rank):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.args.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        args = self.args
        results = {}
        result_dict = {}
        y_prob_list1 = []
        y_prob_list0 = []
        y_prob_pred1 = []
        y_prob_true = []
        y_prob_pred0 = []
        r2_list_prob = []
        r2_list_trans = []
        trans_list = []
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(dataset) if args.num_iters < 0 else args.num_iters
        if local_rank == 0:
            bar = Bar('{}/{}'.format(args.task, args.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(dataset):
            if iter_id >= num_iters:
                break
            if len(self.args.gpus) == 1:
                batch = batch.to(self.args.device)
            data_time.update(time.time() - end)
            hs, hf, y_prob1, y_prob0, loss_stats = model_with_loss(batch,self.args.stage)
                   
            loss = loss_stats['LProb1'] * 0.5 + loss_stats['LProb0'] *0.5
            loss = loss.mean()
 
            loss_stats['loss'] = loss
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model_with_loss.parameters(), args.grad_clip)
                self.optimizer.step()
            elif phase == 'val':
                    if epoch == 0:
                        circuit_names = batch.name if isinstance(batch.name, list) else [batch.name]
                        dff_mask = (batch.gate.squeeze(1) == 3)
                        dff_indices = torch.where(dff_mask)[0]
                        for name in circuit_names:
                            result_dict[name] = [
                                {
                                    "cell_idx": int(idx.item()),
                                    "embedding_h1": hf[idx].detach().cpu(),
                                    "embedding_h0": hs[idx].detach().cpu()
                                }
                                for idx in dff_indices
                            ]
            batch_time.update(time.time() - end)
            end = time.time()
            if phase == 'val':
                pattern = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device=batch.x.device)
                mask = torch.any(batch.x != pattern, dim=1)
                mae = torch.abs(y_prob1[mask] - batch.y_prob1[mask]).mean()
                mae_value1 = mae.item()
                y_prob_list1.append(mae_value1)
                mae = torch.abs(y_prob0[mask] - batch.y_prob0[mask]).mean()
                mae_value0 = mae.item()
                y_prob_list0.append(mae_value0)
                y_prob_pre1 = torch.abs(y_prob1[mask]).mean().item()
                y_prob_pre0 = torch.abs(y_prob0[mask]).mean().item()
                y_true = torch.abs(batch.y_prob1[mask]).mean().item()
                y_prob_pred1.append(y_prob_pre1)
                y_prob_pred0.append(y_prob_pre0)
                y_prob_true.append(y_true)
                
                gt_flat = batch.y_prob1[mask].flatten()
                probs_flat = y_prob1[mask].flatten()
                gt_mean = torch.mean(gt_flat)
                ss_res = torch.sum((gt_flat - probs_flat) ** 2)
                ss_tot = torch.sum((gt_flat - gt_mean) ** 2)+1e-8
                # 计算 R2
                r2 = 1 - (ss_res / ss_tot)
                # r = torch.corrcoef(torch.stack([gt_flat, probs_flat]))[0, 1]
                # r2 = r ** 2
                r2_list_prob.append(r2)

                gt_flat = batch.y_prob0[mask].flatten()
                probs_flat = y_prob0[mask].flatten()
                gt_mean = torch.mean(gt_flat)
                ss_res = torch.sum((gt_flat - probs_flat) ** 2)
                ss_tot = torch.sum((gt_flat - gt_mean) ** 2)+1e-8
                # 计算 R2
                r2 = 1 - (ss_res / ss_tot)
                
                r2_list_trans.append(r2)
            if local_rank == 0:
                Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                    epoch, iter_id, num_iters, phase=phase,
                    total=bar.elapsed_td, eta=bar.eta_td)
                for l in avg_loss_stats:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch.num_graphs * len(hf))
                    Bar.suffix = Bar.suffix + \
                        '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

                if not args.hide_data_time:
                    Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                        '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
                    
                

                if args.print_iter > 0:
                    if iter_id % args.print_iter == 0:
                        print('{}/{}| {}'.format(args.task, args.exp_id, Bar.suffix))
                else:
                    bar.next()
  
            del hs, hf, loss, loss_stats

        if phase == 'val' and args.val_only:
            torch.save(result_dict, "/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/exp/stage1_embedding.pt")
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        if local_rank == 0:
            bar.finish()
            ret['time'] = bar.elapsed_td.total_seconds() / 60.
            if phase == 'val':

                #ret['trans mae'] = np.average(trans_list)
                ret['logic-1 mae'] = np.average(y_prob_list1)
                ret['logic-0 mae'] = np.average(y_prob_list0)
                ret['y_prob_pred1'] = np.average(y_prob_pred1, axis=0)
                # ret['y_prob_pred0'] = np.average(y_prob_pred0, axis=0)
                ret['y_prob_true'] = np.average(y_prob_true, axis=0)
                ret['R2_logic1'] = torch.nanmean(torch.stack(r2_list_prob)).item()
                ret['R2_logic0'] = torch.nanmean(torch.stack(r2_list_trans)).item()

        return ret, results
    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, reg_loss, cls_loss):
        if reg_loss in _loss_factory.keys():
            reg_loss_func = _loss_factory[reg_loss]()
        if cls_loss in _loss_factory.keys():
            cls_loss_func = _loss_factory[cls_loss]()
        loss_states = ['loss', 'LProb1', 'LProb0']
        return loss_states, reg_loss_func, cls_loss_func

    def val(self, epoch, data_loader, local_rank):
        return self.run_epoch('val', epoch, data_loader, local_rank)

    def train(self, epoch, data_loader, local_rank):
        return self.run_epoch('train', epoch, data_loader, local_rank)

class MLPGNNTrainer2(object):
    def __init__(
            self, args, model, optimizer=None):
        self.args = args
        self.optimizer = optimizer
        self.loss_stats, self.reg_loss, self.cls_loss = self._get_losses(args.reg_loss, args.cls_loss)
        self.reg_loss = self.reg_loss.to(self.args.device)
        self.cls_loss = self.cls_loss.to(self.args.device)
        self.model_with_loss = ModelWithLoss2(model, self.reg_loss, self.cls_loss, args.gpus, args.device)
    
    def set_weight(self, w_prob, w_rc, w_func, w_trans, w_seq):
        self.args.prob_weight = w_prob
        self.args.rc_weight = w_rc
        self.args.func_weight = w_func
        self.args.trans_weight = w_trans
        self.args.seq_weight = w_seq
        
    def set_device(self, device, local_rank, gpus):
        if len(gpus)> 1:
            self.model_with_loss = self.model_with_loss.to(device)
            self.model_with_loss = nn.parallel.DistributedDataParallel(self.model_with_loss,
                                                                       device_ids=[local_rank], 
                                                                       find_unused_parameters=True)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, dataset, local_rank, finite_list_full, trans_matrics):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.args.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        args = self.args
        f1_list = []
        precision_list = []
        recall_list = []
        results = []
        groud_truth = []
        accuracy_list = []
        mae_list = []
        mape_list = []
        r2_list = []
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(dataset) if args.num_iters < 0 else args.num_iters
        if local_rank == 0:
            bar = Bar('{}/{}'.format(args.task, args.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(dataset):
            if iter_id >= num_iters:
                break
            if len(self.args.gpus) == 1:
                batch = batch.to(self.args.device)
            data_time.update(time.time() - end)
            pred, loss_stats = model_with_loss(batch,self.args.stage, finite_list_full, trans_matrics)
            loss_reg = loss_stats['reg_loss']
            loss_cls = loss_stats['cls_loss']
            w_reg = 0.2
            w_cls = 0.8
            loss = w_reg * loss_reg + w_cls * loss_cls      
            #loss = loss_stats['cls_loss']*0.5 + loss_stats['reg_loss']*0.5
            loss = loss.mean()
 
            loss_stats['loss'] = loss
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model_with_loss.parameters(), args.grad_clip)
                self.optimizer.step()
            elif phase == 'val':
                    if epoch == 0:
                        results.extend(pred['pred_prob'].detach().cpu().flatten().tolist())
                        groud_truth.extend(pred['ground_true'].detach().cpu().flatten().tolist())
                        # for prob in pred['pred_prob'].detach().cpu().flatten():  # 展平成一维
                        #     results.append(prob.item())
                        # for prob in pred['ground_true'].detach().cpu().flatten():  # 展平成一维
                        #     groud_truth.append(prob.item())
                        # pred_np = np.array(results) > 0
                        # gt_np = np.array(groud_truth)
                        # # True Positives, False Positives, False Negatives (逐行)
                        # TP = np.sum((gt_np == 1) & (pred_np == 1)) #axis = 1
                        # FP = np.sum((gt_np == 0) & (pred_np == 1))
                        # FN = np.sum((gt_np == 1) & (pred_np == 0))
                        # TN = np.sum((gt_np == 0) & (pred_np == 0))
                        # accuracy_per_row = np.divide(TP + TN,
                        #             TP + TN + FP + FN,
                        #             out=np.zeros_like(TP, dtype=float),
                        #             where=(TP + TN + FP + FN) != 0)

                        # accuracy = accuracy_per_row.mean()
                        # # 避免除零
                        # precision_per_row = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
                        # recall_per_row    = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
                        # f1_per_row        = np.divide(2 * precision_per_row * recall_per_row,
                        #                             precision_per_row + recall_per_row,
                        #                             out=np.zeros_like(precision_per_row, dtype=float),
                        #                             where=(precision_per_row + recall_per_row) != 0)

                        # precision = precision_per_row.mean()
                        # recall    = recall_per_row.mean()
                        # f1        = f1_per_row.mean()
            batch_time.update(time.time() - end)
            end = time.time()
            if phase == 'val':
                logits = pred['pred_label']
                labels = pred['ground_true_label']
                probs = pred['pred_prob']
                gt = pred['ground_true']
                labels_np = labels.detach().cpu().numpy().astype(int)         # [S, S]
                logits_np = (logits.detach().cpu().numpy() > 0.1).astype(int)

                # True Positives, False Positives, False Negatives (逐行)
                TP = np.sum((labels_np == 1) & (logits_np == 1)) #axis = 1
                FP = np.sum((labels_np == 0) & (logits_np == 1))
                FN = np.sum((labels_np == 1) & (logits_np == 0))
                TN = np.sum((labels_np == 0) & (logits_np == 0))
                accuracy_per_row = np.divide(TP + TN,
                                TP + TN + FP + FN,
                                out=np.zeros_like(TP, dtype=float),
                                where=(TP + TN + FP + FN) != 0)

                accuracy = accuracy_per_row.mean()
                # 避免除零
                precision_per_row = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
                recall_per_row    = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
                f1_per_row        = np.divide(2 * precision_per_row * recall_per_row,
                                                precision_per_row + recall_per_row,
                                                out=np.zeros_like(precision_per_row, dtype=float),
                                                where=(precision_per_row + recall_per_row) != 0)

                precision = precision_per_row.mean()
                recall    = recall_per_row.mean()
                f1        = f1_per_row.mean()
                mask = gt > 0.001
                if mask.sum() > 0:
                    mape = (torch.abs((gt[mask] - probs[mask]) / gt[mask])).mean().item() * 100
                else:
                    mape = float(0)  
                mae = torch.abs(gt - probs).mean().item()
                gt_flat = gt.flatten()
                probs_flat = probs.flatten()
                r = torch.corrcoef(torch.stack([gt_flat, probs_flat]))[0, 1]
                r2 = r ** 2

                # f1 = f1_score(gt.cpu().numpy(), probs.cpu().numpy() > 0.5, average='binary', zero_division=0)
                # precision = precision_score(gt.cpu().numpy(), probs.cpu().numpy() > 0.5, average='binary', zero_division=0)
                # recall = recall_score(gt.cpu().numpy(), probs.cpu().numpy() > 0.5, average='binary', zero_division=1)
                f1_list.append(f1)
                precision_list.append(precision)    
                recall_list.append(recall)
                accuracy_list.append(accuracy)
                mape_list.append(mape)
                mae_list.append(mae)
                r2_list.append(r2)
                    
            if local_rank == 0:
                Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                    epoch, iter_id, num_iters, phase=phase,
                    total=bar.elapsed_td, eta=bar.eta_td)
                for l in avg_loss_stats:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch.num_graphs * len(pred))
                    Bar.suffix = Bar.suffix + \
                        '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

                if not args.hide_data_time:
                    Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                        '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
                    
                

                if args.print_iter > 0:
                    if iter_id % args.print_iter == 0:
                        print('{}/{}| {}'.format(args.task, args.exp_id, Bar.suffix))
                else:
                    bar.next()
  
            del pred, loss, loss_stats

        if phase == 'val' and args.val_only:
            torch.save(results, "/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/exp/pred.pt")
            torch.save(groud_truth, "/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/exp/groud_truth.pt")
            print("success!")
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        if local_rank == 0:
            bar.finish()
            ret['time'] = bar.elapsed_td.total_seconds() / 60.
            if phase == 'val':
                ret['F1'] = np.average(f1_list)
                ret['Precision'] = np.average(precision_list)
                ret['Recall'] = np.average(recall_list)
                ret['Acc'] = np.average(accuracy_list)
                ret['MAE'] = np.average(mae_list)
                ret['MAPE'] = np.average(mape_list)
                ret['R2'] = torch.nanmean(torch.stack(r2_list)).item()
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, reg_loss, cls_loss):
        if reg_loss in _loss_factory.keys():
            reg_loss_func = _loss_factory[reg_loss]()
        if cls_loss in _loss_factory.keys():
            cls_loss_func = _loss_factory[cls_loss]()
        loss_states = ['cls_loss','reg_loss', 'loss']
        return loss_states, reg_loss_func, cls_loss_func

    def val(self, epoch, data_loader, local_rank, finite_list_full, trans_matrics):
        return self.run_epoch('val', epoch, data_loader, local_rank, finite_list_full, trans_matrics)

    def train(self, epoch, data_loader, local_rank, finite_list_full, trans_matrics):
        return self.run_epoch('train', epoch, data_loader, local_rank, finite_list_full, trans_matrics)
    
class MLPGNNTrainer3(object):
    def __init__(
            self, args, model, optimizer=None):
        self.args = args
        self.optimizer = optimizer
        self.loss_stats, self.reg_loss, self.cls_loss = self._get_losses(args.reg_loss, args.cls_loss)
        self.reg_loss = self.reg_loss.to(self.args.device)
        self.cls_loss = self.cls_loss.to(self.args.device)
        self.model_with_loss = ModelWithLoss3(model, self.reg_loss, self.cls_loss, args.gpus, args.device)
    
    def set_weight(self, w_prob, w_rc, w_func, w_trans, w_seq):
        self.args.prob_weight = w_prob
        self.args.rc_weight = w_rc
        self.args.func_weight = w_func
        self.args.trans_weight = w_trans
        self.args.seq_weight = w_seq
        
    def set_device(self, device, local_rank, gpus):
        if len(gpus)> 1:
            self.model_with_loss = self.model_with_loss.to(device)
            self.model_with_loss = nn.parallel.DistributedDataParallel(self.model_with_loss,
                                                                       device_ids=[local_rank], 
                                                                       find_unused_parameters=True)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, dataset, local_rank, finite_list_full, trans_matrics):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.args.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        args = self.args
        f1_list = []
        precision_list = []
        recall_list = []
        results = []
        groud_truth = []
        y_prob_list = []
        mape_list = []
        r2_list = []
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(dataset) if args.num_iters < 0 else args.num_iters
        if local_rank == 0:
            bar = Bar('{}/{}'.format(args.task, args.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(dataset):
            if iter_id >= num_iters:
                break
            if len(self.args.gpus) == 1:
                batch = batch.to(self.args.device)
            data_time.update(time.time() - end)
            pred, loss_stats = model_with_loss(batch,self.args.stage, finite_list_full, trans_matrics)
                   
            loss = loss_stats['cls_loss']*0.25 + loss_stats['reg_loss']*0.75
            loss = loss.mean()
 
            loss_stats['loss'] = loss
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model_with_loss.parameters(), args.grad_clip)
                self.optimizer.step()
            elif phase == 'val':
                    if epoch == 0:
                        for prob in pred['pred_prob'].detach().cpu().flatten():  # 展平成一维
                            results.append(prob.item())
                        for prob in pred['ground_true'].detach().cpu().flatten():  # 展平成一维
                            groud_truth.append(prob.item())
            batch_time.update(time.time() - end)
            end = time.time()
            if phase == 'val':
                probs = pred['pred_label']
                gt = pred['ground_true_label']
                gt_np = gt.cpu().numpy().astype(int)         # [S, S]
                pred_np = (probs.cpu().numpy() > 0.5).astype(int)

                # True Positives, False Positives, False Negatives (逐行)
                TP = np.sum((gt_np == 1) & (pred_np == 1)) #axis = 1
                FP = np.sum((gt_np == 0) & (pred_np == 1))
                FN = np.sum((gt_np == 1) & (pred_np == 0))

                # 避免除零
                precision_per_row = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
                recall_per_row    = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
                f1_per_row        = np.divide(2 * precision_per_row * recall_per_row,
                                            precision_per_row + recall_per_row,
                                            out=np.zeros_like(precision_per_row, dtype=float),
                                            where=(precision_per_row + recall_per_row) != 0)

                precision = precision_per_row.mean()
                recall    = recall_per_row.mean()
                f1        = f1_per_row.mean()
                # f1 = f1_score(gt.cpu().numpy(), probs.cpu().numpy() > 0.5, average='binary', zero_division=0)
                # precision = precision_score(gt.cpu().numpy(), probs.cpu().numpy() > 0.5, average='binary', zero_division=0)
                # recall = recall_score(gt.cpu().numpy(), probs.cpu().numpy() > 0.5, average='binary', zero_division=1)
                f1_list.append(f1)
                precision_list.append(precision)    
                recall_list.append(recall)
                probs = pred['pred_label']
                gt = pred['ground_true_label']
                mae = torch.abs(probs - gt).mean()
                mae_value = mae.item()
                y_prob_list.append(mae_value)
                eps = 1e-3
                mask = gt.abs() > eps
                if mask.any():
                    mape = (torch.abs(probs - gt)[mask] / gt.abs()[mask]).mean().item()
                else:
                    mape = 0
                mape_list.append(mape)
                y_pred = probs.view(-1)
                y_true = gt.view(-1)
                y_mean = y_true.mean()

                gt_flat = gt.flatten()
                probs_flat = probs.flatten()
                r = torch.corrcoef(torch.stack([gt_flat, probs_flat]))[0, 1]
                r2 = r ** 2
                r2_list.append(r2)
            if local_rank == 0:
                Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                    epoch, iter_id, num_iters, phase=phase,
                    total=bar.elapsed_td, eta=bar.eta_td)
                for l in avg_loss_stats:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch.num_graphs * len(pred))
                    Bar.suffix = Bar.suffix + \
                        '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

                if not args.hide_data_time:
                    Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                        '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
                    
                

                if args.print_iter > 0:
                    if iter_id % args.print_iter == 0:
                        print('{}/{}| {}'.format(args.task, args.exp_id, Bar.suffix))
                else:
                    bar.next()
  
            del pred, loss, loss_stats

        if phase == 'val' and args.val_only:
            torch.save(results, "/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/exp/pred.pt")
            torch.save(groud_truth, "/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/exp/groud_truth.pt")
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        if local_rank == 0:
            bar.finish()
            ret['time'] = bar.elapsed_td.total_seconds() / 60.
            if phase == 'val':
                ret['F1'] = np.average(f1_list)
                ret['Precision'] = np.average(precision_list)
                ret['Recall'] = np.average(recall_list)
                ret['MAE'] = np.average(y_prob_list)
                ret['MAPE'] = np.average(mape_list)
                ret['R2'] = torch.nanmean(torch.stack(r2_list)).item()
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, reg_loss, cls_loss):
        if reg_loss in _loss_factory.keys():
            reg_loss_func = _loss_factory[reg_loss]()
        if cls_loss in _loss_factory.keys():
            cls_loss_func = _loss_factory[cls_loss]()
        loss_states = ['cls_loss','reg_loss', 'loss']
        return loss_states, reg_loss_func, cls_loss_func

    def val(self, epoch, data_loader, local_rank, finite_list_full, trans_matrics):
        return self.run_epoch('val', epoch, data_loader, local_rank, finite_list_full, trans_matrics)

    def train(self, epoch, data_loader, local_rank, finite_list_full, trans_matrics):
        return self.run_epoch('train', epoch, data_loader, local_rank, finite_list_full, trans_matrics)