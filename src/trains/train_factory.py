from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# from .recgnn import RecGNNTrainer
# from .convgnn import ConvGNNTrainer
from .base_trainer import BaseTrainer
from .mlpgnn_trainer import MLPGNNTrainer,MLPGNNTrainer2, MLPGNNTrainer3
from .mlpgnn_trainer_pe import MLPGNNTrainerPE

train_factory = {
  # 'recgnn': RecGNNTrainer,
  'recgnn': BaseTrainer,
  # 'convgnn': ConvGNNTrainer,
  'convgnn': BaseTrainer,
  'dagconvgnn': BaseTrainer,
  'base': BaseTrainer, 
  'mlpgnn': MLPGNNTrainer, 
  'mlpgnn_merge': MLPGNNTrainer, 
  'mlpgnn_dg2': MLPGNNTrainer,
  'mlpgnn_pe' : MLPGNNTrainerPE,
  'mlpgnn_stage2': MLPGNNTrainer2,
  'mlpgnn_ds2': MLPGNNTrainer3,
  'baseline':MLPGNNTrainer,
  'graphGPS':MLPGNNTrainer
}