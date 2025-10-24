import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            '--dataset_name', required=True, help='dataset name for choosing dataset.py')
        self.parser.add_argument('--under_exposure', type=str, default=None)
        self.parser.add_argument('--over_exposure', type=str, default=None)
        self.parser.add_argument(
            '--mode', required=True, help='Train or Test')
        self.parser.add_argument(
            '--warp', type=int, default=0, help='whether to warp the input image')
        self.parser.add_argument(
            '--evidence', type=int, default=0, help='whether to use evidence')
        self.parser.add_argument(
            '--network', required=True, help='choose the network to create')
        self.parser.add_argument(
            '--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument(
            '--CT_patch_ratio', type=int, default=16, help='for CT module')
        self.parser.add_argument(
            '--CT_num_heads', type=int, default=4, help='for CT module')
        self.parser.add_argument(
            '--CT_embedding_dropout', type=float, default=0.1, help='for CT module')
        self.parser.add_argument(
            '--CT_attention_dropout', type=float, default=0.1, help='for CT module')
        self.parser.add_argument(
            '--CT_transformer_dropout', type=float, default=0.0, help='for CT module')
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument(
            '--nThreads', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--drop_last', type=bool, default=False)
        self.parser.add_argument('--evidence_normalization', type=str, default='linear', help='linear or sigmoid')
    
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            print('gpu_ids:', self.opt.gpu_ids)
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        return self.opt
