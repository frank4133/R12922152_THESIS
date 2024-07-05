import os
import torch
from datetime import datetime


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass


    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        from datetime import datetime

# ...

    def save_network(self, network, network_label, epoch_label, gpu_ids, save_path):
        save_folder = os.path.join(save_path, str(epoch_label))
        os.makedirs(save_folder, exist_ok=True)
        save_filename = network_label + '.pth'
        save_path = os.path.join(save_folder, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        print('saved net: %s' % save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, weight_path):
        save_filename = '%s.pth' % network_label
        save_path = os.path.join(weight_path, save_filename)
        print('loading the model from %s' % save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate():
        pass
