from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--save_root', type=str, default='./output')
        self.parser.add_argument('--checkpoint_path', type=str, default='none')
        self.isTrain = False