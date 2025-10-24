from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--main_loss_weight', type=float, default=1)
        self.parser.add_argument('--crop_size', type=int, default='360', help='then crop to this size')
        self.parser.add_argument('--main_loss_type', type=str, default='l1', help = 'l1, mse, charbonnier, etc')
        self.parser.add_argument('--vgg_weight',type=float, default=0.1)
        self.parser.add_argument('--evid_weight',type=float, default=0.1)
        self.parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', type=int, default=0,help='continue training: load the latest model')
        self.parser.add_argument('--continue_path', type=str, default=None, help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--continue_epoch',type=int,default=0,help='which epoch do you want to restore')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam')
        self.parser.add_argument('--ssim_loss_weight',type=float,default=0)
        self.isTrain = True
