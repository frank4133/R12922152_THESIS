import numpy
import torch
from numpy import double
import util.util as util
from options.test_options import TestOptions
from PIL import Image
import cv2
import os
from datetime import datetime
from data.__init__ import create_dataloader
from models.__init__ import create_model

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb: 128'

if __name__ == '__main__':
    opt = TestOptions().parse()
    test_loader = create_dataloader(opt)
    current_time = datetime.now().strftime('%m%d-%H%M%S')
    save_dir = os.path.join(opt.save_root, os.path.normpath(opt.checkpoint_path).split(os.sep)[-3], current_time)
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    model = create_model(opt)
    model.load_network(opt.checkpoint_path)
    model.network.eval().half()
    logger = util.MetricsLogger()
    filename = os.path.splitext(os.path.basename(opt.checkpoint_path))[0]
    dirname = os.path.basename(os.path.dirname(opt.checkpoint_path))
    log_dir = os.path.join(save_dir, dirname + '-' + filename + '.txt')
    total_time = 0
    with torch.no_grad():
        for image_dataset in test_loader:
            label_path = image_dataset['label_path'][0]
            filename = os.path.basename(label_path)
            base, ext = os.path.splitext(filename)
            model.set_input(image_dataset)
            print(f'Inferencing {label_path}')
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            if opt.evidence == 1:
                pred, evid = model.forward()
                util.save_alpha_map(evid, save_dir, prefix=base + '_evid')
            else:
                pred = model.forward()
            end_event.record()
            torch.cuda.synchronize()
            print(f'Time taken: {start_event.elapsed_time(end_event)}')
            total_time += start_event.elapsed_time(end_event)
            inputs = torch.stack([model.under, model.over], dim=0)  # (2, 1, 3, H, W) -> (2, 3, H, W)
            inputs = inputs.squeeze(1)  # 移除 batch 維度 -> (2, 3, H, W)
            logger.update(pred, model.label, inputs, log_dir, base)
            pred = util.tensor2im(pred.detach())
            pred = Image.fromarray(numpy.uint8(pred))
            under = util.tensor2im(image_dataset['under'])
            under = Image.fromarray(numpy.uint8(under))
            save_filename = f"{base}_pred{ext}"  # 例如：image1_pred.png
            save_path = os.path.join(save_dir, save_filename)
            pred.save(save_path)
            # save_filename = f"{base}_under_aligned{ext}"
            # save_path = os.path.join(save_dir, save_filename)
            # under.save(save_path)

    logger.save_summary(log_dir, total_time)