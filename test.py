import numpy
import torch
from numpy import double
import util.util as util
from options.train_options import TrainOptions
from models.models import create_model
from data.data_loader import *
from PIL import Image
import cv2
import os
from datetime import datetime

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb: 128'

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

def fuse(opt, model, imgA_path, imgB_path, imgM_path, img_fused, scale):
    imgA = cv2.imread(imgA_path)
    imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
    imgB = cv2.imread(imgB_path)
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
    imgA = double(imgA) / 255
    imgB = double(imgB) / 255

    # medium exposure
    imgM = cv2.imread(imgM_path)
    imgM = cv2.cvtColor(imgM, cv2.COLOR_BGR2RGB)
    imgM = double(imgM) / 255


    # original size (4k*3k) of ASUS data would be out of memory
    h = imgA.shape[0]
    w = imgA.shape[1]
    if w > h:
        imgA = cv2.resize(imgA, (592 * scale, 400 * scale), interpolation=cv2.INTER_AREA)
        imgB = cv2.resize(imgB, (592 * scale, 400 * scale), interpolation=cv2.INTER_AREA)
        imgM = cv2.resize(imgM, (592 * scale, 400 * scale), interpolation=cv2.INTER_AREA)

    else:
        imgA = cv2.resize(imgA, (400 * scale, 592 * scale), interpolation=cv2.INTER_AREA)
        imgB = cv2.resize(imgB, (400 * scale, 592 * scale), interpolation=cv2.INTER_AREA)
        imgM = cv2.resize(imgM, (400 * scale, 592 * scale), interpolation=cv2.INTER_AREA)

    imgA = torch.from_numpy(imgA)
    imgB = torch.from_numpy(imgB)

    imgA = imgA.unsqueeze(0)
    imgB = imgB.unsqueeze(0)

    imgA = imgA.permute(0, 3, 2, 1).float()
    imgB = imgB.permute(0, 3, 2, 1).float()

    imgA = imgA.cuda().half()
    imgB = imgB.cuda().half()

    # medium exposure
    imgM = torch.from_numpy(imgM)
    imgM = imgM.unsqueeze(0)
    imgM = imgM.permute(0, 3, 2, 1).float()
    imgM = imgM.cuda().half()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with torch.no_grad():
        output4 = model.MEF.forward(imgA, imgM, imgB)
        end_event.record()
        torch.cuda.synchronize()
        print(f'Time taken: {start_event.elapsed_time(end_event)}')
        output4 = util.tensor2im_2(output4.detach())
        output4 = cv2.resize(output4, (w, h), interpolation=cv2.INTER_CUBIC)
        outputimage3 = Image.fromarray(numpy.uint8(output4))
        outputimage3.save(img_fused)

def testphotos(u_path, m_path, o_path, save_path):
    model = create_model(opt)
    scale = 6
    weight_path = r'./checkpoints/uo_v5/04-25-23-04/1500'
    model.load_network(model.MEF, 'MEF', weight_path)
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    model.MEF.eval().half()

    img_names = sorted(os.listdir(u_path))

    for i in range(len(img_names)):
        name = img_names[i]
        u = os.path.join(u_path, name)
        o = os.path.join(o_path, name)

        # medium exposure
        m = os.path.join(m_path, name)

        save = os.path.join(save_path, name)
        fuse(opt, model, u, o, m, save, scale)
        print(save)

    info_dict = {'Checkpoint': weight_path,
                 'Scale': scale}
    with open(save_path + '/info.txt', 'w') as f:
        for key, value in info_dict.items():
            f.write(f'{key}: {value}\n')

if __name__ == '__main__':
    opt = TrainOptions().parse()
    asus_path = [os.path.join('..', 'datasets/asus_test/under'), os.path.join('..', 'datasets/asus_test/medium'), os.path.join('..', 'datasets/asus_test/over')]
    dmef_path = [os.path.join('..', 'datasets/DMEF/Test/test_1-1'), os.path.join('..', 'datasets/DMEF/Test/test_4-2'), os.path.join('..', 'datasets/DMEF/Test/test_7-1')]
    inference_path = {'dmef': dmef_path, 'asus': asus_path}
    current_time = datetime.now().strftime('%m%d-%H%M%S')
    save_root = r'./output'
    save_dir = os.path.join(save_root, opt.git_tag, current_time)
    for key, value in inference_path.items():
        u_path, m_path, o_path = value
        save_path = os.path.join(save_dir, key)
        print(save_path)
        testphotos(u_path, m_path, o_path, save_path)

