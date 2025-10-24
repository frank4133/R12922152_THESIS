import time
from options.train_options import TrainOptions
from data.__init__ import create_dataloader
from models.__init__ import create_model
from datetime import datetime
import os
import json
from tqdm import tqdm
import math


if __name__ == '__main__':
    opt = TrainOptions().parse()
    # config = get_config(opt.config)
    data_loader = create_dataloader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    current_time = datetime.now().strftime('%m-%d-%H-%M')

    if opt.continue_train:
        model.continue_train()
        init = opt.continue_epoch + 1
    else:
        init = 1

    save_path = os.path.join('./checkpoints', opt.network, current_time)
    os.makedirs(save_path, exist_ok=True)
    print(f'save_path: {save_path}')
    with open(save_path + '/info.json', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    n_batches = math.ceil(len(dataset) / opt.batch_size)
    for epoch in range(init, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        for i, data in enumerate(tqdm(dataset, total=n_batches, desc=f"Epoch {epoch}", unit="batch")): # data is a dictionary = {'A': A_img, 'B': B_img, 'C': C_img, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'C_gray': C_d}
            model.set_input(data)
            model.optimize_parameters(epoch)

        loss = model.get_current_errors(epoch)
        print(f'Loss: {loss}')

        if epoch % opt.save_epoch_freq == 0:
            model.write_loss(save_path, epoch, loss)
            model.save_network(epoch, save_path)

        if epoch > opt.niter:
            model.update_learning_rate()
