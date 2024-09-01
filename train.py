import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from datetime import datetime
import os
import json


def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

def delete_checkpoints(path, training_epochs):
    import shutil
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isdir(file_path) and int(filename) < (training_epochs - 500):
            shutil.rmtree(file_path)
            print(f"Deleted {file_path}")

if __name__ == '__main__':
    opt = TrainOptions().parse()
    # config = get_config(opt.config)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)

    if opt.continue_train:
        if opt.continue_epoch > opt.niter:
            scalar = opt.continue_epoch - opt.niter
            opt.lr = opt.lr - scalar * (opt.lr / opt.niter_decay)
        init = opt.continue_epoch + 1
        model.load_network(model.MEF, 'MEF', opt.continue_path)
        model.MEF.train()
        save_path = os.path.dirname(opt.continue_path)
        print(f'save_path: {save_path}')
    else:
        init = 1
        current_time = datetime.now().strftime('%m-%d-%H-%M')
        save_path = os.path.join('./checkpoints', opt.git_tag, current_time)
        os.makedirs(save_path, exist_ok=True)
        print(f'save_path: {save_path}')


        with open(save_path + '/info.json', 'w') as f:
            json.dump(opt.__dict__, f, indent=2)

    total_steps = 0


    for epoch in range(init, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        for i, data in enumerate(dataset): # data is a dictionary = {'A': A_img, 'B': B_img, 'C': C_img, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'C_gray': C_d}
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter = total_steps - dataset_size * (epoch - 1)
            model.set_input(data)
            model.optimize_parameters(epoch)

        error = model.get_current_errors(epoch)
        t = (time.time() - iter_start_time) / opt.batchSize

            # if total_steps % opt.save_latest_freq == 0:
            #     print('saving the latest model (epoch %d, total_steps %d)' %
            #           (epoch, total_steps))
            #     model.save('latest')

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        print(f'Error: {error}')

        if epoch % opt.save_epoch_freq == 0:
            model.write_loss(save_path, epoch, error)
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save(epoch, save_path)

        if epoch > opt.niter:
            model.update_learning_rate()

    delete_checkpoints(save_path, opt.niter + opt.niter_decay)
