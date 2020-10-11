import os
import torch
import random
import logging
import numpy as np
import argparse
from pathlib import Path
from dataset import split_dataset, Probe_Dataset
from torch.utils.data import DataLoader
from initialization import initialization
from learning import train, validate

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='Vnet', help='model architecture')
    parser.add_argument('--data_mode', type=str, default='2.5D', help='data mode')
    parser.add_argument('--dataset_mode', type=str, default='main_branch', help='dataset mode be to used')
    parser.add_argument('--slices', type=int, default=7, help='slices used in the 2.5D mode')
    parser.add_argument('--n_classes', type=int, default=4, help='classes for segmentation')
    parser.add_argument('--seed', type=int, default=123, help='set seed point')
    parser.add_argument('--crop_size', type=int, default=96, help='size for square patch')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch Size during training [default: 256]')
    parser.add_argument('--epoch', default=300, type=int, help='Epoch to run [default: 300]')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--clip', type=float, default=0.4, help='gradient clip, (default: 0.4)')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--loss_func', type=str, default='dice', help='Loss function used for training [default: dice]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--step_size', type=int, default=50, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--k_fold', default=0, type=int, help='k-fold cross validation')
    parser.add_argument('--train_num', default=0.8, type=int, help='folder name for training set')
    parser.add_argument('--val_num', default=0.2, type=int, help='folder name for validation set')
    parser.add_argument('--data_dir', default='/Users/gaoyibo/plaques/all_subset', help='folder name for training set')

    # do not change following flags
    parser.add_argument('--n_weights', type=int, default=None, help='Weights for classes of segmentation or classification')
    parser.add_argument('--experiment_dir', type=str, default=None, help='Experiment path [default: None]')
    parser.add_argument('--checkpoints_dir', type=str, default=None, help='Experiment path [default: None]')
    parser.add_argument('--logger', default=None, help='logger')
    parser.add_argument('--log_string', type=str, default=None, help='log string wrapper [default: None]')
    parser.add_argument('--cur_fold', type=int, default=None, help='log string wrapper [default: None]')
    parser.add_argument('--device', type=str, default=None, help='set device type')
    return parser.parse_args()

def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

def make_dir_log(args):
    # setup experimental logs dir ---------------------------------------
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)

    if args.log_dir is None:
        args.log_dir = 'data_' + args.dataset_mode + '-' + args.model + '-' + args.data_mode + '-' + str(args.n_classes) + '-' + str(args.slices) + '-' + args.loss_func
        if args.k_fold > 1:
            args.log_dir = 'data_' + args.data_mode + '-fold_' + str(args.cur_fold)
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)

    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    args.experiment_dir = experiment_dir
    args.checkpoints_dir = checkpoints_dir
    args.log_dir = log_dir

    # set logs format, file writing and level -----------------------------------
    def log_string(str):
        logger.info(str)
        print(str)
    args.log_string = log_string

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER...')
    log_string(args)
    args.logger = logger

def main(args):
    # set device used -----------------------------------------------
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.log_string('Device using: %s' % args.device)

    # prepare dataset --------------------------------------------
    train_dir, val_dir, _ = split_dataset(args, cur_loop)
    train_dataset = Probe_Dataset(train_dir, args, augmentation=True)
    val_dataset = Probe_Dataset(val_dir, args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    args.log_string("The number of training data is %d" % len(train_dataset))
    args.log_string("The number of validation data is %d" % len(val_dataset))
    args.n_weights = torch.tensor(train_dataset.labelweights).float().to(args.device)
    args.log_string("Weights for classes:{}".format(args.n_weights))

    # initialization -----------------------------------------------------
    model, optimizer, criterion, start_epoch = initialization(args)

    global_epoch = 0
    best_loss = 0
    best_epoch = 0
    best_metric = None
    LEARNING_RATE_CLIP = 1e-5

    for epoch in range(start_epoch, args.epoch):
        args.log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))

        # adjust hyper parameters ---------------------------------------------------------
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        args.log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # train --------------------------------------------------------------
        train(args, global_epoch, train_loader, model, optimizer, criterion)

        if epoch % 5 == 0:
            savepath = str(args.checkpoints_dir) + '/model.pth'
            args.log_string('Saving at %s' %savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            args.log_string('Saving model...')

        # validate ------------------------------------------------------------
        if epoch % 2 == 0:
            val_result = validate(args, global_epoch, val_loader, model, optimizer, criterion)

            if val_result[0] > best_loss:
                best_loss = val_result[0]
                best_metric = val_result[1]
                best_epoch = epoch

                savepath = str(args.checkpoints_dir) + '/best_model.pth'
                args.log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                args.log_string('Saving model...')
            args.log_string('Best Epoch, Loss and Result: %f, %f, %s' %(best_epoch, best_loss, best_metric))

        global_epoch += 1

    return best_metric


if __name__ == "__main__":
    args = parse_args()

    result_list = []

    if args.k_fold > 1:
        loop_time = args.k_fold
    else:
        loop_time = 1

    for cur_loop in range(loop_time):
        args = parse_args()
        set_seed(args)

        args.cur_fold = cur_loop
        make_dir_log(args)
        cur_result = main(args)
        result_list.append(cur_result)

        if cur_loop < loop_time - 1:
            handlers = args.logger.handlers[:]
            for handler in handlers:
                handler.close()
                args.logger.removeHandler(handler)

    args.log_string('Final result -----------------------------')
    for idx in range(len(result_list)):
        args.log_string('Fold {}:{}\n'.format(idx, result_list[idx]))
    final_result = np.zeros((4, 1))
    for item in result_list:
        for idx, value in enumerate(item):
            final_result /= args.k_fold
    final_result /= args.k_fold

    args.log_string('Average result for all folds: {}'.format(list(final_result)))
    args.log_string('Average result for all: {}'.format(np.mean(list(final_result[1:]))))