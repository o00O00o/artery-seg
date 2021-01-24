import torch
import random
import logging
import numpy as np
import argparse
from pathlib import Path
from dataset import split_dataset, Probe_Dataset, count_dataset, record_dataset
from torch.utils.data import DataLoader, ConcatDataset
from initialization import initialization
from learning import validate, train_mean_teacher, train
from over_sample import AugmentDataset


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='Vnet', help='model architecture: Vnet, cosnet')
    parser.add_argument('--data_mode', type=str, default='2D', help='data mode')
    parser.add_argument('--dataset_mode', type=str, default='all_branch', help='dataset mode be to used: main_branch or all_branch')
    parser.add_argument('--slices', type=int, default=7, help='slices used in the 2.5D mode')
    parser.add_argument('--n_classes', type=int, default=4, help='classes for segmentation')
    parser.add_argument('--seed', type=int, default=4, help='set seed point')
    parser.add_argument('--crop_size', type=int, default=64, help='size for square patch')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 256]')
    parser.add_argument('--epoch', default=800, type=int, help='Epoch to run [default: 300]')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--lr_clip', type=float, default=1e-5, help='learning rate clip')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--loss_func', type=str, default='dice', help='Loss function used for training [default: dice]')
    # parser.add_argument('--data_dir', default='/mnt/lustre/wanghuan3/gaoyibo/all_subset_v3', help='folder name for training set')
    parser.add_argument('--data_dir', default='/Users/gaoyibo/Datasets/plaques/all_subset_v3', help='folder name for training set')
    parser.add_argument('--step_size', type=int, default=50, help='Decay step')

    # do not change following flags
    parser.add_argument('--n_weights', type=int, default=None, help='Weights for classes of segmentation or classification')
    parser.add_argument('--experiment_dir', type=str, default=None, help='Experiment path [default: None]')
    parser.add_argument('--checkpoints_dir', type=str, default=None, help='Experiment path [default: None]')
    parser.add_argument('--logger', default=None, help='logger')
    parser.add_argument('--log_string', type=str, default=None, help='log string wrapper [default: None]')
    parser.add_argument('--device', type=str, default=None, help='set device type')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')

    # mean-teacher configurations
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--consistency-type', type=str, default='mse', help='select the type of consistency criterion')
    parser.add_argument('--consistency', type=float, default=1.0)
    parser.add_argument('--consistency_rampup', type=float, default=600.0)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--all_label', action='store_true', help='full supervised configuration if set true')
    parser.add_argument('--case_num', type=int, default=150, help='the num of total case')
    parser.add_argument('--unlabeled_num', default=75, type=int, help='the num of unlabeded case')
    parser.add_argument('--labeled_num', default=50, type=int, help='the num of labeled case')
    parser.add_argument('--times', default=5, type=int)
    parser.add_argument('--aug_list_dir', default='./plaque_info.csv', type=str)
    parser.add_argument('--over_sample', action="store_true")
    
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
        args.log_dir = 'labeled-' + str(args.labeled_num) + '-unlabeled-' + str(args.unlabeled_num) + '-weight-' + str(args.consistency) + ('-basline' if args.baseline else '')
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
    log_string(args)
    args.logger = logger

def main(args):
    # set device used -----------------------------------------------
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print dataset information ------------------------------------
    # record_dataset(args)
    # count_dataset(args)
    
    # prepare dataset --------------------------------------------
    unlabeled_dir, labeled_dir, val_dir = split_dataset(args)

    unlabeled_set = Probe_Dataset(unlabeled_dir, args)
    labeled_set = Probe_Dataset(labeled_dir, args)
    val_set = Probe_Dataset(val_dir, args)

    args.n_weights = torch.tensor(labeled_set.labelweights).float().to(args.device)
    args.log_string("Weights for classes:{}".format(args.n_weights))

    if args.over_sample:
        unlabeled_set = ConcatDataset([AugmentDataset(args, 'unlabel'), unlabeled_set])
        labeled_set = ConcatDataset([AugmentDataset(args, 'label'), labeled_set])
        # labeled_set = AugmentDataset(args, 'label')

    try:
        labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    except:
        print("Empty unlabel_set")

    args.log_string("The number of unlabeled data is %d" % len(unlabeled_set))
    args.log_string("The number of labeled data is %d" % len(labeled_set))
    args.log_string("The number of validation data is %d" % len(val_set))

    # initialization -----------------------------------------------------
    model, ema_model, optimizer, criterion, start_epoch, writer = initialization(args)

    global_epoch = 0
    best_epoch = 0
    best_dice = 0

    for epoch in range(start_epoch, args.epoch):
        args.log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))

        # adjust hyper parameters ---------------------------------------------------------
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), args.lr_clip)
        writer.add_scalar('config/lr', lr, epoch)
        args.log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # train --------------------------------------------------------------
        if args.all_label:
            train_mean_teacher(args, global_epoch, labeled_loader, labeled_loader, model, ema_model, optimizer, criterion, writer)
            # train(args, global_epoch, labeled_loader, model, optimizer, criterion, writer)
        else:
            train_mean_teacher(args, global_epoch, labeled_loader, unlabeled_loader, model, ema_model, optimizer, criterion, writer)

        if epoch % 5 == 0:
            savepath = str(args.checkpoints_dir) + '/model.pth'
            args.log_string('Saving at %s' %savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }

            if not args.baseline:
                state['ema_model_state_dict'] = ema_model.state_dict()
            
            torch.save(state, savepath)

        # validate ------------------------------------------------------------
        val_result = validate(args, global_epoch, val_loader, model, optimizer, criterion, writer, is_ema=False)

        if not args.baseline:
            ema_val_result = validate(args, global_epoch, val_loader, ema_model, optimizer, criterion, writer, is_ema=True)
            if ema_val_result[0] > val_result[0]:
                val_result = ema_val_result
        
        args.log_string('Val class dice %s:' % (val_result[1]))
        args.log_string('Val mean dice %s:' % (val_result[0]))

        if val_result[0] > best_dice:
            best_dice = val_result[0]
            best_metric = val_result[1]
            best_epoch = epoch

            savepath = str(args.checkpoints_dir) + '/best_model.pth'
            args.log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }

            if not args.baseline:
                state['ema_model_state_dict'] = ema_model.state_dict()
            
            torch.save(state, savepath)

        args.log_string('Best Epoch, Dice and Result: %d, %f, %s' %(best_epoch, best_dice, best_metric))
        
        global_epoch += 1

    return best_dice, best_metric


if __name__ == "__main__":

    args = parse_args()

    if args.all_label:
        args.labeled_num = args.labeled_num + args.unlabeled_num
        args.unlabeled_num = 0

    set_seed(args)
    make_dir_log(args)
    best_mean_dice, best_class_dice = main(args)

    handlers = args.logger.handlers[:]
    for handler in handlers:
        handler.close()
        args.logger.removeHandler(handler)

    args.log_string('Final result -----------------------------------------')
    args.log_string('Best mean dice: {}'.format(best_mean_dice))
    args.log_string('Best class dice: {}'.format(best_class_dice))
