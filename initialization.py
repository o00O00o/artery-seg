import os
import importlib
import shutil
import torch
from losses import CrossEntropy, DiceLossMulticlass_CW
from tensorboardX import SummaryWriter

def initialization(args):
    MODEL = importlib.import_module(args.model)
    shutil.copy('%s.py' % args.model, str(args.experiment_dir))

    # decide the input channel of the network according to the data_mode
    if args.data_mode == '2D':
        initial_channel = 1
    elif args.data_mode == '2.5D':
        initial_channel = args.slices
    else:
        raise NotImplementedError

    model = MODEL.get_module(initial_channel, args.n_classes, 4, 4, True, True).to(args.device)
    ema_model = MODEL.get_module(initial_channel, args.n_classes, 4, 4, True, True).to(args.device)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data)
    print(str(args.checkpoints_dir) + '/best_model.pth')

    try:
        checkpoint = torch.load(str(args.checkpoints_dir) + '/model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        if not args.baseline:
            ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        args.log_string('Use pretrain model')
    except:
        args.log_string('No existing model, starting training from scratch...')
        model = model.apply(weights_init)
        ema_model = ema_model.apply(weights_init)
        start_epoch = 0

    # optimizer initialization -----------------------------------------
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    # loss initialization ---------------------------------------------
    if args.loss_func == 'dice':
        criterion = DiceLossMulticlass_CW()
    elif args.loss_func == 'cross_entropy':
        criterion = CrossEntropy()
    else:
        print('unknown loss function:{}'.format(args.loss_func))

    # writer initializtion ---------------------------------------------
    writer = SummaryWriter(os.path.join(args.log_dir, 'runs'))

    return model, ema_model, optimizer, criterion, start_epoch, writer
