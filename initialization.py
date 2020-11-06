import importlib
import shutil
import torch
from losses import CrossEntropy, DiceLossMulticlass_CW

def initialization(args):
    MODEL = importlib.import_module(args.model)
    shutil.copy('%s.py' % args.model, str(args.experiment_dir))

    # decide the input channel of the network according to the data_mode
    if args.data_mode == '2D':
        initial_channel = 1
    elif args.data_mode == '2.5D':
        initial_channel = args.slices
    elif args.data_mode == 'image_pair':
        initial_channel = 1
    elif args.data_mode == '2.5D_pair':
        initial_channel = args.slices
    else:
        raise NotImplementedError

    # network initialization -------------------------------------------------
    if args.model == "cosnet":
        network = MODEL.get_module(args.n_classes)
    elif args.model == "Vnet":
        network = MODEL.get_module(initial_channel, args.n_classes, 4, 4, True, True).to(args.device)
    elif args.model == "cosunet":
        network = MODEL.get_module(initial_channel, args.n_classes)
    elif args.model == "cosunetd":
        network = MODEL.get_module(initial_channel, args.n_classes)

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
        checkpoint = torch.load(str(args.checkpoints_dir) + '/best_model.pth')
        start_epoch = checkpoint['epoch']
        network.load_state_dict(checkpoint['model_state_dict'])
        args.log_string('Use pretrain model')
    except:
        args.log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        # network = network.apply(weights_init)

    # optimizer initialization -----------------------------------------
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = torch.optim.SGD(network.parameters(), lr=args.learning_rate, momentum=0.9)

    # loss initialization ---------------------------------------------
    if args.loss_func == 'dice':
        criterion = DiceLossMulticlass_CW()
    elif args.loss_func == 'cross_entropy':
        criterion = CrossEntropy(topk_rate=args.topk_rate)
    elif args.loss_func == 'cross_entropy_and_dice':
        criterion_1 = CrossEntropy(topk_rate=args.topk_rate)
        criterion_2 = DiceLossMulticlass_CW()
        criterion = (criterion_1, criterion_2)
    else:
        print('unknown loss function:{}'.format(args.loss_func))

    return network, optimizer, criterion, start_epoch
