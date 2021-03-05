import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from losses import softmax_mse_loss
from transformations import *
from utils import dice_coef
import torchvision.utils as vutils


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(args, epoch):
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)  # add_(other， alpha)为torch.add()的in-place版， 直接替换，加上other * alpha

def train(args, global_epoch, labeled_loader, unlabeled_loader, optimizer, criterion, writer, *models):

    stu_model = models[0]
    ema_model = models[1]
    stu_model.train()
    ema_model.train()

    if args.stage == 'fine' or args.stage == 'soft':
        coarse_model = models[2]
        coarse_model.eval()

    labeled_num_batches = len(labeled_loader)
    unlabeled_num_batches = len(unlabeled_loader)

    if not args.baseline:
        num_iteration_per_epoch = max(labeled_num_batches, unlabeled_num_batches)
    else:
        num_iteration_per_epoch = labeled_num_batches

    dice_tensor = np.zeros((num_iteration_per_epoch, args.n_classes))

    for batch_idx in tqdm(range(num_iteration_per_epoch)):

        try:
            data = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_loader)
            data = labeled_train_iter.next()
        
        inputs_x, targets_x = data['img'], data['mask']
        visual_image = vutils.make_grid(inputs_x[:, 3, :, :].unsqueeze(1), normalize=True, scale_each=True)
        visual_mask = vutils.make_grid(targets_x.unsqueeze(1), normalize=True, scale_each=True)

        inputs_x, targets_x = inputs_x.to(args.device).float(), targets_x.to(args.device)

        if args.stage == 'fine' or args.stage == 'soft':
            coarse_output = coarse_model(inputs_x)
            inputs_x = torch.cat((coarse_output, inputs_x), dim=1)

        if not args.baseline:
            try:
                data = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_loader)
                data = unlabeled_train_iter.next()
            
            inputs_stu = data['img']
            inputs_stu = inputs_stu.to(args.device).float()

            if args.stage == 'fine' or args.stage == 'soft':
                coarse_output = coarse_model(inputs_stu)
                inputs_stu = torch.cat((coarse_output, inputs_stu), dim=1)

            inputs_ema = torch.clone(inputs_stu)
            
            with torch.no_grad():
                # trans_inputs_u2 = transforms_for_noise(inputs_u2)  # noise transform
                trans_inputs_ema, rot_mask = transforms_for_rot(inputs_ema)  # rotation transform
                trans_inputs_ema, flip_mask = transforms_for_flip(trans_inputs_ema)  # flip transform
                trans_inputs_ema, scale_mask = transforms_for_scale(trans_inputs_ema)  # scale transform

                outputs_ema = ema_model(trans_inputs_ema)
                outputs_stu = stu_model(inputs_stu)

                trans_outputs_stu = transforms_back_scale(outputs_stu, scale_mask)
                trans_outputs_stu = transforms_back_flip(trans_outputs_stu, flip_mask)
                trans_outputs_stu = transforms_back_rot(trans_outputs_stu, rot_mask)
        
        iter_num = batch_idx + global_epoch * num_iteration_per_epoch

        logits_x = stu_model(inputs_x)

        visual_preds = torch.sigmoid(logits_x).detach().cpu()

        visual_preds_Ch1 = vutils.make_grid(visual_preds[:, 0, :, :].unsqueeze(1), normalize=True, scale_each=True)
        if not args.stage == 'soft':
            visual_preds_Ch2 = vutils.make_grid(visual_preds[:, 1, :, :].unsqueeze(1), normalize=True, scale_each=True)

        writer.add_image('img/train_img', visual_image, iter_num)
        writer.add_image('img/train_mask', visual_mask, iter_num)
        writer.add_image('img/train_preds_1Ch', visual_preds_Ch1, iter_num)
        if not args.stage == 'soft':
            writer.add_image('img/train_preds_2Ch', visual_preds_Ch2, iter_num)

        logits_x = logits_x.reshape(logits_x.size(0), args.n_classes, -1)
        targets_x = targets_x.reshape(targets_x.size(0), 1, -1)

        Lx = criterion(logits_x, targets_x.float(), args.n_weights)

        if not args.baseline:
            consistency_weight = get_current_consistency_weight(args, global_epoch)
            consistency_dist = softmax_mse_loss(outputs_ema, trans_outputs_stu).mean()
            Lu = consistency_weight * consistency_dist
            loss = Lx + Lu
        else:
            loss = Lx

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits_x.detach().cpu()
        mask = targets_x.detach().cpu()

        batch_dice, batch_cat_dice = dice_coef(preds, mask, args.stage)
        dice_tensor[batch_idx] = batch_cat_dice

        writer.add_scalar('dice/train_batch_dice', batch_dice, iter_num)
        writer.add_scalar('loss/train_loss', loss, iter_num)
        writer.add_scalar('loss/train_loss_supervised', Lx, iter_num)

        if not args.baseline:
            update_ema_variables(stu_model, ema_model, args.ema_decay, iter_num)
            writer.add_scalar('loss/train_loss_un', Lu, iter_num)
            writer.add_scalar('misc/consistency_weight', consistency_weight, iter_num)

    dice_cat_list = np.round(dice_tensor.mean(0), 4)
    ave_dice = dice_cat_list.mean()
    args.log_string('Training class dice %s:' %(dice_cat_list.tolist()))
    args.log_string('Training mean dice %s:' %(ave_dice.item()))

def validate(args, global_epoch, val_loader, optimizer, criterion, writer, is_ema, *models):

    if is_ema:
        model = models[1]
    else:
        model = models[0]

    model.eval()

    if args.stage == 'fine' or args.stage =='soft':
        coarse_model = models[2]
        coarse_model.eval()
    
    loss_sum = 0
    dice_tensor = np.zeros((len(val_loader), args.n_classes))

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):

            img, mask = data['img'], data['mask']
            img = img.to(args.device).float()  # (batch_size, 1, 96, 96)
            mask = mask.to(args.device).float()

            if args.stage == 'fine' or args.stage == 'soft':
                coarse_output = coarse_model(img)
                img = torch.cat((coarse_output, img), dim=1)

            output = model(img)

            output = output.reshape(output.size(0), args.n_classes, -1)
            mask = mask.reshape(mask.size(0), 1, -1)

            loss = criterion(output, mask, weights=args.n_weights)
            loss_sum += loss.item()

            preds = output.detach().cpu()
            mask = mask.detach().cpu()

            batch_dice, batch_cat_dice = dice_coef(preds, mask, args.stage)
            dice_tensor[batch_idx] = batch_cat_dice

        mean_loss = loss_sum / len(val_loader)
        dice_cat_list = np.round(dice_tensor.mean(0), 4)
        mean_dice = dice_cat_list.mean().item()

        if is_ema:
            loss_name = 'loss/ema_val_loss'
            dice_name = 'dice/ema_val_dice'
        else:
            loss_name = 'loss/val_loss'
            dice_name = 'dice/val_dice'

        writer.add_scalar(loss_name, mean_loss, global_epoch)
        writer.add_scalar(dice_name, mean_dice, global_epoch)

    return (mean_dice, dice_cat_list.tolist(), mean_loss)
