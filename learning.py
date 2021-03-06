import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from losses import softmax_mse_loss
from transformations import *


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

def train(args, global_epoch, train_loader, model, optimizer, criterion, writer):

    model.train()
    loss_sum = 0
    total_inter_class = [0 for _ in range(args.n_classes)]
    total_union_class = [0 for _ in range(args.n_classes)]
    num_batches = len(train_loader)

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        total_inter_class_tmp = [0 for _ in range(args.n_classes)]
        total_union_class_tmp = [0 for _ in range(args.n_classes)]

        img, mask = data['img'], data['mask']
        img = img.permute(0,3,1,2).to(args.device).float()
        mask = mask.permute(0,3,1,2).to(args.device).float()

        output = model(img)

        output = output.contiguous().view(output.size(0), args.n_classes, -1)
        mask = mask.contiguous().view(mask.size(0), 1, -1)

        loss = criterion(output, mask, args.n_classes, weights=args.n_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = F.softmax(output, dim=1).data.max(1)[1]
        mask = torch.squeeze(mask, 1)

        preds = preds.cpu().numpy()
        mask = mask.cpu().numpy()

        for l in range(args.n_classes):
            total_inter_class_tmp[l] += np.sum((preds == l) & (mask == l))
            total_union_class_tmp[l] += np.sum((preds == l) | (mask == l))
            total_inter_class[l] += total_inter_class_tmp[l]
            total_union_class[l] += total_union_class_tmp[l]

        loss_sum += loss
        iter_num = global_epoch * num_batches + i

        writer.add_scalar('loss/train_loss', loss, iter_num)

    loss_sum /= num_batches
    dice_classes = (np.array(total_inter_class) * 2) / (np.array(total_inter_class) + np.array(total_union_class))

    args.log_string('Training mean loss: %f' %(loss_sum))
    args.log_string('Training class dice %s:' %(np.around(dice_classes, 4)))
    args.log_string('Training mean dice %s:' %(np.around(np.mean(dice_classes), 4)))

    writer.add_scalar('dice/train_dice', np.mean(dice_classes), global_epoch)

def validate(args, global_epoch, val_loader, model, optimizer, criterion, writer, is_ema):

    with torch.no_grad():

        model.eval()
        loss_sum = 0
        total_inter_class = [0 for _ in range(args.n_classes)]
        total_union_class = [0 for _ in range(args.n_classes)]

        for i, data in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
            total_inter_class_tmp = [0 for _ in range(args.n_classes)]
            total_union_class_tmp = [0 for _ in range(args.n_classes)]

            img, mask = data['img'], data['mask']
            img = img.permute(0,3,1,2).to(args.device).float()  # (batch_size, 1, 96, 96)
            mask = mask.permute(0,3,1,2).to(args.device)

            output = model(img)
            output = output.contiguous().view(output.size(0), args.n_classes, -1)  # (batch_size, 4, 96 * 96)
            mask = mask.contiguous().view(mask.size(0), 1, -1)  # (batch_size, 1, 96 * 96)

            loss = criterion(output, mask, args.n_classes, weights=args.n_weights)
            loss_sum += loss.item()

            preds = F.softmax(output, dim=1).data.max(1)[1]
            mask = torch.squeeze(mask, 1)

            preds = preds.cpu().numpy()
            mask = mask.cpu().numpy()

            for l in range(args.n_classes):
                total_inter_class_tmp[l] += np.sum((preds == l) & (mask == l))
                total_union_class_tmp[l] += np.sum((preds == l) | (mask == l))
                total_inter_class[l] += total_inter_class_tmp[l]
                total_union_class[l] += total_union_class_tmp[l]

        mean_loss = loss_sum / len(val_loader)
        dice_classes = (np.array(total_inter_class) * 2) / (np.array(total_inter_class) + np.array(total_union_class))
        dice_classes = np.around(dice_classes, 4)
        mean_dice = np.mean(dice_classes)

        if is_ema:
            loss_name = 'loss/ema_val_loss'
            dice_name = 'dice/ema_val_dice'
        else:
            loss_name = 'loss/val_loss'
            dice_name = 'dice/val_dice'

        writer.add_scalar(loss_name, mean_loss, global_epoch)
        writer.add_scalar(dice_name, mean_dice, global_epoch)

    return (mean_dice, dice_classes, mean_loss)

def train_mean_teacher(args, global_epoch, labeled_loader, unlabeled_loader, stu_model, ema_model, optimizer, criterion, writer):

    total_inter_class = [0 for _ in range(args.n_classes)]
    total_union_class = [0 for _ in range(args.n_classes)]

    labeled_num_batches = len(labeled_loader)
    unlabeled_num_batches = len(unlabeled_loader)

    if not args.baseline:
        num_iteration_per_epoch = max(labeled_num_batches, unlabeled_num_batches)
    else:
        num_iteration_per_epoch = labeled_num_batches

    stu_model.train()
    ema_model.train()

    for batch_idx in tqdm(range(num_iteration_per_epoch)):

        total_inter_class_tmp = [0 for _ in range(args.n_classes)]
        total_union_class_tmp = [0 for _ in range(args.n_classes)]

        try:
            data = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_loader)
            data = labeled_train_iter.next()
        
        inputs_x, targets_x = data['img'], data['mask']
        inputs_x = inputs_x.permute(0,3,1,2).to(args.device).float()
        targets_x = targets_x.permute(0,3,1,2).to(args.device)

        inputs_x, targets_x = inputs_x.to(args.device), targets_x.to(args.device)

        if not args.baseline:
            try:
                data = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_loader)
                data = unlabeled_train_iter.next()
            
            inputs_stu = data['img']
            inputs_stu = inputs_stu.permute(0, 3, 1, 2).to(args.device).float()  # (12, 1, 96, 96)
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
        logits_x = logits_x.contiguous().view(logits_x.size(0), args.n_classes, -1)  # (batch_size, 4, 96 * 96)
        targets_x = targets_x.contiguous().view(targets_x.size(0), 1, -1)  # (batch_size, 1, 96 * 96)

        Lx = criterion(logits_x, targets_x.long(), args.n_classes, args.n_weights)

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

        if not args.baseline:
            update_ema_variables(stu_model, ema_model, args.ema_decay, iter_num)
        
        writer.add_scalar('loss/train_loss', loss, iter_num)
        writer.add_scalar('loss/train_loss_supervised', Lx, iter_num)
        if not args.baseline:
            writer.add_scalar('loss/train_loss_un', Lu, iter_num)
            writer.add_scalar('misc/consistency_weight', consistency_weight, iter_num)

        preds = F.softmax(logits_x, dim=1).data.max(1)[1]
        mask = torch.squeeze(targets_x, 1)

        preds = preds.cpu().numpy()
        mask = mask.cpu().numpy()

        for l in range(args.n_classes):
            total_inter_class_tmp[l] += np.sum((preds == l) & (mask == l))
            total_union_class_tmp[l] += np.sum((preds == l) | (mask == l))
            total_inter_class[l] += total_inter_class_tmp[l]
            total_union_class[l] += total_union_class_tmp[l]

    dice_classes = (np.array(total_inter_class) * 2) / (np.array(total_inter_class) + np.array(total_union_class))

    args.log_string('Training class dice %s:' %(np.around(dice_classes, 4)))
    args.log_string('Training mean dice %s:' %(np.around(np.mean(dice_classes), 4)))
