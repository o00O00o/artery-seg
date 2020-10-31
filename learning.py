import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def train(args, global_epoch, train_loader, model, optimizer, criterion):
    model.train()

    eps = 1e-6
    loss_sum = 0
    total_correct, total_seen = 0, 0
    total_seen_class = [0 for _ in range(args.n_classes)]
    total_inter_class = [0 for _ in range(args.n_classes)]
    total_union_class = [0 for _ in range(args.n_classes)]
    num_batches = len(train_loader)

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        total_seen_class_tmp = [0 for _ in range(args.n_classes)]
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

        if args.loss_func.startswith('dice'):
            preds = F.softmax(output, dim=1).data.max(1)[1]
        elif args.loss_func.startswith('cross_entropy'):
            preds = F.log_softmax(output, dim=1).data.max(1)[1]

        mask = torch.squeeze(mask, 1)

        preds = preds.cpu().numpy()
        mask = mask.cpu().numpy()

        total_correct += np.sum((preds == mask))
        total_seen += preds.shape[0] * preds.shape[1]

        for l in range(args.n_classes):
            total_seen_class_tmp[l] += np.sum((mask == l))
            total_inter_class_tmp[l] += np.sum((preds == l) & (mask == l))
            total_union_class_tmp[l] += np.sum((preds == l) | (preds == l))
            total_seen_class[l] += total_seen_class_tmp[l]
            total_inter_class[l] += total_inter_class_tmp[l]
            total_union_class[l] += total_union_class_tmp[l]
        loss_sum += loss

    loss_sum /= num_batches
    acc_classes = np.array(total_inter_class) / np.array(total_seen_class)
    iou_classes = np.array(np.array(total_inter_class) / np.array(total_union_class) + eps)
    dice_classes = (np.array(total_inter_class) * 2) / (np.array(total_inter_class) + np.array(total_union_class))

    args.log_string('Training mean loss: %f' %(loss_sum))
    args.log_string('Training class acc %s:' %(np.around(acc_classes, 4)))
    args.log_string('Training mean acc %s:' %(np.around(np.mean(acc_classes[1:]), 4)))
    args.log_string('Training class iou %s:' %(np.around(iou_classes, 4)))
    args.log_string('Training mean iou %s:' %(np.around(np.mean(iou_classes[1:]), 4)))
    args.log_string('Training class dice %s:' %(np.around(dice_classes, 4)))
    args.log_string('Training mean dice %s:' %(np.around(np.mean(dice_classes[1:]), 4)))

def validate(args, global_epoch, val_loader, model, optimizer, criterion):

    with torch.no_grad():

        model.eval()

        eps = 1e-6
        loss_sum = 0
        total_correct, total_seen = 0, 0
        total_seen_class = [0 for _ in range(args.n_classes)]
        total_inter_class = [0 for _ in range(args.n_classes)]
        total_union_class = [0 for _ in range(args.n_classes)]
        num_batches = len(val_loader)

        for i, data in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
            total_seen_class_tmp = [0 for _ in range(args.n_classes)]
            total_inter_class_tmp = [0 for _ in range(args.n_classes)]
            total_union_class_tmp = [0 for _ in range(args.n_classes)]

            img, mask = data['img'], data['mask']
            img = img.permute(0,3,1,2).to(args.device).float()
            mask = mask.permute(0,3,1,2).to(args.device)

            output = model(img)
            output = output.contiguous().view(output.size(0), args.n_classes, -1)
            mask = mask.contiguous().view(mask.size(0), 1, -1)

            loss = criterion(output, mask, args.n_classes, weights=args.n_weights)

            if args.loss_func.startswith('dice'):
                preds = F.softmax(output, dim=1).data.max(1)[1]
            elif args.loss_func.startswith('cross_entropy') or args.loss_func.startswith('focal'):
                preds = F.log_softmax(output, dim=1).data.max(1)[1]
            mask = torch.squeeze(mask, 1)

            preds = preds.cpu().numpy()
            mask = mask.cpu().numpy()

            total_correct += np.sum((preds == mask))
            total_seen += preds.shape[0] * preds.shape[1]

            for l in range(args.n_classes):
                total_seen_class_tmp[l] += np.sum((mask == l))
                total_inter_class_tmp[l] += np.sum((preds == l) & (mask == l))
                total_union_class_tmp[l] += np.sum((preds == l) | (mask == l))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_inter_class[l] += total_inter_class_tmp[l]
                total_union_class[l] += total_union_class_tmp[l]
            loss_sum += loss

        loss_sum /= num_batches
        acc_classes = np.array(total_inter_class) / np.array(total_seen_class)
        iou_classes = np.array(np.array(total_inter_class) / (np.array(total_union_class) + eps))
        dice_classes = (np.array(total_inter_class) * 2) / (np.array(total_inter_class) + np.array(total_union_class))

        args.log_string('Val mean loss: %f' % (loss_sum))
        args.log_string('Val  class acc %s:' % (np.around(acc_classes, 4)))
        args.log_string('Val  mean acc %s:' % (np.around(np.mean(acc_classes[1:]), 4)))
        args.log_string('Val  class iou %s:' % (np.around(iou_classes, 4)))
        args.log_string('Val  mean iou %s:' % (np.around(np.mean(iou_classes[1:]), 4)))
        args.log_string('Val  class dice %s:' % (np.around(dice_classes, 4)))
        args.log_string('Val  mean dice %s:' % (np.around(np.mean(dice_classes[1:]), 4)))

    return (np.mean(dice_classes[1:]), dice_classes)

def cos_train(args, global_epoch, train_loader, model, optimizer, criterion):
    model.train()

    eps = 1e-6
    loss_sum = 0
    total_correct, total_seen = 0, 0
    total_seen_class = [0 for _ in range(args.n_classes)]
    total_inter_class = [0 for _ in range(args.n_classes)]
    total_union_class = [0 for _ in range(args.n_classes)]
    num_batches = len(train_loader)

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        total_seen_class_tmp = [0 for _ in range(args.n_classes)]
        total_inter_class_tmp = [0 for _ in range(args.n_classes)]
        total_union_class_tmp = [0 for _ in range(args.n_classes)]

        img, mask = data['img'], data['mask']
        target = img[:, :, :, 0].unsqueeze(3).permute(0, 3, 1, 2).to(args.device).float()
        search = img[:, :, :, 1].unsqueeze(3).permute(0, 3, 1, 2).to(args.device).float()
        target_mask = mask[:, :, :, 0].unsqueeze(3).permute(0, 3, 1, 2).to(args.device).float()
        search_mask = mask[:, :, :, 1].unsqueeze(3).permute(0, 3, 1, 2).to(args.device).float()

        target_pred, search_pred = model(target, search)

        target_pred = target_pred.contiguous().view(target_pred.size(0), args.n_classes, -1)
        target_mask = target_mask.contiguous().view(target_mask.size(0), 1, -1)

        search_pred = search_pred.contiguous().view(search_pred.size(0), args.n_classes, -1)
        search_mask = search_mask.contiguous().view(search_mask.size(0), 1, -1)
	
        loss = criterion(target_pred, target_mask, args.n_classes, weights=args.n_weights) + criterion(search_pred, search_mask, args.n_classes, weights=args.n_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.loss_func.startswith('dice'):
            preds = target_pred.data.max(1)[1]
        elif args.loss_func.startswith('cross_entropy'):
            preds = target_pred.data.max(1)[1]

        mask = torch.squeeze(target_mask, 1)

        preds = preds.cpu().numpy()
        mask = mask.cpu().numpy()

        total_correct += np.sum((preds == mask))
        total_seen += preds.shape[0] * preds.shape[1]

        for l in range(args.n_classes):
            total_seen_class_tmp[l] += np.sum((mask == l))
            total_inter_class_tmp[l] += np.sum((preds == l) & (mask == l))
            total_union_class_tmp[l] += np.sum((preds == l) | (preds == l))
            total_seen_class[l] += total_seen_class_tmp[l]
            total_inter_class[l] += total_inter_class_tmp[l]
            total_union_class[l] += total_union_class_tmp[l]
        loss_sum += loss

    loss_sum /= num_batches
    acc_classes = np.array(total_inter_class) / np.array(total_seen_class)
    iou_classes = np.array(np.array(total_inter_class) / np.array(total_union_class) + eps)
    dice_classes = (np.array(total_inter_class) * 2) / (np.array(total_inter_class) + np.array(total_union_class))

    args.log_string('Training mean loss: %f' %(loss_sum))
    args.log_string('Training class acc %s:' %(np.around(acc_classes, 4)))
    args.log_string('Training mean acc %s:' %(np.around(np.mean(acc_classes[1:]), 4)))
    args.log_string('Training class iou %s:' %(np.around(iou_classes, 4)))
    args.log_string('Training mean iou %s:' %(np.around(np.mean(iou_classes[1:]), 4)))
    args.log_string('Training class dice %s:' %(np.around(dice_classes, 4))) 
    args.log_string('Training mean dice %s:' %(np.around(np.mean(dice_classes[1:]), 4)))
def cos_validate(args, global_epoch, val_loader, model, optimizer, criterion):

    with torch.no_grad():

        model.eval()
        model.cuda()

        eps = 1e-6
        loss_sum = 0
        total_correct, total_seen = 0, 0
        total_seen_class = [0 for _ in range(args.n_classes)]
        total_inter_class = [0 for _ in range(args.n_classes)]
        total_union_class = [0 for _ in range(args.n_classes)]
        num_batches = len(val_loader)

        for i, data in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
            total_seen_class_tmp = [0 for _ in range(args.n_classes)]
            total_inter_class_tmp = [0 for _ in range(args.n_classes)]
            total_union_class_tmp = [0 for _ in range(args.n_classes)]

            img, mask = data['img'], data['mask']
            target = img[:, :, :, 0].unsqueeze(3).permute(0, 3, 1, 2).to(args.device).float() 
            search = img[:, :, :, 1].unsqueeze(3).permute(0, 3, 1, 2).to(args.device).float()
            mask = mask[:, :, :, 0].unsqueeze(3).permute(0, 3, 1, 2).to(args.device).float()
            target_pred, search_pred = model(target, search)
            target_pred = target_pred.contiguous().view(target_pred.size(0), args.n_classes, -1)
            mask = mask.contiguous().view(mask.size(0), 1, -1)

            loss = criterion(target_pred, mask, args.n_classes, weights=args.n_weights, validation=True)

            if args.loss_func.startswith('dice'):
                preds = F.softmax(target_pred, dim=1).data.max(1)[1]
            elif args.loss_func.startswith('cross_entropy') or args.loss_func.startswith('focal'):
                preds = F.log_softmax(output, dim=1).data.max(1)[1]
            mask = torch.squeeze(mask, 1)

            preds = preds.cpu().numpy()
            mask = mask.cpu().numpy()

            total_correct += np.sum((preds == mask))
            total_seen += preds.shape[0] * preds.shape[1]

            for l in range(args.n_classes):
                total_seen_class_tmp[l] += np.sum((mask == l))
                total_inter_class_tmp[l] += np.sum((preds == l) & (mask == l))
                total_union_class_tmp[l] += np.sum((preds == l) | (mask == l))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_inter_class[l] += total_inter_class_tmp[l]
                total_union_class[l] += total_union_class_tmp[l]
            loss_sum += loss

        loss_sum /= num_batches
        acc_classes = np.array(total_inter_class) / np.array(total_seen_class)
        iou_classes = np.array(np.array(total_inter_class) / (np.array(total_union_class) + eps))
        dice_classes = (np.array(total_inter_class) * 2) / (np.array(total_inter_class) + np.array(total_union_class))

        args.log_string('Val mean loss: %f' % (loss_sum))
        args.log_string('Val  class acc %s:' % (np.around(acc_classes, 4)))
        args.log_string('Val  mean acc %s:' % (np.around(np.mean(acc_classes[1:]), 4)))
        args.log_string('Val  class iou %s:' % (np.around(iou_classes, 4)))
        args.log_string('Val  mean iou %s:' % (np.around(np.mean(iou_classes[1:]), 4)))
        args.log_string('Val  class dice %s:' % (np.around(dice_classes, 4)))
        args.log_string('Val  mean dice %s:' % (np.around(np.mean(dice_classes[1:]), 4)))

    return (np.mean(dice_classes[1:]), dice_classes)

