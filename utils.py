import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import torch


def mask2onehot(mask, stage):

    if stage == 'coarse':
        cat_index = [0, 1]
    elif stage == 'fine':
        cat_index = [2, 3]
    else:
        raise NotImplementedError

    _mask = [mask == i for i in cat_index]
    _mask = np.array(_mask).astype(np.uint8)
    return _mask


def dice_coef(inputs, targets, stage):

    inputs = inputs.permute(0,2,1).contiguous().view(-1, inputs.size(1))
    targets = targets.permute(0,2,1).contiguous().view(-1, targets.size(1)).long()

    t_one_hot = inputs.new_zeros(inputs.size(0), 4)
    t_one_hot.scatter_(1, targets, 1.)

    prob = torch.sigmoid(inputs)

    if stage == 'coarse':
        t_one_hot = t_one_hot[:, 0:2]
    elif stage == 'fine':
        t_one_hot = t_one_hot[:, 2:4]
    elif stage == 'soft':
        t_one_hot = t_one_hot[:, 3:4]
    else:
        pass

    assert(t_one_hot.size() == inputs.size()), print("shape not match")

    intersection = (prob * t_one_hot).sum(0)
    dice_classes = ((2. * intersection) / (prob.sum() + t_one_hot.sum() + 1e-7)).numpy()
    dice = np.round(dice_classes.mean().item(), 4)
    return dice, dice_classes

# def dice_coef(preds, mask, stage):

#     inter = np.zeros(4)
#     union = np.zeros(4)
    
#     if stage == 'fine' or stage == 'soft':
#         preds += 2

#     for l in range(4):
#         inter[l] += np.sum((preds == l) & (mask == l))
#         union[l] += np.sum((preds == l) | (mask == l))
    
#     dice_classes = (np.array(inter) * 2) / (np.array(inter) + np.array(union) + 1e-7)
    
#     if stage == 'coarse':
#         dice_classes = dice_classes[0: 2]
#     elif stage == 'fine':
#         dice_classes = dice_classes[2: 4]
#     else:
#         raise NotImplementedError

#     dice = np.round(dice_classes.mean().item(), 4)

#     return dice, dice_classes


def record_dataset(args):
    target_paths = [os.path.join(args.data_dir, str(i)) for i in range(150)]

    if args.dataset_mode == 'main_branch':
        name_list = ['1', '13', '20']
    elif args.dataset_mode == 'all_branch':
        name_list = [str(i) for i in range(25)]

    dataset = []
    for path in target_paths:
        for tar_name in name_list:
            if os.path.exists(os.path.join(path, tar_name, 'mask_refine_checked.nii.gz')):
                dataset.append(os.path.join(path, tar_name))
            elif os.path.exists(os.path.join(path, tar_name, 'mask_refine.nii.gz')):
                dataset.append(os.path.join(path, tar_name))
            elif os.path.exists(os.path.join(path, tar_name, 'mask.nii.gz')):
                dataset.append(os.path.join(path, tar_name))
    
    case_list = []
    branch_list = []
    slice_list = []

    for file_path in dataset:
        if os.path.exists(os.path.join(file_path, 'mask_refine_checked.nii.gz')):
            mask_path = os.path.join(file_path, 'mask_refine_checked.nii.gz')
        elif os.path.exists(os.path.join(file_path, 'mask_refine.nii.gz')):
            mask_path = os.path.join(file_path, 'mask_refine.nii.gz')
        else:
            mask_path = os.path.join(file_path, 'mask.nii.gz')

        mask_itk = sitk.ReadImage(mask_path)
        mask_vol = sitk.GetArrayFromImage(mask_itk)

        # remove anchor voxels
        mask_vol[mask_vol>5] = 0
        mask_vol[mask_vol==4]=0

        for i in range(mask_vol.shape[0]):
            if mask_vol[i, int((mask_vol.shape[1] - 1) / 2), int((mask_vol.shape[2] - 1) / 2)] != 0:
                unique = np.unique(mask_vol[i])
                if 2 in unique or 3 in unique:
                    case_list.append(file_path.split("/")[6])
                    branch_list.append(file_path.split("/")[7])
                    slice_list.append(i)
        
    df = pd.DataFrame({'case_id': case_list, 'branch_id': branch_list, 'slice_id': slice_list})
    df.to_csv('./plaque_info.csv', index=False)


def count_dataset(args):
    target_paths = [os.path.join(args.data_dir, str(i)) for i in range(150)]

    if args.dataset_mode == 'main_branch':
        name_list = ['1', '13', '20']
    elif args.dataset_mode == 'all_branch':
        name_list = [str(i) for i in range(25)]

    dataset = []
    for path in target_paths:
        for tar_name in name_list:
            if os.path.exists(os.path.join(path, tar_name, 'mask_refine_checked.nii.gz')):
                dataset.append(os.path.join(path, tar_name))
            elif os.path.exists(os.path.join(path, tar_name, 'mask_refine.nii.gz')):
                dataset.append(os.path.join(path, tar_name))
            elif os.path.exists(os.path.join(path, tar_name, 'mask.nii.gz')):
                dataset.append(os.path.join(path, tar_name))
    
    case_count_list = []
    branch_count = 0
    slice_count = 0
    total_list = np.zeros(4)
    
    for file_path in dataset:
        label_count = np.zeros(4)

        if os.path.exists(os.path.join(file_path, 'mask_refine_checked.nii.gz')):
            mask_path = os.path.join(file_path, 'mask_refine_checked.nii.gz')
        elif os.path.exists(os.path.join(file_path, 'mask_refine.nii.gz')):
            mask_path = os.path.join(file_path, 'mask_refine.nii.gz')
        else:
            mask_path = os.path.join(file_path, 'mask.nii.gz')

        mask_itk = sitk.ReadImage(mask_path)
        mask_vol = sitk.GetArrayFromImage(mask_itk)

        # remove anchor voxels
        mask_vol[mask_vol>5] = 0
        mask_vol[mask_vol==4]=0

        for i in range(mask_vol.shape[0]):
            if mask_vol[i, int((mask_vol.shape[1]-1)/2), int((mask_vol.shape[2]-1)/2)] != 0:
                unique = np.unique(mask_vol[i])
                label_count[unique.astype(int)] += 1
        
        if label_count[3] != 0:
            branch_count += 1
            case_count_list.append(file_path.split('/')[6])
            print(file_path)
            # print(label_count[3])
        
        slice_count += label_count[3]
        total_list += label_count
        
    case_count = len(set(case_count_list))
    print("Case num contains soft plaque: {}".format(case_count))
    print("Branch num contains soft plaque: {}".format(branch_count))
    print("Slice num contains soft plaque: {}".format(int(slice_count)))
    print("Slice num of each class in the whole dataset is: {}".format(total_list))
