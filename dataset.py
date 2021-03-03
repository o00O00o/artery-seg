import os
import random
import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset


def split_dataset(args):
    # get the path list of all annotated data -----------------------------------
    target_paths = [os.path.join(args.data_dir, str(i)) for i in range(args.case_num)]

    # split the dataset -----------------------------------
    unlabeled_dirs = target_paths[:args.unlabeled_num]
    labeled_dirs = target_paths[args.unlabeled_num:args.unlabeled_num + args.labeled_num]
    val_dirs = target_paths[args.unlabeled_num + args.labeled_num:]

    if args.dataset_mode == 'main_branch':
        name_list = ['1', '13', '20']
    elif args.dataset_mode == 'all_branch':
        name_list = [str(i) for i in range(25)]

    unlabeled_set, labeled_set, val_set = [], [], []
    for item in zip((unlabeled_set, labeled_set, val_set), (unlabeled_dirs, labeled_dirs, val_dirs)):
        tar_set, tar_dirs = item
        for path in tar_dirs:
            for tar_name in name_list:
                if os.path.exists(os.path.join(path, tar_name, 'mask_refine_checked.nii.gz')):
                    tar_set.append(os.path.join(path, tar_name))
                elif os.path.exists(os.path.join(path, tar_name, 'mask_refine.nii.gz')):
                    tar_set.append(os.path.join(path, tar_name))
                elif os.path.exists(os.path.join(path, tar_name, 'mask.nii.gz')):
                    tar_set.append(os.path.join(path, tar_name))

    return unlabeled_set, labeled_set, val_set


def prepare_data(data_paths, stage):

    all_idx_list = []
    env_dict = {}
    env_count = 0
    labelweights = np.ones(4).astype(np.long)

    for file_path in data_paths:

        if os.path.exists(os.path.join(file_path, 'mpr_100.nii.gz')):
            mpr_path = os.path.join(file_path, 'mpr_100.nii.gz')
        else:
            mpr_path = os.path.join(file_path, 'mpr.nii.gz')

        if os.path.exists(os.path.join(file_path, 'mask_refine_checked.nii.gz')):
            mask_path = os.path.join(file_path, 'mask_refine_checked.nii.gz')
        elif os.path.exists(os.path.join(file_path, 'mask_refine.nii.gz')):
            mask_path = os.path.join(file_path, 'mask_refine.nii.gz')
        else:
            mask_path = os.path.join(file_path, 'mask.nii.gz')

        mpr_itk = sitk.ReadImage(mpr_path)
        mask_itk = sitk.ReadImage(mask_path)
        mpr_vol = sitk.GetArrayFromImage(mpr_itk)
        mask_vol = sitk.GetArrayFromImage(mask_itk)
        assert mpr_vol.shape == mask_vol.shape, print('Wrong shape')

        # remove anchor voxels
        mask_vol[mask_vol>3] = 0

        if stage == 'coarse':
            mask_vol[mask_vol == 2] = 1
            mask_vol[mask_vol == 3] = 1

        unique, counts = np.unique(mask_vol, return_counts=True)
        labelweights[unique] += counts

        for i in range(mask_vol.shape[0]):
            if mask_vol[i, int((mask_vol.shape[1] - 1) / 2), int((mask_vol.shape[2] - 1) / 2)] != 0:
                all_idx_list.append((i, env_count))

        env_dict[env_count] = {'img': mpr_vol, 'mask': mask_vol}
        env_count += 1

    labelweights = labelweights / np.sum(labelweights)
    labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 1.0)

    if stage == 'coarse':
        labelweights = labelweights[0:2]
    elif stage == 'fine':
        labelweights = labelweights[2:4]
    elif stage == 'soft':
        labelweights = np.array([1.])
    else:
        raise NotImplementedError

    return all_idx_list, env_dict, labelweights

def center_crop(img, mask, crop_size):
    channel, width, height = np.shape(img)
    assert width >= crop_size, "crop_size should be smaller than img size"

    gap_w, gap_h = int((width - crop_size) / 2), int((height - crop_size) / 2)
    img = img[:, gap_w:gap_w + crop_size, gap_h:gap_h + crop_size]
    mask = mask[gap_w:gap_w + crop_size, gap_h: gap_h + crop_size]

    return img, mask

def adjust_HU(img, value_range):
    min_v, max_v = value_range
    diff = random.randrange(min_v, max_v)
    img += diff
    return img

def normalize(img):
    img = np.clip(img, -360, 840)
    img = (img + 360) / 1200
    return img

class Probe_Dataset(Dataset):
    def __init__(self, data_paths, args):
        self.data_paths = data_paths
        self.args = args
        # labelweights is used in the main function to alleviate unbalance problem
        self.idx_list, self.env_dict, self.labelweights = prepare_data(self.data_paths, args.stage)

    def __len__(self):
        length = len(self.idx_list)
        return length

    def __getitem__(self, idx):

        pt_idx, env_idx = self.idx_list[idx]

        img_stack_list = []
        img_stack_list.append(self.env_dict[env_idx]['img'][pt_idx].astype(np.float))
        step = int((self.args.slices - 1) / 2)
        for i in range(step):
            s_idx = max(pt_idx - sum([i for i in range(i+1)]), 0)
            e_idx = min(pt_idx + sum([i for i in range(i+1)]), len(self.env_dict[env_idx]['img']) - 1)
            img_stack_list.insert(0, self.env_dict[env_idx]['img'][s_idx].astype(np.float))
            img_stack_list.insert(-1, self.env_dict[env_idx]['img'][e_idx].astype(np.float))
        probe_img = np.stack(img_stack_list, axis=0)

        probe_mask = self.env_dict[env_idx]['mask'][pt_idx].astype(np.float)

        probe_img, probe_mask = center_crop(probe_img, probe_mask, self.args.crop_size)
        # probe_img = normalize(probe_img)
        sample = {'img': probe_img, 'mask': probe_mask}

        return sample


# if __name__ == "__main__":
#     args = parse_args()
#     args.data_dir = "/Users/gaoyibo/plaques/all_subset"
#     args.crop_size = 100
#     train_dir, val_dir, _ = split_dataset(args, 0)
#     train_dataset = Probe_Dataset(train_dir, args, augmentation=True)
#     print(len(train_dataset))
#     probe_img = train_dataset[42]['img']
#     probe_mask = train_dataset[42]['mask']
#     print(probe_img.shape, probe_mask.shape)
#     f = plt.figure()
#     f.add_subplot(1,4, 1)
#     plt.imshow(probe_img[:,:,0], cmap='gray')
#     f.add_subplot(1,4, 2)
#     plt.imshow(probe_img[:,:,1], cmap='gray')
#     f.add_subplot(1,4, 3)
#     plt.imshow(probe_img[:,:,2], cmap='gray')
#     f.add_subplot(1,4, 4)
#     plt.imshow(probe_mask[:,:,0], cmap='gray')
#     plt.show(block=True)
