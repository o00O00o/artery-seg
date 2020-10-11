import os
import glob
import math
import random
import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
# import matplotlib.pyplot as plt


def split_dataset(args, cur_loop):
    # return lists of train, val and test dir paths
    train_num, val_num, data_dir, seed = args.train_num, args.val_num, args.data_dir, args.seed
    random.seed(seed)

    # get the path list of all annotated data -----------------------------------
    all_dataset = []
    target_paths = sorted(glob.glob(data_dir + '/**/'))
    for path in target_paths:
        annotated = False
        for dir_path, dir_names, file_names in os.walk(path):
            for filename in file_names:
                if 'mask' in filename:
                    annotated = True
                    break
            if annotated:
                all_dataset.append(path)
    random.shuffle(all_dataset)
    print('All dataset num: {}'.format(len(all_dataset)))

    # split the dataset -----------------------------------
    if args.k_fold > 1:  # if cross_validation is implemented
        fold_length = int(len(all_dataset) / args.k_fold)
        start_idx = cur_loop * fold_length
        end_idx = (cur_loop + 1) * fold_length
        if end_idx > len(all_dataset) - 1:
            end_idx = len(all_dataset) - 1
        val_dirs = all_dataset[start_idx:end_idx]
        train_dirs = [i for i in all_dataset if i not in val_dirs]
        test_dirs = []
    else:  # split the dataset according to the proportion
        train_count = int(math.floor(len(all_dataset) * train_num))
        val_count = int(math.floor(len(all_dataset) * val_num))
        train_dirs = all_dataset[:train_count]
        val_dirs = all_dataset[train_count:train_count+val_count]
        test_dirs = all_dataset[train_count + val_count:]

    # select data according to the dataset_mode
    train_set, val_set, test_set = [], [], []
    for item in zip((train_set, val_set, test_set), (train_dirs, val_dirs, test_dirs)):
        tar_set, tar_dirs = item
        for path in tar_dirs:
            if args.dataset_mode == 'main_branch':
                for tar_name in ['1', '13', '20']:  # 1,13,20 are the main branches of the arterney
                    if os.path.exists(path + tar_name + '/' + 'mask.nii.gz'):
                        tar_set.append(path + tar_name)
            elif args.dataset_mode == 'all_branch':
                for tar_name in [str(i) for i in range(25)]:
                    if os.path.exists(path + tar_name + '/' + 'mask.nii.gz'):
                        tar_set.append(path + tar_name)

    return train_set, val_set, test_set

def prepare_data(data_paths, n_classes):
    all_idx_list = []
    env_dict = {}
    env_count = 0
    labelweights = np.zeros(n_classes)
    temp = 0

    # read the actual image from the path ---------------------------
    for file_path in data_paths:
        mpr_path = file_path + '/mpr_100.nii.gz'
        mask_path = file_path + '/mask.nii.gz'

        mpr_itk = sitk.ReadImage(mpr_path)
        mask_itk = sitk.ReadImage(mask_path)

        mpr_vol = sitk.GetArrayFromImage(mpr_itk)
        mask_vol = sitk.GetArrayFromImage(mask_itk)

        assert mpr_vol.shape == mask_vol.shape, print('Wrong shape')

        if n_classes == 4:
            mask_vol[mask_vol == 4] = 0
        else:
            pass

        # np.unique returns the unique elements and counts of the array
        unique, counts = np.unique(mask_vol, return_counts=True)
        labelweights[unique] += counts

        for i in range(mask_vol.shape[0]):
            if mask_vol[i, int((mask_vol.shape[1] - 1) / 2), int((mask_vol.shape[2] - 1) / 2)] != 0:
                all_idx_list.append((i, env_count))

        env_dict[env_count] = {'img':mpr_vol, 'mask':mask_vol}
        env_count += 1

        temp += 1

    labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 1.0)
    labelweights /= np.sum(labelweights)

    # all_idx_list contains the labelled position of an annotation and the annotation index
    # env_dict contains the mpr image and the corresponding mask
    return all_idx_list, env_dict, labelweights

def center_crop(img, mask, crop_size):
    width, height, channel = np.shape(img)
    assert width >= crop_size, "crop_size should be smaller than img size"

    gap_w, gap_h = int((width - crop_size) / 2), int((height - crop_size) / 2)
    img = img[gap_w:gap_w + crop_size, gap_h:gap_h + crop_size, :]
    mask = mask[gap_w:gap_w + crop_size, gap_h: gap_h + crop_size, :]

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
    def __init__(self, data_paths, args, augmentation=False):
        self.data_paths = data_paths
        self.augmentation = augmentation
        self.args = args
        # labelweights is used in the main function to alleviate unbalance problem
        self.idx_list, self.env_dict, self.labelweights = prepare_data(self.data_paths, args.n_classes)

    def __len__(self):
        length = len(self.idx_list)
        return length

    def __getitem__(self, idx):

        pt_idx, env_idx = self.idx_list[idx]

        if self.args.data_mode == '2D':
            probe_img = self.env_dict[env_idx]['img'][pt_idx].astype(np.float)
            probe_mask = self.env_dict[env_idx]['mask'][pt_idx].astype(np.float)
            probe_img = np.expand_dims(probe_img, axis=-1)
            probe_mask = np.expand_dims(probe_mask, axis=-1)
        elif self.args.data_mode == '2.5D':
            img_stack_list = []
            img_stack_list.append(self.env_dict[env_idx]['img'][pt_idx].astype(np.float))
            step = int((self.args.slices - 1) / 2)
            for i in range(step):
                s_idx = max(pt_idx - sum([i for i in range(i+1)]), 0)
                e_idx = min(pt_idx + sum([i for i in range(i+1)]), len(self.env_dict[env_idx]['img']) - 1)
                img_stack_list.insert(0, self.env_dict[env_idx]['img'][s_idx].astype(np.float))
                img_stack_list.insert(-1, self.env_dict[env_idx]['img'][e_idx].astype(np.float))

            probe_img = np.stack(img_stack_list, axis=-1)
            probe_mask = self.env_dict[env_idx]['mask'][pt_idx].astype(np.float)
            probe_mask = np.expand_dims(probe_mask, axis=-1)
        else:
            raise NotImplementedError

        # crop img to target size
        probe_img, probe_mask = center_crop(probe_img, probe_mask, self.args.crop_size)
        probe_mask = probe_mask.astype(np.int32)

        # img transformation
        if self.augmentation:
            seg_map = SegmentationMapsOnImage(probe_mask, shape=probe_img.shape)

            if random.uniform(0, 1) > 0.3:
                aug_affine = iaa.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-360, 360), shear=(-20, 20), mode='edge')
                probe_img, seg_map = aug_affine(image=probe_img, segmentation_maps=seg_map)
                probe_mask = seg_map.get_arr()

            if random.uniform(0, 1) > 0.3:
                probe_img = adjust_HU(probe_img, value_range=(-50, 50))

        probe_img = normalize(probe_img)
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
