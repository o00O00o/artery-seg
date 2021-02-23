import os
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
import imgaug as ia
import imgaug.augmenters as iaa
from torch.utils.data import Dataset
from dataset import center_crop
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
# from dataset import Probe_Dataset, split_dataset, normalize, center_crop, adjust_HU
# from torch.utils.data import ConcatDataset, Dataset


def parse_args():
    parser = argparse.ArgumentParser('Model')
    # parser.add_argument('--data_dir', default='/mnt/lustre/wanghuan3/gaoyibo/plaques_v2', help='folder name for training set')
    parser.add_argument('--data_dir', default='/Users/gaoyibo/Datasets/plaques/all_subset_v3', help='folder name for training set')
    parser.add_argument('--crop_size', type=int, default=64, help='size for square patch')
    parser.add_argument('--case_num', type=int, default=150, help='the num of total case')
    parser.add_argument('--unlabeled_num', default=0, type=int, help='the num of unlabeded case')
    parser.add_argument('--labeled_num', default=120, type=int, help='the num of labeled case')
    parser.add_argument('--times', default=5, type=int)
    parser.add_argument('--slices', type=int, default=7, help='slices used in the 2.5D mode')
    parser.add_argument('--aug_list_dir', default='./plaque_info.csv', type=str)
    parser.add_argument('--over_sample', action="store_true")
    parser.add_argument('--loss_func', type=str, default='dice', help='Loss function used for training [default: dice]')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 256]')
    parser.add_argument('--all_label', action='store_true', help='full supervised configuration if set true')
    parser.add_argument('--data_mode', type=str, default='2D', help='data mode')
    parser.add_argument('--dataset_mode', type=str, default='all_branch', help='dataset mode be to used: main_branch or all_branch')
    parser.add_argument('--n_classes', type=int, default=4, help='classes for segmentation')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--baseline', action='store_true')
    
    return parser.parse_args()

def prepare_data(data_paths, query_table, times):

    all_idx_list = []
    env_dict = {}
    env_count = 0

    for file_path in data_paths:

        case_id = int(file_path.split('/')[6])
        branch_id = int(file_path.split('/')[7])
        slice_id = query_table['slice_id'].loc[(query_table['case_id'] == case_id) & (query_table['branch_id'] == branch_id)].tolist()  # 查找相应case和branch的切片id,并转化为列表

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

        mask_vol[mask_vol>3] = 0

        for idx in range(mask_vol.shape[0]):
            if idx in slice_id:
                for time in range(times):  # 重复times次，作为复制
                    all_idx_list.append((idx, env_count))

        env_dict[env_count] = {'img': mpr_vol, 'mask': mask_vol}
        env_count += 1

    return all_idx_list, env_dict

class AugmentDataset(Dataset):
    def __init__(self, args, type='unlabel', augmentation=False):
        self.args = args
        self.augmentation = augmentation
        df = pd.read_csv(args.aug_list_dir)

        if type == 'unlabel':
            query_table = df.loc[df['case_id'].isin(range(args.unlabeled_num))]
        elif type == 'label':
            query_table = df.loc[df['case_id'].isin(range(args.unlabeled_num, args.unlabeled_num + args.labeled_num))]
        
        self.dataset = []
        for idx, row in query_table.iterrows():
            self.dataset.append(os.path.join(args.data_dir, str(row['case_id']), str(row['branch_id'])))
        
        self.dataset = sorted(set(self.dataset), key=self.dataset.index)  # 去除重复元素并保留之前顺序
        self.idx_list, self.env_dict = prepare_data(self.dataset, query_table, args.times)

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        ia.seed(idx + 1)
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
            probe_img = np.stack(img_stack_list, axis=0)
            probe_mask = self.env_dict[env_idx]['mask'][pt_idx].astype(np.float)
        else:
            print(self.args.data_mode + " is not implemented.")
            raise NotImplementedError

        probe_img, probe_mask = center_crop(probe_img, probe_mask, self.args.crop_size)

        # augmentation
        if self.augmentation:
            seg_map = SegmentationMapsOnImage(probe_mask, shape=probe_img.shape)
            aug_affine = iaa.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-360, 360), shear=(-20, 20), mode='edge')
            probe_img, seg_map = aug_affine(image=probe_img, segmentation_maps=seg_map)
            probe_mask = seg_map.get_arr()

        sample = {'img': probe_img, 'mask': probe_mask}
        return sample


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    args = parse_args()
    aug_dataset = AugmentDataset(args, 'label')
    img1 = aug_dataset[14]['img']
    mask1 = aug_dataset[14]['mask']
    print(img1.shape)
    # show_two_img(img1, mask1)

    # unlabeled_dir, labeled_dir, val_dir = split_dataset(args)
    # labeled_set = Probe_Dataset(labeled_dir, args)
    # img2 = labeled_set[342]['img']
    # mask2 = labeled_set[342]['mask']
    # show_two_img(img2, mask2)
    # img2 = (img2.reshape(64, 64) - 0.3) * 10000
    # print(img2)
    # img2 = np.around(img2, 0)
    # print(img2)
    # for i in range(len(aug_dataset)):
    #     mask = aug_dataset[i]['mask']
    #     unique = np.unique(mask)
    #     if 2 not in unique and 3 not in unique:
    #         print(i)

    # unlabeled_dir, labeled_dir, val_dir = split_dataset(args)
    # labeled_set = Probe_Dataset(labeled_dir, args)
    # union_set = ConcatDataset([labeled_set, aug_dataset])
    # print(union_set.labelweights)
    # print(len(union_set))
