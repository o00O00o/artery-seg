import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import imgaug.augmenters as iaa
from main import parse_args
from dataset import Probe_Dataset, split_dataset, normalize, center_crop, adjust_HU
from torch.utils.data import ConcatDataset, Dataset
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


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

        for i in range(mask_vol.shape[0]):
            if i in slice_id:
                for i in range(times):  # 重复times次，作为复制
                    all_idx_list.append((i, env_count))

        env_dict[env_count] = {'img': mpr_vol, 'mask': mask_vol}
        env_count += 1

    return all_idx_list, env_dict

class AugmentDataset(Dataset):
    def __init__(self, args, type='unlabel'):
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
        pt_idx, env_idx = self.idx_list[idx]
        probe_img = self.env_dict[env_idx]['img'][pt_idx].astype(np.float)
        probe_mask = self.env_dict[env_idx]['mask'][pt_idx].astype(np.float)
        probe_img = np.expand_dims(probe_img, axis=-1)
        probe_mask = np.expand_dims(probe_mask, axis=-1)

        probe_img, probe_mask = center_crop(probe_img, probe_mask, args.crop_size)
        probe_mask = probe_mask.astype(np.int32)

        # augmentation
        seg_map = SegmentationMapsOnImage(probe_mask, shape=probe_img.shape)
        aug_affine = iaa.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-360, 360), shear=(-20, 20), mode='edge')
        probe_img, seg_map = aug_affine(image=probe_img, segmentation_maps=seg_map)
        probe_mask = seg_map.get_arr()
        probe_img = adjust_HU(probe_img, value_range=(-50, 50))

        probe_img = normalize(probe_img)
        sample = {'img': probe_img, 'mask': probe_mask}

        return sample


if __name__ == "__main__":
    args = parse_args()
    args.aug_list_dir = './plaque_info.csv'
    aug_dataset = AugmentDataset(args, 'label')
    # for i in range(len(aug_dataset)):
    #     img = aug_dataset[i]['img']
    #     mask = aug_dataset[i]['mask']
    #     unique = np.unique(mask)
    #     if 2 in unique or 3 in unique:
    #         print("Correct")

    unlabeled_dir, labeled_dir, val_dir = split_dataset(args)
    labeled_set = Probe_Dataset(labeled_dir, args)
    union_set = ConcatDataset([labeled_set, aug_dataset])
    print(union_set.labelweights)
    print(len(union_set))