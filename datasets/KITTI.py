import os
import torch.utils.data as data
from utils.pointcloud import make_point_cloud, estimate_normal
from utils.SE3 import *

class KITTIDataset(data.Dataset):
    def __init__(self,
                root,
                split='train',
                descriptor='fcgf',
                in_dim=6,
                inlier_threshold=0.60,
                num_node=5000,
                use_mutual=True,
                downsample=0.30,
                augment_axis=0,
                augment_rotation=1.0,
                augment_translation=0.01,
                ):
        self.root = root
        self.split = split
        self.descriptor = descriptor
        assert descriptor in ['fcgf', 'fpfh']
        self.in_dim = in_dim
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.downsample = downsample
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation

        # containers
        self.ids_list = []

        for filename in os.listdir(f"{self.root}/{descriptor}_{split}/"):
            self.ids_list.append(os.path.join(f"{self.root}/{descriptor}_{split}/", filename))

        # self.ids_list = sorted(self.ids_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    def __getitem__(self, index):
        # load meta data
        filename = self.ids_list[index]
        data = np.load(filename)
        src_keypts = data['xyz0']
        tgt_keypts = data['xyz1']
        src_features = data['features0']
        tgt_features = data['features1']
        if self.descriptor == 'fpfh':
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # compute ground truth transformation
        orig_trans = data['gt_trans']
        # data augmentation
        if self.split == 'train':
            src_keypts += np.random.rand(src_keypts.shape[0], 3) * 0.05
            tgt_keypts += np.random.rand(tgt_keypts.shape[0], 3) * 0.05
        aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
        aug_T = translation_matrix(self.augment_translation)
        aug_trans = integrate_trans(aug_R, aug_T)
        tgt_keypts = transform(tgt_keypts, aug_trans)
        gt_trans = concatenate(aug_trans, orig_trans)

        # select {self.num_node} numbers of keypoints
        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]
        src_sel_ind = np.arange(N_src)
        tgt_sel_ind = np.arange(N_tgt)
        if self.num_node != 'all' and N_src > self.num_node:
            src_sel_ind = np.random.choice(N_src, self.num_node, replace=False)
        if self.num_node != 'all' and N_tgt > self.num_node:
            tgt_sel_ind = np.random.choice(N_tgt, self.num_node, replace=False)
        src_desc = src_features[src_sel_ind, :]
        tgt_desc = tgt_features[tgt_sel_ind, :]
        src_keypts = src_keypts[src_sel_ind, :]
        tgt_keypts = tgt_keypts[tgt_sel_ind, :]

        # construct the correspondence set by mutual nn in feature space.
        distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
        source_idx = np.argmin(distance, axis=1)
        if self.use_mutual:
            target_idx = np.argmin(distance, axis=0)
            mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
            corr = np.concatenate([np.where(mutual_nearest == 1)[0][:,None], source_idx[mutual_nearest][:,None]], axis=-1)
        else:
            corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)

        # compute the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        labels = (distance < self.inlier_threshold).astype(np.int)

        # add random outlier to input data
        if self.split == 'train' and np.mean(labels) > 0.5:
            num_outliers = int(0.0 * len(corr))
            src_outliers = np.random.randn(num_outliers, 3) * np.mean(src_keypts, axis=0)
            tgt_outliers = np.random.randn(num_outliers, 3) * np.mean(tgt_keypts, axis=0)
            input_src_keypts = np.concatenate( [src_keypts[corr[:, 0]], src_outliers], axis=0)
            input_tgt_keypts = np.concatenate( [tgt_keypts[corr[:, 1]], tgt_outliers], axis=0)
            labels = np.concatenate( [labels, np.zeros(num_outliers)], axis=0)
        else:
            # prepare input to the network
            input_src_keypts = src_keypts[corr[:, 0]]
            input_tgt_keypts = tgt_keypts[corr[:, 1]]

        if self.in_dim == 3:
            corr_pos = input_src_keypts - input_tgt_keypts
        elif self.in_dim == 6:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
            # move the center of each point cloud to (0,0,0).
            corr_pos = corr_pos - corr_pos.mean(0)
        elif self.in_dim == 9:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts, input_src_keypts-input_tgt_keypts], axis=-1)
        elif self.in_dim == 12:
            src_pcd = make_point_cloud(src_keypts)
            tgt_pcd = make_point_cloud(tgt_keypts)
            estimate_normal(src_pcd, radius=self.downsample*2)
            estimate_normal(tgt_pcd, radius=self.downsample*2)
            src_normal = np.array(src_pcd.normals)
            tgt_normal = np.array(tgt_pcd.normals)
            src_normal = src_normal[src_sel_ind, :]
            tgt_normal = tgt_normal[tgt_sel_ind, :]
            input_src_normal = src_normal[corr[:, 0]]
            input_tgt_normal = tgt_normal[corr[:, 1]]
            corr_pos = np.concatenate([input_src_keypts, input_src_normal, input_tgt_keypts, input_tgt_normal], axis=-1)

        return corr_pos.astype(np.float32), \
            input_src_keypts.astype(np.float32), \
            input_tgt_keypts.astype(np.float32), \
            gt_trans.astype(np.float32), \
            labels.astype(np.float32),

    def __len__(self):
        return len(self.ids_list)

if __name__ == "__main__":
    dset = KITTIDataset(
                    root='/data/KITTI/',
                    split='test',
                    descriptor='fcgf',
                    num_node=5000,
                    use_mutual=False,
                    augment_axis=0,
                    augment_rotation=0,
                    augment_translation=0.00
                    )
    print(len(dset))
    for i in range(dset.__len__()):
        ret_dict = dset[i]
