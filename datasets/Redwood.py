import os
import pickle
import numpy as np
import torch.utils.data as data
from utils.SE3 import *
from utils.pointcloud import make_point_cloud


class RedwoodDataset(data.Dataset):
    def __init__(self,
                 root,
                 descriptor='fpfh',
                 min_overlap=0.30,
                 in_dim=6,
                 inlier_threshold=0.10,
                 num_node=5000,
                 use_mutual=True,
                 downsample=0.30,
                 select_scene=None,
                 augment_axis=0,
                 augment_rotation=1.0,
                 augment_translation=0.00,
                 ):
        self.root = root
        self.descriptor = descriptor
        assert descriptor in ['fcgf', 'fpfh']
        self.in_dim = in_dim
        self.min_overlap = min_overlap
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.downsample = downsample
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation

        self.scene_list = [
            'livingroom1-simulated',
            'livingroom2-simulated',
            'office1-simulated',
            'office2-simulated'
        ]
        if select_scene in self.scene_list:
            self.scene_list = [select_scene]
        else:
            scene_id = input("Please select one scene [1-4]:")
            self.scene_list = [ self.scene_list[int(scene_id)-1] ]
            print(f"Select scene {self.scene_list[0]}")

        assert len(self.scene_list) == 1
        scene = self.scene_list[0]

        # containers
        self.ids_list = []
        self.gt_trans = {}
        self.gt_overlap = {}
        self.gt_trajectory = []

        scene_path = f'{self.root}/{scene}/fragments'
        pcd_list = [filename for filename in os.listdir(scene_path) if filename.endswith('npz')]
        pcd_list = sorted(pcd_list, key=lambda x: int(x[:-4].split("_")[-2])) # sort the point cloud
        self.num_pcds = int(pcd_list[-1][:-4].split("_")[-2]) + 1

        # prepare gt_trajectory and gt_trans
        for src_id in range(self.num_pcds):
            pose_i = np.load(f"{self.root}/{scene}/fragments/fragment_{str(src_id).zfill(3)}.npy")
            self.gt_trajectory.append(pose_i)
            for tgt_id in range(src_id + 1, self.num_pcds):
                pose_j = np.load(f"{self.root}/{scene}/fragments/fragment_{str(tgt_id).zfill(3)}.npy")
                # pose_i : from src to world
                # pose_j : from tgt to world
                self.gt_trans[f'{scene}@{src_id}_{tgt_id}'] = np.linalg.inv(pose_j) @ pose_i # src -> world -> tgt
        assert self.num_pcds == len(self.gt_trajectory)

        return
        ## the following are used to evalute the pariwise registration results on Redwood
        # compute the overlapping ratio for the scene and save in a .pkl file
        gt_overlapping_filename = f"{self.root}/{scene}-overlap.pkl"
        if os.path.exists(gt_overlapping_filename):
            with open(gt_overlapping_filename, 'rb') as file:
                self.gt_overlap = pickle.load(file)
                # print(f'Load overlapping info from {gt_overlapping_filename}')
        else:
            for src_id in range(self.num_pcds):
                for tgt_id in range(src_id + 1, self.num_pcds):
                    # check whether overlapping is large than 30%
                    # trans = self.gt_trans[f'{scene}@{src_id}_{tgt_id}']
                    # R, t = trans[:3, :3].T, trans[:3, -1]
                    src_data = np.load(f"{self.root}/{scene}/fragments/fragment_{str(src_id).zfill(3)}_fpfh.npz")
                    tgt_data = np.load(f"{self.root}/{scene}/fragments/fragment_{str(tgt_id).zfill(3)}_fpfh.npz")
                    src_keypts = src_data['xyz']
                    tgt_keypts = tgt_data['xyz']
                    src_pcd = make_point_cloud(src_keypts)
                    src_pcd.transform(self.gt_trans[f"{scene}@{src_id}_{tgt_id}"])
                    frag1 = np.array(src_pcd.points)
                    frag2 = tgt_keypts
                    # frag1_warp = frag1 @ R + t
                    distance = np.linalg.norm(frag1[None, :, :] - frag2[:, None, :], axis=-1)
                    labels_1 = (distance.min(-1) < self.inlier_threshold).astype(np.int)
                    labels_2 = (distance.min(-2) < self.inlier_threshold).astype(np.int)
                    overlap = max(np.mean(labels_1), np.mean(labels_2))
                    self.gt_overlap[f'{scene}@{src_id}_{tgt_id}'] = overlap
                    print(f"Pair {src_id} and {tgt_id} overlapping {overlap*100:2f}%")
            with open(gt_overlapping_filename, 'wb') as file:
                pickle.dump(self.gt_overlap, file)
            print(f'Save overlapping info into {gt_overlapping_filename}')

        for k,v in self.gt_overlap.items():
            if v < self.min_overlap:
                self.gt_trans.pop(k)

        # print(f"Overlapping > {self.min_overlap}: totally {len(self.gt_trans.keys())} pairs.")

    def __getitem__(self, index):
        sorted_keys = sorted(self.gt_trans.keys(), key=lambda x: (int(x.split('@')[1].split('_')[0]), int(x.split('@')[1].split('_')[1])))
        key = sorted_keys[index]

        scene = key.split('@')[0]
        src_id = key.split('@')[1].split('_')[0]
        tgt_id = key.split('@')[1].split('_')[1]

        if self.descriptor == 'fcgf':
            src_data = np.load(f"{self.root}/{scene}/fragments/fragment_{src_id.zfill(3)}_fcgf.npz")
            tgt_data = np.load(f"{self.root}/{scene}/fragments/fragment_{tgt_id.zfill(3)}_fcgf.npz")
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
        elif self.descriptor == 'fpfh':
            src_data = np.load(f"{self.root}/{scene}/fragments/fragment_{src_id.zfill(3)}_fpfh.npz")
            tgt_data = np.load(f"{self.root}/{scene}/fragments/fragment_{tgt_id.zfill(3)}_fpfh.npz")
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
            # l2 normalize to fpfh descriptor
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # src_pose = np.load(f"{self.root}/{scene}/fragments/fragment_{str(src_id).zfill(3)}.npy")
        # tgt_pose = np.load(f"{self.root}/{scene}/fragments/fragment_{str(tgt_id).zfill(3)}.npy")
        # src_pcd = make_point_cloud(src_keypts)
        # src_pcd.transform(np.linalg.inv(src_pose)) # move pcd to local coord, now need (src_pose) to world
        # tgt_pcd = make_point_cloud(tgt_keypts)
        # tgt_pcd.transform(np.linalg.inv(tgt_pose)) # move pcd to local coord, now need (tgt_pose) to world
        # src_keypts = np.array(src_pcd.points)
        # tgt_keypts = np.array(tgt_pcd.points)

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
            corr = np.concatenate([np.where(mutual_nearest == 1)[0][:, None], source_idx[mutual_nearest][:, None]], axis=-1)
        else:
            corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)

        gt_trans = self.gt_trans[key] # the given ground truth trans is src -> tgt
        # # convert the gt_trans to our convention
        # # for gt_trans: gt_trans @ src[4,N] => tgt[4,N]
        # # for ours: src[N,3] @ gt_trans[0:3, 0:3] + gt_trans[0:3, -1] => tgt[3, N]
        # gt_trans[0:3, 0:3] = gt_trans[0:3, 0:3].T
        # gt_R = gt_trans[:3, :3] 
        # gt_t = gt_trans[:3, -1] 

        # build the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        # frag1_warp = frag1 @ gt_R + gt_t
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        labels = (distance < self.inlier_threshold).astype(np.int)

        # prepare input to the network
        input_src_keypts = src_keypts[corr[:, 0]]
        input_tgt_keypts = tgt_keypts[corr[:, 1]]

        if self.in_dim == 6:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
            # move the center of each point cloud to (0,0,0).
            corr_pos = corr_pos - corr_pos.mean(0)

        return corr_pos.astype(np.float32), \
               input_src_keypts.astype(np.float32), \
               input_tgt_keypts.astype(np.float32), \
               gt_trans.astype(np.float32), \
               labels.astype(np.float32), key

    def __len__(self):
        return self.gt_trans.keys().__len__()

    def __loadlog__(self, gtpath):
        with open(os.path.join(gtpath, 'gt.log')) as f:
            content = f.readlines()
        result = {}
        i = 0
        while i < len(content):
            line = content[i].replace("\n", "").split("\t")[0:3]
            trans = np.zeros([4, 4])
            trans[0] = np.fromstring(content[i + 1], dtype=float, sep=' \t')
            trans[1] = np.fromstring(content[i + 2], dtype=float, sep=' \t')
            trans[2] = np.fromstring(content[i + 3], dtype=float, sep=' \t')
            trans[3] = np.fromstring(content[i + 4], dtype=float, sep=' \t')
            i = i + 5
            result[f'{int(line[0])}_{int(line[1])}'] = trans

        return result


if __name__ == "__main__":
    dset = RedwoodDataset(
        root='/ssd2/xuyang/Augmented_ICL-NUIM_raw/',
        num_node='all',
        min_overlap=0.0,
        descriptor='fpfh',
        select_scene='livingroom1-simulated',
    )
    for i in range(len(dset)):
        _, _, _, _, _, key = dset[i]
        print(key)
