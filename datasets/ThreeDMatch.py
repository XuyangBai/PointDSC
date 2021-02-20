import os
from os.path import join, exists
import pickle
import glob
import random
import torch.utils.data as data
from utils.pointcloud import make_point_cloud, estimate_normal
from utils.SE3 import *

class ThreeDMatchTrainVal(data.Dataset):
    def __init__(self, 
                 root, 
                 split, 
                 descriptor='fcgf',
                 in_dim=6,
                 inlier_threshold=0.10,
                 num_node=5000, 
                 use_mutual=True,
                 downsample=0.03, 
                 augment_axis=1, 
                 augment_rotation=1.0,
                 augment_translation=0.01,
                 ):
        self.root = root
        self.split = split
        self.descriptor = descriptor
        assert descriptor in ['fpfh', 'fcgf']
        self.in_dim = in_dim
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.downsample = downsample
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation

        # use point cloud pairs with more than 30% overlapping as the training/validation set
        OVERLAP_RATIO = 0.3 
        DATA_FILES = {
            'train': './misc/split/train_3dmatch.txt',
            'val': './misc/split/val_3dmatch.txt',
            # 'test': './mic/test_3dmatch.txt'
        }
        subset_names = open(DATA_FILES[split]).read().split()
        self.files = []
        self.length = 0
        for name in subset_names:
            fname = name + "*%.2f.txt" % OVERLAP_RATIO
            fnames_txt = glob.glob(root + f"/threedmatch/" + fname)
            assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                    self.files.append([fname[0], fname[1]])
                    self.length += 1
        
    def __getitem__(self, index):
        # load meta data
        src_id, tgt_id = self.files[index][0], self.files[index][1]
        if random.random() > 0.5:
            src_id, tgt_id = tgt_id, src_id
        
        # load point coordinates and pre-computed per-point local descriptors
        if self.descriptor == 'fcgf':
            src_data = np.load(f"{self.root}/threedmatch_feat/{src_id}".replace('.npz', '_fcgf.npz'))
            tgt_data = np.load(f"{self.root}/threedmatch_feat/{tgt_id}".replace('.npz', '_fcgf.npz'))
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
        elif self.descriptor == 'fpfh':
            src_data = np.load(f"{self.root}/threedmatch_feat/{src_id}".replace('.npz', '_fpfh.npz'))
            tgt_data = np.load(f"{self.root}/threedmatch_feat/{tgt_id}".replace('.npz', '_fpfh.npz'))
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
            np.nan_to_num(src_features)
            np.nan_to_num(tgt_features)
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # compute ground truth transformation
        orig_trans = np.eye(4).astype(np.float32)                
        # data augmentation (add data augmentation to original transformation)
        src_keypts += np.random.rand(src_keypts.shape[0], 3) * 0.005
        tgt_keypts += np.random.rand(tgt_keypts.shape[0], 3) * 0.005
        aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
        aug_T = translation_matrix(self.augment_translation)
        aug_trans = integrate_trans(aug_R, aug_T)
        tgt_keypts = transform(tgt_keypts, aug_trans)
        gt_trans = concatenate(aug_trans, orig_trans)

        # select {self.num_node} numbers of keypoints
        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]
        if self.num_node == 'all':
            src_sel_ind = np.arange(N_src)
            tgt_sel_ind = np.arange(N_tgt)
        else:
            src_sel_ind = np.random.choice(N_src, self.num_node)
            tgt_sel_ind = np.random.choice(N_tgt, self.num_node)
        src_desc = src_features[src_sel_ind, :]
        tgt_desc = tgt_features[tgt_sel_ind, :]
        src_keypts = src_keypts[src_sel_ind, :]
        tgt_keypts = tgt_keypts[tgt_sel_ind, :]            
        
        # construct the correspondence set by nearest neighbor searching in feature space.
        distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
        source_idx = np.argmin(distance, axis=1)
        source_dis = np.min(distance, axis=1)
        if self.use_mutual:
            target_idx = np.argmin(distance, axis=0)
            mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
            corr = np.concatenate([np.where(mutual_nearest == 1)[0][:,None], source_idx[mutual_nearest][:,None]], axis=-1)
        else:
            corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
        if len(corr) < 10:
            # skip pairs with too few correspondences.
            return  self.__getitem__(int(np.random.choice(self.__len__(),1)))
        
        # compute the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        labels = (distance < self.inlier_threshold).astype(np.int)

        # prepare input to the network
        if self.split == 'train' and np.mean(labels) > 0.5:
            # add random outlier to input data (deprecated)
            num_outliers = int(0 * len(corr))
            src_outliers = np.random.randn(num_outliers, 3) * np.mean(src_keypts, axis=0)
            tgt_outliers = np.random.randn(num_outliers, 3) * np.mean(tgt_keypts, axis=0)
            input_src_keypts = np.concatenate( [src_keypts[corr[:, 0]], src_outliers], axis=0)
            input_tgt_keypts = np.concatenate( [tgt_keypts[corr[:, 1]], tgt_outliers], axis=0)
            labels = np.concatenate([labels, np.zeros(num_outliers)], axis=0)
        else:
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
        elif self.in_dim == 70:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
            # move the center of each point cloud to (0,0,0).
            corr_pos = corr_pos - corr_pos.mean(0)
            corr_pos = np.concatenate([corr_pos, src_desc[corr[:,0]], tgt_desc[corr[:,1]]], axis=-1)
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
        return self.length


class ThreeDMatchTest(data.Dataset):
    def __init__(self, 
                root, 
                descriptor='fcgf',
                in_dim=6,
                inlier_threshold=0.10,
                num_node=5000, 
                use_mutual=True,
                downsample=0.03, 
                augment_axis=0, 
                augment_rotation=1.0,
                augment_translation=0.01,
                select_scene=None,
                ):
        self.root = root
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
        # assert augment_axis == 0
        # assert augment_rotation == 0
        # assert augment_translation == 0
        
        # containers
        self.gt_trans = {}
        
        self.scene_list = [
            '7-scenes-redkitchen',
            'sun3d-home_at-home_at_scan1_2013_jan_1',
            'sun3d-home_md-home_md_scan9_2012_sep_30',
            'sun3d-hotel_uc-scan3',
            'sun3d-hotel_umd-maryland_hotel1',
            'sun3d-hotel_umd-maryland_hotel3',
            'sun3d-mit_76_studyroom-76-1studyroom2',
            'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
        ]
        if select_scene in self.scene_list:
            self.scene_list = [select_scene]
        
        # load ground truth transformation
        for scene in self.scene_list:
            scene_path = f'{self.root}/fragments/{scene}'
            gt_path = f'{self.root}/gt_result/{scene}-evaluation'
            for k, v in self.__loadlog__(gt_path).items():
                self.gt_trans[f'{scene}@{k}'] = v
        
                  
    def __getitem__(self, index):
        # load meta data
        key = list(self.gt_trans.keys())[index]      
        scene = key.split('@')[0]
        src_id = key.split('@')[1].split('_')[0]
        tgt_id = key.split('@')[1].split('_')[1]
        
        # load point coordinates and pre-computed per-point local descriptors
        if self.descriptor == 'fcgf':
            src_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{src_id}_fcgf.npz")
            tgt_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{tgt_id}_fcgf.npz")
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
        elif self.descriptor == 'fpfh':
            src_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{src_id}_fpfh.npz")
            tgt_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{tgt_id}_fpfh.npz")
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # compute ground truth transformation
        orig_trans = np.linalg.inv(self.gt_trans[key])  # the given ground truth trans is target-> source   
        # data augmentation
        aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
        aug_T = translation_matrix(self.augment_translation)
        aug_trans = integrate_trans(aug_R, aug_T)
        tgt_keypts = transform(tgt_keypts, aug_trans)
        gt_trans = concatenate(aug_trans, orig_trans)

        # select {self.num_node} numbers of keypoints
        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]
        # use all point during test.
        if self.num_node == 'all':
            src_sel_ind = np.arange(N_src)
            tgt_sel_ind = np.arange(N_tgt)
        else:
            src_sel_ind = np.random.choice(N_src, self.num_node)
            tgt_sel_ind = np.random.choice(N_tgt, self.num_node)
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
             
        # build the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        labels = (distance < self.inlier_threshold).astype(np.int)
        
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
        elif self.in_dim == 70:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
            # move the center of each point cloud to (0,0,0).
            corr_pos = corr_pos - corr_pos.mean(0)
            corr_pos = np.concatenate([corr_pos, src_desc[corr[:,0]], tgt_desc[corr[:,1]]], axis=-1)
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
        return self.gt_trans.keys().__len__()
    
    def __loadlog__(self, gtpath):
        with open(os.path.join(gtpath, 'gt.log')) as f:
            content = f.readlines()
        result = {}
        i = 0
        while i < len(content):
            line = content[i].replace("\n", "").split("\t")[0:3]
            trans = np.zeros([4, 4])
            trans[0] = np.fromstring(content[i+1], dtype=float, sep=' \t')
            trans[1] = np.fromstring(content[i+2], dtype=float, sep=' \t')
            trans[2] = np.fromstring(content[i+3], dtype=float, sep=' \t')
            trans[3] = np.fromstring(content[i+4], dtype=float, sep=' \t')
            i = i + 5
            result[f'{int(line[0])}_{int(line[1])}'] = trans
        return result

class ThreeDLOMatchTest(data.Dataset):
    def __init__(self, 
            root, 
            descriptor='fcgf',
            in_dim=6,
            inlier_threshold=0.10,
            num_node=5000, 
            use_mutual=True,
            downsample=0.03, 
            augment_axis=0, 
            augment_rotation=1.0,
            augment_translation=0.01,
            select_scene=None,
            ):
        self.root = root
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

        with open('misc/3DLoMatch.pkl', 'rb') as f:
            self.infos = pickle.load(f)
    
    def __len__(self):
        return len(self.infos['rot'])

    def __getitem__(self,item): 
        # get meta data
        gt_trans = integrate_trans(self.infos['rot'][item], self.infos['trans'][item])  
        scene = self.infos['src'][item].split('/')[1]
        src_id = self.infos['src'][item].split('/')[-1].split('_')[-1].replace('.pth', '')
        tgt_id = self.infos['tgt'][item].split('/')[-1].split('_')[-1].replace('.pth', '')

        # load point coordinates and pre-computed per-point local descriptors
        if self.descriptor == 'fcgf':
            src_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{src_id}_fcgf.npz")
            tgt_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{tgt_id}_fcgf.npz")
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
        elif self.descriptor == 'fpfh':
            src_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{src_id}_fpfh.npz")
            tgt_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{tgt_id}_fpfh.npz")
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # select {self.num_node} numbers of keypoints
        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]
        # use all point during test.
        if self.num_node == 'all':
            src_sel_ind = np.arange(N_src)
            tgt_sel_ind = np.arange(N_tgt)
        else:
            src_sel_ind = np.random.choice(N_src, self.num_node)
            tgt_sel_ind = np.random.choice(N_tgt, self.num_node)
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
        
        # build the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
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
            labels.astype(np.float32),


if __name__ == "__main__":
    dset = ThreeDMatchTrainVal(root='/data/3DMatch', 
                       split='train',   
                       descriptor='fcgf',
                       num_node='all', 
                       use_mutual=False,
                       augment_axis=0, 
                       augment_rotation=0, 
                       augment_translation=0.00
                       )
    print(len(dset))  
    for i in range(dset.__len__()):
        ret_dict = dset[i]
