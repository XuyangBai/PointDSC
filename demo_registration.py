import json
import copy
import argparse
from easydict import EasyDict as edict
from models.PointDSC import PointDSC
from utils.pointcloud import estimate_normal
import torch
import numpy as np
import open3d as o3d 

def extract_fcgf_features(pcd_path, downsample, device, weight_path='misc/ResUNetBN2C-feat32-3dmatch-v0.05.pth'):
    raw_src_pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.array(raw_src_pcd.points)
    from misc.fcgf import ResUNetBN2C as FCGF
    from misc.cal_fcgf import extract_features
    fcgf_model = FCGF(
        1,
        32,
        bn_momentum=0.05,
        conv1_kernel_size=7,
        normalize_feature=True
    ).to(device)
    checkpoint = torch.load(weight_path)
    fcgf_model.load_state_dict(checkpoint['state_dict'])
    fcgf_model.eval()

    xyz_down, features = extract_features(
        fcgf_model,
        xyz=pts,
        rgb=None,
        normal=None,
        voxel_size=downsample,
        skip_check=True,
    )
    return raw_src_pcd, xyz_down.astype(np.float32), features.detach().cpu().numpy()

def extract_fpfh_features(pcd_path, downsample, device):
    raw_src_pcd = o3d.io.read_point_cloud(pcd_path)
    estimate_normal(raw_src_pcd, radius=downsample*2)
    src_pcd = raw_src_pcd.voxel_down_sample(downsample)
    src_features = o3d.registration.compute_fpfh_feature(src_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=downsample * 5, max_nn=100))
    src_features = np.array(src_features.data).T
    src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
    return raw_src_pcd, np.array(src_pcd.points), src_features

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not source_temp.has_normals():
        estimate_normal(source_temp)
        estimate_normal(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == '__main__':
    from config import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='PointDSC_3DMatch_release', type=str, help='snapshot dir')
    parser.add_argument('--pcd1', default='demo_data/cloud_bin_0.ply', type=str)
    parser.add_argument('--pcd2', default='demo_data/cloud_bin_1.ply', type=str)
    parser.add_argument('--descriptor', default='fcgf', type=str, choices=['fcgf', 'fpfh'])
    parser.add_argument('--use_gpu', default=True, type=str2bool)
    args = parser.parse_args()

    config_path = f'snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    if args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = PointDSC(
        in_dim=config.in_dim,
        num_layers=config.num_layers,
        num_channels=config.num_channels,
        num_iterations=config.num_iterations,
        ratio=config.ratio,
        sigma_d=config.sigma_d,
        k=config.k,
        nms_radius=config.inlier_threshold,
    ).to(device)
    miss = model.load_state_dict(torch.load(f'snapshot/{args.chosen_snapshot}/models/model_best.pkl', map_location=device), strict=False)
    print(miss)
    model.eval()

    # extract features
    if args.descriptor == 'fpfh':
        raw_src_pcd, src_pts, src_features = extract_fpfh_features(args.pcd1, config.downsample, device)
        raw_tgt_pcd, tgt_pts, tgt_features = extract_fpfh_features(args.pcd2, config.downsample, device)
    else:
        raw_src_pcd, src_pts, src_features = extract_fcgf_features(args.pcd1, config.downsample, device)
        raw_tgt_pcd, tgt_pts, tgt_features = extract_fcgf_features(args.pcd2, config.downsample, device)

    # matching
    distance = np.sqrt(2 - 2 * (src_features @ tgt_features.T) + 1e-6)
    source_idx = np.argmin(distance, axis=1)
    source_dis = np.min(distance, axis=1)
    corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
    src_keypts = src_pts[corr[:,0]]
    tgt_keypts = tgt_pts[corr[:,1]]
    corr_pos = np.concatenate([src_keypts, tgt_keypts], axis=-1)
    corr_pos = corr_pos - corr_pos.mean(0)

    # outlier rejection
    data = {
            'corr_pos': torch.from_numpy(corr_pos)[None].to(device).float(),
            'src_keypts': torch.from_numpy(src_keypts)[None].to(device).float(),
            'tgt_keypts': torch.from_numpy(tgt_keypts)[None].to(device).float(),
            'testing': True,
            }
    res = model(data)

   # First plot the original state of the point clouds
    draw_registration_result(raw_src_pcd, raw_tgt_pcd, np.identity(4))

    # Plot point clouds after registration
    draw_registration_result(raw_src_pcd, raw_tgt_pcd, res['final_trans'][0].detach().cpu().numpy())