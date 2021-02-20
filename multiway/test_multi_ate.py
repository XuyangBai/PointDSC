import json
import os
import sys
import argparse
import logging
import torch
import numpy as np
import importlib
import open3d as o3d
from tqdm import tqdm
from easydict import EasyDict as edict
from libs.loss import TransformationLoss, ClassificationLoss
from datasets.Redwood import RedwoodDataset
from datasets.dataloader import get_dataloader
from utils.pointcloud import make_point_cloud
from evaluation.benchmark_utils import set_seed, icp_refine
from models.common import rigid_transform_3d
from utils.SE3 import *
set_seed()


class MatchingResult:
    def __init__(self, s, t, trans):
        self.s = s
        self.t = t
        self.success = False
        self.transformation = trans
        self.infomation = np.identity(6)


def align(model, data):
    """
    Align two trajectory 
    Inputs:
        - model: [num_frag, 3] source trajectory
        - data:  [num_frag, 3] traget trajectory
    Outputs
        - trans: [4, 4] the transformation to align two trajectories
        - trans_error [num_frag, 1] the trajectory error for each fragment
        - model_aligned []
    """
    model = torch.from_numpy(model).float().T[None]
    data = torch.from_numpy(data).float().T[None]
    trans = rigid_transform_3d(model, data)

    aligned_model = transform(model, trans)
    aligned_error = aligned_model - data
    trans_error = torch.norm(aligned_error, dim=-1)[0]

    trans_error = trans_error.detach().cpu().numpy() * 100 # m -> cm
    return trans[0], trans_error


def multi_scale_icp(src_pcd, tgt_pcd, voxel_size, max_iter, trans):
    current_trans = trans
    for i, scale in enumerate(range(len(max_iter))):
        iter = max_iter[scale]
        source_down = src_pcd.voxel_down_sample(voxel_size[scale])
        target_down = tgt_pcd.voxel_down_sample(voxel_size[scale])
        distance_threshold = 0.05 * 1.4
        result_icp = o3d.registration.registration_icp(
            source_down, target_down, distance_threshold,
            current_trans,
            o3d.registration.TransformationEstimationPointToPoint(False),
            o3d.registration.ICPConvergenceCriteria(max_iteration=iter)
        )
        current_trans = result_icp.transformation
        if i == len(max_iter) - 1:
            information_matrix = o3d.registration.get_information_matrix_from_point_clouds(
                source_down, target_down, voxel_size[scale] * 1.4,
                result_icp.transformation
            )
    return result_icp.transformation, information_matrix


def local_refinement(src, tgt, trans):
    src_pcd, tgt_pcd = make_point_cloud(src), make_point_cloud(tgt)

    trans, info = multi_scale_icp(src_pcd, tgt_pcd,
                                  voxel_size=[0.05, 0.05 / 2.0, 0.05 / 4.0],
                                  max_iter=[50, 30, 14],
                                  trans=trans)
    return trans, info


def eval_redwood_scene(model, dloader, config, posegraph_name, use_icp):
    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))

    orig_points_dict = {}
    num_pair = dloader.dataset.__len__()
    dloader_iter = dloader.__iter__()
    num_pcd = dloader.dataset.num_pcds
    #################################
    # pairwise registration between each pair of fragments
    #################################
    with torch.no_grad():
        for _ in tqdm(range(num_pair)):
            #################################
            # load data 
            #################################
            corr, src_keypts, tgt_keypts, gt_trans, gt_labels, key = dloader_iter.next()
            corr, src_keypts, tgt_keypts, gt_trans, gt_labels = \
                corr.cuda(), src_keypts.cuda(), tgt_keypts.cuda(), gt_trans.cuda(), gt_labels.cuda()
            data = {
                'corr_pos': corr,
                'src_keypts': src_keypts,
                'tgt_keypts': tgt_keypts,
                'testing': True,
            }
            source_id, target_id = int(key[0].split('@')[1].split('_')[0]), int(key[0].split('@')[1].split('_')[1])

            #################################
            # forward pass 
            #################################
            if target_id == source_id + 1:  # odometry case
                scene = dloader.dataset.scene_list[0]
                pose_graph_frag = o3d.io.read_pose_graph(os.path.join(f"{config.root}/{scene}/", "fragments/fragment_optimized_%03d.json" % source_id))
                n_nodes = len(pose_graph_frag.nodes)
                transformation_init = np.linalg.inv(pose_graph_frag.nodes[n_nodes - 1].pose)
                transformation, information = local_refinement(
                    # src_keypts[0].detach().cpu().numpy(),
                    # tgt_keypts[0].detach().cpu().numpy(),
                    np.load(f"{config.root}/{scene}/fragments/fragment_{str(source_id).zfill(3)}_fpfh.npz")['xyz'],
                    np.load(f"{config.root}/{scene}/fragments/fragment_{str(target_id).zfill(3)}_fpfh.npz")['xyz'],
                    transformation_init
                )
                odometry = np.dot(transformation, odometry)
                pose_graph.nodes.append(o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(o3d.registration.PoseGraphEdge(source_id,
                                                                       target_id,
                                                                       transformation,
                                                                       information,
                                                                       uncertain=False))
            else:
                res = model(data)
                pred_trans, pred_labels = res['final_trans'], res['final_labels']

                transformation = pred_trans.detach().cpu().numpy()[0]
                information = o3d.registration.get_information_matrix_from_point_clouds(
                    make_point_cloud(src_keypts[0].detach().cpu().numpy()),
                    make_point_cloud(tgt_keypts[0].detach().cpu().numpy()),
                    max_correspondence_distance=0.05 * 1.4,
                    transformation=transformation
                )
                if information[5, 5] / min(src_keypts.shape[1], tgt_keypts.shape[1]) < 0.30 or transformation.trace() == 4.0:
                    # consider the pair has too small overlapping
                    continue
                pose_graph.edges.append(o3d.registration.PoseGraphEdge(source_id,
                                                                       target_id,
                                                                       transformation,
                                                                       information,
                                                                       uncertain=True))
            src_keypts = np.load(f"{config.root}/{scene}/fragments/fragment_{str(source_id).zfill(3)}_fpfh.npz")['xyz']
            tgt_keypts = np.load(f"{config.root}/{scene}/fragments/fragment_{str(target_id).zfill(3)}_fpfh.npz")['xyz']
            orig_points_dict[f'{source_id}_{target_id}'] = [src_keypts, tgt_keypts]

    o3d.io.write_pose_graph(posegraph_name + '_0.json', pose_graph)

    #################################
    # pose graph optimization for pruning false-positive loop closure
    #################################
    print(f"Before optimization {len(pose_graph.nodes)} nodes {len(pose_graph.edges)} edges")
    print("Optimizing PoseGraph ...")
    option = o3d.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.05 * 1.4,
        edge_prune_threshold=0.25,
        preference_loop_closure=20.0,
        reference_node=0)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    o3d.registration.global_optimization(
        pose_graph, o3d.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.registration.GlobalOptimizationConvergenceCriteria(), option)
    print(f"After optimization {len(pose_graph.nodes)} nodes {len(pose_graph.edges)} edges")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    o3d.io.write_pose_graph(posegraph_name + '_1.json', pose_graph)
    if use_icp is False:
        return pose_graph

    #################################
    # Refine each edge with ICP, and second time pose graph optimization for further refinement
    #################################    
    print("Refine each edge with ICP ...")
    # save the matching_result for survived pairs
    matching_results = {}
    for edge in pose_graph.edges:
        s = edge.source_node_id
        e = edge.target_node_id
        matching_results[f'{s}_{e}'] = MatchingResult(s, e, edge.transformation)

    # use icp to refine the transformation for each survived pairs
    pose_graph_new = o3d.registration.PoseGraph()
    odometry = np.eye(4)
    pose_graph_new.nodes.append(o3d.registration.PoseGraphNode(odometry))
    for k, result in matching_results.items():
        src_keypts, tgt_keypts = orig_points_dict[k]
        source_id, target_id = int(k.split('_')[0]), int(k.split('_')[1])
        transformation, information = local_refinement(src_keypts, tgt_keypts, result.transformation)
        if target_id == source_id + 1:
            odometry = np.dot(transformation, odometry)
            pose_graph_new.nodes.append(o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
            pose_graph_new.edges.append(o3d.registration.PoseGraphEdge(source_id,
                                                                       target_id,
                                                                       transformation,
                                                                       information,
                                                                       uncertain=False))
        else:
            pose_graph_new.edges.append(o3d.registration.PoseGraphEdge(source_id,
                                                                       target_id,
                                                                       transformation,
                                                                       information,
                                                                       uncertain=True))
    print(f"Before optimization {len(pose_graph_new.nodes)} nodes {len(pose_graph_new.edges)} edges")
    print("Optimizing PoseGraph ...")
    option = o3d.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.05 * 1.4,
        edge_prune_threshold=0.25,
        preference_loop_closure=20.0,
        reference_node=0)
    o3d.registration.global_optimization(
        pose_graph_new, o3d.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.registration.GlobalOptimizationConvergenceCriteria(), option)
    print(f"After optimization {len(pose_graph_new.nodes)} nodes {len(pose_graph_new.edges)} edges")
    o3d.io.write_pose_graph(posegraph_name + '_2.json', pose_graph_new)
    return pose_graph_new


def eval_redwood(model, config, args):
    scene_list = [
        'livingroom1-simulated',
        'livingroom2-simulated',
        'office1-simulated',
        'office2-simulated'
    ]

    scene_result_list = []
    for scene_ind, scene in enumerate(scene_list):
        dset = RedwoodDataset(root=config.root,
                              descriptor=config.descriptor,
                              min_overlap=0.0,  # do not have overlap prior.
                              in_dim=6,
                              inlier_threshold=0.10,
                              num_node=20000,
                              use_mutual=False,
                              augment_axis=0,
                              augment_rotation=0.00,
                              augment_translation=0.0,
                              select_scene=scene,
                              )
        posegraph_name = f"multiway/{scene}_{config.descriptor}"
        posegraph_filename = posegraph_name + '_2.json'
        if os.path.exists(posegraph_filename):
            pose_graph = o3d.io.read_pose_graph(posegraph_filename)
            print(f"Load pose graph from {posegraph_filename}, Pose graph has {len(pose_graph.nodes)} nodes {len(pose_graph.edges)} edges")
        else:
            dloader = torch.utils.data.DataLoader(
                dset,
                batch_size=1,
                shuffle=False,
                num_workers=16,
            )
            pose_graph = eval_redwood_scene(model, dloader, config, posegraph_name, args.use_icp)

        assert len(pose_graph.nodes) == dset.num_pcds

        X, Y, Z = [], [], []
        _X, _Y, _Z = [], [], []
        start = np.array([0, 0, 0, 1]).T
        relative_trans = np.eye(4)
        for i in range(len(dset.gt_trajectory)):
            gt_loc = dset.gt_trajectory[i] @ relative_trans @ start
            est_loc = pose_graph.nodes[i].pose @ start
            assert gt_loc[-1] == 1 and est_loc[-1] == 1
            X.append(gt_loc[0])
            Y.append(gt_loc[1])
            Z.append(gt_loc[2])
            _X.append(est_loc[0])
            _Y.append(est_loc[1])
            _Z.append(est_loc[2])
        src = np.array([X, Y, Z])
        tgt = np.array([_X, _Y, _Z])
        # traj = np.array([src, tgt])
        # np.save('traj.npy', traj)
        trans, trans_error = align(src, tgt)
        ate_rmse = np.sqrt(np.mean(trans_error**2))
        # print(rmse)
        print(f"Mean Absolute Trajectory Error: {ate_rmse:.2f}cm")
        scene_result_list.append(ate_rmse)
        
        # make a combined point cloud and visualize
        if args.visualize:
            pcd_combined = o3d.geometry.PointCloud()
            for i in range(len(pose_graph.nodes)):
                pcd_path = f"{config.root}/{scene}/fragments/fragment_{str(i).zfill(3)}.ply"
                pcd = o3d.io.read_point_cloud(pcd_path)
                # pcd = pcd.voxel_down_sample(voxel_size=0.025)
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.025 * 2, max_nn=30))
                pose = np.load(f"{config.root}/{scene}/fragments/fragment_{str(i).zfill(3)}.npy")
                pcd.transform(np.linalg.inv(pose))

                pcd.transform(pose_graph.nodes[i].pose)
                pcd_combined += pcd
            pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.01)
            o3d.io.write_point_cloud(f'{scene}.ply', pcd_combined_down)
            o3d.visualization.draw_geometries(pcd_combined_down)

    print(f"All 4 scene ATE(cm): {scene_result_list}")
    print(f"Mean ATE(cm): {np.mean(scene_result_list):.2f}cm")


if __name__ == '__main__':
    from config import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='', type=str, required=True, help='snapshot dir')
    parser.add_argument('--solver', default='SVD', type=str, choices=['SVD', 'RANSAC'])
    parser.add_argument('--use_icp', default=True, type=str2bool)
    parser.add_argument('--save_npy', default=False, type=str2bool)
    parser.add_argument('--visualize', default=False, type=str2bool)
    args = parser.parse_args()

    config_path = f'snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)
    config.root = '/data/Augmented_ICL-NUIM'
    config.descriptor = 'fpfh'

    from models.PointSM import PointSM

    model = PointSM(
        in_dim=config.in_dim,
        num_layers=config.num_layers,
        num_channels=config.num_channels,
        num_iterations=config.num_iterations,
        ratio=config.ratio,
        sigma_d=config.sigma_d,
        k=config.k,
        nms_radius=config.inlier_threshold,
    )
    miss = model.load_state_dict(torch.load(f'snapshot/{args.chosen_snapshot}/models/model_best.pkl'), strict=False)
    print(miss)
    model.eval()

    eval_redwood(model.cuda(), config, args)
