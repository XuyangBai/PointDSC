import json
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
from datasets.ThreeDMatch import ThreeDLOMatchTest
from datasets.dataloader import get_dataloader
from utils.pointcloud import make_point_cloud
from evaluation.benchmark_utils import set_seed
from evaluation.benchmark_utils_predator import *
from utils.timer import Timer
from utils.SE3 import *
set_seed()


def get_predator_data(pair_idx, n_points):
    data_dict = torch.load(f'/data/OverlapPredator/snapshot/predator_3dmatch/3DLoMatch/{pair_idx}.pth')
    len_src = data_dict['len_src']
    src_pcd = data_dict['pcd'][:len_src, :].cuda()
    tgt_pcd = data_dict['pcd'][len_src:, :].cuda()
    src_feats = data_dict['feats'][:len_src].cuda()
    tgt_feats = data_dict['feats'][len_src:].cuda()
    saliency, overlap =  data_dict['saliency'], data_dict['overlaps']
    src_overlap, src_saliency = overlap[:len_src], saliency[:len_src]
    tgt_overlap, tgt_saliency = overlap[len_src:], saliency[len_src:]
    src_scores = src_overlap * src_saliency
    tgt_scores = tgt_overlap * tgt_saliency
    if(src_pcd.size(0) > n_points):
        idx = np.arange(src_pcd.size(0))
        probs = (src_scores / src_scores.sum()).numpy().flatten()
        idx = np.random.choice(idx, size= n_points, replace=False, p=probs)
        src_pcd, src_feats = src_pcd[idx], src_feats[idx]
    if(tgt_pcd.size(0) > n_points):
        idx = np.arange(tgt_pcd.size(0))
        probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
        idx = np.random.choice(idx, size= n_points, replace=False, p=probs)
        tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

    dists = torch.einsum('ac,bc->ab', src_feats, tgt_feats)
    source_idx = torch.argmax(dists, dim=-1)
    corr_pos = torch.cat([src_pcd[None], tgt_pcd[source_idx][None]], dim=-1)
    corr_pos = corr_pos - corr_pos.mean(1, keepdims=True)
    data = {
        'corr_pos': corr_pos,
        'src_keypts': src_pcd[None],
        'tgt_keypts': tgt_pcd[source_idx][None],
        'testing': True,
    }
    
    gt_trans = integrate_trans(data_dict['rot'], data_dict['trans']).cuda()  
    warped_src_pcd = transform(src_pcd, gt_trans)
    distance = torch.norm(warped_src_pcd - tgt_pcd[source_idx], dim=-1)
    gt_labels = (distance < 0.10).float()
    return data, gt_trans[None], gt_labels[None]


def eval_3DMatch_scene(model, scene_ind, dloader, config, args):
    num_pair = dloader.dataset.__len__()
    # 0.success, 1.RE, 2.TE, 3.input inlier number, 4.input inlier ratio,  5. output inlier number 
    # 6. output inlier precision, 7. output inlier recall, 8. output inlier F1 score 9. model_time, 10. data_time 11. scene_ind
    stats = np.zeros([num_pair, 12])
    final_poses = np.zeros([num_pair, 4, 4])
    dloader_iter = dloader.__iter__()
    class_loss = ClassificationLoss()
    evaluate_metric = TransformationLoss(re_thre=config.re_thre, te_thre=config.te_thre)
    data_timer, model_timer = Timer(), Timer()
    with torch.no_grad():
        for i in tqdm(range(num_pair)):
            #################################
            # load data 
            #################################
            data_timer.tic()
            if args.descriptor == 'fcgf':
                # using FCGF 5cm to build the initial correspondence
                corr, src_keypts, tgt_keypts, gt_trans, gt_labels = dloader_iter.next()
                corr, src_keypts, tgt_keypts, gt_trans, gt_labels = \
                    corr.cuda(), src_keypts.cuda(), tgt_keypts.cuda(), gt_trans.cuda(), gt_labels.cuda()
                data = {
                    'corr_pos': corr,
                    'src_keypts': src_keypts,
                    'tgt_keypts': tgt_keypts,
                    'testing': True,
                }
            elif args.descriptor == 'predator':
                # use Predator to build the inital correspondence 
                data, gt_trans, gt_labels = get_predator_data(i, args.num_points)
                src_keypts, tgt_keypts = data['src_keypts'], data['tgt_keypts']

            data_time = data_timer.toc()

            #################################
            # forward pass 
            #################################
            model_timer.tic()
            res = model(data)
            pred_trans, pred_labels = res['final_trans'], res['final_labels']

            # evaluate raw FCGF + ransac   
            # src_pcd = make_point_cloud(src_keypts[0].detach().cpu().numpy())
            # tgt_pcd = make_point_cloud(tgt_keypts[0].detach().cpu().numpy())
            # correspondence = np.array([np.arange(src_keypts.shape[1]), np.arange(src_keypts.shape[1])])
            # correspondence = o3d.utility.Vector2iVector(correspondence.T)
            # reg_result = o3d.registration.registration_ransac_based_on_correspondence(
            #     src_pcd, tgt_pcd, correspondence,
            #     max_correspondence_distance=config.inlier_threshold,
            #     estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
            #     ransac_n=3,
            #     criteria=o3d.registration.RANSACConvergenceCriteria(max_iteration=50000, max_validation=1000)
            # )
            # pred_trans = torch.eye(4)[None].to(gt_trans.device)
            # pred_trans[0, :4, :4] = torch.from_numpy(reg_result.transformation)
            # pred_labels = torch.zeros_like(gt_labels)
            # pred_labels[0, np.array(reg_result.correspondence_set)[:, 0]] = 1

            model_time = model_timer.toc()
            class_stats = class_loss(pred_labels, gt_labels)
            loss, recall, Re, Te, rmse = evaluate_metric(pred_trans, gt_trans, src_keypts, tgt_keypts, pred_labels)

            #################################
            # record the evaluation results.
            #################################
            # save statistics
            stats[i, 0] = float(recall / 100.0)                      # success
            stats[i, 1] = float(Re)                                  # Re (deg)
            stats[i, 2] = float(Te)                                  # Te (cm)
            stats[i, 3] = int(torch.sum(gt_labels))                  # input inlier number
            stats[i, 4] = float(torch.mean(gt_labels.float()))       # input inlier ratio
            stats[i, 5] = int(torch.sum(gt_labels[pred_labels > 0])) # output inlier number
            stats[i, 6] = float(class_stats['precision'])            # output inlier precision
            stats[i, 7] = float(class_stats['recall'])               # output inlier recall
            stats[i, 8] = float(class_stats['f1'])                   # output inlier f1 score
            stats[i, 9] = model_time
            stats[i, 10] = data_time
            stats[i, 11] = scene_ind

            final_poses[i] = pred_trans[0].detach().cpu().numpy()

    return stats, final_poses


def eval_3DMatch(model, config, args):
    dset = ThreeDLOMatchTest(root='/data/3DMatch',
                            descriptor='fcgf',
                            in_dim=config.in_dim,
                            inlier_threshold=config.inlier_threshold,
                            num_node=args.num_points,
                            use_mutual=config.use_mutual,
                            augment_axis=0,
                            augment_rotation=0.00,
                            augment_translation=0.0,
                            )
    dloader = get_dataloader(dset, batch_size=1, num_workers=16, shuffle=False)
    allpair_stats, allpair_poses = eval_3DMatch_scene(model, 0, dloader, config, args)
    
    # benchmarking using the registration recall defined in 3DMatch to compare with Predator
    # np.save('predator.npy', allpair_poses)
    benchmark_predator(allpair_poses, gt_folder='/data/OverlapPredator/configs/benchmarks/3DLoMatch')
    
    # benchmarking using the registration recall defined in DGR 
    allpair_average = allpair_stats.mean(0)
    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0)
    logging.info(f"*" * 40)
    logging.info(f"All {allpair_stats.shape[0]} pairs, Mean Success Rate={allpair_average[0] * 100:.2f}%, Mean Re={correct_pair_average[1]:.2f}, Mean Te={correct_pair_average[2]:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={allpair_average[3]:.2f}(ratio={allpair_average[4] * 100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={allpair_average[5]:.2f}(precision={allpair_average[6] * 100:.2f}%, recall={allpair_average[7] * 100:.2f}%, f1={allpair_average[8] * 100:.2f}%)")
    logging.info(f"\tMean model time: {allpair_average[9]:.2f}s, Mean data time: {allpair_average[10]:.2f}s")

    return allpair_stats


def benchmark_predator(pred_poses, gt_folder):
    scenes = sorted(os.listdir(gt_folder))
    scene_names = [os.path.join(gt_folder,ele) for ele in scenes]

    re_per_scene = defaultdict(list)
    te_per_scene = defaultdict(list)
    re_all, te_all, precision, recall = [], [], [], []
    n_valids= []

    short_names=['Kitchen','Home 1','Home 2','Hotel 1','Hotel 2','Hotel 3','Study','MIT Lab']
    logging.info(("Scene\t¦ prec.\t¦ rec.\t¦ re\t¦ te\t¦ samples\t¦"))
    
    start_ind = 0
    for idx,scene in enumerate(scene_names):
        # ground truth info
        gt_pairs, gt_traj = read_trajectory(os.path.join(scene, "gt.log"))
        n_valid=0
        for ele in gt_pairs:
            diff=abs(int(ele[0])-int(ele[1]))
            n_valid+=diff>1
        n_valids.append(n_valid)

        n_fragments, gt_traj_cov = read_trajectory_info(os.path.join(scene,"gt.info"))

        # estimated info
        # est_pairs, est_traj = read_trajectory(os.path.join(est_folder,scenes[idx],'est.log'))
        est_traj = pred_poses[start_ind:start_ind + len(gt_pairs)]
        start_ind = start_ind + len(gt_pairs)

        temp_precision, temp_recall,c_flag = evaluate_registration(n_fragments, est_traj, gt_pairs, gt_pairs, gt_traj, gt_traj_cov)
        
        # Filter out the estimated rotation matrices
        ext_gt_traj = extract_corresponding_trajectors(gt_pairs,gt_pairs, gt_traj)

        re = rotation_error(torch.from_numpy(ext_gt_traj[:,0:3,0:3]), torch.from_numpy(est_traj[:,0:3,0:3])).cpu().numpy()[np.array(c_flag)==0]
        te = translation_error(torch.from_numpy(ext_gt_traj[:,0:3,3:4]), torch.from_numpy(est_traj[:,0:3,3:4])).cpu().numpy()[np.array(c_flag)==0]

        re_per_scene['mean'].append(np.mean(re))
        re_per_scene['median'].append(np.median(re))
        re_per_scene['min'].append(np.min(re))
        re_per_scene['max'].append(np.max(re))
        
        te_per_scene['mean'].append(np.mean(te))
        te_per_scene['median'].append(np.median(te))
        te_per_scene['min'].append(np.min(te))
        te_per_scene['max'].append(np.max(te))


        re_all.extend(re.reshape(-1).tolist())
        te_all.extend(te.reshape(-1).tolist())

        precision.append(temp_precision)
        recall.append(temp_recall)

        logging.info("{}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:3d}¦".format(short_names[idx], temp_precision, temp_recall, np.median(re), np.median(te), n_valid))
        # np.save(f'{est_folder}/{scenes[idx]}/flag.npy',c_flag)
    
    weighted_precision = (np.array(n_valids) * np.array(precision)).sum() / np.sum(n_valids)

    logging.info("Mean precision: {:.3f}: +- {:.3f}".format(np.mean(precision),np.std(precision)))
    logging.info("Weighted precision: {:.3f}".format(weighted_precision))

    logging.info("Mean median RRE: {:.3f}: +- {:.3f}".format(np.mean(re_per_scene['median']), np.std(re_per_scene['median'])))
    logging.info("Mean median RTE: {:.3F}: +- {:.3f}".format(np.mean(te_per_scene['median']),np.std(te_per_scene['median'])))
    

if __name__ == '__main__':
    from config import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='', type=str, help='snapshot dir')
    parser.add_argument('--descriptor', default='fcgf', type=str, choices=['fcgf', 'predator'])
    parser.add_argument('--num_points', default=5000, type=int)
    parser.add_argument('--use_icp', default=False, type=str2bool)
    parser.add_argument('--save_npy', default=False, type=str2bool)
    args = parser.parse_args()

    config_path = f'snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    log_filename = f'logs/3DLoMatch_{args.chosen_snapshot}-{args.descriptor}-{args.num_points}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='a',
                        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))   

    ## load the model from models/PointDSC.py
    from models.PointDSC import PointDSC

    model = PointDSC(
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

    # evaluate on the test set
    stats = eval_3DMatch(model.cuda(), config, args)

    if args.save_npy:
        save_path = log_filename.replace('.log', '.npy')
        np.save(save_path, stats)
        print(f"Save the stats in {save_path}")
