import torch
import math
import sys
import argparse
import logging
import numpy as np
from tqdm import tqdm
from models.common import rigid_transform_3d
from libs.loss import TransformationLoss, ClassificationLoss
from datasets.ThreeDMatch import ThreeDMatchTest
from datasets.dataloader import get_dataloader
from utils.timer import Timer
from utils.pointcloud import make_point_cloud
import open3d as o3d
from config import str2bool
import pdb


def SM(corr, src_keypts, tgt_keypts, args, top_ratio=0.1):
    diff = corr - corr.permute(1, 0, 2)
    M = torch.sum(diff[:, :, 0:3] ** 2, dim=-1) ** 0.5 - torch.sum(diff[:, :, 3:6] ** 2, dim=-1) ** 0.5
    M = M[None, :, :]

    ## exponetial function
    # M = torch.max(torch.zeros_like(M), torch.exp(-torch.abs(M)))

    ## binary function
    # thre = 0.10
    # binary_M = torch.ones_like(M) # use 0.10 as the threshold, binarize the M
    # binary_M[torch.abs(M) < thre] = 1
    # binary_M[torch.abs(M) >= thre] = 0
    # M = binary_M

    ## polynomial funcition
    sigma = args.inlier_threshold / 3
    M = torch.max(torch.zeros_like(M), 4.5 - M ** 2 / 2 / sigma ** 2)
    M[:, torch.arange(M.shape[1]), torch.arange(M.shape[1])] = 0

    # calculate the principal eigenvector
    leading_eig = torch.ones_like(M[:, :, 0:1])
    for _ in range(10):
        leading_eig = torch.bmm(M, leading_eig)
        leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
    leading_eig = leading_eig.squeeze(-1)

    # select top-10% as the inliers
    top_10p = torch.argsort(leading_eig, dim=1, descending=True)[:, 0:int(leading_eig.shape[1] * top_ratio)]
    pred_labels = torch.zeros_like(leading_eig)
    pred_labels[0, top_10p[0]] = 1 # assert bs = 1

    # compute the transformation
    pred_trans = rigid_transform_3d(src_keypts, tgt_keypts, leading_eig * pred_labels)
    return pred_trans, pred_labels


def PMC(corr, src_keypts, tgt_keypts, args):
    # max clique algorithm https://github.com/ryanrossi/pmc
    from utils.max_clique import pmc
    
    # construct the edge matrix
    num_nodes = corr.shape[1]
    corr = corr[0].detach().cpu().numpy()
    edges = []
    for ind_1 in range(num_nodes):
        for ind_2 in range(0, ind_1):
            diff = np.sum((corr[ind_1][0:3] - corr[ind_2][0:3]) ** 2) - np.sum((corr[ind_1][3:] - corr[ind_2][3:]) ** 2)
            if np.abs(diff) < args.inlier_threshold:
                edges.append((ind_1, ind_2))
    edges = np.array(edges)
    max_clique = pmc(edges[:, 0], edges[:, 1], num_nodes, len(edges))
    
    pred_labels = torch.zeros_like(src_keypts[:, :, 0])
    pred_labels[0, max_clique] = 1 # assert bs = 1

    # compute the transformation
    pred_trans = rigid_transform_3d(src_keypts, tgt_keypts, pred_labels)
    return pred_trans, pred_labels


def RANSAC(corr, src_keypts, tgt_keypts, args):
    src_pcd = make_point_cloud(src_keypts[0])
    tgt_pcd = make_point_cloud(tgt_keypts[0])
    corr = np.array([np.arange(src_keypts.shape[1]), np.arange(src_keypts.shape[1])])
    corr = o3d.utility.Vector2iVector(corr.T)
    reg_result = o3d.registration.registration_ransac_based_on_correspondence(
        src_pcd, tgt_pcd, corr,
        max_correspondence_distance=args.inlier_threshold,
        estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        # for ransac_based_on_correspondence max_iteration and max_validation are the same
        criteria=o3d.registration.RANSACConvergenceCriteria(max_iteration=args.max_iteration, max_validation=args.max_validation)
    )
    inliers = np.array(reg_result.correspondence_set)
    pred_labels = torch.zeros_like(src_keypts[:, :, 0])
    pred_labels[0, inliers[:, 0]] = 1
    pred_trans = torch.eye(4)[None].to(src_keypts.device)
    pred_trans[:, :4, :4] = torch.from_numpy(reg_result.transformation)
    return pred_trans, pred_labels


def GCRANSAC(corr, src_keypts, tgt_keypts, args):
    # https://github.com/danini/graph-cut-ransac
    import pygcransac
    src_pts = src_keypts[0].detach().cpu().numpy()
    tgt_pts = tgt_keypts[0].detach().cpu().numpy()

    pose, mask = pygcransac.findRigidTransform(
        src_pts,
        tgt_pts,
        threshold=args.inlier_threshold,
        conf=0.99999999,
        spatial_coherence_weight=0.1, # default is 0.975
        max_iters=args.max_iteration,
        use_sprt=True,
        min_inlier_ratio_for_sprt=0.1
    )  
    if mask.sum() == 0:
        pose = np.eye(4)
    pose = pose.T # diffenet convention 
    pred_trans  = torch.eye(4)[None].to(src_keypts.device)
    pred_trans[:, :4, :4] = torch.from_numpy(pose)
    pred_labels = torch.from_numpy(mask)[None].float().to(src_keypts.device)
    return pred_trans, pred_labels


def eval_3DMatch_scene(method, scene, scene_ind, dloader, args):
    """
    Evaluate baseline methods on 3DMatch testset [scene]
    """
    num_pair = dloader.dataset.__len__()
    # 0.success, 1.RE, 2.TE, 3.input inlier number, 4.input inlier ratio,  5. output inlier number
    # 6. output inlier precision, 7. output inlier recall, 8. output inlier F1 score 9. model_time, 10. data_time 11. scene_ind
    stats = np.zeros([num_pair, 12])
    dloader_iter = dloader.__iter__()
    class_loss = ClassificationLoss()
    evaluation_metric = TransformationLoss(re_thre=15, te_thre=30)
    data_timer, model_timer = Timer(), Timer()
    with torch.no_grad():
        for i in tqdm(range(num_pair)):
            #################################
            # load data 
            #################################
            data_timer.tic()    
            corr, src_keypts, tgt_keypts, gt_trans, gt_labels = dloader_iter.next()
            corr, src_keypts, tgt_keypts, gt_trans, gt_labels = \
                corr.cuda(), src_keypts.cuda(), tgt_keypts.cuda(), gt_trans.cuda(), gt_labels.cuda()
            data = {
                'corr_pos': corr,
                'src_keypts': src_keypts,
                'tgt_keypts': tgt_keypts,
                'testing': True,
            }
            data_time = data_timer.toc()

            #################################
            # forward pass 
            #################################
            model_timer.tic()

            if method == 'SM':
                pred_trans, pred_labels = SM(corr, src_keypts, tgt_keypts, args)

            elif method == 'PMC':
                pred_trans, pred_labels = PMC(corr, src_keypts, tgt_keypts, args)

            elif method == 'RANSAC':
                pred_trans, pred_labels = RANSAC(corr, src_keypts, tgt_keypts, args)

            elif method == 'MAGSAC':
                import pymagsac
                # No rigid transformation support now.

            elif method == 'GCRANSAC':
                pred_trans, pred_labels = GCRANSAC(corr, src_keypts, tgt_keypts, args)        

            elif method == 'LS':
                # compute the transformation matrix by solving the least-square system
                corr = corr[:, gt_labels[0].bool(), ]
                src_keypts = src_keypts[:, gt_labels[0].bool(), ]
                tgt_keypts = tgt_keypts[:, gt_labels[0].bool(), ]
                # svd_trans = rigid_transform_3d(src_keypts, tgt_keypts)
                A = torch.cat([src_keypts, torch.ones_like(src_keypts[:,:,0:1])], dim=-1)
                B = torch.cat([tgt_keypts, torch.ones_like(tgt_keypts[:,:,0:1])], dim=-1)

                # https://pytorch.org/docs/stable/generated/torch.lstsq.html
                # Ct = torch.lstsq(input=B[0], A=A[0])
                # pred_trans = Ct.solution[0:4].T
                # pred_trans = pred_trans[None]
                # pred_labels = gt_labels
                BB = B.permute(0,2,1) # [bs,4,n]
                AA = A.permute(0,2,1) # [bs,4,n]
                pred_trans = torch.bmm(BB, torch.pinverse(AA))
                pred_labels = gt_labels 

            else:
                exit(-1)

            model_time = model_timer.toc()     
            loss, recall, Re, Te, RMSE = evaluation_metric(pred_trans, gt_trans, src_keypts, tgt_keypts, pred_labels)
            class_stats = class_loss(pred_labels, gt_labels)

            #################################
            # record the evaluation results.
            #################################
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

    return stats


def eval_3DMatch(method, args):
    """
    Collect the evaluation results on each scene of 3DMatch testset, write the result to a .log file.
    """
    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    all_stats = {}
    for scene_ind, scene in enumerate(scene_list):
        dset = ThreeDMatchTest(root='/data/3DMatch/',
                               descriptor=args.descriptor,
                               in_dim=6,
                               inlier_threshold=args.inlier_threshold,
                               num_node='all',
                               use_mutual=args.use_mutual,
                               augment_axis=0,
                               augment_rotation=0.00,
                               augment_translation=0.0,
                               select_scene=scene,
                               )
        dloader = get_dataloader(dset, batch_size=1, num_workers=8, shuffle=False)
        scene_stats = eval_3DMatch_scene(method, scene, scene_ind, dloader, args)
        all_stats[scene] = scene_stats

    scene_vals = np.zeros([len(scene_list), 12])
    scene_ind = 0
    for scene, stats in all_stats.items():
        correct_pair = np.where(stats[:, 0] == 1)
        scene_vals[scene_ind] = stats.mean(0)
        # for Re and Te, we only count the matched pairs.
        scene_vals[scene_ind, 1] = stats[correct_pair].mean(0)[1]
        scene_vals[scene_ind, 2] = stats[correct_pair].mean(0)[2]
        logging.info(f"Scene {scene_ind}th:"
                     f" Reg Recall={scene_vals[scene_ind, 0]*100:.2f}% "
                     f" Mean RE={scene_vals[scene_ind, 1]:.2f} "
                     f" Mean TE={scene_vals[scene_ind, 2]:.2f} "
                     f" Mean Precision={scene_vals[scene_ind, 6]*100:.2f}% "
                     f" Mean Recall={scene_vals[scene_ind, 7]*100:.2f}% "
                     f" Mean F1={scene_vals[scene_ind, 8]*100:.2f}%"
                     )
        scene_ind += 1

    # scene level average
    average = scene_vals.mean(0)
    logging.info(f"All {len(scene_list)} scenes, Mean Reg Recall={average[0]*100:.2f}%, Mean Re={average[1]:.2f}, Mean Te={average[2]:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={average[3]:.2f}(ratio={average[4]*100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={average[5]:.2f}(precision={average[6]*100:.2f}%, recall={average[7]*100:.2f}%, f1={average[8]*100:.2f}%)")
    logging.info(f"\tMean model time: {average[9]:.4f}s, Mean data time: {average[10]:.4f}s")

    # pair level average 
    stats_list = [stats for _, stats in all_stats.items()]
    allpair_stats = np.concatenate(stats_list, axis=0)
    allpair_average = allpair_stats.mean(0)
    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0)
    logging.info(f"*" * 40)
    logging.info(f"All {allpair_stats.shape[0]} pairs, Mean Reg Recall={allpair_average[0]*100:.2f}%, Mean Re={correct_pair_average[1]:.2f}, Mean Te={correct_pair_average[2]:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={allpair_average[3]:.2f}(ratio={allpair_average[4]*100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={allpair_average[5]:.2f}(precision={allpair_average[6]*100:.2f}%, recall={allpair_average[7]*100:.2f}%, f1={allpair_average[8]*100:.2f}%)")
    logging.info(f"\tMean model time: {allpair_average[9]:.4f}s, Mean data time: {allpair_average[10]:.4f}s")

    all_stats_npy = np.concatenate([v for k, v in all_stats.items()], axis=0)
    return all_stats_npy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='SM', type=str, choices=['SM', 'RANSAC', 'PMC', 'MAGSAC', 'GCRANSAC', 'LS'])
    parser.add_argument('--inlier_threshold', default=0.10, type=float)
    parser.add_argument('--max_iteration', default=1000, type=int) # for ransac
    parser.add_argument('--max_validation', default=1000, type=int) # for ransac
    parser.add_argument('--descriptor', default='fcgf', type=str, choices=['fcgf', 'fpfh'])
    parser.add_argument('--use_mutual', default=False, type=str2bool)
    parser.add_argument('--save_npy', default=False, type=str2bool)
    args = parser.parse_args()

    if args.method in ['RANSAC', 'GCRANSAC', 'MAGSAC']:
        log_filename = f'baseline_logs/3DMatch/{args.method}_{args.max_iteration}_{args.descriptor}.log'
    else:
        log_filename = f'baseline_logs/3DMatch/{args.method}_{args.descriptor}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='a',
                        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) 

    stats = eval_3DMatch(args.method, args)

    if args.save_npy:
        save_path = log_filename.replace('.log', '.npy')
        np.save(save_path, stats)
        print(f"Save the stats in {save_path}")
