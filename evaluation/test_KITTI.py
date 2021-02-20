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
from datasets.KITTI import KITTIDataset
from datasets.dataloader import get_dataloader
from utils.pointcloud import make_point_cloud
from evaluation.benchmark_utils import set_seed, icp_refine
from utils.timer import Timer
set_seed()


def eval_KITTI_per_pair(model, dloader, config, use_icp):
    """
    Evaluate our model on KITTI testset.
    """
    num_pair = dloader.dataset.__len__()
    # 0.success, 1.RE, 2.TE, 3.input inlier number, 4.input inlier ratio,  5. output inlier number 
    # 6. output inlier precision, 7. output inlier recall, 8. output inlier F1 score 9. model_time, 10. data_time 11. scene_ind
    stats = np.zeros([num_pair, 12])
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
            res = model(data)
            pred_trans, pred_labels = res['final_trans'], res['final_labels']

            if args.solver == 'SVD':
                pass
            
            elif args.solver == 'RANSAC':
                # our method can be used with RANSAC as a outlier pre-filtering step.
                src_pcd = make_point_cloud(src_keypts[0].detach().cpu().numpy())
                tgt_pcd = make_point_cloud(tgt_keypts[0].detach().cpu().numpy())
                corr = np.array([np.arange(src_keypts.shape[1]), np.arange(src_keypts.shape[1])])
                pred_inliers = np.where(pred_labels.detach().cpu().numpy() > 0)[1]
                corr = o3d.utility.Vector2iVector(corr[:, pred_inliers].T)
                reg_result = o3d.registration.registration_ransac_based_on_correspondence(
                    src_pcd, tgt_pcd, corr,
                    max_correspondence_distance=config.inlier_threshold,
                    estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=3,
                    criteria=o3d.registration.RANSACConvergenceCriteria(max_iteration=5000, max_validation=5000)
                )
                inliers = np.array(reg_result.correspondence_set)
                pred_labels = torch.zeros_like(gt_labels)
                pred_labels[0, inliers[:, 0]] = 1
                pred_trans = torch.eye(4)[None].to(src_keypts.device)
                pred_trans[:, :4, :4] = torch.from_numpy(reg_result.transformation)
  
            if use_icp:
                pred_trans = icp_refine(src_keypts, tgt_keypts, pred_trans)

            model_time = model_timer.toc()
            class_stats = class_loss(pred_labels, gt_labels)
            loss, recall, Re, Te, rmse = evaluate_metric(pred_trans, gt_trans, src_keypts, tgt_keypts, pred_labels)
            pred_trans = pred_trans[0]

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
            stats[i, 11] = -1

            if recall == 0:
                from evaluation.benchmark_utils import rot_to_euler
                R_gt, t_gt = gt_trans[0][:3, :3], gt_trans[0][:3, -1]
                euler = rot_to_euler(R_gt.detach().cpu().numpy())

                input_ir = float(torch.mean(gt_labels.float()))
                input_i = int(torch.sum(gt_labels))
                output_i = int(torch.sum(gt_labels[pred_labels > 0]))
                logging.info(f"Pair {i}, GT Rot: {euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}, Trans: {t_gt[0]:.2f}, {t_gt[1]:.2f}, {t_gt[2]:.2f}, RE: {float(Re):.2f}, TE: {float(Te):.2f}")
                logging.info((f"\tInput Inlier Ratio :{input_ir*100:.2f}%(#={input_i}), Output: IP={float(class_stats['precision'])*100:.2f}%(#={output_i}) IR={float(class_stats['recall'])*100:.2f}%"))

    return stats

def eval_KITTI(model, config, use_icp):
    dset = KITTIDataset(root='/data/KITTI',
                    split='test',
                    descriptor=config.descriptor,
                    in_dim=config.in_dim,
                    inlier_threshold=config.inlier_threshold,
                    num_node=12000,
                    use_mutual=config.use_mutual,
                    augment_axis=0, 
                    augment_rotation=0.00, 
                    augment_translation=0.0,
                    )
    dloader = get_dataloader(dset, batch_size=1, num_workers=16, shuffle=False)
    stats = eval_KITTI_per_pair(model, dloader, config, use_icp)
    logging.info(f"Max memory allicated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")

    # pair level average 
    allpair_stats = stats
    allpair_average = allpair_stats.mean(0)
    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0)
    logging.info(f"*"*40)
    logging.info(f"All {allpair_stats.shape[0]} pairs, Mean Success Rate={allpair_average[0]*100:.2f}%, Mean Re={correct_pair_average[1]:.2f}, Mean Te={correct_pair_average[2]:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={allpair_average[3]:.2f}(ratio={allpair_average[4]*100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={allpair_average[5]:.2f}(precision={allpair_average[6]*100:.2f}%, recall={allpair_average[7]*100:.2f}%, f1={allpair_average[8]*100:.2f}%)")
    logging.info(f"\tMean model time: {allpair_average[9]:.2f}s, Mean data time: {allpair_average[10]:.2f}s")

    return allpair_stats

if __name__ == '__main__':
    from config import str2bool
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='', type=str, help='snapshot dir')
    parser.add_argument('--solver', default='SVD', type=str, choices=['SVD', 'RANSAC'])
    parser.add_argument('--use_icp', default=False, type=str2bool)
    parser.add_argument('--save_npz', default=False, type=str2bool)
    args = parser.parse_args()

    if args.use_icp:
        log_filename = f'logs/{args.chosen_snapshot}-{args.solver}-ICP.log'
    else:
        log_filename = f'logs/{args.chosen_snapshot}-{args.solver}.log'
    logging.basicConfig(level=logging.INFO, 
        filename=log_filename, 
        filemode='a', 
        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) 
    
    config_path = f'snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    ## in case test the generalization ability of model trained on 3DMatch
    config.inlier_threshold = 0.6
    config.sigma_d = 1.2
    config.re_thre = 5
    config.te_thre = 60
    config.descriptor = 'fcgf'

    ## dynamically load the model from snapshot
    # module_file_path = f'snapshot/{args.chosen_snapshot}/model.py'
    # module_name = 'model'
    # module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    # module = importlib.util.module_from_spec(module_spec)
    # module_spec.loader.exec_module(module)
    # PointSM = module.PointSM
    
    # load from models/PointSM.py
    from models.PointSM import PointSM
    model = PointSM(
            in_dim=config.in_dim,
            num_layers=config.num_layers,
            num_channels=config.num_channels,
            num_iterations=config.num_iterations,
            ratio=config.ratio,
            inlier_threshold=config.inlier_threshold,
            sigma_d=config.sigma_d,
            k=config.k,
            nms_radius=config.inlier_threshold,
            )
    miss = model.load_state_dict(torch.load(f'snapshot/{args.chosen_snapshot}/models/model_best.pkl'), strict=False)
    print(miss)
    model.eval()

    # evaluate on the test set
    stats = eval_KITTI(model.cuda(), config, args.use_icp)

    if args.save_npz:
        save_path = log_filename.replace('.log', '.npy')
        np.save(save_path, stats)
        print(f"Save the stats in {save_path}")
