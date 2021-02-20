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
from datasets.ThreeDMatch import ThreeDMatchTest
from datasets.dataloader import get_dataloader
from utils.pointcloud import make_point_cloud
from evaluation.benchmark_utils import set_seed, icp_refine
from utils.timer import Timer
set_seed()


def eval_3DMatch_scene(model, scene, scene_ind, dloader, config, use_icp):
    """
    Evaluate our model on 3DMatch testset [scene]
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

    return stats


def eval_3DMatch(model, config, use_icp):
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
        dset = ThreeDMatchTest(root='/data/3DMatch',
                               descriptor=config.descriptor,
                               in_dim=config.in_dim,
                               inlier_threshold=config.inlier_threshold,
                               num_node='all',
                               use_mutual=config.use_mutual,
                               augment_axis=0,
                               augment_rotation=0.00,
                               augment_translation=0.0,
                               select_scene=scene,
                               )
        dloader = get_dataloader(dset, batch_size=1, num_workers=16, shuffle=False)
        scene_stats = eval_3DMatch_scene(model, scene, scene_ind, dloader, config, use_icp)
        all_stats[scene] = scene_stats
    logging.info(f"Max memory allicated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")

    # result for each scene
    scene_vals = np.zeros([len(scene_list), 12])
    scene_ind = 0
    for scene, stats in all_stats.items():
        correct_pair = np.where(stats[:, 0] == 1)
        scene_vals[scene_ind] = stats.mean(0)
        # for Re and Te, we only average over the successfully matched pairs.
        scene_vals[scene_ind, 1] = stats[correct_pair].mean(0)[1]
        scene_vals[scene_ind, 2] = stats[correct_pair].mean(0)[2]
        logging.info(f"Scene {scene_ind}th:"
                     f" Reg Recall={scene_vals[scene_ind, 0] * 100:.2f}% "
                     f" Mean RE={scene_vals[scene_ind, 1]:.2f} "
                     f" Mean TE={scene_vals[scene_ind, 2]:.2f} "
                     f" Mean Precision={scene_vals[scene_ind, 6] * 100:.2f}% "
                     f" Mean Recall={scene_vals[scene_ind, 7] * 100:.2f}% "
                     f" Mean F1={scene_vals[scene_ind, 8] * 100:.2f}%"
                     )
        scene_ind += 1

    # scene level average
    average = scene_vals.mean(0)
    logging.info(f"All {len(scene_list)} scenes, Mean Reg Recall={average[0] * 100:.2f}%, Mean Re={average[1]:.2f}, Mean Te={average[2]:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={average[3]:.2f}(ratio={average[4] * 100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={average[5]:.2f}(precision={average[6] * 100:.2f}%, recall={average[7] * 100:.2f}%, f1={average[8] * 100:.2f}%)")
    logging.info(f"\tMean model time: {average[9]:.2f}s, Mean data time: {average[10]:.2f}s")

    # pair level average 
    stats_list = [stats for _, stats in all_stats.items()]
    allpair_stats = np.concatenate(stats_list, axis=0)
    allpair_average = allpair_stats.mean(0)
    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0)
    logging.info(f"*" * 40)
    logging.info(f"All {allpair_stats.shape[0]} pairs, Mean Reg Recall={allpair_average[0] * 100:.2f}%, Mean Re={correct_pair_average[1]:.2f}, Mean Te={correct_pair_average[2]:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={allpair_average[3]:.2f}(ratio={allpair_average[4] * 100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={allpair_average[5]:.2f}(precision={allpair_average[6] * 100:.2f}%, recall={allpair_average[7] * 100:.2f}%, f1={allpair_average[8] * 100:.2f}%)")
    logging.info(f"\tMean model time: {allpair_average[9]:.2f}s, Mean data time: {allpair_average[10]:.2f}s")

    all_stats_npy = np.concatenate([v for k, v in all_stats.items()], axis=0)
    return all_stats_npy


if __name__ == '__main__':
    from config import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='', type=str, help='snapshot dir')
    parser.add_argument('--solver', default='SVD', type=str, choices=['SVD', 'RANSAC'])
    parser.add_argument('--use_icp', default=False, type=str2bool)
    parser.add_argument('--save_npy', default=False, type=str2bool)
    args = parser.parse_args()

    config_path = f'snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)
    # config.descriptor = 'fpfh'

    if args.use_icp:
        log_filename = f'logs/{args.chosen_snapshot}-{args.solver}-{config.descriptor}-ICP.log'
    else:
        log_filename = f'logs/{args.chosen_snapshot}-{args.solver}-{config.descriptor}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='a',
                        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))   

    ## dynamically load the model from snapshot
    # module_file_path = f'snapshot/{args.chosen_snapshot}/model.py'
    # module_name = 'model'
    # module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    # module = importlib.util.module_from_spec(module_spec)
    # module_spec.loader.exec_module(module)
    # PointSM = module.PointSM

    ## load the model from models/PointSM.py
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

    # evaluate on the test set
    stats = eval_3DMatch(model.cuda(), config, args.use_icp)

    # claculate area under the cumulative error curve.
    # re_auc = exact_auc(stats[:, 1], thresholds=[5, 10, 15])
    # te_auc = exact_auc(stats[:, 2], thresholds=[5, 10, 15, 20, 25, 30])
    # print(f"RE AUC:", re_auc)
    # print(f"TE AUC:", te_auc)

    if args.save_npy:
        save_path = log_filename.replace('.log', '.npy')
        np.save(save_path, stats)
        print(f"Save the stats in {save_path}")
