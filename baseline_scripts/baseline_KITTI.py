import torch
import sys
import argparse
import logging
import numpy as np
from tqdm import tqdm
from models.common import rigid_transform_3d
from libs.loss import TransformationLoss, ClassificationLoss
from datasets.KITTI import KITTIDataset
from datasets.dataloader import get_dataloader
from utils.timer import Timer
from baseline_scripts.baseline_3DMatch import SM, RANSAC, GCRANSAC
from config import str2bool


def eval_KITTI_scene(method, dloader, args):
    """
    Evaluate baseline methods on KITTI testset.
    """
    num_pair = dloader.dataset.__len__()
    # 0.success, 1.RE, 2.TE, 3.input inlier number, 4.input inlier ratio,  5. output inlier number
    # 6. output inlier precision, 7. output inlier recall, 8. output inlier F1 score 9. model_time, 10. data_time 11. scene_ind
    stats = np.zeros([num_pair, 12])
    dloader_iter = dloader.__iter__()
    class_loss = ClassificationLoss()
    evaluation_metric = TransformationLoss(re_thre=5, te_thre=60)
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
                pred_trans, pred_labels = SM(corr, src_keypts, tgt_keypts, args, top_ratio=0.05)

            elif method == 'RANSAC':
                pred_trans, pred_labels = RANSAC(corr, src_keypts, tgt_keypts, args)

            elif method == 'GCRANSAC':
                pred_trans, pred_labels = GCRANSAC(corr, src_keypts, tgt_keypts, args)
            
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


def eval_KITTI(method, args):
    dset = KITTIDataset(root='/data/KITTI/',
                        split='test',
                        descriptor=args.descriptor,
                        in_dim=6,
                        inlier_threshold=args.inlier_threshold,
                        num_node=15000,
                        use_mutual=args.use_mutual,
                        augment_axis=0,
                        augment_rotation=0.00,
                        augment_translation=0.0,
                        )
    dloader = get_dataloader(dset, batch_size=1, num_workers=8, shuffle=False)
    stats = eval_KITTI_scene(method, dloader, args)

    # pair level average
    allpair_stats = stats
    allpair_average = allpair_stats.mean(0)
    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0)
    logging.info(f"*" * 40)
    logging.info(f"All {allpair_stats.shape[0]} pairs, Mean Reg Recall={allpair_average[0] * 100:.2f}%, Mean Re={correct_pair_average[1]:.2f}, Mean Te={correct_pair_average[2]:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={allpair_average[3]:.2f}(ratio={allpair_average[4] * 100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={allpair_average[5]:.2f}(precision={allpair_average[6] * 100:.2f}%, recall={allpair_average[7] * 100:.2f}%, f1={allpair_average[8] * 100:.2f}%)")
    logging.info(f"\tMean model time: {allpair_average[9]:.2f}s, Mean data time: {allpair_average[10]:.2f}s")

    return allpair_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='RANSAC', type=str, choices=['SM', 'RANSAC', 'GCRANSAC'])
    parser.add_argument('--inlier_threshold', default=0.6, type=float)
    parser.add_argument('--max_iteration', default=1000, type=int)  # for RANSAC
    parser.add_argument('--max_validation', default=1000, type=int)  # for RANSAC
    parser.add_argument('--descriptor', default='fcgf', type=str, choices=['fcgf', 'fpfh'])
    parser.add_argument('--use_mutual', default=False, type=str2bool)
    parser.add_argument('--save_npy', default=False, type=str2bool)
    args = parser.parse_args()

    if args.method in ['RANSAC', 'GCRANSAC']:
        log_filename = f'baseline_logs/KITTI/{args.method}_{args.max_iteration}_{args.descriptor}.log'
    else:
        log_filename = f'baseline_logs/KITTI/{args.method}_{args.descriptor}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='w',
                        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) 

    stats = eval_KITTI(args.method, args)

    if args.save_npy:
        save_path = log_filename.replace('.log', '.npy')
        np.save(save_path, stats)
        print(f"Save the stats in {save_path}")
