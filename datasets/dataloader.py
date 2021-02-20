import numpy as np
import torch
import random


def collate_fn(list_data):
    min_num = 1e10
    # clip the pair having more correspondence during training.
    for ind, (corr_pos, src_keypts, tgt_keypts, gt_trans, gt_labels) in enumerate(list_data):
        if len(gt_labels) < min_num:
            min_num = min(min_num, len(gt_labels))

    batched_corr_pos = []
    batched_src_keypts = []
    batched_tgt_keypts = []
    batched_gt_trans = []
    batched_gt_labels = []
    for ind, (corr_pos, src_keypts, tgt_keypts, gt_trans, gt_labels) in enumerate(list_data):
        sel_ind = np.random.choice(len(gt_labels), min_num, replace=False)
        batched_corr_pos.append(corr_pos[sel_ind, :][None,:,:])
        batched_src_keypts.append(src_keypts[sel_ind, :][None,:,:])
        batched_tgt_keypts.append(tgt_keypts[sel_ind, :][None,:,:])
        batched_gt_trans.append(gt_trans[None,:,:])
        batched_gt_labels.append(gt_labels[sel_ind][None, :])
    
    batched_corr_pos = torch.from_numpy(np.concatenate(batched_corr_pos, axis=0))
    batched_src_keypts = torch.from_numpy(np.concatenate(batched_src_keypts, axis=0))
    batched_tgt_keypts = torch.from_numpy(np.concatenate(batched_tgt_keypts, axis=0))
    batched_gt_trans = torch.from_numpy(np.concatenate(batched_gt_trans, axis=0))
    batched_gt_labels = torch.from_numpy(np.concatenate(batched_gt_labels, axis=0))
    return batched_corr_pos, batched_src_keypts, batched_tgt_keypts, batched_gt_trans, batched_gt_labels


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4, fix_seed=True):
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        num_workers=num_workers, 
    )