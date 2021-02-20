from tqdm import tqdm
import numpy as np
from datasets.ThreeDMatch import ThreeDMatchTrainVal, ThreeDMatchTest
from datasets.dataloader import get_dataloader
import matplotlib.pyplot as plt


def process_split(split='test', descriptor='fcgf'):
    if split == 'test':
        dset = ThreeDMatchTest(root='/data/3DMatch', 
                                descriptor=descriptor,
                                num_node='all', 
                                augment_axis=0,
                                augment_rotation=0.0,
                                augment_translation=0.0,
                                )
    else:
        dset = ThreeDMatchTrainVal(root='/data/3DMatch', 
                                descriptor=descriptor,
                                split=split,  
                                num_node='all', 
                                augment_axis=0,
                                augment_rotation=0.0,
                                augment_translation=0.0,
                                )
    dloader = get_dataloader(dset, batch_size=1, num_workers=16, shuffle=False)
    dloader_iter = dloader.__iter__()
    inlier_ratio_list = []
    for i in tqdm(range(len(dset))):
        corr, src_keypts, tgt_keypts, gt_trans, gt_labels = dloader_iter.next()
        inlier_ratio = gt_labels.mean()
        inlier_ratio_list.append(float(inlier_ratio)*100)
    return np.array(inlier_ratio_list)


if __name__ == '__main__':
    descriptor = 'fcgf'
    num_bins = 25
    test_inlier_ratio_list = process_split(split='test', descriptor=descriptor)
    plt.hist(test_inlier_ratio_list, num_bins, histtype='step', density=True, label='test')
    val_inlier_ratio_list = process_split(split='val', descriptor=descriptor)
    plt.hist(val_inlier_ratio_list, num_bins, histtype='step', density=True, label='val')
    train_inlier_ratio_list = process_split(split='train', descriptor=descriptor)
    plt.hist(train_inlier_ratio_list, num_bins, histtype='step', density=True, label='train')
    plt.legend(loc='best')
    plt.xlabel('Inlier Ratio')
    plt.ylabel('Density (%)')
    plt.xlim([0, 100])
    plt.savefig(f'statis_{descriptor}.png')