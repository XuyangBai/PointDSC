import os
import open3d as o3d
import numpy as np
from utils.pointcloud import make_point_cloud


def process_3dmatch(voxel_size=0.05):
    root = "/data/3DMatch/threedmatch/"
    save_path = "/data/3DMatch/threedmatch_feat/"
    pcd_list = os.listdir(root)
    for pcd_path in pcd_list:
        if pcd_path.endswith('.npz') is False:
            continue
        full_path = os.path.join(root, pcd_path)
        data = np.load(full_path)
        pts = data['pcd']
        if pts.shape[0] == 0:
            print(f"{full_path} error: do not have any points.")
            continue
        orig_pcd = make_point_cloud(pts)
        # voxel downsample 
        pcd = orig_pcd.voxel_down_sample(voxel_size=voxel_size)

        # estimate the normals and compute fpfh descriptor
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        fpfh_np = np.array(fpfh.data).T
        
        # save the data for training.
        np.savez_compressed(
            os.path.join(save_path, pcd_path.replace('.npz', '_fpfh.npz')),
            points=np.array(orig_pcd.points).astype(np.float32),
            xyz=np.array(pcd.points).astype(np.float32),
            feature=fpfh_np.astype(np.float32),
        )
        print(full_path, fpfh_np.shape)


def process_3dmatch_test(voxel_size=0.05):
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
    for scene in scene_list:
        scene_path = os.path.join("/ssd2/xuyang/3DMatch/fragments/", scene)
        pcd_list = os.listdir(scene_path)
        for pcd_path in pcd_list:
            if not pcd_path.endswith('.ply'):
                continue
            full_path = os.path.join(scene_path, pcd_path)
            orig_pcd = o3d.io.read_point_cloud(full_path)
            # voxel downsample 
            pcd = orig_pcd.voxel_down_sample(voxel_size=voxel_size)

            # estimate the normals and compute fpfh descriptor 
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
            fpfh = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
            fpfh_np = np.array(fpfh.data).T

            # save the data for training.
            np.savez_compressed(
                full_path.replace('.ply', '_fpfh'),
                points=np.array(orig_pcd.points).astype(np.float32),
                xyz=np.array(pcd.points).astype(np.float32),
                feature=fpfh_np.astype(np.float32),
            )
            print(full_path, fpfh_np.shape)


def process_redwood(voxel_size=0.05):
    scene_list = [
        'livingroom1-simulated',
        'livingroom2-simulated',
        'office1-simulated',
        'office2-simulated'
    ]
    for scene in scene_list:
        scene_path = os.path.join("/data/Augmented_ICL-NUIM/", scene + '/fragments')
        pcd_list = os.listdir(scene_path)
        for pcd_path in pcd_list:
            if not pcd_path.endswith('.ply'):
                continue
            full_path = os.path.join(scene_path, pcd_path)
            orig_pcd = o3d.io.read_point_cloud(full_path)
            # voxel downsample 
            pcd = orig_pcd.voxel_down_sample(voxel_size=voxel_size)

            # estimate the normals and compute fpfh descriptor
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
            fpfh = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
            fpfh_np = np.array(fpfh.data).T

            # save the data for training.
            np.savez_compressed(
                full_path.replace('.ply', '_fpfh'),
                points=np.array(orig_pcd.points).astype(np.float32),
                xyz=np.array(pcd.points).astype(np.float32),
                feature=fpfh_np.astype(np.float32),
            )
            print(full_path, fpfh_np.shape)


if __name__ == '__main__':
    # process_3dmatch(voxel_size=0.05)
    # process_3dmatch_test(voxel_size=0.05)
    process_redwood(voxel_size=0.05)
