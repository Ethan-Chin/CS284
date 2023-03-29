import numpy as np
from glob import glob
import open3d as o3d
import matplotlib.pyplot as plt
import pickle
from main import update_point_cloud



def build_transform_matrix(transform_list):
    """
    build a matrix T: NxNx4x4; T[j][i] is the transform from frame i to frame j
    param transform_list: list of transforms (4x4); transform_list[i] is the transform from frame i to frame i+1
    """
    num_frames = len(transform_list) + 1
    T = np.zeros((num_frames, num_frames, 4, 4))
    for i in range(num_frames):
        T[i][i] = np.eye(4)
    for i in range(num_frames-1):
        for j in range(i+1, num_frames):
            T[j][i] = transform_list[j-1]@T[j-1][i]
            T[i][j] = np.linalg.inv(T[j][i])
    return T


def point_cloud_registration(pc_frames, T, coord_idx):
    """
    align all frames into the coordinates of frame idx, using the transform matrix T
    param pc_frames: list of point clouds; pc_frames[i] is the point cloud of frame i
    param T: transform matrix; T[j][i] is the transform from frame i to frame j
    param coord_idx: the index of the frame to align to
    """
    num_frames = len(pc_frames)
    assert num_frames == T.shape[0] == T.shape[1], "T should be a NxNx4x4 matrix"
    assert coord_idx < num_frames, "coord_idx should be less than the number of frames"
    pc_frames_aligned = [0]*num_frames
    for i in range(num_frames):
        pc_frames_aligned[i] = update_point_cloud(pc_frames[i], T[coord_idx][i])
    return pc_frames_aligned


def camera_trajectory(T, coord_idx):
    coord_frame_list = list()
    for i in range(len(T)):
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8, origin=np.zeros((3, 1), dtype=np.float32))
        coord_frame_list.append(coord_frame.transform(T[coord_idx][i]))
    return coord_frame_list


def visualize_point_cloud(pc, c_tr, color=None):
    """
    visualize point cloud using open3d
    param pc: point cloud Nx3
    param color: color 1x3
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd = pcd.voxel_down_sample(voxel_size=0.125)
    if color is None:
        color = [0, 0.1843, 0.6549] # Klein Blue
    pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries([pcd, *c_tr])


def refine_T(T: np.ndarray, finetuned_transform_list, coord_idx):
    T = T.copy()
    # for i, t in enumerate(finetuned_transform_list):
    #     T[coord_idx][i] = t
    return T
        




if __name__ == "__main__":
    transform_list = pickle.load(open("./transform_list.pkl", "rb"))
    errors_list = pickle.load(open("./errors_list.pkl", "rb"))
    outlier_ratios_list = pickle.load(open("./outlier_ratios_list.pkl", "rb"))
    # frame_list = sorted(glob("./hw1_dataset/frame_*.xyz"))
    # pc_frames = [np.loadtxt(frame) for frame in frame_list]


    # T = build_transform_matrix(transform_list)
    # # pc_frames_aligned = point_cloud_registration(pc_frames, T, 19)
    # # pc = np.vstack(pc_frames_aligned)
    # T_refined = refine_T(T, pickle.load(open("./finetuned_transform_list.pkl", "rb")), 19)
    # pc_frames_aligned_refined = point_cloud_registration(pc_frames, T_refined, 19)
    # pc_refined = np.vstack(pc_frames_aligned_refined)

    # c_tr = camera_trajectory(T_refined, 19)

    # # print(f"point cloud loaded: {pc.shape}")
    # # visualize_point_cloud(pc)
    # visualize_point_cloud(pc_refined, c_tr)

    # plot errors
    plt.subplots(4, 5, figsize=(20, 10))
    for i in range(len(errors_list)):
        plt.subplot(4, 5, i+1)
        plt.plot(errors_list[i])
        plt.xlabel("iteration")
        plt.ylabel("errors")
    plt.savefig('./errors.png', dpi=600)

    plt.subplots(4, 5, figsize=(20, 10))
    for i in range(len(outlier_ratios_list)):
        plt.subplot(4, 5, i+1)
        plt.plot(outlier_ratios_list[i])
        plt.xlabel("iteration")
        plt.ylabel("outlier ratios")
    plt.savefig('./outliers.png', dpi=600)
