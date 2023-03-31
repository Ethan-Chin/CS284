import numpy as np
import cv2
import glob
from camera_calibration import get_corners, findHomography, solve_camera_parameters
from scipy.spatial.transform import Rotation
from itertools import combinations
import matplotlib.pyplot as plt
import open3d as o3d
import pickle

def solve(image_sq, pose_sq):
    assert len(image_sq) == len(pose_sq), "must have the same length"
    corners_list = get_corners(image_sq)
    H_list, error_list = findHomography(corners_list)
    print(f"Homography found with error: {np.mean(error_list)}")
    _, R_list, t_list = solve_camera_parameters(H_list)
    A_inv_list = []
    for R, t in zip (R_list, t_list):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        A_inv_list.append(T)

    B_list = []
    for p in pose_sq:
        R = Rotation.from_quat(p[:4]).as_matrix()
        t = np.array(p[4:])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        B_list.append(T)
    RA_list, RB_list, tA_list, tB_list, S_list = [], [], [], [], []
    for comb in combinations(range(len(A_inv_list)), 2):
        i, j = comb
        C = np.linalg.inv(A_inv_list[i]) @ A_inv_list[j]
        D = np.linalg.inv(B_list[i]) @ B_list[j]
        RA = C[:3, :3]
        RB = D[:3, :3]
        tA = C[:3, 3]
        tB = D[:3, 3]
        S = np.kron(RA, np.eye(3)) - np.kron(np.eye(3), RB.T)
        RA_list.append(RA)
        RB_list.append(RB)
        tA_list.append(tA)
        tB_list.append(tB)
        S_list.append(S)
    S = np.vstack(S_list)
    _, _, VT = np.linalg.svd(S)
    RX = VT[-1, :]
    RX = RX.reshape(3, 3)
    U, _, VT = np.linalg.svd(RX)
    RX = U @ VT

    left_list = []
    right_list = []
    for i in range(len(RA_list)):
        left = (RA_list[i] - np.eye(3)).reshape(3, 3)
        right = (RX @ tB_list[i] - tA_list[i]).reshape(3, 1)
        left_list.append(left)
        right_list.append(right)
    L = np.vstack(left_list)
    R = np.vstack(right_list)
    # tX, res, _, _ = lstsq(L, R)
    tX = np.linalg.inv(L.T @ L) @ L.T @ R
    tX = tX.reshape(-1)
    TX = np.eye(4)
    TX[:3, :3] = RX
    TX[:3, 3] = tX

    return TX, R_list, t_list


def draw_cameras(K, R_list, t_list, vis, color):
    for R, t in zip(R_list, t_list):
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = t
        camera = o3d.geometry.LineSet.create_camera_visualization(2448, 2048, K, extrinsics, scale=0.05)
        camera.paint_uniform_color(color)
        vis.add_geometry(camera)


if __name__ == '__main__':
    Y_list = []
    USE_CV2 = False
    fig, axes = plt.subplots(1, 2)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='visualization')
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=np.zeros(3))
    vis.add_geometry(coord)
    for seq in ['A', 'B']:
        K = pickle.load(open(f'./outputs/partA/K_{seq}.pkl', 'rb'))
        image_sq = [cv2.imread(file) for file in sorted(glob.glob(f'./data/partB/{seq}*.png'))]
        pose_sq = [list(map(lambda x: float(x), open(file, 'r').readlines()[1].split())) for file in sorted(glob.glob(f'./data/partB/hw3_v2_{seq}*.pose'))]
        Y, R_list, t_list = solve(image_sq, pose_sq)
        Y_list.append(Y)
        draw_cameras(K, R_list, t_list, vis, (1, 0, 0) if seq == 'A' else (0, 0, 1))
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=np.zeros(3))
        coord.transform(Y)
        vis.add_geometry(coord)

    vis.run()
    print(f"Y difference between A and B: {np.linalg.norm(np.linalg.inv(Y_list[0])@Y_list[1] - np.eye(4)):.4f}")
    axes[0].matshow(Y_list[0])
    axes[1].matshow(Y_list[1])
    # draw the values of the matrix
    for (i, j), z in np.ndenumerate(Y_list[0]):
        axes[0].text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    for (i, j), z in np.ndenumerate(Y_list[1]):
        axes[1].text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    plt.savefig('./outputs/partB/Y.png', dpi=300)





    