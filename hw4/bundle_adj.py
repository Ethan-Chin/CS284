import numpy as np
import pickle as pkl
from main import get_view_coords, K, plot_table
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import matplotlib.pyplot as plt
import ipdb
from scipy.optimize import least_squares
import open3d as o3d

def f(input):
    # input: (129, 1)
    # output: (450, 1)
    points = input[:75].reshape(25, 3)
    output = np.zeros((450, 1))
    for i in range(9):
        extrinsic = np.eye(4)
        extrinsic_param = input[75 + 6 * i: 75 + 6 * (i + 1)]
        extrinsic[:3, :3] = Rotation.from_euler("xyz", extrinsic_param[:3].squeeze()).as_matrix()
        extrinsic[:3, 3] = extrinsic_param[3:].squeeze()
        view_coords = get_view_coords(points, K, extrinsic) # (25, 2)
        output[50 * i: 50 * (i + 1)] = view_coords.reshape(50, 1)
    return (output).squeeze()


def Jacobian(input, f0):
    eps = 1e-10
    J = np.zeros((f0.shape[0], input.shape[0]))
    for i in range(input.shape[0]):
        increased_input = input.copy()
        increased_input[i] += eps
        f1 = f(increased_input)
        J[:, i] = (f1 - f0) / eps
    return J


def GN(X, input):
    # X: (450, 1)
    # input: (129, 1)
    loss = []
    for i in tqdm(range(10)):
        f0 = f(input)
        e0 = (X - f0.reshape(450, 1))
        loss.append(np.linalg.norm(e0))
        J = Jacobian(input, f0)
        H = J.T @ J
        delta = np.linalg.inv(H + np.eye(H.shape[0])) @ J.T @ e0
        input += 0.5*delta
    return input, loss



gt_data = pkl.load(open("data.pkl", "rb"))
noise_data = pkl.load(open("noise_data.pkl", "rb"))

X = np.zeros((450, 1))
input = np.zeros((129, 1))
input[:75] = noise_data["world_points"].reshape(75, 1)
for i in range(9):
    X_name = f"image_coords_c{i+1:01d}"
    coord = noise_data[X_name].copy()
    # coord[:, 0] = (coord[:, 0] - 640) / 500
    # coord[:, 1] = (coord[:, 1] - 480) / 500
    X[50 * i: 50 * (i + 1)] = coord.reshape(50, 1)
    extrinsic = noise_data[f"extrinsic_c{i+1:01d}"]
    r = Rotation.from_matrix(extrinsic[:3, :3])
    angles = r.as_euler("xyz")
    input[75 + 6 * i: 75 + 6 * (i + 1)] = np.concatenate((angles, extrinsic[:3, 3])).reshape(6, 1)
input, loss = GN(X, input)
plot_table(input[:75].reshape(25, 3).T, ["x", "y", "z"], save_path="./bundle_adj_world_points.png")


print(input[:75].reshape(25, 3))
print(input[75:].reshape(-1, 6))
plt.figure(figsize=(8, 6))
plt.plot(loss)
plt.xlim([0, 9])
plt.xlabel("Iteration")
plt.yticks(np.arange(0, 110, 10))
plt.ylim([0, 110])
plt.ylabel("Loss (l2 norm of residuals)")
plt.grid()
plt.savefig('loss.png', dpi=300)


gt_points = gt_data['world_points']
noise_points = noise_data['world_points']

gt_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_points))
gt_pc.paint_uniform_color([1, 0, 0])
noise_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(noise_points))
noise_pc.paint_uniform_color([0, 0, 1])
adj_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input[:75].reshape(25, 3)))
adj_pc.paint_uniform_color([0, 1, 0])
geo = [gt_pc, noise_pc, adj_pc]

for i in range(9):
    camera = o3d.geometry.LineSet.create_camera_visualization(640, 480, K, gt_data[f"extrinsic_c{i+1:01d}"], scale=0.02)
    camera.paint_uniform_color([1, 0, 0])
    noise_camera = o3d.geometry.LineSet.create_camera_visualization(640, 480, K, noise_data[f"extrinsic_c{i+1:01d}"], scale=0.02)
    noise_camera.paint_uniform_color([0, 0, 1])
    extrinsic_adj = np.eye(4)
    extrinsic_param = input[75 + 6 * i: 75 + 6 * (i + 1)]
    extrinsic_adj[:3, :3] = Rotation.from_euler("xyz", extrinsic_param[:3].squeeze()).as_matrix()
    extrinsic_adj[:3, 3] = extrinsic_param[3:].squeeze()
    adj_camera = o3d.geometry.LineSet.create_camera_visualization(640, 480, K, extrinsic_adj, scale=0.02)
    adj_camera.paint_uniform_color([0, 1, 0])
    geo.extend([camera, noise_camera, adj_camera])
o3d.visualization.draw_geometries(geo)

