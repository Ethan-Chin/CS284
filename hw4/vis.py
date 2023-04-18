import open3d as o3d
import pickle as pkl
from main import K

def vis(gt_data, noise_data):
    gt_points = gt_data['world_points']
    noise_points = noise_data['world_points']

    gt_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_points))
    gt_pc.paint_uniform_color([1, 0, 0])
    noise_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(noise_points))
    noise_pc.paint_uniform_color([0, 0, 1])
    geo = [gt_pc, noise_pc]

    for k in gt_data:
        if k.startswith("extrinsic"):
            camera = o3d.geometry.LineSet.create_camera_visualization(640, 480, K, gt_data[k], scale=0.02)
            camera.paint_uniform_color([1, 0, 0])
            noise_camera = o3d.geometry.LineSet.create_camera_visualization(640, 480, K, noise_data[k], scale=0.02)
            noise_camera.paint_uniform_color([0, 0, 1])
            geo.extend([camera, noise_camera])
    o3d.visualization.draw_geometries(geo)

