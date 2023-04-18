import numpy as np
import pickle as pkl
from scipy.spatial.transform import Rotation



def add_pixel_noise(coords):
    # coords: (N, 2)
    return coords + np.random.normal(0, 0.25, coords.shape)

def add_points_noise(points):
    # points: (N, 3)
    return points + np.random.normal(0, 0.02, points.shape)

def add_extrinsic_noise(extrinsic):
    # extrinsic: (4, 4)
    extrinsic[:3, 3] += np.random.normal(0, 0.002, 3)

    # perturb the rotation by multiplying a random rotation matrix generated by euler angles
    euler_angles = np.random.normal(0, 0.002, 3)
    R = Rotation.from_euler("xyz", euler_angles).as_matrix()
    extrinsic[:3, :3] = R @ extrinsic[:3, :3]
    return extrinsic


if __name__ == '__main__':
    data = pkl.load(open("data.pkl", "rb"))
    noise_data = data.copy()
    for k in data:
        if k.startswith("image_coords"):
            noise_data[k] = add_pixel_noise(data[k])
        elif k.startswith("world_points"):
            noise_data[k] = add_points_noise(data[k])
        elif k.startswith("extrinsic"):
            noise_data[k] = add_extrinsic_noise(data[k])
        print(noise_data[k])

    pkl.dump(noise_data, open("noise_data.pkl", "wb"))
