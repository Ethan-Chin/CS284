import numpy as np
from glob import glob
import pickle
from multiprocessing import Pool

MAX_ITER = 50
RADIUS = 0.2

def icp(point_cloud_1, point_cloud_2, init_transform=None):
    """
    icp algorithm
    point_cloud_1: Nx3
    point_cloud_2: Nx3
    """
    transform = init_transform if (init_transform is not None) else np.eye(4)
    errors, outlier_ratios = [], []
    transform_ = transform.copy()
    for i in range(MAX_ITER):

        # update point cloud
        point_cloud_1 = update_point_cloud(point_cloud_1, transform)

        # find nearest neighbor
        corresponding_points, outlier_ratio = find_nearest_neighbor(
            point_cloud_1[np.random.choice(len(point_cloud_1), 1000, replace=False)],
            point_cloud_2
        )
        point_cloud_1_corres, point_cloud_2_corres = corresponding_points[:, 0], corresponding_points[:, 1]

        if init_transform is None:
            outlier_ratios.append(outlier_ratio)
            # calculate error
            errors.append(calculate_error(point_cloud_1_corres, point_cloud_2_corres))

        # calculate transform
        transform = calculate_transform(point_cloud_1_corres, point_cloud_2_corres)
        transform_ = transform @ transform_

    return transform_, errors, outlier_ratios


def find_nearest_neighbor(point_cloud_1, point_cloud_2):
    """
    find nearest neighbor
    """
    corresponding_points = []
    for point in point_cloud_1:
        distances = np.linalg.norm(point_cloud_2 - point, axis=1)
        min_idx = np.argmin(distances)
        # reject outliers
        if distances[min_idx] < RADIUS:
            corresponding_points.append([point, point_cloud_2[min_idx]])
    outlier_ratio = 1 - len(corresponding_points) / len(point_cloud_1)

    return np.array(corresponding_points), outlier_ratio


def calculate_transform(point_cloud_1, point_cloud_2):
    """
    calculate transform
    """
    mean1 = np.mean(point_cloud_1, axis=0, keepdims=True)
    mean2 = np.mean(point_cloud_2, axis=0, keepdims=True)
    centered_point_cloud_1 = point_cloud_1 - mean1
    centered_point_cloud_2 = point_cloud_2 - mean2
    Q = centered_point_cloud_2.T @ centered_point_cloud_1
    U, S, V = np.linalg.svd(Q)
    R = U @ V
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    t = mean2.reshape(3, 1) - R @ mean1.T
    return np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))


def update_point_cloud(point_cloud, transform):
    """
    update point cloud
    """
    return (np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))@transform.T)[:, :3]



def calculate_error(point_cloud_1, point_cloud_2):
    """
    calculate error
    """
    return np.mean(np.linalg.norm(point_cloud_1 - point_cloud_2, axis=1))



def icp_process(point_cloud_1, point_cloud_2, idx, init_transform=None):
    """
    multiprocessing wrapper
    """
    transform, errors, outlier_ratios = icp(point_cloud_1, point_cloud_2, init_transform)
    print(f"idx {idx} done!")
    return transform, errors, outlier_ratios, idx


def run_icp(frame_list: list):
    """
    run icp on a list of files
    :param frame_list: list of files
    """
    num_frames = len(frame_list)
    assert num_frames > 1, "frame_list should contain at least 2 files"
    transform_list = [0]*(num_frames-1)
    errors_list = [0]*(num_frames-1)
    outlier_ratios_list = [0]*(num_frames-1)
    pool = Pool(4)
    results = []
    point_cloud_1 = np.loadtxt(frame_list[0])
    for i in range(1, num_frames): # i: 1, 2, ..., num_frames-1
        point_cloud_2 = np.loadtxt(frame_list[i])
        results.append(pool.apply_async(icp_process, args=(point_cloud_1, point_cloud_2, i-1)))
        print(f"processing frame {i+1}/{num_frames}; point_cloud_1: {point_cloud_1.shape}; point_cloud_2: {point_cloud_2.shape}")
        point_cloud_1 = point_cloud_2
    for i, res in enumerate(results):
        transform, errors, outlier_ratios, idx = res.get()
        transform_list[idx] = transform
        errors_list[idx] = errors
        outlier_ratios_list[idx] = outlier_ratios
    pool.close()
    return transform_list, errors_list, outlier_ratios_list


def icp_finetune(frame_list, target_frame, init_transform_list):
    global MAX_ITER
    MAX_ITER = 25
    num_frames = len(frame_list)
    assert num_frames==len(init_transform_list), "wrong number of frames"
    transform_list = [0]*(num_frames)
    pool = Pool(4)
    results = []
    point_cloud_target = np.loadtxt(target_frame)
    for i in range(num_frames):
        point_cloud_query = np.loadtxt(frame_list[i])
        results.append(pool.apply_async(icp_process, args=(point_cloud_query, point_cloud_target, i, init_transform_list[i])))
        print(f"processing frame {i+1}/{num_frames}; point_cloud_query: {point_cloud_query.shape}; point_cloud_target: {point_cloud_target.shape}")
    for i, res in enumerate(results):
        transform, _, __, idx = res.get()
        transform_list[idx] = transform
    pool.close()
    return transform_list

if __name__ == "__main__":
    # frame_list = sorted(glob("./hw1_dataset/frame_*.xyz"))
    # transform_list, errors_list, outlier_ratios_list = run_icp(frame_list)
    # pickle.dump(transform_list, open("./transform_list.pkl", "wb"))
    # pickle.dump(errors_list, open("./errors_list.pkl", "wb"))
    # pickle.dump(outlier_ratios_list, open("./outlier_ratios_list.pkl", "wb"))

    frame_list = sorted(glob("./hw1_dataset/frame_*.xyz"))
    target_frame = frame_list.pop()
    init_transform_list = pickle.load(open("./transform_list.pkl", "rb"))
    assert isinstance(init_transform_list, list), f"expected type {list}, but got {type(init_transform_list)}"
    transform_list = icp_finetune(frame_list, target_frame, init_transform_list)
    pickle.dump(transform_list, open("./finetuned_transform_list.pkl", "wb"))

    print("done!")
