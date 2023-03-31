"""
    Zhengyou Zhang's method
"""
import cv2
import numpy as np
import glob
import sys
import ipdb
import pickle

def read_frames(files_path):
    images = []
    for file in sorted(glob.glob(files_path)):
        images.append(cv2.imread(file))
    print(f"Read {len(images)} frames!")
    return images

def get_corners(images):
    """
    Get the corners of the chessboard
    :param images: list of images
    :return: list of corners
    """
    corners_list = []
    for image in images:
        # try different resize and apply the findChessboardCorners, if not found, resize again
        for i in [2, 4, 8, 1]:
            resized_image = cv2.resize(image, (0, 0), fx=1 / i, fy=1 / i)
            ret, corners = cv2.findChessboardCorners(resized_image, (7, 8), None, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                corners = corners * i
                # adjust the corner locations with sub-pixel accuracy
                grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.cornerSubPix(grayed_image, corners, (15, 15), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

                cv2.drawChessboardCorners(image, (7, 8), corners, ret)
                cv2.imshow('img', image)
                cv2.waitKey(500)
                corners_list.append(corners)
                break
        else:
            print("Can't find the corners!")
            sys.exit(1)
    return corners_list


def findHomography(corners_list):
    """
    Find the homography matrix
    :param corners_list: list of corners
    :return: list of homography matrices
    """
    H_list = []
    error_list = []
    for corners in corners_list:
        # 3D points
        objp = np.ones((7 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:8].T.reshape(-1, 2)*0.033

        # 2D points
        imgp = corners.squeeze()
        imgp = np.concatenate((imgp, np.ones((imgp.shape[0], 1))), axis=1)
        # find the homography matrix
        H, _ = cv2.findHomography(objp, imgp, 0)
        H_list.append(H)
        # cal the reprojection error
        dst = (H @ objp.T).T
        dst = dst / dst[:, 2].reshape(-1, 1)
        error_list.append(np.mean((dst - imgp)**2))
    return H_list, error_list


def solve_camera_parameters(H_list):
    """
    Solve the camera parameters
    :param H_list: list of homography matrices
    :return: camera parameters
    """
    A = []
    for H in H_list:
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        h11, h12, h13 = h1[0], h1[1], h1[2]
        h21, h22, h23 = h2[0], h2[1], h2[2]
        h31, h32, h33 = h3[0], h3[1], h3[2]

        v11 = np.array(
            [
                h11*h11,
                h11*h12 + h12*h11,
                h12*h12,
                h11*h13 + h13*h11,
                h12*h13 + h13*h12,
                h13*h13
            ]
        )
        v12 = np.array(
            [
                h11*h21,
                h11*h22 + h12*h21,
                h12*h22,
                h11*h23 + h13*h21,
                h12*h23 + h13*h22,
                h13*h23
            ]
        )
        v22 = np.array(
            [
                h21*h21,
                h21*h22 + h22*h21,
                h22*h22,
                h21*h23 + h23*h21,
                h22*h23 + h23*h22,
                h23*h23
            ]
        )
        A.append(
            np.array(
                [
                    v12.reshape(1, -1),
                    (v11 - v22).reshape(1, -1)
                ]
            ).reshape(2, -1)
        )
    A = np.array(A).reshape(-1, 6)
    # do SVD to solve Ab = 0, b is the last column of V
    U, S, V = np.linalg.svd(A)
    b = V[-1, :]
    B11, B12, B22, B13, B23, B33 = b[0], b[1], b[2], b[3], b[4], b[5]
    # find the camera parameters
    v0 = (B12*B13 - B11*B23) / (B11*B22 - B12*B12)
    lmd = B33 - (B13*B13 + v0 * (B12*B13 - B11*B23)) / B11
    fx = np.sqrt(lmd / B11)
    fy = np.sqrt(lmd * B11 / (B11*B22 - B12*B12))
    gamma = -B12 * fx**2 * fy / lmd
    u0 = gamma * v0 / fy - B13 * fx**2 / lmd

    # camera matrix
    K = np.array(
        [
            [fx, gamma, u0],
            [0, fy, v0],
            [0, 0, 1]
        ]
    )
    
    R_list = []
    t_list = []
    for H in H_list:
        inv_K = np.linalg.inv(K)
        Z = 2 / (np.linalg.norm(inv_K @ H[:, 0], 2) + np.linalg.norm(inv_K @ H[:, 1], 2))
        r1 = Z * (inv_K @ H[:, 0])
        r2 = Z * (inv_K @ H[:, 1])
        r3 = np.cross(r1, r2)
        t = Z * (inv_K @ H[:, 2])
        R = np.array([r1, r2, r3]).T
        R_list.append(R)
        t_list.append(t)
    return K, R_list, t_list




if __name__ == '__main__':
    for seq in ['A', 'B']:
        images = read_frames(f'/Users/chenyc/Documents/CS284/hw3/data/partA/calibration_seq_{seq}_frames/*.png')
        corners_list = get_corners(images)
        H_list, error_list = findHomography(corners_list)
        K, R_list, t_list = solve_camera_parameters(H_list)
        pickle.dump(H_list, open(f'./outputs/partA/H_list_{seq}_error_{np.mean(error_list):.3f}.pkl', 'wb'))
        pickle.dump(K, open(f'./outputs/partA/K_{seq}.pkl', 'wb'))
        pickle.dump(R_list, open(f'./outputs/partA/R_list_{seq}.pkl', 'wb'))
        pickle.dump(t_list, open(f'./outputs/partA/t_list_{seq}.pkl', 'wb'))
        