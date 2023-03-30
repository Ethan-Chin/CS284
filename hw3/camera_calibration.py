"""
    Zhengyou Zhang's method
"""
import cv2
import numpy as np
import glob



def read_frames(files_path):
    images = []
    for file in glob.glob(files_path):
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
    return corners_list


def findHomography(corners_list):
    """
    Find the homography matrix
    :param corners_list: list of corners
    :return: list of homography matrices
    """
    H_list = []
    for corners in corners_list:
        # 3D points
        objp = np.zeros((7 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:8].T.reshape(-1, 2)
        print(objp)

        # 2D points
        imgp = corners
        print(imgp)

        # find the homography matrix
        H, _ = cv2.findHomography(objp, imgp, cv2.RANSAC, 5.0)
        print(H)
        H_list.append(H)
    return H_list 


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
    b = V[:, -1]
    # find the camera parameters
    v0 = (b[1]*b[3] - b[0]*b[4]) / (b[0]*b[2] - b[1]**2)
    lmd = b[5] - (b[3]**2 + v0*(b[1]*b[3] - b[0]*b[4])) / b[0]
    fx = np.sqrt(lmd / b[0])
    fy = np.sqrt(lmd*b[0] / (b[0]*b[2] - b[1]**2))
    gamma = -b[1] * fx**2 * fy / lmd
    u0 = gamma * v0 / fy - b[3] * fx**2 / lmd

    # camera matrix
    M = np.array(
        [
            [fx, gamma, u0],
            [0, fy, v0],
            [0, 0, 1]
        ]
    )
    # find the rotation and translation matrix by SVD

        
        




if __name__ == '__main__':
    images = read_frames('/Users/chenyc/Documents/CS284/hw3/data/partA/calibration_seq_A_frames/*.png')
    corners_list = get_corners(images)
    H_list = findHomography(corners_list)