import numpy as np
import pickle as pkl
def back_proj(image_coords, intrinsic):
    # assume the extrinsic matrix is identity
    # assume the depth is 2.0 m
    # image_coords: (N, 2)

    inv_K = np.linalg.inv(intrinsic)
    points = np.zeros((image_coords.shape[0], 3))
    points[:, :2] = image_coords
    points[:, 2] = 1.0
    points *= 2.0
    points = (inv_K @ points.T).T

    return points



def get_view_coords(points, intrinsic, extrinsic):
    # points: (N, 3)
    # intrinsic: (3, 3)
    # extrinsic: (4, 4)

    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    intrinsic = np.concatenate([intrinsic, np.zeros((3, 1))], axis=1)
    points = (extrinsic @ points.T).T
    points = (intrinsic @ points.T).T
    points = points[:, :2] / points[:, 2:]

    return points


if __name__ == '__main__':
    image_coords_c5 = np.array([[50, 40],[185, 40],[320, 40],[455, 40],[590, 40],[50, 140],[185, 140],[320, 140],[455, 140],[590, 140],[50, 240],[185, 240],[320, 240],[455, 240],[590, 240],[50, 340],[185, 340],[320, 340],[455, 340],[590, 340],[50, 440],[185, 440],[320, 440],[455, 440],[590, 440]])
    K = np.array(
        [[500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]]
    )

    points = back_proj(image_coords_c5, K)
    print('world points:')
    print(points)
    extrinsic = np.array(
        [[1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]]
    )

    save = {"image_coords_c5": image_coords_c5, "world_points": points, "extrinsic_c5": extrinsic}


    print("************** C1 view **************")
    extrinsic_c1 = extrinsic.copy()
    extrinsic_c1[0, 3] = 0.1
    extrinsic_c1[1, 3] = 0.1
    image_coords_c1 = get_view_coords(points, K, extrinsic_c1)
    save["image_coords_c1"] = image_coords_c1
    save["extrinsic_c1"] = extrinsic_c1
    print(image_coords_c1)

    print("************** C2 view **************")
    extrinsic_c2 = extrinsic.copy()
    extrinsic_c2[1, 3] = 0.1
    image_coords_c2 = get_view_coords(points, K, extrinsic_c2)
    save["image_coords_c2"] = image_coords_c2
    save["extrinsic_c2"] = extrinsic_c2
    print(image_coords_c2)

    print("************** C3 view **************")
    extrinsic_c3 = extrinsic.copy()
    extrinsic_c3[0, 3] = -0.1
    extrinsic_c3[1, 3] = 0.1
    image_coords_c3 = get_view_coords(points, K, extrinsic_c3)
    save["image_coords_c3"] = image_coords_c3
    save["extrinsic_c3"] = extrinsic_c3
    print(image_coords_c3)

    print("************** C4 view **************")
    extrinsic_c4 = extrinsic.copy()
    extrinsic_c4[0, 3] = 0.1
    image_coords_c4 = get_view_coords(points, K, extrinsic_c4)
    save["image_coords_c4"] = image_coords_c4
    save["extrinsic_c4"] = extrinsic_c4
    print(image_coords_c4)

    print("************** C6 view **************")
    extrinsic_c6 = extrinsic.copy()
    extrinsic_c6[0, 3] = -0.1
    image_coords_c6 = get_view_coords(points, K, extrinsic_c6)
    save["image_coords_c6"] = image_coords_c6
    save["extrinsic_c6"] = extrinsic_c6
    print(image_coords_c6)

    print("************** C7 view **************")
    extrinsic_c7 = extrinsic.copy()
    extrinsic_c7[0, 3] = 0.1
    extrinsic_c7[1, 3] = -0.1
    image_coords_c7 = get_view_coords(points, K, extrinsic_c7)
    save["image_coords_c7"] = image_coords_c7
    save["extrinsic_c7"] = extrinsic_c7
    print(image_coords_c7)

    print("************** C8 view **************")
    extrinsic_c8 = extrinsic.copy()
    extrinsic_c8[1, 3] = -0.1
    image_coords_c8 = get_view_coords(points, K, extrinsic_c8)
    save["image_coords_c8"] = image_coords_c8
    save["extrinsic_c8"] = extrinsic_c8
    print(image_coords_c8)

    print("************** C9 view **************")
    extrinsic_c9 = extrinsic.copy()
    extrinsic_c9[0, 3] = -0.1
    extrinsic_c9[1, 3] = -0.1
    image_coords_c9 = get_view_coords(points, K, extrinsic_c9)
    save["image_coords_c9"] = image_coords_c9
    save["extrinsic_c9"] = extrinsic_c9
    print(image_coords_c9)


    with open('./data.pkl', 'wb') as f:
        pkl.dump(save, f)



