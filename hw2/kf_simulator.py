import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
Vector3dVector = o3d.utility.Vector3dVector


def rot_mat_2d(theta):
    return np.array([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)]).reshape([2, 2])


def rot_mat_3d(theta):
    return np.array([np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1]).reshape([3, 3])


R_offset = np.array([1.0000000,  0.0000000,  0.0000000,
                     0.0000000,  0.0000000,  1.0000000,
                     0.0000000,  -1.0000000,  0.0000000]).reshape(3, 3)

if __name__ == "__main__":
    num_step = 360

    r0 = np.pi/2
    t0 = np.array([1.0, 0.0]).reshape([1, 2])

    ts = []
    rs = []

    for i in range(num_step//8*7):
        ri = r0+i*2*np.pi/num_step
        Ri = rot_mat_2d(ri)
        ti = t0@Ri.T
        ts.append(ti)
        rs.append(ri)

    rs = np.array(rs)
    ts = np.array(ts).reshape([-1, 2])

    landmark_A = np.zeros([ts.shape[0], 3])
    landmark_A[:, :2] = ts + np.random.randn(*ts.shape)*0.05
    landmark_A[:, -1] = rs + np.random.randn(*rs.shape)*0.03

    odometry = np.zeros([ts.shape[0]-1, 3])
    for i in range(ts.shape[0]-1):
        delta_trans = np.sqrt(np.sum((ts[i+1]-ts[i])**2))
        delta_rot1 = np.arctan2(ts[i+1, 1]-ts[i, 1],
                                ts[i+1, 0]-ts[i, 0]) - rs[i]
        delta_rot2 = rs[i+1]-rs[i]-delta_rot1
        odometry[i, 0] = delta_rot1
        odometry[i, 1] = delta_trans
        odometry[i, 2] = delta_rot2



    landmark_B = np.zeros([ts.shape[0], 3])
    landmark_B[0, 0] = t0[0, 0]
    landmark_B[0, 1] = t0[0, 1]
    landmark_B[0, 2] = r0

    # speed = [[ts[i+1, 0]-ts[i, 0], ts[i+1, 1]-ts[i, 1], rs[i+1]-rs[i]]
    #          for i in range(ts.shape[0]-1)]
    # speed = np.array(speed)
    # speed += np.random.randn(*speed.shape)*0.01

    speed = np.zeros([ts.shape[0]-1, 3])
    speed[:, 0] = 0.1+np.random.randn(speed.shape[0])*0.05
    speed[:, 1] = 0.2+np.random.randn(speed.shape[0])*0.05
    speed[:, 2] = np.pi/180 + np.random.randn(speed.shape[0])*0.02
    for i in range(ts.shape[0]-1):
        landmark_B[i+1, 0] = landmark_B[i, 0] + \
            speed[i, 0]
        landmark_B[i+1, 1] = landmark_B[i, 1] + \
            speed[i, 1]
        landmark_B[i+1, 2] = landmark_B[i, 2] + \
            speed[i, 2]
    landmark_B_gt = landmark_B.copy()
    landmark_B += np.random.randn(*landmark_B.shape)*0.05
    ts = np.hstack((ts, np.zeros([ts.shape[0], 1])))
    data = o3d.geometry.PointCloud()
    data.points = Vector3dVector(ts)

    np.savetxt("landmark_A.txt", landmark_A)
    np.savetxt("odometry.txt", odometry)
    np.savetxt("landmark_B.txt", landmark_B)
    np.savetxt("landmark_B_gt.txt", landmark_B_gt)
    np.savetxt("speed.txt", speed)

    plt.scatter(ts[:,0],ts[:,1])
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()
    # arrow_start = o3d.geometry.TriangleMesh.create_arrow(
    #     cylinder_radius=0.01, cone_radius=0.03, cylinder_height=0.1, cone_height=0.03).rotate(R_offset).rotate(rot_mat_3d(rs[0])).translate(ts[0]).paint_uniform_color((1, 0, 0))
    # arrow_end = o3d.geometry.TriangleMesh.create_arrow(
    #     cylinder_radius=0.01, cone_radius=0.03, cylinder_height=0.1, cone_height=0.03).rotate(R_offset).rotate(rot_mat_3d(rs[-1])).translate(ts[-1]).paint_uniform_color((0, 1, 0))

    # # arrow.rotate()
    # o3d.visualization.draw([arrow_start, arrow_end])
