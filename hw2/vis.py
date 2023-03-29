import numpy as np
import open3d as o3d
import sys
from kf import ekf_A, ekf_B
import matplotlib.pyplot as plt


def state2transform(state_sequence: np.ndarray):
    # state_sequence: Nx3, (x, y, theta) for each

    # R: rotation matrix Nx3x3
    R = np.expand_dims(np.eye(3, dtype=np.float32), axis=0).repeat(state_sequence.shape[0], axis=0)
    R[:, (0, 1), (0, 1)] = np.expand_dims(np.cos(state_sequence[:, 2]), axis=1).repeat(2, axis=1)
    R[:, (0, 1), (1, 0)] = np.expand_dims(np.sin(state_sequence[:, 2]), axis=1).repeat(2, axis=1)
    R[:, 0, 1] = -R[:, 0, 1]

    # t: translation vector Nx3x1
    t = np.hstack((state_sequence[:, :2], np.zeros((state_sequence.shape[0], 1), dtype=np.float32)))
    t = np.expand_dims(t, axis=2)

    # T: transformation matrix Nx4x4
    T = np.expand_dims(np.eye(4, dtype=np.float32), axis=0).repeat(state_sequence.shape[0], axis=0)
    T[:, :3, :3] = R
    T[:, :3, 3:] = t

    return T


def draw_trajectory_3d(T: np.ndarray):
    agent_list = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=np.zeros(3))]
    for i in T:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.zeros(3))
        agent_list.append(coord.transform(i))
    o3d.visualization.draw_geometries(agent_list)


def get_odometry_states(data, init_state = np.zeros(3, dtype=np.float32)):
    states = [init_state]
    x, y, theta = init_state
    for d in data:
        delta_rot1, delta_trans, delta_rot2 = d
        angle = delta_rot1 + theta
        x += delta_trans * np.cos(angle)
        y += delta_trans * np.sin(angle)
        theta = angle + delta_rot2
        states.append(np.array([x, y, theta]))
    return np.array(states)


def xyTheta2Line(states, line_length=0.5):
    assert states.shape[1] == 3
    lines = []
    for i in range(states.shape[0]):
        x, y, theta = states[i]
        lines.append(np.array([[x, y], [x + line_length * np.cos(theta), y + line_length * np.sin(theta)]]))
    return np.array(lines)


def draw_trajectory_A_2d(lm, odo, kf):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.grid(True)
    lm_orientations = xyTheta2Line(lm, 0.05)
    ax.scatter(lm[:, 0], lm[:, 1], color='crimson', s=6)
    ax.scatter(odo[:, 0], odo[:, 1], color='deepskyblue', s=6)
    ax.scatter(kf[:, 0], kf[:, 1], color='mediumpurple', s=6)
    for i in lm_orientations:
        ax.plot(i[:, 0], i[:, 1], 'gray', linewidth=2, alpha=0.4)
    odo_orientations = xyTheta2Line(odo, 0.05)
    for i in odo_orientations:
        ax.plot(i[:, 0], i[:, 1], 'gray', linewidth=2, alpha=0.4)
    kf_orientations = xyTheta2Line(kf, 0.05)
    for i in kf_orientations:
        ax.plot(i[:, 0], i[:, 1], 'gray', linewidth=2, alpha=0.4)
    ax.set_aspect('equal')
    ax.legend(['noisy landmark', 'noisy odometry', 'kalman filter', 'orientation'])
    ax.set_title('Trajectory Comparison of Part A')
    plt.savefig('trajectory_A.png', dpi=400, bbox_inches='tight', pad_inches=0.1)
     

def draw_trajectory_B_2d(lm, kf):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].scatter(lm[:, 0], lm[:, 1], color='crimson', s=6)
    ax[1].scatter(kf[:, 0], kf[:, 1], color='mediumpurple', s=6)
    lm_orientations = xyTheta2Line(lm, 5)
    for i in lm_orientations:
        ax[0].plot(i[:, 0], i[:, 1], 'gray', linewidth=2, alpha=0.4)
    kf_orientations = xyTheta2Line(kf, 5)
    for i in kf_orientations:
        ax[1].plot(i[:, 0], i[:, 1], 'gray', linewidth=2, alpha=0.4)
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[0].legend(['noisy landmark', 'orientation'])
    ax[1].legend(['kalman filter', 'orientation'])
    ax[0].set_title('Landmark Trajectory of Part B')
    ax[1].set_title('Kalman Filter Trajectory of Part B')
    ax[0].set_xlim(0, 35)
    ax[0].set_ylim(0, 70)
    ax[1].set_xlim(0, 35)
    ax[1].set_ylim(0, 70)
    plt.savefig('trajectory_B.png', dpi=400, bbox_inches='tight', pad_inches=0.1)


def draw_velocity_B(noise_v, kf_v):
    assert noise_v.shape[1] == kf_v.shape[1] == 3
    noise_axis_x = np.arange(noise_v.shape[0])
    kf_axis_x = np.arange(kf_v.shape[0])
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)

    # plot x speed
    ax[0].scatter(noise_axis_x, noise_v[:, 0], color='crimson', label='noisy observation')
    ax[0].scatter(kf_axis_x, kf_v[:, 0], color='mediumpurple', label='kalman filter')
    ax[0].legend()
    ax[0].set_title('x speed')
    ax[0].set_xlabel('time step')

    # plot y speed
    ax[1].scatter(noise_axis_x, noise_v[:, 1], color='crimson', label='noisy observation')
    ax[1].scatter(kf_axis_x, kf_v[:, 1], color='mediumpurple', label='kalman filter')
    ax[1].legend()
    ax[1].set_title('y speed')
    ax[1].set_xlabel('time step')

    # plot theta speed
    ax[2].scatter(noise_axis_x, noise_v[:, 2], color='crimson', label='noisy observation')
    ax[2].scatter(kf_axis_x, kf_v[:, 2], color='mediumpurple', label='kalman filter')
    ax[2].legend()
    ax[2].set_title('angular speed')
    ax[2].set_xlabel('time step')

    plt.savefig('velocity_B.png', dpi=400, bbox_inches='tight', pad_inches=0.1)





if __name__ == '__main__':
    lm_A_traj = np.loadtxt('./hw2_dataset/landmark_A.txt')
    odo_A = np.loadtxt('./hw2_dataset/odometry.txt')
    odo_A_traj = get_odometry_states(odo_A, init_state=lm_A_traj[0])
    kf_A_traj = ekf_A(lm_A_traj, odo_A)
    draw_trajectory_A_2d(lm_A_traj, odo_A_traj, kf_A_traj)

    lm_B_traj = np.loadtxt('./hw2_dataset/landmark_B.txt')
    kf_B_traj = ekf_B(lm_B_traj)
    draw_trajectory_B_2d(lm_B_traj, kf_B_traj[:, :3])
    draw_velocity_B(lm_B_traj[1:] - lm_B_traj[:-1], kf_B_traj[:, 3:])


