import numpy as np
from tqdm import tqdm

def ekf_A(obs_sequence: np.ndarray, odo_sequence: np.ndarray):
    """
        obs_sequence: from z_0 to z_N, (x, y, theta)
        odo_sequence: from u_0 to u_N-1, (delta_1, delta_t, delta_2)
    """
    status_update_list = [obs_sequence[0].reshape((3, 1))]
    Q = np.eye(3); Q[(0, 2), (0, 2)] = 0.02**2; Q[1, 1] = 0.5**2 
    R = np.eye(3); R[(0, 1), (0, 1)] = 0.05**2; R[2, 2] = 0.03**2
    P_update = R.copy()
    N = len(odo_sequence)
    for t in tqdm(range(N)): # t = 0, 1, 2, ..., N

        # data prepare
        u = odo_sequence[t].reshape((3, 1)) # u = [delta_1, delta_t, delta_2]
        z = obs_sequence[t+1].reshape((3, 1))
        angle = u[0] + status_update_list[-1][2] # angle = delta_1 + theta
        cos_term = u[1] * np.cos(angle)
        sin_term = u[1] * np.sin(angle)

        # predict
        status_pred = np.array(
            (
                status_update_list[-1][0] + cos_term,
                status_update_list[-1][1] + sin_term,
                angle + u[2]
            )
        ).reshape((3, 1))
        F = np.eye(3); F[0, 2] = -sin_term; F[1, 2] = cos_term
        L = np.eye(3); L[(0, 1), 0] = F[(0, 1), 2].copy()
        L[2, 0] = 1
        L[0, 1] = cos_term; L[1, 1] = sin_term
        P_pred = F @ P_update @ F.T + L @ Q @ L.T

        # update
        y = z - status_pred
        S = P_pred + R
        K = P_pred @ np.linalg.inv(S)
        status_update_list.append(status_pred + K @ y)
        P_update = (np.eye(3) - K) @ P_pred
    
    return np.array(status_update_list).squeeze()


def ekf_B(obs_sequence: np.ndarray):
    """
        obs_sequence: (x, y, theta)
    """
    status_update_list = [
        np.hstack((obs_sequence[0].reshape((3, 1)), (obs_sequence[1] - obs_sequence[0]).reshape((3, 1)))).reshape((6, 1))
    ]
    Q = np.eye(6); Q[(0, 1, 2), (0, 1, 2)] = 0.0; Q[(3, 4), (3, 4)] = 0.05**2; Q[5, 5] = 0.02**2
    R = np.eye(3); R *= 0.05**2
    H = np.hstack((np.eye(3), np.eye(3)))
    P_update = np.eye(6); P_update[(0, 1, 2), (0, 1, 2)] = 0.05**2; P_update[(3, 4, 5), (3, 4, 5)] = 2 * 0.05**2
    N = len(obs_sequence)
    for t in tqdm(range(1, N)): # t = 1, 2, ..., N

        # data prepare
        z = obs_sequence[t].reshape((3, 1))

        # predict
        status_pred = np.array(
            (
                status_update_list[-1][0] + status_update_list[-1][3],
                status_update_list[-1][1] + status_update_list[-1][4],
                status_update_list[-1][2] + status_update_list[-1][5],
                status_update_list[-1][3],
                status_update_list[-1][4],
                status_update_list[-1][5]
            )
        ).reshape((6, 1))
        F = np.eye(6); F[(0, 1, 2), (3, 4, 5)] = 1
        P_pred = F @ P_update @ F.T + Q

        # update
        h_status_pred = np.array(
            (
                status_pred[0] + status_pred[3],
                status_pred[1] + status_pred[4],
                status_pred[2] + status_pred[5]
            )
        ).reshape((3, 1))
        y = z - h_status_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        status_update = status_pred + K @ y
        status_update_list.append(status_update)
        P_update = (np.eye(6) - K @ H) @ P_pred
    
    return np.array(status_update_list).squeeze()

