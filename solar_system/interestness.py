import numpy as np

from numpy.typing import NDArray

from solar_system.simulation import Body


WINDOW_DAYS = 90


def calculate_trajectory_interestness(traj: NDArray, control_traj: NDArray, t_step: float):
    window = int(WINDOW_DAYS / t_step)
    traj_len = traj.shape[0]
    traj_interestness = np.zeros((traj.shape[0]))
    for window_end in range(window, traj_len):
        tp = traj[window_end - window : window_end]
        control_tp = control_traj[window_end - window : window_end]
        
        delta_x_mat = tp[:, 0].reshape((-1, 1)) - control_tp[:, 0]
        delta_y_mat = tp[:, 1].reshape((-1, 1)) - control_tp[:, 1]
        r_mat = np.sqrt(delta_x_mat ** 2 + delta_y_mat ** 2)
        traj_interestness[window_end] = np.mean(np.min(r_mat, axis=1))  # mean trajectory point divergence = interestness
    traj_interestness[: window] = traj_interestness[window]
    return traj_interestness


def rate_bodies(bodies: list[Body], control_bodies: list[Body], t_step: float):
    control_bodies_by_name = {b.name: (b, i) for i, b in enumerate(control_bodies)}
    for b in bodies:
        cb, _ = control_bodies_by_name[b.control_body_name]
        b.trajectory_interestness = calculate_trajectory_interestness(b.trajectory, cb.trajectory, t_step)

    # for i_step in range(bodies[0].trajectory_length):

    #     def _dist_to_closest(bodies: list[Body]):
    #         positions = np.concatenate([b.position(i_step).reshape((1, -1)) for b in bodies])
    #         delta_x_mat = positions[:, 0].reshape((-1, 1)) - positions[:, 0]
    #         delta_y_mat = positions[:, 1].reshape((-1, 1)) - positions[:, 1]
    #         r_mat = np.sqrt(delta_x_mat ** 2 + delta_y_mat ** 2)
    #         r_mat[r_mat < 1e-6] = np.inf
    #         return np.min(r_mat, axis=1)
        
    #     d2c_control = _dist_to_closest(control_bodies)
    #     d2c = _dist_to_closest(bodies)
    #     for i_body in range(len(bodies)):
    #         body = bodies[i_body]
    #         _, i_control_body = control_bodies_by_name[body.control_body_name]
    #         body.trajectory_interestness[i_step] *= d2c_control[i_control_body] / d2c[i_body]
