from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from typing import Optional
from numpy.typing import NDArray


Vector = NDArray[np.float64]


@dataclass
class Body:
    name: str
    color: str
    m: float  # in M☉
    r_0: Vector  # in AU
    v_0: Vector  # in AU / yr
    
    trajectory: Optional[NDArray] = None

G = 39.478  # AU^3 M☉^-1 yr^-2


def random_body_state_on_elliptical_orbit(
    a: float,
    ecc: float,
    inverse: bool = False,
    center: Vector = np.array([0.0, 0.0]),
    center_mass: float = 1.0,
) -> tuple[Vector, Vector]:
    # working in "local" coordinate system = centered in the central body
    def random_angle() -> float:
        return np.random.random() * 2 * np.pi

    major_axis_angle = random_angle()  # oribital ellips is oriented randomly
    orbit_phase = random_angle()
    r = (
        a * (1 - ecc**2) / (1 + ecc * np.cos(orbit_phase))
    )  # ellipse polar form relative to focus
    phi = major_axis_angle + orbit_phase
    r_0_local = np.array([r * np.cos(phi), r * np.sin(phi)])

    linear_ecc = a * ecc
    b = a * np.sqrt(1 - ecc**2)
    x_p_canonic = r * np.cos(orbit_phase) - linear_ecc
    y_p_canonic = r * np.sin(orbit_phase)
    tangent_angle_canonic = np.arctan(- (x_p_canonic * b ** 2) / (y_p_canonic * a ** 2))
    if y_p_canonic < 0:
        tangent_angle_canonic += np.pi
    tangent_angle = tangent_angle_canonic + major_axis_angle
    v = np.sqrt(G * center_mass * (2 / r - 1 / a))  # vis-viva equation
    v_0 = np.array([v * np.cos(tangent_angle), v * np.sin(tangent_angle)])
    if inverse:
        v_0 = -1 * v_0

    r_0_global = center + r_0_local
    return r_0_global, v_0


SOLAR_SYSTEM = [
    Body(
        "Sun",
        "#ffed69",
        1.0,
        r_0=np.array([0.0, 0.0]),
        v_0=np.array([0.0, 0.0]),
    ),
    Body(
        "Mercury",
        "#ff7a66",
        1.652e-7,
        *random_body_state_on_elliptical_orbit(a=0.387, ecc=0.206),
    ),
    Body(
        "Venus",
        "#de8d00",
        2.45e-6,
        *random_body_state_on_elliptical_orbit(a=0.723, ecc=0.007),
    )
]


def simulate_trajectories(bodies: list[Body], dt: float, n_steps: int):
    """Credit: _calc_trajectories
    @ https://github.com/robolamp/3_body_problem_bot/blob/master/generate_3_body_simulation.py
    """
    n_bodies = len(bodies)

    m_vec = np.array([body.m for body in bodies])
    m_col = m_vec.reshape((-1, 1))
    m_mat = m_col @ m_col.T  # m_mat[i][j] = m_i * m_j

    def encode_state(x: NDArray, y: NDArray, xdot: NDArray, ydot: NDArray) -> NDArray:
        return np.concatenate([x, y, xdot, ydot])

    def decode_state(state: NDArray) -> tuple[NDArray, ...]:
        x = state[0:n_bodies]
        y = state[n_bodies:2 *n_bodies]
        xdot = state[2 * n_bodies:3 *n_bodies]
        ydot = state[3 * n_bodies:]
        return x, y, xdot, ydot

    def newtons_law(t: float, state: NDArray) -> NDArray:
        x, y, xdot, ydot = decode_state(state)

        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        delta_x_mat = x @ x.T
        delta_y_mat = y @ y.T
        dist_mat = np.sqrt(delta_x_mat ** 2 + delta_y_mat ** 2)

        def acceleration(delta_coords_mat: NDArray):
            forces = G * m_mat * delta_coords_mat / (dist_mat ** 3)
            # no self-action!
            forces[np.isnan(forces)] = 0.
            forces[np.isinf(forces)] = 0.
            return np.sum(forces, axis=1) / m_vec

        return encode_state(
            x=xdot,
            y=ydot,
            xdot=acceleration(delta_x_mat),
            ydot=acceleration(delta_y_mat),
        )

    solution = solve_ivp(
        fun=newtons_law,
        t_span=[0, dt * n_steps],
        t_eval=np.linspace(0, dt * n_steps, n_steps),
        y0=encode_state(
            x=np.array([b.r_0[0] for b in bodies]),
            y=np.array([b.r_0[1] for b in bodies]),
            xdot=np.array([b.v_0[0] for b in bodies]),
            ydot=np.array([b.v_0[1] for b in bodies]),
        ),
        atol=1e-6,
        method="RK45",  # DOP853
    )

    if not solution.success:
        raise RuntimeError(solution.message)
    for i, body in enumerate(bodies):
        traj_x: NDArray = solution.y[i, :]
        traj_y: NDArray = solution.y[n_bodies + i, :]
        body.trajectory = np.concatenate(
            [traj_x.reshape((-1, 1)), traj_y.reshape((-1, 1))],
            axis=1,
        )


if __name__ == "__main__":
    bodies = SOLAR_SYSTEM
    simulate_trajectories(bodies, 1 / 365, 365)
    print(bodies[1].trajectory)
