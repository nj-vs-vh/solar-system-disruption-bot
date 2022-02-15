from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from typing import Optional
from numpy.typing import NDArray


Vector = NDArray[np.float64]


NO_VALUE = object()


@dataclass
class Body:
    name: str
    color: str
    m: float  # in M☉
    r_0: Vector  # in AU
    v_0: Vector  # in AU / yr

    radius: float = None  # plotting radius, not physical!
    control_body_name: Optional[str] = NO_VALUE

    trajectory: Optional[NDArray] = None  # size: (nsteps x 4), 0 - x, 1 - y, 2 - vx, 3 - vy
    trajectory_interestness: Optional[NDArray] = None

    def __post_init__(self):
        if self.radius is None:
            sun_plot_radius = 0.2
            sun_real_radius = 1.0
            mercury_plot_radius = 0.05
            mercury_real_radius = 1.65e-7 ** (1 / 3)  # up to a constant

            radius_real = self.m ** (1 / 3)

            self.radius = np.interp(
                [np.log(radius_real)],
                [np.log(mercury_real_radius), np.log(sun_real_radius)],
                [mercury_plot_radius, sun_plot_radius],
            )
        if self.control_body_name is NO_VALUE:
            self.control_body_name = self.name

    def successor(self) -> Body:
        self._assert_trajectory_calculated()
        return Body(
            name=self.name,
            color=self.color,
            m=self.m,
            r_0=self.trajectory[-1, 0:2],
            v_0=self.trajectory[-1, 2:],
        )

    @property
    def marker_size(self) -> float:
        return 1e4 * self.radius**2

    def _assert_trajectory_calculated(self):
        assert self.trajectory is not None, f"Trajectory is not calculated for body {self}"

    @property
    def x_traj(self) -> NDArray:
        self._assert_trajectory_calculated()
        return self.trajectory[:, 0]

    @property
    def y_traj(self) -> NDArray:
        self._assert_trajectory_calculated()
        return self.trajectory[:, 1]

    @property
    def trajectory_length(self) -> int:
        self._assert_trajectory_calculated()
        return self.trajectory.shape[0]

    def position(self, step: int) -> NDArray:
        return self.trajectory[step, 0:2]


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
    r = a * (1 - ecc**2) / (1 + ecc * np.cos(orbit_phase))  # ellipse polar form relative to focus
    phi = major_axis_angle + orbit_phase
    r_0_local = np.array([r * np.cos(phi), r * np.sin(phi)])

    linear_ecc = a * ecc
    b = a * np.sqrt(1 - ecc**2)
    x_p_canonic = r * np.cos(orbit_phase) - linear_ecc
    y_p_canonic = r * np.sin(orbit_phase)
    tangent_angle_canonic = np.arctan(-(x_p_canonic * b**2) / (y_p_canonic * a**2))
    if y_p_canonic < 0:
        tangent_angle_canonic += np.pi
    tangent_angle = tangent_angle_canonic + major_axis_angle
    v = np.sqrt(G * center_mass * (2 / r - 1 / a))  # vis-viva equation
    # -1 is for counter-clockwise rotation
    v_0 = -1 * np.array([v * np.cos(tangent_angle), v * np.sin(tangent_angle)])
    if inverse:
        v_0 = -1 * v_0

    r_0_global = center + r_0_local
    return r_0_global, v_0


SOLAR_SYSTEM = [
    Body(
        "Sun",
        "#FEB238",
        1.0,
        r_0=np.array([0.0, 0.0]),
        v_0=np.array([0.0, 0.0]),
    ),
    Body(
        "Mercury",
        "#7F7A79",
        1.65e-7,
        *random_body_state_on_elliptical_orbit(a=0.387, ecc=0.206),
    ),
    Body(
        "Venus",
        "#C8A972",
        2.45e-6,
        *random_body_state_on_elliptical_orbit(a=0.723, ecc=0.007),
    ),
    Body(
        "Earth",
        "#4A6F99",
        3.45e-6,
        *random_body_state_on_elliptical_orbit(a=1.0, ecc=0.017),
    ),
    Body(
        "Mars",
        "#FD8660",
        3.21e-7,
        *random_body_state_on_elliptical_orbit(a=1.523, ecc=0.093),
    ),
    Body(
        "Jupiter",
        "#BE955F",
        9.55e-4,
        *random_body_state_on_elliptical_orbit(a=5.204, ecc=0.0489),
    ),
    Body(
        "Saturn",
        "#F3E2AE",
        2.86e-4,
        *random_body_state_on_elliptical_orbit(a=9.583, ecc=0.0565),
    ),
    Body(
        "Uranus",
        "#66747D",
        4.36e-5,
        *random_body_state_on_elliptical_orbit(a=19.191, ecc=0.047),
    ),
    Body(
        "Neptune",
        "#61739C",
        5.15e-5,
        *random_body_state_on_elliptical_orbit(a=30.07, ecc=0.008),
    ),
]


def simulate_trajectories(bodies: list[Body], t_step: float, t_total_yrs: float):
    """Credit: _calc_trajectories
    @ https://github.com/robolamp/3_body_problem_bot/blob/master/generate_3_body_simulation.py

    Args:
        bodies (list[Body])
        t_step_days (float): simulation step in days
        t_total_yrs (float): total simulation time in years
    """
    n_bodies = len(bodies)

    m_vec = np.array([body.m for body in bodies])
    m_col = m_vec.reshape((-1, 1))
    m_prod_mat = m_col @ m_col.T  # m_prod_mat[i][j] = m_i * m_j

    def encode_state(x: NDArray, y: NDArray, xdot: NDArray, ydot: NDArray) -> NDArray:
        return np.concatenate([x, y, xdot, ydot])

    def decode_state(state: NDArray) -> tuple[NDArray, ...]:
        x = state[0:n_bodies]
        y = state[n_bodies : 2 * n_bodies]
        xdot = state[2 * n_bodies : 3 * n_bodies]
        ydot = state[3 * n_bodies :]
        return x, y, xdot, ydot

    def newtons_law(t: float, state: NDArray) -> NDArray:
        x, y, xdot, ydot = decode_state(state)

        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        delta_x_mat = x - x.T
        delta_y_mat = y - y.T
        dist_mat = np.sqrt(delta_x_mat**2 + delta_y_mat**2)

        def acceleration(delta_coords_mat: NDArray):
            force_components = -G * m_prod_mat * delta_coords_mat / (dist_mat**3)
            # no self-action!
            force_components[np.isnan(force_components)] = 0.0
            force_components[np.isinf(force_components)] = 0.0
            return np.sum(force_components, axis=1) / m_vec

        return encode_state(
            x=xdot,
            y=ydot,
            xdot=acceleration(delta_x_mat),
            ydot=acceleration(delta_y_mat),
        )

    n_steps = int(t_total_yrs * 365 / t_step)
    solution = solve_ivp(
        fun=newtons_law,
        t_span=[0, t_total_yrs],
        t_eval=np.linspace(0, t_total_yrs, n_steps),
        y0=encode_state(
            x=np.array([b.r_0[0] for b in bodies]),
            y=np.array([b.r_0[1] for b in bodies]),
            xdot=np.array([b.v_0[0] for b in bodies]),
            ydot=np.array([b.v_0[1] for b in bodies]),
        ),
        atol=1e-5,
        method="DOP853",
    )

    if not solution.success:
        raise RuntimeError(solution.message)
    x, y, xdot, ydot = decode_state(solution.y)
    for i, body in enumerate(bodies):
        body.trajectory = np.concatenate(
            [
                x[i, :].reshape((-1, 1)),
                y[i, :].reshape((-1, 1)),
                xdot[i, :].reshape((-1, 1)),
                ydot[i, :].reshape((-1, 1)),
            ],
            axis=1,
        )


def _distance_matrix(x: NDArray, y: NDArray) -> NDArray:
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    delta_x_mat = x - x.T
    delta_y_mat = y - y.T
    return np.sqrt(delta_x_mat**2 + delta_y_mat**2)


def concatenate_trajectories(bodies: list[Body]) -> NDArray:
    traj_x_all = np.empty((0,))
    traj_y_all = np.empty((0,))
    for b in bodies:
        traj_x_all = np.concatenate([traj_x_all, b.x_traj])
        traj_y_all = np.concatenate([traj_y_all, b.x_traj])
    return np.concatenate([traj_x_all.reshape((-1, 1)), traj_y_all.reshape((-1, 1))], axis=1)


def random_between(min_: float, max_: float) -> float:
    return min_ + np.random.random() * (max_ - min_)


def generate_disruption(bodies: list[Body], t_disruption_years: float) -> tuple[list[Body], list[Body], str]:
    # will be simulated further without disruption
    control = [b.successor() for b in bodies]
    to_disrupt = [b.successor() for b in bodies]
    disrupted = to_disrupt.copy()

    SPLIT_PROB = 0.5
    MAX_SPLIT_TO = 7
    MAX_SPLIT_BODIES = 5
    SPLIT_ENERGY_MIN = 0.3  # relative to debris potential energy after split
    SPLIT_ENERGY_MAX = 0.9

    MAX_VISITORS = 4
    VISITOR_MIN_MASS = 0.1
    VISITOR_MAX_MASS = 2
    VISITOR_R_SPAWN_MIN = 45
    VISITOR_R_SPAWN_MAX = 70

    descriptions: list[str] = []

    split_bodies_info: list[tuple[str, int]] = []

    for _ in range(MAX_SPLIT_BODIES):
        if np.random.random() < SPLIT_PROB:
            # the sun is excluded here and cannot split!
            split_idx = np.random.randint(low=1, high=len(to_disrupt))
            disrupted.pop(split_idx)
            split_body = to_disrupt.pop(split_idx)
            n_split = np.random.randint(2, MAX_SPLIT_TO)
            split_bodies_info.append((split_body.name, n_split))
            debris_weights = np.random.random((n_split,))
            radii_debris: NDArray = ((split_body.radius**3) * debris_weights / debris_weights.sum()) ** (1 / 3)
            m_debris: NDArray = split_body.m * debris_weights / debris_weights.sum()
            m_debris = m_debris.reshape((-1, 1))
            # debris positions in the COM coordinate system, uniformly across r_start ring
            r_start = 0.01  # AU
            phis = np.linspace(0, 2 * np.pi, n_split + 1)[:-1]
            phi_step = phis[1] - phis[0]
            r_debris = np.concatenate(
                [
                    (r_start * np.cos(phis)).reshape((-1, 1)),
                    (r_start * np.sin(phis)).reshape((-1, 1)),
                ],
                axis=1,
            )
            # debris momenta in the COM coordinate system
            p_debris = np.zeros_like(r_debris)
            for i in range(n_split - 1):
                p_phi_min = phis[i] - phi_step * 0.45  # generally everyone is going outward
                p_phi_max = phis[i] + phi_step * 0.45
                p_phi = random_between(p_phi_min, p_phi_max)
                p_len = random_between(0, 1)
                p_debris[i, 0] = p_len * np.cos(p_phi)
                p_debris[i, 1] = p_len * np.sin(p_phi)
            p_debris[-1, :] = -(p_debris[:-1, :]).sum(axis=0)  # conservation of momentum

            m_prod_mat = m_debris @ m_debris.T
            dist_mat = _distance_matrix(r_debris[:, 0], r_debris[:, 1])
            potential_energy_contributions = -G * m_prod_mat / dist_mat
            potential_energy_contributions[np.isinf(potential_energy_contributions)] = 0.0
            potential_energy_contributions[np.isnan(potential_energy_contributions)] = 0.0
            potential_energy = potential_energy_contributions.sum()
            debris_kinetic_energy = -potential_energy * random_between(SPLIT_ENERGY_MIN, SPLIT_ENERGY_MAX)
            generated_kinetic_energy = (0.5 * (p_debris[:, 0] ** 2 + p_debris[:, 1] ** 2) / m_debris.T).sum()
            print(
                f"{split_body.name} shuttered into {n_split} pieces: "
                + f"{100 * debris_kinetic_energy / np.abs(potential_energy):.2f} "
                + "% of gravitational potential energy converted to kinetic"
            )
            p_debris = p_debris * np.sqrt(
                debris_kinetic_energy / generated_kinetic_energy
            )  # energy deposit from explosion

            for i in range(n_split):
                disrupted.append(
                    Body(
                        name=f"{split_body.name} debris #{i+1}",
                        color=split_body.color,
                        m=m_debris[i, 0],
                        r_0=r_debris[i, :] + split_body.r_0,
                        v_0=(p_debris[i, :] / m_debris[i]) + split_body.v_0,
                        radius=radii_debris[i],
                        control_body_name=split_body.name,
                    )
                )

    if split_bodies_info:
        first, *rest = split_bodies_info
        first_name, first_n_split = first
        split_strs = [f"{first_name.upper()} IS SHATTERED INTO {first_n_split} PIECES"]
        for name, n_split in rest:
            split_strs.append(f"{name.upper()} - INTO {n_split}")
        boom_emojis = ["💥", "💣", "✨"]
        descriptions.append(boom_emojis[np.random.randint(0, len(boom_emojis))] + " " + ", ".join(split_strs))

    n_visitors = np.random.randint(1, MAX_VISITORS)
    visitor_colors = ["#114a6f", "#459eb0", "#904449", "#db8183"][:n_visitors]
    visitor_masses = []
    for i in range(n_visitors):
        r_spawn = random_between(VISITOR_R_SPAWN_MIN, VISITOR_R_SPAWN_MAX)
        phi_spawn = random_between(0, 2 * np.pi)
        t_impact = 0.3 + np.random.random() * 0.5 * t_disruption_years
        v_spawn = r_spawn / t_impact
        delta_theta = np.arctan(30 / r_spawn)
        theta = phi_spawn + np.pi + random_between(-delta_theta, delta_theta)
        visitor_mass = np.exp(random_between(np.log(VISITOR_MIN_MASS), np.log(VISITOR_MAX_MASS)))
        visitor_masses.append(visitor_mass)
        print(f"Visitor with mass {visitor_mass}, R_spawn = {r_spawn}, v_spawn = {v_spawn}")
        disrupted.append(
            Body(
                name=f"Visitor #{i+1}",
                color=visitor_colors[i],
                m=visitor_mass,
                r_0=np.array([r_spawn * np.cos(phi_spawn), r_spawn * np.sin(phi_spawn)]),
                v_0=np.array([v_spawn * np.cos(theta), v_spawn * np.sin(theta)]),
                control_body_name=None,
            )
        )

    masses_str = ', '.join([f'{m:.1f}' for m in visitor_masses])
    if n_visitors == 1:
        visitors_str = f"VISITOR WITH MASS {masses_str}M☉"
    else:
        visitors_str = f"{n_visitors} VISITORS WITH MASSES {masses_str} M☉"
    visitors_emojis = ["🌠", "☄️", "🌀", "💫", "⭐"]
    descriptions.append(visitors_emojis[np.random.randint(0, len(visitors_emojis))] + " " + visitors_str)

    return disrupted, control, "\n".join(descriptions)


if __name__ == "__main__":
    bodies = SOLAR_SYSTEM[:3]
    simulate_trajectories(bodies, 1, 1)
    print(*generate_disruption(bodies))
