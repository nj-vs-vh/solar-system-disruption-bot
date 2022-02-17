from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import date
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter
from scipy import interpolate
from pathlib import Path

from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.text import Text

from numpy.typing import NDArray

from solar_system.simulation import Body, concatenate_trajectories


FIGSIZE = (10, 10)


def plot_orbits(bodies: list[Body]):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for b in bodies:
        ax.scatter([b.r_0[0]], [b.r_0[1]], color=b.color, marker="x")
        ax.plot(b.x_traj, b.y_traj, label=b.name, color=b.color)
        ax.scatter([b.x_traj[-1]], [b.y_traj[-1]], s=b.marker_size, c=b.color, marker="o")
    ax.axis("equal")
    ax.legend()
    plt.savefig("orbits.png", bbox_inches="tight")


@dataclass
class DrawedBody:
    body: Body
    traj_line_major: Line2D
    traj_line_minor: Line2D
    body_dot: Circle

    @classmethod
    def from_body(cls, b: Body, ax: plt.Axes):
        (traj_line_major,) = ax.plot([b.r_0[0]], [b.r_0[1]], color=b.color, linewidth=1.5, animated=True, visible=False)
        (traj_line_minor,) = ax.plot([b.r_0[0]], [b.r_0[1]], color=b.color, linewidth=0.6, animated=True, visible=False)
        body_dot = Circle((b.r_0), radius=b.radius, color=b.color, animated=True, visible=False)
        ax.add_patch(body_dot)
        return DrawedBody(b, traj_line_major, traj_line_minor, body_dot)

    @property
    def artists(self) -> list[Artist]:
        return [self.traj_line_major, self.body_dot]


@dataclass
class Camera:
    center: NDArray
    full_halfwidth: float
    scale: float

    @classmethod
    def from_bodies(cls, bodies: list[Body]) -> Camera:
        trajectories = concatenate_trajectories(bodies)
        max_radius = np.max(np.sqrt(trajectories[:, 0] ** 2 + trajectories[:, 1] ** 2))
        return Camera(
            center=np.array([0, 0]),
            full_halfwidth=max_radius,
            scale=1.05,
        )

    def apply(self, ax: plt.Axes):
        x_c = self.center[0]
        y_c = self.center[1]
        halfwidth = self.full_halfwidth * self.scale
        ax.set_xlim(x_c - halfwidth, x_c + halfwidth)
        ax.set_ylim(y_c - halfwidth, y_c + halfwidth)

    def create_movement(
        self,
        bodies_disrupted: list[Body],
        bodies_calm: list[Body],
        init_scale: float,
        steps_per_frame: int,
    ) -> NDArray:  # (nsteps, 3): x_c, y_c, scale
        def mean_relative_r_from_bodies(bodies: list[Body]):
            calm_positions = np.concatenate([b.position(0).reshape((1, -1)) for b in bodies])
            calm_r_from_com = np.sqrt(calm_positions[:, 0] ** 2 + calm_positions[:, 1] ** 2)
            return np.mean(calm_r_from_com / calm_r_from_com[-1])

        mean_relative_r_calm_start = mean_relative_r_from_bodies(bodies_calm)
        mean_relative_r_calm_end = mean_relative_r_from_bodies(bodies_calm[:7])  # up to Jupiter

        print(f"Max mean relative R: {mean_relative_r_calm_start} -> {mean_relative_r_calm_end}")

        bodies_disrupted = [b for b in bodies_disrupted if not b.name.startswith("Visitor")]

        def get_center(coord: int) -> tuple[NDArray, NDArray]:
            traj_sample = np.zeros((bodies_disrupted[0].trajectory_length, len(bodies_disrupted)))
            masses = np.array([b.m for b in bodies_disrupted])
            for i_body, b in enumerate(bodies_disrupted):
                traj_sample[:, i_body] = b.trajectory[:, coord]

            com = np.sum(traj_sample * masses, axis=1) / masses.sum()
            traj_samle_relative_to_com = traj_sample - com.reshape((-1, 1))
            return com, traj_samle_relative_to_com

        x_com, x_traj_sample_rel = get_center(0)
        y_com, y_traj_sample_rel = get_center(1)
        x_com[0] = 0.0
        y_com[0] = 0.0

        r_from_com_sample = np.sqrt(x_traj_sample_rel**2 + y_traj_sample_rel**2)

        r_camera = []
        for i_step in range(bodies_disrupted[0].trajectory_length):
            r_bodies = np.sort(r_from_com_sample[i_step, :])
            #                       sum of R up to ith     n umber of summed terms       normalizing terms to max R
            mean_relative_r_up_to = np.cumsum(r_bodies) / (1 + np.arange(len(r_bodies))) / r_bodies
            min_mean_relative_r = mean_relative_r_calm_start + (i_step / bodies_disrupted[0].trajectory_length) * (
                mean_relative_r_calm_end - mean_relative_r_calm_start
            )
            min_mean_relative_r *= 0.6
            r_bodies_in_frame = r_bodies[mean_relative_r_up_to > min_mean_relative_r]
            r_camera.append(r_bodies_in_frame.max())

        r_camera = np.array(r_camera)

        scale = r_camera / self.full_halfwidth
        scale[: 2 * steps_per_frame] = init_scale
        scale[scale > 1] = 1 + (scale[scale > 1] - 1) * 0.1

        movement = np.concatenate(
            [
                x_com.reshape((-1, 1)),
                y_com.reshape((-1, 1)),
                scale.reshape((-1, 1)),
            ],
            axis=1,
        )

        def smooth(param: NDArray, spline_points: int, is_quadratic: bool = False) -> NDArray:
            length = len(param)
            MOVING_AVERAGE_WINDOW = 5 * steps_per_frame  # simulation steps
            param_ma = np.convolve(param, np.ones(MOVING_AVERAGE_WINDOW) / MOVING_AVERAGE_WINDOW, 'valid')
            valid_ma_start = int(MOVING_AVERAGE_WINDOW / 2)
            param[valid_ma_start : valid_ma_start + len(param_ma)] = param_ma
            indices = np.arange(0, length)
            sample_idx = np.linspace(0, length - 1, spline_points).astype(np.int64)
            return interpolate.interp1d(
                x=indices[sample_idx], y=param[sample_idx], kind='quadratic' if is_quadratic else 'linear'
            )(indices)

        movement[:, 0] = smooth(movement[:, 0], spline_points=8, is_quadratic=True)
        movement[:, 1] = smooth(movement[:, 1], spline_points=8, is_quadratic=True)
        movement[:, 2] = smooth(movement[:, 2], spline_points=4, is_quadratic=False)
        return movement


def animate_trajectories(
    bodies_calm: list[Body],
    bodies_disrupted: list[Body],
    t_step: float,
    days_per_frame: float,
    output_file: Path,
):
    fig = plt.figure(figsize=FIGSIZE, frameon=False)

    ax: plt.Axes = fig.add_axes((0, 0, 1, 1), facecolor="k")

    time_text = fig.text(0.01, 0.01, "", ha="left", va="bottom", color="w", fontfamily="Ubuntu")
    fig.text(
        0.99,
        0.01,
        "All sounds stolen from 65dos Wreckage Systems stream",
        ha="right",
        va="bottom",
        color="w",
        fontfamily="Ubuntu",
    )

    dbs_calm = [DrawedBody.from_body(b, ax) for b in bodies_calm]
    dbs_disrupted = [DrawedBody.from_body(b, ax) for b in bodies_disrupted]

    camera = Camera.from_bodies(bodies_calm)
    camera_scale_calm_start = 1 / camera.full_halfwidth  # starting at 1 AU
    camera_scale_calm_end = 1  # ending at full solar system
    camera.apply(ax)

    camera_movement = camera.create_movement(
        bodies_disrupted, bodies_calm, init_scale=camera_scale_calm_end, steps_per_frame=int(days_per_frame / t_step)
    )

    n_calm_steps = bodies_calm[0].trajectory_length - 1
    n_disr_steps = bodies_disrupted[0].trajectory_length
    frame_count = int((n_calm_steps + n_disr_steps) * t_step / days_per_frame)

    def update(frame: int):
        if frame % 30 == 0:
            print(f"{100 * frame / frame_count : .0f} %")

        current_day = frame * days_per_frame
        simulation_date = date.fromordinal(int(1 + current_day))
        time_text.set_text(f"Y {simulation_date.year : <4} M {simulation_date.month : <4} D {simulation_date.day}")

        current_step_global = int(current_day / t_step)
        for dbs, starts_at_step, length in zip(
            (dbs_calm, dbs_disrupted), (0, n_calm_steps), (n_calm_steps, n_disr_steps)
        ):
            current_step_local = current_step_global - starts_at_step
            if current_step_local < 0:
                continue
            for db in dbs:
                body_v0_abs = np.sqrt(np.sum(db.body.v_0**2))
                if body_v0_abs < 1e-3:
                    major_traj_line_len = 180
                else:
                    major_traj_line_len = np.pi * np.sqrt(np.sum(db.body.r_0**2)) / body_v0_abs
                major_line_start_step = max(min(current_step_local - int(major_traj_line_len / t_step), length), 0)
                # if major_line_start_step == length:
                #     continue
                db.traj_line_major.set(
                    data=(
                        db.body.x_traj[major_line_start_step : min(current_step_local, length - 1)],
                        db.body.y_traj[major_line_start_step : min(current_step_local, length - 1)],
                    ),
                    visible=current_step_local < length - 1,
                )
                db.traj_line_minor.set(
                    data=(
                        db.body.x_traj[: min(current_step_local, length) + 1],
                        db.body.y_traj[: min(current_step_local, length) + 1],
                    ),
                    visible=True,
                )
                db.body_dot.set(
                    center=(
                        db.body.x_traj[min(current_step_local, length - 1)],
                        db.body.y_traj[min(current_step_local, length - 1)],
                    ),
                    visible=current_step_local < length - 1,
                )

        if current_step_global < n_calm_steps:
            camera.scale = np.exp(
                np.log(camera_scale_calm_start)
                + (np.log(camera_scale_calm_end) - np.log(camera_scale_calm_start))
                * current_step_global
                / (n_calm_steps)
            )
            camera.apply(ax)
        else:
            x_c, y_c, scale = camera_movement[current_step_global - n_calm_steps, :]
            camera.center = np.array([x_c, y_c])
            camera.scale = scale
            camera.apply(ax)

    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=range(frame_count),
        blit=False,
    )

    writer = FFMpegWriter(fps=30, metadata=dict(artist="njvsvh"), bitrate=1800)
    anim.save(output_file, writer=writer)
