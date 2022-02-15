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

    def create_timeline(self, bodies: list[Body], init_scale: float) -> NDArray:  # (nsteps, 3): x_c, y_c, scale
        bodies = [b for b in bodies if not b.name.startswith("Visitor")]

        def get_center_range(coord: int):
            traj_sample = np.zeros((bodies[0].trajectory_length, len(bodies)))
            masses = np.array([b.m for b in bodies])
            for i_body, b in enumerate(bodies):
                traj_sample[:, i_body] = b.trajectory[:, coord]

            center_preliminary = np.mean(traj_sample * masses, axis=1) / masses.sum()
            MAX_DELTA = 25
            bad_mask = np.abs(traj_sample - center_preliminary.reshape((-1, 1))) > MAX_DELTA
            for i_step in range(bodies[0].trajectory_length):
                traj_sample[i_step, bad_mask[i_step, :]] = center_preliminary[i_step]
            median_masked = np.median(traj_sample, axis=1)
            max_delta_masked = 1.1 * np.max(median_masked.reshape((-1, 1)) - traj_sample, axis=1)
            MOVING_AVERAGE_WINDOW = 31
            median_masked = np.convolve(median_masked, np.ones(MOVING_AVERAGE_WINDOW) / MOVING_AVERAGE_WINDOW, 'same')
            max_delta_masked = np.convolve(max_delta_masked, np.ones(MOVING_AVERAGE_WINDOW) / MOVING_AVERAGE_WINDOW, 'same')
            return median_masked, max_delta_masked

        x_center, x_delta = get_center_range(0)
        y_center, y_delta = get_center_range(1)
        x_center[0] = 0.0
        y_center[0] = 0.0
        std = np.minimum(x_delta, y_delta)
        scale = init_scale * std / std[0]
        timeline = np.concatenate(
            [
                x_center.reshape((-1, 1)),
                y_center.reshape((-1, 1)),
                scale.reshape((-1, 1)),
            ],
            axis=1,
        )

        def smooth(param: NDArray) -> NDArray:
            length = len(param)
            SPLINE_POINTS = 3
            indices = np.arange(0, length)
            sample_idx = np.linspace(0, length - 1, SPLINE_POINTS).astype(np.int64)
            return interpolate.interp1d(x=indices[sample_idx], y=param[sample_idx], kind='quadratic')(indices)

        for i in range(3):
            timeline[:, i] = smooth(timeline[:, i])
        return timeline


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
        "sounds stolen from 65 Wreckage System stream",
        ha="right",
        va="bottom",
        color="w",
        fontfamily="Ubuntu",
    )

    dbs_calm = [DrawedBody.from_body(b, ax) for b in bodies_calm]
    dbs_disrupted = [DrawedBody.from_body(b, ax) for b in bodies_disrupted]

    camera = Camera.from_bodies(bodies_calm)
    camera_scale_calm_start = 1.01 / camera.full_halfwidth
    camera_scale_calm_end = 1.01
    camera.apply(ax)

    camera_timeline = camera.create_timeline(bodies_disrupted, init_scale=camera_scale_calm_end)

    n_calm_steps = bodies_calm[0].trajectory_length - 1
    n_disr_steps = bodies_disrupted[0].trajectory_length
    frame_count = int((n_calm_steps + n_disr_steps) * t_step / days_per_frame)

    def update(frame: int):
        if frame % 30 == 0:
            print(f"{100 * frame / frame_count : .0f} %")

        current_day = frame * days_per_frame
        simulation_date = date.fromordinal(int(1 + current_day))
        time_text.set_text(f"Y {simulation_date.year : <6} M {simulation_date.month : <6} D {simulation_date.day}")

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
                if major_line_start_step == length:
                    continue
                db.traj_line_major.set(
                    data=(
                        db.body.x_traj[major_line_start_step : min(current_step_local, length - 1)],
                        db.body.y_traj[major_line_start_step : min(current_step_local, length - 1)],
                    ),
                    visible=current_step_local < length,
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
                    visible=current_step_local < length,
                )

        if current_step_global < n_calm_steps:
            camera.scale = np.exp(
                np.log(camera_scale_calm_start)
                + (np.log(camera_scale_calm_end) - np.log(camera_scale_calm_start))
                * current_step_global
                / (n_calm_steps - 10)
            )
            camera.apply(ax)
        else:
            x_c, y_c, scale = camera_timeline[current_step_global - n_calm_steps, :]
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
