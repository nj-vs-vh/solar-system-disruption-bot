from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import date
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

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
        ax.scatter(
            [b.r_0[0]],
            [b.r_0[1]],
            color=b.color,
            marker="x",
        )
        ax.plot(
            b.x_traj,
            b.y_traj,
            label=b.name,
            color=b.color,
        )
        ax.scatter(
            [b.x_traj[-1]],
            [b.y_traj[-1]],
            s=b.marker_size,
            c=b.color,
            marker="o",
        )
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
        (traj_line_major,) = ax.plot(
            [b.r_0[0]], [b.r_0[1]], color=b.color, linewidth=1.0, animated=True
        )
        (traj_line_minor,) = ax.plot(
            [b.r_0[0]], [b.r_0[1]], color=b.color, linewidth=0.3, animated=True
        )
        body_dot = Circle((b.r_0), radius=b.radius, color=b.color, animated=True)
        ax.add_patch(body_dot)
        return DrawedBody(b, traj_line_major, traj_line_minor, body_dot)

    @property
    def artists(self) -> list[Artist]:
        return [self.traj_line_major, self.body_dot]


@dataclass
class Camera:
    center: tuple[float, float]
    full_halfwidth: float
    scale: float

    @classmethod
    def from_bodies(cls, bodies: list[Body], static: bool = False) -> Camera:
        if static:
            trajectories = concatenate_trajectories(bodies)
        else:
            trajectories = np.concatenate([b.position(0).reshape((1, -1)) for b in bodies])
        max_radius = np.max(np.sqrt(trajectories[:, 0] ** 2 + trajectories[:, 1] ** 2))
        return Camera(
            center=(0, 0),
            full_halfwidth=max_radius,
            scale=1.05,
        )

    def apply(self, ax: plt.Axes):
        x_c, y_c = self.center
        halfwidth = self.full_halfwidth * self.scale
        ax.set_xlim(x_c - halfwidth, x_c + halfwidth)
        ax.set_ylim(y_c - halfwidth, y_c + halfwidth)


def animate_trajectories(
    bodies: list[Body], t_step: float, days_per_frame: float = 1.0
):
    fig = plt.figure(figsize=FIGSIZE, frameon=False)

    ax: plt.Axes = fig.add_axes((0, 0, 1, 1), facecolor="k")

    # trajectories = _concatenate_trajectories(bodies)
    # ax.set_xticks(list(np.linspace(np.min(trajectories[:,0]), np.max(trajectories[:,0]), 15)))
    # ax.set_yticks(list(np.linspace(np.min(trajectories[:,1]), np.max(trajectories[:,1]), 15)))
    # ax.grid(color="w", linestyle="dotted", linewidth=0.5)

    time_text = fig.text(
        0.01, 0.01, "", ha="left", va="bottom", color="w", fontfamily="Ubuntu"
    )

    dbs = [DrawedBody.from_body(b, ax) for b in bodies]

    camera = Camera.from_bodies(bodies, static=False)
    camera.apply(ax)

    def update(frame: int) -> list[Artist]:
        current_day = frame * days_per_frame
        simulation_date = date.fromordinal(int(1 + current_day))
        time_text.set_text(
            f"Y {simulation_date.year : <3} M {simulation_date.month : <3} D {simulation_date.day}"
        )

        current_step = int(current_day / t_step)
        MAJOR_TRAJ_LINE_LEN = 180  # days
        major_line_start_step = max(current_step - int(MAJOR_TRAJ_LINE_LEN / t_step), 0)
        for db in dbs:
            db.traj_line_major.set(
                data=(
                    db.body.x_traj[major_line_start_step:current_step],
                    db.body.y_traj[major_line_start_step:current_step],
                )
            )
            db.traj_line_minor.set(
                data=(
                    db.body.x_traj[: major_line_start_step + 1],
                    db.body.y_traj[: major_line_start_step + 1],
                )
            )
            db.body_dot.set(center=(db.body.x_traj[current_step], db.body.y_traj[current_step]))
        # camera.scale *= 0.995
        # camera.apply(ax)

    frame_count = int((bodies[0].trajectory_length - 1) * t_step / days_per_frame)
    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=range(frame_count),
        blit=False,
    )

    writer = FFMpegWriter(fps=30, metadata=dict(artist="njvsvh"), bitrate=1800)
    anim.save("./simulation.mp4", writer=writer)
