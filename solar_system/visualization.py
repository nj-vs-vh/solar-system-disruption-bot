from matplotlib import pyplot as plt

from .simulation import Body


FIGSIZE = (10, 10)


def visualize_orbits(bodies: list[Body]):
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
            s=b.size_normalized,
            c=b.color,
            marker="o",
        )
    ax.axis('equal')
    ax.legend()
    plt.savefig("orbits.png", bbox_inches="tight")
