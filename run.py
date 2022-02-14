from solar_system import simulation, visualization


def main():
    bodies = simulation.SOLAR_SYSTEM
    t_step = 1
    simulation.simulate_trajectories(bodies, t_step=t_step, t_total_yrs=30)
    visualization.animate_trajectories(bodies, t_step=t_step, days_per_frame=14)


if __name__ == "__main__":
    main()
