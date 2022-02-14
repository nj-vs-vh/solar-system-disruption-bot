from solar_system import simulation, visualization, interestness


def main():
    t_step = 1
    t_yrs_pre = 20
    t_yrs_post = 100

    days_per_frame = 31

    bodies_calm = simulation.SOLAR_SYSTEM
    simulation.simulate_trajectories(bodies_calm, t_step, t_yrs_pre)

    bodies_disurpted, bodies_control, description = simulation.generate_disruption(bodies_calm, t_yrs_post)
    simulation.simulate_trajectories(bodies_disurpted, t_step, t_yrs_post)
    simulation.simulate_trajectories(bodies_control, t_step, t_yrs_post)
    # interestness.rate_bodies(bodies_disurpted, bodies_control, t_step)

    visualization.plot_orbits(bodies_disurpted)
    visualization.animate_trajectories(bodies_calm, bodies_disurpted, t_step, days_per_frame=days_per_frame)


if __name__ == "__main__":
    main()
