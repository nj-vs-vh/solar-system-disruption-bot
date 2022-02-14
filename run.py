from solar_system import simulation, visualization, interestness


def main():
    t_step = 7
    t_yrs_pre = 0.1
    t_yrs_post = 5

    bodies = simulation.SOLAR_SYSTEM[:3]
    simulation.simulate_trajectories(bodies, t_step, t_yrs_pre)

    bodies_disurpted, bodies_control, description = simulation.generate_disruption(bodies)
    print(description)
    simulation.simulate_trajectories(bodies_disurpted, t_step, t_yrs_post)
    simulation.simulate_trajectories(bodies_control, t_step, t_yrs_post)
    interestness.rate_bodies(bodies_disurpted, bodies_control, t_step)

    visualization.plot_orbits(bodies_disurpted)
    visualization.animate_trajectories(bodies_disurpted, t_step, days_per_frame=14)


if __name__ == "__main__":
    main()
