from pathlib import Path

from solar_system import simulation, visualization, interestness, steal_sounds


def main():
    t_step = 1
    t_yrs_pre = 30
    t_yrs_post = 100
    days_per_frame = 31
    output_file = Path("simulation.mp4")

    bodies_calm = simulation.SOLAR_SYSTEM
    simulation.simulate_trajectories(bodies_calm, t_step, t_yrs_pre)

    bodies_disurpted, bodies_control, description = simulation.generate_disruption(bodies_calm, t_yrs_post)
    simulation.simulate_trajectories(bodies_disurpted, t_step, t_yrs_post)
    # simulation.simulate_trajectories(bodies_control, t_step, t_yrs_post)
    # interestness.rate_bodies(bodies_disurpted, bodies_control, t_step)

    visualization.plot_orbits(bodies_disurpted)

    video_quiet = Path("simulation-quiet.mp4")
    visualization.animate_trajectories(
        bodies_calm, bodies_disurpted, t_step, days_per_frame=days_per_frame, output_file=video_quiet
    )

    audio = Path("audio.mp4")
    approx_video_length = (t_yrs_pre + t_yrs_post) * 365 / 30
    steal_sounds.download(audio, duration=approx_video_length * 2)
    steal_sounds.add_audio(video_quiet, audio, output_file)


if __name__ == "__main__":
    main()
