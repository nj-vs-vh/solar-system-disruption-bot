from pathlib import Path
import traceback

from solar_system import simulation, stolen_sounds, visualization, config, telegram_bot


def main():
    conf = config.load_config()
    t_step = conf['step_days']
    days_per_frame = conf['days_per_frame']
    t_yrs_pre = conf['calm_years']
    t_yrs_post = conf['post_disruption_years']

    output_file = Path("simulation.mp4")

    n_try = 5
    for i_try in range(n_try):
        try:
            bodies_calm = simulation.SOLAR_SYSTEM
            simulation.simulate_trajectories(bodies_calm, t_step, t_yrs_pre)

            bodies_disurpted, _, description = simulation.generate_disruption(bodies_calm, t_yrs_post)
            simulation.simulate_trajectories(bodies_disurpted, t_step, t_yrs_post)
            visualization.plot_orbits(bodies_disurpted)

            video_quiet = Path("simulation-quiet.mp4")
            visualization.animate_trajectories(
                bodies_calm, bodies_disurpted, t_step, days_per_frame=days_per_frame, output_file=video_quiet
            )
            break
        except Exception as e:
            print(f"\n\nERROR DURING SIMULATION / VIDEO EXPORT ({i_try}/{n_try} try): {e}")
            traceback.print_exc()
            if i_try == n_try - 1:
                print("\n\nNO MORE TRIES, EXITING")
                return

    try:
        print("Retrieving audio")
        audio = Path("audio.mp4")
        approx_video_length = (t_yrs_pre + t_yrs_post) * 365 / days_per_frame / 30
        print(f"Video length ~{approx_video_length} sec")
        stolen_sounds.download(audio, duration=approx_video_length * 1.3)
        stolen_sounds.add_audio(video_quiet, audio, output_file)
    except Exception as e:
        print(f"\n\nERROR WHILE STEALING AUDIO: {e}")
        traceback.print_exc()
        output_file = video_quiet

    msg = "SOLAR SYSTEM DISRUPTION:\n\n" + description
    telegram_bot.send_to_channel(msg, output_file)


if __name__ == "__main__":
    main()
