from solar_system.simulation import SOLAR_SYSTEM, simulate_trajectories
from solar_system.visualization import visualize_orbits
from solar_system import utils


bodies = SOLAR_SYSTEM
simulate_trajectories(bodies, *utils.get_dt_nsteps(total_years=10, step_days=1))
visualize_orbits(bodies)
