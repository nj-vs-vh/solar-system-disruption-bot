from solar_system.simulation import SOLAR_SYSTEM, simulate_trajectories
from solar_system.visualization import visualize_orbits


bodies = SOLAR_SYSTEM
simulate_trajectories(bodies, dt=1/365, n_steps=int(100 * 365))
visualize_orbits(bodies)
