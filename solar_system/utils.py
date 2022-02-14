def get_dt_nsteps(total_years: float, step_days: float = 1):
    dt = step_days / 365
    return dt, int(total_years / dt)
