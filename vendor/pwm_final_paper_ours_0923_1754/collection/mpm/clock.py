import numpy as np
from .contracts import Clock
SPEED_BUDGET = 8.

def wave_speeds(E, nu, density):
    E, nu, density = np.broadcast_arrays(np.asarray(E, float), np.asarray(nu, float), np.asarray(density, float))
    if (not all(np.isfinite(a).all() for a in (E, nu, density)) or np.any(E < 0)
            or np.any(nu <= -1) or np.any(nu >= .5) or np.any(density <= 0)):
        raise ValueError('Invalid material wave parameters')
    mu = E/(2*(1+nu)); lam = E*nu/((1+nu)*(1-2*nu))
    return np.sqrt((lam+2*mu)/density)


def choose_clock(static, *, dx_m=1/64):
    if not np.isfinite(dx_m) or dx_m<=0:raise ValueError('Positive finite physical grid spacing required')
    cp = float(wave_speeds(static['E'], static['nu'], static['density']).max())
    for divisor in (2, 4, 8, 16):
        clock = Clock(1e-4/divisor, 417*divisor, 48)
        courant = clock.substep_dt_s*(cp+SPEED_BUDGET)/dx_m
        if courant <= .4:
            np.testing.assert_array_equal(clock.state_time_s, Clock(1e-4, 417, 48).state_time_s)
            return clock, dict(linear_wave_speed_m_s=cp, advective_budget_m_s=SPEED_BUDGET,
                initial_budgeted_courant=courant, accuracy_certificate=False)
    raise ValueError('Outside predeclared fixed-clock budget')
