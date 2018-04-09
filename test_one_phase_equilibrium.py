import numpy as np
from interfacial_transport import compute, compute_mass_balance

# Check one-phase equilibrium for errors
def test_no_errors_one_phase_equilibrium_interface():
    alpha = 6.86e-6#*50.       # 1/s
    beta = 22.1#*50.           # m**3/mol/s
    n = 0.460             #
    kappa = 10.3          #
    D = 3.8e-10           # m**2/s
    Gamma_infty = 2.25e-6 # mol/m**2

    R = 1.9e-3
    L = 1e-2            # m
    m = 101

    a = alpha/beta
    c0 = 0.0006

    times_to_save = np.logspace(-3., 4.)
    c_save, Gamma_save = compute(m, D, c0, R, L, a, Gamma_infty, kappa, n, times_to_save)

    assert(c_save.ndim == 2)
    assert(Gamma_save.ndim == 1)
    assert(len(Gamma_save) == len(times_to_save))
    assert(len(c_save) == len(times_to_save))


def test_no_errors_compute_mass_balance():
    Gamma = 1.
    c = np.ones(5)
    r = np.linspace(1., 2., 5)
    compute_mass_balance(Gamma, c, r, r[0])

    Gamma = np.ones(10)
    c = np.ones((10, 20))
    r = np.linspace(1., 2., 20)
    compute_mass_balance(Gamma, c, r, r[0])



def test_validate_one_phase_equilibrium_interface():
    pass
