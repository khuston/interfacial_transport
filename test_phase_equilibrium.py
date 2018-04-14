import numpy as np
import logging
from interfacial_transport import compute_one_phase_equilibrium, compute_mass_balance_one_phase
from interfacial_transport import compute_two_phase_equilibrium, compute_mass_balance_two_phase
from numpy.testing import assert_almost_equal

logger = logging.getLogger(__name__)

# Check one-phase equilibrium for errors
def test_no_errors_one_phase_equilibrium_interface():
    alpha = 6.86e-6       # 1/s
    beta = 22.1           # m**3/mol/s
    n = 0.460             #
    kappa = 10.3          #
    D = 3.8e-10           # m**2/s
    Gamma_infty = 2.25e-6 # mol/m**2

    R = 1.9e-3

    a = alpha/beta
    c0 = 0.0006

    times_to_save = np.logspace(-3., 4.)
    r, c_save, Gamma_save = compute_one_phase_equilibrium(D, c0, R, a, Gamma_infty, kappa, n, times_to_save)

    assert(c_save.ndim == 2)
    assert(Gamma_save.ndim == 1)
    assert(r.ndim == 1)
    assert(len(Gamma_save) == len(times_to_save))
    assert(len(c_save) == len(times_to_save))
    assert(len(r) == c_save.shape[1])


def test_no_errors_two_phase_equilibrium_interface():
    alpha = 6.86e-6       # 1/s
    beta = 22.1           # m**3/mol/s
    n = 0.460             #
    kappa = 10.3          #
    DA = 3.8e-10/48.      # m**2/s
    DB = 3.8e-10          # m**2/s
    Gamma_infty = 2.25e-6 # mol/m**2

    R = 1.9e-3

    aB = alpha/beta
    aA = alpha/beta*1.45
    c0A = 0.
    c0B = 0.0006

    times_to_save = np.logspace(-3., 4.)
    rA, rB, cA_save, cB_save, Gamma_save = compute_two_phase_equilibrium(DA, DB, c0A, c0B, R, aA, aB, Gamma_infty, kappa, n, times_to_save)

    assert(cA_save.ndim == 2)
    assert(cB_save.ndim == 2)
    assert(Gamma_save.ndim == 1)
    assert(rA.ndim == 1)
    assert(rB.ndim == 1)
    assert(len(Gamma_save) == len(times_to_save))
    assert(len(cA_save) == len(times_to_save))
    assert(len(cB_save) == len(times_to_save))
    assert(len(rA) == cA_save.shape[1])
    assert(len(rB) == cB_save.shape[1])



def test_no_errors_compute_mass_balance_one_phase():
    Gamma = 1.
    c = np.ones(5)
    r = np.linspace(1., 2., 5)
    compute_mass_balance_one_phase(Gamma, c, r, r[0])

    Gamma = np.ones(10)
    c = np.ones((10, 20))
    r = np.linspace(1., 2., 20)
    compute_mass_balance_one_phase(Gamma, c, r, r[0])


def test_no_errors_compute_mass_balance_two_phase():
    Gamma = 1.
    cA = np.ones(5)
    rA = np.linspace(0., 1., 5)
    cB = np.ones(5)
    rB = np.linspace(1., 2., 5)
    compute_mass_balance_two_phase(Gamma, cA, rA, cA, rB, rB[0])

    Gamma = np.ones(10)
    cA = np.ones((10, 20))
    rA = np.linspace(0., 1., 20)
    cB = np.ones((10, 20))
    rB = np.linspace(1., 2., 20)
    compute_mass_balance_two_phase(Gamma, cA, rA, cB, rB, rB[0])


def test_one_phase_mass_balance_accuracy():
    alpha = 6.86e-6       # 1/s
    beta = 22.1           # m**3/mol/s
    n = 0.460             #
    kappa = 10.3          #
    D = 3.8e-10           # m**2/s
    Gamma_infty = 2.25e-6 # mol/m**2

    R = 1.9e-3

    a = alpha/beta
    c0 = 0.0006

    times_to_save = np.logspace(-3., 4.)
    r, c_save, Gamma_save = compute_one_phase_equilibrium(D, c0, R, a, Gamma_infty, kappa, n, times_to_save)
    compute_mass_balance_one_phase(Gamma_save, c_save, r, R)
    error = (compute_mass_balance_one_phase(Gamma_save, c_save, r, R) - compute_mass_balance_one_phase(0., np.array([0.]+[c0]*(len(r)-1)), r, R))
    assert_almost_equal(error[-1]/Gamma_save[-1], 0., decimal=2)


def test_two_phase_mass_balance_accuracy():
    alpha = 6.86e-6       # 1/s
    beta = 22.1           # m**3/mol/s
    n = 0.460             #
    kappa = 10.3          #
    DA = 3.8e-10/48.           # m**2/s
    DB = 3.8e-10           # m**2/s
    Gamma_infty = 2.25e-6 # mol/m**2

    R = 1.9e-3

    aB = alpha/beta
    aA = alpha/beta*1.45
    c0A = 0.
    c0B = 0.0006

    times_to_save = np.logspace(-3., 4.)
    rA, rB, cA_save, cB_save, Gamma_save = compute_two_phase_equilibrium(DA, DB, c0A, c0B, R, aA, aB, Gamma_infty, kappa, n, times_to_save)
    compute_mass_balance_two_phase(Gamma_save, cA_save, rA, cB_save, rB, R)
    logger.info('rA={}'.format(rA))
    logger.info('rB={}'.format(rB))
    logger.info('cA={}'.format(cA_save))
    logger.info('cB={}'.format(cB_save))
    error = (compute_mass_balance_two_phase(Gamma_save, cA_save, rA, cB_save, rB, R) - compute_mass_balance_two_phase(0., np.array([c0A]*(len(rA)-1)+[0.]), rA, np.array([0.]+[c0B]*(len(rB)-1)), rB, R))
    assert_almost_equal(error[-1]/Gamma_save[-1], 0., decimal=2)


def test_validate_one_phase_equilibrium_interface():
    pass
