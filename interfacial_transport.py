import numpy as np
import logging
from math import exp, factorial
from scipy.integrate import simps

logger = logging.getLogger(__name__)


def finite_difference_weights(x, x_bar, k):
    """Calculate the finite difference weights for approximating
    the kth derivative at x_bar based on points x. Based on the approach
    described by Randall J. LeVeque in Chapter 1 of Finite Difference
    Methods for Ordinary and Partial Differential Equations

    Parameters
    ----------
    x : ndarray
        Vector of positions at which data will be used to approximate

    x_bar : float
        Position at which the derivative should be approximated

    k : int
        Order of the derivative

    Returns
    -------
    c : ndarray
        Vector of weights to appriximate the kth derivative at x_bar from
        data at positions x

    Example
    -------
    # Approximate the 1st derivative of exp(x) w.rt. x at x = 0
    x = np.linspace(0., 1., 6)
    c = finite_difference_weights(x, x[0], 1)
    result = np.sum(c*np.exp(x))
    """
    n = len(x)
    L = np.zeros((n, n))
    for i in range(1, n+1):
        for j in range(1, n+1):
            L[i-1, j-1] = 1./factorial(i-1.)*(x[j-1]-x_bar)**(i-1.)

    r = np.zeros(n)
    r[k] = 1.
    c = np.linalg.solve(L, r)

    return c


def compute_2nd_order_finite_difference_coefficients(x):
    """Compute 2nd-order finite difference coefficients for a centered,
    three-node stencil. 

    Parameters
    ----------
    x : ndarray
        1-D vector of node positions for which to compute the finite
        difference approximations.

    Returns
    -------
    am, a0, ap : ndarray
        Three 1-D vectors containing the coefficients to compute the
        finite difference approximate 2nd order derivative at the N-2
        inner nodes of x.

    Notes
    -----
    len(am) == len(a0) == len(ap) == len(x) - 2

    Example
    -------
    # Get the finite difference approximate second derivative of sin(x)
    x = np.linspace(0., 1.)
    am, a0, ap = compute_2nd_order_finite_difference_coefficients(x)
    y = np.sin(x)
    result = am * y[:-2] + a0 * y[1:-1] + ap * y[2:]
    
    """
    h = x[1:] - x[:-1]
    hm = h[:-1]
    hp = h[1:]
    am = 2./(hm*hp + hm*hm)
    a0 = -2./(hm*hp)
    ap = 2./(hm*hp + hp*hp)

    return am, a0, ap


def compute_2nd_order_finite_difference_coefficient_matrix(x):
    """Compute 2nd-order finite difference coefficient matrix for a centered,
    three-node stencil. 

    Parameters
    ----------
    x : ndarray
        1-D vector of node positions for which to compute the finite
        difference approximations.

    Returns
    -------
    A : ndarray
        Three 1-D vectors containing the coefficients to compute the
        finite difference approximate 2nd order derivative at the N-2
        inner nodes of x.

    Example
    -------
    x = np.linspace(0., 1.)
    A = compute_2nd_order_finite_difference_coefficient_matrix(x)
    y = np.sin(x)
    result = A.dot(y)
    
    """
    m = len(x)
    A = np.zeros((m, m))
    am, a0, ap = compute_2nd_order_finite_difference_coefficients(x)
    for i in range(m-2):
        A[i+1, i+1] = a0[i]
        A[i+1, i] = am[i]
        A[i+1, i+2] = ap[i]

    return A


def make_cs_from_Gamma(a, Gamma_infty, kappa, n):
    def cs_from_Gamma(Gamma):
        return a*Gamma*exp(kappa*(Gamma/Gamma_infty)**n)/(Gamma_infty-Gamma)
    return cs_from_Gamma


def compute(m, D, c0, R, L, a, Gamma_infty, kappa, n, times_to_save, noflux_bc=False):
    Gamma = 0.
    c = np.array([c0]*m)
    r = np.linspace(R, R+L, m)
    A = compute_2nd_order_finite_difference_coefficient_matrix(r)
    A *= D
    cs_from_Gamma = make_cs_from_Gamma(a, Gamma_infty, kappa, n)

    nGamma = 5
    cGamma = D*finite_difference_weights(r[:nGamma], r[0], 1)

    # TRANSFORM c -> cr
    cr = c*r

    max_dt = 0.4*np.min(r[1:]-r[:-1])**2./D
    t = 0.

    logging.debug('max_dt = {}; nGamma = {}'.format(max_dt, nGamma))

    times_to_save = list(times_to_save)
    cr_save = []
    Gamma_save = []

    # This converts the spherical radial diffusion equation into a linear
    # diffusion equation
    # EQUILIBRATE INTERFACE
    while len(times_to_save) > 0:
        dt = max_dt
        if t + dt >= times_to_save[0]:
            dt = times_to_save[0] - t

        if noflux_bc:
            cr[0] = cs_from_Gamma(Gamma)*r[0]

        # EXPLICIT EULER TIME INTEGRATION

        # Update Gamma based on finite difference approximation of
        # concentration gradient at surface with nGamma-point stencil
        Gamma += dt*np.sum(cGamma*cr[:nGamma]/r[:nGamma])

        # Update the radius-weighted concentrations with the planar
        # Laplacian approximated with centered finite differences
        cr += dt*A.dot(cr)
        cr[-1] = cr[-2]*r[-1]/r[-2]

        logging.debug('time = {}; Gamma = {:1.5f}'.format(t, Gamma))
        if t == times_to_save[0]:
            times_to_save.pop(0)
            logging.debug('saving at time = {}:'.format(t))
            logging.debug('saving cr: {}'.format(cr))
            logging.debug('saving Gamma: {}'.format(Gamma))
            cr_save.append(list(cr))
            Gamma_save.append(Gamma)
        t += dt

    # TRANSFORM cr -> c
    c_save = []
    for cr in cr_save:
        c_save.append(list(cr/r))

    return np.array(c_save), np.array(Gamma_save)


def compute_mass_balance(Gamma, c, r, R):
    if c.ndim == 1:
        axis=0
    elif c.ndim == 2:
        axis=1
    else:
        raise ValueError('Unexpected c.ndim == {}'.format(c.ndim))
    total_mass = Gamma + simps(c*r**2./R**2., r, axis=axis)
    return total_mass
