import numpy as np
import logging
from math import exp, sqrt, factorial, ceil
from scipy.integrate import simps
from scipy.optimize import newton

logger = logging.getLogger(__name__)

NGAMMA = 5


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


def make_Gamma_from_cs(a, Gamma_infty, kappa, n):
    cs_from_Gamma = make_cs_from_Gamma(a, Gamma_infty, kappa, n)
    def Gamma_from_cs(cs):
        return newton(lambda x: cs_from_Gamma(x) - cs, Gamma_infty/2.)
    return Gamma_from_cs


def pick_outer_grid(Gamma_from_cs, c0, alpha, R):
    h_predict = Gamma_from_cs(c0)/c0
    min_dr = (3. * R**2. * alpha * h_predict)**(1./3.)
    L = (3. * R**2. * 100. * h_predict)**(1./3.)
    r = R + np.array([0] + (np.arange(sqrt(min_dr), sqrt(L)+sqrt(min_dr), sqrt(min_dr))**2.).tolist())
    return r


def pick_quartic_grid(xmin, xmax, dxmin):
    # Given points distributed between $\{ 0, \ldots , 1\}$ nonuniformly by
    # mapping points distributed uniformly between 0 and 1 with the function
    # $(x_j - 2)^2 (x_j)^2$. Given we want a minimum grid size on left or
    # right, how many grid points do we need?
    # Let $j \in \{ 0, \ldots, m-1 \}$
    # Given $y_j = \left( \frac{j}{m-1} - 2 \right)^2 \left( \frac{j}{m-1} \right)^2 R$
    # m = ceil(max(np.roots([y, -4.*y, 6.*y-4., -4.*y+12., -8.]))) # second-smallest interval greater than y (left)
    # m = ceil(max(np.roots([y, -4.*y, 6.*y-2., -4.*y+4., y-1.]))) # smallest interval greater than y(right)

    y = dxmin/(xmax-xmin)
    m = int(ceil(max(np.roots([y, -4.*y, 6.*y-4., -4.*y+12., -8.])))) # second-smallest interval greater than y (left)
    # densify the grid so that NGAMMA grid points are found within dxmin
    x = np.linspace(0., 1., m)
    return (x**2.*(x-2.)**2.)*(xmax-xmin)+xmin


def pick_inner_grid(Gamma_from_cs, c0, alpha, R):
    h_predict = Gamma_from_cs(c0)/c0
    min_dr = (3. * R**2. * alpha * h_predict)**(1./3.)
    r = pick_quartic_grid(0., R, min_dr)
    return r


def pick_inner_and_outer_grids(Gamma_from_csA, Gamma_from_csB, c0A, c0B, alpha, R):
    if c0A > 0.:
        rA = pick_inner_grid(Gamma_from_csA, c0A, alpha, R)
    else:
        rA = np.nan
    if c0B > 0.:
        rB = pick_outer_grid(Gamma_from_csB, c0B, alpha, R)
    else:
        rB = np.nan

    if rA == np.nan and rB == np.nan:
        raise ValueError('Both rA and rB are nan')

    if rA is np.nan:
        min_dr = min(rB[1:]-rB[:-1])
    elif rB is np.nan:
        min_dr = min(rA[1:]-rA[:-1])
    else:
        min_dr = min(min(rA[1:]-rA[:-1]), min(rB[1:]-rB[:-1]))

    rA = pick_quartic_grid(0., R, min_dr)
    if rB is np.nan:
        h_predict = Gamma_from_csB(c0A)/c0A
        L = (3. * R**2. * 100. * h_predict)**(1./3.)
        rB = R + np.array([0] + (np.arange(sqrt(min_dr), sqrt(L)+sqrt(min_dr), sqrt(min_dr))**2.).tolist())

    ## NOTE IN PROGRESS
    ##if c0B > 0.:
    ##    h_predict = Gamma_from_csB(c0B)/c0B
    ##else:
    ##    h_predict = Gamma_from_csB(c0A)/c0A
    ##L = (3. * R**2. * 100. * h_predict)**(1./3.)
    ##rB = R + np.array([0] + (np.arange(sqrt(min_dr), sqrt(L)+sqrt(min_dr), sqrt(min_dr))**2.).tolist())

    # ensure minimum number of grid points
    #m_min = 6
    #x = np.linspace(0., 1., m_min)
    #if len(rA) < m_min:
    #    rA = R - (x**2.*(x-2.)**2.)*R
    #    rA = rA[::-1]
    #if len(rB) < m_min:
    #    rB = R + (x**2.*(x-2.)**2.)*L

    return rA, rB


def compute_one_phase_equilibrium(D, c0, R, a, Gamma_infty, kappa, n, times_to_save, noflux_bc=False, alpha=1e-5):
    Gamma = 0.

    cs_from_Gamma = make_cs_from_Gamma(a, Gamma_infty, kappa, n)
    Gamma_from_cs = make_Gamma_from_cs(a, Gamma_infty, kappa, n)

    r = pick_outer_grid(Gamma_from_cs, c0, alpha, R)
    m = len(r)

    c = np.array([c0]*m)

    A = compute_2nd_order_finite_difference_coefficient_matrix(r)
    A *= D

    nGamma = min(NGAMMA, m)
    cGamma = D*finite_difference_weights(r[:nGamma], r[0], 1)

    # TRANSFORM c -> cr
    # This converts the spherical radial diffusion equation into a linear
    # diffusion equation
    cr = c*r

    max_dt = 0.4*np.min(r[1:]-r[:-1])**2./D
    t = 0.

    times_to_save = list(times_to_save)
    cr_save = []
    Gamma_save = []

    while len(times_to_save) > 0:
        dt = max_dt
        if t + dt >= times_to_save[0]:
            dt = times_to_save[0] - t

        cr[0] = cs_from_Gamma(Gamma)*r[0]

        # EXPLICIT EULER TIME INTEGRATION
        # Update Gamma based on finite difference approximation of
        # concentration gradient at surface with nGamma-point stencil
        Gamma += dt*np.sum(cGamma*cr[:nGamma]/r[:nGamma])

        # Update the radius-weighted concentrations with the planar
        # Laplacian approximated with centered finite differences
        cr += dt*A.dot(cr)

        # outer no-flux boundary condition, if enabled
        if noflux_bc:
            cr[-1] = cr[-2]*r[-1]/r[-2]

        if t == times_to_save[0]:
            times_to_save.pop(0)
            cr_save.append(list(cr))
            Gamma_save.append(Gamma)
        t += dt

    # TRANSFORM cr -> c
    c_save = []
    for cr in cr_save:
        c_save.append(list(cr/r))

    return r, np.array(c_save), np.array(Gamma_save)


def compute_two_phase_equilibrium(DA, DB, c0A, c0B, R, aA, aB, Gamma_infty,
                                  kappa, n, times_to_save,
                                  noflux_bc=False, alpha=1e-5):
    Gamma = 0.

    csA_from_Gamma = make_cs_from_Gamma(aA, Gamma_infty, kappa, n)
    Gamma_from_csA = make_Gamma_from_cs(aA, Gamma_infty, kappa, n)
    csB_from_Gamma = make_cs_from_Gamma(aB, Gamma_infty, kappa, n)
    Gamma_from_csB = make_Gamma_from_cs(aB, Gamma_infty, kappa, n)

    # TODO I cannot separate `pick_inner_grid` and `pick_outer_grid`
    # entirely as I have tried to do here, because I need to use a
    # shared `dr_min` for the two. I can keep the separate functions
    # as I have written them, but after calling both I will need to
    # replace the coarser grid with a finer grid, I think.
    # It is possible that they should be different for maximum
    # stability or decent accuracy, but for now I will just make them
    # the same.

    rA, rB = pick_inner_and_outer_grids(Gamma_from_csA, Gamma_from_csB, c0A, c0B, alpha, R)
    mA = len(rA)
    mB = len(rB)

    cA = np.array([c0A]*mA)
    cB = np.array([c0B]*mB)

    A = compute_2nd_order_finite_difference_coefficient_matrix(rA)
    A *= DA
    B = compute_2nd_order_finite_difference_coefficient_matrix(rB)
    B *= DB

    nGammaA = min(NGAMMA, mA)
    nGammaB = min(NGAMMA, mB)
    logger.info('rA = {}'.format(rA))
    logger.info('rB = {}'.format(rB))
    cGammaA = -DA*finite_difference_weights(rA[-nGammaA:], rA[-1], 1)
    cGammaB = DB*finite_difference_weights(rB[:nGammaB], rB[0], 1)

    # TRANSFORM c -> cr
    # This converts the spherical radial diffusion equation into a linear
    # diffusion equation
    crA = cA*rA
    crB = cB*rB

    # NOTE will I have an instability if rA[1]-rA[0] = rA[1]-0 = rA[1] is too small?
    max_dt = 0.4*min(min(rA[1:]-rA[:-1]), min(rB[2:]-rB[1:-1]))**2./max(DA, DB)
    t = 0.

    times_to_save = list(times_to_save)
    crA_save = []
    crB_save = []
    Gamma_save = []


    while len(times_to_save) > 0:
        dt = max_dt
        if t + dt >= times_to_save[0]:
            dt = times_to_save[0] - t

        crA[-1] = csA_from_Gamma(Gamma)*rA[-1]
        crB[0] = csB_from_Gamma(Gamma)*rB[0]

        # EXPLICIT EULER TIME INTEGRATION
        # Update Gamma based on finite difference approximation of
        # concentration gradient at surface with nGamma-point stencil
        Gamma += dt*np.sum(cGammaA*crA[-nGammaA:]/rA[-nGammaA:])
        Gamma += dt*np.sum(cGammaB*crB[:nGammaB]/rB[:nGammaB])

        # Update the radius-weighted concentrations with the planar
        # Laplacian approximated with centered finite differences
        crA += dt*A.dot(crA)
        crB += dt*B.dot(crB)

        # inner no-flux boundary condition
        ##crA[0] = crA[1]*rA[0]/rA[1]
        crA[0] = 0.
        # outer no-flux boundary condition, if enabled
        if noflux_bc:
            crB[-1] = crB[-2]*rB[-1]/rB[-2]

        if t == times_to_save[0]:
            times_to_save.pop(0)
            crA_save.append(list(crA))
            crB_save.append(list(crB))
            Gamma_save.append(Gamma)
        t += dt

    # TRANSFORM cr -> c
    cA_save = []
    for cr in crA_save:
        cA = [0.]+list(cr[1:]/rA[1:])
        # inner no-flux boundary condition
        cA[0] = cA[1]
        cA_save.append(cA)

    cB_save = []
    for cr in crB_save:
        cB_save.append(list(cr/rB))

    return rA, rB, np.array(cA_save), np.array(cB_save), np.array(Gamma_save)


def compute_mass_balance_one_phase(Gamma, c, r, R):
    if c.ndim == 1:
        axis=0
    elif c.ndim == 2:
        axis=1
    else:
        raise ValueError('Unexpected c.ndim == {}'.format(c.ndim))

    total_mass = Gamma + simps(c*r**2./R**2., r, axis=axis)
    return total_mass


def compute_mass_balance_two_phase(Gamma, cA, rA, cB, rB, R):
    if cA.ndim == 1:
        axis=0
    elif cA.ndim == 2:
        axis=1
    else:
        raise ValueError('Unexpected cA.ndim == {}'.format(cA.ndim))

    if cB.ndim == 1:
        axis=0
    elif cB.ndim == 2:
        axis=1
    else:
        raise ValueError('Unexpected cB.ndim == {}'.format(cB.ndim))

    total_mass = Gamma + simps(cA*rA**2./R**2., rA, axis=axis) + simps(cB*rB**2./R**2., rB, axis=axis)
    return total_mass
