import numpy as np
from control import StateSpace, ControlSlycot

__all__ = ('hinfsyn',)


def hinfsyn(P, nmeas, ncon, gamma_tol):
    """
    Function calling sb10ad routine from slycot
    """
    try:
        from slycot import sb10ad
    except ImportError:
        raise ControlSlycot("can't find slycot subroutine sb10ad")

    n = np.size(P.A, 0)
    m = np.size(P.B, 1)
    np_ = np.size(P.C, 0)
    gamma = 1.e100
    out = sb10ad(n, m, np_, ncon, nmeas, gamma, P.A, P.B, P.C, P.D, gtol=gamma_tol)
    gam = out[0]
    Ak = out[1]
    Bk = out[2]
    Ck = out[3]
    Dk = out[4]
    Ac = out[5]
    Bc = out[6]
    Cc = out[7]
    Dc = out[8]
    rcond = out[9]

    K = StateSpace(Ak, Bk, Ck, Dk)
    CL = StateSpace(Ac, Bc, Cc, Dc)

    return K, CL, gam, rcond
