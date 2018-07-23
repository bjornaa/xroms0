# -*- coding: utf-8 -*-

"""Vertical structure functions for ROMS

:func:`sdepth`
  Depth of s-levels
:func:`zslice`
  Slice a 3D field in s-coordinates to fixed depth
:func:`multi_zslice`
  Slice a 3D field to several depth levels
:func:`z_average`
  Vertical average of a 3D field
:func:`s_stretch`
  Compute vertical stretching arrays Cs_r or Cs_w

"""

# -----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# Bergen, Norway
# 2010-09-30
# -----------------------------------

from __future__ import (absolute_import, division)

import numpy as np


def sdepth(H, Hc, C, stagger="rho", Vtransform=1):
    """Depth of s-levels

    *H* : arraylike
      Bottom depths [meter, positive]

    *Hc* : scalar
       Critical depth

    *cs_r* : 1D array
       s-level stretching curve

    *stagger* : [ 'rho' | 'w' ]

    *Vtransform* : [ 1 | 2 ]
       defines the transform used, defaults 1 = Song-Haidvogel

    Returns an array with ndim = H.ndim + 1 and
    shape = cs_r.shape + H.shape with the depths of the
    mid-points in the s-levels.

    Typical usage::

    >>> fid = Dataset(roms_file)
    >>> H = fid.variables['h'][:, :]
    >>> C = fid.variables['Cs_r'][:]
    >>> Hc = fid.variables['hc'].getValue()
    >>> z_rho = sdepth(H, Hc, C)

    """
    H = np.asarray(H)
    Hshape = H.shape      # Save the shape of H
    H = H.ravel()         # and make H 1D for easy shape manipulation
    C = np.asarray(C)
    N = len(C)
    outshape = (N,) + Hshape       # Shape of output
    if stagger == 'rho':
        S = -1.0 + (0.5+np.arange(N))/N    # Unstretched coordinates
    elif stagger == 'w':
        S = np.linspace(-1.0, 0.0, N)
    else:
        raise ValueError("stagger must be 'rho' or 'w'")

    if Vtransform == 1:         # Default transform by Song and Haidvogel
        A = Hc * (S - C)[:, None]
        B = np.outer(C, H)
        return (A + B).reshape(outshape)

    elif Vtransform == 2:       # New transform by Shchepetkin
        N = Hc*S[:, None] + np.outer(C, H)
        D = (1.0 + Hc/H)
        return (N/D).reshape(outshape)

    else:
        raise ValueError("Unknown Vtransform")

# ------------------------------------


def sdepth_w(H, Hc, cs_w):
    """Return depth of w-points in s-levels

    Kept for backwards compatibility
    use *sdepth(H, Hc, cs_w, stagger='w')* instead

    """
    return sdepth(H, Hc, cs_w, stagger='w')

# ------------------------------------------
# Vertical slicing e.t.c.
# ------------------------------------------


def zslice(F, S, z):
    """Vertical slice of a 3D ROMS field

    Vertical interpolation of a field in s-coordinates to
    (possibly varying) depth level

    *F* : array with vertical profiles, first dimension is vertical

    *S* : array with depths of the F-values,

    *z* : Depth level(s) for output, scalar or ``shape = F.shape[1:]``
          The z values should be negative

    Return value : array, `shape = F.shape[1:]`, the vertical slice

    Example:
    H is an array of depths (positive values)
    Hc is the critical depth
    C is 1D containing the s-coordinate stretching at rho-points
    returns F50, interpolated values at 50 meter with F50.shape = H.shape

    >>> z_rho = sdepth(H, Hc, C)
    >>> F50 = zslice(F, z_rho, -50.0)

    """

    # TODO:
    # Option to Save A, D, Dm
    #   => faster interpolate more fields to same depth

    F = np.asarray(F)
    S = np.asarray(S)
    z = np.asarray(z, dtype='float')
    Fshape = F.shape  # Save original shape
    if S.shape != Fshape:
        raise ValueError("F and z_r must have same shape")
    if z.shape and z.shape != Fshape[1:]:
        raise ValueError("z must be scalar or have shape = F.shape[1:]")

    # Flatten all non-vertical dimensions
    N = F.shape[0]        # Length of vertical dimension
    M = F.size // N        # Combined length of horizontal dimension(s)
    F = F.reshape((N, M))
    S = S.reshape((N, M))
    if z.shape:
        z = z.reshape((M,))

    # Find integer array C with shape (M,)
    # with S[C[i]-1, i] < z <= S[C[i], i]
    # C = np.apply_along_axis(np.searchsorted, 0, S, z)
    # but the following is much faster
    C = np.sum(S < z, axis=0)
    C = C.clip(1, N-1)

    # For vectorisation
    # construct index array tuples D and Dm such that
    #   F[D][i]  = F[C[i], i]
    #   F[Dm][i] = F[C[i]-1, i]
    I = np.arange(M, dtype='int')
    D = (C, I)
    Dm = (C-1, I)

    # Compute interpolation weights
    A = (z - S[Dm]) / (S[D]-S[Dm])
    A = A.clip(0.0, 1.0)   # Control the extrapolation

    # Do the linear interpolation
    R = (1-A)*F[Dm]+A*F[D]

    # Give the result the correct s
    R = R.reshape(Fshape[1:])

    return R

# -----------------------------------------------


def s_stretch(N, theta_s, theta_b, stagger='rho', Vstretching=1):
    """Compute a s-level stretching array

    *N* : Number of vertical levels

    *theta_s* : Surface stretching factor

    *theta_b* : Bottom stretching factor

    *stagger* : "rho"|"w"

    *Vstretching* : 1|2|4

    """

    if stagger == 'rho':
        S = -1.0 + (0.5+np.arange(N))/N
    elif stagger == "w":
        S = np.linspace(-1.0, 0.0, N+1)
    else:
        raise ValueError("stagger must be 'rho' or 'w'")

    if Vstretching == 1:
        cff1 = 1.0 / np.sinh(theta_s)
        cff2 = 0.5 / np.tanh(0.5*theta_s)
        return ((1.0-theta_b)*cff1*np.sinh(theta_s*S)
                + theta_b*(cff2*np.tanh(theta_s*(S+0.5))-0.5))

    elif Vstretching == 2:
        a, b = 1.0, 1.0
        Csur = (1 - np.cosh(theta_s * S))/(np.cosh(theta_s) - 1)
        Cbot = np.sinh(theta_b * (S+1)) / np.sinh(theta_b) - 1
        mu = (S+1)**a * (1 + (a/b)*(1-(S+1)**b))
        return mu*Csur + (1-mu)*Cbot

    elif Vstretching == 4:
        C = (1 - np.cosh(theta_s * S)) / (np.cosh(theta_s) - 1)
        C = (np.exp(theta_b * C) - 1) / (1 - np.exp(-theta_b))
        return C

    else:
        raise ValueError("Unknown Vstretching")
