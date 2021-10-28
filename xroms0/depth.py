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

from typing import Union, List
import numpy as np
import xarray as xr

Surface = Union[float, np.ndarray]


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
    Hshape = H.shape  # Save the shape of H
    H = H.ravel()  # and make H 1D for easy shape maniplation
    C = np.asarray(C)
    N = len(C)
    outshape = (N,) + Hshape  # Shape of output
    if stagger == "rho":
        S = -1.0 + (0.5 + np.arange(N)) / N  # Unstretched coordinates
    elif stagger == "w":
        S = np.linspace(-1.0, 0.0, N)
    else:
        raise ValueError("stagger must be 'rho' or 'w'")

    if Vtransform == 1:  # Default transform by Song and Haidvogel
        A = Hc * (S - C)[:, None]
        B = np.outer(C, H)
        return (A + B).reshape(outshape)

    elif Vtransform == 2:  # New transform by Shchepetkin
        N = Hc * S[:, None] + np.outer(C, H)
        D = 1.0 + Hc / H
        return (N / D).reshape(outshape)

    else:
        raise ValueError("Unknown Vtransform")


# ------------------------------------


def sdepth_w(H, Hc, cs_w):
    """Return depth of w-points in s-levels

    Kept for backwards compatibility
    use *sdepth(H, Hc, cs_w, stagger='w')* instead

    """
    return sdepth(H, Hc, cs_w, stagger="w")


# ------------------------------------------
# Vertical slicing e.t.c.
# ------------------------------------------


def zslice2(F, S, z):
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
    z = np.asarray(z, dtype="float")
    Fshape = F.shape  # Save original shape
    if S.shape != Fshape:
        raise ValueError("F and z_r must have same shape")
    if z.shape and z.shape != Fshape[1:]:
        raise ValueError("z must be scalar or have shape = F.shape[1:]")

    # Flatten all non-vertical dimensions
    N = F.shape[0]  # Length of vertical dimension
    M = F.size // N  # Combined length of horizontal dimension(s)
    F = F.reshape((N, M))
    S = S.reshape((N, M))
    if z.shape:
        z = z.reshape((M,))

    # Find integer array C with shape (M,)
    # with S[C[i]-1, i] < z <= S[C[i], i]
    # C = np.apply_along_axis(np.searchsorted, 0, S, z)
    # but the following is much faster
    C = np.sum(S < z, axis=0)
    C = C.clip(1, N - 1)

    # For vectorization
    # construct index array tuples D and Dm such that
    #   F[D][i]  = F[C[i], i]
    #   F[Dm][i] = F[C[i]-1, i]
    I = np.arange(M, dtype="int")
    D = (C, I)
    Dm = (C - 1, I)

    # Compute interpolation weights
    A = (z - S[Dm]) / (S[D] - S[Dm])
    A = A.clip(0.0, 1.0)  # Control the extrapolation

    # Do the linear interpolation
    R = (1 - A) * F[Dm] + A * F[D]

    # Give the result the correct s
    R = R.reshape(Fshape[1:])

    return R


# -----------------------------------------------


def s_stretch(N, theta_s, theta_b, stagger="rho", Vstretching=1):
    """Compute a s-level stretching array

    *N* : Number of vertical levels

    *theta_s* : Surface stretching factor

    *theta_b* : Bottom stretching factor

    *stagger* : "rho"|"w"

    *Vstretching* : 1|2|4

    """

    if stagger == "rho":
        S = -1.0 + (0.5 + np.arange(N)) / N
    elif stagger == "w":
        S = np.linspace(-1.0, 0.0, N + 1)
    else:
        raise ValueError("stagger must be 'rho' or 'w'")

    if Vstretching == 1:
        cff1 = 1.0 / np.sinh(theta_s)
        cff2 = 0.5 / np.tanh(0.5 * theta_s)
        return (1.0 - theta_b) * cff1 * np.sinh(theta_s * S) + theta_b * (
            cff2 * np.tanh(theta_s * (S + 0.5)) - 0.5
        )

    elif Vstretching == 2:
        a, b = 1.0, 1.0
        Csur = (1 - np.cosh(theta_s * S)) / (np.cosh(theta_s) - 1)
        Cbot = np.sinh(theta_b * (S + 1)) / np.sinh(theta_b) - 1
        mu = (S + 1) ** a * (1 + (a / b) * (1 - (S + 1) ** b))
        return mu * Csur + (1 - mu) * Cbot

    elif Vstretching == 4:
        C = (1 - np.cosh(theta_s * S)) / (np.cosh(theta_s) - 1)
        C = (np.exp(theta_b * C) - 1) / (1 - np.exp(-theta_b))
        return C

    else:
        raise ValueError("Unknown Vstretching")


def invert_s(F: xr.DataArray, value: Surface):
    """Return highest (shallowest) s-value such that F(s,...) = value

    F = DataArray with z_rho as coordinate

    The vertical dimension in F must be first, axis=0
    F must not have a time dimension

    Returns D, Dm, a
    F[Dm] <= value <= F[D] (or opposite inequalities)
    and a is the interpolation weight:
    value = (1-a)*F(K-1) + a*F(K)
    a = nan if this is not possible

    """

    val = value
    # Work on numpy arrays
    F0 = F.values
    # z_rho = F.z_rho.values
    # s_rho = F.s_rho.values
    val = np.asarray(val, dtype="float")
    # Fshape = F.shape  # Save original shape
    # if val.shape and val.shape != Fshape[1:]:
    #     raise ValueError("z must be scalar or have shape = F.shape[1:]")

    # Flatten all non-vertical dimensions
    N = F.shape[0]  # Length of vertical dimension
    M = F0.size // N  # Combined length of horizontal dimensions
    F0 = F0.reshape((N, M))
    if val.shape:  # Value may be space dependent
        val = val.reshape((M,))

    # Look for highest s-value where G is negative
    G = (F0[1:, :] - val) * (F0[:-1, :] - val)
    G = G[::-1, :]  # Reverse
    K = N - 1 - (G <= 0).argmax(axis=0)

    # Define D such that F[D][i] = F[K[i], i]
    I = np.arange(M)
    D = (K, I)
    Dm = (K - 1, I)

    # Compute interpolation weights
    a = (val - F0[Dm]) / (F0[D] - F0[Dm] + 1e-30)
    # Only use 0 <= a <= 1
    a[np.abs(a - 0.5) > 0.5] = np.nan  #

    return D, Dm, a


class HorizontalSlicer:
    """Reduce to horizontal view by slicing

    F = DataArray,  time-independent, first dimension is vertical
    value = slice value

    If F is not monotonous, returns the shallowest depth where F = value

    """

    def __init__(self, F: xr.DataArray, value: Surface) -> None:
        self.D, self.Dm, self.a = invert_s(F, value)
        self.M = len(self.a)
        # self.dims = F.dims

    def __call__(self, G: xr.DataArray) -> xr.DataArray:
        """G must have same vertical and horizontal dimensions as F"""

        if "ocean_time" in G.dims:
            ntimes = G.shape[0]
            kmax = G.shape[1]
            R: List[np.ndarray] = []
            for t in range(ntimes):
                G0 = G.isel(ocean_time=t).values
                G0 = G0.reshape((kmax, self.M))
                R0 = (1 - self.a) * G0[self.Dm] + self.a * G0[self.D]
                R0 = R0.reshape(G.shape[2:])
                R.append(R0)
            R1 = np.array(R)
        else:
            kmax = G.shape[0]
            G0 = G.values
            G0 = G0.reshape((kmax, self.M))
            R1 = (1 - self.a) * G0[self.Dm] + self.a * G0[self.D]
            R1 = R1.reshape(G.shape[1:])

        # Return a DataArray
        # Should have something on z_rho?
        dims = list(G.dims)
        dims.remove("s_rho")
        coords = {dim: G.coords[dim] for dim in dims}

        coords["lon_rho"] = G.coords["lon_rho"]
        coords["lat_rho"] = G.coords["lat_rho"]
        return xr.DataArray(R1, dims=dims, coords=coords, attrs=G.attrs)
