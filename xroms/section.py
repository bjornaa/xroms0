import numpy as np
import xarray as xr
import xroms


def section(A, X, Y):
    """Slice a Dataset along a vertical section"""

    if 's_rho' not in A:
        raise ValueError("Section requires vertical dimension s_rho")

    Npoints = len(X)

    # Make temporary DataArrays for initial interpolations
    X0 = xr.DataArray(X, dims=['distance'])
    Y0 = xr.DataArray(Y, dims=['distance'])
    # Distance between section points
    pm = A.pm.interp(xi_rho=X0, eta_rho=Y0).values
    pn = A.pn.interp(xi_rho=X0, eta_rho=Y0).values
    dX = 2 * np.diff(X) / (pm[:-1] + pm[1:])
    dY = 2 * np.diff(Y) / (pn[:-1] + pn[1:])
    dS = np.sqrt(dX * dX + dY * dY)
    # Cumulative distance along the section
    distance = np.concatenate(([0], np.cumsum(dS))) / 1000.0  # unit = km
    X0['distance'] = distance
    Y0['distance'] = distance

    # Interpolate to the section making an intermediate Dataset
    B0 = A.interp(xi_rho=X0, eta_rho=Y0)

    # Initialize the proper section Dataset
    B = xr.Dataset(dict(xi_rho=X, eta_rho=Y, s_rho=A.s_rho))

    # Weights for trapezoidal integration
    V = 0.5*(np.concatenate(([0], dS)) + np.concatenate((dS, [0])))
    B['dS'] = xr.DataArray(V, dims=['distance'], coords=dict(distance=distance))

    # Vertical coordinates
    B.coords['z_rho'] = B0.z_rho
    B.coords['depth'] = -B0.z_rho

    # Scalar data variables
    for var in ['h', 'temp', 'salt', 'zeta']:
        if var in B0:
            B[var] = B0[var]

    # Velocity
    if 'u' in A:
        B['u'] = A.u.interp(xi_u=X0, eta_u=Y0)
        B = B.drop(['xi_u', 'eta_u'])
    if 'v' in A:
        B['v'] = A.v.interp(xi_v=X0, eta_v=Y0)
        B = B.drop(['xi_v', 'eta_v'])

    # Normal velocity ++
    if 'u' in A:
        dX = diff2(X)
        dY = diff2(Y)
        norm = np.sqrt(dX*dX + dY*dY)
        # Unit normal vector
        nX = -dY / norm
        nY = dX / norm
        B['u_norm'] = B['u'] * nX + B['v'] * nY

    # More vertical structure
    B.coords['z_w'] = B0.z_w
    V = B.z_w.values[1:, :] - B.z_w.values[:-1, :]
    B['dZ'] = xr.DataArray(V, dims=['s_rho', 'distance'])
    B['area'] = B.dZ * B.dS   # Unit = m**2

    return B


def diff2(X):
    """Central differences of a sequence"""
    n = len(X)
    Y = np.empty(n+2, dtype=X.dtype)
    Y[1:-1] = X
    Y[0] = X[0]
    Y[-1] = X[-1]
    return Y[2:] - Y[:-2]
