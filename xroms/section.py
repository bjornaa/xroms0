import numpy as np
import xarray as xr
import xroms


def section(A, X, Y):
    """Slice a Dataset along a vertical section"""

    Npoints = len(X)

    # Make temporary DataArrays for initial interpolations
    X0 = xr.DataArray(X, dims=['q'], coords=dict(q=np.arange(Npoints)))
    Y0 = xr.DataArray(Y, dims=['q'], coords=dict(q=np.arange(Npoints)))
    # Distance between section points
    pm = A.pm.interp(xi_rho=X0, eta_rho=Y0).values
    pn = A.pn.interp(xi_rho=X0, eta_rho=Y0).values
    dX = 2 * np.diff(X) / (pm[:-1] + pn[1:])
    dY = 2 * np.diff(Y) / (pm[:-1] + pn[1:])
    dS = np.sqrt(dX * dX + dY * dY)
    # Cumulative distance along the section
    distance = np.concatenate(([0], np.cumsum(dS))) / 1000.0  # unit = km

    # Make DataArrays, dimension = distance
    X = xr.DataArray(X, dims=['distance'], coords=dict(distance=distance))
    Y = xr.DataArray(Y, dims=['distance'], coords=dict(distance=distance))

    # Temporary section Dataset
    B0 = A.interp(xi_rho=X, eta_rho=Y)

    # Initiate the section Dataset
    B = xr.Dataset(dict(xi_rho=X, eta_rho=Y, s_rho=B0.s_rho))

    # Weights for trapezoidal integration
    V = 0.5*(np.concatenate(([0], dS)) + np.concatenate((dS, [0])))
    B['dS'] = xr.DataArray(V, dims=['distance'], coords=dict(distance=distance))

    # Vertical coordinates
    B.coords['z_rho'] = B0.z_rho
    B.coords['depth'] = -B0.z_rho

    # Scalar data variables
    B['h'] = B0.h
    if 'temp' in B0:
        B['temp'] = B0.temp
    if 'salt' in B0:
        B['salt'] = B0.salt
    if 'zeta' in B0:
        B['zeta'] = B0.zeta

    # Velocity
    if 'u' in A:
        B['u'] = A.interp(xi_u=X, eta_u=Y).u
        B = B.drop(['xi_u', 'eta_u'])
    if 'v' in A:
        B['v'] = A.interp(xi_v=X, eta_v=Y).v
        B = B.drop(['xi_v', 'eta_v'])

    # Normal velocity
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
        # B['dZ'] = B.z_w.diff(dim='s_w') # wrong dimensions
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


if __name__ == '__main__':
    A = xroms.roms_dataset('ocean_avg_0014.nc')
    x0, y0 = 70, 94
    x1, y1 = 120, 80
    Npoints = 50
    X = np.linspace(x0, x1, Npoints)
    Y = np.linspace(y0, y1, Npoints)
    B = section(A, X, Y)
