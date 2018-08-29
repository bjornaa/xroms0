from typing import Union, Sequence, Tuple
import numpy as np
from scipy.interpolate import griddata
import xarray as xr
# from sample import bilin_inv

# Desired behavior:
# x, y should have same shape,
# return lon, lat DataArrays of same shape
# return DataArr out if DataArr in?
#    array, list in -> array out
#     scalars in -> scalars out
#
# Alternative: work like xarray indexing and interp
#

ArrayLike = Union[float, Sequence[float]]
Array = Union[float, np.ndarray]


def xy2ll(A: xr.Dataset,
          x: ArrayLike,
          y: ArrayLike) -> Tuple[Array, Array]:
    """Convert from grid coordinates to longitude/latitude"""

    xb, yb = np.broadcast_arrays(x, y)
    x_da = xr.DataArray(xb)
    y_da = xr.DataArray(yb)
    lon = A['lon_rho'].interp(xi_rho=x_da, eta_rho=y_da).values
    lat = A['lat_rho'].interp(xi_rho=x_da, eta_rho=y_da).values

    # Return scalars if both x and y are scalars
    if np.isscalar(x) and np.isscalar(y):
        lon = float(lon)
        lat = float(lat)

    return lon, lat


def ll2xy(A: xr.Dataset,
          lon: ArrayLike,
          lat: ArrayLike) -> Tuple[Array, Array]:
    """Convert from longitude/latitude to grid coordinates"""
    # Choose projection method
    x, y = ll2xy1(A, lon, lat)

    # Return scalars if both lon and lat are scalars
    if np.isscalar(lon) and np.isscalar(lat):
        x = float(x)
        y = float(y)

    return x, y


def ll2xy1(A: xr.Dataset,
           lon: ArrayLike,
           lat: ArrayLike) -> Tuple[Array, Array]:
    gLon = A['lon_rho'].data.ravel()
    gLat = A['lat_rho'].data.ravel()
    X0 = A['xi_rho'].data
    Y0 = A['eta_rho'].data
    gX, gY = np.meshgrid(X0, Y0)
    gX = gX.ravel()
    gY = gY.ravel()
    x = griddata((gLon, gLat), gX, (lon, lat), 'linear')
    y = griddata((gLon, gLat), gY, (lon, lat), 'linear')
    return x, y

# def ll2xy2(A, lon, lat):
#     y, x = bilin_inv(lon, lat, A['lon_rho'].data, A['lat_rho'].data)
#     return x, y


# More accurate, but slower than ll2xy1
def ll2xy3(A: xr.Dataset,
           lon: ArrayLike,
           lat: ArrayLike) -> Tuple[Array, Array]:
    gLon = A['lon_rho'].data.ravel()
    gLat = A['lat_rho'].data.ravel()
    X0 = A['xi_rho'].data
    Y0 = A['eta_rho'].data
    gX, gY = np.meshgrid(X0, Y0)
    gX = gX.ravel()
    gY = gY.ravel()
    x = griddata((gLon, gLat), gX, (lon, lat), 'cubic')
    y = griddata((gLon, gLat), gY, (lon, lat), 'cubic')
    return x, y
