from pytest import approx
import numpy as np
import xroms
from xroms import ll2xy, xy2ll

# A = xroms.roms_dataset('../examples/ocean_avg_0014.nc')
A = xroms.roms_dataset('/home/bjorn/python/xroms/examples/ocean_avg_0014.nc')
A = xroms.roms_dataset('examples/ocean_avg_0014.nc')


def test_xy2ll2xy():
    """grid to geography to grid should be identity"""
    x, y = 100, 110
    lon, lat = xy2ll(A, x, y)
    x1, y1 = ll2xy(A, lon, lat)
    assert (x1 == approx(x))
    assert (y1 == approx(y))


def test_ll2xy2ll():
    """geography to grid to geography should be identity"""
    lon, lat = 5, 60
    x, y = ll2xy(A, lon, lat)
    lon1, lat1 = xy2ll(A, x, y)
    print(float(lon1), float(lat1))
    assert (lon1 == approx(lon, abs=1e-4))
    assert (lat1 == approx(lat))


def test_ll2xy_edge():
    """Testing edge cases, literally"""

    x = [0, 0, 0, 100, 181, 181, 181]
    y = [0, 100, 191, 191, 191, 100, 0]
    lon, lat = xy2ll(A, x, y)
    x1, y1 = ll2xy(A, lon, lat)
    # print(x1)
    # print(y1)
    assert (~np.any(np.isnan(x1)))
    assert (~np.any(np.isnan(y1)))


def test_xy2ll_shape():
    """Should return same shape"""
    # Scalars
    x, y = 20, 50
    lon, lat = xy2ll(A, x, y)
    assert(np.isscalar(lon))
    assert(np.isscalar(lat))

    # 1D arrays, same shape
    x = [20, 25]
    y = [50, 55]
    lon, lat = xy2ll(A, x, y)
    assert(lon.shape == (2,))
    assert(lat.shape == (2,))

    # Broadcast
    x = 20
    y = [50, 55]
    lon, lat = xy2ll(A, x, y)
    assert(lon.shape == np.broadcast(x, y).shape)
    assert(lat.shape == np.broadcast(x, y).shape)


def test_ll2xy_shape():
    """Should return same shape"""
    # Scalars
    lon, lat = 5, 60
    x, y = ll2xy(A, lon, lat)
    assert(np.isscalar(x))
    assert(np.isscalar(y))

    # 1D arrays, same shape
    lon = [5, 6]
    lat = [60, 58]
    x, y = ll2xy(A, lon, lat)
    assert(x.shape == (2,))
    assert(y.shape == (2,))

    # Broadcast
    lon = [4, 5, 6,]
    lat = 60
    x, y = ll2xy(A, lon, lat)
    assert(x.shape == np.broadcast(lon, lat).shape)
    assert(y.shape == np.broadcast(lon, lat).shape)


if __name__ == '__main__':
    test_xy2ll_shape()
