from pytest import approx
import numpy as np
import xroms
from xroms import ll2xy, xy2ll

A = xroms.roms_dataset('../examples/ocean_avg_0014.nc')


def test_xy2ll2xy():
    x, y = 100, 110
    lon, lat = xy2ll(A, x, y)
    x1, y1 = ll2xy(A, lon, lat)
    assert (x1 == approx(x))
    assert (y1 == approx(y))


def test_ll2xy2ll():
    lon, lat = 5, 60
    x, y = ll2xy(A, lon, lat)
    lon1, lat1 = xy2ll(A, x, y)
    print(float(lon1), float(lat1))
    assert (lon1 == approx(lon, abs=1e-4))
    assert (lat1 == approx(lat))


def test_ll2xy():
    """Testing edge cases, literally"""

    x = [0, 0, 0, 100, 181, 181, 181]
    y = [0, 100, 191, 191, 191, 100, 0]
    lon, lat = xy2ll(A, x, y)
    x1, y1 = ll2xy(A, lon, lat)
    # print(x1)
    # print(y1)
    assert (~np.any(np.isnan(x1)))
    assert (~np.any(np.isnan(y1)))


if __name__ == '__main__':
    test_ll2xy()
