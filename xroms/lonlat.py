import numpy as np
from scipy.interpolate import griddata
# from sample import bilin_inv


def xy2ll(A, x, y):

    lon = A['lon_rho'].interp(xi_rho=x, eta_rho=y)
    lat = A['lat_rho'].interp(xi_rho=x, eta_rho=y)

    # return lon, lat
    return lon, lat


# Metoden som velges må returnere Dataarray, slik som xy2ll
def ll2xy1(A, lon, lat):
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


# ll2xy2 er raskere (2000 ganger!)
# Har litt mer trøbbel nær randen.
# Mer nøyaktig?? I forhold til xy2ll (som selv er unøyaktig)
# Bedre: sammenligne med eksplisitt kartprojeksjon

# def ll2xy2(A, lon, lat):
#     y, x = bilin_inv(lon, lat, A['lon_rho'].data, A['lat_rho'].data)
#     return x, y

# Nøyaktigst, men litt seinere enn ll2xy1
def ll2xy3(A, lon, lat):
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


if __name__ == '__main__':

    import xroms

    A = xroms.roms_dataset('ocean_avg_0014.nc')

    x, y = 170.0, 120.0
    print(x, y)
    lon, lat = xy2ll(A, x=x, y=y)
    print('lon, lat', lon.data, lat.data)
    print('-----------')
    x1, y1 = ll2xy1(A, lon, lat)
    print(x1-x, y1-y)
    x2, y2 = ll2xy2(A, lon, lat)
    print(x2-x, y2-y)
    x3, y3 = ll2xy3(A, lon, lat)
    print(x3 - x, y3 - y)
    # print(A.lon_rho.sel(xi_rho=x, eta_rho=y))
    # print('-----------')
    # print(A.lat_rho.sel(xi_rho=x, eta_rho=y))

