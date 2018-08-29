import numpy as np
import xarray as xr
from . import depth


def roms_dataset(roms_file, subgrid=None):
    """Make a ROMS xarray Dataset from a ROMS file"""

    # Variables we care about
    grid_vars = ['mask_rho', 'h', 'lon_rho', 'lat_rho', 'pm', 'pn',
                 'ocean_time', 's_rho', 's_w', 'ocean_time']
    data_vars = ['zeta', 'u', 'v', 'temp', 'salt']

    # Read the ROMS file
    A0 = xr.open_dataset(roms_file)
    # Old ROMS ouptut have dimension 'time' instead of 'ocean_time'
    if 'time' in A0.dims:
        A0.rename({'time': 'ocean_time'}, inplace=True)

    # Select the variables
    variables = [var for var in grid_vars + data_vars if var in A0]

    A = xr.Dataset({var: A0[var] for var in variables})

    # Fill in missing coordinate variables
    # --- Horizontal ---
    imax = len(A.xi_rho)
    jmax = len(A.eta_rho)
    # xi_rho and eta_rho are always present
    A['xi_rho'] = np.arange(imax)
    A['eta_rho'] = np.arange(jmax)
    if 'xi_u' in A.dims:
        A['xi_u'] = np.arange(0.5, imax - 1)
    if 'xi_v' in A.dims:
        A['xi_v'] = np.arange(imax)
    if 'eta_u' in A.dims:
        A['eta_u'] = np.arange(jmax)
    if 'eta_v' in A.dims:
        A['eta_v'] = np.arange(0.5, jmax - 1)

    # --- Vertical handling ---
    if 's_rho' in A.dims:
        # Evenly distributed coordinates from -1 to 0
        kmax = len(A.s_rho)
        A['s_rho'] = -1.0 + (0.5 + np.arange(kmax)) / kmax
        if 's_w' in A0.dims:
            A['s_w'] = np.linspace(-1.0, 0.0, kmax + 1)

        # Make the z_rho array
        if 'Vtransform' not in A0:
            Vtransform = 1
        else:
            Vtransform = int(A0.Vtransform)
        # Should handle different ways to geg  vertical structure

        z_rho = depth.sdepth(
            A.h, np.float32(A0.hc), A0.Cs_r,
            stagger='rho', Vtransform=Vtransform)
        z_rho = xr.DataArray(z_rho.astype('float32'),
                             dims=('s_rho', 'eta_rho', 'xi_rho'),
                             attrs={'long_name': 'depth of s-surfaces',
                                    'units': 'meter',
                                    'positive': 'up'})

        z_w = depth.sdepth(A.h, np.float32(A0.hc), A0.Cs_w,
                           stagger='w', Vtransform=Vtransform)
        z_w = xr.DataArray(z_w.astype('float32'),
                           dims=('s_w', 'eta_rho', 'xi_rho'),
                           attrs={'long_name': 'depth of s-interfaces',
                                  'units': 'meter',
                                  'positive': 'up'})

        A.coords['z_rho'] = (('s_rho', 'eta_rho', 'xi_rho'), z_rho)
        A.coords['z_w'] = z_w

    # Add geographic coordinates
    if 'lon_rho' in A0:
        A.coords['lat_rho'] = (('eta_rho', 'xi_rho'), A0.lat_rho)
        A.coords['lon_rho'] = (('eta_rho', 'xi_rho'), A0.lon_rho)

    return A


# ---------------------------------
def zslice_da(F, z):
    """Horizontal slice of DataArray at fixed depth"""

    z0 = -abs(z)
    vslice = depth.HorizontalSlicer(F.z_rho, z0)
    G = vslice(F)
    G['z_rho'] = z0
    return G


def zslice_ds(A, z):
    """Horizontal slice av ROMS Dataset at fixed depth"""
    # May be unnecessary, hard to get enough flexibility?

    data = dict(z_rho=-abs(z))
    if 'u' in A:
        data['u'] = zslice_da(A.u, z)
    if 'v' in A:
        data['v'] = zslice_da(A.v, z)
    if 'temp' in A:
        data['temp'] = zslice_da(A.temp, z)
    if 'salt' in A:
        data['salt'] = zslice_da(A.salt, z)

    B = xr.Dataset(data)
    return B


def zslice(D, z):
    if isinstance(D, xr.DataArray):
        return zslice_da(D, z)
    else:
        return zslice_ds(D, z)


# ------------------------------------------------------------
def subgrid(A, subgrid_spec, stagger='outer'):
    """Make a ROMS xarray Dataset on a horizontal subgrid"""

    # Suppese xi_rho and eta_rho allways present
    # imax = len(A.xi_rho)
    # jmax = len(A.eta_rho)

    # How about slicing, inner or outer velocity
    # input: depends on (len(A.xi_u) vs. len(A.xi_rho)

    x0, x1, y0, y1 = subgrid_spec

    sub = dict(xi_rho=slice(x0, x1), eta_rho=slice(y0, y1))
    sub['xi_u'] = np.arange(x0-0.5, x1+1)
    sub['xi_v'] = slice(x0, x1)
    sub['eta_u'] = slice(y0, y1)
    sub['eta_v'] = np.arange(y0-0.5, y1+1)

    return A.sel(**sub)


def get_stagger(A):
    if 'xi_rho' in A.dims:
        len_rho = len(A.xi_rho)
        len_u = len(A.xi_u)
        if len_rho > len_u:
            return 'inner'
        else:
            return 'outer'


if __name__ == '__main__':
    A = roms_dataset('ocean_avg_0014.nc')

    # B = zslice(A, 'temp', 50)
