import numpy as np
import xarray as xr
import depth


def roms_dataset(roms_file, subgrid=None):
    """Make a ROMS xarray Dataset from a ROMS file"""

    # Variables we care about
    # grid_vars = ['mask_rho', 'h', 'lon_rho', 'lat_rho', 'pm', 'pn',
    #              'time', 's_rho', 's_w', 'time']
    grid_vars = ['mask_rho', 'h', 'pm', 'pn',
                 'time', 's_rho', 's_w', 'time']
    data_vars = ['zeta', 'u', 'v', 'temp', 'salt']

    # Read the ROMS file
    A0 = xr.open_dataset(roms_file)

    # Select the variables
    vars = [var for var in grid_vars + data_vars if var in A0]

    A = xr.Dataset({var: A0[var] for var in vars})

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

    # --- Temporal ---
    if 'time' in A.dims:
        A['time'] = A0.ocean_time

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
        # ta omvei om Cs_r ikke finnes

        z_rho = depth.sdepth(
            A.h, np.float32(A0.hc), A0.Cs_r,
            stagger='rho', Vtransform=Vtransform)
        z_rho = xr.DataArray(z_rho.astype('float32'),
                             dims=('s_rho', 'eta_rho', 'xi_rho'),
                             attrs={'long_name': 'depth of s-surfaces',
                                    'units': 'meter',
                                    'positive': 'up'})

        z_w = depth.sdepth(
            A.h, np.float32(A0.hc), A0.Cs_w,
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
    """Horizontal slice of a ROMS DataArray"""

    # Slice
    if 'time' in F.dims:
        V = []
        for t in range(len(F['time'])):
            V0 = depth.zslice(F[t, ...], F.z_rho, -abs(z))
            V0 = np.where(abs(z) + F.z_rho[0] <= 0, V0, np.nan)
            V.append(V0)
        V = np.array(V)
    else:
        V = depth.zslice(F, F.z_rho, -abs(z))
        V = np.where(abs(z) + F.z_rho[0] <= 0, V, np.nan)  # Mask out values below bottom

    # Make a DataArray
    dims = list(F.dims)
    dims.remove('s_rho')
    coords = {dim: F.coords[dim] for dim in dims}
    coords['z_rho'] = -abs(z)
    coords['lon_rho'] = F.coords['lon_rho']
    coords['lat_rho'] = F.coords['lat_rho']
    attrs = F.attrs
    # attrs['depth'] = abs(z)
    return xr.DataArray(V, dims=dims, coords=coords, attrs=attrs)


def zslice_ds(A, z):
    """Horizontal slice av ROMS Dataset"""
    # Kanskje dårlig idé. Har ikke z_rho på u og v
    # Må ha z_u og z_v dersom skikkelig.
    # eller gjøre det på u_rho

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


def zslice(D, z):
    if isinstance(D, xr.DataArray):
        return zslice_da(D, abs(z))
    else:
        return zslice_ds(D, abs(z))


# ------------------------------------------------------------
def subgrid(A, subgrid, stagger='outer'):
    """Make a ROMS xarray Dataset on a horizontal subgrid"""

    # Bestemme om endepunkter skal med eller ikke
    # Få alternativ, med koordinater

    # Antar at xi_rho og eta_rho alltid er med
    # imax = len(A.xi_rho)
    # jmax = len(A.eta_rho)

    # Hva med slicing, inner or outer velocity?
    # Vite hva man har (len(A.xi_u) vs. len(A.xi_rho)

    x0, x1, y0, y1 = subgrid

    # Mangler test om plass
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

