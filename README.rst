XROMS
=====

xroms is a package of post-processing tools for the ROMS
ocean model.

The implementation is based on the versatile xarray package,
giving DataArrays with named dimensions. It supersedes the older roppy package.

Naming follows the ROMS conventions, with a few exceptions.
In ROMS netCDF files, the dimension is 'time', while the
corresponding variable is 'ocean_time'. In xroms, the
variable is named 'time', making it a proper coordinate.

  Bjørn Ådlandsvik <bjorn at imr.no>
  Institute of Marine Research
  Bergen, Norway