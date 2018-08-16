=========
Functions
=========

XROMS provides some convenient functions for working with ROMS output.
The most important tool is the

.. function:: roms_dataset(roms_file)

   :arg str roms_file: Name of ROMS output file

Opens a ROMS output file as an `xarray` Dataset. It provides the necessary
coordinate variables. If the file contains 3D data, the function also adds
the variable `z_rho` with the local depth of the s-surfaces.

.. class:: HorizontalSlicer(F, value)

    :arg DataArray F:  3D time indepent ROMS DataArray

    :arg float value:  The F-value to slice at.

If F is not vertically monotonous uses the shallowest depth where F = value.

A HorizontalSlicer instance is callable, and slices a 3D or 4D ROMS DataArray
at the depth where F = value.


.. function:: zslice(G, depth)

   :arg DataArray G:  3D/4D ROMS field
   :arg float depth:  The depth of the slice

Convenience function for::

   slicer = HorizontalSlicer(z_rho, -abs(depth))
   G_slice = slicer(G)

To slice many DataArrays at the same depth, it is faster to make the slicer
object once and reuse it.
