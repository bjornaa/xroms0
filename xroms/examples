# -*- coding: utf-8 -*-

"""Extract a regional coast line as a WKB file

Extracts a coast line in lat-lon from GSHHS by
using cartopy and the underlying shapely library

"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute if Marine Research
# 2016-08-12
# ----------------------------------

import shapely.geometry as geom
import shapely.wkb as wkb
import cartopy.feature as cfeature


def main():
    """Main function if used as script"""

    coastfile = "coast.wkb"

    # Choose between c, l, i, h, f resolutions
    GSHHSres = 'i'

    # Define regional domain
    lonmin, lonmax, latmin, latmax = -21, 18, 47, 67

    coast2wkb(**locals())


def coast2wkb(lonmin, lonmax, latmin, latmax, GSHHSres, coastfile):

    # Global cartopy feature from GSHHS
    gfeat = cfeature.GSHHSFeature(scale=GSHHSres)
    # As shapely collection generator
    coll = gfeat.geometries()

    # Polygon representation of  the regional domain
    frame = geom.box(lonmin, latmin, lonmax, latmax)

    # The intersection
    B = (frame.intersection(p) for p in coll if frame.intersects(p))

    # Save to file
    with open(coastfile, mode='wb') as fp:
        wkb.dump(geom.MultiPolygon(flatten(B)), fp, output_dimension=2)


# The intersection may consist of both polygons and multipolygons,
# flatten it to an iterator of polygons
def flatten(B):
    """Generator to flatten an iterator of Polygons and MultiPolygons"""
    for p in B:
        if type(p) == geom.polygon.Polygon:
            yield p
        else:  # MultiPolygon
            for q in p:
                yield q


if __name__ == '__main__':
    main()
