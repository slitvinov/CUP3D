import h5py
import numpy as np
import sys
import os
import re
import xml.etree.ElementTree

path = sys.argv[1]
x = xml.etree.ElementTree.parse(path)
root = x.getroot()
hdf_data = root.find('.//DataItem[@Format="HDF"]')
hdf_path = hdf_data.text
hdf_path = re.sub("^[ \t\n]*", "", hdf_path)
hdf_path = re.sub("[ \t\n]*$", "", hdf_path)
hdf_data = re.sub("^.*:", "", hdf_path)
hdf_path = re.sub(":[^:]*$", "", hdf_path)
dirname = os.path.dirname(path)
hdf_path = os.path.join(dirname, hdf_path)
with h5py.File(hdf_path) as f, open("a.txt", "w") as fo:
    offset = 0
    for grid in root.findall('.//Grid[@GridType="Uniform"]'):
        topology = grid.find('./Topology')
        geometry = grid.find('./Geometry[@GeometryType="ORIGIN_DXDYDZ"]')
        dorg, dspa = geometry.findall("./DataItem")
        ddim = topology.get("Dimensions")
        ox, oy, oz = [float(e) for e in dorg.text.split()]
        sx, sy, sz = [float(e) for e in dspa.text.split()]
        nx, ny, nz = [int(e) for e in ddim.split()]
        size = (nx - 1) * (ny - 1) * (nz - 1)
        data = f[hdf_data][offset:offset + size]
        data = data.reshape((nx - 1, ny - 1, nz - 1))
        ix, iy, iz = np.where(data > 0)
        x = (ix + 1 / 2) * sx + ox
        y = (iy + 1 / 2) * sy + oy
        z = (iz + 1 / 2) * sz + oz
        np.savetxt(fo, np.column_stack((x, y, z)))
        offset += size
