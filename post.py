import h5py
import numpy as np
import os
import re
import xml.etree.ElementTree

# f = h5py.File("data/chi_000000001.h5")
# print(f)

path = "data/chi_000000001.xmf"
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

with h5py.File(hdf_path) as f:
    offset = 0
    for grid in root.findall('.//Grid[@GridType="Uniform"]'):
        topology = grid.find('./Topology')
        geometry = grid.find('./Geometry[@GeometryType="ORIGIN_DXDYDZ"]')
        dorg, dspa = geometry.findall("./DataItem")
        ddim = topology.get("Dimensions")
        org = [float(e) for e in dorg.text.split()]
        spa = [float(e) for e in dspa.text.split()]
        nx, ny, nz = [int(e) for e in ddim.split()]
        size = (nx - 1) * (ny - 1) * (nz - 1)
        data = f[hdf_data][offset:offset + size]
        offset += size
        print(data)
