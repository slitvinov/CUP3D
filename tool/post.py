import numpy as np
import sys
import os
import re
import xml.etree.ElementTree
import matplotlib.pyplot as plt
import matplotlib.animation

def plot(path):
    x = xml.etree.ElementTree.parse(path)
    root = x.getroot()
    bin_data = root.find('.//DataItem[@Format="Binary"]')
    bin_path = bin_data.text
    bin_path = re.sub("^[ \t\n]*", "", bin_path)
    bin_path = re.sub("[ \t\n]*$", "", bin_path)
    bin_data = re.sub("^.*:", "", bin_path)
    bin_path = re.sub(":[^:]*$", "", bin_path)
    dirname = os.path.dirname(path)
    bin_path = os.path.join(dirname, bin_path)
    xx = [ ]
    yy = [ ]
    zz = [ ]
    with h5py.File(bin_path) as f:
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
            data = f[bin_data][offset:offset + size]
            data = data.reshape((nx - 1, ny - 1, nz - 1))
            ix, iy, iz = np.where(data > 0)
            x = (ix + 1 / 2) * sx + ox
            y = (iy + 1 / 2) * sy + oy
            z = (iz + 1 / 2) * sz + oz
            xx.append(x)
            yy.append(y)
            zz.append(z)
            offset += size
    points.set_data(np.concatenate(zz), np.concatenate(yy))

plt.axis((0, 1, 0, 1))
plt.xticks([], [])
plt.axis("equal")
points, = plt.plot([], [], 'o', alpha=0.1)
anim = matplotlib.animation.FuncAnimation(plt.gcf(), plot, sys.argv[1:])
anim.save("plot.mp4")
