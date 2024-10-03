import numpy as np
import sys
import os
import re
import xml.etree.ElementTree
import matplotlib.pyplot as plt
import matplotlib.animation


def norm(path):
    path = re.sub("^[ \t\n]*", "", path)
    path = re.sub("[ \t\n]*$", "", path)
    return path


def plot(path):
    x = xml.etree.ElementTree.parse(path)
    root = x.getroot()
    xyz, attr = root.findall('.//DataItem[@Format="Binary"]')
    dirname = os.path.dirname(path)
    xyz_path = os.path.join(dirname, norm(xyz.text))
    attr_path = os.path.join(dirname, norm(attr.text))

    xyz = np.memmap(xyz_path, np.dtype("f4"), "r", order="C")
    xyz = np.reshape(xyz, (-1, 8, 3))
    xyz = xyz[:, 0, :]

    attr = np.memmap(attr_path, np.dtype("f4"), "r", order="C")
    attr = np.reshape(attr, (-1, 3))

    xx = []
    yy = []
    zz = []
    for (x, y, z), (vx, vy, vz) in zip(xyz, attr):
        if abs(vx) or abs(vy) or abs(vz):
            xx.append(x)
            yy.append(y)
            zz.append(z)
    points.set_data(xx, yy)


plt.axis("equal")
plt.axis((0, 1, 0, 1))
points, = plt.plot([], [], 'o', alpha=0.001)
anim = matplotlib.animation.FuncAnimation(plt.gcf(), plot, sys.argv[1:])
anim.save("post.mp4")
