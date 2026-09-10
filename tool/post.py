#!/usr/bin/env python3
import numpy as np
import sys
import os
import xml.etree.ElementTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation


def read(path):
    root = xml.etree.ElementTree.parse(path).getroot()
    xyz = root.find('.//Geometry/DataItem')
    attr = root.find('.//Attribute[@Name="chi"]/DataItem')
    dirname = os.path.dirname(path)
    xyz = np.memmap(os.path.join(dirname, xyz.text.strip()), "f4", "r")
    attr = np.memmap(os.path.join(dirname, attr.text.strip()), "f4", "r")
    xyz = np.reshape(xyz, (-1, 8, 3))
    lo = xyz[:, 0, :]
    hi = xyz[:, 6, :]
    return (lo + hi) / 2, lo, hi, attr


def plot(path):
    center, lo, hi, attr = read(path)
    mask = attr > 0
    points.set_data(center[mask, 0], center[mask, 1])
    fig.savefig(os.path.splitext(path)[0] + ".png")
    return points,


if len(sys.argv) < 2:
    sys.stderr.write("usage: post.py FILE.xdmf2 [FILE.xdmf2 ...]\n")
    sys.exit(1)

center, lo, hi, attr = read(sys.argv[1])
fig, ax = plt.subplots()
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(lo[:, 0].min(), hi[:, 0].max())
ax.set_ylim(lo[:, 1].min(), hi[:, 1].max())
points, = ax.plot([], [], 'o', alpha=0.1)
anim = matplotlib.animation.FuncAnimation(fig, plot, sys.argv[1:],
                                          cache_frame_data=False)
anim.save("post.mp4")
