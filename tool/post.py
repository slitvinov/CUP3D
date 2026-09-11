#!/usr/bin/env python3
import numpy as np
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation


def read(path):
    base = os.path.splitext(path)[0]
    blk = np.fromfile(base + ".blk.raw", "f4").reshape(-1, 6)
    attr = np.fromfile(base + ".attr.raw", "f4")
    origin = blk[:, 2::-1]
    h = blk[:, 3]
    i = np.arange(512)
    idx = np.stack([i % 8, (i // 8) % 8, i // 64], 1)
    lo = origin[:, None, :] + h[:, None, None] * idx[None, :, :]
    lo = lo.reshape(-1, 3)
    hi = lo + np.repeat(h, 512)[:, None]
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
