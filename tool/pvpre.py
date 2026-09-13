import os
import sys
import numpy as np

if len(sys.argv) < 3:
    sys.stderr.write("usage: pvpre.py LEVEL FILE.xdmf2 [FILE.xdmf2 ...]\n")
    sys.exit(1)
spec = sys.argv[1]
tag = "q%g" % float(spec[1:]) if spec.startswith("q") else "omega%g" % float(spec)
m = 0.0
body = [1e30, -1e30, 1e30, -1e30, 1e30, -1e30]
all = [1e30, -1e30, 1e30, -1e30, 1e30, -1e30]


def grow(bb, x):
    if x.size == 0:
        return bb
    lo = x.min(0)
    hi = x.max(0)
    return [min(bb[0], lo[0]), max(bb[1], hi[0]), min(bb[2], lo[1]),
            max(bb[3], hi[1]), min(bb[4], lo[2]), max(bb[5], hi[2])]


for path in sys.argv[2:]:
    base = os.path.splitext(path)[0]
    a = np.fromfile("%s.%s.attr.raw" % (base, tag), np.float32)
    if a.size:
        m = max(m, float(np.abs(a).max()))
    b = np.fromfile(base + ".body.xyz.raw", np.float32).reshape(-1, 3)
    o = np.fromfile("%s.%s.xyz.raw" % (base, tag), np.float32).reshape(-1, 3)
    body = grow(body, b)
    all = grow(grow(all, b), o)
sys.stdout.write(",".join("%.9g" % v for v in [m] + body + all) + "\n")
