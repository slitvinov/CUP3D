import os
import sys
import numpy as np
import amriso

if len(sys.argv) < 3:
    sys.stderr.write("usage: iso.py LEVEL[,LEVEL...] FILE.xdmf2 [FILE.xdmf2 ...]\n"
                     "       LEVEL is an |omega| value or qVALUE for the Q criterion\n")
    sys.exit(1)
levels = sys.argv[1].split(",")
corner = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
                   [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]], np.float32)
i = np.arange(512)
idx = np.stack([i % 8, (i // 8) % 8, i // 64], 1).astype(np.float32)


def extract(origin, h, scalar, field, level):
    if hasattr(amriso, "extract3d_blocks"):
        return amriso.extract3d_blocks(origin, h, 8, scalar, field, level)
    coords = origin[:, None, None, :] + h[:, None, None, None] * (idx[None, :, None, :] + corner[None, None, :, :])
    coords = np.ascontiguousarray(coords.reshape(-1, 8, 3))
    return amriso.extract3d(coords, scalar, field, level)


for path in sys.argv[2:]:
    base = os.path.splitext(path)[0]
    blk = np.fromfile(base + ".blk.raw", np.float32).reshape(-1, 6)
    chi = np.fromfile(base + ".attr.raw", np.float32)
    vort = np.fromfile(base + ".vort.raw", np.float32).reshape(-1, 3)
    origin = np.ascontiguousarray(blk[:, 2::-1])
    h = np.ascontiguousarray(blk[:, 3])
    omega = np.sqrt((vort * vort).sum(1))
    wz = np.ascontiguousarray(vort[:, 2])
    x, t, a = extract(origin, h, chi, omega, 0.5)
    amriso.dump3d(base + ".body", x, t, a)
    sys.stderr.write("iso.py: %s.body ntri=%d\n" % (base, len(t)))
    for spec in levels:
        if spec.startswith("q"):
            lv = float(spec[1:])
            if not os.path.exists(base + ".q.raw"):
                sys.exit("iso.py: %s.q.raw not found" % base)
            f = np.fromfile(base + ".q.raw", np.float32)
            name = "%s.q%g" % (base, lv)
        else:
            lv = float(spec)
            f = omega
            name = "%s.omega%g" % (base, lv)
        x, t, a = extract(origin, h, f, wz, lv)
        amriso.dump3d(name, x, t, a)
        sys.stderr.write("iso.py: %s ntri=%d\n" % (name, len(t)))
