import os
import sys
import numpy as np
import amriso

if len(sys.argv) < 3:
    sys.stderr.write("usage: iso.py LEVEL[,LEVEL...] FILE.xdmf2 [FILE.xdmf2 ...]\n")
    sys.exit(1)
levels = [float(s) for s in sys.argv[1].split(",")]
corner = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
                   [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]], np.float32)
i = np.arange(512)
idx = np.stack([i % 8, (i // 8) % 8, i // 64], 1).astype(np.float32)
for path in sys.argv[2:]:
    base = os.path.splitext(path)[0]
    blk = np.fromfile(base + ".blk.raw", np.float32).reshape(-1, 6)
    chi = np.fromfile(base + ".attr.raw", np.float32)
    vort = np.fromfile(base + ".vort.raw", np.float32).reshape(-1, 3)
    origin = blk[:, 2::-1]
    h = blk[:, 3]
    coords = origin[:, None, None, :] + h[:, None, None, None] * (idx[None, :, None, :] + corner[None, None, :, :])
    coords = np.ascontiguousarray(coords.reshape(-1, 8, 3))
    omega = np.sqrt((vort * vort).sum(1))
    x, t, a = amriso.extract3d(coords, chi, omega, 0.5)
    amriso.dump3d(base + ".body", x, t, a)
    sys.stderr.write("iso.py: %s.body ntri=%d\n" % (base, len(t)))
    for lv in levels:
        x, t, a = amriso.extract3d(coords, omega, np.ascontiguousarray(vort[:, 2]), lv)
        amriso.dump3d("%s.omega%g" % (base, lv), x, t, a)
        sys.stderr.write("iso.py: %s.omega%g ntri=%d\n" % (base, lv, len(t)))
