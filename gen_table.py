#!/usr/bin/env python3
"""
Generate ghost-cell fill tables for main.c.

One file per stencil configuration (ss = ghost width, te = tensorial):
    lab_ss<ss>_t<te>.bin

Each op is 14 int32: type, bd, dst, bs, src, n, a[8].
Buffers: 0 = fine cache, 1 = coarse buffer, 2+k = neighbor block slot k.
Op types:
    COPY    n elements, contiguous in x in both source and destination
    AVG8    dst = 0.125 * (sum of 8 sources a[0..7], in that order)
    INTERP  27-cell coarse cluster (base at src) -> up to 8 fine cells a[r]
    FD      one-sided 5-point face interpolation with blend, see main.c
    BC      dst = src, normal component negated for vector labs (a[0] = dir)

Sections (start, count pairs into the op array), 27 slots indexed by icode:
    same_copy[27], same_cfill[27], fine[27], coarse[27][8],
    interp[27], own_avg, bc[6][2]
relevant[27][64]: neighbor codes whose coarseness forces the coarse version
of the same-level neighbor at icode, indexed by wall state (4 per dim).
"""
import struct
import sys

BS = 8
CBS = BS // 2
IS, IE = -1, 2
COPY, AVG8, INTERP, FD, BC = range(5)
MAGIC = 0x4C414231


def cdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def code_of(icode):
    return (icode % 3 - 1, (icode // 3) % 3 - 1, (icode // 9) % 3 - 1)


class Gen:
    def __init__(self, ss, te):
        self.ss, self.te = ss, te
        self.S, self.E = -ss, ss + 1
        self.cn = BS + self.E - self.S - 1
        self.offset = cdiv(self.S - 1, 2) + IS
        self.ec = cdiv(self.E, 2) + 1 + IE - 1
        self.cc = CBS + self.ec - self.offset - 1
        self.use_averages = te or self.S < -2 or self.E > 3
        self.ops = []

    def lab(self, x, y, z):
        S = self.S
        return ((z - S) * self.cn + (y - S)) * self.cn + (x - S)

    def coa(self, x, y, z):
        o = self.offset
        return ((z - o) * self.cc + (y - o)) * self.cc + (x - o)

    @staticmethod
    def blk(x, y, z):
        return (z * BS + y) * BS + x

    def op(self, type, bd, dst, bs, src, n=1, a=()):
        a = list(a) + [0] * (8 - len(a))
        self.ops.append((type, bd, dst, bs, src, n, *a))

    def section(self, fn, *args):
        start = len(self.ops)
        fn(*args)
        return (start, len(self.ops) - start)

    def region(self, code):
        S, E = self.S, self.E
        s = [S if c < 0 else (0 if c == 0 else BS) for c in code]
        e = [0 if c < 0 else (BS if c == 0 else BS + E - 1) for c in code]
        return s, e

    def skip_code(self, code):
        return not self.te and not self.use_averages and sum(map(abs, code)) > 1

    def same_copy(self, code):
        if self.skip_code(code):
            return
        s, e = self.region(code)
        for iz in range(s[2], e[2]):
            for iy in range(s[1], e[1]):
                self.op(COPY, 0, self.lab(s[0], iy, iz), 2,
                        self.blk(s[0] - code[0] * BS, iy - code[1] * BS, iz - code[2] * BS),
                        e[0] - s[0])

    def fine(self, code):
        if self.skip_code(code):
            return
        s, e = self.region(code)
        S = self.S
        c0, c1, c2 = code
        nx_len = abs(c0) * (e[0] - s[0]) + (1 - abs(c0)) * cdiv(e[0] - s[0], 2)
        ystep = 2 if c1 == 0 else 1
        zstep = 2 if c2 == 0 else 1
        tot = abs(c0) + abs(c1) + abs(c2)
        bstep = 3 if tot == 2 else (4 if tot == 3 else 1)
        for B in range(0, 4, bstep):
            aux = B % 2 if abs(c0) == 1 else B // 2
            my_ix = abs(c0) * (s[0] - S) + (1 - abs(c0)) * (s[0] - S + (B % 2) * cdiv(e[0] - s[0], 2))
            XX = s[0] - c0 * BS + min(c0, 0) * (e[0] - s[0])
            for iz in range(s[2], e[2], zstep):
                ZZ = 2 * (iz - c2 * BS) + min(c2, 0) * BS if abs(c2) == 1 else iz
                my_iz = abs(c2) * (iz - S) + (1 - abs(c2)) * (cdiv(iz, 2) - S + (B // 2) * cdiv(e[2] - s[2], 2))
                for iy in range(s[1], e[1], ystep):
                    my_iy = abs(c1) * (iy - S) + (1 - abs(c1)) * (cdiv(iy, 2) - S + aux * cdiv(e[1] - s[1], 2))
                    YY = 2 * (iy - c1 * BS) + min(c1, 0) * BS if abs(c1) == 1 else iy
                    for ee in range(nx_len):
                        X = XX + 2 * ee
                        dst = (my_iz * self.cn + my_iy) * self.cn + my_ix + ee
                        self.op(AVG8, 0, dst, 2 + B, 0, 1,
                                [self.blk(X, YY, ZZ), self.blk(X, YY, ZZ + 1),
                                 self.blk(X, YY + 1, ZZ), self.blk(X, YY + 1, ZZ + 1),
                                 self.blk(X + 1, YY, ZZ), self.blk(X + 1, YY, ZZ + 1),
                                 self.blk(X + 1, YY + 1, ZZ), self.blk(X + 1, YY + 1, ZZ + 1)])

    def coarse(self, code, par):
        o, E = self.offset, self.E
        s = [o if c < 0 else (0 if c == 0 else CBS) for c in code]
        e = [0 if c < 0 else (CBS if c == 0 else CBS + cdiv(E, 2) + IE - 1) for c in code]
        if e[0] - s[0] <= 0:
            return
        base = [(par[d] + code[d]) % 2 for d in range(3)]
        edge = [0 if code[d] == 0 else
                (1 if (par[d] == 0 and code[d] > 0) or (par[d] == 1 and code[d] < 0) else 0)
                for d in range(3)]
        start = [max(code[d], 0) * BS // 2 + (1 - abs(code[d])) * base[d] * BS // 2 -
                 code[d] * BS + edge[d] * code[d] * BS // 2 for d in range(3)]
        for iz in range(s[2], e[2]):
            for iy in range(s[1], e[1]):
                self.op(COPY, 1, self.coa(s[0], iy, iz), 2,
                        self.blk(s[0] + start[0], iy + start[1], iz + start[2]), e[0] - s[0])

    def same_cfill(self, code):
        o = self.offset
        eC = cdiv(self.E, 2) + IE
        s = [o if c < 0 else (0 if c == 0 else CBS) for c in code]
        e = [0 if c < 0 else (CBS if c == 0 else CBS + eC - 1) for c in code]
        if e[0] - s[0] <= 0:
            return
        start = [s[d] + max(code[d], 0) * CBS - code[d] * BS + min(code[d], 0) * (e[d] - s[d])
                 for d in range(3)]
        XX = start[0]
        for iz in range(s[2], e[2]):
            ZZ = 2 * (iz - s[2]) + start[2]
            for iy in range(s[1], e[1]):
                if (code[1] == 0 and code[2] == 0 and iy > -IS and iy < CBS - IE and
                        iz > -IS and iz < CBS - IE):
                    continue
                YY = 2 * (iy - s[1]) + start[1]
                for ee in range(e[0] - s[0]):
                    X = XX + 2 * ee
                    self.op(AVG8, 1, self.coa(s[0] + ee, iy, iz), 2, 0, 1,
                            [self.blk(X, YY, ZZ), self.blk(X, YY, ZZ + 1),
                             self.blk(X, YY + 1, ZZ), self.blk(X, YY + 1, ZZ + 1),
                             self.blk(X + 1, YY, ZZ), self.blk(X + 1, YY, ZZ + 1),
                             self.blk(X + 1, YY + 1, ZZ), self.blk(X + 1, YY + 1, ZZ + 1)])

    def relevant(self, code, wall):
        if not self.use_averages:
            return 0
        lo, hi = [], []
        for d in range(3):
            lo.append(0 if code[d] > 0 else -1)
            hi.append(0 if code[d] < 0 else 1)
            if wall[d] & 1 and code[d] == 0:
                lo[d] = 0
            if wall[d] & 2 and code[d] == 0:
                hi[d] = 0
        mask = 0
        for i2 in range(lo[2], hi[2] + 1):
            for i1 in range(lo[1], hi[1] + 1):
                for i0 in range(lo[0], hi[0] + 1):
                    mask |= 1 << ((i0 + 1) + 3 * (i1 + 1) + 9 * (i2 + 1))
        return mask

    def own_avg(self):
        for kk in range(CBS):
            for j in range(CBS):
                for i in range(CBS):
                    if i > -IS and i < CBS - IE and j > -IS and j < CBS - IE and kk > -IS and kk < CBS - IE:
                        continue
                    x, y, z = 2 * i, 2 * j, 2 * kk
                    self.op(AVG8, 1, self.coa(i, j, kk), 0, 0, 1,
                            [self.lab(x, y, z), self.lab(x + 1, y, z), self.lab(x, y + 1, z),
                             self.lab(x + 1, y + 1, z), self.lab(x, y, z + 1), self.lab(x + 1, y, z + 1),
                             self.lab(x, y + 1, z + 1), self.lab(x + 1, y + 1, z + 1)])

    def bc(self, dir, side, coarse):
        if coarse:
            beg = [self.offset] * 3
            end = [self.ec] * 3
            bsize = [CBS] * 3
            idx = self.coa
        else:
            beg = [self.S] * 3
            end = [self.E] * 3
            bsize = [BS] * 3
            idx = self.lab
        s = [0, 0, 0]
        e = list(bsize)
        s[dir] = beg[dir] if side == 0 else bsize[dir]
        e[dir] = 0 if side == 0 else bsize[dir] + end[dir] - 1
        for iz in range(s[2], e[2]):
            for iy in range(s[1], e[1]):
                for ix in range(s[0], e[0]):
                    src = [ix, iy, iz]
                    src[dir] = 0 if side == 0 else bsize[dir] - 1
                    self.op(BC, 1 if coarse else 0, idx(ix, iy, iz), 1 if coarse else 0, idx(*src), 1, [dir])
        s[dir] = beg[dir] * (1 - side) + bsize[dir] * side
        e[dir] = (bsize[dir] - 1 + end[dir]) * side
        d1 = (dir + 1) % 3
        d2 = (dir + 2) % 3
        for b in range(2):
            for a in range(2):
                s[d1] = beg[d1] + a * b * (bsize[d1] - beg[d1])
                s[d2] = beg[d2] + (a - a * b) * (bsize[d2] - beg[d2])
                e[d1] = (1 - b + a * b) * (bsize[d1] - 1 + end[d1])
                e[d2] = (a + b - a * b) * (bsize[d2] - 1 + end[d2])
                for iz in range(s[2], e[2]):
                    for iy in range(s[1], e[1]):
                        for ix in range(s[0], e[0]):
                            src = [ix, iy, iz]
                            src[dir] = side * (bsize[dir] - 1)
                            self.op(BC, 1 if coarse else 0, idx(ix, iy, iz), 1 if coarse else 0, idx(*src), 1, [dir])

    def interp(self, code):
        if self.skip_code(code):
            return
        S, E, o = self.S, self.E, self.offset
        s, e = self.region(code)
        sC = [cdiv(S - 1, 2) if c < 0 else (0 if c == 0 else CBS) for c in code]
        if e[0] - s[0] <= 0:
            return
        if self.use_averages:
            for iz in range(s[2], e[2], 2):
                t = iz - s[2] - min(0, code[2]) * ((e[2] - s[2]) % 2)
                ZZ = cdiv(t, 2) + sC[2]
                izp = -1 if abs(iz) % 2 == 1 else 1
                rzp = 1 if izp == 1 else 0
                rz = 0 if izp == 1 else 1
                for iy in range(s[1], e[1], 2):
                    t = iy - s[1] - min(0, code[1]) * ((e[1] - s[1]) % 2)
                    YY = cdiv(t, 2) + sC[1]
                    iyp = -1 if abs(iy) % 2 == 1 else 1
                    ryp = 1 if iyp == 1 else 0
                    ry = 0 if iyp == 1 else 1
                    for ix in range(s[0], e[0], 2):
                        t = ix - s[0] - min(0, code[0]) * ((e[0] - s[0]) % 2)
                        XX = cdiv(t, 2) + sC[0]
                        ixp = -1 if abs(ix) % 2 == 1 else 1
                        rxp = 1 if ixp == 1 else 0
                        rx = 0 if ixp == 1 else 1
                        dst = [-1] * 8
                        for (x, y, z, r) in [(ix, iy, iz, rx + 2 * ry + 4 * rz),
                                             (ix + ixp, iy, iz, rxp + 2 * ry + 4 * rz),
                                             (ix, iy + iyp, iz, rx + 2 * ryp + 4 * rz),
                                             (ix + ixp, iy + iyp, iz, rxp + 2 * ryp + 4 * rz),
                                             (ix, iy, iz + izp, rx + 2 * ry + 4 * rzp),
                                             (ix + ixp, iy, iz + izp, rxp + 2 * ry + 4 * rzp),
                                             (ix, iy + iyp, iz + izp, rx + 2 * ryp + 4 * rzp),
                                             (ix + ixp, iy + iyp, iz + izp, rxp + 2 * ryp + 4 * rzp)]:
                            if s[0] <= x < e[0] and s[1] <= y < e[1] and s[2] <= z < e[2]:
                                dst[r] = self.lab(x, y, z)
                        self.op(INTERP, 0, 0, 1, self.coa(XX - 1, YY - 1, ZZ - 1), 1, dst)
        if sum(map(abs, code)) == 1:
            coef = [min(0, code[d]) * ((e[d] - s[d]) % 2) for d in range(3)]
            lo = [max(s[d], -2) for d in range(3)]
            hi = [min(e[d], BS + 2) for d in range(3)]
            dir = [d for d in range(3) if code[d] != 0][0]
            tang = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[dir]
            ccc = sum(code)
            for iz in range(lo[2], hi[2]):
                for iy in range(lo[1], hi[1]):
                    for ix in range(lo[0], hi[0]):
                        i = [ix, iy, iz]
                        C = [cdiv(i[d] - s[d] - coef[d], 2) + sC[d] - o for d in range(3)]
                        p = [abs(i[d] - s[d] - coef[d]) % 2 for d in range(3)]
                        kind = []
                        for d in tang:
                            inner = (C[d] + o != 0) and (C[d] + o != CBS - 1)
                            kind.append(0 if inner else (1 if C[d] + o == 0 else 2))
                        xyz = sum(abs(code[d]) * p[d] for d in range(3))
                        bb = self.lab(*[i[d] + cdiv(-3 * code[d] + 1, 2) - p[d] * abs(code[d]) for d in range(3)])
                        cc = self.lab(*[i[d] + cdiv(-5 * code[d] + 1, 2) - p[d] * abs(code[d]) for d in range(3)])
                        src = (C[2] * self.cc + C[1]) * self.cc + C[0]
                        self.op(FD, 0, self.lab(ix, iy, iz), 1, src, 1,
                                [dir, kind[0], kind[1], p[0] | (p[1] << 1) | (p[2] << 2), bb, cc, ccc, xyz])

    def build(self, fname):
        sec = {}
        sec["same_copy"] = [self.section(self.same_copy, code_of(ic)) if ic != 13 else (0, 0) for ic in range(27)]
        sec["same_cfill"] = [self.section(self.same_cfill, code_of(ic)) if ic != 13 and self.use_averages else (0, 0)
                             for ic in range(27)]
        sec["fine"] = [self.section(self.fine, code_of(ic)) if ic != 13 else (0, 0) for ic in range(27)]
        sec["coarse"] = [[self.section(self.coarse, code_of(ic), (par & 1, (par >> 1) & 1, par >> 2))
                          if ic != 13 else (0, 0) for par in range(8)] for ic in range(27)]
        sec["interp"] = [self.section(self.interp, code_of(ic)) if ic != 13 else (0, 0) for ic in range(27)]
        sec["own_avg"] = self.section(self.own_avg)
        sec["bc"] = [[self.section(self.bc, f // 2, f % 2, coarse) for coarse in range(2)] for f in range(6)]
        rel = [[self.relevant(code_of(ic), (w & 3, (w >> 2) & 3, w >> 4)) if ic != 13 else 0
                for w in range(64)] for ic in range(27)]
        with open(fname, "wb") as f:
            f.write(struct.pack("<4i", MAGIC, self.ss, self.te, len(self.ops)))
            for ic in range(27):
                f.write(struct.pack("<2i", *sec["same_copy"][ic]))
            for ic in range(27):
                f.write(struct.pack("<2i", *sec["same_cfill"][ic]))
            for ic in range(27):
                f.write(struct.pack("<2i", *sec["fine"][ic]))
            for ic in range(27):
                for par in range(8):
                    f.write(struct.pack("<2i", *sec["coarse"][ic][par]))
            for ic in range(27):
                f.write(struct.pack("<2i", *sec["interp"][ic]))
            f.write(struct.pack("<2i", *sec["own_avg"]))
            for fc in range(6):
                for coarse in range(2):
                    f.write(struct.pack("<2i", *sec["bc"][fc][coarse]))
            for ic in range(27):
                for w in range(64):
                    f.write(struct.pack("<i", rel[ic][w]))
            for o in self.ops:
                f.write(struct.pack("<14i", *o))
        print(f"{fname}: {len(self.ops)} ops, {len(self.ops) * 56 // 1024} KB")


if __name__ == "__main__":
    for ss, te in [(1, 1), (1, 0), (2, 1), (3, 0)]:
        Gen(ss, te).build(f"lab_ss{ss}_t{te}.bin")
