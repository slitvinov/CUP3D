#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef double Real;
#define MPI_Real MPI_DOUBLE
enum { BS = 8 };
enum { P_INT, P_REAL, P_BOOL, P_STR, P_STRLEN = 32 };
struct Param {
  char *name;
  size_t off;
  int type;
  void *base;
};
struct Midline {
  Real length, period, phase, h, wavelen, amp;
  Real frac_ref, frac_mid, ds_mid_tgt, ds_ref_tgt, ds_mid, ds_ref;
  int nmid, nend, nm;
  Real *rs;
  Real (*r)[3], (*v)[3], (*nor)[3], (*vnor)[3], (*bin)[3], (*vbin)[3];
  Real *width, *height;
  Real *rk, *vk, *rc, *vc, *rt, *vt;
  Real quat_int[4], omega_int[3];
  Real time0, timeshift;
  Real sched_p0[6], sched_p1[6], sched_dp0[6], sched_t0, sched_t1;
  Real alpha, dalpha, beta, dbeta, gamma, dgamma;
};
struct Fish {
  Real length;
  Real position[3], abs_pos[3], quaternion[4], vel[3], omega[3];
  Real vel_imposed[3];
  int fix[3], forced[3], block_rot[3];
  int correct_pos, correct_z, correct_roll, planar;
  Real angle, orig[3], wyp, wzp;
  char hprof[P_STRLEN], wprof[P_STRLEN];
  Real old_position[3], old_abs_pos[3], old_quaternion[4];
  Real com[3], mass, J[6], vel_corr[3], omega_corr[3];
  Real pen_m, pen_cm[3], pen_j[6], pen_lmom[3], pen_amom[3];
  Real hit_time, hit_vel[3], hit_omega[3];
  Real (*r_axis)[4];
  int nr_axis;
  struct Midline m;
  struct ObstacleBlock *pool;
  int *slot;
};
enum { F_CHI = 0, F_PRES = 1, F_VEL = 2, F_TMP = 5, F_LHS = 8, F_N = 9 };
enum { BS3 = BS * BS * BS, BLK_S = F_N * BS3 };
enum { SS_MAX = 3, NC_MAX = 6, CN_MAX = BS + 2 * SS_MAX, CC_MAX = BS / 2 + 6 };
#define IDX(X, Y, Z) (((Z) * BS + (Y)) * BS + (X))
struct Blk {
  int level, ix, iy, iz;
  long long Z;
  Real h, origin[3];
};
enum {
  M_V,
  M_FX,
  M_FY,
  M_FZ,
  M_TX,
  M_TY,
  M_TZ,
  M_J0,
  M_J1,
  M_J2,
  M_J3,
  M_J4,
  M_J5,
  M_GfX,
  M_GpX,
  M_GpY,
  M_GpZ,
  M_Gj0,
  M_Gj1,
  M_Gj2,
  M_Gj3,
  M_Gj4,
  M_Gj5,
  M_GuX,
  M_GuY,
  M_GuZ,
  M_GaX,
  M_GaY,
  M_GaZ,
  M_N
};
struct ObstacleBlock {
  Real chi[BS][BS][BS];
  Real udef[BS][BS][BS][3];
  Real sdf[BS + 2][BS + 2][BS + 2];
  Real com[3], mass;
  Real mom[M_N];
};
struct Segment {
  Real safe_distance;
  int s0, s1;
  Real ni[3], nj[3], nk[3];
  Real w[3], c[3];
  Real box_lab[3][2], box_obj[3][2];
};
static struct Sim {
  MPI_Comm comm;
  int rank, size;
  int bpdx, bpdy, bpdz, level_max, level_start;
  Real extents[3], extent_max, hmin, h0;
  Real rtol, ctol, cfl, nu, tend, tdump;
  Real umax, dlm, ptol, ptol_rel;
  int static_obst;
  int rampup, nsteps, mean_constraint;
  int nfish;
} sim;
static struct Sta {
  Real lambda, dt, dt_old, time, next_dump, umax;
  Real uinf[3], coef_u[3];
  int step, mesh_changed;
  long long nblk;
  struct Blk *blk;
  Real *fld;
  struct Fish *fish;
} sta;
enum { STEP_2ND = 2 };
#define BLK(i) (sta.fld + (long long)(i) * BLK_S)
static Real *fld(long long i, int f) { return BLK(i) + f * BS3; }
static void fatal(char *fmt, ...) __attribute__((noreturn));
static void fatal(char *fmt, ...) {
  va_list ap;
  fprintf(stderr, "main.c: rank %d: ", sim.rank);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
  abort();
}
static void *emalloc(size_t n) {
  void *p = malloc(n > 0 ? n : 1);
  if (p == NULL)
    fatal("out of memory (%zu bytes)", n);
  return p;
}
static void *ecalloc(size_t n, size_t m) {
  void *p = calloc(n > 0 ? n : 1, m);
  if (p == NULL)
    fatal("out of memory (%zu x %zu bytes)", n, m);
  return p;
}
static void *erealloc(void *q, size_t n) {
  void *p = realloc(q, n > 0 ? n : 1);
  if (p == NULL)
    fatal("out of memory (%zu bytes)", n);
  return p;
}
#define SIMP(name, field, type) {name, offsetof(struct Sim, field), type, &sim}
#define STAP(name, field, type) {name, offsetof(struct Sta, field), type, &sta}
#define FISHP(name, field, type) {name, offsetof(struct Fish, field), type, NULL}
static struct Param sim_params[] = {
    SIMP("bpdx", bpdx, P_INT),
    SIMP("bpdy", bpdy, P_INT),
    SIMP("bpdz", bpdz, P_INT),
    SIMP("levelMax", level_max, P_INT),
    SIMP("levelStart", level_start, P_INT),
    SIMP("extent", extent_max, P_REAL),
    SIMP("Rtol", rtol, P_REAL),
    SIMP("Ctol", ctol, P_REAL),
    SIMP("CFL", cfl, P_REAL),
    SIMP("nu", nu, P_REAL),
    SIMP("tend", tend, P_REAL),
    SIMP("tdump", tdump, P_REAL),
    SIMP("umax", umax, P_REAL),
    SIMP("use-dlm", dlm, P_REAL),
    SIMP("poissonTol", ptol, P_REAL),
    SIMP("poissonTolRel", ptol_rel, P_REAL),
    SIMP("rampup", rampup, P_INT),
    SIMP("nsteps", nsteps, P_INT),
    SIMP("bMeanConstraint", mean_constraint, P_INT),
    SIMP("StaticObstacles", static_obst, P_BOOL),
    STAP("dt", dt, P_REAL),
    STAP("lambda", lambda, P_REAL),
    STAP("uinfx", uinf[0], P_REAL),
    STAP("uinfy", uinf[1], P_REAL),
    STAP("uinfz", uinf[2], P_REAL),
};
static struct Param fish_params[] = {
    FISHP("L", length, P_REAL),
    FISHP("xpos", position[0], P_REAL),
    FISHP("ypos", position[1], P_REAL),
    FISHP("zpos", position[2], P_REAL),
    FISHP("planarAngle", angle, P_REAL),
    FISHP("T", m.period, P_REAL),
    FISHP("phi", m.phase, P_REAL),
    FISHP("amplitudeFactor", m.amp, P_REAL),
    FISHP("heightProfile", hprof, P_STR),
    FISHP("widthProfile", wprof, P_STR),
    FISHP("bFixFrameOfRef_x", fix[0], P_BOOL),
    FISHP("bFixFrameOfRef_y", fix[1], P_BOOL),
    FISHP("bFixFrameOfRef_z", fix[2], P_BOOL),
    FISHP("bForcedInSimFrame_x", forced[0], P_BOOL),
    FISHP("bForcedInSimFrame_y", forced[1], P_BOOL),
    FISHP("bForcedInSimFrame_z", forced[2], P_BOOL),
    FISHP("xvel", vel_imposed[0], P_REAL),
    FISHP("yvel", vel_imposed[1], P_REAL),
    FISHP("zvel", vel_imposed[2], P_REAL),
    FISHP("bFixToPlanar", planar, P_BOOL),
    FISHP("CorrectPosition", correct_pos, P_BOOL),
    FISHP("CorrectPositionZ", correct_z, P_BOOL),
    FISHP("CorrectRoll", correct_roll, P_BOOL),
    FISHP("wyp", wyp, P_REAL),
    FISHP("wzp", wzp, P_REAL),
};
enum { N_SIMP = sizeof sim_params / sizeof *sim_params, N_FISHP = sizeof fish_params / sizeof *fish_params };
static void param_fail(char *key, char *val, char *why) __attribute__((noreturn));
static void param_fail(char *key, char *val, char *why) { fatal("parameter '%s' = '%s': %s", key, val, why); }
static void param_store(struct Param *p, void *base, char *key, char *val) {
  char *end;
  void *dst = (char *)(base ? base : p->base) + p->off;
  long iv;
  double rv;
  errno = 0;
  switch (p->type) {
  case P_INT:
    iv = strtol(val, &end, 10);
    if (end == val || *end != '\0' || errno != 0 || iv < INT_MIN || iv > INT_MAX)
      param_fail(key, val, "not an integer");
    *(int *)dst = (int)iv;
    break;
  case P_REAL:
    rv = strtod(val, &end);
    if (end == val || *end != '\0' || errno != 0 || !isfinite(rv))
      param_fail(key, val, "not a finite number");
    *(Real *)dst = rv;
    break;
  case P_BOOL:
    if (strcmp(val, "0") == 0 || strcmp(val, "false") == 0)
      *(int *)dst = 0;
    else if (strcmp(val, "1") == 0 || strcmp(val, "true") == 0)
      *(int *)dst = 1;
    else
      param_fail(key, val, "not a boolean (0, 1, true, false)");
    break;
  case P_STR:
    if (strlen(val) >= P_STRLEN)
      param_fail(key, val, "string too long");
    strcpy((char *)dst, val);
    break;
  }
}
static void param_apply(struct Param *tab, int n, void *base, char *key, char *val, int *seen) {
  int i;
  for (i = 0; i < n; i++)
    if (strcmp(tab[i].name, key) == 0) {
      if (seen[i])
        param_fail(key, val, "given twice");
      seen[i] = 1;
      param_store(&tab[i], base, key, val);
      return;
    }
  param_fail(key, val, "unknown parameter");
}
static void param_check(struct Param *tab, int n, int *seen, char *what) {
  int i, missing = 0;
  for (i = 0; i < n; i++)
    if (!seen[i]) {
      fprintf(stderr, "main.c: %s: missing parameter '%s'\n", what, tab[i].name);
      missing = 1;
    }
  if (missing)
    fatal("%s: missing parameters", what);
}
static Real *ralloc(int n) { return emalloc(n * sizeof(Real)); }

static Real mid_dds(struct Midline *m, int idx, Real (*vals)[3], int c, int maxidx) {
  Real *rs = m->rs;
  if (idx == 0)
    return (vals[idx + 1][c] - vals[idx][c]) / (rs[idx + 1] - rs[idx]);
  else if (idx == maxidx - 1)
    return (vals[idx][c] - vals[idx - 1][c]) / (rs[idx] - rs[idx - 1]);
  else
    return 0.5 * ((vals[idx + 1][c] - vals[idx][c]) / (rs[idx + 1] - rs[idx]) +
                  (vals[idx][c] - vals[idx - 1][c]) / (rs[idx] - rs[idx - 1]));
}
static void natural_cubic_spline(Real *x, Real *y, unsigned n, Real *xx, Real *yy, unsigned nn) {
  Real *y2 = ralloc(n);
  Real *u = ralloc(n - 1);
  unsigned i;
  unsigned k;
  unsigned j;
  y2[0] = u[0] = 0.0;
  for (i = 1; i < n - 1; i++) {
    Real sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
    Real p = sig * y2[i - 1] + 2.0;
    y2[i] = (sig - 1.0) / p;
    u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
  }
  y2[n - 1] = 0;
  for (k = n - 2; k > 0; k--)
    y2[k] = y2[k] * y2[k + 1] + u[k];
  for (j = 0; j < nn; j++) {
    unsigned klo = 0;
    unsigned khi = n - 1;
    unsigned k = 0;
    Real h;
    Real a;
    Real b;
    while (khi - klo > 1) {
      k = (khi + klo) >> 1;
      if (x[k] > xx[j])
        khi = k;
      else
        klo = k;
    }
    h = x[khi] - x[klo];
    if (fabs(h) < 2.2e-16)
      fatal("interpolation points must be distinct");
    a = (x[khi] - xx[j]) / h;
    b = (xx[j] - x[klo]) / h;
    yy[j] = a * y[klo] + b * y[khi] + ((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * (h * h) / 6;
  }
  free(y2);
  free(u);
}
static void cubic_interpolation(Real x0, Real x1, Real x, Real y0, Real y1, Real dy0, Real dy1, Real *y,
                                Real *dy) {
  Real xrel = (x - x0);
  Real deltax = (x1 - x0);
  Real a = (dy0 + dy1) / (deltax * deltax) - 2 * (y1 - y0) / (deltax * deltax * deltax);
  Real b = (-2 * dy0 - dy1) / deltax + 3 * (y1 - y0) / (deltax * deltax);
  Real c = dy0;
  Real d = y0;
  *y = a * xrel * xrel * xrel + b * xrel * xrel + c * xrel + d;
  *dy = 3 * a * xrel * xrel + 2 * b * xrel + c;
}
static void sched_set(struct Midline *m, Real t, Real tstart, Real tend, Real p0[6], Real p1[6]) {
  int i;
  if (t < tstart || t > tend)
    return;
  if (tstart < m->sched_t0)
    return;
  m->sched_t0 = tstart;
  m->sched_t1 = tend;
  for (i = 0; i < 6; i++) {
    m->sched_p0[i] = p0[i];
    m->sched_p1[i] = p1[i];
  }
}
static void sched_get(struct Midline *m, Real t, Real positions[6], int Nfine, Real *positions_fine,
                      Real *parameters_fine, Real *dparameters_fine) {
  Real *p0f = ralloc(Nfine);
  Real *p1f = ralloc(Nfine);
  Real *dp0f = ralloc(Nfine);
  natural_cubic_spline(positions, m->sched_p0, 6, positions_fine, p0f, Nfine);
  natural_cubic_spline(positions, m->sched_p1, 6, positions_fine, p1f, Nfine);
  natural_cubic_spline(positions, m->sched_dp0, 6, positions_fine, dp0f, Nfine);
  if (t < m->sched_t0 || m->sched_t0 < 0) {
    int i;
    for (i = 0; i < Nfine; ++i) {
      parameters_fine[i] = p0f[i];
      dparameters_fine[i] = 0.0;
    }
  } else if (t > m->sched_t1) {
    int i;
    for (i = 0; i < Nfine; ++i) {
      parameters_fine[i] = p1f[i];
      dparameters_fine[i] = 0.0;
    }
  } else {
    int i;
    for (i = 0; i < Nfine; ++i)
      cubic_interpolation(m->sched_t0, m->sched_t1, t, p0f[i], p1f[i], dp0f[i], 0.0, &parameters_fine[i],
                          &dparameters_fine[i]);
  }
  free(p0f);
  free(p1f);
  free(dp0f);
}
static void mid_init(struct Midline *m, Real L, Real h) {
  int nm;
  Real *buf;
  Real *rs;
  int nend, nmid;
  Real ds_ref, ds_mid;
  int i, k;
  m->length = L;
  m->h = h;
  m->wavelen = 1;
  m->frac_ref = 0.1;
  m->frac_mid = 1 - 2 * m->frac_ref;
  m->ds_mid_tgt = h / sqrt(3);
  m->ds_ref_tgt = 0.125 * h;
  m->nmid = (int)ceil(L * m->frac_mid / m->ds_mid_tgt / 8) * 8;
  m->ds_mid = L * m->frac_mid / m->nmid;
  m->nend = (int)ceil(m->frac_ref * L * 2 / (m->ds_mid + m->ds_ref_tgt) / 4) * 4;
  m->ds_ref = m->frac_ref * L * 2 / m->nend - m->ds_mid;
  while (m->ds_ref < 0 && m->nend > 4) {
    m->nend -= 4;
    m->ds_ref = m->frac_ref * L * 2 / m->nend - m->ds_mid;
  }
  m->nm = m->nmid + 2 * m->nend + 1;
  nm = m->nm;
  buf = ralloc(27 * nm);
  m->rs = buf;
  m->width = buf + 1 * nm;
  m->height = buf + 2 * nm;
  m->rk = buf + 3 * nm;
  m->vk = buf + 4 * nm;
  m->rc = buf + 5 * nm;
  m->vc = buf + 6 * nm;
  m->rt = buf + 7 * nm;
  m->vt = buf + 8 * nm;
  m->r = (Real(*)[3])(buf + 9 * nm);
  m->v = (Real(*)[3])(buf + 12 * nm);
  m->nor = (Real(*)[3])(buf + 15 * nm);
  m->vnor = (Real(*)[3])(buf + 18 * nm);
  m->bin = (Real(*)[3])(buf + 21 * nm);
  m->vbin = (Real(*)[3])(buf + 24 * nm);
  rs = m->rs;
  nend = m->nend;
  nmid = m->nmid;
  ds_ref = m->ds_ref;
  ds_mid = m->ds_mid;
  rs[0] = 0;
  k = 0;
  for (i = 0; i < nend; ++i, k++)
    rs[k + 1] = rs[k] + ds_ref + (ds_mid - ds_ref) * i / ((Real)nend - 1.);
  for (i = 0; i < nmid; ++i, k++)
    rs[k + 1] = rs[k] + ds_mid;
  for (i = 0; i < nend; ++i, k++)
    rs[k + 1] = rs[k] + ds_ref + (ds_mid - ds_ref) * (nend - i - 1) / ((Real)nend - 1.);
  rs[k] = (L < rs[k]) ? L : rs[k];
  assert(k + 1 == nm);
  m->quat_int[0] = 1;
  m->quat_int[1] = m->quat_int[2] = m->quat_int[3] = 0;
  m->omega_int[0] = m->omega_int[1] = m->omega_int[2] = 0;
  m->time0 = 0;
  m->timeshift = 0;
  for (i = 0; i < 6; i++)
    m->sched_p0[i] = m->sched_p1[i] = m->sched_dp0[i] = 0;
  m->sched_t0 = -1;
  m->sched_t1 = 0;
  m->alpha = 1;
  m->dalpha = 0;
  m->beta = 0;
  m->dbeta = 0;
  m->gamma = 0;
  m->dgamma = 0;
}
static void bspline_basis(Real x, Real *t, int n, Real *B) {
  enum { K = 4 };
  Real b[K], deltal[K], deltar[K];
  int i, j, left;
  if (x >= t[n + K - 1]) {
    left = n - 1;
  } else {
    int lo = 0, hi = n + K - 1;
    while (hi > lo + 1) {
      int mid = (hi + lo) >> 1;
      if (t[mid] > x)
        hi = mid;
      else
        lo = mid;
    }
    left = lo;
  }
  b[0] = 1;
  for (j = 0; j < K - 1; j++) {
    Real saved;
    deltar[j] = t[left + j + 1] - x;
    deltal[j] = x - t[left - j];
    saved = 0;
    for (i = 0; i <= j; i++) {
      Real term = b[i] / (deltar[i] + deltal[j - i]);
      b[i] = saved + deltar[i] * term;
      saved = deltal[j - i] * term;
    }
    b[j + 1] = saved;
  }
  for (i = 0; i < n; i++)
    B[i] = 0;
  for (i = 0; i < K; i++)
    B[left - K + 1 + i] = b[i];
}
static void integrate_bspline(Real *xc, Real *yc, int n, Real length, Real *rs, Real *res, int nm) {
  enum { K = 4 };
  Real len = 0;
  int i;
  Real *t;
  Real *B;
  Real delta;
  Real ti;
  for (i = 0; i < n - 1; i++) {
    len += sqrt(pow(xc[i] - xc[i + 1], 2) + pow(yc[i] - yc[i + 1], 2));
  }
  t = ralloc(n + K);
  B = ralloc(n);
  delta = len / (n - 3);
  for (i = 0; i < K; i++)
    t[i] = 0;
  for (i = 0; i < n - 4; i++)
    t[K + i] = (i + 1) * delta;
  for (i = n; i < n + K; i++)
    t[i] = len;
  ti = 0;
  for (i = 0; i < nm; ++i) {
    res[i] = 0;
    if (rs[i] > 0 && rs[i] < length) {
      Real dtt = (rs[i] - rs[i - 1]) / 1e3;
      int j;
      for (;;) {
        Real xi = 0;
        bspline_basis(ti, t, n, B);
        for (j = 0; j < n; j++)
          xi += xc[j] * B[j];
        if (xi >= rs[i])
          break;
        if (ti + dtt > len)
          break;
        else
          ti += dtt;
      }
      for (j = 0; j < n; j++)
        res[i] += yc[j] * B[j];
    }
  }
  free(t);
  free(B);
}
static void w_stefan(Real L, Real *rs, Real *res, int nm) {
  Real sb = .04 * L;
  Real st = .95 * L;
  Real wt = .01 * L;
  Real wh = .04 * L;
  int i;
  for (i = 0; i < nm; ++i) {
    if (rs[i] <= 0 || rs[i] >= L)
      res[i] = 0;
    else {
      Real s = rs[i];
      res[i] =
          (s < sb ? sqrt(2.0 * wh * s - s * s)
                  : (s < st ? wh - (wh - wt) * pow((s - sb) / (st - sb), 2) : (wt * (L - s) / (L - st))));
    }
  }
}
static void h_stefan(Real L, Real *rs, Real *res, int nm) {
  Real a = 0.51 * L;
  Real b = 0.08 * L;
  int i;
  for (i = 0; i < nm; ++i) {
    if (rs[i] <= 0 || rs[i] >= L)
      res[i] = 0;
    else {
      Real s = rs[i];
      res[i] = b * sqrt(1 - pow((s - a) / a, 2));
    }
  }
}
static void w_danio(Real L, Real *rs, Real *res, int nm) {
  enum { n_breaks_w = 11 };
  Real bw[n_breaks_w] = {0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0};
  Real coeffs_w[n_breaks_w - 1][4] = {{0.0015713, 2.6439, 0, -15410},
                                      {0.012865, 1.4882, -231.15, 15598},
                                      {0.016476, 0.34647, 2.8156, -39.328},
                                      {0.032323, 0.38294, -1.9038, 0.7411},
                                      {0.046803, 0.19812, -1.7926, 5.4876},
                                      {0.054176, 0.0042136, -0.14638, 0.077447},
                                      {0.049783, -0.045043, -0.099907, -0.12599},
                                      {0.03577, -0.10012, -0.1755, 0.62019},
                                      {0.013687, -0.0959, 0.19662, 0.82341},
                                      {0.0065049, 0.018665, 0.56715, -3.781}};
  int i;
  for (i = 0; i < nm; ++i) {
    if (rs[i] <= 0 || rs[i] >= L)
      res[i] = 0;
    else {
      Real sn = rs[i] / L;
      int iw = 1;
      Real *cw;
      Real xw;
      while (sn >= bw[iw])
        iw++;
      iw--;
      cw = coeffs_w[iw];
      xw = sn - bw[iw];
      res[i] = L * (cw[0] + cw[1] * xw + cw[2] * pow(xw, 2) + cw[3] * pow(xw, 3));
    }
  }
}
static void h_danio(Real L, Real *rs, Real *res, int nm) {
  enum { n_breaks_h = 15 };
  Real bh[n_breaks_h] = {0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.87, 0.9, 0.993, 0.996, 0.998, 1};
  Real coeffs_h[n_breaks_h - 1][4] = {
      {0.0011746, 1.345, 2.2204e-14, -578.62},   {0.014046, 1.1715, -17.359, 128.6},
      {0.041361, 0.40004, -1.9268, 9.7029},      {0.057759, 0.28013, -0.47141, -0.08102},
      {0.094281, 0.081843, -0.52002, -0.76511},  {0.083728, -0.21798, -0.97909, 3.9699},
      {0.032727, -0.13323, 1.4028, 2.5693},      {0.036002, 0.22441, 2.1736, -13.194},
      {0.051007, 0.34282, 0.19446, 16.642},      {0.058075, 0.37057, 1.193, -17.944},
      {0.069781, 0.3937, -0.42196, -29.388},     {0.079107, -0.44731, -8.6211, -1.8283e+05},
      {0.072751, -5.4355, -1654.1, -2.9121e+05}, {0.052934, -15.546, -3401.4, 5.6689e+05}};
  int i;
  for (i = 0; i < nm; ++i) {
    if (rs[i] <= 0 || rs[i] >= L)
      res[i] = 0;
    else {
      Real sn = rs[i] / L;
      int ih = 1;
      Real *ch;
      Real xh;
      while (sn >= bh[ih])
        ih++;
      ih--;
      ch = coeffs_h[ih];
      xh = sn - bh[ih];
      res[i] = L * (ch[0] + ch[1] * xh + ch[2] * pow(xh, 2) + ch[3] * pow(xh, 3));
    }
  }
}
static void h_default(Real L, Real *rs, Real *res, int nm) {
  Real x[8] = {0, 0, .2 * L, .4 * L, .6 * L, .8 * L, L, L};
  Real y[8] = {0, .055 * L, .068 * L, .076 * L, .064 * L, .0072 * L, .11 * L, 0};
  integrate_bspline(x, y, 8, L, rs, res, nm);
}
static void w_default(Real L, Real *rs, Real *res, int nm) {
  Real x[6] = {0, 0, L / 3., 2 * L / 3., L, L};
  Real y[6] = {0, 8.9e-2 * L, 1.7e-2 * L, 1.6e-2 * L, 1.3e-2 * L, 0};
  integrate_bspline(x, y, 6, L, rs, res, nm);
}
struct Profile {
  char *name;
  void (*fn)(Real L, Real *rs, Real *res, int nm);
};
static struct Profile height_profiles[] = {
    {"danio", h_danio},
    {"stefan", h_stefan},
    {"default", h_default},
};
static struct Profile width_profiles[] = {
    {"danio", w_danio},
    {"stefan", w_stefan},
    {"default", w_default},
};
enum {
  NHEIGHT = sizeof height_profiles / sizeof *height_profiles,
  NWIDTH = sizeof width_profiles / sizeof *width_profiles
};
static void profile_set(struct Profile *tab, int n, char *what, char *name, Real L, Real *rs, Real *res,
                        int nm) {
  int i;
  for (i = 0; i < n; i++)
    if (strcmp(tab[i].name, name) == 0) {
      tab[i].fn(L, rs, res, nm);
      return;
    }
  fatal("unknown %s profile '%s'", what, name);
}

static Real dot3(Real a[3], Real b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static void cross3(Real out[3], Real a[3], Real b[3]) {
  int d;
  for (d = 0; d < 3; d++) {
    int e = (d + 1) % 3, f = (d + 2) % 3;
    out[d] = a[e] * b[f] - a[f] * b[e];
  }
}
static void normalize3(Real a[3]) {
  Real d = dot3(a, a);
  if (d > DBL_EPSILON) {
    Real f = 1.0 / sqrt(d);
    int k;
    for (k = 0; k < 3; k++)
      a[k] *= f;
  }
}
static void mid_frenet(struct Midline *m) {
  int nm = m->nm;
  Real *rs = m->rs, *curv = m->rk, *curv_dt = m->vk, *tors = m->rt, *tors_dt = m->vt;
  Real(*r)[3] = m->r, (*v)[3] = m->v, (*nor)[3] = m->nor, (*vnor)[3] = m->vnor, (*bin)[3] = m->bin,
  (*vbin)[3] = m->vbin;
  Real ksi[3] = {1.0, 0.0, 0.0}, v_ksi[3] = {0.0, 0.0, 0.0};
  int d;
  int i;
  for (d = 0; d < 3; d++) {
    r[0][d] = v[0][d] = vnor[0][d] = vbin[0][d] = 0.0;
    nor[0][d] = d == 1;
    bin[0][d] = d == 2;
  }
  for (i = 1; i < nm; i++) {
    Real k = curv[i - 1], kt = curv_dt[i - 1], tau = tors[i - 1], taut = tors_dt[i - 1];
    Real ds = rs[i] - rs[i - 1];
    int d;
    for (d = 0; d < 3; d++) {
      Real dksi = k * nor[i - 1][d];
      Real dnu = -k * ksi[d] + tau * bin[i - 1][d];
      Real dbin = -tau * nor[i - 1][d];
      Real dv_ksi = kt * nor[i - 1][d] + k * vnor[i - 1][d];
      Real dv_nu = -kt * ksi[d] - k * v_ksi[d] + taut * bin[i - 1][d] + tau * vbin[i - 1][d];
      Real dv_bin = -taut * nor[i - 1][d] - tau * vnor[i - 1][d];
      r[i][d] = r[i - 1][d] + ds * ksi[d];
      nor[i][d] = nor[i - 1][d] + ds * dnu;
      ksi[d] += ds * dksi;
      bin[i][d] = bin[i - 1][d] + ds * dbin;
      v[i][d] = v[i - 1][d] + ds * v_ksi[d];
      vnor[i][d] = vnor[i - 1][d] + ds * dv_nu;
      v_ksi[d] += ds * dv_ksi;
      vbin[i][d] = vbin[i - 1][d] + ds * dv_bin;
    }
    normalize3(ksi);
    normalize3(nor[i]);
    normalize3(bin[i]);
  }
}

static void mid_frame(struct Midline *m, int i, Real t[3], Real dt[3]) {
  Real *nor = m->nor[i], *vnor = m->vnor[i], *bin = m->bin[i], *vbin = m->vbin[i];
  Real BD[3] = {nor[0], nor[1], nor[2]}, d_bd[3] = {vnor[0], vnor[1], vnor[2]};
  Real dot = dot3(BD, t);
  Real ddot = dot3(d_bd, t) + BD[0] * dt[0] + BD[1] * dt[1] + BD[2] * dt[2];
  int d;
  Real inormn;
  Real inormb;
  int a;
  for (d = 0; d < 3; d++)
    nor[d] = BD[d] - dot * t[d];
  inormn = 1.0 / sqrt(dot3(nor, nor));
  for (d = 0; d < 3; d++) {
    nor[d] *= inormn;
    vnor[d] = d_bd[d] - ddot * t[d] - dot * dt[d];
  }
  cross3(bin, t, nor);
  inormb = 1.0 / sqrt(dot3(bin, bin));
  for (d = 0; d < 3; d++)
    bin[d] *= inormb;
  for (a = 0; a < 3; a++) {
    int b = (a + 1) % 3, c = (a + 2) % 3;
    vbin[a] = (dt[b] * nor[c] + t[b] * vnor[c]) - (dt[c] * nor[b] + t[c] * vnor[b]);
  }
}
static void mid_frames(struct Midline *m) {
  int nm = m->nm;
  Real *rs = m->rs;
  Real(*r)[3] = m->r, (*v)[3] = m->v;
  int i;
#pragma omp parallel for
  for (i = 1; i < nm - 1; i++) {
    Real hp = rs[i + 1] - rs[i];
    Real hm = rs[i] - rs[i - 1];
    Real frac = hp / hm;
    Real am = -frac * frac;
    Real a = frac * frac - 1.0;
    Real ap = 1.0;
    Real denom = 1.0 / (hp * (1.0 + frac));
    Real t[3], dt[3];
    int d;
    for (d = 0; d < 3; d++) {
      t[d] = (am * r[i - 1][d] + a * r[i][d] + ap * r[i + 1][d]) * denom;
      dt[d] = (am * v[i - 1][d] + a * v[i][d] + ap * v[i + 1][d]) * denom;
    }
    mid_frame(m, i, t, dt);
  }
  for (i = 0; i <= nm - 1; i += nm - 1) {
    int ipm = (i == nm - 1) ? i - 1 : i + 1;
    Real ids = 1.0 / (rs[ipm] - rs[i]);
    Real t[3], dt[3];
    int d;
    for (d = 0; d < 3; d++) {
      t[d] = (r[ipm][d] - r[i][d]) * ids;
      dt[d] = (v[ipm][d] - v[i][d]) * ids;
    }
    mid_frame(m, i, t, dt);
  }
}
static void mid_pitch(struct Midline *m) {
  int nm = m->nm;
  Real(*r)[3] = m->r, (*v)[3] = m->v;
  Real gamma = m->gamma, dgamma = m->dgamma;
  Real R, Rdot;
  Real x0_n;
  Real y0_n;
  Real x0_ndot;
  Real y0_ndot;
  Real phi;
  Real phidot;
  Real M;
  Real Mdot;
  Real cosphi;
  Real sinphi;
  int i;
  if (fabs(gamma) > 1e-10) {
    R = 1.0 / gamma;
    Rdot = -1.0 / gamma / gamma * dgamma;
  } else {
    R = gamma >= 0 ? 1e10 : -1e10;
    Rdot = 0.0;
  }
  x0_n = r[nm - 1][0];
  y0_n = r[nm - 1][1];
  x0_ndot = v[nm - 1][0];
  y0_ndot = v[nm - 1][1];
  phi = atan2(y0_n, x0_n);
  phidot = 1.0 / (1.0 + pow(y0_n / x0_n, 2)) * (y0_ndot / x0_n - y0_n * x0_ndot / x0_n / x0_n);
  M = pow(x0_n * x0_n + y0_n * y0_n, 0.5);
  Mdot = (x0_n * x0_ndot + y0_n * y0_ndot) / M;
  cosphi = cos(phi);
  sinphi = sin(phi);
#pragma omp parallel for
  for (i = 0; i < nm; i++) {
    double x0 = r[i][0];
    double y0 = r[i][1];
    double x0dot = v[i][0];
    double y0dot = v[i][1];
    double x1 = cosphi * x0 - sinphi * y0;
    double y1 = sinphi * x0 + cosphi * y0;
    double x1dot = cosphi * x0dot - sinphi * y0dot + (-sinphi * x0 - cosphi * y0) * phidot;
    double y1dot = sinphi * x0dot + cosphi * y0dot + (cosphi * x0 - sinphi * y0) * phidot;
    double theta = (M - x1) / R;
    double costheta = cos(theta);
    double sintheta = sin(theta);
    double x2 = M - R * sintheta;
    double z2 = R - R * costheta;
    double thetadot = (Mdot - x1dot) / R - (M - x1) / R / R * Rdot;
    double x2dot = Mdot - Rdot * sintheta - R * costheta * thetadot;
    double z2dot = Rdot - Rdot * costheta + R * sintheta * thetadot;
    r[i][0] = x2;
    r[i][1] = y1;
    r[i][2] = z2;
    v[i][0] = x2dot;
    v[i][1] = y1dot;
    v[i][2] = z2dot;
  }
  mid_frames(m);
}
static void mid_step(struct Midline *m, Real t) {
  int nm = m->nm;
  Real length = m->length, period = m->period;
  Real curv_s[6];
  Real curv_k[6];
  Real curv_0[6];
  Real darg;
  Real arg0;
  Real alpha, dalpha, beta, dbeta, amp, wavelen;
  Real *rs, *rc, *vc;
  Real *rk, *vk, *rt, *vt;
  int i;
  if (0 < t && t < 0.1 * period) {
    m->timeshift = (t - m->time0) / period + m->timeshift;
    m->time0 = t;
  }
  curv_s[0] = 0.0;
  curv_s[1] = 0.15 * length;
  curv_s[2] = 0.4 * length;
  curv_s[3] = 0.65 * length;
  curv_s[4] = 0.9 * length;
  curv_s[5] = length;
  curv_k[0] = 0.82014 / length;
  curv_k[1] = 1.46515 / length;
  curv_k[2] = 2.57136 / length;
  curv_k[3] = 3.75425 / length;
  curv_k[4] = 5.09147 / length;
  curv_k[5] = 5.70449 / length;
  curv_0[0] = 0;
  curv_0[1] = 0;
  curv_0[2] = 0;
  curv_0[3] = 0;
  curv_0[4] = 0;
  curv_0[5] = 0;
  sched_set(m, 0, 0, period, curv_0, curv_k);
  sched_get(m, t, curv_s, nm, m->rs, m->rc, m->vc);
  darg = 2 * M_PI / period;
  arg0 = 2 * M_PI * ((t - m->time0) / period + m->timeshift) + M_PI * m->phase;
  alpha = m->alpha;
  dalpha = m->dalpha;
  beta = m->beta;
  dbeta = m->dbeta;
  amp = m->amp;
  wavelen = m->wavelen;
  rs = m->rs;
  rc = m->rc;
  vc = m->vc;
  rk = m->rk;
  vk = m->vk;
  rt = m->rt;
  vt = m->vt;
#pragma omp parallel for
  for (i = 0; i < nm; ++i) {
    Real arg = arg0 - 2 * M_PI * rs[i] / length / wavelen;
    Real curv = sin(arg) + beta;
    Real dcurv = cos(arg) * darg + dbeta;
    rk[i] = alpha * amp * rc[i] * curv;
    vk[i] = alpha * amp * (vc[i] * curv + rc[i] * dcurv) + dalpha * amp * rc[i] * curv;
    rt[i] = 0;
    vt[i] = 0;
  }
  mid_frenet(m);
  mid_pitch(m);
}

static void mid_lin(struct Midline *m) {
  int nm = m->nm;
  Real *rs = m->rs, *width = m->width, *height = m->height;
  Real(*r)[3] = m->r, (*v)[3] = m->v, (*nor)[3] = m->nor, (*vnor)[3] = m->vnor, (*bin)[3] = m->bin,
  (*vbin)[3] = m->vbin;
  Real V = 0, cm[3] = {0, 0, 0}, lm[3] = {0, 0, 0};
  int i;
  Real volume;
  Real aux;
  int d;

  for (i = 0; i < nm; ++i) {
    Real ds =
        0.5 * ((i == 0) ? rs[1] - rs[0] : ((i == nm - 1) ? rs[nm - 1] - rs[nm - 2] : rs[i + 1] - rs[i - 1]));
    Real c[3], xdot[3], ndot[3], bdot[3];
    int d;
    Real w;
    Real H;
    Real aux1;
    Real aux2;
    Real aux3;
    cross3(c, nor[i], bin[i]);
    for (d = 0; d < 3; d++) {
      xdot[d] = mid_dds(m, i, r, d, nm);
      ndot[d] = mid_dds(m, i, nor, d, nm);
      bdot[d] = mid_dds(m, i, bin, d, nm);
    }
    w = width[i];
    H = height[i];
    aux1 = w * H * dot3(c, xdot) * ds;
    aux2 = 0.25 * w * w * w * H * dot3(c, ndot) * ds;
    aux3 = 0.25 * w * H * H * H * dot3(c, bdot) * ds;
    V += aux1;
    for (d = 0; d < 3; d++) {
      cm[d] += r[i][d] * aux1 + nor[i][d] * aux2 + bin[i][d] * aux3;
      lm[d] += v[i][d] * aux1 + vnor[i][d] * aux2 + vbin[i][d] * aux3;
    }
  }
  volume = V * M_PI;
  aux = M_PI / volume;
  for (d = 0; d < 3; d++) {
    cm[d] *= aux;
    lm[d] *= aux;
  }
#pragma omp parallel for schedule(static)
  for (i = 0; i < nm; ++i) {
    int d;
    for (d = 0; d < 3; d++) {
      r[i][d] -= cm[d];
      v[i][d] -= lm[d];
    }
  }
}

static void rotate_pair(Real R[3][3], Real w[3], Real p[3], Real vp[3]) {
  Real p0[3] = {p[0], p[1], p[2]}, v0[3] = {vp[0], vp[1], vp[2]};
  int a;
  for (a = 0; a < 3; a++) {
    p[a] = R[a][0] * p0[0] + R[a][1] * p0[1] + R[a][2] * p0[2];
    vp[a] = R[a][0] * v0[0] + R[a][1] * v0[1] + R[a][2] * v0[2];
  }
  for (a = 0; a < 3; a++) {
    int b = (a + 1) % 3, c = (a + 2) % 3;
    vp[a] += w[c] * p[b] - w[b] * p[c];
  }
}
static void quat_rate(Real q[4], Real w[3], Real dq[4]) {
  dq[0] = 0.5 * (-w[0] * q[1] - w[1] * q[2] - w[2] * q[3]);
  dq[1] = 0.5 * (+w[0] * q[0] + w[1] * q[3] - w[2] * q[2]);
  dq[2] = 0.5 * (-w[0] * q[3] + w[1] * q[0] + w[2] * q[1]);
  dq[3] = 0.5 * (+w[0] * q[2] - w[1] * q[1] + w[2] * q[0]);
}
static void quat_normalize(Real q[4]) {
  Real inv_d = 1.0 / sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  int d;
  for (d = 0; d < 4; d++)
    q[d] *= inv_d;
}
static void quat_to_rotation(Real q[4], Real R[3][3]) {
  R[0][0] = 1 - 2 * (q[2] * q[2] + q[3] * q[3]);
  R[0][1] = 2 * (q[1] * q[2] - q[3] * q[0]);
  R[0][2] = 2 * (q[1] * q[3] + q[2] * q[0]);
  R[1][0] = 2 * (q[1] * q[2] + q[3] * q[0]);
  R[1][1] = 1 - 2 * (q[1] * q[1] + q[3] * q[3]);
  R[1][2] = 2 * (q[2] * q[3] - q[1] * q[0]);
  R[2][0] = 2 * (q[1] * q[3] - q[2] * q[0]);
  R[2][1] = 2 * (q[2] * q[3] + q[1] * q[0]);
  R[2][2] = 1 - 2 * (q[1] * q[1] + q[2] * q[2]);
}
static void mat3_apply(Real R[3][3], Real x[3]) {
  Real p[3] = {x[0], x[1], x[2]};
  int a;
  for (a = 0; a < 3; a++)
    x[a] = R[a][0] * p[0] + R[a][1] * p[1] + R[a][2] * p[2];
}
static void mat3_apply_t(Real R[3][3], Real x[3]) {
  Real p[3] = {x[0], x[1], x[2]};
  int a;
  for (a = 0; a < 3; a++)
    x[a] = R[0][a] * p[0] + R[1][a] * p[1] + R[2][a] * p[2];
}
static void inertia_add(Real *J, Real f, Real p[3]) {
  J[0] += f * (p[1] * p[1] + p[2] * p[2]);
  J[1] += f * (p[0] * p[0] + p[2] * p[2]);
  J[2] += f * (p[0] * p[0] + p[1] * p[1]);
  J[3] -= f * p[0] * p[1];
  J[4] -= f * p[0] * p[2];
  J[5] -= f * p[1] * p[2];
}
static void mid_ang(struct Midline *m, Real dt) {
  int nm = m->nm;
  Real *rs = m->rs, *width = m->width, *height = m->height;
  Real(*r)[3] = m->r, (*v)[3] = m->v, (*nor)[3] = m->nor, (*vnor)[3] = m->vnor, (*bin)[3] = m->bin,
  (*vbin)[3] = m->vbin;
  Real *quat_int = m->quat_int;
  Real *omega_int = m->omega_int;
  Real Jd[3] = {0, 0, 0};
  Real Jo[3] = {0, 0, 0};
  Real AM[3] = {0, 0, 0};
  int i;
  Real eps;
  int d;
  Real m00, m01, m02, m11, m12, m22;
  Real a00;
  Real a01;
  Real a02;
  Real a11;
  Real a12;
  Real a22;
  Real determinant;
  Real dqdt[4];
  Real R[3][3];

  for (i = 0; i < nm; ++i) {
    Real ds =
        0.5 * ((i == 0) ? rs[1] - rs[0] : ((i == nm - 1) ? rs[nm - 1] - rs[nm - 2] : rs[i + 1] - rs[i - 1]));
    Real c[3], xdot[3], ndot[3], bdot[3];
    int d;
    Real M00;
    Real M11;
    Real M22;
    Real c_r;
    Real c_n;
    Real c_b;
    Real XX;
    Real YY;
    Real ZZ;
    cross3(c, nor[i], bin[i]);
    for (d = 0; d < 3; d++) {
      xdot[d] = mid_dds(m, i, r, d, nm);
      ndot[d] = mid_dds(m, i, nor, d, nm);
      bdot[d] = mid_dds(m, i, bin, d, nm);
    }
    M00 = width[i] * height[i];
    M11 = 0.25 * width[i] * width[i] * width[i] * height[i];
    M22 = 0.25 * width[i] * height[i] * height[i] * height[i];
    c_r = dot3(c, xdot);
    c_n = dot3(c, ndot);
    c_b = dot3(c, bdot);
#define J2(a, b)                                                                                             \
  (c_r * (r[i][a] * r[i][b] * M00 + nor[i][a] * nor[i][b] * M11 + bin[i][a] * bin[i][b] * M22) +             \
   c_n * M11 * (r[i][a] * nor[i][b] + r[i][b] * nor[i][a]) +                                                 \
   c_b * M22 * (r[i][a] * bin[i][b] + r[i][b] * bin[i][a]))
#define K(a, b)                                                                                              \
  (c_r * (v[i][a] * r[i][b] * M00 + vnor[i][a] * nor[i][b] * M11 + vbin[i][a] * bin[i][b] * M22) +           \
   c_n * M11 * (v[i][a] * nor[i][b] + r[i][b] * vnor[i][a]) +                                                \
   c_b * M22 * (v[i][a] * bin[i][b] + r[i][b] * vbin[i][a]))
    Jo[0] += -ds * J2(0, 1);
    Jo[2] += -ds * J2(2, 0);
    Jo[1] += -ds * J2(1, 2);
    XX = ds * J2(0, 0);
    YY = ds * J2(1, 1);
    ZZ = ds * J2(2, 2);
    Jd[0] += YY + ZZ;
    Jd[1] += ZZ + XX;
    Jd[2] += YY + XX;
    AM[0] += (K(2, 1) - K(1, 2)) * ds;
    AM[1] += (K(0, 2) - K(2, 0)) * ds;
    AM[2] += (K(1, 0) - K(0, 1)) * ds;
#undef J2
#undef K
  }
  eps = DBL_EPSILON;
  for (d = 0; d < 3; d++) {
    if (Jd[d] < eps)
      Jd[d] += eps;
    Jd[d] *= M_PI;
    Jo[d] *= M_PI;
    AM[d] *= M_PI;
  }
  m00 = Jd[0];
  m01 = Jo[0];
  m02 = Jo[2];
  m11 = Jd[1];
  m12 = Jo[1];
  m22 = Jd[2];
  a00 = m22 * m11 - m12 * m12;
  a01 = m02 * m12 - m22 * m01;
  a02 = m01 * m12 - m02 * m11;
  a11 = m22 * m00 - m02 * m02;
  a12 = m01 * m02 - m00 * m12;
  a22 = m00 * m11 - m01 * m01;
  determinant = 1.0 / ((m00 * a00) + (m01 * a01) + (m02 * a02));
  omega_int[0] = (a00 * AM[0] + a01 * AM[1] + a02 * AM[2]) * determinant;
  omega_int[1] = (a01 * AM[0] + a11 * AM[1] + a12 * AM[2]) * determinant;
  omega_int[2] = (a02 * AM[0] + a12 * AM[1] + a22 * AM[2]) * determinant;
  quat_rate(quat_int, omega_int, dqdt);
  for (d = 0; d < 4; d++)
    quat_int[d] -= dt * dqdt[d];
  quat_normalize(quat_int);
  quat_to_rotation(quat_int, R);
  for (i = 0; i < nm; ++i) {
    rotate_pair(R, omega_int, r[i], v[i]);
    rotate_pair(R, omega_int, nor[i], vnor[i]);
    rotate_pair(R, omega_int, bin[i], vbin[i]);
  }
}
static void fish_init(struct Fish *f) {
  Real angle = f->angle / 180 * M_PI;
  Real *q = f->quaternion;
  Real enforced_velocity[3];
  int d;
  int any_vel_forced;
  q[0] = cos(0.5 * angle);
  q[1] = 0;
  q[2] = 0;
  q[3] = sin(0.5 * angle);
  for (d = 0; d < 3; d++) {
    enforced_velocity[d] = -f->vel_imposed[d];
    f->abs_pos[d] = f->position[d];
    f->vel[d] = f->omega[d] = f->vel_imposed[d] = 0;
    f->block_rot[d] = 0;
  }
  if (f->length < 5 * DBL_EPSILON)
    fatal("fish length %g must be positive", f->length);
  for (d = 0; d < 3; ++d) {
    if (f->forced[d]) {
      f->vel_imposed[d] = f->vel[d] = enforced_velocity[d];
    }
  }
  any_vel_forced = f->forced[0] || f->forced[1] || f->forced[2];
  if (any_vel_forced)
    f->block_rot[0] = f->block_rot[1] = f->block_rot[2] = 1;
  if (f->planar) {
    f->forced[2] = 1;
    f->vel_imposed[2] = 0;
    f->block_rot[1] = 1;
    f->block_rot[0] = 1;
  }
  if ((f->correct_pos || f->correct_z || f->correct_roll) && fabs(f->quaternion[0] - 1) > 1e-6)
    fatal("PID controller only works for zero initial angles");
  mid_init(&f->m, f->length, sim.hmin);
  profile_set(height_profiles, NHEIGHT, "height", f->hprof, f->length, f->m.rs, f->m.height, f->m.nm);
  profile_set(width_profiles, NWIDTH, "width", f->wprof, f->length, f->m.rs, f->m.width, f->m.nm);

  for (d = 1; d < f->m.nm - 1; d++) {
    if (f->m.height[d] <= 0)
      f->m.height[d] = 1e-10;
    if (f->m.width[d] <= 0)
      f->m.width[d] = 1e-10;
  }
  f->orig[0] = f->position[0];
  f->orig[1] = f->position[1];
  f->orig[2] = f->position[2];
  if (sim.rank == 0)
    printf("nMidline=%d, length=%f, Tperiod=%f, phaseShift=%f, height=%s, width=%s\n", f->m.nm, f->length,
           f->m.period, f->m.phase, f->hprof, f->wprof);
}
static void fish_parse(char *content) {
  char *text = strdup(content);
  char *save;
  char *line;
  sim.nfish = 0;
  sta.fish = NULL;
  for (line = strtok_r(text, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
    char *save2;
    char *tok;
    struct Fish *fish;
    int seen[N_FISHP];
    while (isspace(*line))
      line++;
    if (*line == '\0' || *line == '#')
      continue;
    sta.fish = erealloc(sta.fish, (sim.nfish + 1) * sizeof *sta.fish);
    fish = &sta.fish[sim.nfish];
    memset(fish, 0, sizeof *fish);
    memset(seen, 0, sizeof seen);
    for (tok = strtok_r(line, " \t", &save2); tok != NULL; tok = strtok_r(NULL, " \t", &save2)) {
      char *eq = strchr(tok, '=');
      if (eq == NULL || eq == tok)
        param_fail(tok, "", "expected key=value");
      *eq = '\0';
      param_apply(fish_params, N_FISHP, fish, tok, eq + 1, seen);
    }
    param_check(fish_params, N_FISHP, seen, "fish");
    fish_init(fish);
    sim.nfish++;
  }
  if (sim.nfish == 0 && sim.rank == 0)
    fprintf(stderr, "[CUP3D] OBSTACLE FACTORY did not create any obstacles.\n");
  free(text);
}

static struct Sfc {
  int BX, BY, BZ, level_max, is_regular, base_level;
  long long *Zsave;
  int *i_inv, *j_inv, *k_inv;
} sfc;
static long long axes_to_transpose(int *X_in, int b) {
  int n;
  int X[3];
  int M, P, Q, t;
  int i;
  long long retval;
  long long a;
  long long one;
  long long two;
  long long level;
  if (b == 0)
    return 0;
  n = 3;
  X[0] = X_in[0];
  X[1] = X_in[1];
  X[2] = X_in[2];
  M = 1 << (b - 1);
  for (Q = M; Q > 1; Q >>= 1) {
    P = Q - 1;
    for (i = 0; i < n; i++)
      if (X[i] & Q)
        X[0] ^= P;
      else {
        t = (X[0] ^ X[i]) & P;
        X[0] ^= t;
        X[i] ^= t;
      }
  }
  for (i = 1; i < n; i++)
    X[i] ^= X[i - 1];
  t = 0;
  for (Q = M; Q > 1; Q >>= 1)
    if (X[n - 1] & Q)
      t ^= Q - 1;
  for (i = 0; i < n; i++)
    X[i] ^= t;
  retval = 0;
  a = 0;
  one = 1;
  two = 2;
  for (level = 0; level < b; level++) {
    long long a0 = ((one) << (a)) * ((long long)X[2] >> level & one);
    long long a1 = ((one) << (a + one)) * ((long long)X[1] >> level & one);
    long long a2 = ((one) << (a + two)) * ((long long)X[0] >> level & one);
    retval += a0 + a1 + a2;
    a += 3;
  }
  return retval;
}
static void transpose_to_axes(long long index, long long *X, int b) {
  int n = 3;
  long long aa;
  long long one;
  long long two;
  long long i;
  int N, P, Q, t;
  int i2;
  X[0] = 0;
  X[1] = 0;
  X[2] = 0;
  if (b == 0 && index == 0)
    return;
  aa = 0;
  one = 1;
  two = 2;
  for (i = 0; index > 0; i++) {
    long long x2 = index % two;
    long long x1;
    long long x0;
    index = index / two;
    x1 = index % two;
    index = index / two;
    x0 = index % two;
    index = index / two;
    X[0] += x0 * (one << aa);
    X[1] += x1 * (one << aa);
    X[2] += x2 * (one << aa);
    aa += 1;
  }
  N = 2 << (b - 1);
  t = X[n - 1] >> 1;
  for (i2 = n - 1; i2 >= 1; i2--)
    X[i2] ^= X[i2 - 1];
  X[0] ^= t;
  for (Q = 2; Q != N; Q <<= 1) {
    P = Q - 1;
    for (i2 = n - 1; i2 >= 0; i2--)
      if (X[i2] & Q)
        X[0] ^= P;
      else {
        t = (X[0] ^ X[i2]) & P;
        X[0] ^= t;
        X[i2] ^= t;
      }
  }
}
static void sfc_init(int BX, int BY, int BZ, int lmax) {
  int n_max;
  int n0;
  int i;
  int k;
  int j;
  sfc.BX = BX;
  sfc.BY = BY;
  sfc.BZ = BZ;
  sfc.level_max = lmax;
  n_max = BX > BY ? BX : BY;
  if (BZ > n_max)
    n_max = BZ;
  sfc.base_level = (log(n_max) / log(2));
  if (sfc.base_level < (double)(log(n_max) / log(2)))
    sfc.base_level++;
  n0 = BX * BY * BZ;
  sfc.Zsave = emalloc(n0 * sizeof *sfc.Zsave);
  sfc.i_inv = emalloc(n0 * sizeof *sfc.i_inv);
  sfc.j_inv = emalloc(n0 * sizeof *sfc.j_inv);
  sfc.k_inv = emalloc(n0 * sizeof *sfc.k_inv);
  for (i = 0; i < n0; i++)
    sfc.Zsave[i] = sfc.i_inv[i] = sfc.j_inv[i] = sfc.k_inv[i] = -1;
  sfc.is_regular = 1;
  for (k = 0; k < BZ; k++)
    for (j = 0; j < BY; j++)
      for (i = 0; i < BX; i++) {
        int c[3] = {i, j, k};
        long long index = axes_to_transpose(c, sfc.base_level);
        long long substract = 0;
        long long h;
        for (h = 0; h < index; h++) {
          long long X[3] = {0, 0, 0};
          transpose_to_axes(h, X, sfc.base_level);
          if (X[0] >= BX || X[1] >= BY || X[2] >= BZ)
            substract++;
        }
        index -= substract;
        if (substract > 0)
          sfc.is_regular = 0;
        sfc.i_inv[index] = i;
        sfc.j_inv[index] = j;
        sfc.k_inv[index] = k;
        sfc.Zsave[k * BX * BY + j * BX + i] = index;
      }
}
static long long sfc_forward(int l, int i, int j, int k) {
  int aux = 1 << l;
  long long retval;
  if (l >= sfc.level_max)
    return 0;
  if (!sfc.is_regular) {
    int I = i / aux;
    int J = j / aux;
    int K = k / aux;
    int c2_a[3] = {i - I * aux, j - J * aux, k - K * aux};
    retval = axes_to_transpose(c2_a, l);
    retval += sfc.Zsave[(J + K * sfc.BY) * sfc.BX + I] * aux * aux * aux;
  } else {
    int c2_a[3] = {i, j, k};
    retval = axes_to_transpose(c2_a, l + sfc.base_level);
  }
  return retval;
}
static void sfc_inverse(long long Z, int l, int *i, int *j, int *k) {
  if (sfc.is_regular) {
    long long X[3] = {0, 0, 0};
    transpose_to_axes(Z, X, l + sfc.base_level);
    *i = X[0];
    *j = X[1];
    *k = X[2];
  } else {
    long long aux = 1 << l;
    long long Zloc = Z % (aux * aux * aux);
    long long X[3] = {0, 0, 0};
    long long index;
    transpose_to_axes(Zloc, X, l);
    index = Z / (aux * aux * aux);
    *i = X[0] + sfc.i_inv[index] * aux;
    *j = X[1] + sfc.j_inv[index] * aux;
    *k = X[2] + sfc.k_inv[index] * aux;
  }
}
static void blk_fill(struct Blk *b, int level, long long Z) {
  int i, j, k;
  sfc_inverse(Z, level, &i, &j, &k);
  b->level = level;
  b->Z = Z;
  b->ix = i;
  b->iy = j;
  b->iz = k;
  b->h = sim.h0 / (1 << level);
  b->origin[0] = i * BS * b->h;
  b->origin[1] = j * BS * b->h;
  b->origin[2] = k * BS * b->h;
}
static void blk_pos(struct Blk *b, int ix, int iy, int iz, Real p[3]) {
  p[0] = b->origin[0] + b->h * (ix + 0.5);
  p[1] = b->origin[1] + b->h * (iy + 0.5);
  p[2] = b->origin[2] + b->h * (iz + 0.5);
}
static void mpi_check(int err, char *what) {
  char msg[MPI_MAX_ERROR_STRING];
  int len;
  if (err == MPI_SUCCESS)
    return;
  MPI_Error_string(err, msg, &len);
  fatal("%s: %s", what, msg);
}

static void io_write(char *path, float *buf, long n, int m, long off) {
  MPI_File fp;
  MPI_Datatype t;
  MPI_Type_contiguous(m, MPI_FLOAT, &t);
  MPI_Type_commit(&t);
  mpi_check(MPI_File_open(sim.comm, path, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp), path);
  mpi_check(MPI_File_write_at_all(fp, (MPI_Offset)off * m * sizeof *buf, buf, (int)n, t, MPI_STATUS_IGNORE),
            path);
  mpi_check(MPI_File_close(&fp), path);
  MPI_Type_free(&t);
}
static void vorticity(void);
static void qcrit(void);
static void io_dump(Real time, char *path) {
  long i, j, l, m, ncell, nblk_total, offset, boff;
  char attr_path[FILENAME_MAX], vort_path[FILENAME_MAX], q_path[FILENAME_MAX], blk_path[FILENAME_MAX],
      xdmf_path[FILENAME_MAX], *attr_base, *vort_base, *q_base, *blk_base;
  FILE *xmf;
  float *attr, *vort, *q, *blk, *all;
  int *cnt, *dsp;
  int n, r, root;
  vorticity();
  qcrit();
  snprintf(attr_path, sizeof attr_path, "%s.attr.raw", path);
  snprintf(vort_path, sizeof vort_path, "%s.vort.raw", path);
  snprintf(q_path, sizeof q_path, "%s.q.raw", path);
  snprintf(blk_path, sizeof blk_path, "%s.blk.raw", path);
  snprintf(xdmf_path, sizeof xdmf_path, "%s.xdmf2", path);
  attr_base = attr_path;
  vort_base = vort_path;
  q_base = q_path;
  blk_base = blk_path;
  for (j = 0; attr_path[j] != '\0'; j++) {
    if (attr_path[j] == '/' && attr_path[j + 1] != '\0') {
      attr_base = &attr_path[j + 1];
      vort_base = &vort_path[j + 1];
      q_base = &q_path[j + 1];
      blk_base = &blk_path[j + 1];
    }
  }
  ncell = sta.nblk * BS3;
  if (ncell > INT_MAX)
    fatal("dump: %ld cells on one rank exceed the MPI-IO count limit", ncell);
  MPI_Exscan(&ncell, &offset, 1, MPI_LONG, MPI_SUM, sim.comm);
  if (sim.rank == 0)
    offset = 0;
  boff = offset / BS3;
  blk = emalloc(6 * (sta.nblk > 0 ? sta.nblk : 1) * sizeof *blk);
  for (i = 0; i < sta.nblk; i++) {
    struct Blk *b = &sta.blk[i];
    blk[6 * i] = b->origin[2];
    blk[6 * i + 1] = b->origin[1];
    blk[6 * i + 2] = b->origin[0];
    blk[6 * i + 3] = blk[6 * i + 4] = blk[6 * i + 5] = b->h;
  }
  root = sim.size - 1;
  n = 6 * (int)sta.nblk;
  cnt = emalloc(sim.size * sizeof *cnt);
  dsp = emalloc(sim.size * sizeof *dsp);
  MPI_Gather(&n, 1, MPI_INT, cnt, 1, MPI_INT, root, sim.comm);
  all = NULL;
  if (sim.rank == root) {
    dsp[0] = 0;
    for (r = 1; r < sim.size; r++)
      dsp[r] = dsp[r - 1] + cnt[r - 1];
    all = emalloc((dsp[sim.size - 1] + cnt[sim.size - 1] + 1) * sizeof *all);
  }
  MPI_Gatherv(blk, n, MPI_FLOAT, all, cnt, dsp, MPI_FLOAT, root, sim.comm);
  if (sim.rank == root) {
    long k;
    nblk_total = (offset + ncell) / BS3;
    xmf = fopen(xdmf_path, "w");
    if (xmf == NULL)
      fatal("cannot write %s", xdmf_path);
    fprintf(xmf,
            "<Xdmf Version=\"2.0\">\n"
            " <Domain>\n"
            "  <Grid GridType=\"Collection\" CollectionType=\"Spatial\">\n"
            "   <Time Value=\"%.16e\"/>\n",
            time);
    for (k = 0; k < nblk_total; k++)
      fprintf(xmf,
              "   <Grid GridType=\"Uniform\">\n"
              "    <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"%d %d %d\"/>\n"
              "    <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
              "     <DataItem Dimensions=\"3\" Format=\"Binary\" Seek=\"%ld\">%s</DataItem>\n"
              "     <DataItem Dimensions=\"3\" Format=\"Binary\" Seek=\"%ld\">%s</DataItem>\n"
              "    </Geometry>\n"
              "    <Attribute Name=\"chi\" Center=\"Cell\">\n"
              "     <DataItem Dimensions=\"%d %d %d\" Format=\"Binary\" Seek=\"%ld\">%s</DataItem>\n"
              "    </Attribute>\n"
              "    <Attribute Name=\"vorticity\" AttributeType=\"Vector\" Center=\"Cell\">\n"
              "     <DataItem Dimensions=\"%d %d %d 3\" Format=\"Binary\" Seek=\"%ld\">%s</DataItem>\n"
              "    </Attribute>\n"
              "    <Attribute Name=\"q\" Center=\"Cell\">\n"
              "     <DataItem Dimensions=\"%d %d %d\" Format=\"Binary\" Seek=\"%ld\">%s</DataItem>\n"
              "    </Attribute>\n"
              "   </Grid>\n",
              BS + 1, BS + 1, BS + 1, 24 * k, blk_base, 24 * k + 12, blk_base, BS, BS, BS, 4L * BS3 * k,
              attr_base, BS, BS, BS, 12L * BS3 * k, vort_base, BS, BS, BS, 4L * BS3 * k, q_base);
    fprintf(xmf, "  </Grid>\n </Domain>\n</Xdmf>\n");
    fclose(xmf);
    free(all);
  }
  free(cnt);
  free(dsp);
  attr = emalloc(ncell * sizeof *attr);
  vort = emalloc(3 * ncell * sizeof *vort);
  q = emalloc(ncell * sizeof *q);
  l = 0;
  m = 0;
  for (i = 0; i < sta.nblk; i++) {
    Real *chi = fld(i, F_CHI);
    Real *om = fld(i, F_TMP);
    Real *qq = fld(i, F_LHS);
    for (j = 0; j < BS3; j++) {
      vort[m++] = om[j];
      vort[m++] = om[BS3 + j];
      vort[m++] = om[2 * BS3 + j];
      q[l] = qq[j];
      attr[l++] = chi[j];
    }
  }
  io_write(attr_path, attr, ncell, 1, offset);
  free(attr);
  io_write(vort_path, vort, ncell, 3, offset);
  free(vort);
  io_write(q_path, q, ncell, 1, offset);
  free(q);
  io_write(blk_path, blk, sta.nblk, 6, boff);
  free(blk);
}
static char *param_parse(int argc, char **argv) {
  int seen[N_SIMP];
  char *content = NULL;
  int i;
  int aux;
  Real NFE[3];
  Real maxbpd;
  int maxb;
  memset(seen, 0, sizeof seen);
  for (i = 1; i < argc; i += 2) {
    if (argv[i][0] != '-' || i + 1 >= argc)
      fatal("expected '-key value' at '%s'", argv[i]);
    if (strcmp(argv[i] + 1, "factory-content") == 0) {
      if (content != NULL)
        param_fail("factory-content", "", "given twice");
      content = argv[i + 1];
    } else
      param_apply(sim_params, N_SIMP, NULL, argv[i] + 1, argv[i + 1], seen);
  }
  param_check(sim_params, N_SIMP, seen, "command line");
  if (content == NULL)
    fatal("command line: missing parameter 'factory-content'");
  if (sim.bpdx < 1 || sim.bpdy < 1 || sim.bpdz < 1)
    fatal("invalid bpd: %d x %d x %d", sim.bpdx, sim.bpdy, sim.bpdz);
  aux = 1 << (sim.level_max - 1);
  NFE[0] = (Real)sim.bpdx * aux * BS;
  NFE[1] = (Real)sim.bpdy * aux * BS;
  NFE[2] = (Real)sim.bpdz * aux * BS;
  maxbpd = NFE[0];
  if (NFE[1] > maxbpd)
    maxbpd = NFE[1];
  if (NFE[2] > maxbpd)
    maxbpd = NFE[2];
  maxb = sim.bpdx;
  if (sim.bpdy > maxb)
    maxb = sim.bpdy;
  if (sim.bpdz > maxb)
    maxb = sim.bpdz;
  sim.h0 = sim.extent_max / maxb / BS;
  sim.extents[0] = (NFE[0] / maxbpd) * sim.extent_max;
  sim.extents[1] = (NFE[1] / maxbpd) * sim.extent_max;
  sim.extents[2] = (NFE[2] / maxbpd) * sim.extent_max;
  sim.hmin = sim.extents[0] / NFE[0];
  return content;
}
static void sta_init(void) {
  sta.coef_u[0] = 1.5;
  sta.coef_u[1] = -2.0;
  sta.coef_u[2] = 0.5;
  sta.mesh_changed = 1;
}

static Real min3(Real a, Real b, Real c) {
  Real m = a;
  if (b < m)
    m = b;
  if (c < m)
    m = c;
  return m;
}
static Real max4(Real a, Real b, Real c, Real d) {
  Real m = a;
  if (b > m)
    m = b;
  if (c > m)
    m = c;
  if (d > m)
    m = d;
  return m;
}
static Real min4(Real a, Real b, Real c, Real d) {
  Real m = a;
  if (b < m)
    m = b;
  if (c < m)
    m = c;
  if (d < m)
    m = d;
  return m;
}
static void seg_normalize(struct Segment *s) {
  Real *n[3] = {s->ni, s->nj, s->nk};
  int k;
  for (k = 0; k < 3; k++) {
    Real inv = (Real)1 / sqrt(dot3(n[k], n[k]));
    int i;
    for (i = 0; i < 3; ++i)
      n[k][i] = fabs(n[k][i]) * inv;
  }
}
static void seg_prepare(struct Segment *s, int s0, int s1, Real bbox[3][2], Real h) {
  int i;
  s->safe_distance = (1 + 2) * h;
  s->s0 = s0;
  s->s1 = s1;
  s->ni[0] = 1;
  s->ni[1] = 0;
  s->ni[2] = 0;
  s->nj[0] = 0;
  s->nj[1] = 1;
  s->nj[2] = 0;
  s->nk[0] = 0;
  s->nk[1] = 0;
  s->nk[2] = 1;
  for (i = 0; i < 3; ++i) {
    s->w[i] = (bbox[i][1] - bbox[i][0]) / 2 + s->safe_distance;
    s->c[i] = (bbox[i][1] + bbox[i][0]) / 2;
  }
}
static void seg_to_frame(struct Segment *s, Real position[3], Real quaternion[4]) {
  Real R[3][3];
  int i;
  quat_to_rotation(quaternion, R);
  mat3_apply(R, s->c);
  mat3_apply(R, s->ni);
  mat3_apply(R, s->nj);
  mat3_apply(R, s->nk);
  for (i = 0; i < 3; ++i)
    s->c[i] += position[i];
  seg_normalize(s);
  for (i = 0; i < 3; ++i) {
    Real wx = s->w[0] * s->ni[i], wy = s->w[1] * s->nj[i], wz = s->w[2] * s->nk[i];
    s->box_lab[i][0] = s->c[i] - wx - wy - wz;
    s->box_lab[i][1] = s->c[i] + wx + wy + wz;
    s->box_obj[i][0] = s->c[i] - s->w[i];
    s->box_obj[i][1] = s->c[i] + s->w[i];
  }
}
static int seg_intersects(struct Segment *s, Real start[3], Real end[3]) {
  Real AABB_w[3] = {(end[0] - start[0]) / 2 + s->safe_distance, (end[1] - start[1]) / 2 + s->safe_distance,
                    (end[2] - start[2]) / 2 + s->safe_distance};
  Real AABB_c[3] = {(end[0] + start[0]) / 2, (end[1] + start[1]) / 2, (end[2] + start[2]) / 2};
  Real AABB_box[3][2] = {{AABB_c[0] - AABB_w[0], AABB_c[0] + AABB_w[0]},
                         {AABB_c[1] - AABB_w[1], AABB_c[1] + AABB_w[1]},
                         {AABB_c[2] - AABB_w[2], AABB_c[2] + AABB_w[2]}};
  int d;
  Real *N[3];
  Real box[3][2];
  for (d = 0; d < 3; d++) {
    Real lo = s->box_lab[d][0] > AABB_box[d][0] ? s->box_lab[d][0] : AABB_box[d][0];
    Real hi = s->box_lab[d][1] < AABB_box[d][1] ? s->box_lab[d][1] : AABB_box[d][1];
    if (hi - lo < 0)
      return 0;
  }
  N[0] = s->ni;
  N[1] = s->nj;
  N[2] = s->nk;
  for (d = 0; d < 3; d++) {
    Real wx = AABB_w[0] * N[d][0], wy = AABB_w[1] * N[d][1], wz = AABB_w[2] * N[d][2];
    box[d][0] = AABB_c[d] - wx - wy - wz;
    box[d][1] = AABB_c[d] + wx + wy + wz;
  }
  for (d = 0; d < 3; d++) {
    Real lo = box[d][0] > s->box_obj[d][0] ? box[d][0] : s->box_obj[d][0];
    Real hi = box[d][1] < s->box_obj[d][1] ? box[d][1] : s->box_obj[d][1];
    if (hi - lo < 0)
      return 0;
  }
  return 1;
}
struct Frame {
  struct Midline *m;
  Real position[3], quaternion[4], R[3][3];
};
static void frame_init(struct Frame *f, struct Fish *fish) {
  Real *q = fish->quaternion;
  int i;
  f->m = &fish->m;
  for (i = 0; i < 3; i++)
    f->position[i] = fish->position[i];
  for (i = 0; i < 4; i++)
    f->quaternion[i] = q[i];
  quat_to_rotation(q, f->R);
}
static Real euler_dist_sq(Real a[3], Real b[3]) {
  return pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2);
}
static void vel_to_frame(struct Frame *f, Real x[3]) { mat3_apply(f->R, x); }
static void to_frame(struct Frame *f, Real x[3]) {
  int d;
  mat3_apply(f->R, x);
  for (d = 0; d < 3; d++)
    x[d] += f->position[d];
}
static void from_frame(struct Frame *f, Real x[3]) {
  int d;
  for (d = 0; d < 3; d++)
    x[d] -= f->position[d];
  mat3_apply_t(f->R, x);
}
static Real dist_plane(Real p1[3], Real p2[3], Real p3[3], Real s[3], Real IN[3]) {
  Real t[3] = {s[0] - p1[0], s[1] - p1[1], s[2] - p1[2]};
  Real u[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
  Real v[3] = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};
  Real i[3] = {IN[0] - p1[0], IN[1] - p1[1], IN[2] - p1[2]};
  Real n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
  Real proj_inner = i[0] * n[0] + i[1] * n[1] + i[2] * n[2];
  Real sign_in = proj_inner > 0 ? 1 : -1;
  Real norm = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  return sign_in * (t[0] * n[0] + t[1] * n[1] + t[2] * n[2]) / norm;
}
static void geom_in(struct Frame *fr, Real h, Real ox, Real oy, Real oz, struct ObstacleBlock *defblock,
                    struct Segment **segs, int nseg) {
  struct Midline *cfish = fr->m;
  Real org[3] = {ox - h, oy - h, oz - h};
  Real invh = 1.0 / h;
  int BSP[3] = {BS + 2, BS + 2, BS + 2};
  Real(*r)[3] = cfish->r, (*v)[3] = cfish->v, (*nor)[3] = cfish->nor, (*vnor)[3] = cfish->vnor,
  (*bin)[3] = cfish->bin, (*vbin)[3] = cfish->vbin;
  Real *width = cfish->width, *height = cfish->height;
  int i;
  for (i = 0; i < nseg; ++i) {
    int first_segm = segs[i]->s0 > 1 ? segs[i]->s0 : 1;
    int last_segm = segs[i]->s1 < cfish->nm - 2 ? segs[i]->s1 : cfish->nm - 2;
    int ss;
    for (ss = first_segm; ss <= last_segm; ++ss) {
      Real my_width = width[ss], my_height = height[ss];
      int Nh = floor(my_height / h);
      int ih;
      for (ih = -Nh + 1; ih < Nh; ++ih) {
        Real offset_h = ih * h;
        Real curr_width = my_width * sqrt(1 - pow(offset_h / my_height, 2));
        int Nw = floor(curr_width / h);
        int iw;
        for (iw = -Nw + 1; iw < Nw; ++iw) {
          Real offset_w = iw * h;
          Real xp[3], udef[3];
          int d;
          Real ap[3];
          int iap[3];
          Real wghts[3][2];
          int c;
          int z0;
          int z1;
          int y0;
          int y1;
          int x0;
          int x1;
          int idz;
          int idy;
          int idx;
          for (d = 0; d < 3; d++) {
            xp[d] = r[ss][d] + offset_w * nor[ss][d] + offset_h * bin[ss][d];
            udef[d] = v[ss][d] + offset_w * vnor[ss][d] + offset_h * vbin[ss][d];
          }
          to_frame(fr, xp);
          for (d = 0; d < 3; d++)
            xp[d] = (xp[d] - org[d]) * invh;
          ap[0] = floor(xp[0]);
          ap[1] = floor(xp[1]);
          ap[2] = floor(xp[2]);
          iap[0] = (int)ap[0];
          iap[1] = (int)ap[1];
          iap[2] = (int)ap[2];
          if (iap[0] + 2 <= 0 || iap[0] >= BSP[0])
            continue;
          if (iap[1] + 2 <= 0 || iap[1] >= BSP[1])
            continue;
          if (iap[2] + 2 <= 0 || iap[2] >= BSP[2])
            continue;
          vel_to_frame(fr, udef);
          for (c = 0; c < 3; ++c) {
            Real t[2] = {fabs(xp[c] - ap[c]), fabs(xp[c] - (ap[c] + 1))};
            wghts[c][0] = 1.0 - t[0];
            wghts[c][1] = 1.0 - t[1];
          }
          z0 = iap[2] > 0 ? iap[2] : 0;
          z1 = iap[2] + 2 < BSP[2] ? iap[2] + 2 : BSP[2];
          y0 = iap[1] > 0 ? iap[1] : 0;
          y1 = iap[1] + 2 < BSP[1] ? iap[1] + 2 : BSP[1];
          x0 = iap[0] > 0 ? iap[0] : 0;
          x1 = iap[0] + 2 < BSP[0] ? iap[0] + 2 : BSP[0];
          for (idz = z0; idz < z1; ++idz)
            for (idy = y0; idy < y1; ++idy)
              for (idx = x0; idx < x1; ++idx) {
                int sx = idx - iap[0], sy = idy - iap[1], sz = idz - iap[2];
                Real wxwywz = wghts[2][sz] * wghts[1][sy] * wghts[0][sx];
                if (idz - 1 >= 0 && idz - 1 < BS && idy - 1 >= 0 && idy - 1 < BS && idx - 1 >= 0 &&
                    idx - 1 < BS) {
                  int d;
                  for (d = 0; d < 3; d++)
                    defblock->udef[idz - 1][idy - 1][idx - 1][d] += wxwywz * udef[d];
                  defblock->chi[idz - 1][idy - 1][idx - 1] += wxwywz;
                }
                if (fabs(defblock->sdf[idz][idy][idx] + 1) < DBL_EPSILON)
                  defblock->sdf[idz][idy][idx] = 1;
              }
        }
      }
    }
  }
}
static void ellipse_point(struct Midline *m, int s, Real costh, Real sinth, Real out[3]) {
  int d;
  for (d = 0; d < 3; d++)
    out[d] = m->r[s][d] + m->width[s] * costh * m->nor[s][d] + m->height[s] * sinth * m->bin[s][d];
}
static void ellipse_offset(struct Midline *m, int s, Real costh, Real sinth, Real out[3]) {
  int d;
  for (d = 0; d < 3; d++)
    out[d] = m->width[s] * costh * m->nor[s][d] + m->height[s] * sinth * m->bin[s][d];
}
static void ellipse_velocity(struct Midline *m, int s, Real costh, Real sinth, Real out[3]) {
  int d;
  for (d = 0; d < 3; d++)
    out[d] = m->v[s][d] + m->width[s] * costh * m->vnor[s][d] + m->height[s] * sinth * m->vbin[s][d];
}
static void geom_surf(struct Frame *fr, Real h, Real ox, Real oy, Real oz, struct ObstacleBlock *defblock,
                      struct Segment **segs, int nseg) {
  struct Midline *cfish = fr->m;
  Real(*r)[3] = cfish->r, (*nor)[3] = cfish->nor, (*bin)[3] = cfish->bin;
  Real *width = cfish->width;
  Real *height = cfish->height;
  Real org[3] = {ox - h, oy - h, oz - h};
  Real invh = 1.0 / h;
  int BSP[3] = {BS + 2, BS + 2, BS + 2};
  Real pc[3];
  int i;
  for (i = 0; i < nseg; ++i) {
    int first_segm = segs[i]->s0 > 1 ? segs[i]->s0 : 1;
    int last_segm = segs[i]->s1 < cfish->nm - 2 ? segs[i]->s1 : cfish->nm - 2;
    int ss;
    for (ss = first_segm; ss <= last_segm; ++ss) {
      Real major_axis;
      Real dtheta_tgt;
      int Ntheta;
      Real dtheta;
      Real offset;
      int tt;
      major_axis = height[ss] > width[ss] ? height[ss] : width[ss];
      dtheta_tgt = fabs(asin(h / (major_axis + h) / 2));
      Ntheta = ceil(2 * M_PI / dtheta_tgt);
      if (Ntheta % 2 == 1)
        Ntheta++;
      dtheta = 2 * M_PI / ((Real)Ntheta);
      offset = height[ss] > width[ss] ? M_PI / 2 : 0;
      for (tt = 0; tt < Ntheta; ++tt) {
        Real theta = tt * dtheta + offset;
        Real sinth = sin(theta), costh = cos(theta);
        int iap[3];
        int nei;
        int ST[3];
        int EN[3];
        Real p_p[3], p_m[3], udef[3];
        int z0, z1;
        int y0, y1;
        int x0, x1;
        int sz;
        int sy;
        int sx;
        ellipse_point(cfish, ss, costh, sinth, pc);
        to_frame(fr, pc);
        iap[0] = (int)floor((pc[0] - org[0]) * invh);
        iap[1] = (int)floor((pc[1] - org[1]) * invh);
        iap[2] = (int)floor((pc[2] - org[2]) * invh);
        nei = 3;
        ST[0] = iap[0] - nei;
        ST[1] = iap[1] - nei;
        ST[2] = iap[2] - nei;
        EN[0] = iap[0] + nei;
        EN[1] = iap[1] + nei;
        EN[2] = iap[2] + nei;
        if (EN[0] <= 0 || ST[0] > BSP[0])
          continue;
        if (EN[1] <= 0 || ST[1] > BSP[1])
          continue;
        if (EN[2] <= 0 || ST[2] > BSP[2])
          continue;
        ellipse_point(cfish, ss + 1, costh, sinth, p_p);
        ellipse_point(cfish, ss - 1, costh, sinth, p_m);
        to_frame(fr, p_m);
        to_frame(fr, p_p);
        ellipse_velocity(cfish, ss, costh, sinth, udef);
        vel_to_frame(fr, udef);
        z0 = ST[2] > 0 ? ST[2] : 0;
        z1 = EN[2] < BSP[2] ? EN[2] : BSP[2];
        y0 = ST[1] > 0 ? ST[1] : 0;
        y1 = EN[1] < BSP[1] ? EN[1] : BSP[1];
        x0 = ST[0] > 0 ? ST[0] : 0;
        x1 = EN[0] < BSP[0] ? EN[0] : BSP[0];
        for (sz = z0; sz < z1; ++sz)
          for (sy = y0; sy < y1; ++sy)
            for (sx = x0; sx < x1; ++sx) {
              Real p[3];
              Real dist0;
              Real dp;
              Real dm;
              int close_s, secnd_s;
              Real dist1, dist2;
              Real Wc;
              Real W;
              int in_range;
              Real R1[3], nn[3], P1[3], P2[3], center_close[3], center_second[3];
              int d;
              Real norm_r1;
              Real base1;
              Real base2;
              Real radius_close;
              Real radius_second;
              Real ds2;
              Real corr;
              p[0] = ox + h * (sx - 1 + 0.5);
              p[1] = oy + h * (sy - 1 + 0.5);
              p[2] = oz + h * (sz - 1 + 0.5);
              dist0 = euler_dist_sq(p, pc);
              dp = euler_dist_sq(p, p_p);
              dm = euler_dist_sq(p, p_m);
              if (fabs(defblock->sdf[sz][sy][sx]) < min3(dist0, dp, dm))
                continue;
              if (min3(dist0, dp, dm) > 4 * h * h)
                continue;
              from_frame(fr, p);
              close_s = ss;
              secnd_s = ss + (dp < dm ? 1 : -1);
              dist1 = dist0;
              dist2 = dp < dm ? dp : dm;
              if (dp < dist0 || dm < dist0) {
                dist1 = dist2;
                dist2 = dist0;
                close_s = secnd_s;
                secnd_s = ss;
              }
              Wc = 1 - sqrt(dist1) * (invh / 3);
              W = Wc > (Real)0 ? Wc : (Real)0;
              in_range =
                  (sz - 1 >= 0 && sz - 1 < BS && sy - 1 >= 0 && sy - 1 < BS && sx - 1 >= 0 && sx - 1 < BS);
              if (in_range) {
                int d;
                for (d = 0; d < 3; d++)
                  defblock->udef[sz - 1][sy - 1][sx - 1][d] = W * udef[d];
                defblock->chi[sz - 1][sy - 1][sx - 1] = W;
              }
              for (d = 0; d < 3; d++)
                R1[d] = r[secnd_s][d] - r[close_s][d];
              norm_r1 = 1.0 / (1e-21 + sqrt(dot3(R1, R1)));
              for (d = 0; d < 3; d++)
                nn[d] = R1[d] * norm_r1;
              ellipse_offset(cfish, close_s, costh, sinth, P1);
              ellipse_offset(cfish, secnd_s, costh, sinth, P2);
              base1 = dot3(P1, R1) * norm_r1;
              base2 = dot3(P2, R1) * norm_r1;
              radius_close = pow(width[close_s] * costh, 2) + pow(height[close_s] * sinth, 2) - base1 * base1;
              radius_second =
                  pow(width[secnd_s] * costh, 2) + pow(height[secnd_s] * sinth, 2) - base2 * base2;
              ds2 = 0;
              for (d = 0; d < 3; d++) {
                center_close[d] = r[close_s][d] - nn[d] * base1;
                center_second[d] = r[secnd_s][d] + nn[d] * base2;
                ds2 += pow(center_close[d] - center_second[d], 2);
              }
              corr = 2 * sqrt(radius_close * radius_second);
              if (close_s == cfish->nm - 2 || secnd_s == cfish->nm - 2) {
                int TT = cfish->nm - 1, TS = cfish->nm - 2;
                Real *PC = r[TT], *PF = r[TS];
                Real proj_w = 0, proj_h = 0, PT[3], PP[3];
                int d;
                int sign_w;
                int sign_h;
                Real dplane;
                for (d = 0; d < 3; d++) {
                  proj_w += (width[TS] * nor[TS][d]) * (p[d] - PF[d]);
                  proj_h += (height[TS] * bin[TS][d]) * (p[d] - PF[d]);
                }
                sign_w = proj_w > 0 ? 1 : -1;
                sign_h = proj_h > 0 ? 1 : -1;
                for (d = 0; d < 3; d++) {
                  PT[d] = r[TS][d] + sign_h * height[TS] * bin[TS][d];
                  PP[d] = r[TS][d] + sign_w * width[TS] * nor[TS][d];
                }
                dplane = dist_plane(PC, PT, PP, p, PF);
                defblock->sdf[sz][sy][sx] = dplane * fabs(dplane);
              } else if (ds2 >= radius_close + radius_second - corr) {
                Real grd2_ml = euler_dist_sq(p, r[close_s]);
                Real sign = grd2_ml > radius_close ? -1 : 1;
                defblock->sdf[sz][sy][sx] = sign * dist1;
              } else {
                Real Rsq = (radius_close + radius_second - corr + ds2) *
                           (radius_close + radius_second + corr + ds2) / 4 / ds2;
                Real max_ax = radius_close > radius_second ? radius_close : radius_second;
                Real d = sqrt((Rsq - max_ax) / ds2);
                Real *big = radius_close > radius_second ? center_close : center_second;
                Real *small = radius_close > radius_second ? center_second : center_close;
                Real x_midl[3];
                int k;
                Real grd2_core;
                Real sign;
                for (k = 0; k < 3; k++)
                  x_midl[k] = big[k] + (big[k] - small[k]) * d;
                grd2_core = euler_dist_sq(p, x_midl);
                sign = grd2_core > Rsq ? -1 : 1;
                defblock->sdf[sz][sy][sx] = sign * dist1;
              }
            }
      }
    }
  }
}
static void geom_end(struct ObstacleBlock *defblock) {
  int iz;
  int iy;
  int ix;
  for (iz = 0; iz < BS + 2; iz++)
    for (iy = 0; iy < BS + 2; iy++)
      for (ix = 0; ix < BS + 2; ix++) {
        if (iz < BS && iy < BS && ix < BS) {
          if (defblock->chi[iz][iy][ix] > DBL_EPSILON) {
            Real normfac = 1.0 / defblock->chi[iz][iy][ix];
            defblock->udef[iz][iy][ix][0] *= normfac;
            defblock->udef[iz][iy][ix][1] *= normfac;
            defblock->udef[iz][iy][ix][2] *= normfac;
          }
        }
        defblock->sdf[iz][iy][ix] = defblock->sdf[iz][iy][ix] >= 0 ? sqrt(defblock->sdf[iz][iy][ix])
                                                                   : -sqrt(-defblock->sdf[iz][iy][ix]);
      }
}
static void geom_blk(struct Frame *fr, Real h, Real ox, Real oy, Real oz, struct ObstacleBlock *o,
                     struct Segment **segs, int nseg) {
  Real *sdf;
  int i;
  memset(o->chi, 0, sizeof o->chi);
  memset(o->udef, 0, sizeof o->udef);
  sdf = &o->sdf[0][0][0];
  for (i = 0; i < (BS + 2) * (BS + 2) * (BS + 2); i++)
    sdf[i] = -1.;
  geom_in(fr, h, ox, oy, oz, o, segs, nseg);
  geom_surf(fr, h, ox, oy, oz, o, segs, nseg);
  geom_end(o);
}
static struct ObstacleBlock *oblock(struct Fish *f, long long i) {
  int k = f->slot[i];
  return k < 0 ? NULL : &f->pool[k];
}
static void fish_free_blk(struct Fish *f) {
  free(f->pool);
  free(f->slot);
  f->pool = NULL;
  f->slot = NULL;
}
struct Geom {
  struct Segment *segs;
  int *myblk, *seg_start, *seg_idx;
  int nmyblk;
  struct Frame fr;
};
static void fish_geom(struct Fish *f, struct Geom *g) {
  struct Midline *m = &f->m;
  int nm;
  int Nsegments;
  struct Segment *segs;
  int i;
  long long i2;
  int *myblk, *seg_start, *seg_idx;
  int nmyblk, nseg_idx, cap_blk, cap_seg;
  int d;
  Real fb[3][2];
  mid_step(m, sta.time);
  mid_lin(m);
  mid_ang(m, sta.dt);
  nm = m->nm;
  Nsegments = ceil((nm - 1.) / 8);
  segs = emalloc(Nsegments * sizeof *segs);
  for (i = 0; i < Nsegments; ++i) {
    int nextidx = (i + 1) * (nm - 1) / Nsegments;
    int idx = i * (nm - 1) / Nsegments;
    Real bbox[3][2] = {{1e9, -1e9}, {1e9, -1e9}, {1e9, -1e9}};
    int ss;
    for (ss = idx; ss <= nextidx; ++ss) {
      int d;
      for (d = 0; d < 3; d++) {
        Real bnd[4] = {m->r[ss][d] + m->nor[ss][d] * m->width[ss], m->r[ss][d] - m->nor[ss][d] * m->width[ss],
                       m->r[ss][d] + m->bin[ss][d] * m->height[ss],
                       m->r[ss][d] - m->bin[ss][d] * m->height[ss]};
        Real mx = max4(bnd[0], bnd[1], bnd[2], bnd[3]);
        Real mn = min4(bnd[0], bnd[1], bnd[2], bnd[3]);
        bbox[d][0] = mn < bbox[d][0] ? mn : bbox[d][0];
        bbox[d][1] = mx > bbox[d][1] ? mx : bbox[d][1];
      }
    }
    seg_prepare(&segs[i], idx, nextidx, bbox, sim.hmin);
    seg_to_frame(&segs[i], f->position, f->quaternion);
  }
  fish_free_blk(f);
  f->slot = emalloc(sta.nblk * sizeof *f->slot);
  cap_blk = 64;
  cap_seg = 512;
  myblk = emalloc(cap_blk * sizeof *myblk);
  seg_start = emalloc((cap_blk + 1) * sizeof *seg_start);
  seg_idx = emalloc(cap_seg * sizeof *seg_idx);
  nmyblk = 0;
  nseg_idx = 0;
  for (d = 0; d < 3; d++) {
    fb[d][0] = segs[0].box_lab[d][0];
    fb[d][1] = segs[0].box_lab[d][1];
    for (i = 1; i < Nsegments; ++i) {
      if (segs[i].box_lab[d][0] < fb[d][0])
        fb[d][0] = segs[i].box_lab[d][0];
      if (segs[i].box_lab[d][1] > fb[d][1])
        fb[d][1] = segs[i].box_lab[d][1];
    }
  }
  for (i2 = 0; i2 < sta.nblk; ++i2) {
    struct Blk *b = &sta.blk[i2];
    Real MINP[3], MAXP[3];
    int s;
    int out = 0;
    blk_pos(b, 0, 0, 0, MINP);
    blk_pos(b, BS - 1, BS - 1, BS - 1, MAXP);
    f->slot[i2] = -1;
    for (d = 0; d < 3; d++) {
      Real w = (MAXP[d] - MINP[d]) / 2 + segs[0].safe_distance, c = (MAXP[d] + MINP[d]) / 2;
      Real lo = fb[d][0] > c - w ? fb[d][0] : c - w, hi = fb[d][1] < c + w ? fb[d][1] : c + w;
      if (hi - lo < 0)
        out = 1;
    }
    if (out)
      continue;
    for (s = 0; s < Nsegments; ++s)
      if (seg_intersects(&segs[s], MINP, MAXP)) {
        if (f->slot[i2] < 0) {
          if (nmyblk == cap_blk) {
            cap_blk *= 2;
            myblk = erealloc(myblk, cap_blk * sizeof *myblk);
            seg_start = erealloc(seg_start, (cap_blk + 1) * sizeof *seg_start);
          }
          f->slot[i2] = nmyblk;
          myblk[nmyblk] = i2;
          seg_start[nmyblk] = nseg_idx;
          nmyblk++;
        }
        if (nseg_idx == cap_seg) {
          cap_seg *= 2;
          seg_idx = erealloc(seg_idx, cap_seg * sizeof *seg_idx);
        }
        seg_idx[nseg_idx++] = s;
      }
  }
  seg_start[nmyblk] = nseg_idx;
  f->pool = emalloc(nmyblk * sizeof *f->pool);
  frame_init(&g->fr, f);
  g->segs = segs;
  g->myblk = myblk;
  g->seg_start = seg_start;
  g->seg_idx = seg_idx;
  g->nmyblk = nmyblk;
}
static void geom_job(struct Fish *f, struct Geom *g, int j) {
  int n = g->seg_start[j + 1] - g->seg_start[j];
  struct Segment **S = emalloc(n * sizeof *S);
  int k;
  struct Blk *b;
  for (k = 0; k < n; k++)
    S[k] = &g->segs[g->seg_idx[g->seg_start[j] + k]];
  b = &sta.blk[g->myblk[j]];
  geom_blk(&g->fr, b->h, b->origin[0], b->origin[1], b->origin[2], &f->pool[j], S, n);
  free(S);
}
static void geom_free(struct Geom *g) {
  free(g->segs);
  free(g->myblk);
  free(g->seg_start);
  free(g->seg_idx);
}
static void clip(Real fmax, Real dfmax, Real dt, int zero, Real fcandidate, Real dfcandidate, Real *f,
                 Real *df) {
  if (zero) {
    *f = 0;
    *df = 0;
  } else if (fabs(dfcandidate) > dfmax) {
    *df = dfcandidate > 0 ? +dfmax : -dfmax;
    *f = *f + dt * *df;
  } else if (fabs(fcandidate) < fmax) {
    *f = fcandidate;
    *df = dfcandidate;
  } else {
    *f = fcandidate > 0 ? fmax : -fmax;
    *df = 0;
  }
}
static void fish_create(struct Fish *f, struct Geom *g) {
  struct Midline *mid = &f->m;
  int nm = mid->nm;
  Real *q = f->quaternion;
  Real R[3][3], dv[3];
  int d;
  Real dn;
  Real xx2;
  Real pitch;
  Real roll;
  Real yaw;
  int roll_is_small;
  int yaw_is_small;
  quat_to_rotation(q, R);
  for (d = 0; d < 3; d++)
    dv[d] = mid->r[0][d] - mid->r[nm / 2][d];
  dn = pow(dot3(dv, dv), 0.5) + 1e-21;
  xx2 = R[2][0] * (dv[0] / dn) + R[2][1] * (dv[1] / dn) + R[2][2] * (dv[2] / dn);
  xx2 = xx2 > 1 ? 1 : (xx2 < -1 ? -1 : xx2);
  pitch = asin(xx2);
  roll = atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
  yaw = atan2(2.0 * (q[3] * q[0] + q[1] * q[2]), -1.0 + 2.0 * (q[0] * q[0] + q[1] * q[1]));
  roll_is_small = fabs(roll) < M_PI / 9.;
  yaw_is_small = fabs(yaw) < M_PI / 9.;
  if (f->correct_pos) {
    Real y;
    Real ytgt;
    Real dy;
    Real sign_y;
    Real yaw_tgt;
    Real dphi;
    Real b;
    Real dbdt;
    mid->alpha = 1.0 + (f->position[0] - f->orig[0]) / f->length;
    mid->dalpha = (f->vel[0] + sta.uinf[0]) / f->length;
    if (roll_is_small == 0) {
      mid->alpha = 1.0;
      mid->dalpha = 0.0;
    } else if (mid->alpha < 0.9) {
      mid->alpha = 0.9;
      mid->dalpha = 0.0;
    } else if (mid->alpha > 1.1) {
      mid->alpha = 1.1;
      mid->dalpha = 0.0;
    }
    y = f->abs_pos[1];
    ytgt = f->orig[1];
    dy = (ytgt - y) / f->length;
    sign_y = dy > 0 ? 1 : -1;
    yaw_tgt = 0;
    dphi = yaw - yaw_tgt;
    b = roll_is_small ? f->wyp * sign_y * dy * dphi : 0;
    dbdt = sta.step > 1 ? (b - mid->beta) / sta.dt : 0;
    clip(1.0, 5.0, sta.dt, 0, b, dbdt, &mid->beta, &mid->dbeta);
  }
  if (f->correct_z) {
    Real pitch_tgt = 0;
    Real dphi = pitch - pitch_tgt;
    Real z = f->abs_pos[2];
    Real ztgt = f->orig[2];
    Real dz = (ztgt - z) / f->length;
    Real sign_z = dz > 0 ? 1 : -1;
    Real g = (roll_is_small && yaw_is_small) ? -f->wzp * dphi * dz * sign_z : 0.0;
    Real dgdt = sta.step > 1 ? (g - mid->gamma) / sta.dt : 0.0;
    Real gmax = 0.10 / f->length;
    Real d_rdtmax = 0.1 * f->length / mid->period;
    Real dgdtmax = fabs(gmax * gmax * d_rdtmax);
    clip(gmax, dgdtmax, sta.dt, 0, g, dgdt, &mid->gamma, &mid->dgamma);
  }
  fish_geom(f, g);
}
static void fish_update(struct Fish *f) {
  Real *position = f->position, *abs_pos = f->abs_pos, *quaternion = f->quaternion;
  Real *omega = f->omega, *vel = f->vel;
  Real dqdt[4];
  quat_rate(quaternion, omega, dqdt);
  if (sta.step < STEP_2ND) {
    int d;
    for (d = 0; d < 3; d++) {
      f->old_position[d] = position[d];
      f->old_abs_pos[d] = abs_pos[d];
    }
    for (d = 0; d < 4; d++)
      f->old_quaternion[d] = quaternion[d];
    position[0] += sta.dt * (vel[0] + sta.uinf[0]);
    position[1] += sta.dt * (vel[1] + sta.uinf[1]);
    position[2] += sta.dt * (vel[2] + sta.uinf[2]);
    abs_pos[0] += sta.dt * vel[0];
    abs_pos[1] += sta.dt * vel[1];
    abs_pos[2] += sta.dt * vel[2];
    quaternion[0] += sta.dt * dqdt[0];
    quaternion[1] += sta.dt * dqdt[1];
    quaternion[2] += sta.dt * dqdt[2];
    quaternion[3] += sta.dt * dqdt[3];
  } else {
    Real aux = 1.0 / sta.coef_u[0];
    Real temp[10] = {position[0], position[1],   position[2],   abs_pos[0],    abs_pos[1],
                     abs_pos[2],  quaternion[0], quaternion[1], quaternion[2], quaternion[3]};
    int d;
    for (d = 0; d < 3; d++)
      position[d] = aux * (sta.dt * (vel[d] + sta.uinf[d]) +
                           (-sta.coef_u[1] * position[d] - sta.coef_u[2] * f->old_position[d]));
    for (d = 0; d < 3; d++)
      abs_pos[d] =
          aux * (sta.dt * (vel[d]) + (-sta.coef_u[1] * abs_pos[d] - sta.coef_u[2] * f->old_abs_pos[d]));
    for (d = 0; d < 4; d++)
      quaternion[d] = aux * (sta.dt * (dqdt[d]) +
                             (-sta.coef_u[1] * quaternion[d] - sta.coef_u[2] * f->old_quaternion[d]));
    for (d = 0; d < 3; d++) {
      f->old_position[d] = temp[d];
      f->old_abs_pos[d] = temp[3 + d];
    }
    for (d = 0; d < 4; d++)
      f->old_quaternion[d] = temp[6 + d];
  }
  quat_normalize(quaternion);
}
static void sta_uinf(void) {
  int n_sum[3] = {0, 0, 0};
  Real u_sum[3] = {0, 0, 0};
  int i;
  int d;
  for (i = 0; i < sim.nfish; i++) {
    struct Fish *f = &sta.fish[i];
    int d;
    for (d = 0; d < 3; d++)
      if (f->fix[d]) {
        n_sum[d] += 1;
        u_sum[d] -= f->vel[d];
      }
  }
  for (d = 0; d < 3; d++)
    if (n_sum[d] > 0)
      u_sum[d] = u_sum[d] / n_sum[d];
  for (d = 0; d < 3; d++)
    sta.uinf[d] = u_sum[d];
}
static void geom_chi(long long i) {
  struct Blk *blk = &sta.blk[i];
  Real *b = fld(i, F_CHI);
  Real h = blk->h, inv2h = .5 / h, vol = h * h * h;
  int gp = 1;
  int obst_id;
  for (obst_id = 0; obst_id < sim.nfish; obst_id++) {
    struct ObstacleBlock *o = oblock(&sta.fish[obst_id], i);
    int z;
    int y;
    int x;
    if (o == NULL)
      continue;
    o->com[0] = 0;
    o->com[1] = 0;
    o->com[2] = 0;
    o->mass = 0;
    for (z = 0; z < BS; ++z)
      for (y = 0; y < BS; ++y)
        for (x = 0; x < BS; ++x) {
          Real p[3];
          int j;
          if (o->sdf[z + 1][y + 1][x + 1] > +gp * h || o->sdf[z + 1][y + 1][x + 1] < -gp * h) {
            o->chi[z][y][x] = o->sdf[z + 1][y + 1][x + 1] > 0 ? 1 : 0;
          } else {
            Real grad_u[3], grad_i[3];
            int a;
            Real grad_usq;
            for (a = 0; a < 3; a++) {
              Real d_p = o->sdf[z + 1 + (a == 2)][y + 1 + (a == 1)][x + 1 + (a == 0)];
              Real d_m = o->sdf[z + 1 - (a == 2)][y + 1 - (a == 1)][x + 1 - (a == 0)];
              grad_u[a] = inv2h * (d_p - d_m);
              grad_i[a] = inv2h * ((d_p > 0.0 ? d_p : 0.0) - (d_m > 0.0 ? d_m : 0.0));
            }
            grad_usq = dot3(grad_u, grad_u) + DBL_EPSILON;
            o->chi[z][y][x] = dot3(grad_i, grad_u) / grad_usq;
          }
          blk_pos(blk, x, y, z, p);
          j = z * BS * BS + y * BS + x;
          b[j] = o->chi[z][y][x] < b[j] ? b[j] : o->chi[z][y][x];
          o->com[0] += o->chi[z][y][x] * vol * p[0];
          o->com[1] += o->chi[z][y][x] * vol * p[1];
          o->com[2] += o->chi[z][y][x] * vol * p[2];
          o->mass += o->chi[z][y][x] * vol;
        }
  }
}
static void invert_sym(Real J[6], Real inv[6]) {
  Real jdet = J[0] * (J[1] * J[2] - J[5] * J[5]) + J[3] * (J[4] * J[5] - J[2] * J[3]) +
              J[4] * (J[3] * J[5] - J[1] * J[4]);
  if (fabs(jdet) <= DBL_MIN) {
    int q;
    for (q = 0; q < 6; q++)
      inv[q] = 0;
  } else {
    inv[0] = (J[1] * J[2] - J[5] * J[5]) / jdet;
    inv[1] = (J[0] * J[2] - J[4] * J[4]) / jdet;
    inv[2] = (J[0] * J[1] - J[3] * J[3]) / jdet;
    inv[3] = (J[4] * J[5] - J[2] * J[3]) / jdet;
    inv[4] = (J[3] * J[5] - J[1] * J[4]) / jdet;
    inv[5] = (J[3] * J[4] - J[0] * J[5]) / jdet;
  }
}
static void fish_com(void) {
  int k;
  for (k = 0; k < sim.nfish; k++) {
    struct Fish *f = &sta.fish[k];
    Real com[4] = {0.0, 0.0, 0.0, 0.0};
    long long i;
    for (i = 0; i < sta.nblk; i++) {
      struct ObstacleBlock *o = oblock(f, i);
      if (o == NULL)
        continue;
      com[0] += o->mass;
      com[1] += o->com[0];
      com[2] += o->com[1];
      com[3] += o->com[2];
    }
    MPI_Allreduce(MPI_IN_PLACE, com, 4, MPI_Real, MPI_SUM, sim.comm);
    if (com[0] <= 0)
      continue;
    f->com[0] = com[1] / com[0];
    f->com[1] = com[2] / com[0];
    f->com[2] = com[3] / com[0];
  }
}
static void fish_udef_blk(long long i) {
  struct Blk *b = &sta.blk[i];
  int k;
  for (k = 0; k < sim.nfish; k++) {
    struct Fish *f = &sta.fish[k];
    struct ObstacleBlock *o = oblock(f, i);
    Real *CM;
    Real *M;
    int q;
    int z;
    int y;
    int x;
    if (o == NULL)
      continue;
    CM = f->com;
    M = o->mom;
    for (q = 0; q < 13; q++)
      M[q] = 0;
    for (z = 0; z < BS; ++z)
      for (y = 0; y < BS; ++y)
        for (x = 0; x < BS; ++x) {
          Real p[3];
          Real dv, X;
          Real *U;
          Real px_u[3];
          int d;
          if (o->chi[z][y][x] <= 0)
            continue;
          blk_pos(b, x, y, z, p);
          dv = b->h * b->h * b->h;
          X = o->chi[z][y][x];
          U = o->udef[z][y][x];
          p[0] -= CM[0];
          p[1] -= CM[1];
          p[2] -= CM[2];
          cross3(px_u, p, U);
          M[M_V] += X * dv;
          for (d = 0; d < 3; d++) {
            int b = d == 0 ? 1 : 0, c = d == 2 ? 1 : 2;
            M[M_FX + d] += X * U[d] * dv;
            M[M_TX + d] += X * px_u[d] * dv;
            M[M_J0 + d] += X * (p[b] * p[b] + p[c] * p[c]) * dv;
          }
          M[M_J3] -= X * p[0] * p[1] * dv;
          M[M_J4] -= X * p[0] * p[2] * dv;
          M[M_J5] -= X * p[1] * p[2] * dv;
        }
  }
}
static void fish_udef_mom(void) {
  int k;
  for (k = 0; k < sim.nfish; k++) {
    struct Fish *f = &sta.fish[k];
    Real M[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    long long i;
    Real AM[3];
    Real J[6];
    Real jinv[6];
    int q;
    for (i = 0; i < sta.nblk; i++) {
      struct ObstacleBlock *o = oblock(f, i);
      int q;
      if (o == NULL)
        continue;
      for (q = 0; q < 13; q++)
        M[q] += o->mom[q];
    }
    MPI_Allreduce(MPI_IN_PLACE, M, 13, MPI_Real, MPI_SUM, sim.comm);
    if (M[0] <= 0) {
      int d;
      int q;
      f->mass = 0;
      for (d = 0; d < 3; d++)
        f->vel_corr[d] = f->omega_corr[d] = 0;
      for (q = 0; q < 6; q++)
        f->J[q] = 0;
      continue;
    }
    AM[0] = M[4];
    AM[1] = M[5];
    AM[2] = M[6];
    J[0] = M[7];
    J[1] = M[8];
    J[2] = M[9];
    J[3] = M[10];
    J[4] = M[11];
    J[5] = M[12];
    invert_sym(J, jinv);
    f->mass = M[0];
    f->vel_corr[0] = M[1] / M[0];
    f->vel_corr[1] = M[2] / M[0];
    f->vel_corr[2] = M[3] / M[0];
    for (q = 0; q < 6; q++)
      f->J[q] = J[q];
    f->omega_corr[0] = jinv[0] * AM[0] + jinv[3] * AM[1] + jinv[4] * AM[2];
    f->omega_corr[1] = jinv[3] * AM[0] + jinv[1] * AM[1] + jinv[5] * AM[2];
    f->omega_corr[2] = jinv[4] * AM[0] + jinv[5] * AM[1] + jinv[2] * AM[2];
  }
}
static void fish_udef_fix(void) {
  int k;
  for (k = 0; k < sim.nfish; k++) {
    struct Fish *f = &sta.fish[k];
    Real *av = f->omega_corr;
    Real *tv = f->vel_corr;
    Real *CM = f->com;
    long long i;
#pragma omp parallel for schedule(dynamic, 1)
    for (i = 0; i < sta.nblk; i++) {
      struct ObstacleBlock *o = oblock(f, i);
      struct Blk *b;
      int z;
      int y;
      int x;
      if (o == NULL)
        continue;
      b = &sta.blk[i];
      for (z = 0; z < BS; ++z)
        for (y = 0; y < BS; ++y)
          for (x = 0; x < BS; ++x) {
            Real p[3];
            Real rot[3];
            int d;
            blk_pos(b, x, y, z, p);
            p[0] -= CM[0];
            p[1] -= CM[1];
            p[2] -= CM[2];
            cross3(rot, av, p);
            for (d = 0; d < 3; d++)
              o->udef[z][y][x][d] -= tv[d] + rot[d];
          }
    }
  }
}
static void fish_build(void) {
  long long i;
  int k;
  struct Geom *g;
  int *job_fish, *job_start;
  int njob;
  if (sim.nfish == 0)
    return;
  if (sta.mesh_changed == 0 && sim.static_obst)
    return;
  sta.mesh_changed = 0;
#pragma omp parallel for schedule(static)
  for (i = 0; i < sta.nblk; ++i) {
    memset(fld(i, F_CHI), 0, BS3 * sizeof(Real));
  }
  sta_uinf();
  for (k = 0; k < sim.nfish; k++)
    fish_update(&sta.fish[k]);
  g = emalloc(sim.nfish * sizeof *g);
  job_start = emalloc((sim.nfish + 1) * sizeof *job_start);
#pragma omp parallel for schedule(dynamic, 1)
  for (k = 0; k < sim.nfish; k++)
    fish_create(&sta.fish[k], &g[k]);
  job_start[0] = 0;
  for (k = 0; k < sim.nfish; k++)
    job_start[k + 1] = job_start[k] + g[k].nmyblk;
  njob = job_start[sim.nfish];
  job_fish = emalloc(njob * sizeof *job_fish);
  for (k = 0; k < sim.nfish; k++)
    for (i = job_start[k]; i < job_start[k + 1]; i++)
      job_fish[i] = k;
#pragma omp parallel for schedule(dynamic, 1)
  for (i = 0; i < njob; i++)
    geom_job(&sta.fish[job_fish[i]], &g[job_fish[i]], (int)(i - job_start[job_fish[i]]));
  for (k = 0; k < sim.nfish; k++)
    geom_free(&g[k]);
  free(g);
  free(job_fish);
  free(job_start);
#pragma omp parallel for
  for (i = 0; i < sta.nblk; ++i) {
    geom_chi(i);
  }
  fish_com();
#pragma omp parallel for schedule(dynamic, 1)
  for (i = 0; i < sta.nblk; ++i) {
    fish_udef_blk(i);
  }
  fish_udef_mom();
  fish_udef_fix();
}
enum { TAG_KEEP = 0, TAG_REF = 1, TAG_COMP = -1 };
struct Node {
  long long key;
  int pos;
  int local;
  int halo;
  signed char state;
  signed char used;
};
static struct {
  struct Node *tab;
  long long cap, n;
  long long *level_base;
} nodes;
static long long node_key(int level, long long Z) { return nodes.level_base[level] + Z; }
static struct Node *node_find(long long key, int create);
static void nodes_rehash(void) {
  struct Node *old = nodes.tab;
  long long oldcap = nodes.cap;
  long long i;
  nodes.cap = oldcap ? 2 * oldcap : 1 << 16;
  nodes.tab = ecalloc(nodes.cap, sizeof *nodes.tab);
  nodes.n = 0;
  for (i = 0; i < oldcap; i++)
    if (old[i].used) {
      struct Node *nn = node_find(old[i].key, 1);
      *nn = old[i];
    }
  free(old);
}
static struct Node *node_find(long long key, int create) {
  unsigned long long h;
  long long i;
  if (nodes.cap == 0) {
    if (!create)
      return NULL;
    nodes_rehash();
  } else if (create && 2 * (nodes.n + 1) > nodes.cap)
    nodes_rehash();
  h = (unsigned long long)key * 0x9E3779B97F4A7C15ULL;
  i = (long long)(h >> 20) & (nodes.cap - 1);
  for (;;) {
    struct Node *nd = &nodes.tab[i];
    if (!nd->used) {
      if (!create)
        return NULL;
      nd->used = 1;
      nd->key = key;
      nd->pos = -3;
      nd->local = -1;
      nd->halo = -1;
      nd->state = TAG_KEEP;
      nodes.n++;
      return nd;
    }
    if (nd->key == key)
      return nd;
    i = (i + 1) & (nodes.cap - 1);
  }
}
static struct Node *node(int level, long long Z) { return node_find(node_key(level, Z), 1); }
static struct Node *node_get(int level, long long Z) { return node_find(node_key(level, Z), 0); }
static void nodes_reset(void) {
  free(nodes.tab);
  nodes.tab = NULL;
  nodes.cap = nodes.n = 0;
  nodes_rehash();
}
static void nodes_init(void) {
  int m;
  nodes.level_base = emalloc(sim.level_max * sizeof(long long));
  for (m = 0; m < sim.level_max; m++) {
    long long TwoPower = 1 << m;
    long long Ntot = (long long)sim.bpdx * sim.bpdy * sim.bpdz * TwoPower * TwoPower * TwoPower;
    nodes.level_base[m] = m == 0 ? Ntot : nodes.level_base[m - 1] + Ntot;
  }
  nodes.tab = NULL;
  nodes_reset();
}
static int nblocks_dim(int d, int level) {
  int b = d == 0 ? sim.bpdx : d == 1 ? sim.bpdy : sim.bpdz;
  return b * (1 << level);
}
static long long zforward(int level, int i, int j, int k) {
  int NX = sim.bpdx, NY = sim.bpdy, NZ = sim.bpdz;
  int TwoPower = 1 << level;
  int ix = (i + TwoPower * NX) % (NX * TwoPower);
  int iy = (j + TwoPower * NY) % (NY * TwoPower);
  int iz = (k + TwoPower * NZ) % (NZ * TwoPower);
  return sfc_forward(level, ix, iy, iz);
}
static long long znei(struct Blk *b, int i, int j, int k) {
  return zforward(b->level, b->ix + i, b->iy + j, b->iz + k);
}
static long long zparent(struct Blk *b) {
  return b->level == 0 ? 0 : zforward(b->level - 1, b->ix / 2, b->iy / 2, b->iz / 2);
}
static long long zchild(struct Blk *b, int i, int j, int k) {
  return sfc_forward(b->level + 1, 2 * b->ix + i, 2 * b->iy + j, 2 * b->iz + k);
}
static long long encode(int level, long long Z, int ix, int iy, int iz) {
  int lmax = sim.level_max;
  long long retval = 0;
  int l;
  int i, j, k;
  for (l = level; l >= 0; l--) {
    long long Zp = sfc_forward(l, ix, iy, iz);
    retval += Zp;
    ix /= 2;
    iy /= 2;
    iz /= 2;
  }
  sfc_inverse(Z, level, &i, &j, &k);
  ix = 2 * i;
  iy = 2 * j;
  iz = 2 * k;
  for (l = level + 1; l < lmax; l++) {
    long long Zc = sfc_forward(l, ix, iy, iz);
    int ix1, iy1, iz1;
    Zc -= Zc % 8;
    retval += Zc;
    sfc_inverse(Zc, l, &ix1, &iy1, &iz1);
    ix = 2 * ix1;
    iy = 2 * iy1;
    iz = 2 * iz1;
  }
  retval += level;
  return retval;
}
static long long blk_id(struct Blk *b) { return encode(b->level, b->Z, b->ix, b->iy, b->iz); }
static long long fld_cap;
static long long blk_alloc(int level, long long Z) {
  long long i;
  struct Node *nd;
  if (sta.nblk == fld_cap) {
    Real *nf;
    fld_cap = fld_cap ? 2 * fld_cap : 64;
    sta.blk = erealloc(sta.blk, fld_cap * sizeof *sta.blk);
    nf = emalloc(fld_cap * BLK_S * sizeof(Real));
    if (sta.fld) {
      memcpy(nf, sta.fld, sta.nblk * BLK_S * sizeof(Real));
      free(sta.fld);
    }
    sta.fld = nf;
  }
  i = sta.nblk++;
  blk_fill(&sta.blk[i], level, Z);
  nd = node(level, Z);
  nd->pos = sim.rank;
  nd->local = i;
  return i;
}
static void blk_remove(long long i) {
  struct Node *nd = node(sta.blk[i].level, sta.blk[i].Z);
  long long last;
  if (nd->local == i)
    nd->local = -1;
  last = sta.nblk - 1;
  if (i != last) {
    sta.blk[i] = sta.blk[last];
    memcpy(BLK(i), BLK(last), BLK_S * sizeof(Real));
    node(sta.blk[i].level, sta.blk[i].Z)->local = i;
  }
  sta.nblk--;
}
static int blk_cmp(const void *a, const void *b) {
  long long ia = *(long long *)a, ib = *(long long *)b;
  return (ia > ib) - (ia < ib);
}
static void blk_sort(void) {
  long long *keys = emalloc(2 * sta.nblk * sizeof *keys);
  long long i;
  struct Blk *nb;
  Real *nf;
  for (i = 0; i < sta.nblk; i++) {
    keys[2 * i] = blk_id(&sta.blk[i]);
    keys[2 * i + 1] = i;
  }
  qsort(keys, sta.nblk, 2 * sizeof *keys, blk_cmp);
  nb = emalloc(fld_cap * sizeof *nb);
  nf = emalloc(fld_cap * BLK_S * sizeof(Real));
  for (i = 0; i < sta.nblk; i++) {
    long long j = keys[2 * i + 1];
    nb[i] = sta.blk[j];
    memcpy(nf + i * BLK_S, BLK(j), BLK_S * sizeof(Real));
    node(nb[i].level, nb[i].Z)->local = i;
  }
  free(sta.blk);
  free(sta.fld);
  sta.blk = nb;
  sta.fld = nf;
  free(keys);
}
static int imax(int a, int b) { return a > b ? a : b; }
static void nei_code(int icode, int code[3]) {
  code[0] = icode % 3 - 1;
  code[1] = (icode / 3) % 3 - 1;
  code[2] = icode / 9 - 1;
}
static int nei_outside(struct Blk *b, int code[3]) {
  int ix[3] = {b->ix, b->iy, b->iz};
  int d;
  for (d = 0; d < 3; d++) {
    int n = nblocks_dim(d, b->level);
    if (code[d] == (ix[d] == 0 ? -1 : 1) && (ix[d] == 0 || ix[d] == n - 1))
      return 1;
  }
  return 0;
}
static int nei_next(struct Blk *b, int *icode, int code[3]) {
  while (++*icode < 27) {
    if (*icode == 13)
      continue;
    nei_code(*icode, code);
    if (!nei_outside(b, code))
      return 1;
  }
  return 0;
}
static int nei_bstep(int code[3]) {
  int t = abs(code[0]) + abs(code[1]) + abs(code[2]);
  return t == 2 ? 3 : t == 3 ? 4 : 1;
}
static long long nei_fine(struct Blk *b, int code[3], int B) {
  int a = abs(code[0]) == 1 ? B % 2 : B / 2;
  int ci = 2 * b->ix + imax(code[0], 0) + code[0] + (B % 2) * imax(0, 1 - abs(code[0]));
  int cj = 2 * b->iy + imax(code[1], 0) + code[1] + a * imax(0, 1 - abs(code[1]));
  int ck = 2 * b->iz + imax(code[2], 0) + code[2] + (B / 2) * imax(0, 1 - abs(code[2]));
  return zforward(b->level + 1, ci, cj, ck);
}
static long long nei_coarse(struct Blk *b, int code[3]) {
  int NX = nblocks_dim(0, b->level), NY = nblocks_dim(1, b->level), NZ = nblocks_dim(2, b->level);
  int idx[3] = {(b->ix + code[0] + NX) % NX, (b->iy + code[1] + NY) % NY, (b->iz + code[2] + NZ) % NZ};
  return zforward(b->level - 1, idx[0] / 2, idx[1] / 2, idx[2] / 2);
}
struct Xch {
  int *scnt, *sdsp, *rcnt, *rdsp;
};
static void xch_free(struct Xch *x) {
  free(x->scnt);
  free(x->sdsp);
  free(x->rcnt);
  free(x->rdsp);
  x->scnt = x->sdsp = x->rcnt = x->rdsp = NULL;
}
static void xch_reset(struct Xch *x) {
  xch_free(x);
  x->scnt = ecalloc(sim.size, sizeof *x->scnt);
  x->sdsp = ecalloc(sim.size, sizeof *x->sdsp);
  x->rcnt = ecalloc(sim.size, sizeof *x->rcnt);
  x->rdsp = ecalloc(sim.size, sizeof *x->rdsp);
}
static void xch_dsp(struct Xch *x) {
  int r;
  for (r = 1; r < sim.size; r++) {
    x->sdsp[r] = x->sdsp[r - 1] + x->scnt[r - 1];
    x->rdsp[r] = x->rdsp[r - 1] + x->rcnt[r - 1];
  }
}
static int xch_nsend(struct Xch *x) { return x->sdsp[sim.size - 1] + x->scnt[sim.size - 1]; }
static int xch_nrecv(struct Xch *x) { return x->rdsp[sim.size - 1] + x->rcnt[sim.size - 1]; }
static void xch_exec(struct Xch *x, void *sbuf, void *rbuf, int m, MPI_Datatype base) {
  MPI_Datatype t;
  MPI_Type_contiguous(m, base, &t);
  MPI_Type_commit(&t);
  MPI_Alltoallv(sbuf, x->scnt, x->sdsp, t, rbuf, x->rcnt, x->rdsp, t, sim.comm);
  MPI_Type_free(&t);
}
#define HALO_BASE (1LL << 40)
enum { NEI_MAX = 26 * 4 };
struct Halo {
  int nhalo, nsend, f0, nc;
  struct Xch x;
  long long *send, *rkey;
  Real *buf, *sbuf;
  signed char *sst, *rst;
};
static struct Halo halo;
static long long *slot;
#define SLOT(i) (slot ? slot[i] : (i))
static long long blk_avail(int level, long long Z) {
  struct Node *nd = node_get(level, Z);
  if (nd == NULL)
    return -1;
  if (nd->pos == sim.rank)
    return nd->local;
  if (nd->halo >= 0)
    return HALO_BASE + nd->halo;
  return -1;
}
static void tree_sync(void) {
  int *cnt = emalloc(sim.size * sizeof *cnt);
  int *dsp = emalloc(sim.size * sizeof *dsp);
  int n = 2 * (int)sta.nblk;
  int total;
  int r;
  long long *mine;
  long long i;
  long long *all;
  int j;
  MPI_Allgather(&n, 1, MPI_INT, cnt, 1, MPI_INT, sim.comm);
  total = 0;
  for (r = 0; r < sim.size; r++) {
    dsp[r] = total;
    total += cnt[r];
  }
  mine = emalloc(n * sizeof *mine);
  for (i = 0; i < sta.nblk; i++) {
    mine[2 * i] = sta.blk[i].level;
    mine[2 * i + 1] = sta.blk[i].Z;
  }
  all = emalloc(total * sizeof *all);
  MPI_Allgatherv(mine, n, MPI_LONG_LONG, all, cnt, dsp, MPI_LONG_LONG, sim.comm);
  nodes_reset();
  for (r = 0; r < sim.size; r++)
    for (j = dsp[r]; j < dsp[r] + cnt[r]; j += 2) {
      int level = (int)all[j];
      long long Z = all[j + 1];
      struct Blk b;
      int k;
      int jj;
      int i;
      blk_fill(&b, level, Z);
      node(level, Z)->pos = r;
      if (level < sim.level_max - 1)
        for (k = 0; k < 2; k++)
          for (jj = 0; jj < 2; jj++)
            for (i = 0; i < 2; i++)
              node(level + 1, zchild(&b, i, jj, k))->pos = -2;
      if (level > 0)
        node(level - 1, zparent(&b))->pos = -1;
    }
  for (i = 0; i < sta.nblk; i++)
    node(sta.blk[i].level, sta.blk[i].Z)->local = i;
  free(cnt);
  free(dsp);
  free(mine);
  free(all);
}
static int blk_remote_neighbors(struct Blk *b, long long *keys, int *ranks, int *lv) {
  int n = 0;
  int icode = -1, code[3];
  while (nei_next(b, &icode, code)) {
    long long zn = znei(b, code[0], code[1], code[2]);
    struct Node *nd = node(b->level, zn);
    if (nd->pos >= 0) {
      if (nd->pos != sim.rank) {
        keys[n] = node_key(b->level, zn);
        lv[n] = b->level;
        ranks[n++] = nd->pos;
      }
    } else if (nd->pos == -2) {
      long long zp = nei_coarse(b, code);
      struct Node *np = node(b->level - 1, zp);
      if (np->pos != sim.rank) {
        keys[n] = node_key(b->level - 1, zp);
        lv[n] = b->level - 1;
        ranks[n++] = np->pos;
      }
    } else if (nd->pos == -1) {
      int B;
      for (B = 0; B <= 3; B += nei_bstep(code)) {
        long long zf = nei_fine(b, code, B);
        struct Node *nf = node(b->level + 1, zf);
        if (nf->pos >= 0 && nf->pos != sim.rank) {
          keys[n] = node_key(b->level + 1, zf);
          lv[n] = b->level + 1;
          ranks[n++] = nf->pos;
        }
      }
    }
  }
  return n;
}
static int pair_cmp(const void *a, const void *b) {
  long long *x = (long long *)a, *y = (long long *)b;
  if (x[0] != y[0])
    return (x[0] > y[0]) - (x[0] < y[0]);
  return (x[1] > y[1]) - (x[1] < y[1]);
}
static long long pair_unique(long long *p, long long n) {
  long long m = 0;
  long long i;
  for (i = 0; i < n; i++)
    if (i == 0 || p[2 * i] != p[2 * (m - 1)] || p[2 * i + 1] != p[2 * (m - 1) + 1]) {
      p[2 * m] = p[2 * i];
      p[2 * m + 1] = p[2 * i + 1];
      m++;
    }
  return m;
}
static int halo_lvl = -1;
static void halo_build(void) {
  int k;
  long long cap;
  long long *sp;
  long long *rp;
  long long ns, nr;
  long long keys[NEI_MAX];
  int ranks[NEI_MAX];
  int lv[NEI_MAX];
  long long i;
  long long k2;
  for (k = 0; k < halo.nhalo; k++) {
    struct Node *nd = node_find(halo.rkey[k], 0);
    if (nd)
      nd->halo = -1;
  }
  free(halo.send);
  free(halo.rkey);
  xch_reset(&halo.x);
  cap = 2 * NEI_MAX * sta.nblk;
  sp = emalloc(cap * sizeof *sp);
  rp = emalloc(cap * sizeof *rp);
  ns = 0;
  nr = 0;
  for (i = 0; i < sta.nblk; i++) {
    int n = blk_remote_neighbors(&sta.blk[i], keys, ranks, lv);
    long long mykey = node_key(sta.blk[i].level, sta.blk[i].Z);
    int j;
    for (j = 0; j < n; j++) {
      if (halo_lvl < 0 || lv[j] == halo_lvl) {
        sp[2 * ns] = ranks[j];
        sp[2 * ns + 1] = mykey;
        ns++;
      }
      if (halo_lvl < 0 || sta.blk[i].level == halo_lvl) {
        rp[2 * nr] = ranks[j];
        rp[2 * nr + 1] = keys[j];
        nr++;
      }
    }
  }
  qsort(sp, ns, 2 * sizeof *sp, pair_cmp);
  qsort(rp, nr, 2 * sizeof *rp, pair_cmp);
  ns = pair_unique(sp, ns);
  nr = pair_unique(rp, nr);
  halo.nsend = (int)ns;
  halo.nhalo = (int)nr;
  halo.send = emalloc(ns * sizeof *halo.send);
  halo.rkey = emalloc(nr * sizeof *halo.rkey);
  free(halo.buf);
  free(halo.sbuf);
  free(halo.sst);
  free(halo.rst);
  halo.buf = emalloc(nr * NC_MAX * BS3 * sizeof(Real));
  halo.sbuf = emalloc(ns * NC_MAX * BS3 * sizeof(Real));
  halo.sst = emalloc(ns);
  halo.rst = emalloc(nr);
  for (k2 = 0; k2 < ns; k2++) {
    halo.x.scnt[sp[2 * k2]]++;
    halo.send[k2] = node_find(sp[2 * k2 + 1], 1)->local;
  }
  for (k2 = 0; k2 < nr; k2++) {
    halo.x.rcnt[rp[2 * k2]]++;
    halo.rkey[k2] = rp[2 * k2 + 1];
    node_find(rp[2 * k2 + 1], 1)->halo = (int)k2;
  }
  xch_dsp(&halo.x);
  free(sp);
  free(rp);
}
static Real *view_in, *view_out;
static void halo_sync(int f, int nc) {
  long long m = (long long)nc * BS3;
  int k;
  halo.f0 = f;
  halo.nc = nc;
#pragma omp parallel for
  for (k = 0; k < halo.nsend; k++) {
    memcpy(halo.sbuf + k * m, view_in ? view_in + SLOT(halo.send[k]) * BS3 : fld(halo.send[k], f),
           m * sizeof(Real));
  }
  xch_exec(&halo.x, halo.sbuf, halo.buf, (int)m, MPI_Real);
}
static void states_sync(void) {
  int k;
  for (k = 0; k < halo.nsend; k++) {
    struct Blk *b = &sta.blk[halo.send[k]];
    halo.sst[k] = node(b->level, b->Z)->state;
  }
  xch_exec(&halo.x, halo.sst, halo.rst, 1, MPI_SIGNED_CHAR);
  for (k = 0; k < halo.nhalo; k++)
    node_find(halo.rkey[k], 1)->state = halo.rst[k];
}
static Real *fld_ptr(long long i, int f, int c) {
  if (i < HALO_BASE)
    return view_in ? view_in + (SLOT(i) + c) * BS3 : BLK(i) + (f + c) * BS3;
  return halo.buf + ((i - HALO_BASE) * halo.nc + (f - halo.f0) + c) * BS3;
}
#define CELL(i, f, c, x, y, z) (fld_ptr(i, f, c)[((z) * BS + (y)) * BS + (x)])
static Real *out_ptr(long long i, int f, int c) {
  return view_out ? view_out + (SLOT(i) + c) * BS3 : BLK(i) + (f + c) * BS3;
}
#define OUT(i, f, c, x, y, z) (out_ptr(i, f, c)[((z) * BS + (y)) * BS + (x)])
static struct {
  int nface;
  int *idx;
  Real *data;
  long long nsend, nrecv, nrface;
  long long *send, *recv, *rface;
  struct Xch x;
  Real *sbuf, *rbuf;
} fc;
enum { FC_NC = 3, FC_Q = 16 * FC_NC };
static Real *fc_face(long long i, int face, int c) {
  int slot = fc.idx[6 * i + face];
  return slot < 0 ? NULL : fc.data + ((long long)slot * 3 + c) * BS * BS;
}
static int fc_cmp(const void *a, const void *b) {
  long long *x = (long long *)a, *y = (long long *)b;
  int q;
  for (q = 0; q < 3; q++)
    if (x[q] != y[q])
      return (x[q] > y[q]) - (x[q] < y[q]);
  return 0;
}
static void fc_prepare(void) {
  static int fcode[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  long long q;
  long long i;
  long long k;
  free(fc.idx);
  free(fc.data);
  free(fc.send);
  free(fc.recv);
  free(fc.rface);
  free(fc.sbuf);
  free(fc.rbuf);
  xch_reset(&fc.x);
  fc.idx = emalloc(6 * sta.nblk * sizeof *fc.idx);
  for (q = 0; q < 6 * sta.nblk; q++)
    fc.idx[q] = -1;
  fc.nface = 0;
  fc.send = emalloc(6 * sta.nblk * 4 * sizeof *fc.send);
  fc.recv = emalloc(24 * sta.nblk * 6 * sizeof *fc.recv);
  fc.rface = emalloc(6 * sta.nblk * 2 * sizeof *fc.rface);
  fc.nsend = fc.nrecv = fc.nrface = 0;
  for (i = 0; i < sta.nblk; i++) {
    struct Blk *b = &sta.blk[i];
    int f;
    for (f = 0; f < 6; f++) {
      int *code = fcode[f];
      int d;
      int face;
      struct Node *nd;
      if (nei_outside(b, code))
        continue;
      d = f / 2;
      face = 2 * d + (code[d] > 0);
      nd = node(b->level, znei(b, code[0], code[1], code[2]));
      if (nd->pos >= 0)
        continue;
      fc.idx[6 * i + face] = fc.nface++;
      if (nd->pos == -2) {
        long long zp = nei_coarse(b, code);
        long long *e = fc.send + 4 * fc.nsend++;
        e[0] = node(b->level - 1, zp)->pos;
        e[1] = node_key(b->level, b->Z);
        e[2] = face;
        e[3] = i;
      } else if (nd->pos == -1) {
        int B;
        fc.rface[2 * fc.nrface] = i;
        fc.rface[2 * fc.nrface + 1] = face;
        fc.nrface++;
        for (B = 0; B <= 3; B++) {
          long long zc = nei_fine(b, code, B);
          long long *e = fc.recv + 6 * fc.nrecv++;
          e[0] = node(b->level + 1, zc)->pos;
          e[1] = node_key(b->level + 1, zc);
          e[2] = face ^ 1;
          e[3] = i;
          e[4] = face;
          e[5] = B;
        }
      }
    }
  }
  qsort(fc.send, fc.nsend, 4 * sizeof *fc.send, fc_cmp);
  qsort(fc.recv, fc.nrecv, 6 * sizeof *fc.recv, fc_cmp);
  for (k = 0; k < fc.nsend; k++)
    fc.x.scnt[fc.send[4 * k]]++;
  for (k = 0; k < fc.nrecv; k++)
    fc.x.rcnt[fc.recv[6 * k]]++;
  xch_dsp(&fc.x);
  fc.data = ecalloc(fc.nface * 3 * BS * BS, sizeof(Real));
  fc.sbuf = emalloc(fc.nsend * FC_Q * sizeof(Real));
  fc.rbuf = emalloc(fc.nrecv * FC_Q * sizeof(Real));
}
static void fc_fill(int f, int nc) {
  int Q = 16 * nc;
  Real *sbuf = fc.sbuf, *rbuf = fc.rbuf;
  long long k;
  int d;
  if (nc > FC_NC)
    fatal("fc_fill nc=%d exceeds FC_NC=%d", nc, FC_NC);
#pragma omp parallel for
  for (k = 0; k < fc.nsend; k++) {
    long long *e = fc.send + 4 * k;
    int c;
    for (c = 0; c < nc; c++) {
      Real *F = fc_face(e[3], (int)e[2], c);
      int i1;
      int i2;
      for (i1 = 0; i1 < BS; i1 += 2)
        for (i2 = 0; i2 < BS; i2 += 2)
          sbuf[k * Q + c * 16 + (i1 / 2) * 4 + i2 / 2] =
              ((F[i2 + i1 * BS] + F[i2 + 1 + i1 * BS]) + (F[i2 + (i1 + 1) * BS] + F[i2 + 1 + (i1 + 1) * BS]));
    }
  }
  xch_exec(&fc.x, sbuf, rbuf, Q, MPI_Real);
#pragma omp parallel for
  for (k = 0; k < fc.nrecv; k++) {
    long long *e = fc.recv + 6 * k;
    int B = (int)e[5];
    int base = B == 1 ? BS / 2 : B == 2 ? (BS / 2) * BS : B == 3 ? BS / 2 + (BS / 2) * BS : 0;
    int c;
    for (c = 0; c < nc; c++) {
      Real *F = fc_face(e[3], (int)e[4], c);
      int i1;
      int i2;
      for (i1 = 0; i1 < BS; i1 += 2)
        for (i2 = 0; i2 < BS; i2 += 2)
          F[base + i2 / 2 + (i1 / 2) * BS] += rbuf[k * Q + c * 16 + (i1 / 2) * 4 + i2 / 2];
    }
  }
  for (d = 0; d < 3; d++) {
#pragma omp parallel for
    for (k = 0; k < fc.nrface; k++) {
      long long i = fc.rface[2 * k];
      int face = (int)fc.rface[2 * k + 1];
      int j;
      int c;
      if (face / 2 != d)
        continue;
      j = (face % 2 == 0) ? 0 : BS - 1;
      for (c = 0; c < nc; c++) {
        Real *F = fc_face(i, face, c);
        int i1;
        int i2;
        for (i1 = 0; i1 < BS; i1++)
          for (i2 = 0; i2 < BS; i2++) {
            if (d == 0)
              OUT(i, f, c, j, i2, i1) += F[i2 + i1 * BS];
            else if (d == 1)
              OUT(i, f, c, i2, j, i1) += F[i2 + i1 * BS];
            else
              OUT(i, f, c, i2, i1, j) += F[i2 + i1 * BS];
            F[i2 + i1 * BS] = 0;
          }
      }
    }
  }
  memset(fc.data, 0, (size_t)fc.nface * 3 * BS * BS * sizeof(Real));
}
static void mg_build(void);
static void mesh_init(void) {
  int level;
  long long aux;
  long long total;
  long long my_blocks;
  long long n_start;
  long long Z;
  sfc_init(sim.bpdx, sim.bpdy, sim.bpdz, sim.level_max);
  nodes_init();
  level = sim.level_start;
  aux = 1 << level;
  total = (long long)sim.bpdx * sim.bpdy * sim.bpdz * aux * aux * aux;
  my_blocks = total / sim.size;
  if ((long long)sim.rank < total % sim.size)
    my_blocks++;
  n_start = sim.rank * (total / sim.size);
  if (total % sim.size > 0) {
    if ((long long)sim.rank < total % sim.size)
      n_start += sim.rank;
    else
      n_start += total % sim.size;
  }
  sta.nblk = 0;
  fld_cap = 0;
  sta.blk = NULL;
  sta.fld = NULL;
  for (Z = n_start; Z < n_start + my_blocks; Z++)
    blk_alloc(level, Z);
  memset(sta.fld, 0, sta.nblk * BLK_S * sizeof(Real));
  blk_sort();
  tree_sync();
  halo_build();
  fc_prepare();
  mg_build();
}
enum { OP_COPY, OP_AVG8, OP_INTERP, OP_FD, OP_BC };
struct Op {
  int32_t type, bd, dst, bs, src, n, a[8];
};
struct LabTab {
  int32_t magic, ss, te, nops;
  int32_t same_copy[27][2], same_cfill[27][2], fine[27][2], coarse[27][8][2], interp[27][2], own_avg[2],
      bc[6][2][2];
  int32_t relevant[27][64];
  struct Op *ops;
};
static struct LabTab lab_tab[4];
static void lab_tables(void) {
  static int cfg[4][2] = {{1, 1}, {1, 0}, {2, 1}, {3, 0}};
  int k;
  for (k = 0; k < 4; k++) {
    struct LabTab *T = &lab_tab[k];
    char name[64];
    FILE *fp;
    size_t hdr;
    snprintf(name, sizeof name, "lab_ss%d_t%d.bin", cfg[k][0], cfg[k][1]);
    fp = fopen(name, "rb");
    if (fp == NULL)
      fatal("cannot open %s (run gen_table.py)", name);
    hdr = offsetof(struct LabTab, ops);
    if (fread(T, 1, hdr, fp) != hdr || T->magic != 0x4C414231 || T->ss != cfg[k][0] || T->te != cfg[k][1])
      fatal("bad table %s", name);
    T->ops = emalloc(T->nops * sizeof *T->ops);
    if (fread(T->ops, sizeof *T->ops, T->nops, fp) != (size_t)T->nops)
      fatal("short read from %s", name);
    fclose(fp);
  }
}
struct Lab {
  int f, nc, vflip;
  int ss[3], se[3];
  int cn[3], cc[3];
  struct LabTab *tab;
  Real cache[CN_MAX * CN_MAX * CN_MAX * NC_MAX];
  Real coarse[CC_MAX * CC_MAX * CC_MAX * NC_MAX];
};
static double d_coef_plus[9] = {-0.09375, 0.4375,   0.15625, 0.15625, -0.5625,
                                0.90625,  -0.09375, 0.4375,  0.15625};
static double d_coef_minus[9] = {0.15625, -0.5625, 0.90625, -0.09375, 0.4375,
                                 0.15625, 0.15625, 0.4375,  -0.09375};
#define LAB(l, ix, iy, iz) ((l)->cache + (((iz) * (l)->cn[1] + (iy)) * (l)->cn[0] + (ix)) * (l)->nc)
static void lab_init(struct Lab *l, int f, int nc, int ss, int te, int vflip) {
  int k;
  int d;
  if (ss > SS_MAX || nc > NC_MAX)
    fatal("lab ss=%d nc=%d exceeds SS_MAX=%d NC_MAX=%d", ss, nc, SS_MAX, NC_MAX);
  l->f = f;
  l->nc = nc;
  l->vflip = vflip;
  l->tab = NULL;
  for (k = 0; k < 4; k++)
    if (lab_tab[k].ss == ss && lab_tab[k].te == te)
      l->tab = &lab_tab[k];
  if (l->tab == NULL)
    fatal("no table for ss=%d te=%d", ss, te);
  for (d = 0; d < 3; d++) {
    int offset;
    int e;
    l->ss[d] = -ss;
    l->se[d] = ss + 1;
    l->cn[d] = BS + 2 * ss;
    offset = (l->ss[d] - 1) / 2 - 1;
    e = l->se[d] / 2 + 2;
    l->cc[d] = BS / 2 + e - offset - 1;
  }
}
static void lab_exec(struct Lab *l, int32_t sec[2], Real **nb) {
  struct Op *ops = l->tab->ops + sec[0];
  int nc = l->nc;
  int cc = l->cc[0];
  Real *buf[2] = {l->cache, l->coarse};
  Real R[8 * F_N];
  int k;
  for (k = 0; k < sec[1]; k++) {
    struct Op *o = &ops[k];
    int32_t *a = o->a;
    switch (o->type) {
    case OP_COPY: {
      Real *d = buf[o->bd] + (long long)o->dst * nc;
      Real *s = nb[o->bs - 2] + o->src;
      int i;
      int c;
      for (i = 0; i < o->n; i++)
        for (c = 0; c < nc; c++)
          d[i * nc + c] = s[c * BS3 + i];
      break;
    }
    case OP_AVG8: {
      Real *d = buf[o->bd] + (long long)o->dst * nc;
      if (o->bs >= 2) {
        Real *s = nb[o->bs - 2];
        int c;
        for (c = 0; c < nc; c++)
          d[c] = 0.125 * (s[c * BS3 + a[0]] + s[c * BS3 + a[1]] + s[c * BS3 + a[2]] + s[c * BS3 + a[3]] +
                          s[c * BS3 + a[4]] + s[c * BS3 + a[5]] + s[c * BS3 + a[6]] + s[c * BS3 + a[7]]);
      } else {
        Real *s = buf[o->bs];
        int c;
        for (c = 0; c < nc; c++)
          d[c] = 0.125 * (s[a[0] * nc + c] + s[a[1] * nc + c] + s[a[2] * nc + c] + s[a[3] * nc + c] +
                          s[a[4] * nc + c] + s[a[5] * nc + c] + s[a[6] * nc + c] + s[a[7] * nc + c]);
      }
      break;
    }
    case OP_INTERP: {
      int c;
      int r;
#define C3(I, J, K) (l->coarse[(o->src + ((K) * cc + (J)) * cc + (I)) * nc + c])
      for (c = 0; c < nc; c++) {
        Real dudx = 0.125 * (C3(2, 1, 1) - C3(0, 1, 1));
        Real dudy = 0.125 * (C3(1, 2, 1) - C3(1, 0, 1));
        Real dudz = 0.125 * (C3(1, 1, 2) - C3(1, 1, 0));
        Real dudxdy = 0.015625 * (C3(0, 0, 1) + C3(2, 2, 1) - C3(2, 0, 1) - C3(0, 2, 1));
        Real dudxdz = 0.015625 * (C3(0, 1, 0) + C3(2, 1, 2) - C3(2, 1, 0) - C3(0, 1, 2));
        Real dudydz = 0.015625 * (C3(1, 0, 0) + C3(1, 2, 2) - C3(1, 2, 0) - C3(1, 0, 2));
        Real lap = C3(1, 1, 1) + 0.03125 * (C3(0, 1, 1) + C3(2, 1, 1) + C3(1, 0, 1) + C3(1, 2, 1) +
                                            C3(1, 1, 0) + C3(1, 1, 2) + (-6.0) * C3(1, 1, 1));
        int q;
        for (q = 0; q < 8; q++) {
          Real sx = q & 1 ? 1.0 : -1.0, sy = q & 2 ? 1.0 : -1.0, sz = q & 4 ? 1.0 : -1.0;
          R[q * nc + c] = lap + sx * dudx + sy * dudy + sz * dudz + sx * sy * dudxdy + sx * sz * dudxdz +
                          sy * sz * dudydz;
        }
      }
#undef C3
      for (r = 0; r < 8; r++)
        if (a[r] >= 0)
          memcpy(l->cache + (long long)a[r] * nc, R + r * nc, nc * sizeof(Real));
      break;
    }
    case OP_FD: {
      static int tang[3][2] = {{1, 2}, {0, 2}, {0, 1}};
      int stride[3] = {1, cc, cc * cc};
      int t1 = tang[a[0]][0], t2 = tang[a[0]][1];
      int s1 = stride[t1], s2 = stride[t2];
      double d1 = 0.25 * (2 * ((a[3] >> t1) & 1) - 1);
      double d2 = 0.25 * (2 * ((a[3] >> t2) & 1) - 1);
      double *c1 = d1 > 0 ? d_coef_plus : d_coef_minus;
      double *c2 = d2 > 0 ? d_coef_plus : d_coef_minus;
      Real *dst = l->cache + (long long)o->dst * nc;
      Real *bb = l->cache + (long long)a[4] * nc;
      Real *cq = l->cache + (long long)a[5] * nc;
      int c;
      for (c = 0; c < nc; c++) {
#define CO(OFF) (l->coarse[(o->src + (OFF)) * nc + c])
        Real x1_d, x2_d;
        double mixed_coef = 1.0;
        int P1, M1, P2, M2;
        Real mixed;
        Real v;
        int first;
        if (a[1] == 0) {
          x1_d = (c1[6] * CO(-s1) + c1[8] * CO(s1)) + c1[7] * CO(0);
          P1 = s1;
          M1 = -s1;
          mixed_coef *= 0.5;
        } else if (a[1] == 1) {
          x1_d = (c1[0] * CO(2 * s1) + c1[1] * CO(s1)) + c1[2] * CO(0);
          P1 = s1;
          M1 = 0;
        } else {
          x1_d = (c1[3] * CO(-2 * s1) + c1[4] * CO(-s1)) + c1[5] * CO(0);
          P1 = 0;
          M1 = -s1;
        }
        if (a[2] == 0) {
          x2_d = (c2[6] * CO(-s2) + c2[8] * CO(s2)) + c2[7] * CO(0);
          P2 = s2;
          M2 = -s2;
          mixed_coef *= 0.5;
        } else if (a[2] == 1) {
          x2_d = (c2[0] * CO(2 * s2) + c2[1] * CO(s2)) + c2[2] * CO(0);
          P2 = s2;
          M2 = 0;
        } else {
          x2_d = (c2[3] * CO(-2 * s2) + c2[4] * CO(-s2)) + c2[5] * CO(0);
          P2 = 0;
          M2 = -s2;
        }
        mixed = mixed_coef * d1 * d2 * ((CO(M1 + M2) + CO(P1 + P2)) - (CO(P1 + M2) + CO(M1 + P2)));
#undef CO
        v = (x1_d + x2_d) + mixed;
        first = a[6] == 1 ? a[7] == 0 : a[7] == 1;
        v = first ? (1.0 / 15.0) * (8.0 * v + (10.0 * bb[c] - 3.0 * cq[c]))
                  : (1.0 / 15.0) * (24.0 * v + (-15.0 * bb[c] + 6 * cq[c]));
        dst[c] = v;
      }
      break;
    }
    case OP_BC: {
      Real *d = buf[o->bd] + (long long)o->dst * nc;
      Real *s = buf[o->bs] + (long long)o->src * nc;
      memcpy(d, s, nc * sizeof(Real));
      if (l->vflip >= 0) {
        int c;
        for (c = l->vflip + a[0]; c < nc; c += 3)
          d[c] = (-1.) * s[c];
      }
      break;
    }
    }
  }
}
static Real *lab_block(struct Lab *l, int level, long long Z) {
  long long i = blk_avail(level, Z);
  if (i < 0)
    fatal("block level %d Z %lld not available", level, Z);
  return fld_ptr(i, l->f, 0);
}
static void lab_load(struct Lab *l, long long ib) {
  struct Blk *b = &sta.blk[ib];
  struct LabTab *T = l->tab;
  int c;
  int wall[6];
  int w;
  int par;
  int same[26], coarse[26], nsame, ncoarse;
  unsigned coarse_mask;
  int icode, code[3];
  int coarsened;
  int k;
  int f;
  for (c = 0; c < l->nc; c++) {
    Real *src = fld_ptr(ib, l->f, c);
    int iz;
    int iy;
    int ix;
    for (iz = 0; iz < BS; iz++)
      for (iy = 0; iy < BS; iy++)
        for (ix = 0; ix < BS; ix++)
          LAB(l, ix - l->ss[0], iy - l->ss[1], iz - l->ss[2])[c] = src[(iz * BS + iy) * BS + ix];
  }
  wall[0] = b->ix == 0;
  wall[1] = b->ix == nblocks_dim(0, b->level) - 1;
  wall[2] = b->iy == 0;
  wall[3] = b->iy == nblocks_dim(1, b->level) - 1;
  wall[4] = b->iz == 0;
  wall[5] = b->iz == nblocks_dim(2, b->level) - 1;
  w = (wall[0] | wall[1] << 1) | (wall[2] | wall[3] << 1) << 2 | (wall[4] | wall[5] << 1) << 4;
  par = (b->ix & 1) | (b->iy & 1) << 1 | (b->iz & 1) << 2;
  nsame = 0;
  ncoarse = 0;
  coarse_mask = 0;
  icode = -1;
  while (nei_next(b, &icode, code)) {
    long long zn = znei(b, code[0], code[1], code[2]);
    struct Node *nd = node_get(b->level, zn);
    if (nd == NULL)
      fatal("lab_load: no tree entry for neighbor level %d Z %lld of level %d Z %lld", b->level, zn, b->level,
            b->Z);
    if (nd->pos >= 0) {
      Real *nb;
      same[nsame++] = icode;
      nb = lab_block(l, b->level, zn);
      lab_exec(l, T->same_copy[icode], &nb);
    } else if (nd->pos == -2) {
      Real *nb;
      coarse[ncoarse++] = icode;
      coarse_mask |= 1u << icode;
      nb = lab_block(l, b->level - 1, nei_coarse(b, code));
      lab_exec(l, T->coarse[icode][par], &nb);
    } else if (nd->pos == -1) {
      Real *nb[4] = {NULL, NULL, NULL, NULL};
      int B;
      for (B = 0; B <= 3; B += nei_bstep(code))
        nb[B] = lab_block(l, b->level + 1, nei_fine(b, code, B));
      lab_exec(l, T->fine[icode], nb);
    }
  }
  coarsened = 0;
  if (ncoarse > 0)
    for (k = 0; k < nsame; k++) {
      int icode = same[k];
      if (T->relevant[icode][w] & coarse_mask) {
        Real *nb;
        nei_code(icode, code);
        nb = lab_block(l, b->level, znei(b, code[0], code[1], code[2]));
        lab_exec(l, T->same_cfill[icode], &nb);
        coarsened = 1;
      }
    }
  if (coarsened)
    lab_exec(l, T->own_avg, NULL);
  for (f = 0; f < 6; f++)
    if (wall[f])
      lab_exec(l, T->bc[f][1], NULL);
  for (k = 0; k < ncoarse; k++)
    lab_exec(l, T->interp[coarse[k]], NULL);
  for (f = 0; f < 6; f++)
    if (wall[f])
      lab_exec(l, T->bc[f][0], NULL);
}
struct Stencil {
  int f, nc, ss, te, vflip, out, outc;
  void (*kernel)(struct Lab *, long long);
};
static void stencil_run(struct Stencil *st, long long *list, long long n) {
  halo_sync(st->f, st->nc);
#pragma omp parallel
  {
    struct Lab l;
    long long k;
    lab_init(&l, st->f, st->nc, st->ss, st->te, st->vflip);
#pragma omp for schedule(dynamic, 1)
    for (k = 0; k < n; k++) {
      long long i = list ? list[k] : k;
      lab_load(&l, i);
      st->kernel(&l, i);
    }
  }
  if (st->outc > 0)
    fc_fill(st->out, st->outc);
}
static void stencil_apply(struct Stencil *st) { stencil_run(st, NULL, sta.nblk); }
static void k_gradchi(struct Lab *l, long long ib) {
  struct Blk *b = &sta.blk[ib];
  Real *TMP0 = fld(ib, F_TMP), *TMP1 = TMP0 + BS3, *TMP2 = TMP1 + BS3;
  int done = 0;
  int offset = (b->level == sim.level_max - 1) ? 2 : 1;
  int z;
  int y;
  int x;
  for (z = -offset; z < BS + offset; ++z)
    for (y = -offset; y < BS + offset; ++y)
      for (x = -offset; x < BS + offset; ++x) {
        Real *v;
        if (done)
          break;
        v = LAB(l, x - l->ss[0], y - l->ss[1], z - l->ss[2]);
        v[0] = (Real)1.0 < v[0] ? (Real)1.0 : v[0];
        v[0] = v[0] < (Real)0.0 ? (Real)0.0 : v[0];
        if (v[0] > 0.00001 && v[0] < 0.9) {
          int q;
          for (q = 0; q < 8; q++)
            TMP0[IDX(BS / 2 - 1 + (q & 1), BS / 2 - 1 + ((q >> 1) & 1), BS / 2 - 1 + (q >> 2))] = 1e10;
          done = 1;
          break;
        } else if (v[0] > 0.9 && z >= 0 && z < BS && y >= 0 && y < BS && x >= 0 && x < BS) {
          int j = (z * BS + y) * BS + x;
          TMP0[j] = 0.0;
          TMP1[j] = 0.0;
          TMP2[j] = 0.0;
        }
      }
}
static struct Stencil st_gradchi = {F_CHI, 1, 2, 1, -1, 0, 0, k_gradchi};
static int mesh_tag_blk(long long i) {
  Real *u0 = fld(i, F_TMP), *u1 = u0 + BS3, *u2 = u1 + BS3;
  double Linf = 0.0;
  int j;
  for (j = 0; j < BS3; j++) {
    double m = fabs(sqrt(u0[j] * u0[j] + u1[j] * u1[j] + u2[j] * u2[j]));
    Linf = m > Linf ? m : Linf;
  }
  if (Linf > sim.rtol)
    return TAG_REF;
  else if (Linf < sim.ctol)
    return TAG_COMP;
  return TAG_KEEP;
}
static void set_state(long long i, int st) { node_get(sta.blk[i].level, sta.blk[i].Z)->state = st; }
static int get_state(long long i) { return node_get(sta.blk[i].level, sta.blk[i].Z)->state; }
static int mesh_tag(void) {
  int changed = 0;
  long long i;
#pragma omp parallel for reduction(| : changed)
  for (i = 0; i < sta.nblk; i++) {
    int st = mesh_tag_blk(i);
    int level = sta.blk[i].level;
    if ((st == TAG_REF && level == sim.level_max - 1) || (st == TAG_COMP && level == 0))
      st = TAG_KEEP;
    set_state(i, st);
    if (st != TAG_KEEP)
      changed = 1;
  }
  return changed;
}
static void mesh_fix(void) {
  int level_min = 0;
  int level_max = sim.level_max;
  long long j;
  int m;
  for (j = 0; j < sta.nblk; j++) {
    int st = get_state(j);
    if ((st == TAG_REF && sta.blk[j].level == level_max - 1) ||
        (st == TAG_COMP && sta.blk[j].level == level_min))
      set_state(j, TAG_KEEP);
  }
  for (m = level_max - 1; m >= level_min; m--) {
    long long j;
    for (j = 0; j < sta.nblk; j++) {
      struct Blk *b = &sta.blk[j];
      if (b->level == m && get_state(j) != TAG_REF && b->level != level_max - 1) {
        int icode = -1, code[3];
        while (nei_next(b, &icode, code)) {
          if (get_state(j) == TAG_REF)
            break;
          if (node(m, znei(b, code[0], code[1], code[2]))->pos == -1) {
            int B;
            if (get_state(j) == TAG_COMP)
              set_state(j, TAG_KEEP);
            for (B = 0; B <= 3; B += nei_bstep(code))
              if (node(m + 1, nei_fine(b, code, B))->state == TAG_REF) {
                set_state(j, TAG_REF);
                break;
              }
          }
        }
      }
    }
    states_sync();
    if (m == level_min)
      break;
    for (j = 0; j < sta.nblk; j++) {
      struct Blk *b = &sta.blk[j];
      if (b->level == m && get_state(j) == TAG_COMP) {
        int icode = -1, code[3];
        while (nei_next(b, &icode, code)) {
          struct Node *nd = node(m, znei(b, code[0], code[1], code[2]));
          if (nd->pos >= 0 && nd->state == TAG_REF) {
            set_state(j, TAG_KEEP);
            break;
          }
        }
      }
    }
  }
  for (j = 0; j < sta.nblk; j++) {
    struct Blk *b = &sta.blk[j];
    int m = b->level;
    int found = 0;
    int ii;
    int jj;
    int kk;
    for (ii = 2 * (b->ix / 2); ii <= 2 * (b->ix / 2) + 1 && !found; ii++)
      for (jj = 2 * (b->iy / 2); jj <= 2 * (b->iy / 2) + 1 && !found; jj++)
        for (kk = 2 * (b->iz / 2); kk <= 2 * (b->iz / 2) + 1; kk++) {
          struct Node *nd = node(m, zforward(m, ii, jj, kk));
          if (nd->pos < 0 || nd->state != TAG_COMP) {
            found = 1;
            if (get_state(j) == TAG_COMP)
              set_state(j, TAG_KEEP);
            break;
          }
        }
    if (found)
      for (ii = 2 * (b->ix / 2); ii <= 2 * (b->ix / 2) + 1; ii++)
        for (jj = 2 * (b->iy / 2); jj <= 2 * (b->iy / 2) + 1; jj++)
          for (kk = 2 * (b->iz / 2); kk <= 2 * (b->iz / 2) + 1; kk++) {
            struct Node *nd = node(m, zforward(m, ii, jj, kk));
            if (nd->pos >= 0 && nd->state == TAG_COMP)
              nd->state = TAG_KEEP;
          }
  }
}
static void mesh_refine(struct Lab *l, long long B[8], int f, int nc) {
  int nx = BS, ny = BS, nz = BS;
  int offset_x[2] = {0, nx / 2}, offset_y[2] = {0, ny / 2}, offset_z[2] = {0, nz / 2};
  int K;
  int J;
  int I;
  for (K = 0; K < 2; K++)
    for (J = 0; J < 2; J++)
      for (I = 0; I < 2; I++) {
        long long ib = B[K * 4 + J * 2 + I];
        int k;
        int j;
        int i;
        for (k = 0; k < nz; k += 2)
          for (j = 0; j < ny; j += 2)
            for (i = 0; i < nx; i += 2) {
              int x = i / 2 + offset_x[I];
              int y = j / 2 + offset_y[J];
              int z = k / 2 + offset_z[K];
              int c;
              for (c = 0; c < nc; c++) {
#define L(X, Y, Z) (LAB(l, (X) - l->ss[0], (Y) - l->ss[1], (Z) - l->ss[2])[c])
                Real dudx = 0.5 * (L(x + 1, y, z) - L(x - 1, y, z));
                Real dudy = 0.5 * (L(x, y + 1, z) - L(x, y - 1, z));
                Real dudz = 0.5 * (L(x, y, z + 1) - L(x, y, z - 1));
                Real dudx2 = (L(x + 1, y, z) + L(x - 1, y, z)) - 2.0 * L(x, y, z);
                Real dudy2 = (L(x, y + 1, z) + L(x, y - 1, z)) - 2.0 * L(x, y, z);
                Real dudz2 = (L(x, y, z + 1) + L(x, y, z - 1)) - 2.0 * L(x, y, z);
                Real dudxdy = 0.25 * ((L(x + 1, y + 1, z) + L(x - 1, y - 1, z)) -
                                      (L(x + 1, y - 1, z) + L(x - 1, y + 1, z)));
                Real dudxdz = 0.25 * ((L(x + 1, y, z + 1) + L(x - 1, y, z - 1)) -
                                      (L(x + 1, y, z - 1) + L(x - 1, y, z + 1)));
                Real dudydz = 0.25 * ((L(x, y + 1, z + 1) + L(x, y - 1, z - 1)) -
                                      (L(x, y + 1, z - 1) + L(x, y - 1, z + 1)));
                Real u = L(x, y, z);
                Real lap = 0.03125 * (dudx2 + dudy2 + dudz2);
                int q;
#undef L
                for (q = 0; q < 8; q++) {
                  Real sx = q & 1 ? 1.0 : -1.0, sy = q & 2 ? 1.0 : -1.0, sz = q & 4 ? 1.0 : -1.0;
                  CELL(ib, f, c, i + (q & 1), j + ((q >> 1) & 1), k + (q >> 2)) =
                      u + 0.25 * (sx * dudx + sy * dudy + sz * dudz) + lap +
                      0.0625 * (sx * sy * dudxdy + sx * sz * dudxdz + sy * sz * dudydz);
                }
              }
            }
      }
}
static void blk_pack(Real *dst, long long i) {
  dst[0] = sta.blk[i].level;
  dst[1] = (Real)sta.blk[i].Z;
  memcpy(dst + 2, BLK(i), BLK_S * sizeof(Real));
}
static void blk_unpack(Real *src) {
  int level = (int)src[0];
  long long Z = (long long)src[1];
  long long i = blk_alloc(level, Z);
  memcpy(BLK(i), src + 2, BLK_S * sizeof(Real));
}
static void blk_remove_key(int level, long long Z) {
  struct Node *nd = node(level, Z);
  if (nd->local >= 0)
    blk_remove(nd->local);
}
enum { PACK_S = BLK_S + 2 };
static void blk_migrate(int *dst) {
  struct Xch x = {NULL, NULL, NULL, NULL};
  int *fill = ecalloc(sim.size, sizeof *fill);
  long long i;
  long long ns, nr;
  int r;
  Real *sbuf;
  Real *rbuf;
  int k;
  long long k2;
  xch_reset(&x);
  for (i = 0; i < sta.nblk; i++)
    if (dst[i] >= 0)
      x.scnt[dst[i]]++;
  MPI_Alltoall(x.scnt, 1, MPI_INT, x.rcnt, 1, MPI_INT, sim.comm);
  xch_dsp(&x);
  ns = xch_nsend(&x);
  nr = xch_nrecv(&x);
  sbuf = emalloc(ns * PACK_S * sizeof(Real));
  rbuf = emalloc(nr * PACK_S * sizeof(Real));
  for (i = 0; i < sta.nblk; i++)
    if (dst[i] >= 0)
      blk_pack(sbuf + (long long)(x.sdsp[dst[i]] + fill[dst[i]]++) * PACK_S, i);
  xch_exec(&x, sbuf, rbuf, PACK_S, MPI_Real);
  for (r = 0; r < sim.size; r++)
    for (k = 0; k < x.scnt[r]; k++) {
      Real *p = sbuf + (long long)(x.sdsp[r] + k) * PACK_S;
      blk_remove_key((int)p[0], (long long)p[1]);
      node((int)p[0], (long long)p[1])->pos = r;
    }
  for (k2 = 0; k2 < nr; k2++)
    blk_unpack(rbuf + k2 * PACK_S);
  free(sbuf);
  free(rbuf);
  free(fill);
  xch_free(&x);
}
static void mesh_gather(void) {
  int *dst = emalloc(sta.nblk * sizeof *dst);
  long long i;
  for (i = 0; i < sta.nblk; i++) {
    struct Blk *b = &sta.blk[i];
    long long zb = zforward(b->level, 2 * (b->ix / 2), 2 * (b->iy / 2), 2 * (b->iz / 2));
    struct Node *base = node(b->level, zb);
    dst[i] = -1;
    if (base->pos >= 0 && base->state == TAG_COMP && b->Z != zb && base->pos != sim.rank)
      dst[i] = base->pos;
  }
  blk_migrate(dst);
  free(dst);
}
static int mesh_bal_all(long long *all_b) {
  int size = sim.size, rank = sim.rank;
  long long total_load;
  int r;
  long long *index_start;
  long long b1, b2;
  int *dst;
  long long i;
  long long front, back;
  int q;
  blk_sort();
  total_load = 0;
  for (r = 0; r < size; r++)
    total_load += all_b[r];
  index_start = emalloc(size * sizeof *index_start);
  index_start[0] = 0;
  for (r = 1; r < size; r++)
    index_start[r] = index_start[r - 1] + all_b[r - 1];
  b1 = index_start[rank];
  b2 = index_start[rank] + all_b[rank] - 1;
  dst = emalloc(sta.nblk * sizeof *dst);
  for (i = 0; i < sta.nblk; i++)
    dst[i] = -1;
  front = 0;
  back = 0;
  for (q = 0; q < size - 1; q++) {
    int r = q < rank ? q : size - 1 - (q - rank);
    long long other_load = total_load / size + (r < total_load % size);
    long long a1 = (total_load / size) * r + ((r < total_load % size) ? r : total_load % size);
    long long a2 = a1 + other_load - 1;
    long long c1 = a1 > b1 ? a1 : b1;
    long long c2 = a2 < b2 ? a2 : b2;
    long long k;
    for (k = 0; k < c2 - c1 + 1; k++)
      if (r < rank)
        dst[front++] = r;
      else
        dst[sta.nblk - 1 - back++] = r;
  }
  blk_migrate(dst);
  free(dst);
  free(index_start);
  return 1;
}
static int mesh_bal(long long *dist) {
  int size = sim.size, rank = sim.rank;
  int right;
  int left;
  int my_blocks;
  int right_blocks, left_blocks;
  int nu;
  int flux_left;
  int flux_right;
  int *dst;
  long long i;
  int k;
  int moved;
  {
    long long max_b = dist[0], min_b = dist[0];
    int r;
    double ratio;
    for (r = 0; r < size; r++) {
      max_b = dist[r] > max_b ? dist[r] : max_b;
      min_b = dist[r] < min_b ? dist[r] : min_b;
    }
    ratio = (double)max_b / min_b;
    if (ratio > 1.01 || min_b == 0)
      return mesh_bal_all(dist);
  }
  right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
  left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
  my_blocks = (int)sta.nblk;
  MPI_Sendrecv(&my_blocks, 1, MPI_INT, left, 0, &right_blocks, 1, MPI_INT, right, 0, sim.comm,
               MPI_STATUS_IGNORE);
  MPI_Sendrecv(&my_blocks, 1, MPI_INT, right, 0, &left_blocks, 1, MPI_INT, left, 0, sim.comm,
               MPI_STATUS_IGNORE);
  nu = 4;
  flux_left = (rank == 0) ? 0 : (my_blocks - left_blocks) / nu;
  flux_right = (rank == size - 1) ? 0 : (my_blocks - right_blocks) / nu;
  if (flux_right != 0 || flux_left != 0)
    blk_sort();
  dst = emalloc(sta.nblk * sizeof *dst);
  for (i = 0; i < sta.nblk; i++)
    dst[i] = -1;
  for (k = 0; k < flux_left; k++)
    dst[k] = left;
  for (k = 0; k < flux_right; k++)
    dst[my_blocks - k - 1] = right;
  moved = flux_left != 0 || flux_right != 0;
  blk_migrate(dst);
  free(dst);
  MPI_Allreduce(MPI_IN_PLACE, &moved, 1, MPI_INT, MPI_SUM, sim.comm);
  return moved >= 1;
}
static void vorticity(void);
static void mesh_adapt(void) {
  int changed;
  long long nref, ncom;
  long long *ref;
  long long *com;
  long long blocks_after;
  long long i;
  int temp[2], result[2];
  long long *dist;
  struct Lab lab;
  long long r;
  long long *dead;
  long long ndead;
  Real *tmp;
  long long d;
  int moved;
  vorticity();
  stencil_apply(&st_gradchi);
  changed = mesh_tag();
  MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_SUM, sim.comm);
  if (changed)
    mesh_fix();
  states_sync();
  nref = 0;
  ncom = 0;
  ref = emalloc(sta.nblk * sizeof *ref);
  com = emalloc(sta.nblk * sizeof *com);
  blocks_after = sta.nblk;
  for (i = 0; i < sta.nblk; i++) {
    struct Blk *b = &sta.blk[i];
    int st = get_state(i);
    if (st == TAG_REF) {
      ref[nref++] = node_key(b->level, b->Z);
      blocks_after += 7;
    } else if (st == TAG_COMP && b->ix % 2 == 0 && b->iy % 2 == 0 && b->iz % 2 == 0)
      com[ncom++] = node_key(b->level, b->Z);
    else if (st == TAG_COMP)
      blocks_after--;
  }
  temp[0] = (int)nref;
  temp[1] = (int)ncom;
  MPI_Allreduce(temp, result, 2, MPI_INT, MPI_SUM, sim.comm);
  dist = emalloc(sim.size * sizeof *dist);
  MPI_Allgather(&blocks_after, 1, MPI_LONG_LONG, dist, 1, MPI_LONG_LONG, sim.comm);
  halo_sync(F_PRES, 4);
  lab_init(&lab, F_PRES, 4, 1, 1, 1);
  for (r = 0; r < nref; r++) {
    struct Node *pn = node_find(ref[r], 1);
    long long ip = pn->local;
    struct Blk parent = sta.blk[ip];
    long long B[8];
    int k;
    int j;
    int i;
    int q;
    pn->state = TAG_KEEP;
    lab_load(&lab, ip);
    for (k = 0; k < 2; k++)
      for (j = 0; j < 2; j++)
        for (i = 0; i < 2; i++) {
          long long nc = zchild(&parent, i, j, k);
          long long ic = blk_alloc(parent.level + 1, nc);
          struct Node *cn = node(parent.level + 1, nc);
          cn->state = TAG_KEEP;
          cn->pos = -2;
          B[k * 4 + j * 2 + i] = ic;
        }
    for (q = 0; q < 8; q++)
      memset(BLK(B[q]), 0, BLK_S * sizeof(Real));
    mesh_refine(&lab, B, F_PRES, 4);
  }
  for (r = 0; r < nref; r++) {
    struct Node *pn = node_find(ref[r], 1);
    struct Blk parent = sta.blk[pn->local];
    int k;
    int j;
    int i;
    pn->pos = -1;
    pn->state = TAG_KEEP;
    for (k = 0; k < 2; k++)
      for (j = 0; j < 2; j++)
        for (i = 0; i < 2; i++) {
          long long nc = zchild(&parent, i, j, k);
          struct Node *cn = node(parent.level + 1, nc);
          cn->pos = sim.rank;
          if (parent.level + 2 < sim.level_max) {
            struct Blk cb = sta.blk[cn->local];
            int i0;
            int i1;
            int i2;
            for (i0 = 0; i0 < 2; i0++)
              for (i1 = 0; i1 < 2; i1++)
                for (i2 = 0; i2 < 2; i2++)
                  node(parent.level + 2, zchild(&cb, i0, i1, i2))->pos = -2;
          }
        }
  }
  for (r = nref - 1; r >= 0; r--) {
    struct Node *nd = node_find(ref[r], 1);
    if (nd->local >= 0)
      blk_remove(nd->local);
  }
  mesh_gather();
  dead = emalloc(7 * ncom * sizeof *dead);
  ndead = 0;
  tmp = emalloc(BLK_S * sizeof(Real));
  for (r = 0; r < ncom; r++) {
    struct Node *nd = node_find(com[r], 1);
    struct Blk info = sta.blk[nd->local];
    int level = info.level;
    long long B[8];
    int K;
    int J;
    int I;
    int offs[2];
    long long np;
    struct Node *pn;
    long long ib0;
    for (K = 0; K < 2; K++)
      for (J = 0; J < 2; J++)
        for (I = 0; I < 2; I++) {
          long long zs = zforward(level, info.ix + I, info.iy + J, info.iz + K);
          B[K * 4 + J * 2 + I] = node(level, zs)->local;
          if (B[K * 4 + J * 2 + I] < 0)
            fatal("compress: sibling level %d Z %lld of Z %lld is not on this rank", level, zs, info.Z);
        }
    offs[0] = 0;
    offs[1] = BS / 2;
    for (K = 0; K < 2; K++)
      for (J = 0; J < 2; J++)
        for (I = 0; I < 2; I++) {
          long long ib = B[K * 4 + J * 2 + I];
          int f;
          int k;
          int j;
          int i;
          for (f = 0; f < F_N; f++)
            for (k = 0; k < BS; k += 2)
              for (j = 0; j < BS; j += 2)
                for (i = 0; i < BS; i += 2)
                  tmp[f * BS3 + ((k / 2 + offs[K]) * BS + (j / 2 + offs[J])) * BS + (i / 2 + offs[I])] =
                      0.125 * ((CELL(ib, f, 0, i, j, k) + CELL(ib, f, 0, i + 1, j + 1, k + 1)) +
                               (CELL(ib, f, 0, i + 1, j, k) + CELL(ib, f, 0, i, j + 1, k + 1)) +
                               (CELL(ib, f, 0, i, j + 1, k) + CELL(ib, f, 0, i + 1, j, k + 1)) +
                               (CELL(ib, f, 0, i + 1, j + 1, k) + CELL(ib, f, 0, i, j, k + 1)));
        }
    np = zforward(level - 1, info.ix / 2, info.iy / 2, info.iz / 2);
    pn = node(level - 1, np);
    pn->pos = sim.rank;
    pn->state = TAG_KEEP;
    if (level - 2 >= 0) {
      struct Blk pb;
      blk_fill(&pb, level - 1, np);
      node(level - 2, zparent(&pb))->pos = -1;
    }
    ib0 = B[0];
    blk_fill(&sta.blk[ib0], level - 1, np);
    memcpy(BLK(ib0), tmp, BLK_S * sizeof(Real));
    pn->local = ib0;
    for (K = 0; K < 2; K++)
      for (J = 0; J < 2; J++)
        for (I = 0; I < 2; I++) {
          long long n = zforward(level, info.ix + I, info.iy + J, info.iz + K);
          struct Node *cn = node(level, n);
          if (I + J + K != 0)
            dead[ndead++] = node_key(level, n);
          else
            cn->local = -1;
          cn->pos = -2;
          cn->state = TAG_KEEP;
        }
  }
  free(tmp);
  for (d = 0; d < ndead; d++) {
    struct Node *nd = node_find(dead[d], 1);
    if (nd->local >= 0)
      blk_remove(nd->local);
  }
  moved = mesh_bal(dist);
  sta.mesh_changed = result[0] > 0 || result[1] > 0 || moved;
  free(ref);
  free(com);
  free(dead);
  free(dist);
  blk_sort();
  tree_sync();
  halo_build();
  fc_prepare();
  mg_build();
}
static void sta_zero(void) {
  long long i;
#pragma omp parallel for
  for (i = 0; i < sta.nblk; i++) {
    memset(fld(i, F_PRES), 0, BS3 * sizeof(Real));
    memset(fld(i, F_VEL), 0, 3 * BS3 * sizeof(Real));
    memset(fld(i, F_TMP), 0, 3 * BS3 * sizeof(Real));
    memset(fld(i, F_LHS), 0, BS3 * sizeof(Real));
  }
}
static void sta_fields(void) {
  int lmax;
  int l;
  fish_build();
  sta_zero();
  lmax = sim.static_obst ? sim.level_max : 3 * sim.level_max;
  for (l = 0; l < lmax; l++) {
    mesh_adapt();
    fish_build();
    sta_zero();
  }
}
#define L(X, Y, Z, C) (LAB(l, (X) - l->ss[0], (Y) - l->ss[1], (Z) - l->ss[2])[C])
#define LS(A, K, C) L(x + ((A) == 0) * (K), y + ((A) == 1) * (K), z + ((A) == 2) * (K), C)

static void face_cell(int f, int k, int c[3], int n[3]) {
  int d = f / 2, t1 = d == 0 ? 1 : 0, t2 = d == 2 ? 1 : 2;
  c[d] = f % 2 ? BS - 1 : 0;
  c[t1] = k % BS;
  c[t2] = k / BS;
  n[0] = c[0];
  n[1] = c[1];
  n[2] = c[2];
  n[d] += f % 2 ? 1 : -1;
}
#define LC(P, C) L((P)[0], (P)[1], (P)[2], C)
static void face_grad(struct Lab *l, long long i, int comp, Real coef) {
  int f;
  for (f = 0; f < 6; f++) {
    Real *F = fc_face(i, f, comp);
    int k;
    if (F == NULL)
      continue;
    for (k = 0; k < BS * BS; k++) {
      int c[3], n[3];
      face_cell(f, k, c, n);
      F[k] = coef * (LC(c, comp) - LC(n, comp));
    }
  }
}
static void face_sum(struct Lab *l, long long i, int f, int in, int out, Real coef) {
  Real *F = fc_face(i, f, out);
  Real s;
  int k;
  if (F == NULL)
    return;
  s = f % 2 ? -coef : coef;
  for (k = 0; k < BS * BS; k++) {
    int c[3], n[3];
    face_cell(f, k, c, n);
    F[k] = s * (LC(n, in) + LC(c, in));
  }
}
static void k_lhs(struct Lab *l, long long i) {
  Real h = sta.blk[i].h;
  Real *o = out_ptr(i, F_LHS, 0);
  int z;
  int y;
  int x;
  for (z = 0; z < BS; ++z)
    for (y = 0; y < BS; ++y)
      for (x = 0; x < BS; ++x)
        o[IDX(x, y, z)] = h * (L(x - 1, y, z, 0) + L(x + 1, y, z, 0) + L(x, y - 1, z, 0) + L(x, y + 1, z, 0) +
                               L(x, y, z - 1, 0) + L(x, y, z + 1, 0) - 6.0 * L(x, y, z, 0));
  face_grad(l, i, 0, h);
}
static struct Stencil st_lhs = {F_PRES, 1, 1, 0, -1, F_LHS, 1, k_lhs};
static void k_mg(struct Lab *l, long long i) {
  Real h = sta.blk[i].h;
  Real *o = out_ptr(i, F_LHS, 0);
  int z;
  int y;
  int x;
  for (z = 0; z < BS; ++z)
    for (y = 0; y < BS; ++y)
      for (x = 0; x < BS; ++x)
        o[IDX(x, y, z)] = h * (L(x - 1, y, z, 0) + L(x + 1, y, z, 0) + L(x, y - 1, z, 0) + L(x, y + 1, z, 0) +
                               L(x, y, z - 1, 0) + L(x, y, z + 1, 0) - 6.0 * L(x, y, z, 0));
}
static struct Stencil st_mg = {F_PRES, 1, 1, 0, -1, F_LHS, 0, k_mg};
static long long pois_pin;
static void pois_op(Real *in, Real *out) {
  Real avg_p = 0;
  long long i;
  if (sim.mean_constraint <= 2 && sim.mean_constraint > 0) {
#pragma omp parallel for reduction(+ : avg_p)
    for (i = 0; i < sta.nblk; ++i) {
      struct Blk *b = &sta.blk[i];
      Real *Z = in + i * BS3;
      Real h3 = b->h * b->h * b->h;
      int j;
      for (j = 0; j < BS3; j++)
        avg_p += Z[j] * h3;
    }
    MPI_Allreduce(MPI_IN_PLACE, &avg_p, 1, MPI_Real, MPI_SUM, sim.comm);
  }
  view_in = in;
  view_out = out;
  stencil_apply(&st_lhs);
  view_in = NULL;
  view_out = NULL;
  if (sim.mean_constraint == 0)
    return;
  if (sim.mean_constraint <= 2 && sim.mean_constraint > 0) {
    if (sim.mean_constraint == 1 && pois_pin != -1) {
      out[pois_pin * BS3] = avg_p;
    } else if (sim.mean_constraint == 2) {
#pragma omp parallel for
      for (i = 0; i < sta.nblk; ++i) {
        Real *LHS = out + i * BS3;
        Real h3 = sta.blk[i].h * sta.blk[i].h * sta.blk[i].h;
        int j;
        for (j = 0; j < BS3; j++)
          LHS[j] += avg_p * h3;
      }
    }
  } else if (pois_pin != -1) {
    out[pois_pin * BS3] = in[pois_pin * BS3];
  }
}
static Real pre_s[BS][BS], pre_w[BS3];
static void pois_init(void) {
  int i, j, k;
  Real lam[BS];
  for (j = 0; j < BS; j++) {
    lam[j] = 2 * cos(M_PI * (j + 1) / (BS + 1)) - 2;
    for (k = 0; k < BS; k++)
      pre_s[j][k] = sqrt(2.0 / (BS + 1)) * sin(M_PI * (j + 1) * (k + 1) / (BS + 1));
  }
  for (k = 0; k < BS; k++)
    for (j = 0; j < BS; j++)
      for (i = 0; i < BS; i++)
        pre_w[IDX(i, j, k)] = 1 / (lam[i] + lam[j] + lam[k]);
}
static void pre_x(Real *in, Real *out) {
  int z, y, j, k;
  for (z = 0; z < BS; z++)
    for (y = 0; y < BS; y++)
      for (j = 0; j < BS; j++) {
        Real a = 0;
        for (k = 0; k < BS; k++)
          a += pre_s[j][k] * in[IDX(k, y, z)];
        out[IDX(j, y, z)] = a;
      }
}
static void pre_y(Real *in, Real *out) {
  int z, x, j, k;
  for (z = 0; z < BS; z++)
    for (j = 0; j < BS; j++)
      for (x = 0; x < BS; x++) {
        Real a = 0;
        for (k = 0; k < BS; k++)
          a += pre_s[j][k] * in[IDX(x, k, z)];
        out[IDX(x, j, z)] = a;
      }
}
static void pre_z(Real *in, Real *out) {
  int y, x, j, k;
  for (j = 0; j < BS; j++)
    for (y = 0; y < BS; y++)
      for (x = 0; x < BS; x++) {
        Real a = 0;
        for (k = 0; k < BS; k++)
          a += pre_s[j][k] * in[IDX(x, y, k)];
        out[IDX(x, y, j)] = a;
      }
}
static void pre_blk(Real *src, Real *dst, Real invh, Real *a, Real *b) {
  int j;
  for (j = 0; j < BS3; j++)
    a[j] = invh * src[j];
  pre_x(a, b);
  pre_y(b, a);
  pre_z(a, b);
  for (j = 0; j < BS3; j++)
    b[j] *= pre_w[j];
  pre_x(b, a);
  pre_y(a, b);
  pre_z(b, dst);
}
static void field_set(int f, Real *in) {
  long long i;
#pragma omp parallel for
  for (i = 0; i < sta.nblk; i++) {
    memcpy(fld(i, f), in + i * BS3, BS3 * sizeof(Real));
  }
}
static void field_get(int f, Real *out) {
  long long i;
#pragma omp parallel for
  for (i = 0; i < sta.nblk; i++) {
    memcpy(out + i * BS3, fld(i, f), BS3 * sizeof(Real));
  }
}
static void pois_op_vec(Real *in, Real *out) { pois_op(in, out); }
static struct Pois {
  long long cap;
  Real *x, *b, *r, *w, *z, *hw, *V;
} pois;
enum { KR_M = 30, KR_MAXIT = 1000 };
static void pois_alloc(long long N) {
  if (N <= pois.cap)
    return;
  N += N / 4;
  free(pois.x);
  free(pois.b);
  free(pois.r);
  free(pois.w);
  free(pois.z);
  free(pois.hw);
  free(pois.V);
  pois.x = ecalloc(N, sizeof(Real));
  pois.b = ecalloc(N, sizeof(Real));
  pois.r = ecalloc(N, sizeof(Real));
  pois.w = ecalloc(N, sizeof(Real));
  pois.z = ecalloc(N, sizeof(Real));
  pois.hw = ecalloc(N / BS3, sizeof(Real));
  pois.V = ecalloc((KR_M + 1) * N, sizeof(Real));
  pois.cap = N;
}
static void vec_copy(Real *dst, Real *src, long long n) {
  long long i;
#pragma omp parallel for
  for (i = 0; i < n; i += BS3)
    memcpy(dst + i, src + i, (n - i < BS3 ? n - i : BS3) * sizeof(Real));
}
static void vec_zero(Real *dst, long long n) {
  long long i;
#pragma omp parallel for
  for (i = 0; i < n; i += BS3)
    memset(dst + i, 0, (n - i < BS3 ? n - i : BS3) * sizeof(Real));
}
enum { MG_PRE = 2, MG_POST = 2, MG_BOT = 50, MG_M = 2 * BS3 / 8 };
static Real mg_omega = 0.8;
struct Ctx {
  long long nblk;
  struct Blk *blk;
  long long *slot;
  struct Node *tab;
  long long cap, n;
  struct Halo halo;
};
struct Lvl {
  struct Ctx c;
  long long nact, *act;
  long long npar, *par;
  long long *pslot, *pos;
  int *oct, *dst;
  struct Xch x, xr;
  long long nsend, nrecv, *rslot;
  int *roct;
  Real *sbuf, *rbuf;
};
static struct {
  int top, cur;
  long long nslot, cap;
  struct Lvl *lv;
  Real *u, *f, *t, *us;
} mg;
static void ctx_swap(struct Ctx *c) {
  struct Ctx t = {sta.nblk, sta.blk, slot, nodes.tab, nodes.cap, nodes.n, halo};
  sta.nblk = c->nblk;
  sta.blk = c->blk;
  slot = c->slot;
  nodes.tab = c->tab;
  nodes.cap = c->cap;
  nodes.n = c->n;
  halo = c->halo;
  *c = t;
}
static void mg_use(int L) {
  if (mg.cur == L)
    return;
  if (mg.cur != mg.top)
    ctx_swap(&mg.lv[mg.cur].c);
  if (L != mg.top)
    ctx_swap(&mg.lv[L].c);
  mg.cur = L;
}
static void ctx_free(struct Ctx *c) {
  free(c->blk);
  free(c->slot);
  free(c->tab);
  free(c->halo.send);
  free(c->halo.rkey);
  free(c->halo.buf);
  free(c->halo.sbuf);
  free(c->halo.sst);
  free(c->halo.rst);
  xch_free(&c->halo.x);
  memset(c, 0, sizeof *c);
}
static void mg_free(void) {
  int L;
  if (mg.lv == NULL)
    return;
  mg_use(mg.top);
  for (L = 0; L <= mg.top; L++) {
    struct Lvl *v = &mg.lv[L];
    if (L < mg.top)
      ctx_free(&v->c);
    free(v->act);
    free(v->par);
    free(v->pslot);
    free(v->pos);
    free(v->oct);
    free(v->dst);
    free(v->rslot);
    free(v->roct);
    free(v->sbuf);
    free(v->rbuf);
    xch_free(&v->x);
    xch_free(&v->xr);
  }
  free(mg.lv);
  mg.lv = NULL;
}
static int mg_cmp(const void *a, const void *b) {
  long long *x = (long long *)a, *y = (long long *)b;
  return (x[0] > y[0]) - (x[0] < y[0]);
}
static void mg_build(void) {
  int L;
  long long i;
  mg_free();
  mg.top = sim.level_max - 1;
  mg.cur = mg.top;
  mg.nslot = sta.nblk;
  mg.lv = ecalloc(mg.top + 1, sizeof *mg.lv);
  for (L = mg.top; L >= 0; L--) {
    struct Lvl *v = &mg.lv[L];
    struct Lvl *w;
    long long *zp, *ent, *key;
    long long npass, nz, k, m;
    struct Blk *cb;
    long long *cs;
    mg_use(L);
    v->act = emalloc((sta.nblk + 1) * sizeof *v->act);
    v->nact = 0;
    for (i = 0; i < sta.nblk; i++)
      if (sta.blk[i].level == L)
        v->act[v->nact++] = i;
    if (L == 0)
      break;
    w = &mg.lv[L - 1];
    zp = emalloc((v->nact + 1) * sizeof *zp);
    v->oct = emalloc((v->nact + 1) * sizeof *v->oct);
    v->dst = emalloc((v->nact + 1) * sizeof *v->dst);
    v->pslot = emalloc((v->nact + 1) * sizeof *v->pslot);
    v->pos = emalloc((v->nact + 1) * sizeof *v->pos);
    ent = emalloc((v->nact + 1) * sizeof *ent);
    nz = 0;
    for (k = 0; k < v->nact; k++) {
      struct Blk *b = &sta.blk[v->act[k]];
      int owner = sim.size;
      int I, J, K;
      struct Blk pb;
      zp[k] = zparent(b);
      v->oct[k] = (b->ix & 1) + 2 * (b->iy & 1) + 4 * (b->iz & 1);
      blk_fill(&pb, L - 1, zp[k]);
      for (K = 0; K < 2; K++)
        for (J = 0; J < 2; J++)
          for (I = 0; I < 2; I++) {
            struct Node *nd = node_get(L, zchild(&pb, I, J, K));
            if (nd == NULL || nd->pos < 0)
              fatal("mg_build: sibling of level %d Z %lld missing", L, b->Z);
            owner = nd->pos < owner ? nd->pos : owner;
          }
      v->dst[k] = owner == sim.rank ? -1 : owner;
      if (owner == sim.rank)
        ent[nz++] = zp[k];
    }
    qsort(ent, nz, sizeof *ent, mg_cmp);
    m = 0;
    for (k = 0; k < nz; k++)
      if (m == 0 || ent[m - 1] != ent[k])
        ent[m++] = ent[k];
    nz = m;
    npass = sta.nblk - v->nact;
    cb = emalloc((npass + nz + 1) * sizeof *cb);
    cs = emalloc((npass + nz + 1) * sizeof *cs);
    key = emalloc(2 * (npass + nz + 1) * sizeof *key);
    m = 0;
    for (i = 0; i < sta.nblk; i++)
      if (sta.blk[i].level < L) {
        cb[m] = sta.blk[i];
        cs[m] = SLOT(i);
        m++;
      }
    for (k = 0; k < nz; k++) {
      blk_fill(&cb[m], L - 1, ent[k]);
      cs[m] = mg.nslot++;
      m++;
    }
    for (i = 0; i < m; i++) {
      key[2 * i] = blk_id(&cb[i]);
      key[2 * i + 1] = i;
    }
    qsort(key, m, 2 * sizeof *key, mg_cmp);
    w->c.nblk = m;
    w->c.blk = emalloc((m + 1) * sizeof *w->c.blk);
    w->c.slot = emalloc((m + 1) * sizeof *w->c.slot);
    for (i = 0; i < m; i++) {
      w->c.blk[i] = cb[key[2 * i + 1]];
      w->c.slot[i] = cs[key[2 * i + 1]];
    }
    free(cb);
    free(cs);
    free(key);
    free(ent);
    mg_use(L - 1);
    tree_sync();
    halo_lvl = L - 1;
    halo_build();
    halo_lvl = -1;
    w->npar = 0;
    w->par = emalloc((nz + 1) * sizeof *w->par);
    for (i = 0; i < sta.nblk; i++)
      if (sta.blk[i].level == L - 1 && slot[i] >= mg.nslot - nz)
        w->par[w->npar++] = i;
    xch_reset(&v->x);
    for (k = 0; k < v->nact; k++) {
      if (v->dst[k] >= 0) {
        v->x.scnt[v->dst[k]]++;
        v->pslot[k] = -1;
      } else {
        struct Node *nd = node_get(L - 1, zp[k]);
        v->pslot[k] = slot[nd->local];
      }
    }
    MPI_Alltoall(v->x.scnt, 1, MPI_INT, v->x.rcnt, 1, MPI_INT, sim.comm);
    xch_dsp(&v->x);
    v->nsend = xch_nsend(&v->x);
    v->nrecv = xch_nrecv(&v->x);
    {
      long long *sk = emalloc(2 * (v->nsend + 1) * sizeof *sk);
      long long *rk = emalloc(2 * (v->nrecv + 1) * sizeof *rk);
      int *fill = ecalloc(sim.size, sizeof *fill);
      for (k = 0; k < v->nact; k++) {
        v->pos[k] = -1;
        if (v->dst[k] >= 0) {
          v->pos[k] = fill[v->dst[k]]++;
          sk[2 * (v->x.sdsp[v->dst[k]] + v->pos[k])] = zp[k];
          sk[2 * (v->x.sdsp[v->dst[k]] + v->pos[k]) + 1] = v->oct[k];
        }
      }
      xch_exec(&v->x, sk, rk, 2, MPI_LONG_LONG);
      v->rslot = emalloc((v->nrecv + 1) * sizeof *v->rslot);
      v->roct = emalloc((v->nrecv + 1) * sizeof *v->roct);
      for (k = 0; k < v->nrecv; k++) {
        struct Node *nd = node_get(L - 1, rk[2 * k]);
        if (nd == NULL || nd->pos != sim.rank)
          fatal("mg_build: received parent level %d Z %lld not local", L - 1, rk[2 * k]);
        v->rslot[k] = slot[nd->local];
        v->roct[k] = (int)rk[2 * k + 1];
      }
      free(sk);
      free(rk);
      free(fill);
    }
    xch_reset(&v->xr);
    memcpy(v->xr.scnt, v->x.rcnt, sim.size * sizeof(int));
    memcpy(v->xr.rcnt, v->x.scnt, sim.size * sizeof(int));
    xch_dsp(&v->xr);
    v->sbuf = emalloc((v->nsend + v->nrecv + 1) * MG_M * sizeof(Real));
    v->rbuf = emalloc((v->nsend + v->nrecv + 1) * MG_M * sizeof(Real));
    free(zp);
  }
  mg_use(mg.top);
  if (mg.nslot > mg.cap) {
    free(mg.u);
    free(mg.f);
    free(mg.t);
    free(mg.us);
    mg.cap = mg.nslot + mg.nslot / 4;
    mg.u = emalloc(mg.cap * BS3 * sizeof(Real));
    mg.f = emalloc(mg.cap * BS3 * sizeof(Real));
    mg.t = emalloc(mg.cap * BS3 * sizeof(Real));
    mg.us = emalloc(mg.cap * BS3 * sizeof(Real));
  }
}
static void mg_op(Real *in, Real *out, long long *list, long long n) {
  view_in = in;
  view_out = out;
  stencil_run(&st_mg, list, n);
  view_in = NULL;
  view_out = NULL;
}
static void mg_smooth(struct Lvl *v, int n) {
  int it;
  for (it = 0; it < n; it++) {
    mg_op(mg.u, mg.t, v->act, v->nact);
#pragma omp parallel
    {
      Real a[BS3], b[BS3], r[BS3];
      long long k;
#pragma omp for schedule(dynamic, 1)
      for (k = 0; k < v->nact; k++) {
        long long i = v->act[k];
        Real *u = mg.u + SLOT(i) * BS3, *f = mg.f + SLOT(i) * BS3, *t = mg.t + SLOT(i) * BS3;
        int j;
        for (j = 0; j < BS3; j++)
          r[j] = f[j] - t[j];
        pre_blk(r, b, 1 / sta.blk[i].h, a, r);
        for (j = 0; j < BS3; j++)
          u[j] += mg_omega * b[j];
      }
    }
  }
}
static void mg_sum(Real *src, Real *out64, Real scale) {
  int x, y, z;
  for (z = 0; z < 4; z++)
    for (y = 0; y < 4; y++)
      for (x = 0; x < 4; x++)
        out64[(z * 4 + y) * 4 + x] =
            scale * (src[IDX(2 * x, 2 * y, 2 * z)] + src[IDX(2 * x + 1, 2 * y, 2 * z)] +
                     src[IDX(2 * x, 2 * y + 1, 2 * z)] + src[IDX(2 * x + 1, 2 * y + 1, 2 * z)] +
                     src[IDX(2 * x, 2 * y, 2 * z + 1)] + src[IDX(2 * x + 1, 2 * y, 2 * z + 1)] +
                     src[IDX(2 * x, 2 * y + 1, 2 * z + 1)] + src[IDX(2 * x + 1, 2 * y + 1, 2 * z + 1)]);
}
static void mg_put(long long ps, int oct, Real *r64, Real *u64) {
  int x, y, z;
  int ox = 4 * (oct & 1), oy = 4 * ((oct >> 1) & 1), oz = 4 * (oct >> 2);
  Real *f = mg.f + ps * BS3, *u = mg.u + ps * BS3;
  for (z = 0; z < 4; z++)
    for (y = 0; y < 4; y++)
      for (x = 0; x < 4; x++) {
        int q = (z * 4 + y) * 4 + x;
        f[IDX(ox + x, oy + y, oz + z)] = r64[q];
        u[IDX(ox + x, oy + y, oz + z)] = u64[q];
      }
}
static void mg_down(struct Lvl *v) {
  long long k;
  mg_op(mg.u, mg.t, v->act, v->nact);
#pragma omp parallel for schedule(dynamic, 1)
  for (k = 0; k < v->nact; k++) {
    long long i = v->act[k], s = SLOT(i);
    Real r[BS3], r64[64], u64[64];
    int j;
    for (j = 0; j < BS3; j++)
      r[j] = mg.f[s * BS3 + j] - mg.t[s * BS3 + j];
    mg_sum(r, r64, 1);
    mg_sum(mg.u + s * BS3, u64, 1.0 / 8);
    if (v->pslot[k] >= 0) {
      mg_put(v->pslot[k], v->oct[k], r64, u64);
    } else {
      long long q = v->x.sdsp[v->dst[k]] + v->pos[k];
      memcpy(v->sbuf + q * MG_M, r64, 64 * sizeof(Real));
      memcpy(v->sbuf + q * MG_M + 64, u64, 64 * sizeof(Real));
    }
  }
  xch_exec(&v->x, v->sbuf, v->rbuf, MG_M, MPI_Real);
  for (k = 0; k < v->nrecv; k++)
    mg_put(v->rslot[k], v->roct[k], v->rbuf + k * MG_M, v->rbuf + k * MG_M + 64);
}
static void mg_tau(struct Lvl *w) {
  long long k;
  mg_op(mg.u, mg.t, w->par, w->npar);
#pragma omp parallel for
  for (k = 0; k < w->npar; k++) {
    long long s = SLOT(w->par[k]);
    int j;
    for (j = 0; j < BS3; j++) {
      mg.f[s * BS3 + j] += mg.t[s * BS3 + j];
      mg.us[s * BS3 + j] = mg.u[s * BS3 + j];
    }
  }
}
static void mg_get(long long ps, int oct, Real *d64) {
  int x, y, z;
  int ox = 4 * (oct & 1), oy = 4 * ((oct >> 1) & 1), oz = 4 * (oct >> 2);
  Real *u = mg.u + ps * BS3, *us = mg.us + ps * BS3;
  for (z = 0; z < 4; z++)
    for (y = 0; y < 4; y++)
      for (x = 0; x < 4; x++)
        d64[(z * 4 + y) * 4 + x] = u[IDX(ox + x, oy + y, oz + z)] - us[IDX(ox + x, oy + y, oz + z)];
}
static void mg_add(Real *u, Real *d64) {
  int x, y, z;
  for (z = 0; z < BS; z++)
    for (y = 0; y < BS; y++)
      for (x = 0; x < BS; x++)
        u[IDX(x, y, z)] += d64[((z / 2) * 4 + y / 2) * 4 + x / 2];
}
static void mg_up(struct Lvl *v) {
  long long k;
  for (k = 0; k < v->nrecv; k++)
    mg_get(v->rslot[k], v->roct[k], v->sbuf + k * MG_M);
  xch_exec(&v->xr, v->sbuf, v->rbuf, MG_M, MPI_Real);
}
static void mg_up2(struct Lvl *v) {
  long long k;
#pragma omp parallel for schedule(dynamic, 1)
  for (k = 0; k < v->nact; k++) {
    long long i = v->act[k], s = SLOT(i);
    Real d64[64];
    if (v->pslot[k] >= 0) {
      mg_get(v->pslot[k], v->oct[k], d64);
      mg_add(mg.u + s * BS3, d64);
    } else {
      long long q = v->xr.rdsp[v->dst[k]] + v->pos[k];
      mg_add(mg.u + s * BS3, v->rbuf + q * MG_M);
    }
  }
}
static void mg_bottom(struct Lvl *v) {
  Real q[2] = {0, 0};
  long long k;
#pragma omp parallel for reduction(+ : q[ : 2])
  for (k = 0; k < v->nact; k++) {
    long long s = SLOT(v->act[k]);
    Real h3 = sta.blk[v->act[k]].h * sta.blk[v->act[k]].h * sta.blk[v->act[k]].h;
    int j;
    for (j = 0; j < BS3; j++)
      q[0] += mg.f[s * BS3 + j] * h3;
    q[1] += BS3 * h3;
  }
  MPI_Allreduce(MPI_IN_PLACE, q, 2, MPI_Real, MPI_SUM, sim.comm);
  q[0] /= q[1];
#pragma omp parallel for
  for (k = 0; k < v->nact; k++) {
    long long s = SLOT(v->act[k]);
    int j;
    for (j = 0; j < BS3; j++)
      mg.f[s * BS3 + j] -= q[0];
  }
  mg_smooth(v, MG_BOT);
}
static void mg_vcycle(Real *in, Real *out) {
  long long N = sta.nblk * BS3;
  int L;
  vec_zero(mg.u, mg.nslot * BS3);
  vec_copy(mg.f, in, N);
  for (L = mg.top; L >= 1; L--) {
    mg_use(L);
    mg_smooth(&mg.lv[L], MG_PRE);
    mg_down(&mg.lv[L]);
    mg_use(L - 1);
    mg_tau(&mg.lv[L - 1]);
  }
  mg_use(0);
  mg_bottom(&mg.lv[0]);
  for (L = 1; L <= mg.top; L++) {
    mg_use(L - 1);
    mg_up(&mg.lv[L]);
    mg_use(L);
    mg_up2(&mg.lv[L]);
    mg_smooth(&mg.lv[L], MG_POST);
  }
  vec_copy(out, mg.u, N);
}
static Real pois_dot(Real *a, Real *b, long long N) {
  Real d = 0;
  long long i;
#pragma omp parallel for reduction(+ : d)
  for (i = 0; i < N; i++)
    d += a[i] * b[i] * pois.hw[i / BS3];
  MPI_Allreduce(MPI_IN_PLACE, &d, 1, MPI_Real, MPI_SUM, sim.comm);
  return d;
}
static void pois_axpy(Real *y, Real a, Real *x, long long N) {
  long long i;
#pragma omp parallel for
  for (i = 0; i < N; i++)
    y[i] += a * x[i];
}
static void pois_scale(Real *y, Real a, long long N) {
  long long i;
#pragma omp parallel for
  for (i = 0; i < N; i++)
    y[i] *= a;
}
static void pois_solve(void) {
  long long N = sta.nblk * BS3;
  Real H[KR_M + 1][KR_M], cs[KR_M], sn[KR_M], g[KR_M + 1], y[KR_M];
  Real vol, bnorm, norm, beta;
  long long i;
  int it, j, k, done;
  pois_alloc(N);
  vol = 0;
  pois_pin = -1;
  for (i = 0; i < sta.nblk; i++) {
    Real h3 = sta.blk[i].h * sta.blk[i].h * sta.blk[i].h;
    pois.hw[i] = 1 / h3;
    vol += BS3 * h3;
    if (sta.blk[i].ix == 0 && sta.blk[i].iy == 0 && sta.blk[i].iz == 0)
      pois_pin = i;
  }
  MPI_Allreduce(MPI_IN_PLACE, &vol, 1, MPI_Real, MPI_SUM, sim.comm);
#pragma omp parallel for
  for (i = 0; i < sta.nblk; i++) {
    Real *rhs = fld(i, F_LHS);
    Real *zz = fld(i, F_PRES);
    struct Blk *bb = &sta.blk[i];
    int j;
    if (sim.mean_constraint == 1 || sim.mean_constraint > 2)
      if (bb->ix == 0 && bb->iy == 0 && bb->iz == 0)
        rhs[0] = 0.0;
    for (j = 0; j < BS3; j++) {
      pois.b[i * BS3 + j] = rhs[j];
      pois.x[i * BS3 + j] = zz[j];
    }
  }
  bnorm = sqrt(pois_dot(pois.b, pois.b, N) / vol);
  pois_op_vec(pois.x, pois.r);
#pragma omp parallel for
  for (i = 0; i < N; i++)
    pois.r[i] = pois.b[i] - pois.r[i];
  it = 0;
  done = 0;
  while (!done) {
    beta = sqrt(pois_dot(pois.r, pois.r, N));
    norm = beta / sqrt(vol);
    if (norm < sim.ptol || norm < sim.ptol_rel * bnorm || it >= KR_MAXIT)
      break;
    vec_copy(pois.V, pois.r, N);
    pois_scale(pois.V, 1 / beta, N);
    memset(g, 0, sizeof g);
    g[0] = beta;
    for (j = 0; j < KR_M; j++) {
      Real *v = pois.V + (long long)(j + 1) * N;
      mg_vcycle(pois.V + (long long)j * N, pois.z);
      pois_op_vec(pois.z, pois.w);
      for (k = 0; k <= j; k++) {
        H[k][j] = pois_dot(pois.w, pois.V + (long long)k * N, N);
        pois_axpy(pois.w, -H[k][j], pois.V + (long long)k * N, N);
      }
      H[j + 1][j] = sqrt(pois_dot(pois.w, pois.w, N));
      vec_copy(v, pois.w, N);
      if (H[j + 1][j] > 0)
        pois_scale(v, 1 / H[j + 1][j], N);
      for (k = 0; k < j; k++) {
        Real t = cs[k] * H[k][j] + sn[k] * H[k + 1][j];
        H[k + 1][j] = -sn[k] * H[k][j] + cs[k] * H[k + 1][j];
        H[k][j] = t;
      }
      {
        Real d = sqrt(H[j][j] * H[j][j] + H[j + 1][j] * H[j + 1][j]);
        cs[j] = d > 0 ? H[j][j] / d : 1;
        sn[j] = d > 0 ? H[j + 1][j] / d : 0;
        H[j][j] = d;
        H[j + 1][j] = 0;
        g[j + 1] = -sn[j] * g[j];
        g[j] = cs[j] * g[j];
      }
      it++;
      norm = fabs(g[j + 1]) / sqrt(vol);
      if (norm < sim.ptol || norm < sim.ptol_rel * bnorm || it >= KR_MAXIT) {
        done = 1;
        j++;
        break;
      }
    }
    for (k = j - 1; k >= 0; k--) {
      int m;
      y[k] = g[k];
      for (m = k + 1; m < j; m++)
        y[k] -= H[k][m] * y[m];
      y[k] = H[k][k] != 0 ? y[k] / H[k][k] : 0;
    }
    vec_zero(pois.w, N);
    for (k = 0; k < j; k++)
      pois_axpy(pois.w, y[k], pois.V + (long long)k * N, N);
    mg_vcycle(pois.w, pois.z);
    pois_axpy(pois.x, 1, pois.z, N);
    pois_op_vec(pois.x, pois.r);
#pragma omp parallel for
    for (i = 0; i < N; i++)
      pois.r[i] = pois.b[i] - pois.r[i];
  }
  field_set(F_PRES, pois.x);
}
static Real derivative(Real U, Real um3, Real um2, Real um1, Real u, Real up1, Real up2, Real up3) {
  if (U > 0)
    return (-2 * um3 + 15 * um2 - 60 * um1 + 20 * u + 30 * up1 - 3 * up2) / 60.;
  else
    return (2 * up3 - 15 * up2 + 60 * up1 - 20 * u - 30 * um1 + 3 * um2) / 60.;
}
static void k_advdiff(struct Lab *l, long long i) {
  Real dt = sta.dt;
  Real mu = sim.nu;
  Real coef = 1.0;
  Real *u_inf = sta.uinf;
  Real h = sta.blk[i].h;
  Real *o = fld(i, F_TMP);
  Real h3 = h * h * h;
  Real fac_a = -dt / h * h3 * coef;
  Real fac_d = (mu / h) * (dt / h) * h3 * coef;
  int z;
  int y;
  int x;
  int c;
  for (z = 0; z < BS; ++z)
    for (y = 0; y < BS; ++y)
      for (x = 0; x < BS; ++x) {
        Real u_abs[3] = {L(x, y, z, 0) + u_inf[0], L(x, y, z, 1) + u_inf[1], L(x, y, z, 2) + u_inf[2]};
        int c;
        for (c = 0; c < 3; c++) {
          Real dd[3], pair[3];
          int a;
          int a1, a2;
          Real adv;
          Real lap;
          for (a = 0; a < 3; a++) {
            dd[a] = derivative(u_abs[a], LS(a, -3, c), LS(a, -2, c), LS(a, -1, c), LS(a, 0, c), LS(a, 1, c),
                               LS(a, 2, c), LS(a, 3, c));
            pair[a] = LS(a, 1, c) + LS(a, -1, c);
          }
          a1 = (c + 1) % 3;
          a2 = (c + 2) % 3;
          adv = u_abs[c] * dd[c] + (u_abs[a1] * dd[a1] + u_abs[a2] * dd[a2]);
          lap = (pair[c] + (pair[a1] + pair[a2])) - 6 * L(x, y, z, c);
          o[c * BS3 + IDX(x, y, z)] += fac_a * adv + fac_d * lap;
        }
      }
  for (c = 0; c < 3; c++)
    face_grad(l, i, c, fac_d);
}
static struct Stencil st_advdiff = {F_VEL, 3, 3, 0, 0, F_TMP, 3, k_advdiff};
static void advdiff(void) {
  Real alpha[3] = {1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0};
  Real beta[3] = {-5.0 / 9.0, -153.0 / 128.0, 0.0};
  long long i;
  int RKstep;
#pragma omp parallel for
  for (i = 0; i < sta.nblk; i++) {
    memset(fld(i, F_TMP), 0, 3 * BS3 * sizeof(Real));
  }
  for (RKstep = 0; RKstep < 3; RKstep++) {
    long long i;
    stencil_apply(&st_advdiff);
#pragma omp parallel for
    for (i = 0; i < sta.nblk; i++) {
      Real h = sta.blk[i].h;
      Real ih3 = alpha[RKstep] / (h * h * h);
      Real *tmpv = fld(i, F_TMP);
      Real *V = fld(i, F_VEL);
      int j;
      for (j = 0; j < BS3; j++) {
        V[0 * BS3 + j] += tmpv[0 * BS3 + j] * ih3;
        V[1 * BS3 + j] += tmpv[1 * BS3 + j] * ih3;
        V[2 * BS3 + j] += tmpv[2 * BS3 + j] * ih3;
        tmpv[0 * BS3 + j] *= beta[RKstep];
        tmpv[1 * BS3 + j] *= beta[RKstep];
        tmpv[2 * BS3 + j] *= beta[RKstep];
      }
    }
  }
}
static void fish_mom_blk(long long i, struct Fish *f) {
  struct ObstacleBlock *o = oblock(f, i);
  struct Blk *b;
  Real lambda, dt;
  Real *CM;
  Real *V;
  Real *M;
  int q;
  Real lambdt;
  int iz;
  int iy;
  int ix;
  if (o == NULL)
    return;
  b = &sta.blk[i];
  lambda = sta.lambda;
  dt = sta.dt;
  CM = f->com;
  V = fld(i, F_VEL);
  M = o->mom;
  for (q = 0; q < M_N; q++)
    M[q] = 0;
  lambdt = lambda * dt;
  for (iz = 0; iz < BS; ++iz)
    for (iy = 0; iy < BS; ++iy)
      for (ix = 0; ix < BS; ++ix) {
        Real p[3];
        Real dv, X;
        Real u[3];
        Real DiffU[3], pxu[3], pxdu[3];
        int d;
        Real X1;
        Real pen_fac;
        if (o->chi[iz][iy][ix] <= 0)
          continue;
        blk_pos(b, ix, iy, iz, p);
        dv = b->h * b->h * b->h;
        X = o->chi[iz][iy][ix];
        u[0] = V[0 * BS3 + IDX(ix, iy, iz)];
        u[1] = V[1 * BS3 + IDX(ix, iy, iz)];
        u[2] = V[2 * BS3 + IDX(ix, iy, iz)];
        for (d = 0; d < 3; d++) {
          p[d] -= CM[d];
          DiffU[d] = u[d] - o->udef[iz][iy][ix][d];
        }
        cross3(pxu, p, u);
        cross3(pxdu, p, DiffU);
        X1 = o->chi[iz][iy][ix] > 0.5 ? 1.0 : 0.0;
        pen_fac = dv * lambdt * X1 / (1 + X1 * lambdt);
        M[M_V] += X * dv;
        M[M_GfX] += pen_fac;
        inertia_add(&M[M_J0], X * dv, p);
        inertia_add(&M[M_Gj0], pen_fac, p);
        for (d = 0; d < 3; d++) {
          M[M_FX + d] += X * dv * u[d];
          M[M_TX + d] += X * dv * pxu[d];
          M[M_GpX + d] += pen_fac * p[d];
          M[M_GuX + d] += pen_fac * DiffU[d];
          M[M_GaX + d] += pen_fac * pxdu[d];
        }
      }
}
static int solve6(double *A, double *b, double *x) {
  enum { N = 6 };
  int i, j, k;
  for (k = 0; k < N; k++) {
    int p = k;
    for (i = k + 1; i < N; i++)
      if (fabs(A[i * N + k]) > fabs(A[p * N + k]))
        p = i;
    if (fabs(A[p * N + k]) <= DBL_MIN)
      return 1;
    if (p != k) {
      double tmp;
      for (j = 0; j < N; j++) {
        tmp = A[k * N + j];
        A[k * N + j] = A[p * N + j];
        A[p * N + j] = tmp;
      }
      tmp = b[k];
      b[k] = b[p];
      b[p] = tmp;
    }
    for (i = k + 1; i < N; i++) {
      double f = A[i * N + k] / A[k * N + k];
      for (j = k + 1; j < N; j++)
        A[i * N + j] -= f * A[k * N + j];
      b[i] -= f * b[k];
    }
  }
  for (i = N - 1; i >= 0; i--) {
    double s = b[i];
    for (j = i + 1; j < N; j++)
      s -= A[i * N + j] * x[j];
    x[i] = s / A[i * N + i];
  }
  return 0;
}
static void fish_solve6(struct Fish *f) {
  double A[36] = {0};
  Real *pen_cm = f->pen_cm, *pen_j = f->pen_j;
  Real pen_m = f->pen_m;
  int d;
  static int jidx[3][3] = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};
  int a;
  int b;
  double b3[6];
  double x[6];
  for (d = 0; d < 3; d++)
    A[d * 6 + d] = pen_m;
  A[0 * 6 + 4] = +pen_cm[2];
  A[0 * 6 + 5] = -pen_cm[1];
  A[1 * 6 + 3] = -pen_cm[2];
  A[1 * 6 + 5] = +pen_cm[0];
  A[2 * 6 + 3] = +pen_cm[1];
  A[2 * 6 + 4] = -pen_cm[0];
  A[3 * 6 + 1] = -pen_cm[2];
  A[3 * 6 + 2] = +pen_cm[1];
  A[4 * 6 + 0] = +pen_cm[2];
  A[4 * 6 + 2] = -pen_cm[0];
  A[5 * 6 + 0] = -pen_cm[1];
  A[5 * 6 + 1] = +pen_cm[0];
  for (a = 0; a < 3; a++)
    for (b = 0; b < 3; b++)
      A[(3 + a) * 6 + 3 + b] = pen_j[jidx[a][b]];
  b3[0] = f->pen_lmom[0];
  b3[1] = f->pen_lmom[1];
  b3[2] = f->pen_lmom[2];
  b3[3] = f->pen_amom[0];
  b3[4] = f->pen_amom[1];
  b3[5] = f->pen_amom[2];
  for (d = 0; d < 3; d++)
    if (f->forced[d]) {
      int j;
      for (j = 0; j < 6; j++)
        if (j != d)
          A[d * 6 + j] = 0;
      b3[d] = pen_m * f->vel_imposed[d];
    }
  for (d = 0; d < 3; d++)
    if (f->block_rot[d]) {
      int j;
      for (j = 0; j < 6; j++)
        if (j != 3 + d)
          A[(3 + d) * 6 + j] = 0;
      b3[3 + d] = 0;
    }
  x[0] = f->vel[0];
  x[1] = f->vel[1];
  x[2] = f->vel[2];
  x[3] = f->omega[0];
  x[4] = f->omega[1];
  x[5] = f->omega[2];
  if (pen_m > 0)
    solve6(A, b3, x);
  for (d = 0; d < 3; d++)
    f->vel[d] = f->forced[d] ? f->vel_imposed[d] : x[d];
  for (d = 0; d < 3; d++)
    f->omega[d] = f->block_rot[d] ? 0 : x[3 + d];
  if (f->hit_time > 0) {
    int d;
    f->hit_time -= sta.dt;
    for (d = 0; d < 3; d++) {
      f->vel[d] = f->hit_vel[d];
      f->omega[d] = f->hit_omega[d];
    }
  }
}
static void fish_solve(struct Fish *f) {
  fish_solve6(f);
  if (f->correct_roll) {
    struct Midline *mid = &f->m;
    Real *q = f->quaternion;
    Real *o = f->omega;
    Real dq[4];
    Real nom;
    Real dnom;
    Real denom;
    Real ddenom;
    Real arg;
    Real darg;
    Real a;
    Real da;
    int nm;
    Real dv[3];
    int d;
    Real dn;
    Real roll_axis[3];
    Real time_roll;
    int elements_to_keep;
    int i;
    int elements_to_delete;
    Real omega_roll;
    Real correction_magnitude, dummy;
    quat_rate(q, o, dq);
    nom = 2.0 * (q[3] * q[2] + q[0] * q[1]);
    dnom = 2.0 * (dq[3] * q[2] + dq[0] * q[1] + q[3] * dq[2] + q[0] * dq[1]);
    denom = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
    ddenom = -2.0 * (2.0 * q[1] * dq[1] + 2.0 * q[2] * dq[2]);
    arg = nom / denom;
    darg = (dnom * denom - nom * ddenom) / denom / denom;
    a = atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
    da = 1.0 / (1.0 + arg * arg) * darg;
    nm = mid->nm;
    for (d = 0; d < 3; d++)
      dv[d] = mid->r[0][d] - mid->r[nm - 1][d];
    dn = pow(dot3(dv, dv), 0.5) + 1e-21;
    f->r_axis = erealloc(f->r_axis, (f->nr_axis + 1) * sizeof *f->r_axis);
    for (d = 0; d < 3; d++)
      f->r_axis[f->nr_axis][d] = -dv[d] / dn;
    f->r_axis[f->nr_axis][3] = sta.dt;
    f->nr_axis++;
    roll_axis[0] = 0.;
    roll_axis[1] = 0.;
    roll_axis[2] = 0.;
    time_roll = 0.0;
    elements_to_keep = 0;
    for (i = f->nr_axis - 1; i >= 0; i--) {
      Real *r = f->r_axis[i];
      Real dt = r[3];
      int d;
      if (time_roll + dt > 5.0)
        break;
      for (d = 0; d < 3; d++)
        roll_axis[d] += r[d] * dt;
      time_roll += dt;
      elements_to_keep++;
    }
    time_roll += 1e-21;
    for (d = 0; d < 3; d++)
      roll_axis[d] /= time_roll;
    elements_to_delete = f->nr_axis - elements_to_keep;
    if (elements_to_delete > 0) {
      memmove(f->r_axis, f->r_axis + elements_to_delete, elements_to_keep * sizeof *f->r_axis);
      f->nr_axis = elements_to_keep;
    }
    if (sta.time < 1.0 || time_roll < 1.0)
      return;
    omega_roll = dot3(o, roll_axis);
    for (d = 0; d < 3; d++)
      o[d] += -omega_roll * roll_axis[d];
    clip(0.025, 1e4, sta.dt, 0, a + 0.05 * da, 0.0, &correction_magnitude, &dummy);
    for (d = 0; d < 3; d++)
      o[d] += -correction_magnitude * roll_axis[d];
  }
}
static void fish_vel(void) {
  long long i;
  int k;
  if (sim.nfish == 0)
    return;
#pragma omp parallel for schedule(dynamic, 1)
  for (i = 0; i < sta.nblk; ++i) {
    int k;
    for (k = 0; k < sim.nfish; k++)
      fish_mom_blk(i, &sta.fish[k]);
  }
  for (k = 0; k < sim.nfish; k++) {
    struct Fish *f = &sta.fish[k];
    Real M[M_N] = {0};
    long long i;
    int q;
    for (i = 0; i < sta.nblk; i++) {
      struct ObstacleBlock *o = oblock(f, i);
      int q;
      if (o == NULL)
        continue;
      for (q = 0; q < M_N; q++)
        M[q] += o->mom[q];
    }
    MPI_Allreduce(MPI_IN_PLACE, M, M_N, MPI_Real, MPI_SUM, sim.comm);
    f->pen_m = M[M_GfX];
    f->pen_cm[0] = M[M_GpX];
    f->pen_cm[1] = M[M_GpY];
    f->pen_cm[2] = M[M_GpZ];
    for (q = 0; q < 6; q++)
      f->pen_j[q] = M[M_Gj0 + q];
    f->pen_lmom[0] = M[M_GuX];
    f->pen_lmom[1] = M[M_GuY];
    f->pen_lmom[2] = M[M_GuZ];
    f->pen_amom[0] = M[M_GaX];
    f->pen_amom[1] = M[M_GaY];
    f->pen_amom[2] = M[M_GaZ];
    fish_solve(f);
  }
}
static void hit_lever(Real *Rc, Real *R, Real *N, Real *I, Real *J) {
  Real m00 = I[0];
  Real m01 = I[3];
  Real m02 = I[4];
  Real m11 = I[1];
  Real m12 = I[5];
  Real m22 = I[2];
  Real a00 = m22 * m11 - m12 * m12;
  Real a01 = m02 * m12 - m22 * m01;
  Real a02 = m01 * m12 - m02 * m11;
  Real a11 = m22 * m00 - m02 * m02;
  Real a12 = m01 * m02 - m00 * m12;
  Real a22 = m00 * m11 - m01 * m01;
  Real determinant = 1.0 / ((m00 * a00) + (m01 * a01) + (m02 * a02));
  Real aux_0;
  Real aux_1;
  Real aux_2;
  a00 *= determinant;
  a01 *= determinant;
  a02 *= determinant;
  a11 *= determinant;
  a12 *= determinant;
  a22 *= determinant;
  aux_0 = (Rc[1] - R[1]) * N[2] - (Rc[2] - R[2]) * N[1];
  aux_1 = (Rc[2] - R[2]) * N[0] - (Rc[0] - R[0]) * N[2];
  aux_2 = (Rc[0] - R[0]) * N[1] - (Rc[1] - R[1]) * N[0];
  J[0] = a00 * aux_0 + a01 * aux_1 + a02 * aux_2;
  J[1] = a01 * aux_0 + a11 * aux_1 + a12 * aux_2;
  J[2] = a02 * aux_0 + a12 * aux_1 + a22 * aux_2;
}
static void hit_impulse(Real m1, Real m2, Real *I1, Real *I2, Real *v1, Real *v2, Real *o1, Real *o2,
                        Real *C1, Real *C2, Real N[3], Real C[3], Real *vc1, Real *vc2, Real *hv1, Real *hv2,
                        Real *ho1, Real *ho2) {
  Real e = 1.0;
  Real J1[3];
  Real J2[3];
  Real nom;
  Real denom;
  Real impulse;
  int d;
  hit_lever(C, C1, N, I1, J1);
  hit_lever(C, C2, N, I2, J2);
  nom = (e + 1) * ((vc1[0] - vc2[0]) * N[0] + (vc1[1] - vc2[1]) * N[1] + (vc1[2] - vc2[2]) * N[2]);
  denom = -(1.0 / m1 + 1.0 / m2) +
          -((J1[1] * (C[2] - C1[2]) - J1[2] * (C[1] - C1[1])) * N[0] +
            (J1[2] * (C[0] - C1[0]) - J1[0] * (C[2] - C1[2])) * N[1] +
            (J1[0] * (C[1] - C1[1]) - J1[1] * (C[0] - C1[0])) * N[2]) -
          ((J2[1] * (C[2] - C2[2]) - J2[2] * (C[1] - C2[1])) * N[0] +
           (J2[2] * (C[0] - C2[0]) - J2[0] * (C[2] - C2[2])) * N[1] +
           (J2[0] * (C[1] - C2[1]) - J2[1] * (C[0] - C2[0])) * N[2]);
  impulse = nom / (denom + 1e-21);
  for (d = 0; d < 3; d++) {
    hv1[d] = v1[d] + (N[d] / m1) * impulse;
    hv2[d] = v2[d] + (-N[d] / m2) * impulse;
    ho1[d] = o1[d] + J1[d] * impulse;
    ho2[d] = o2[d] - J2[d] * impulse;
  }
}
struct CollisionSide {
  Real M, Pos[3], Mom[3], vec[3];
};
struct CollisionInfo {
  struct CollisionSide s[2];
};
typedef char collision_info_is_20_reals[sizeof(struct CollisionInfo) == 20 * sizeof(Real) ? 1 : -1];
static void hit_cell(struct CollisionSide *cs, Real *magmax, struct Fish *f, struct ObstacleBlock *o, int x,
                     int y, int z, Real p[3]) {
  Real *U = f->vel, *om = f->omega, *C = f->com, *ud = o->udef[z][y][x];
  Real Mom[3], vec[3];
  int d;
  Real mag;
  Real norm;
  for (d = 0; d < 3; d++) {
    int e = (d + 1) % 3, f = (d + 2) % 3;
    Mom[d] = U[d] + om[e] * (p[f] - C[f]) - om[f] * (p[e] - C[e]) + ud[d];
    vec[d] = o->sdf[z + 1 + (d == 2)][y + 1 + (d == 1)][x + 1 + (d == 0)] -
             o->sdf[z + 1 - (d == 2)][y + 1 - (d == 1)][x + 1 - (d == 0)];
  }
  mag = dot3(Mom, Mom);
  norm = 1.0 / (sqrt(dot3(vec, vec)) + 1e-21);
  cs->M += 1;
  for (d = 0; d < 3; d++) {
    cs->Pos[d] += p[d];
    cs->vec[d] += vec[d] * norm;
  }
  if (mag > *magmax) {
    int d;
    *magmax = mag;
    for (d = 0; d < 3; d++)
      cs->Mom[d] = Mom[d];
  }
}
static void fish_pairs(int N, long long **start_out, long long **blk_out) {
  long long nb = sta.nblk, k, q, npair = (long long)N * N;
  int *nfish = ecalloc(nb + 1, sizeof *nfish);
  int *fish;
  long long *fstart = ecalloc(nb + 1, sizeof *fstart);
  long long *start = ecalloc(npair + 1, sizeof *start);
  long long *fill = ecalloc(npair, sizeof *fill);
  long long *blk;
  int i, a, b;
  for (i = 0; i < N; i++)
    for (k = 0; k < nb; k++)
      if (sta.fish[i].slot[k] >= 0)
        nfish[k]++;
  for (k = 0; k < nb; k++)
    fstart[k + 1] = fstart[k] + (nfish[k] >= 2 ? nfish[k] : 0);
  fish = emalloc((fstart[nb] + 1) * sizeof *fish);
  memset(nfish, 0, (nb + 1) * sizeof *nfish);
  for (i = 0; i < N; i++)
    for (k = 0; k < nb; k++)
      if (sta.fish[i].slot[k] >= 0 && fstart[k + 1] > fstart[k])
        fish[fstart[k] + nfish[k]++] = i;
  for (k = 0; k < nb; k++)
    for (a = fstart[k]; a < fstart[k + 1]; a++)
      for (b = fstart[k]; b < fstart[k + 1]; b++)
        if (a != b)
          start[(long long)fish[a] * N + fish[b] + 1]++;
  for (q = 1; q <= npair; q++)
    start[q] += start[q - 1];
  blk = emalloc((start[npair] + 1) * sizeof *blk);
  for (k = 0; k < nb; k++)
    for (a = fstart[k]; a < fstart[k + 1]; a++)
      for (b = fstart[k]; b < fstart[k + 1]; b++)
        if (a != b) {
          long long p = (long long)fish[a] * N + fish[b];
          blk[start[p] + fill[p]++] = k;
        }
  free(nfish);
  free(fstart);
  free(fish);
  free(fill);
  *start_out = start;
  *blk_out = blk;
}
static void fish_hit(void) {
  int N = sim.nfish;
  struct CollisionInfo *collisions;
  int i;
  struct {
    double v;
    int r;
  } *mx;
  int s;
  int j;
  long long *pair_start, *pair_blk, k, q;
  if (N < 2)
    return;
  collisions = ecalloc(N, sizeof *collisions);
  fish_pairs(N, &pair_start, &pair_blk);
  for (i = 0; i < N; ++i) {
    struct CollisionInfo *coll = &collisions[i];
    struct Fish *fi = &sta.fish[i];
    int j;
    for (j = 0; j < N; ++j) {
      struct Fish *fj;
      Real imagmax, jmagmax;
      long long p;
      if (i == j)
        continue;
      fj = &sta.fish[j];
      imagmax = 0.0;
      jmagmax = 0.0;
      p = (long long)i * N + j;
      for (q = pair_start[p]; q < pair_start[p + 1]; ++q) {
        struct ObstacleBlock *ib, *jb;
        int z;
        int y;
        int x;
        k = pair_blk[q];
        ib = oblock(fi, k);
        jb = oblock(fj, k);
        for (z = 0; z < BS; ++z)
          for (y = 0; y < BS; ++y)
            for (x = 0; x < BS; ++x) {
              Real pos[3];
              if (ib->chi[z][y][x] <= 0.0 || jb->chi[z][y][x] <= 0.0)
                continue;
              blk_pos(&sta.blk[k], x, y, z, pos);
              hit_cell(&coll->s[0], &imagmax, fi, ib, x, y, z, pos);
              hit_cell(&coll->s[1], &jmagmax, fj, jb, x, y, z, pos);
            }
      }
    }
  }
  free(pair_start);
  free(pair_blk);
  mx = emalloc((2 * N + 1) * sizeof *mx);
  for (i = 0; i < N; i++)
    for (s = 0; s < 2; s++) {
      Real *M = collisions[i].s[s].Mom;
      mx[2 * i + s].v = M[0] * M[0] + M[1] * M[1] + M[2] * M[2];
      mx[2 * i + s].r = sim.rank;
    }
  MPI_Allreduce(MPI_IN_PLACE, mx, 2 * N, MPI_DOUBLE_INT, MPI_MAXLOC, sim.comm);
  for (i = 0; i < N; i++)
    for (s = 0; s < 2; s++) {
      Real *M = collisions[i].s[s].Mom;
      if (mx[2 * i + s].r != sim.rank)
        M[0] = M[1] = M[2] = 0;
    }
  MPI_Allreduce(MPI_IN_PLACE, collisions, 20 * N, MPI_Real, MPI_SUM, sim.comm);
  for (i = 0; i < N; ++i)
    for (j = i + 1; j < N; ++j) {
      struct Fish *fi = &sta.fish[i], *fj = &sta.fish[j];
      struct CollisionSide *a = &collisions[i].s[0], *b = &collisions[i].s[1];
      struct CollisionSide *oa = &collisions[j].s[0], *ob = &collisions[j].s[1];
      Real tolerance = 0.001;
      Real norm_i;
      Real norm_j;
      Real m[3], Nn[3], C[3];
      int d;
      Real inorm;
      Real proj_vel;
      int iforced;
      int jforced;
      Real m1;
      Real m2;
      Real ho1[3], ho2[3], hv1[3], hv2[3];
      if (a->M < tolerance || b->M < tolerance)
        continue;
      if (oa->M < tolerance || ob->M < tolerance)
        continue;
      if (fabs(a->Pos[0] / a->M - oa->Pos[0] / oa->M) > 0.2 ||
          fabs(a->Pos[1] / a->M - oa->Pos[1] / oa->M) > 0.2 ||
          fabs(a->Pos[2] / a->M - oa->Pos[2] / oa->M) > 0.2)
        continue;
      norm_i = sqrt(a->vec[0] * a->vec[0] + a->vec[1] * a->vec[1] + a->vec[2] * a->vec[2]);
      norm_j = sqrt(b->vec[0] * b->vec[0] + b->vec[1] * b->vec[1] + b->vec[2] * b->vec[2]);
      for (d = 0; d < 3; d++)
        m[d] = a->vec[d] / norm_i - b->vec[d] / norm_j;
      inorm = 1.0 / sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
      for (d = 0; d < 3; d++)
        Nn[d] = m[d] * inorm;
      proj_vel =
          (b->Mom[0] - a->Mom[0]) * Nn[0] + (b->Mom[1] - a->Mom[1]) * Nn[1] + (b->Mom[2] - a->Mom[2]) * Nn[2];
      if (proj_vel <= 0)
        continue;
      for (d = 0; d < 3; d++)
        C[d] = 0.5 * (a->Pos[d] * (1.0 / a->M) + b->Pos[d] * (1.0 / b->M));
      iforced = fi->forced[0] || fi->forced[1] || fi->forced[2];
      jforced = fj->forced[0] || fj->forced[1] || fj->forced[2];
      m1 = iforced ? 1e10 * fi->mass : fi->mass;
      m2 = jforced ? 1e10 * fj->mass : fj->mass;
      hit_impulse(m1, m2, fi->J, fj->J, fi->vel, fj->vel, fi->omega, fj->omega, fi->com, fj->com, Nn, C,
                  a->Mom, b->Mom, hv1, hv2, ho1, ho2);
      for (d = 0; d < 3; d++) {
        fi->vel[d] = fi->hit_vel[d] = hv1[d];
        fj->vel[d] = fj->hit_vel[d] = hv2[d];
        fi->omega[d] = fi->hit_omega[d] = ho1[d];
        fj->omega[d] = fj->hit_omega[d] = ho2[d];
      }
      fi->hit_time = 0.01 * sta.dt;
      fj->hit_time = 0.01 * sta.dt;
    }
  free(mx);
  free(collisions);
}
static void fish_pen_blk(long long i, struct Fish *f) {
  struct ObstacleBlock *o = oblock(f, i);
  struct Blk *blk;
  Real dt, lambda;
  Real *b;
  Real *b_chi;
  Real *CM;
  Real *vel;
  Real *omega;
  Real lambda_fac;
  int iz;
  int iy;
  int ix;
  if (o == NULL)
    return;
  blk = &sta.blk[i];
  dt = sta.dt;
  lambda = sta.lambda;
  b = fld(i, F_VEL);
  b_chi = fld(i, F_CHI);
  CM = f->com;
  vel = f->vel;
  omega = f->omega;
  lambda_fac = lambda;
  for (iz = 0; iz < BS; ++iz)
    for (iy = 0; iy < BS; ++iy)
      for (ix = 0; ix < BS; ++ix) {
        Real p[3];
        int d;
        Real *U;
        Real X;
        Real pen_fac;
        if (b_chi[IDX(ix, iy, iz)] > o->chi[iz][iy][ix])
          continue;
        if (o->chi[iz][iy][ix] <= 0)
          continue;
        blk_pos(blk, ix, iy, iz, p);
        for (d = 0; d < 3; d++)
          p[d] -= CM[d];
        U = o->udef[iz][iy][ix];
        X = o->chi[iz][iy][ix] > 0.5 ? 1.0 : 0.0;
        pen_fac = X * lambda_fac / (1 + X * lambda_fac * dt);
        for (d = 0; d < 3; d++) {
          int e = (d + 1) % 3, f = (d + 2) % 3;
          Real U_TOT = vel[d] + omega[e] * p[f] - omega[f] * p[e] + U[d];
          Real *u = &b[d * BS3 + IDX(ix, iy, iz)];
          *u = *u + dt * (pen_fac * (U_TOT - *u));
        }
      }
}
static void fish_pen(void) {
  long long i;
  if (sim.nfish == 0)
    return;
  fish_hit();
#pragma omp parallel for schedule(dynamic, 1)
  for (i = 0; i < sta.nblk; ++i) {
    int k;
    for (k = 0; k < sim.nfish; k++)
      fish_pen_blk(i, &sta.fish[k]);
  }
}
static void k_prhs(struct Lab *l, long long i) {
  Real dt = sta.dt;
  Real h = sta.blk[i].h, fac = 0.5 * h * h / dt;
  Real *c = fld(i, F_CHI);
  Real *p = fld(i, F_LHS);
  int z;
  int y;
  int x;
  int f;
  for (z = 0; z < BS; ++z)
    for (y = 0; y < BS; ++y)
      for (x = 0; x < BS; ++x) {
        Real div_us;
        p[IDX(x, y, z)] = fac * (L(x + 1, y, z, 0) - L(x - 1, y, z, 0) + L(x, y + 1, z, 1) -
                                 L(x, y - 1, z, 1) + L(x, y, z + 1, 2) - L(x, y, z - 1, 2));
        div_us = L(x + 1, y, z, 3) - L(x - 1, y, z, 3) + L(x, y + 1, z, 4) - L(x, y - 1, z, 4) +
                 L(x, y, z + 1, 5) - L(x, y, z - 1, 5);
        p[IDX(x, y, z)] += -c[IDX(x, y, z)] * fac * div_us;
      }
  for (f = 0; f < 6; f++) {
    Real *F = fc_face(i, f, 0);
    int d;
    Real s;
    int k;
    if (F == NULL)
      continue;
    d = f / 2;
    s = f % 2 ? -1.0 : 1.0;
    for (k = 0; k < BS * BS; k++) {
      int cc[3], n[3];
      face_cell(f, k, cc, n);
      F[k] = s * (fac * (LC(n, d) + LC(cc, d)) -
                  c[IDX(cc[0], cc[1], cc[2])] * fac * (LC(n, 3 + d) + LC(cc, 3 + d)));
    }
  }
}
static struct Stencil st_prhs = {F_VEL, 6, 1, 0, 0, F_LHS, 1, k_prhs};
static void k_divp(struct Lab *l, long long i) {
  Real *b = fld(i, F_TMP);
  Real fac = sta.blk[i].h;
  int z;
  int y;
  int x;
  for (z = 0; z < BS; ++z)
    for (y = 0; y < BS; ++y)
      for (x = 0; x < BS; ++x)
        b[IDX(x, y, z)] =
            fac * (L(x + 1, y, z, 0) + L(x - 1, y, z, 0) + L(x, y + 1, z, 0) + L(x, y - 1, z, 0) +
                   L(x, y, z + 1, 0) + L(x, y, z - 1, 0) - 6.0 * L(x, y, z, 0));
  face_grad(l, i, 0, fac);
}
static struct Stencil st_divp = {F_PRES, 1, 1, 0, -1, F_TMP, 3, k_divp};
static void k_gradp(struct Lab *l, long long i) {
  Real dt = sta.dt;
  Real h = sta.blk[i].h;
  Real *o = fld(i, F_TMP);
  Real fac = -0.5 * dt * h * h;
  int z;
  int y;
  int x;
  int f;
  for (z = 0; z < BS; ++z)
    for (y = 0; y < BS; ++y)
      for (x = 0; x < BS; ++x) {
        int a;
        for (a = 0; a < 3; a++)
          o[a * BS3 + IDX(x, y, z)] = fac * (L(x + (a == 0), y + (a == 1), z + (a == 2), 0) -
                                             L(x - (a == 0), y - (a == 1), z - (a == 2), 0));
      }
  for (f = 0; f < 6; f++)
    face_sum(l, i, f, 0, f / 2, fac);
}
static struct Stencil st_gradp = {F_PRES, 1, 1, 0, -1, F_TMP, 3, k_gradp};
static void k_vort(struct Lab *l, long long i) {
  Real h = sta.blk[i].h;
  Real inv2h = .5 * h * h;
  Real *o = fld(i, F_TMP);
  int z;
  int y;
  int x;
  int f;
  for (z = 0; z < BS; ++z)
    for (y = 0; y < BS; ++y)
      for (x = 0; x < BS; ++x) {
        int a;
        for (a = 0; a < 3; a++) {
          int b = (a + 1) % 3, c = (a + 2) % 3;
          o[a * BS3 + IDX(x, y, z)] = inv2h * ((LS(b, 1, c) - LS(b, -1, c)) - (LS(c, 1, b) - LS(c, -1, b)));
        }
      }
  for (f = 0; f < 6; f++) {
    int d = f / 2;
    int a;
    for (a = 0; a < 3; a++)
      if (a != d)
        face_sum(l, i, f, 3 - a - d, a, d == (a + 1) % 3 ? inv2h : -inv2h);
  }
}
static struct Stencil st_vort = {F_VEL, 3, 1, 0, 0, F_TMP, 3, k_vort};
static void k_q(struct Lab *l, long long i) {
  Real inv2h = .5 / sta.blk[i].h;
  Real *o = fld(i, F_LHS);
  int z;
  int y;
  int x;
  for (z = 0; z < BS; ++z)
    for (y = 0; y < BS; ++y)
      for (x = 0; x < BS; ++x) {
        Real g[3][3], q;
        int a;
        int b;
        for (a = 0; a < 3; a++)
          for (b = 0; b < 3; b++)
            g[a][b] = inv2h * (LS(b, 1, a) - LS(b, -1, a));
        q = 0;
        for (a = 0; a < 3; a++)
          for (b = 0; b < 3; b++)
            q -= 0.5 * g[a][b] * g[b][a];
        o[IDX(x, y, z)] = q;
      }
}
static struct Stencil st_q = {F_VEL, 3, 1, 0, 0, F_LHS, 0, k_q};
static void qcrit(void) { stencil_apply(&st_q); }
static void vorticity(void) {
  long long i;
  stencil_apply(&st_vort);
#pragma omp parallel for
  for (i = 0; i < sta.nblk; i++) {
    Real h = sta.blk[i].h;
    Real fac = 1.0 / (h * h * h);
    Real *b = fld(i, F_TMP);
    int j;
    for (j = 0; j < 3 * BS3; j++)
      b[j] *= fac;
  }
}
static void fish_tmpv(void) {
  int k;
  for (k = 0; k < sim.nfish; k++) {
    struct Fish *f = &sta.fish[k];
    long long i;
#pragma omp parallel for schedule(dynamic, 1)
    for (i = 0; i < sta.nblk; ++i) {
      struct ObstacleBlock *o = oblock(f, i);
      Real *c;
      Real *b;
      int z;
      int y;
      int x;
      if (o == NULL)
        continue;
      c = fld(i, F_CHI);
      b = fld(i, F_TMP);
      for (z = 0; z < BS; ++z)
        for (y = 0; y < BS; ++y)
          for (x = 0; x < BS; ++x) {
            if (c[IDX(x, y, z)] > o->chi[z][y][x])
              continue;
            b[0 * BS3 + IDX(x, y, z)] += o->udef[z][y][x][0];
            b[1 * BS3 + IDX(x, y, z)] += o->udef[z][y][x][1];
            b[2 * BS3 + IDX(x, y, z)] += o->udef[z][y][x][2];
          }
    }
  }
}
static void projection(void) {
  static Real *p_old;
  static long long cap;
  long long N = sta.nblk * BS3;
  long long i;
  Real avg;
  Real avg1;
  Real quantities[2];
  if (N > cap) {
    free(p_old);
    p_old = emalloc(N * sizeof(Real));
    cap = N;
  }
#pragma omp parallel for
  for (i = 0; i < sta.nblk; i++) {
    memcpy(p_old + i * BS3, fld(i, F_PRES), BS3 * sizeof(Real));
    memset(fld(i, F_TMP), 0, 3 * BS3 * sizeof(Real));
  }
  if (sim.nfish > 0)
    fish_tmpv();
  stencil_apply(&st_prhs);
  if (sta.step > STEP_2ND) {
    long long i;
    stencil_apply(&st_divp);
#pragma omp parallel for
    for (i = 0; i < sta.nblk; i++) {
      Real *b = fld(i, F_TMP);
      Real *LHS = fld(i, F_LHS);
      Real *p = fld(i, F_PRES);
      int j;
      for (j = 0; j < BS3; j++) {
        LHS[j] -= b[j];
        p[j] = 0;
      }
    }
  } else {
    long long i;
#pragma omp parallel for
    for (i = 0; i < sta.nblk; i++) {
      memset(fld(i, F_PRES), 0, BS3 * sizeof(Real));
    }
  }
  pois_solve();
  avg = 0;
  avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
  for (i = 0; i < sta.nblk; i++) {
    Real *P = fld(i, F_PRES);
    Real vv = sta.blk[i].h * sta.blk[i].h * sta.blk[i].h;
    int j;
    for (j = 0; j < BS3; j++) {
      avg += P[j] * vv;
      avg1 += vv;
    }
  }
  quantities[0] = avg;
  quantities[1] = avg1;
  MPI_Allreduce(MPI_IN_PLACE, quantities, 2, MPI_Real, MPI_SUM, sim.comm);
  avg = quantities[0];
  avg1 = quantities[1];
  avg = avg / avg1;
#pragma omp parallel for
  for (i = 0; i < sta.nblk; i++) {
    Real *P = fld(i, F_PRES);
    int j;
    for (j = 0; j < BS3; j++)
      P[j] -= avg;
  }
  if (sta.step > STEP_2ND) {
    long long i;
#pragma omp parallel for
    for (i = 0; i < sta.nblk; i++) {
      Real *p = fld(i, F_PRES);
      int j;
      for (j = 0; j < BS3; j++)
        p[j] += p_old[i * BS3 + j];
    }
  }
  stencil_apply(&st_gradp);
#pragma omp parallel for
  for (i = 0; i < sta.nblk; i++) {
    Real h = sta.blk[i].h;
    Real fac = 1.0 / (h * h * h);
    Real *grad_p = fld(i, F_TMP);
    Real *v = fld(i, F_VEL);
    int j;
    for (j = 0; j < 3 * BS3; j++)
      v[j] += fac * grad_p[j];
  }
}
static Real sta_umax(void) {
  Real umax = 0;
  long long i;
#pragma omp parallel for reduction(max : umax)
  for (i = 0; i < sta.nblk; i++) {
    Real *b = fld(i, F_VEL);
    int j;
    for (j = 0; j < BS3; j++) {
      Real advu = fabs(b[0 * BS3 + j] + sta.uinf[0]);
      Real advv = fabs(b[1 * BS3 + j] + sta.uinf[1]);
      Real advw = fabs(b[2 * BS3 + j] + sta.uinf[2]);
      Real ul = advu;
      if (ul < advv)
        ul = advv;
      if (ul < advw)
        ul = advw;
      if (umax < ul)
        umax = ul;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_Real, MPI_MAX, sim.comm);
  return umax;
}
static Real sta_dt(void) {
  Real dt_old = sta.dt;
  Real hmin;
  Real cfl;
  sta.dt_old = sta.dt;
  hmin = sim.hmin;
  cfl = sim.cfl;
  sta.umax = sta_umax();
  if (sta.umax > sim.umax)
    fatal("maxU = %g exceeded umax = %g", sta.umax, sim.umax);
  if (cfl > 0) {
    Real dt_diffusion = (1.0 / 6.0) * hmin * hmin / (sim.nu + (1.0 / 6.0) * hmin * sta.umax);
    Real dt_advection = hmin / (sta.umax + 1e-8);
    if (sta.step < sim.rampup) {
      Real x = sta.step / (Real)sim.rampup;
      Real ramp_cfl = exp(log(1e-3) * (1 - x) + log(cfl) * x);
      Real b = ramp_cfl * dt_advection;
      sta.dt = b < dt_diffusion ? b : dt_diffusion;
    } else {
      Real b = cfl * dt_advection;
      sta.dt = b < dt_diffusion ? b : dt_diffusion;
    }
  } else {
    cfl = (sta.umax + 1e-8) * sta.dt / hmin;
  }
  if (sta.dt <= 0)
    fatal("dt <= 0: CFL=%g hmin=%g umax=%g", cfl, hmin, sta.umax);
  if (sim.dlm > 0)
    sta.lambda = sim.dlm / sta.dt;
  if (sim.rank == 0)
    printf("main.c: step: %d, time: %f\n", sta.step, sta.time);
  if (sta.step >= STEP_2ND) {
    Real a = dt_old;
    Real b = sta.dt;
    Real c1 = -(a + b) / (a * b);
    Real c2 = b / (a + b) / a;
    sta.coef_u[0] = -b * (c1 + c2);
    sta.coef_u[1] = b * c1;
    sta.coef_u[2] = b * c2;
  }
  return sta.dt;
}
static int advance(Real dt) {
  if (sim.tdump > 0 && sta.time >= sta.next_dump) {
    char path[FILENAME_MAX];
    sta.next_dump += sim.tdump;
    snprintf(path, sizeof path, "vel.%08d", sta.step);
    fprintf(stderr, "main.c: %s\n", path);
    io_dump(sta.time, path);
  }
  if (sta.step % 20 == 0 || sta.step < 10)
    mesh_adapt();
  fish_build();
  advdiff();
  fish_vel();
  fish_pen();
  projection();
  sta.step++;
  sta.time += dt;
  if ((sim.tend > 0 && sta.time > sim.tend) || (sim.nsteps != 0 && sta.step >= sim.nsteps))
    return 1;
  return 0;
}
static void simulate(void) {
  for (;;) {
    Real dt = sta_dt();
    if (advance(dt))
      break;
  }
}
int main(int argc, char **argv) {
  int provided;
  char *content;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  sim.comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &sim.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &sim.size);
  content = param_parse(argc, argv);
  sta_init();
  lab_tables();
  pois_init();
  fish_parse(content);
  mesh_init();
  sta_fields();
  simulate();
  MPI_Finalize();
}
