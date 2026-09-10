#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double Real;
#define MPI_Real MPI_DOUBLE
enum { BS = 8 };

struct Params {
  int n;
  char *key[256];
  char *val[256];
};
static void params_add(struct Params *p, const char *key, size_t klen,
                       const char *val) {
  for (int i = 0; i < p->n; i++)
    if (strlen(p->key[i]) == klen && strncmp(p->key[i], key, klen) == 0)
      return;
  assert(p->n < 256);
  p->key[p->n] = strndup(key, klen);
  p->val[p->n] = strdup(val);
  p->n++;
}
static void params_free(struct Params *p) {
  for (int i = 0; i < p->n; i++) {
    free(p->key[i]);
    free(p->val[i]);
  }
  p->n = 0;
}
static int params_isnumber(const char *s) {
  char *end;
  strtod(s, &end);
  return end != s;
}
static void params_from_argv(struct Params *p, int argc, char **argv) {
  p->n = 0;
  for (int i = 1; i < argc; i++)
    if (argv[i][0] == '-') {
      int j, count = 0;
      size_t len = 1;
      for (j = i + 1; j < argc; j++) {
        if (argv[j][0] == '-' && !params_isnumber(argv[j]))
          break;
        len += strlen(argv[j]) + 1;
        count++;
      }
      char *values = (char *)malloc(len + 5);
      values[0] = '\0';
      if (count == 0)
        strcpy(values, "true");
      for (j = i + 1; j < i + 1 + count; j++) {
        if (j > i + 1)
          strcat(values, " ");
        strcat(values, argv[j]);
      }
      const char *key = argv[i] + 1;
      if (key[0] == '+')
        key++;
      params_add(p, key, strlen(key), values);
      free(values);
      i += count;
    }
}
static void params_from_line(struct Params *p, const char *line) {
  p->n = 0;
  const char *s = line;
  for (;;) {
    const char *eq = strchr(s, '=');
    if (eq == NULL)
      break;
    const char *k0 = s, *k1 = eq;
    while (k0 < k1 && isspace(*k0))
      k0++;
    while (k1 > k0 && isspace(k1[-1]))
      k1--;
    const char *v0 = eq + 1;
    const char *v1 = strchr(v0, ' ');
    if (v1 == NULL)
      v1 = v0 + strlen(v0);
    s = *v1 ? v1 + 1 : v1;
    const char *w0 = v0, *w1 = v1;
    while (w0 < w1 && isspace(*w0))
      w0++;
    while (w1 > w0 && isspace(w1[-1]))
      w1--;
    char *val = strndup(w0, w1 - w0);
    params_add(p, k0, k1 - k0, val);
    free(val);
  }
}
static const char *param_get(const struct Params *p, const char *key) {
  if (*key == '-')
    key++;
  if (*key == '+')
    key++;
  for (int i = 0; i < p->n; i++)
    if (strcmp(p->key[i], key) == 0)
      return p->val[i];
  return NULL;
}
static Real param_real(const struct Params *p, const char *key, Real def) {
  const char *v = param_get(p, key);
  return v ? atof(v) : def;
}
static int param_int(const struct Params *p, const char *key, int def) {
  const char *v = param_get(p, key);
  return v ? atoi(v) : def;
}
static int param_bool(const struct Params *p, const char *key, int def) {
  const char *v = param_get(p, key);
  if (v == NULL)
    return def;
  if (strcmp(v, "0") == 0 || strcmp(v, "false") == 0)
    return 0;
  return 1;
}
static const char *param_str(const struct Params *p, const char *key,
                             const char *def) {
  const char *v = param_get(p, key);
  return v ? v : def;
}

struct Midline {
  Real length, Tperiod, phaseShift, h, waveLength, amplitudeFactor;
  Real fracRefined, fracMid, dSmid_tgt, dSrefine_tgt, dSmid, dSref;
  int Nmid, Nend, Nm;
  Real *rS, *rX, *rY, *rZ, *vX, *vY, *vZ;
  Real *norX, *norY, *norZ, *vNorX, *vNorY, *vNorZ;
  Real *binX, *binY, *binZ, *vBinX, *vBinY, *vBinZ;
  Real *width, *height;
  Real *rK, *vK, *rC, *vC, *rT, *vT;
  Real quaternion_internal[4], angvel_internal[3];
  Real time0, timeshift;
  Real sched_p0[6], sched_p1[6], sched_dp0[6], sched_t0, sched_t1;
  Real alpha, dalpha, beta, dbeta, gamma, dgamma;
};

struct Fish {
  int id;
  Real length;
  Real position[3], absPos[3], quaternion[4], transVel[3], angVel[3];
  Real transVel_imposed[3];
  int bFixFrameOfRef[3], bForcedInSimFrame[3], bBlockRotation[3];
  int bCorrectPosition, bCorrectPositionZ, bCorrectRoll;
  Real origC[3], wyp, wzp;
  Real old_position[3], old_absPos[3], old_quaternion[4];
  Real centerOfMass[3], mass, J[6], transVel_correction[3], angVel_correction[3];
  Real penalM, penalCM[3], penalJ[6], penalLmom[3], penalAmom[3];
  Real transVel_computed[3], angVel_computed[3];
  Real collision_counter, u_collision[3], o_collision[3];
  Real (*r_axis)[4];
  int nr_axis;
  struct Midline m;
  struct ObstacleBlock **oblock;
  int nmyblk, *myblk, *seg_start, *seg_idx, nseg_idx;
};

enum { F_CHI = 0, F_PRES = 1, F_VEL = 2, F_TMP = 5, F_LHS = 8, F_N = 9 };
enum { BS3 = BS * BS * BS, BLK_S = F_N * BS3 };
struct Blk {
  int level, ix, iy, iz;
  long long Z;
  Real h, origin[3];
};
enum {
  M_V, M_FX, M_FY, M_FZ, M_TX, M_TY, M_TZ, M_J0, M_J1, M_J2, M_J3, M_J4, M_J5,
  M_GfX, M_GpX, M_GpY, M_GpZ, M_Gj0, M_Gj1, M_Gj2, M_Gj3, M_Gj4, M_Gj5,
  M_GuX, M_GuY, M_GuZ, M_GaX, M_GaY, M_GaZ, M_N
};
struct ObstacleBlock {
  Real chi[BS][BS][BS];
  Real udef[BS][BS][BS][3];
  Real sdfLab[BS + 2][BS + 2][BS + 2];
  int nPoints, filled;
  Real CoM_x, CoM_y, CoM_z, mass;
  Real mom[M_N];
};
struct Segment {
  Real safe_distance;
  int s0, s1;
  Real normalI[3], normalJ[3], normalK[3];
  Real w[3], c[3];
  Real objBoxLabFr[3][2], objBoxObjFr[3][2];
};
static struct Sim {
  MPI_Comm comm;
  int rank, size;
  int bpdx, bpdy, bpdz, levelMax, levelStart;
  Real extents[3], maxextent, hmin, hmax, h0;
  Real Rtol, Ctol, CFL, nu, lambda, dt, time, endTime, dumpTime, nextDumpTime;
  Real dt_old, uMax_measured, uMax_allowed, DLM, PoissonErrorTol, PoissonErrorTolRel;
  Real uinf[3], coefU[3];
  int step, step_2nd_start, MeshChanged, StaticObstacles;
  int rampup, nsteps, bMeanConstraint;
  long long nblk;
  struct Blk *blk;
  Real *fld;
  int nfish;
  struct Fish *fish;
} sim;
#define BLK(i) (sim.fld + (long long)(i) * BLK_S)

static Real *ralloc(int n) { return (Real *)malloc(n * sizeof(Real)); }

static Real d_ds(const struct Midline *m, int idx, const Real *vals,
                 int maxidx) {
  const Real *rS = m->rS;
  if (idx == 0)
    return (vals[idx + 1] - vals[idx]) / (rS[idx + 1] - rS[idx]);
  else if (idx == maxidx - 1)
    return (vals[idx] - vals[idx - 1]) / (rS[idx] - rS[idx - 1]);
  else
    return 0.5 * ((vals[idx + 1] - vals[idx]) / (rS[idx + 1] - rS[idx]) +
                  (vals[idx] - vals[idx - 1]) / (rS[idx] - rS[idx - 1]));
}

static void natural_cubic_spline(const Real *x, const Real *y, unsigned n,
                                 const Real *xx, Real *yy, unsigned nn) {
  Real *y2 = ralloc(n);
  Real *u = ralloc(n - 1);
  y2[0] = u[0] = 0.0;
  for (unsigned i = 1; i < n - 1; i++) {
    const Real sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
    const Real p = sig * y2[i - 1] + 2.0;
    y2[i] = (sig - 1.0) / p;
    u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
           (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
  }
  y2[n - 1] = 0;
  for (unsigned k = n - 2; k > 0; k--)
    y2[k] = y2[k] * y2[k + 1] + u[k];
  for (unsigned j = 0; j < nn; j++) {
    unsigned klo = 0;
    unsigned khi = n - 1;
    unsigned k = 0;
    while (khi - klo > 1) {
      k = (khi + klo) >> 1;
      if (x[k] > xx[j])
        khi = k;
      else
        klo = k;
    }
    const Real h = x[khi] - x[klo];
    if (fabs(h) < 2.2e-16) {
      fprintf(stderr, "Interpolation points must be distinct!");
      abort();
    }
    const Real a = (x[khi] - xx[j]) / h;
    const Real b = (xx[j] - x[klo]) / h;
    yy[j] = a * y[klo] + b * y[khi] +
            ((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * (h * h) /
                6;
  }
  free(y2);
  free(u);
}
static void cubic_interpolation(Real x0, Real x1, Real x, Real y0, Real y1,
                                Real dy0, Real dy1, Real *y, Real *dy) {
  const Real xrel = (x - x0);
  const Real deltax = (x1 - x0);
  const Real a = (dy0 + dy1) / (deltax * deltax) -
                 2 * (y1 - y0) / (deltax * deltax * deltax);
  const Real b = (-2 * dy0 - dy1) / deltax + 3 * (y1 - y0) / (deltax * deltax);
  const Real c = dy0;
  const Real d = y0;
  *y = a * xrel * xrel * xrel + b * xrel * xrel + c * xrel + d;
  *dy = 3 * a * xrel * xrel + 2 * b * xrel + c;
}
static void sched_transition(struct Midline *m, Real t, Real tstart, Real tend,
                             const Real p0[6], const Real p1[6]) {
  if (t < tstart || t > tend)
    return;
  if (tstart < m->sched_t0)
    return;
  m->sched_t0 = tstart;
  m->sched_t1 = tend;
  for (int i = 0; i < 6; i++) {
    m->sched_p0[i] = p0[i];
    m->sched_p1[i] = p1[i];
  }
}
static void sched_gimme(struct Midline *m, Real t, const Real positions[6],
                        int Nfine, const Real *positions_fine,
                        Real *parameters_fine, Real *dparameters_fine) {
  Real *p0f = ralloc(Nfine);
  Real *p1f = ralloc(Nfine);
  Real *dp0f = ralloc(Nfine);
  natural_cubic_spline(positions, m->sched_p0, 6, positions_fine, p0f, Nfine);
  natural_cubic_spline(positions, m->sched_p1, 6, positions_fine, p1f, Nfine);
  natural_cubic_spline(positions, m->sched_dp0, 6, positions_fine, dp0f,
                       Nfine);
  if (t < m->sched_t0 || m->sched_t0 < 0) {
    for (int i = 0; i < Nfine; ++i) {
      parameters_fine[i] = p0f[i];
      dparameters_fine[i] = 0.0;
    }
  } else if (t > m->sched_t1) {
    for (int i = 0; i < Nfine; ++i) {
      parameters_fine[i] = p1f[i];
      dparameters_fine[i] = 0.0;
    }
  } else {
    for (int i = 0; i < Nfine; ++i)
      cubic_interpolation(m->sched_t0, m->sched_t1, t, p0f[i], p1f[i], dp0f[i],
                          0.0, &parameters_fine[i], &dparameters_fine[i]);
  }
  free(p0f);
  free(p1f);
  free(dp0f);
}

static void midline_init(struct Midline *m, Real L, Real Tp, Real phi, Real h,
                         Real ampFac) {
  m->length = L;
  m->Tperiod = Tp;
  m->phaseShift = phi;
  m->h = h;
  m->waveLength = 1;
  m->amplitudeFactor = ampFac;
  m->fracRefined = 0.1;
  m->fracMid = 1 - 2 * m->fracRefined;
  m->dSmid_tgt = h / sqrt(3);
  m->dSrefine_tgt = 0.125 * h;
  m->Nmid = (int)ceil(L * m->fracMid / m->dSmid_tgt / 8) * 8;
  m->dSmid = L * m->fracMid / m->Nmid;
  m->Nend =
      (int)ceil(m->fracRefined * L * 2 / (m->dSmid + m->dSrefine_tgt) / 4) * 4;
  m->dSref = m->fracRefined * L * 2 / m->Nend - m->dSmid;
  while (m->dSref < 0 && m->Nend > 4) {
    m->Nend -= 4;
    m->dSref = m->fracRefined * L * 2 / m->Nend - m->dSmid;
  }
  m->Nm = m->Nmid + 2 * m->Nend + 1;
  const int Nm = m->Nm;
  Real **arrays[] = {&m->rS,    &m->rX,    &m->rY,    &m->rZ,    &m->vX,
                     &m->vY,    &m->vZ,    &m->norX,  &m->norY,  &m->norZ,
                     &m->vNorX, &m->vNorY, &m->vNorZ, &m->binX,  &m->binY,
                     &m->binZ,  &m->vBinX, &m->vBinY, &m->vBinZ, &m->width,
                     &m->height, &m->rK,   &m->vK,    &m->rC,    &m->vC,
                     &m->rT,    &m->vT};
  for (size_t i = 0; i < sizeof arrays / sizeof *arrays; i++)
    *arrays[i] = ralloc(Nm);
  Real *rS = m->rS;
  const int Nend = m->Nend, Nmid = m->Nmid;
  const Real dSref = m->dSref, dSmid = m->dSmid;
  rS[0] = 0;
  int k = 0;
  for (int i = 0; i < Nend; ++i, k++)
    rS[k + 1] = rS[k] + dSref + (dSmid - dSref) * i / ((Real)Nend - 1.);
  for (int i = 0; i < Nmid; ++i, k++)
    rS[k + 1] = rS[k] + dSmid;
  for (int i = 0; i < Nend; ++i, k++)
    rS[k + 1] =
        rS[k] + dSref + (dSmid - dSref) * (Nend - i - 1) / ((Real)Nend - 1.);
  rS[k] = (L < rS[k]) ? L : rS[k];
  assert(k + 1 == Nm);
  m->quaternion_internal[0] = 1;
  m->quaternion_internal[1] = m->quaternion_internal[2] =
      m->quaternion_internal[3] = 0;
  m->angvel_internal[0] = m->angvel_internal[1] = m->angvel_internal[2] = 0;
  m->time0 = 0;
  m->timeshift = 0;
  for (int i = 0; i < 6; i++)
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
static void midline_free(struct Midline *m) {
  Real *arrays[] = {m->rS,    m->rX,    m->rY,    m->rZ,    m->vX,    m->vY,
                    m->vZ,    m->norX,  m->norY,  m->norZ,  m->vNorX, m->vNorY,
                    m->vNorZ, m->binX,  m->binY,  m->binZ,  m->vBinX, m->vBinY,
                    m->vBinZ, m->width, m->height, m->rK,   m->vK,    m->rC,
                    m->vC,    m->rT,    m->vT};
  for (size_t i = 0; i < sizeof arrays / sizeof *arrays; i++)
    free(arrays[i]);
}

static void bspline_basis(const Real x, const Real *const t, const int n,
                          Real *const B) {
  enum { K = 4 };
  Real b[K], deltal[K], deltar[K];
  int i, j, left;
  if (x >= t[n + K - 1]) {
    left = n - 1;
  } else {
    int lo = 0, hi = n + K - 1;
    while (hi > lo + 1) {
      const int mid = (hi + lo) >> 1;
      if (t[mid] > x)
        hi = mid;
      else
        lo = mid;
    }
    left = lo;
  }
  b[0] = 1;
  for (j = 0; j < K - 1; j++) {
    deltar[j] = t[left + j + 1] - x;
    deltal[j] = x - t[left - j];
    Real saved = 0;
    for (i = 0; i <= j; i++) {
      const Real term = b[i] / (deltar[i] + deltal[j - i]);
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
static void integrate_bspline(const Real *xc, const Real *yc, int n, Real length,
                              const Real *rS, Real *res, int Nm) {
  enum { K = 4 };
  Real len = 0;
  for (int i = 0; i < n - 1; i++) {
    len += sqrt(pow(xc[i] - xc[i + 1], 2) + pow(yc[i] - yc[i + 1], 2));
  }
  Real *t = ralloc(n + K);
  Real *B = ralloc(n);
  const Real delta = len / (n - 3);
  for (int i = 0; i < K; i++)
    t[i] = 0;
  for (int i = 0; i < n - 4; i++)
    t[K + i] = (i + 1) * delta;
  for (int i = n; i < n + K; i++)
    t[i] = len;
  Real ti = 0;
  for (int i = 0; i < Nm; ++i) {
    res[i] = 0;
    if (rS[i] > 0 && rS[i] < length) {
      const Real dtt = (rS[i] - rS[i - 1]) / 1e3;
      for (;;) {
        Real xi = 0;
        bspline_basis(ti, t, n, B);
        for (int j = 0; j < n; j++)
          xi += xc[j] * B[j];
        if (xi >= rS[i])
          break;
        if (ti + dtt > len)
          break;
        else
          ti += dtt;
      }
      for (int j = 0; j < n; j++)
        res[i] += yc[j] * B[j];
    }
  }
  free(t);
  free(B);
}
static void stefan_width(Real L, const Real *rS, Real *res, int Nm) {
  const Real sb = .04 * L;
  const Real st = .95 * L;
  const Real wt = .01 * L;
  const Real wh = .04 * L;
  for (int i = 0; i < Nm; ++i) {
    if (rS[i] <= 0 || rS[i] >= L)
      res[i] = 0;
    else {
      const Real s = rS[i];
      res[i] = (s < sb ? sqrt(2.0 * wh * s - s * s)
                       : (s < st ? wh - (wh - wt) * pow((s - sb) / (st - sb), 2)
                                 : (wt * (L - s) / (L - st))));
    }
  }
}
static void stefan_height(Real L, const Real *rS, Real *res, int Nm) {
  const Real a = 0.51 * L;
  const Real b = 0.08 * L;
  for (int i = 0; i < Nm; ++i) {
    if (rS[i] <= 0 || rS[i] >= L)
      res[i] = 0;
    else {
      const Real s = rS[i];
      res[i] = b * sqrt(1 - pow((s - a) / a, 2));
    }
  }
}
static void larval_width(Real L, const Real *rS, Real *res, int Nm) {
  const Real sb = .0862 * L;
  const Real st = .3448 * L;
  const Real wh = .0635 * L;
  const Real wt = .0254 * L;
  for (int i = 0; i < Nm; ++i) {
    if (rS[i] <= 0 || rS[i] >= L)
      res[i] = 0;
    else {
      const Real s = rS[i];
      res[i] = s < sb ? wh * sqrt(1 - pow((sb - s) / sb, 2))
                      : (s < st ? (-2 * (wt - wh) - wt * (st - sb)) *
                                          pow((s - sb) / (st - sb), 3) +
                                      (3 * (wt - wh) + wt * (st - sb)) *
                                          pow((s - sb) / (st - sb), 2) +
                                      wh
                                : (wt - wt * (s - st) / (L - st)));
    }
  }
}
static void larval_height(Real L, const Real *rS, Real *res, int Nm) {
  const Real s1 = 0.287 * L;
  const Real h1 = 0.072 * L;
  const Real s2 = 0.844 * L;
  const Real h2 = 0.041 * L;
  const Real s3 = 0.957 * L;
  const Real h3 = 0.071 * L;
  for (int i = 0; i < Nm; ++i) {
    if (rS[i] <= 0 || rS[i] >= L)
      res[i] = 0;
    else {
      const Real s = rS[i];
      res[i] =
          s < s1 ? (h1 * sqrt(1 - pow((s - s1) / s1, 2)))
                 : (s < s2 ? -2 * (h2 - h1) * pow((s - s1) / (s2 - s1), 3) +
                                 3 * (h2 - h1) * pow((s - s1) / (s2 - s1), 2) +
                                 h1
                           : (s < s3 ? -2 * (h3 - h2) *
                                               pow((s - s2) / (s3 - s2), 3) +
                                           3 * (h3 - h2) *
                                               pow((s - s2) / (s3 - s2), 2) +
                                           h2
                                     : (h3 * sqrt(1 - pow((s - s3) / (L - s3),
                                                          3)))));
    }
  }
}
static void danio_width(Real L, const Real *rS, Real *res, int Nm) {
  enum { nBreaksW = 11 };
  const Real breaksW[nBreaksW] = {0,   0.005, 0.01, 0.05, 0.1, 0.2,
                                  0.4, 0.6,   0.8,  0.95, 1.0};
  const Real coeffsW[nBreaksW - 1][4] = {
      {0.0015713, 2.6439, 0, -15410},
      {0.012865, 1.4882, -231.15, 15598},
      {0.016476, 0.34647, 2.8156, -39.328},
      {0.032323, 0.38294, -1.9038, 0.7411},
      {0.046803, 0.19812, -1.7926, 5.4876},
      {0.054176, 0.0042136, -0.14638, 0.077447},
      {0.049783, -0.045043, -0.099907, -0.12599},
      {0.03577, -0.10012, -0.1755, 0.62019},
      {0.013687, -0.0959, 0.19662, 0.82341},
      {0.0065049, 0.018665, 0.56715, -3.781}};
  for (int i = 0; i < Nm; ++i) {
    if (rS[i] <= 0 || rS[i] >= L)
      res[i] = 0;
    else {
      const Real sNormalized = rS[i] / L;
      int currentSegW = 1;
      while (sNormalized >= breaksW[currentSegW])
        currentSegW++;
      currentSegW--;
      const Real *paramsW = coeffsW[currentSegW];
      const Real xxW = sNormalized - breaksW[currentSegW];
      res[i] = L * (paramsW[0] + paramsW[1] * xxW + paramsW[2] * pow(xxW, 2) +
                    paramsW[3] * pow(xxW, 3));
    }
  }
}
static void danio_height(Real L, const Real *rS, Real *res, int Nm) {
  enum { nBreaksH = 15 };
  const Real breaksH[nBreaksH] = {0,   0.01,  0.05,  0.1,   0.3,
                                  0.5, 0.7,   0.8,   0.85,  0.87,
                                  0.9, 0.993, 0.996, 0.998, 1};
  const Real coeffsH[nBreaksH - 1][4] = {
      {0.0011746, 1.345, 2.2204e-14, -578.62},
      {0.014046, 1.1715, -17.359, 128.6},
      {0.041361, 0.40004, -1.9268, 9.7029},
      {0.057759, 0.28013, -0.47141, -0.08102},
      {0.094281, 0.081843, -0.52002, -0.76511},
      {0.083728, -0.21798, -0.97909, 3.9699},
      {0.032727, -0.13323, 1.4028, 2.5693},
      {0.036002, 0.22441, 2.1736, -13.194},
      {0.051007, 0.34282, 0.19446, 16.642},
      {0.058075, 0.37057, 1.193, -17.944},
      {0.069781, 0.3937, -0.42196, -29.388},
      {0.079107, -0.44731, -8.6211, -1.8283e+05},
      {0.072751, -5.4355, -1654.1, -2.9121e+05},
      {0.052934, -15.546, -3401.4, 5.6689e+05}};
  for (int i = 0; i < Nm; ++i) {
    if (rS[i] <= 0 || rS[i] >= L)
      res[i] = 0;
    else {
      const Real sNormalized = rS[i] / L;
      int currentSegH = 1;
      while (sNormalized >= breaksH[currentSegH])
        currentSegH++;
      currentSegH--;
      const Real *paramsH = coeffsH[currentSegH];
      const Real xxH = sNormalized - breaksH[currentSegH];
      res[i] = L * (paramsH[0] + paramsH[1] * xxH + paramsH[2] * pow(xxH, 2) +
                    paramsH[3] * pow(xxH, 3));
    }
  }
}
static void compute_widths_heights(const char *heightName,
                                   const char *widthName, Real L,
                                   const Real *rS, Real *height, Real *width,
                                   int nM) {
  if (!sim.rank) {
    printf("height = %s, width=%s\n", heightName, widthName);
    fflush(NULL);
  }
  if (strcmp(heightName, "largefin") == 0) {
    Real xh[8] = {0, 0, .2 * L, .4 * L, .6 * L, .8 * L, L, L};
    Real yh[8] = {0,        .055 * L, .18 * L,  .2 * L,
                  .064 * L, .002 * L, .325 * L, 0};
    integrate_bspline(xh, yh, 8, L, rS, height, nM);
  } else if (strcmp(heightName, "tunaclone") == 0) {
    Real xh[9] = {0, 0, 0.2 * L, .4 * L, .6 * L, .9 * L, .96 * L, L, L};
    Real yh[9] = {0, .05 * L, .14 * L, .15 * L, .11 * L,
                  0, .1 * L,  .2 * L,  0};
    integrate_bspline(xh, yh, 9, L, rS, height, nM);
  } else if (strcmp(heightName, "danio") == 0) {
    danio_height(L, rS, height, nM);
  } else if (strcmp(heightName, "stefan") == 0) {
    if (!sim.rank)
      printf("Building object's height according to Stefan profile\n");
    stefan_height(L, rS, height, nM);
  } else if (strcmp(heightName, "larval") == 0) {
    if (!sim.rank)
      printf("Building object's height according to Larval profile\n");
    larval_height(L, rS, height, nM);
  } else {
    Real xh[8] = {0, 0, .2 * L, .4 * L, .6 * L, .8 * L, L, L};
    Real yh[8] = {0,        .055 * L,  .068 * L, .076 * L,
                  .064 * L, .0072 * L, .11 * L,  0};
    integrate_bspline(xh, yh, 8, L, rS, height, nM);
  }
  if (strcmp(widthName, "fatter") == 0) {
    Real xw[6] = {0, 0, L / 3., 2 * L / 3., L, L};
    Real yw[6] = {0, 8.9e-2 * L, 7.0e-2 * L, 3.0e-2 * L, 2.0e-2 * L, 0};
    integrate_bspline(xw, yw, 6, L, rS, width, nM);
  } else if (strcmp(widthName, "danio") == 0) {
    danio_width(L, rS, width, nM);
  } else if (strcmp(widthName, "stefan") == 0) {
    stefan_width(L, rS, width, nM);
  } else if (strcmp(widthName, "larval") == 0) {
    larval_width(L, rS, width, nM);
  } else {
    Real xw[6] = {0, 0, L / 3., 2 * L / 3., L, L};
    Real yw[6] = {0, 8.9e-2 * L, 1.7e-2 * L, 1.6e-2 * L, 1.3e-2 * L, 0};
    integrate_bspline(xw, yw, 6, L, rS, width, nM);
  }
}

static void frenet_solve(struct Midline *m) {
  const int Nm = m->Nm;
  const Real *rS = m->rS, *curv = m->rK, *curv_dt = m->vK, *tors = m->rT,
             *tors_dt = m->vT;
  Real *rX = m->rX, *rY = m->rY, *rZ = m->rZ, *vX = m->vX, *vY = m->vY,
       *vZ = m->vZ;
  Real *norX = m->norX, *norY = m->norY, *norZ = m->norZ, *vNorX = m->vNorX,
       *vNorY = m->vNorY, *vNorZ = m->vNorZ;
  Real *binX = m->binX, *binY = m->binY, *binZ = m->binZ, *vBinX = m->vBinX,
       *vBinY = m->vBinY, *vBinZ = m->vBinZ;
  rX[0] = 0.0;
  rY[0] = 0.0;
  rZ[0] = 0.0;
  Real ksiX = 1.0;
  Real ksiY = 0.0;
  Real ksiZ = 0.0;
  norX[0] = 0.0;
  norY[0] = 1.0;
  norZ[0] = 0.0;
  binX[0] = 0.0;
  binY[0] = 0.0;
  binZ[0] = 1.0;
  vX[0] = 0.0;
  vY[0] = 0.0;
  vZ[0] = 0.0;
  Real vKsiX = 0.0;
  Real vKsiY = 0.0;
  Real vKsiZ = 0.0;
  vNorX[0] = 0.0;
  vNorY[0] = 0.0;
  vNorZ[0] = 0.0;
  vBinX[0] = 0.0;
  vBinY[0] = 0.0;
  vBinZ[0] = 0.0;
  for (int i = 1; i < Nm; i++) {
    const Real dksiX = curv[i - 1] * norX[i - 1];
    const Real dksiY = curv[i - 1] * norY[i - 1];
    const Real dksiZ = curv[i - 1] * norZ[i - 1];
    const Real dnuX = -curv[i - 1] * ksiX + tors[i - 1] * binX[i - 1];
    const Real dnuY = -curv[i - 1] * ksiY + tors[i - 1] * binY[i - 1];
    const Real dnuZ = -curv[i - 1] * ksiZ + tors[i - 1] * binZ[i - 1];
    const Real dbinX = -tors[i - 1] * norX[i - 1];
    const Real dbinY = -tors[i - 1] * norY[i - 1];
    const Real dbinZ = -tors[i - 1] * norZ[i - 1];
    const Real dvKsiX =
        curv_dt[i - 1] * norX[i - 1] + curv[i - 1] * vNorX[i - 1];
    const Real dvKsiY =
        curv_dt[i - 1] * norY[i - 1] + curv[i - 1] * vNorY[i - 1];
    const Real dvKsiZ =
        curv_dt[i - 1] * norZ[i - 1] + curv[i - 1] * vNorZ[i - 1];
    const Real dvNuX = -curv_dt[i - 1] * ksiX - curv[i - 1] * vKsiX +
                       tors_dt[i - 1] * binX[i - 1] + tors[i - 1] * vBinX[i - 1];
    const Real dvNuY = -curv_dt[i - 1] * ksiY - curv[i - 1] * vKsiY +
                       tors_dt[i - 1] * binY[i - 1] + tors[i - 1] * vBinY[i - 1];
    const Real dvNuZ = -curv_dt[i - 1] * ksiZ - curv[i - 1] * vKsiZ +
                       tors_dt[i - 1] * binZ[i - 1] + tors[i - 1] * vBinZ[i - 1];
    const Real dvBinX =
        -tors_dt[i - 1] * norX[i - 1] - tors[i - 1] * vNorX[i - 1];
    const Real dvBinY =
        -tors_dt[i - 1] * norY[i - 1] - tors[i - 1] * vNorY[i - 1];
    const Real dvBinZ =
        -tors_dt[i - 1] * norZ[i - 1] - tors[i - 1] * vNorZ[i - 1];
    const Real ds = rS[i] - rS[i - 1];
    rX[i] = rX[i - 1] + ds * ksiX;
    rY[i] = rY[i - 1] + ds * ksiY;
    rZ[i] = rZ[i - 1] + ds * ksiZ;
    norX[i] = norX[i - 1] + ds * dnuX;
    norY[i] = norY[i - 1] + ds * dnuY;
    norZ[i] = norZ[i - 1] + ds * dnuZ;
    ksiX += ds * dksiX;
    ksiY += ds * dksiY;
    ksiZ += ds * dksiZ;
    binX[i] = binX[i - 1] + ds * dbinX;
    binY[i] = binY[i - 1] + ds * dbinY;
    binZ[i] = binZ[i - 1] + ds * dbinZ;
    vX[i] = vX[i - 1] + ds * vKsiX;
    vY[i] = vY[i - 1] + ds * vKsiY;
    vZ[i] = vZ[i - 1] + ds * vKsiZ;
    vNorX[i] = vNorX[i - 1] + ds * dvNuX;
    vNorY[i] = vNorY[i - 1] + ds * dvNuY;
    vNorZ[i] = vNorZ[i - 1] + ds * dvNuZ;
    vKsiX += ds * dvKsiX;
    vKsiY += ds * dvKsiY;
    vKsiZ += ds * dvKsiZ;
    vBinX[i] = vBinX[i - 1] + ds * dvBinX;
    vBinY[i] = vBinY[i - 1] + ds * dvBinY;
    vBinZ[i] = vBinZ[i - 1] + ds * dvBinZ;
    const Real d1 = ksiX * ksiX + ksiY * ksiY + ksiZ * ksiZ;
    const Real d2 = norX[i] * norX[i] + norY[i] * norY[i] + norZ[i] * norZ[i];
    const Real d3 = binX[i] * binX[i] + binY[i] * binY[i] + binZ[i] * binZ[i];
    if (d1 > DBL_EPSILON) {
      const Real normfac = 1.0 / sqrt(d1);
      ksiX *= normfac;
      ksiY *= normfac;
      ksiZ *= normfac;
    }
    if (d2 > DBL_EPSILON) {
      const Real normfac = 1.0 / sqrt(d2);
      norX[i] *= normfac;
      norY[i] *= normfac;
      norZ[i] *= normfac;
    }
    if (d3 > DBL_EPSILON) {
      const Real normfac = 1.0 / sqrt(d3);
      binX[i] *= normfac;
      binY[i] *= normfac;
      binZ[i] *= normfac;
    }
  }
}

static void recompute_normal_vectors(struct Midline *m) {
  const int Nm = m->Nm;
  const Real *rS = m->rS;
  Real *rX = m->rX, *rY = m->rY, *rZ = m->rZ, *vX = m->vX, *vY = m->vY,
       *vZ = m->vZ;
  Real *norX = m->norX, *norY = m->norY, *norZ = m->norZ, *vNorX = m->vNorX,
       *vNorY = m->vNorY, *vNorZ = m->vNorZ;
  Real *binX = m->binX, *binY = m->binY, *binZ = m->binZ, *vBinX = m->vBinX,
       *vBinY = m->vBinY, *vBinZ = m->vBinZ;
#pragma omp parallel for
  for (int i = 1; i < Nm - 1; i++) {
    const Real hp = rS[i + 1] - rS[i];
    const Real hm = rS[i] - rS[i - 1];
    const Real frac = hp / hm;
    const Real am = -frac * frac;
    const Real a = frac * frac - 1.0;
    const Real ap = 1.0;
    const Real denom = 1.0 / (hp * (1.0 + frac));
    const Real tX = (am * rX[i - 1] + a * rX[i] + ap * rX[i + 1]) * denom;
    const Real tY = (am * rY[i - 1] + a * rY[i] + ap * rY[i + 1]) * denom;
    const Real tZ = (am * rZ[i - 1] + a * rZ[i] + ap * rZ[i + 1]) * denom;
    const Real dtX = (am * vX[i - 1] + a * vX[i] + ap * vX[i + 1]) * denom;
    const Real dtY = (am * vY[i - 1] + a * vY[i] + ap * vY[i + 1]) * denom;
    const Real dtZ = (am * vZ[i - 1] + a * vZ[i] + ap * vZ[i + 1]) * denom;
    const Real BDx = norX[i];
    const Real BDy = norY[i];
    const Real BDz = norZ[i];
    const Real dBDx = vNorX[i];
    const Real dBDy = vNorY[i];
    const Real dBDz = vNorZ[i];
    const Real dot = BDx * tX + BDy * tY + BDz * tZ;
    const Real ddot =
        dBDx * tX + dBDy * tY + dBDz * tZ + BDx * dtX + BDy * dtY + BDz * dtZ;
    norX[i] = BDx - dot * tX;
    norY[i] = BDy - dot * tY;
    norZ[i] = BDz - dot * tZ;
    const Real inormn =
        1.0 / sqrt(norX[i] * norX[i] + norY[i] * norY[i] + norZ[i] * norZ[i]);
    norX[i] *= inormn;
    norY[i] *= inormn;
    norZ[i] *= inormn;
    vNorX[i] = dBDx - ddot * tX - dot * dtX;
    vNorY[i] = dBDy - ddot * tY - dot * dtY;
    vNorZ[i] = dBDz - ddot * tZ - dot * dtZ;
    binX[i] = tY * norZ[i] - tZ * norY[i];
    binY[i] = tZ * norX[i] - tX * norZ[i];
    binZ[i] = tX * norY[i] - tY * norX[i];
    const Real inormb =
        1.0 / sqrt(binX[i] * binX[i] + binY[i] * binY[i] + binZ[i] * binZ[i]);
    binX[i] *= inormb;
    binY[i] *= inormb;
    binZ[i] *= inormb;
    vBinX[i] =
        (dtY * norZ[i] + tY * vNorZ[i]) - (dtZ * norY[i] + tZ * vNorY[i]);
    vBinY[i] =
        (dtZ * norX[i] + tZ * vNorX[i]) - (dtX * norZ[i] + tX * vNorZ[i]);
    vBinZ[i] =
        (dtX * norY[i] + tX * vNorY[i]) - (dtY * norX[i] + tY * vNorX[i]);
  }
  for (int i = 0; i <= Nm - 1; i += Nm - 1) {
    const int ipm = (i == Nm - 1) ? i - 1 : i + 1;
    const Real ids = 1.0 / (rS[ipm] - rS[i]);
    const Real tX = (rX[ipm] - rX[i]) * ids;
    const Real tY = (rY[ipm] - rY[i]) * ids;
    const Real tZ = (rZ[ipm] - rZ[i]) * ids;
    const Real dtX = (vX[ipm] - vX[i]) * ids;
    const Real dtY = (vY[ipm] - vY[i]) * ids;
    const Real dtZ = (vZ[ipm] - vZ[i]) * ids;
    const Real BDx = norX[i];
    const Real BDy = norY[i];
    const Real BDz = norZ[i];
    const Real dBDx = vNorX[i];
    const Real dBDy = vNorY[i];
    const Real dBDz = vNorZ[i];
    const Real dot = BDx * tX + BDy * tY + BDz * tZ;
    const Real ddot =
        dBDx * tX + dBDy * tY + dBDz * tZ + BDx * dtX + BDy * dtY + BDz * dtZ;
    norX[i] = BDx - dot * tX;
    norY[i] = BDy - dot * tY;
    norZ[i] = BDz - dot * tZ;
    const Real inormn =
        1.0 / sqrt(norX[i] * norX[i] + norY[i] * norY[i] + norZ[i] * norZ[i]);
    norX[i] *= inormn;
    norY[i] *= inormn;
    norZ[i] *= inormn;
    vNorX[i] = dBDx - ddot * tX - dot * dtX;
    vNorY[i] = dBDy - ddot * tY - dot * dtY;
    vNorZ[i] = dBDz - ddot * tZ - dot * dtZ;
    binX[i] = tY * norZ[i] - tZ * norY[i];
    binY[i] = tZ * norX[i] - tX * norZ[i];
    binZ[i] = tX * norY[i] - tY * norX[i];
    const Real inormb =
        1.0 / sqrt(binX[i] * binX[i] + binY[i] * binY[i] + binZ[i] * binZ[i]);
    binX[i] *= inormb;
    binY[i] *= inormb;
    binZ[i] *= inormb;
    vBinX[i] =
        (dtY * norZ[i] + tY * vNorZ[i]) - (dtZ * norY[i] + tZ * vNorY[i]);
    vBinY[i] =
        (dtZ * norX[i] + tZ * vNorX[i]) - (dtX * norZ[i] + tX * vNorZ[i]);
    vBinZ[i] =
        (dtX * norY[i] + tX * vNorY[i]) - (dtY * norX[i] + tY * vNorX[i]);
  }
}

static void perform_pitching_motion(struct Midline *m) {
  const int Nm = m->Nm;
  Real *rX = m->rX, *rY = m->rY, *rZ = m->rZ, *vX = m->vX, *vY = m->vY,
       *vZ = m->vZ;
  const Real gamma = m->gamma, dgamma = m->dgamma;
  Real R, Rdot;
  if (fabs(gamma) > 1e-10) {
    R = 1.0 / gamma;
    Rdot = -1.0 / gamma / gamma * dgamma;
  } else {
    R = gamma >= 0 ? 1e10 : -1e10;
    Rdot = 0.0;
  }
  const Real x0N = rX[Nm - 1];
  const Real y0N = rY[Nm - 1];
  const Real x0Ndot = vX[Nm - 1];
  const Real y0Ndot = vY[Nm - 1];
  const Real phi = atan2(y0N, x0N);
  const Real phidot = 1.0 / (1.0 + pow(y0N / x0N, 2)) *
                      (y0Ndot / x0N - y0N * x0Ndot / x0N / x0N);
  const Real M = pow(x0N * x0N + y0N * y0N, 0.5);
  const Real Mdot = (x0N * x0Ndot + y0N * y0Ndot) / M;
  const Real cosphi = cos(phi);
  const Real sinphi = sin(phi);
#pragma omp parallel for
  for (int i = 0; i < Nm; i++) {
    const double x0 = rX[i];
    const double y0 = rY[i];
    const double x0dot = vX[i];
    const double y0dot = vY[i];
    const double x1 = cosphi * x0 - sinphi * y0;
    const double y1 = sinphi * x0 + cosphi * y0;
    const double x1dot =
        cosphi * x0dot - sinphi * y0dot + (-sinphi * x0 - cosphi * y0) * phidot;
    const double y1dot =
        sinphi * x0dot + cosphi * y0dot + (cosphi * x0 - sinphi * y0) * phidot;
    const double theta = (M - x1) / R;
    const double costheta = cos(theta);
    const double sintheta = sin(theta);
    const double x2 = M - R * sintheta;
    const double y2 = y1;
    const double z2 = R - R * costheta;
    const double thetadot = (Mdot - x1dot) / R - (M - x1) / R / R * Rdot;
    const double x2dot = Mdot - Rdot * sintheta - R * costheta * thetadot;
    const double y2dot = y1dot;
    const double z2dot = Rdot - Rdot * costheta + R * sintheta * thetadot;
    rX[i] = x2;
    rY[i] = y2;
    rZ[i] = z2;
    vX[i] = x2dot;
    vY[i] = y2dot;
    vZ[i] = z2dot;
  }
  recompute_normal_vectors(m);
}

static void compute_midline(struct Midline *m, Real t) {
  const int Nm = m->Nm;
  const Real length = m->length, Tperiod = m->Tperiod;
  if (0 < t && t < 0.1 * Tperiod) {
    m->timeshift = (t - m->time0) / Tperiod + m->timeshift;
    m->time0 = t;
  }
  const Real curvaturePoints[6] = {0.0,          0.15 * length, 0.4 * length,
                                   0.65 * length, 0.9 * length,  length};
  const Real curvatureValues[6] = {0.82014 / length, 1.46515 / length,
                                   2.57136 / length, 3.75425 / length,
                                   5.09147 / length, 5.70449 / length};
  const Real curvatureZeros[6] = {0, 0, 0, 0, 0, 0};
  sched_transition(m, 0, 0, Tperiod, curvatureZeros, curvatureValues);
  sched_gimme(m, t, curvaturePoints, Nm, m->rS, m->rC, m->vC);
  const Real darg = 2 * M_PI / Tperiod;
  const Real arg0 = 2 * M_PI * ((t - m->time0) / Tperiod + m->timeshift) +
                    M_PI * m->phaseShift;
  const Real alpha = m->alpha, dalpha = m->dalpha, beta = m->beta,
             dbeta = m->dbeta, amplitudeFactor = m->amplitudeFactor,
             waveLength = m->waveLength;
  const Real *rS = m->rS, *rC = m->rC, *vC = m->vC;
  Real *rK = m->rK, *vK = m->vK, *rT = m->rT, *vT = m->vT;
#pragma omp parallel for
  for (int i = 0; i < Nm; ++i) {
    const Real arg = arg0 - 2 * M_PI * rS[i] / length / waveLength;
    const Real curv = sin(arg) + beta;
    const Real dcurv = cos(arg) * darg + dbeta;
    rK[i] = alpha * amplitudeFactor * rC[i] * curv;
    vK[i] = alpha * amplitudeFactor * (vC[i] * curv + rC[i] * dcurv) +
            dalpha * amplitudeFactor * rC[i] * curv;
    rT[i] = 0;
    vT[i] = 0;
  }
  frenet_solve(m);
  perform_pitching_motion(m);
}

static void integrate_linear_momentum(struct Midline *m) {
  const int Nm = m->Nm;
  const Real *rS = m->rS, *width = m->width, *height = m->height;
  Real *rX = m->rX, *rY = m->rY, *rZ = m->rZ, *vX = m->vX, *vY = m->vY,
       *vZ = m->vZ;
  const Real *norX = m->norX, *norY = m->norY, *norZ = m->norZ,
             *vNorX = m->vNorX, *vNorY = m->vNorY, *vNorZ = m->vNorZ;
  const Real *binX = m->binX, *binY = m->binY, *binZ = m->binZ,
             *vBinX = m->vBinX, *vBinY = m->vBinY, *vBinZ = m->vBinZ;
  Real V = 0, cmx = 0, cmy = 0, cmz = 0, lmx = 0, lmy = 0, lmz = 0;
#pragma omp parallel for schedule(static) reduction(+ : V, cmx, cmy, cmz, lmx, lmy, lmz)
  for (int i = 0; i < Nm; ++i) {
    const Real ds = 0.5 * ((i == 0) ? rS[1] - rS[0]
                                    : ((i == Nm - 1) ? rS[Nm - 1] - rS[Nm - 2]
                                                     : rS[i + 1] - rS[i - 1]));
    const Real c0 = norY[i] * binZ[i] - norZ[i] * binY[i];
    const Real c1 = norZ[i] * binX[i] - norX[i] * binZ[i];
    const Real c2 = norX[i] * binY[i] - norY[i] * binX[i];
    const Real x0dot = d_ds(m, i, rX, Nm);
    const Real x1dot = d_ds(m, i, rY, Nm);
    const Real x2dot = d_ds(m, i, rZ, Nm);
    const Real n0dot = d_ds(m, i, norX, Nm);
    const Real n1dot = d_ds(m, i, norY, Nm);
    const Real n2dot = d_ds(m, i, norZ, Nm);
    const Real b0dot = d_ds(m, i, binX, Nm);
    const Real b1dot = d_ds(m, i, binY, Nm);
    const Real b2dot = d_ds(m, i, binZ, Nm);
    const Real w = width[i];
    const Real H = height[i];
    const Real aux1 = w * H * (c0 * x0dot + c1 * x1dot + c2 * x2dot) * ds;
    const Real aux2 =
        0.25 * w * w * w * H * (c0 * n0dot + c1 * n1dot + c2 * n2dot) * ds;
    const Real aux3 =
        0.25 * w * H * H * H * (c0 * b0dot + c1 * b1dot + c2 * b2dot) * ds;
    V += aux1;
    cmx += rX[i] * aux1 + norX[i] * aux2 + binX[i] * aux3;
    cmy += rY[i] * aux1 + norY[i] * aux2 + binY[i] * aux3;
    cmz += rZ[i] * aux1 + norZ[i] * aux2 + binZ[i] * aux3;
    lmx += vX[i] * aux1 + vNorX[i] * aux2 + vBinX[i] * aux3;
    lmy += vY[i] * aux1 + vNorY[i] * aux2 + vBinY[i] * aux3;
    lmz += vZ[i] * aux1 + vNorZ[i] * aux2 + vBinZ[i] * aux3;
  }
  const Real volume = V * M_PI;
  const Real aux = M_PI / volume;
  cmx *= aux;
  cmy *= aux;
  cmz *= aux;
  lmx *= aux;
  lmy *= aux;
  lmz *= aux;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < Nm; ++i) {
    rX[i] -= cmx;
    rY[i] -= cmy;
    rZ[i] -= cmz;
    vX[i] -= lmx;
    vY[i] -= lmy;
    vZ[i] -= lmz;
  }
}

static void integrate_angular_momentum(struct Midline *m, const Real dt) {
  const int Nm = m->Nm;
  const Real *rS = m->rS, *width = m->width, *height = m->height;
  Real *rX = m->rX, *rY = m->rY, *rZ = m->rZ, *vX = m->vX, *vY = m->vY,
       *vZ = m->vZ;
  Real *norX = m->norX, *norY = m->norY, *norZ = m->norZ, *vNorX = m->vNorX,
       *vNorY = m->vNorY, *vNorZ = m->vNorZ;
  Real *binX = m->binX, *binY = m->binY, *binZ = m->binZ, *vBinX = m->vBinX,
       *vBinY = m->vBinY, *vBinZ = m->vBinZ;
  Real *quaternion_internal = m->quaternion_internal;
  Real *angvel_internal = m->angvel_internal;
  Real JXX = 0;
  Real JYY = 0;
  Real JZZ = 0;
  Real JXY = 0;
  Real JYZ = 0;
  Real JZX = 0;
  Real AM_X = 0;
  Real AM_Y = 0;
  Real AM_Z = 0;
#pragma omp parallel for reduction(+ : JXX, JYY, JZZ, JXY, JYZ, JZX, AM_X, AM_Y, AM_Z)
  for (int i = 0; i < Nm; ++i) {
    const Real ds = 0.5 * ((i == 0) ? rS[1] - rS[0]
                                    : ((i == Nm - 1) ? rS[Nm - 1] - rS[Nm - 2]
                                                     : rS[i + 1] - rS[i - 1]));
    const Real c0 = norY[i] * binZ[i] - norZ[i] * binY[i];
    const Real c1 = norZ[i] * binX[i] - norX[i] * binZ[i];
    const Real c2 = norX[i] * binY[i] - norY[i] * binX[i];
    const Real x0dot = d_ds(m, i, rX, Nm);
    const Real x1dot = d_ds(m, i, rY, Nm);
    const Real x2dot = d_ds(m, i, rZ, Nm);
    const Real n0dot = d_ds(m, i, norX, Nm);
    const Real n1dot = d_ds(m, i, norY, Nm);
    const Real n2dot = d_ds(m, i, norZ, Nm);
    const Real b0dot = d_ds(m, i, binX, Nm);
    const Real b1dot = d_ds(m, i, binY, Nm);
    const Real b2dot = d_ds(m, i, binZ, Nm);
    const Real M00 = width[i] * height[i];
    const Real M11 = 0.25 * width[i] * width[i] * width[i] * height[i];
    const Real M22 = 0.25 * width[i] * height[i] * height[i] * height[i];
    const Real cR = c0 * x0dot + c1 * x1dot + c2 * x2dot;
    const Real cN = c0 * n0dot + c1 * n1dot + c2 * n2dot;
    const Real cB = c0 * b0dot + c1 * b1dot + c2 * b2dot;
    JXY += -ds * (cR * (rX[i] * rY[i] * M00 + norX[i] * norY[i] * M11 +
                        binX[i] * binY[i] * M22) +
                  cN * M11 * (rX[i] * norY[i] + rY[i] * norX[i]) +
                  cB * M22 * (rX[i] * binY[i] + rY[i] * binX[i]));
    JZX += -ds * (cR * (rZ[i] * rX[i] * M00 + norZ[i] * norX[i] * M11 +
                        binZ[i] * binX[i] * M22) +
                  cN * M11 * (rZ[i] * norX[i] + rX[i] * norZ[i]) +
                  cB * M22 * (rZ[i] * binX[i] + rX[i] * binZ[i]));
    JYZ += -ds * (cR * (rY[i] * rZ[i] * M00 + norY[i] * norZ[i] * M11 +
                        binY[i] * binZ[i] * M22) +
                  cN * M11 * (rY[i] * norZ[i] + rZ[i] * norY[i]) +
                  cB * M22 * (rY[i] * binZ[i] + rZ[i] * binY[i]));
    const Real XX = ds * (cR * (rX[i] * rX[i] * M00 + norX[i] * norX[i] * M11 +
                                binX[i] * binX[i] * M22) +
                          cN * M11 * (rX[i] * norX[i] + rX[i] * norX[i]) +
                          cB * M22 * (rX[i] * binX[i] + rX[i] * binX[i]));
    const Real YY = ds * (cR * (rY[i] * rY[i] * M00 + norY[i] * norY[i] * M11 +
                                binY[i] * binY[i] * M22) +
                          cN * M11 * (rY[i] * norY[i] + rY[i] * norY[i]) +
                          cB * M22 * (rY[i] * binY[i] + rY[i] * binY[i]));
    const Real ZZ = ds * (cR * (rZ[i] * rZ[i] * M00 + norZ[i] * norZ[i] * M11 +
                                binZ[i] * binZ[i] * M22) +
                          cN * M11 * (rZ[i] * norZ[i] + rZ[i] * norZ[i]) +
                          cB * M22 * (rZ[i] * binZ[i] + rZ[i] * binZ[i]));
    JXX += YY + ZZ;
    JYY += ZZ + XX;
    JZZ += YY + XX;
    const Real xd_y = cR * (vX[i] * rY[i] * M00 + vNorX[i] * norY[i] * M11 +
                            vBinX[i] * binY[i] * M22) +
                      cN * M11 * (vX[i] * norY[i] + rY[i] * vNorX[i]) +
                      cB * M22 * (vX[i] * binY[i] + rY[i] * vBinX[i]);
    const Real x_yd = cR * (rX[i] * vY[i] * M00 + norX[i] * vNorY[i] * M11 +
                            binX[i] * vBinY[i] * M22) +
                      cN * M11 * (rX[i] * vNorY[i] + vY[i] * norX[i]) +
                      cB * M22 * (rX[i] * vBinY[i] + vY[i] * binX[i]);
    const Real xd_z = cR * (rZ[i] * vX[i] * M00 + norZ[i] * vNorX[i] * M11 +
                            binZ[i] * vBinX[i] * M22) +
                      cN * M11 * (rZ[i] * vNorX[i] + vX[i] * norZ[i]) +
                      cB * M22 * (rZ[i] * vBinX[i] + vX[i] * binZ[i]);
    const Real x_zd = cR * (vZ[i] * rX[i] * M00 + vNorZ[i] * norX[i] * M11 +
                            vBinZ[i] * binX[i] * M22) +
                      cN * M11 * (vZ[i] * norX[i] + rX[i] * vNorZ[i]) +
                      cB * M22 * (vZ[i] * binX[i] + rX[i] * vBinZ[i]);
    const Real yd_z = cR * (vY[i] * rZ[i] * M00 + vNorY[i] * norZ[i] * M11 +
                            vBinY[i] * binZ[i] * M22) +
                      cN * M11 * (vY[i] * norZ[i] + rZ[i] * vNorY[i]) +
                      cB * M22 * (vY[i] * binZ[i] + rZ[i] * vBinY[i]);
    const Real y_zd = cR * (rY[i] * vZ[i] * M00 + norY[i] * vNorZ[i] * M11 +
                            binY[i] * vBinZ[i] * M22) +
                      cN * M11 * (rY[i] * vNorZ[i] + vZ[i] * norY[i]) +
                      cB * M22 * (rY[i] * vBinZ[i] + vZ[i] * binY[i]);
    AM_X += (y_zd - yd_z) * ds;
    AM_Y += (xd_z - x_zd) * ds;
    AM_Z += (x_yd - xd_y) * ds;
  }
  const Real eps = DBL_EPSILON;
  if (JXX < eps)
    JXX += eps;
  if (JYY < eps)
    JYY += eps;
  if (JZZ < eps)
    JZZ += eps;
  JXX *= M_PI;
  JYY *= M_PI;
  JZZ *= M_PI;
  JXY *= M_PI;
  JYZ *= M_PI;
  JZX *= M_PI;
  AM_X *= M_PI;
  AM_Y *= M_PI;
  AM_Z *= M_PI;
  const Real m00 = JXX;
  const Real m01 = JXY;
  const Real m02 = JZX;
  const Real m11 = JYY;
  const Real m12 = JYZ;
  const Real m22 = JZZ;
  const Real a00 = m22 * m11 - m12 * m12;
  const Real a01 = m02 * m12 - m22 * m01;
  const Real a02 = m01 * m12 - m02 * m11;
  const Real a11 = m22 * m00 - m02 * m02;
  const Real a12 = m01 * m02 - m00 * m12;
  const Real a22 = m00 * m11 - m01 * m01;
  const Real determinant = 1.0 / ((m00 * a00) + (m01 * a01) + (m02 * a02));
  angvel_internal[0] = (a00 * AM_X + a01 * AM_Y + a02 * AM_Z) * determinant;
  angvel_internal[1] = (a01 * AM_X + a11 * AM_Y + a12 * AM_Z) * determinant;
  angvel_internal[2] = (a02 * AM_X + a12 * AM_Y + a22 * AM_Z) * determinant;
  const Real dqdt[4] = {0.5 * (-angvel_internal[0] * quaternion_internal[1] -
                               angvel_internal[1] * quaternion_internal[2] -
                               angvel_internal[2] * quaternion_internal[3]),
                        0.5 * (+angvel_internal[0] * quaternion_internal[0] +
                               angvel_internal[1] * quaternion_internal[3] -
                               angvel_internal[2] * quaternion_internal[2]),
                        0.5 * (-angvel_internal[0] * quaternion_internal[3] +
                               angvel_internal[1] * quaternion_internal[0] +
                               angvel_internal[2] * quaternion_internal[1]),
                        0.5 * (+angvel_internal[0] * quaternion_internal[2] -
                               angvel_internal[1] * quaternion_internal[1] +
                               angvel_internal[2] * quaternion_internal[0])};
  quaternion_internal[0] -= dt * dqdt[0];
  quaternion_internal[1] -= dt * dqdt[1];
  quaternion_internal[2] -= dt * dqdt[2];
  quaternion_internal[3] -= dt * dqdt[3];
  const Real invD =
      1.0 / sqrt(quaternion_internal[0] * quaternion_internal[0] +
                 quaternion_internal[1] * quaternion_internal[1] +
                 quaternion_internal[2] * quaternion_internal[2] +
                 quaternion_internal[3] * quaternion_internal[3]);
  quaternion_internal[0] *= invD;
  quaternion_internal[1] *= invD;
  quaternion_internal[2] *= invD;
  quaternion_internal[3] *= invD;
  Real R[3][3];
  R[0][0] = 1 - 2 * (quaternion_internal[2] * quaternion_internal[2] +
                     quaternion_internal[3] * quaternion_internal[3]);
  R[0][1] = 2 * (quaternion_internal[1] * quaternion_internal[2] -
                 quaternion_internal[3] * quaternion_internal[0]);
  R[0][2] = 2 * (quaternion_internal[1] * quaternion_internal[3] +
                 quaternion_internal[2] * quaternion_internal[0]);
  R[1][0] = 2 * (quaternion_internal[1] * quaternion_internal[2] +
                 quaternion_internal[3] * quaternion_internal[0]);
  R[1][1] = 1 - 2 * (quaternion_internal[1] * quaternion_internal[1] +
                     quaternion_internal[3] * quaternion_internal[3]);
  R[1][2] = 2 * (quaternion_internal[2] * quaternion_internal[3] -
                 quaternion_internal[1] * quaternion_internal[0]);
  R[2][0] = 2 * (quaternion_internal[1] * quaternion_internal[3] -
                 quaternion_internal[2] * quaternion_internal[0]);
  R[2][1] = 2 * (quaternion_internal[2] * quaternion_internal[3] +
                 quaternion_internal[1] * quaternion_internal[0]);
  R[2][2] = 1 - 2 * (quaternion_internal[1] * quaternion_internal[1] +
                     quaternion_internal[2] * quaternion_internal[2]);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < Nm; ++i) {
    {
      Real p[3] = {rX[i], rY[i], rZ[i]};
      rX[i] = R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2];
      rY[i] = R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2];
      rZ[i] = R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2];
      Real v[3] = {vX[i], vY[i], vZ[i]};
      vX[i] = R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2];
      vY[i] = R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2];
      vZ[i] = R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2];
      vX[i] += angvel_internal[2] * rY[i] - angvel_internal[1] * rZ[i];
      vY[i] += angvel_internal[0] * rZ[i] - angvel_internal[2] * rX[i];
      vZ[i] += angvel_internal[1] * rX[i] - angvel_internal[0] * rY[i];
    }
    {
      Real p[3] = {norX[i], norY[i], norZ[i]};
      norX[i] = R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2];
      norY[i] = R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2];
      norZ[i] = R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2];
      Real v[3] = {vNorX[i], vNorY[i], vNorZ[i]};
      vNorX[i] = R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2];
      vNorY[i] = R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2];
      vNorZ[i] = R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2];
      vNorX[i] += angvel_internal[2] * norY[i] - angvel_internal[1] * norZ[i];
      vNorY[i] += angvel_internal[0] * norZ[i] - angvel_internal[2] * norX[i];
      vNorZ[i] += angvel_internal[1] * norX[i] - angvel_internal[0] * norY[i];
    }
    {
      Real p[3] = {binX[i], binY[i], binZ[i]};
      binX[i] = R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2];
      binY[i] = R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2];
      binZ[i] = R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2];
      Real v[3] = {vBinX[i], vBinY[i], vBinZ[i]};
      vBinX[i] = R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2];
      vBinY[i] = R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2];
      vBinZ[i] = R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2];
      vBinX[i] += angvel_internal[2] * binY[i] - angvel_internal[1] * binZ[i];
      vBinY[i] += angvel_internal[0] * binZ[i] - angvel_internal[2] * binX[i];
      vBinZ[i] += angvel_internal[1] * binX[i] - angvel_internal[0] * binY[i];
    }
  }
}

static void fish_init(struct Fish *f, const struct Params *p) {
  f->length = param_real(p, "L", 0);
  f->position[0] = param_real(p, "xpos", 0);
  f->position[1] = param_real(p, "ypos", sim.extents[1] / 2);
  f->position[2] = param_real(p, "zpos", sim.extents[2] / 2);
  f->quaternion[0] = param_real(p, "quat0", 0.0);
  f->quaternion[1] = param_real(p, "quat1", 0.0);
  f->quaternion[2] = param_real(p, "quat2", 0.0);
  f->quaternion[3] = param_real(p, "quat3", 0.0);
  Real planarAngle = param_real(p, "planarAngle", 0.0) / 180 * M_PI;
  Real *q = f->quaternion;
  const Real q_length = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  q[0] /= q_length;
  q[1] /= q_length;
  q[2] /= q_length;
  q[3] /= q_length;
  if (fabs(q_length - 1.0) > 100 * DBL_EPSILON) {
    q[0] = cos(0.5 * planarAngle);
    q[1] = 0;
    q[2] = 0;
    q[3] = sin(0.5 * planarAngle);
  } else {
    if (fabs(planarAngle) > 0 && sim.rank == 0)
      fprintf(stderr, "WARNING: Obstacle arguments include both quaternions and "
                      "planarAngle.Quaterion arguments have priority and "
                      "therefore planarAngle will be ignored.\n");
    planarAngle = 2 * atan2(q[3], q[0]);
  }
  int bFSM_alldir = param_bool(p, "bForcedInSimFrame", 0);
  f->bForcedInSimFrame[0] = bFSM_alldir || param_bool(p, "bForcedInSimFrame_x", 0);
  f->bForcedInSimFrame[1] = bFSM_alldir || param_bool(p, "bForcedInSimFrame_y", 0);
  f->bForcedInSimFrame[2] = bFSM_alldir || param_bool(p, "bForcedInSimFrame_z", 0);
  Real enforcedVelocity[3];
  enforcedVelocity[0] = -param_real(p, "xvel", 0.0);
  enforcedVelocity[1] = -param_real(p, "yvel", 0.0);
  enforcedVelocity[2] = -param_real(p, "zvel", 0.0);
  const int bFixToPlanar = param_bool(p, "bFixToPlanar", 0);
  int bFOR_alldir = param_bool(p, "bFixFrameOfRef", 0);
  f->bFixFrameOfRef[0] = bFOR_alldir || param_bool(p, "bFixFrameOfRef_x", 0);
  f->bFixFrameOfRef[1] = bFOR_alldir || param_bool(p, "bFixFrameOfRef_y", 0);
  f->bFixFrameOfRef[2] = bFOR_alldir || param_bool(p, "bFixFrameOfRef_z", 0);
  for (int d = 0; d < 3; d++) {
    f->absPos[d] = f->position[d];
    f->transVel[d] = f->angVel[d] = f->transVel_imposed[d] = 0;
    f->bBlockRotation[d] = 0;
  }
  if (f->length < 5 * DBL_EPSILON) {
    fprintf(stderr, "Parsed length is equal to zero. It really ought not to be.\n");
    abort();
  }
  for (int d = 0; d < 3; ++d) {
    if (f->bForcedInSimFrame[d]) {
      f->transVel_imposed[d] = f->transVel[d] = enforcedVelocity[d];
      if (!sim.rank)
        printf("Obstacle forced to move relative to sim domain with constant "
               "%c-vel: %f\n",
               "xyz"[d], f->transVel[d]);
    }
  }
  const int anyVelForced = f->bForcedInSimFrame[0] || f->bForcedInSimFrame[1] ||
                           f->bForcedInSimFrame[2];
  if (anyVelForced) {
    if (!sim.rank)
      printf("Obstacle has no angular velocity.\n");
    f->bBlockRotation[0] = f->bBlockRotation[1] = f->bBlockRotation[2] = 1;
  }
  if (bFixToPlanar) {
    if (!sim.rank)
      printf("Obstacle motion restricted to constant Z-plane.\n");
    f->bForcedInSimFrame[2] = 1;
    f->transVel_imposed[2] = 0;
    f->bBlockRotation[1] = 1;
    f->bBlockRotation[0] = 1;
  }
  const Real Tperiod = param_real(p, "T", 1.0);
  const Real phaseShift = param_real(p, "phi", 0.0);
  const Real ampFac = param_real(p, "amplitudeFactor", 1.0);
  f->bCorrectPosition = param_bool(p, "CorrectPosition", 0);
  f->bCorrectPositionZ = param_bool(p, "CorrectPositionZ", 0);
  f->bCorrectRoll = param_bool(p, "CorrectRoll", 0);
  const char *heightName = param_str(p, "heightProfile", "baseline");
  const char *widthName = param_str(p, "widthProfile", "baseline");
  if ((f->bCorrectPosition || f->bCorrectPositionZ || f->bCorrectRoll) &&
      fabs(f->quaternion[0] - 1) > 1e-6) {
    printf("PID controller only works for zero initial angles.\n");
    MPI_Abort(sim.comm, 1);
  }
  midline_init(&f->m, f->length, Tperiod, phaseShift, sim.hmin, ampFac);
  compute_widths_heights(heightName, widthName, f->length, f->m.rS,
                         f->m.height, f->m.width, f->m.Nm);
  f->origC[0] = f->position[0];
  f->origC[1] = f->position[1];
  f->origC[2] = f->position[2];
  if (sim.rank == 0)
    printf("nMidline=%d, length=%f, Tperiod=%f, phaseShift=%f\n", f->m.Nm,
           f->length, Tperiod, phaseShift);
  f->wyp = param_real(p, "wyp", 1.0);
  f->wzp = param_real(p, "wzp", 1.0);
}

static void add_obstacles(const struct Params *args) {
  const char *content = param_str(args, "factory-content", "");
  if (content[0] == '\0')
    content = param_str(args, "shapes", "");
  const char *fname = param_str(args, "factory", "factory");
  size_t len = strlen(content);
  char *text = (char *)malloc(len + 1);
  memcpy(text, content, len + 1);
  FILE *f = fname[0] ? fopen(fname, "r") : NULL;
  if (f) {
    fseek(f, 0, SEEK_END);
    long flen = ftell(f);
    fseek(f, 0, SEEK_SET);
    text = (char *)realloc(text, len + flen + 2);
    text[len] = '\n';
    flen = fread(text + len + 1, 1, flen, f);
    text[len + 1 + flen] = '\0';
    fclose(f);
  }
  sim.nfish = 0;
  sim.fish = NULL;
  char *save;
  for (char *line = strtok_r(text, "\n", &save); line;
       line = strtok_r(NULL, "\n", &save)) {
    while (isspace(*line))
      line++;
    if (*line == '\0' || *line == '#')
      continue;
    char *rest = line;
    while (*rest && !isspace(*rest))
      rest++;
    size_t idlen = rest - line;
    struct Params p;
    params_from_line(&p, rest);
    if ((idlen == 10 && strncmp(line, "StefanFish", 10) == 0) ||
        (idlen == 10 && strncmp(line, "stefanfish", 10) == 0)) {
      sim.fish = (struct Fish *)realloc(sim.fish, (sim.nfish + 1) * sizeof *sim.fish);
      struct Fish *fish = &sim.fish[sim.nfish];
      memset(fish, 0, sizeof *fish);
      fish->id = sim.nfish;
      fish_init(fish, &p);
      sim.nfish++;
    } else if (sim.rank == 0) {
      fprintf(stderr, "[CUP3D] Case %.*s is not defined: aborting\n", (int)idlen,
              line);
      abort();
    }
    params_free(&p);
  }
  if (sim.nfish == 0 && sim.rank == 0)
    fprintf(stderr, "[CUP3D] OBSTACLE FACTORY did not create any obstacles.\n");
  free(text);
}


static struct Sfc {
  int BX, BY, BZ, levelMax, isRegular, base_level;
  long long *Zsave;
  int *i_inv, *j_inv, *k_inv;
} sfc;
static long long axes_to_transpose(const int *X_in, int b) {
  if (b == 0)
    return 0;
  const int n = 3;
  int X[3] = {X_in[0], X_in[1], X_in[2]};
  int M = 1 << (b - 1), P, Q, t;
  int i;
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
  long long retval = 0;
  long long a = 0;
  const long long one = 1;
  const long long two = 2;
  for (long long level = 0; level < b; level++) {
    const long long a0 = ((one) << (a)) * ((long long)X[2] >> level & one);
    const long long a1 = ((one) << (a + one)) * ((long long)X[1] >> level & one);
    const long long a2 = ((one) << (a + two)) * ((long long)X[0] >> level & one);
    retval += a0 + a1 + a2;
    a += 3;
  }
  return retval;
}
static void transpose_to_axes(long long index, long long *X, int b) {
  const int n = 3;
  X[0] = 0;
  X[1] = 0;
  X[2] = 0;
  if (b == 0 && index == 0)
    return;
  long long aa = 0;
  const long long one = 1;
  const long long two = 2;
  for (long long i = 0; index > 0; i++) {
    long long x2 = index % two;
    index = index / two;
    long long x1 = index % two;
    index = index / two;
    long long x0 = index % two;
    index = index / two;
    X[0] += x0 * (one << aa);
    X[1] += x1 * (one << aa);
    X[2] += x2 * (one << aa);
    aa += 1;
  }
  int N = 2 << (b - 1), P, Q, t;
  int i;
  t = X[n - 1] >> 1;
  for (i = n - 1; i >= 1; i--)
    X[i] ^= X[i - 1];
  X[0] ^= t;
  for (Q = 2; Q != N; Q <<= 1) {
    P = Q - 1;
    for (i = n - 1; i >= 0; i--)
      if (X[i] & Q)
        X[0] ^= P;
      else {
        t = (X[0] ^ X[i]) & P;
        X[0] ^= t;
        X[i] ^= t;
      }
  }
}
static void sfc_init(int BX, int BY, int BZ, int lmax) {
  sfc.BX = BX;
  sfc.BY = BY;
  sfc.BZ = BZ;
  sfc.levelMax = lmax;
  int n_max = BX > BY ? BX : BY;
  if (BZ > n_max)
    n_max = BZ;
  sfc.base_level = (log(n_max) / log(2));
  if (sfc.base_level < (double)(log(n_max) / log(2)))
    sfc.base_level++;
  const int n0 = BX * BY * BZ;
  sfc.Zsave = (long long *)malloc(n0 * sizeof *sfc.Zsave);
  sfc.i_inv = (int *)malloc(n0 * sizeof *sfc.i_inv);
  sfc.j_inv = (int *)malloc(n0 * sizeof *sfc.j_inv);
  sfc.k_inv = (int *)malloc(n0 * sizeof *sfc.k_inv);
  for (int i = 0; i < n0; i++)
    sfc.Zsave[i] = sfc.i_inv[i] = sfc.j_inv[i] = sfc.k_inv[i] = -1;
  sfc.isRegular = 1;
  for (int k = 0; k < BZ; k++)
    for (int j = 0; j < BY; j++)
      for (int i = 0; i < BX; i++) {
        const int c[3] = {i, j, k};
        long long index = axes_to_transpose(c, sfc.base_level);
        long long substract = 0;
        for (long long h = 0; h < index; h++) {
          long long X[3] = {0, 0, 0};
          transpose_to_axes(h, X, sfc.base_level);
          if (X[0] >= BX || X[1] >= BY || X[2] >= BZ)
            substract++;
        }
        index -= substract;
        if (substract > 0)
          sfc.isRegular = 0;
        sfc.i_inv[index] = i;
        sfc.j_inv[index] = j;
        sfc.k_inv[index] = k;
        sfc.Zsave[k * BX * BY + j * BX + i] = index;
      }
}
static long long sfc_forward(int l, int i, int j, int k) {
  const int aux = 1 << l;
  if (l >= sfc.levelMax)
    return 0;
  long long retval;
  if (!sfc.isRegular) {
    const int I = i / aux;
    const int J = j / aux;
    const int K = k / aux;
    const int c2_a[3] = {i - I * aux, j - J * aux, k - K * aux};
    retval = axes_to_transpose(c2_a, l);
    retval += sfc.Zsave[(J + K * sfc.BY) * sfc.BX + I] * aux * aux * aux;
  } else {
    const int c2_a[3] = {i, j, k};
    retval = axes_to_transpose(c2_a, l + sfc.base_level);
  }
  return retval;
}
static void sfc_inverse(long long Z, int l, int *i, int *j, int *k) {
  if (sfc.isRegular) {
    long long X[3] = {0, 0, 0};
    transpose_to_axes(Z, X, l + sfc.base_level);
    *i = X[0];
    *j = X[1];
    *k = X[2];
  } else {
    long long aux = 1 << l;
    long long Zloc = Z % (aux * aux * aux);
    long long X[3] = {0, 0, 0};
    transpose_to_axes(Zloc, X, l);
    long long index = Z / (aux * aux * aux);
    *i = X[0] + sfc.i_inv[index] * aux;
    *j = X[1] + sfc.j_inv[index] * aux;
    *k = X[2] + sfc.k_inv[index] * aux;
  }
}
static void blk_fill(struct Blk *b, int level, long long Z) {
  int i, j, k;
  sfc_inverse(Z, level, &i, &j, &k);
  int nmax = sim.bpdx * BS;
  if (sim.bpdy * BS > nmax)
    nmax = sim.bpdy * BS;
  if (sim.bpdz * BS > nmax)
    nmax = sim.bpdz * BS;
  const double h0 = sim.maxextent / nmax;
  b->level = level;
  b->Z = Z;
  b->ix = i;
  b->iy = j;
  b->iz = k;
  b->h = h0 / (1 << level);
  b->origin[0] = i * BS * b->h;
  b->origin[1] = j * BS * b->h;
  b->origin[2] = k * BS * b->h;
}
static void grid_init_uniform(void) {
  const int level = sim.levelStart;
  const long long aux = 1 << level;
  sim.nblk = (long long)sim.bpdx * sim.bpdy * sim.bpdz * aux * aux * aux;
  sim.blk = (struct Blk *)malloc(sim.nblk * sizeof *sim.blk);
  sim.fld = (Real *)calloc(sim.nblk * BLK_S, sizeof(Real));
  for (long long Z = 0; Z < sim.nblk; Z++)
    blk_fill(&sim.blk[Z], level, Z);
}
static void blk_pos(const struct Blk *b, int ix, int iy, int iz, Real p[3]) {
  p[0] = b->origin[0] + b->h * (ix + 0.5);
  p[1] = b->origin[1] + b->h * (iy + 0.5);
  p[2] = b->origin[2] + b->h * (iz + 0.5);
}
static void dump(Real time, char *path) {
  long i, j, k, l, x, y, z, ncell, ncell_total, offset;
  char xyz_path[FILENAME_MAX], attr_path[FILENAME_MAX], xdmf_path[FILENAME_MAX],
      *xyz_base, *attr_base;
  MPI_File mpi_file;
  FILE *xmf;
  float *xyz, *attr;
  snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", path);
  snprintf(attr_path, sizeof attr_path, "%s.attr.raw", path);
  snprintf(xdmf_path, sizeof xdmf_path, "%s.xdmf2", path);
  xyz_base = xyz_path;
  attr_base = attr_path;
  for (j = 0; xyz_path[j] != '\0'; j++) {
    if (xyz_path[j] == '/' && xyz_path[j + 1] != '\0') {
      xyz_base = &xyz_path[j + 1];
      attr_base = &attr_path[j + 1];
    }
  }
  ncell = sim.nblk * BS3;
  MPI_Exscan(&ncell, &offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (sim.rank == 0)
    offset = 0;
  if (sim.rank == sim.size - 1) {
    ncell_total = ncell + offset;
    xmf = fopen(xdmf_path, "w");
    fprintf(xmf,
            "<Xdmf\n"
            "    Version=\"2.0\">\n"
            "  <Domain>\n"
            "    <Grid>\n"
            "      <Time Value=\"%.16e\"/>\n"
            "      <Topology\n"
            "          Dimensions=\"%ld\"\n"
            "          TopologyType=\"Hexahedron\"/>\n"
            "     <Geometry>\n"
            "       <DataItem\n"
            "           Dimensions=\"%ld 3\"\n"
            "           Format=\"Binary\">\n"
            "         %s\n"
            "       </DataItem>\n"
            "     </Geometry>\n"
            "       <Attribute\n"
            "           Name=\"chi\"\n"
            "           Center=\"Cell\">\n"
            "         <DataItem\n"
            "             Dimensions=\"%ld\"\n"
            "             Format=\"Binary\">\n"
            "           %s\n"
            "         </DataItem>\n"
            "       </Attribute>\n"
            "    </Grid>\n"
            "  </Domain>\n"
            "</Xdmf>\n",
            time, ncell_total, 8 * ncell_total, xyz_base, ncell_total,
            attr_base);
    fclose(xmf);
  }
  xyz = (float *)malloc(3 * 8 * ncell * sizeof *xyz);
  attr = (float *)malloc(ncell * sizeof *attr);
  k = 0;
  l = 0;
  for (i = 0; i < sim.nblk; i++) {
    const struct Blk *b = &sim.blk[i];
    const Real *chi = BLK(i) + F_CHI * BS3;
    j = 0;
    for (z = 0; z < BS; z++)
      for (y = 0; y < BS; y++)
        for (x = 0; x < BS; x++) {
          double u0, v0, w0, u1, v1, w1, h;
          h = sim.h0 / (1 << b->level);
          u0 = b->origin[0] + h * x;
          v0 = b->origin[1] + h * y;
          w0 = b->origin[2] + h * z;
          u1 = u0 + h;
          v1 = v0 + h;
          w1 = w0 + h;
          xyz[k++] = u0;
          xyz[k++] = v0;
          xyz[k++] = w0;
          xyz[k++] = u0;
          xyz[k++] = v0;
          xyz[k++] = w1;
          xyz[k++] = u0;
          xyz[k++] = v1;
          xyz[k++] = w1;
          xyz[k++] = u0;
          xyz[k++] = v1;
          xyz[k++] = w0;
          xyz[k++] = u1;
          xyz[k++] = v0;
          xyz[k++] = w0;
          xyz[k++] = u1;
          xyz[k++] = v0;
          xyz[k++] = w1;
          xyz[k++] = u1;
          xyz[k++] = v1;
          xyz[k++] = w1;
          xyz[k++] = u1;
          xyz[k++] = v1;
          xyz[k++] = w0;
          attr[l++] = chi[j++];
        }
  }
  MPI_File_open(MPI_COMM_WORLD, xyz_path, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &mpi_file);
  MPI_File_write_at_all(mpi_file, 3 * 8 * offset * sizeof *xyz, xyz,
                        3 * 8 * ncell * sizeof *xyz, MPI_BYTE,
                        MPI_STATUS_IGNORE);
  MPI_File_close(&mpi_file);
  free(xyz);
  MPI_File_open(MPI_COMM_WORLD, attr_path, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &mpi_file);
  MPI_File_write_at_all(mpi_file, offset * sizeof *attr, attr,
                        ncell * sizeof *attr, MPI_BYTE, MPI_STATUS_IGNORE);
  MPI_File_close(&mpi_file);
  free(attr);
}

static void parse_arguments(const struct Params *parser) {
  sim.bpdx = param_int(parser, "bpdx", 0);
  sim.bpdy = param_int(parser, "bpdy", 0);
  sim.bpdz = param_int(parser, "bpdz", 0);
  sim.levelMax = param_int(parser, "levelMax", 0);
  sim.levelStart = param_int(parser, "levelStart", sim.levelMax - 1);
  sim.Rtol = param_real(parser, "Rtol", 0);
  sim.Ctol = param_real(parser, "Ctol", 0);
  sim.extents[0] = param_real(parser, "extentx", 0);
  sim.extents[1] = param_real(parser, "extenty", 0);
  sim.extents[2] = param_real(parser, "extentz", 0);
  if (sim.extents[0] + sim.extents[1] + sim.extents[2] < 1e-21)
    sim.extents[0] = param_real(parser, "extent", 1);
  sim.CFL = param_real(parser, "CFL", .1);
  sim.dt = param_real(parser, "dt", 0);
  sim.endTime = param_real(parser, "tend", 0);
  sim.nu = param_real(parser, "nu", 0);
  sim.lambda = param_real(parser, "lambda", 1e6);
  sim.dumpTime = param_real(parser, "tdump", 0.0);
  sim.uinf[0] = param_real(parser, "uinfx", 0.0);
  sim.uinf[1] = param_real(parser, "uinfy", 0.0);
  sim.uinf[2] = param_real(parser, "uinfz", 0.0);
  sim.rampup = param_int(parser, "rampup", 100);
  sim.nsteps = param_int(parser, "nsteps", 0);
  sim.DLM = param_real(parser, "use-dlm", 0);
  sim.PoissonErrorTol = param_real(parser, "poissonTol", 1e-6);
  sim.PoissonErrorTolRel = param_real(parser, "poissonTolRel", 1e-4);
  sim.bMeanConstraint = param_int(parser, "bMeanConstraint", 1);
  sim.uMax_allowed = param_real(parser, "umax", 10.0);
  if (sim.bpdx < 1 || sim.bpdy < 1 || sim.bpdz < 1) {
    fprintf(stderr, "Invalid bpd: %d x %d x %d\n", sim.bpdx, sim.bpdy, sim.bpdz);
    abort();
  }
  const int aux = 1 << (sim.levelMax - 1);
  const Real NFE[3] = {
      (Real)sim.bpdx * aux * BS,
      (Real)sim.bpdy * aux * BS,
      (Real)sim.bpdz * aux * BS,
  };
  Real maxbpd = NFE[0];
  if (NFE[1] > maxbpd)
    maxbpd = NFE[1];
  if (NFE[2] > maxbpd)
    maxbpd = NFE[2];
  sim.maxextent = sim.extents[0];
  if (sim.extents[1] > sim.maxextent)
    sim.maxextent = sim.extents[1];
  if (sim.extents[2] > sim.maxextent)
    sim.maxextent = sim.extents[2];
  int maxb = sim.bpdx;
  if (sim.bpdy > maxb)
    maxb = sim.bpdy;
  if (sim.bpdz > maxb)
    maxb = sim.bpdz;
  sim.h0 = sim.maxextent / maxb / BS;
  if (sim.extents[0] <= 0 || sim.extents[1] <= 0 || sim.extents[2] <= 0) {
    sim.extents[0] = (NFE[0] / maxbpd) * sim.maxextent;
    sim.extents[1] = (NFE[1] / maxbpd) * sim.maxextent;
    sim.extents[2] = (NFE[2] / maxbpd) * sim.maxextent;
  } else {
    fprintf(stderr, "Invalid extent: %f x %f x %f\n", sim.extents[0],
            sim.extents[1], sim.extents[2]);
    abort();
  }
  sim.hmin = sim.extents[0] / NFE[0];
  sim.hmax = sim.extents[0] * aux / NFE[0];
  sim.step_2nd_start = 2;
  sim.coefU[0] = 1.5;
  sim.coefU[1] = -2.0;
  sim.coefU[2] = 0.5;
  sim.MeshChanged = 1;
  sim.StaticObstacles = param_bool(parser, "StaticObstacles", 0);
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
  const Real magI = sqrt(s->normalI[0] * s->normalI[0] +
                         s->normalI[1] * s->normalI[1] +
                         s->normalI[2] * s->normalI[2]);
  const Real magJ = sqrt(s->normalJ[0] * s->normalJ[0] +
                         s->normalJ[1] * s->normalJ[1] +
                         s->normalJ[2] * s->normalJ[2]);
  const Real magK = sqrt(s->normalK[0] * s->normalK[0] +
                         s->normalK[1] * s->normalK[1] +
                         s->normalK[2] * s->normalK[2]);
  const Real invMagI = (Real)1 / magI;
  const Real invMagJ = (Real)1 / magJ;
  const Real invMagK = (Real)1 / magK;
  for (int i = 0; i < 3; ++i) {
    s->normalI[i] = fabs(s->normalI[i]) * invMagI;
    s->normalJ[i] = fabs(s->normalJ[i]) * invMagJ;
    s->normalK[i] = fabs(s->normalK[i]) * invMagK;
  }
}
static void seg_prepare(struct Segment *s, int s0, int s1, const Real bbox[3][2],
                        Real h) {
  s->safe_distance = (1 + 2) * h;
  s->s0 = s0;
  s->s1 = s1;
  s->normalI[0] = 1; s->normalI[1] = 0; s->normalI[2] = 0;
  s->normalJ[0] = 0; s->normalJ[1] = 1; s->normalJ[2] = 0;
  s->normalK[0] = 0; s->normalK[1] = 0; s->normalK[2] = 1;
  for (int i = 0; i < 3; ++i) {
    s->w[i] = (bbox[i][1] - bbox[i][0]) / 2 + s->safe_distance;
    s->c[i] = (bbox[i][1] + bbox[i][0]) / 2;
  }
}
static void seg_to_frame(struct Segment *s, const Real position[3],
                         const Real quaternion[4]) {
  const Real a = quaternion[0];
  const Real x = quaternion[1];
  const Real y = quaternion[2];
  const Real z = quaternion[3];
  const Real Rmatrix[3][3] = {
      {(Real)1. - 2 * (y * y + z * z), (Real)2 * (x * y - z * a),
       (Real)2 * (x * z + y * a)},
      {(Real)2 * (x * y + z * a), (Real)1. - 2 * (x * x + z * z),
       (Real)2 * (y * z - x * a)},
      {(Real)2 * (x * z - y * a), (Real)2 * (y * z + x * a),
       (Real)1. - 2 * (x * x + y * y)}};
  const Real p[3] = {s->c[0], s->c[1], s->c[2]};
  const Real nx[3] = {s->normalI[0], s->normalI[1], s->normalI[2]};
  const Real ny[3] = {s->normalJ[0], s->normalJ[1], s->normalJ[2]};
  const Real nz[3] = {s->normalK[0], s->normalK[1], s->normalK[2]};
  for (int i = 0; i < 3; ++i) {
    s->c[i] = Rmatrix[i][0] * p[0] + Rmatrix[i][1] * p[1] + Rmatrix[i][2] * p[2];
    s->normalI[i] =
        Rmatrix[i][0] * nx[0] + Rmatrix[i][1] * nx[1] + Rmatrix[i][2] * nx[2];
    s->normalJ[i] =
        Rmatrix[i][0] * ny[0] + Rmatrix[i][1] * ny[1] + Rmatrix[i][2] * ny[2];
    s->normalK[i] =
        Rmatrix[i][0] * nz[0] + Rmatrix[i][1] * nz[1] + Rmatrix[i][2] * nz[2];
  }
  s->c[0] += position[0];
  s->c[1] += position[1];
  s->c[2] += position[2];
  seg_normalize(s);
  const Real widthXvec[] = {s->w[0] * s->normalI[0], s->w[0] * s->normalI[1],
                            s->w[0] * s->normalI[2]};
  const Real widthYvec[] = {s->w[1] * s->normalJ[0], s->w[1] * s->normalJ[1],
                            s->w[1] * s->normalJ[2]};
  const Real widthZvec[] = {s->w[2] * s->normalK[0], s->w[2] * s->normalK[1],
                            s->w[2] * s->normalK[2]};
  for (int i = 0; i < 3; ++i) {
    s->objBoxLabFr[i][0] = s->c[i] - widthXvec[i] - widthYvec[i] - widthZvec[i];
    s->objBoxLabFr[i][1] = s->c[i] + widthXvec[i] + widthYvec[i] + widthZvec[i];
    s->objBoxObjFr[i][0] = s->c[i] - s->w[i];
    s->objBoxObjFr[i][1] = s->c[i] + s->w[i];
  }
}
static int seg_intersects(const struct Segment *s, const Real start[3],
                          const Real end[3]) {
  const Real AABB_w[3] = {(end[0] - start[0]) / 2 + s->safe_distance,
                          (end[1] - start[1]) / 2 + s->safe_distance,
                          (end[2] - start[2]) / 2 + s->safe_distance};
  const Real AABB_c[3] = {(end[0] + start[0]) / 2, (end[1] + start[1]) / 2,
                          (end[2] + start[2]) / 2};
  const Real AABB_box[3][2] = {{AABB_c[0] - AABB_w[0], AABB_c[0] + AABB_w[0]},
                               {AABB_c[1] - AABB_w[1], AABB_c[1] + AABB_w[1]},
                               {AABB_c[2] - AABB_w[2], AABB_c[2] + AABB_w[2]}};
  for (int d = 0; d < 3; d++) {
    const Real lo = s->objBoxLabFr[d][0] > AABB_box[d][0] ? s->objBoxLabFr[d][0]
                                                          : AABB_box[d][0];
    const Real hi = s->objBoxLabFr[d][1] < AABB_box[d][1] ? s->objBoxLabFr[d][1]
                                                          : AABB_box[d][1];
    if (hi - lo < 0)
      return 0;
  }
  const Real widthXbox[3] = {AABB_w[0] * s->normalI[0], AABB_w[0] * s->normalJ[0],
                             AABB_w[0] * s->normalK[0]};
  const Real widthYbox[3] = {AABB_w[1] * s->normalI[1], AABB_w[1] * s->normalJ[1],
                             AABB_w[1] * s->normalK[1]};
  const Real widthZbox[3] = {AABB_w[2] * s->normalI[2], AABB_w[2] * s->normalJ[2],
                             AABB_w[2] * s->normalK[2]};
  const Real boxBox[3][2] = {
      {AABB_c[0] - widthXbox[0] - widthYbox[0] - widthZbox[0],
       AABB_c[0] + widthXbox[0] + widthYbox[0] + widthZbox[0]},
      {AABB_c[1] - widthXbox[1] - widthYbox[1] - widthZbox[1],
       AABB_c[1] + widthXbox[1] + widthYbox[1] + widthZbox[1]},
      {AABB_c[2] - widthXbox[2] - widthYbox[2] - widthZbox[2],
       AABB_c[2] + widthXbox[2] + widthYbox[2] + widthZbox[2]}};
  for (int d = 0; d < 3; d++) {
    const Real lo = boxBox[d][0] > s->objBoxObjFr[d][0] ? boxBox[d][0]
                                                        : s->objBoxObjFr[d][0];
    const Real hi = boxBox[d][1] < s->objBoxObjFr[d][1] ? boxBox[d][1]
                                                        : s->objBoxObjFr[d][1];
    if (hi - lo < 0)
      return 0;
  }
  return 1;
}

struct Frame {
  const struct Midline *m;
  Real position[3], quaternion[4], R[3][3];
};
static void frame_init(struct Frame *f, const struct Fish *fish) {
  const Real *q = fish->quaternion;
  f->m = &fish->m;
  for (int i = 0; i < 3; i++)
    f->position[i] = fish->position[i];
  for (int i = 0; i < 4; i++)
    f->quaternion[i] = q[i];
  f->R[0][0] = 1 - 2 * (q[2] * q[2] + q[3] * q[3]);
  f->R[0][1] = 2 * (q[1] * q[2] - q[3] * q[0]);
  f->R[0][2] = 2 * (q[1] * q[3] + q[2] * q[0]);
  f->R[1][0] = 2 * (q[1] * q[2] + q[3] * q[0]);
  f->R[1][1] = 1 - 2 * (q[1] * q[1] + q[3] * q[3]);
  f->R[1][2] = 2 * (q[2] * q[3] - q[1] * q[0]);
  f->R[2][0] = 2 * (q[1] * q[3] - q[2] * q[0]);
  f->R[2][1] = 2 * (q[2] * q[3] + q[1] * q[0]);
  f->R[2][2] = 1 - 2 * (q[1] * q[1] + q[2] * q[2]);
}
static Real euler_dist_sq(const Real a[3], const Real b[3]) {
  return pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2);
}
static void vel_to_frame(const struct Frame *f, Real x[3]) {
  const Real p[3] = {x[0], x[1], x[2]};
  x[0] = f->R[0][0] * p[0] + f->R[0][1] * p[1] + f->R[0][2] * p[2];
  x[1] = f->R[1][0] * p[0] + f->R[1][1] * p[1] + f->R[1][2] * p[2];
  x[2] = f->R[2][0] * p[0] + f->R[2][1] * p[1] + f->R[2][2] * p[2];
}
static void to_frame(const struct Frame *f, Real x[3]) {
  const Real p[3] = {x[0], x[1], x[2]};
  x[0] = f->R[0][0] * p[0] + f->R[0][1] * p[1] + f->R[0][2] * p[2];
  x[1] = f->R[1][0] * p[0] + f->R[1][1] * p[1] + f->R[1][2] * p[2];
  x[2] = f->R[2][0] * p[0] + f->R[2][1] * p[1] + f->R[2][2] * p[2];
  x[0] += f->position[0];
  x[1] += f->position[1];
  x[2] += f->position[2];
}
static void from_frame(const struct Frame *f, Real x[3]) {
  const Real p[3] = {x[0] - f->position[0], x[1] - f->position[1],
                     x[2] - f->position[2]};
  x[0] = f->R[0][0] * p[0] + f->R[1][0] * p[1] + f->R[2][0] * p[2];
  x[1] = f->R[0][1] * p[0] + f->R[1][1] * p[1] + f->R[2][1] * p[2];
  x[2] = f->R[0][2] * p[0] + f->R[1][2] * p[1] + f->R[2][2] * p[2];
}
static Real dist_plane(const Real p1[3], const Real p2[3], const Real p3[3],
                       const Real s[3], const Real IN[3]) {
  const Real t[3] = {s[0] - p1[0], s[1] - p1[1], s[2] - p1[2]};
  const Real u[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
  const Real v[3] = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};
  const Real i[3] = {IN[0] - p1[0], IN[1] - p1[1], IN[2] - p1[2]};
  const Real n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                     u[0] * v[1] - u[1] * v[0]};
  const Real projInner = i[0] * n[0] + i[1] * n[1] + i[2] * n[2];
  const Real signIn = projInner > 0 ? 1 : -1;
  const Real norm = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  return signIn * (t[0] * n[0] + t[1] * n[1] + t[2] * n[2]) / norm;
}
static void construct_internal(const struct Frame *fr, Real h, Real ox, Real oy,
                               Real oz, struct ObstacleBlock *defblock,
                               const struct Segment *const *vSegments, int nseg) {
  const struct Midline *cfish = fr->m;
  Real org[3] = {ox - h, oy - h, oz - h};
  const Real invh = 1.0 / h;
  const int BSP[3] = {BS + 2, BS + 2, BS + 2};
  const Real *rX = cfish->rX, *norX = cfish->norX, *vBinX = cfish->vBinX;
  const Real *rY = cfish->rY, *norY = cfish->norY, *vBinY = cfish->vBinY;
  const Real *rZ = cfish->rZ, *norZ = cfish->norZ, *vBinZ = cfish->vBinZ;
  const Real *vX = cfish->vX, *vNorX = cfish->vNorX, *binX = cfish->binX;
  const Real *vY = cfish->vY, *vNorY = cfish->vNorY, *binY = cfish->binY;
  const Real *vZ = cfish->vZ, *vNorZ = cfish->vNorZ, *binZ = cfish->binZ;
  const Real *width = cfish->width, *height = cfish->height;
  for (int i = 0; i < nseg; ++i) {
    const int firstSegm = vSegments[i]->s0 > 1 ? vSegments[i]->s0 : 1;
    const int lastSegm =
        vSegments[i]->s1 < cfish->Nm - 2 ? vSegments[i]->s1 : cfish->Nm - 2;
    for (int ss = firstSegm; ss <= lastSegm; ++ss) {
      const Real myWidth = width[ss], myHeight = height[ss];
      const int Nh = floor(myHeight / h);
      for (int ih = -Nh + 1; ih < Nh; ++ih) {
        const Real offsetH = ih * h;
        const Real currWidth = myWidth * sqrt(1 - pow(offsetH / myHeight, 2));
        const int Nw = floor(currWidth / h);
        for (int iw = -Nw + 1; iw < Nw; ++iw) {
          const Real offsetW = iw * h;
          Real xp[3] = {rX[ss] + offsetW * norX[ss] + offsetH * binX[ss],
                        rY[ss] + offsetW * norY[ss] + offsetH * binY[ss],
                        rZ[ss] + offsetW * norZ[ss] + offsetH * binZ[ss]};
          to_frame(fr, xp);
          xp[0] = (xp[0] - org[0]) * invh;
          xp[1] = (xp[1] - org[1]) * invh;
          xp[2] = (xp[2] - org[2]) * invh;
          const Real ap[3] = {floor(xp[0]), floor(xp[1]), floor(xp[2])};
          const int iap[3] = {(int)ap[0], (int)ap[1], (int)ap[2]};
          if (iap[0] + 2 <= 0 || iap[0] >= BSP[0])
            continue;
          if (iap[1] + 2 <= 0 || iap[1] >= BSP[1])
            continue;
          if (iap[2] + 2 <= 0 || iap[2] >= BSP[2])
            continue;
          Real udef[3] = {vX[ss] + offsetW * vNorX[ss] + offsetH * vBinX[ss],
                          vY[ss] + offsetW * vNorY[ss] + offsetH * vBinY[ss],
                          vZ[ss] + offsetW * vNorZ[ss] + offsetH * vBinZ[ss]};
          vel_to_frame(fr, udef);
          Real wghts[3][2];
          for (int c = 0; c < 3; ++c) {
            const Real t[2] = {fabs(xp[c] - ap[c]), fabs(xp[c] - (ap[c] + 1))};
            wghts[c][0] = 1.0 - t[0];
            wghts[c][1] = 1.0 - t[1];
          }
          const int z0 = iap[2] > 0 ? iap[2] : 0;
          const int z1 = iap[2] + 2 < BSP[2] ? iap[2] + 2 : BSP[2];
          const int y0 = iap[1] > 0 ? iap[1] : 0;
          const int y1 = iap[1] + 2 < BSP[1] ? iap[1] + 2 : BSP[1];
          const int x0 = iap[0] > 0 ? iap[0] : 0;
          const int x1 = iap[0] + 2 < BSP[0] ? iap[0] + 2 : BSP[0];
          for (int idz = z0; idz < z1; ++idz)
            for (int idy = y0; idy < y1; ++idy)
              for (int idx = x0; idx < x1; ++idx) {
                const int sx = idx - iap[0], sy = idy - iap[1],
                          sz = idz - iap[2];
                const Real wxwywz = wghts[2][sz] * wghts[1][sy] * wghts[0][sx];
                if (idz - 1 >= 0 && idz - 1 < BS && idy - 1 >= 0 &&
                    idy - 1 < BS && idx - 1 >= 0 && idx - 1 < BS) {
                  defblock->udef[idz - 1][idy - 1][idx - 1][0] += wxwywz * udef[0];
                  defblock->udef[idz - 1][idy - 1][idx - 1][1] += wxwywz * udef[1];
                  defblock->udef[idz - 1][idy - 1][idx - 1][2] += wxwywz * udef[2];
                  defblock->chi[idz - 1][idy - 1][idx - 1] += wxwywz;
                }
                if (fabs(defblock->sdfLab[idz][idy][idx] + 1) < DBL_EPSILON)
                  defblock->sdfLab[idz][idy][idx] = 1;
              }
        }
      }
    }
  }
}
static void construct_surface(const struct Frame *fr, Real h, Real ox, Real oy,
                              Real oz, struct ObstacleBlock *defblock,
                              const struct Segment *const *vSegments, int nseg) {
  const struct Midline *cfish = fr->m;
  const Real *rX = cfish->rX, *norX = cfish->norX, *vBinX = cfish->vBinX;
  const Real *rY = cfish->rY, *norY = cfish->norY, *vBinY = cfish->vBinY;
  const Real *rZ = cfish->rZ, *norZ = cfish->norZ, *vBinZ = cfish->vBinZ;
  const Real *vX = cfish->vX, *vNorX = cfish->vNorX, *binX = cfish->binX;
  const Real *vY = cfish->vY, *vNorY = cfish->vNorY, *binY = cfish->binY;
  const Real *vZ = cfish->vZ, *vNorZ = cfish->vNorZ, *binZ = cfish->binZ;
  Real *width = cfish->width;
  Real *height = cfish->height;
  const Real org[3] = {ox - h, oy - h, oz - h};
  const Real invh = 1.0 / h;
  const int BSP[3] = {BS + 2, BS + 2, BS + 2};
  Real myP[3] = {rX[0], rY[0], rZ[0]};
  to_frame(fr, myP);
  for (int i = 0; i < nseg; ++i) {
    const int firstSegm = vSegments[i]->s0 > 1 ? vSegments[i]->s0 : 1;
    const int lastSegm =
        vSegments[i]->s1 < cfish->Nm - 2 ? vSegments[i]->s1 : cfish->Nm - 2;
    for (int ss = firstSegm; ss <= lastSegm; ++ss) {
      if (height[ss] <= 0)
        height[ss] = 1e-10;
      if (width[ss] <= 0)
        width[ss] = 1e-10;
      const Real major_axis = height[ss] > width[ss] ? height[ss] : width[ss];
      const Real dtheta_tgt = fabs(asin(h / (major_axis + h) / 2));
      int Ntheta = ceil(2 * M_PI / dtheta_tgt);
      if (Ntheta % 2 == 1)
        Ntheta++;
      const Real dtheta = 2 * M_PI / ((Real)Ntheta);
      const Real offset = height[ss] > width[ss] ? M_PI / 2 : 0;
      for (int tt = 0; tt < Ntheta; ++tt) {
        const Real theta = tt * dtheta + offset;
        const Real sinth = sin(theta), costh = cos(theta);
        myP[0] = rX[ss] + width[ss] * costh * norX[ss] +
                 height[ss] * sinth * binX[ss];
        myP[1] = rY[ss] + width[ss] * costh * norY[ss] +
                 height[ss] * sinth * binY[ss];
        myP[2] = rZ[ss] + width[ss] * costh * norZ[ss] +
                 height[ss] * sinth * binZ[ss];
        to_frame(fr, myP);
        const int iap[3] = {(int)floor((myP[0] - org[0]) * invh),
                            (int)floor((myP[1] - org[1]) * invh),
                            (int)floor((myP[2] - org[2]) * invh)};
        const int nei = 3;
        const int ST[3] = {iap[0] - nei, iap[1] - nei, iap[2] - nei};
        const int EN[3] = {iap[0] + nei, iap[1] + nei, iap[2] + nei};
        if (EN[0] <= 0 || ST[0] > BSP[0])
          continue;
        if (EN[1] <= 0 || ST[1] > BSP[1])
          continue;
        if (EN[2] <= 0 || ST[2] > BSP[2])
          continue;
        Real pP[3] = {rX[ss + 1] + width[ss + 1] * costh * norX[ss + 1] +
                          height[ss + 1] * sinth * binX[ss + 1],
                      rY[ss + 1] + width[ss + 1] * costh * norY[ss + 1] +
                          height[ss + 1] * sinth * binY[ss + 1],
                      rZ[ss + 1] + width[ss + 1] * costh * norZ[ss + 1] +
                          height[ss + 1] * sinth * binZ[ss + 1]};
        Real pM[3] = {rX[ss - 1] + width[ss - 1] * costh * norX[ss - 1] +
                          height[ss - 1] * sinth * binX[ss - 1],
                      rY[ss - 1] + width[ss - 1] * costh * norY[ss - 1] +
                          height[ss - 1] * sinth * binY[ss - 1],
                      rZ[ss - 1] + width[ss - 1] * costh * norZ[ss - 1] +
                          height[ss - 1] * sinth * binZ[ss - 1]};
        to_frame(fr, pM);
        to_frame(fr, pP);
        Real udef[3] = {vX[ss] + width[ss] * costh * vNorX[ss] +
                            height[ss] * sinth * vBinX[ss],
                        vY[ss] + width[ss] * costh * vNorY[ss] +
                            height[ss] * sinth * vBinY[ss],
                        vZ[ss] + width[ss] * costh * vNorZ[ss] +
                            height[ss] * sinth * vBinZ[ss]};
        vel_to_frame(fr, udef);
        const int z0 = ST[2] > 0 ? ST[2] : 0, z1 = EN[2] < BSP[2] ? EN[2] : BSP[2];
        const int y0 = ST[1] > 0 ? ST[1] : 0, y1 = EN[1] < BSP[1] ? EN[1] : BSP[1];
        const int x0 = ST[0] > 0 ? ST[0] : 0, x1 = EN[0] < BSP[0] ? EN[0] : BSP[0];
        for (int sz = z0; sz < z1; ++sz)
          for (int sy = y0; sy < y1; ++sy)
            for (int sx = x0; sx < x1; ++sx) {
              Real p[3];
              p[0] = ox + h * (sx - 1 + 0.5);
              p[1] = oy + h * (sy - 1 + 0.5);
              p[2] = oz + h * (sz - 1 + 0.5);
              const Real dist0 = euler_dist_sq(p, myP);
              const Real distP = euler_dist_sq(p, pP);
              const Real distM = euler_dist_sq(p, pM);
              if (fabs(defblock->sdfLab[sz][sy][sx]) < min3(dist0, distP, distM))
                continue;
              if (min3(dist0, distP, distM) > 4 * h * h)
                continue;
              from_frame(fr, p);
              int close_s = ss, secnd_s = ss + (distP < distM ? 1 : -1);
              Real dist1 = dist0, dist2 = distP < distM ? distP : distM;
              if (distP < dist0 || distM < dist0) {
                dist1 = dist2;
                dist2 = dist0;
                close_s = secnd_s;
                secnd_s = ss;
              }
              const Real Wc = 1 - sqrt(dist1) * (invh / 3);
              const Real W = Wc > (Real)0 ? Wc : (Real)0;
              const int inRange =
                  (sz - 1 >= 0 && sz - 1 < BS && sy - 1 >= 0 && sy - 1 < BS &&
                   sx - 1 >= 0 && sx - 1 < BS);
              if (inRange) {
                defblock->udef[sz - 1][sy - 1][sx - 1][0] = W * udef[0];
                defblock->udef[sz - 1][sy - 1][sx - 1][1] = W * udef[1];
                defblock->udef[sz - 1][sy - 1][sx - 1][2] = W * udef[2];
                defblock->chi[sz - 1][sy - 1][sx - 1] = W;
              }
              const Real R1[3] = {rX[secnd_s] - rX[close_s],
                                  rY[secnd_s] - rY[close_s],
                                  rZ[secnd_s] - rZ[close_s]};
              const Real normR1 =
                  1.0 / (1e-21 + sqrt(R1[0] * R1[0] + R1[1] * R1[1] + R1[2] * R1[2]));
              const Real nn[3] = {R1[0] * normR1, R1[1] * normR1, R1[2] * normR1};
              const Real P1[3] = {width[close_s] * costh * norX[close_s] +
                                      height[close_s] * sinth * binX[close_s],
                                  width[close_s] * costh * norY[close_s] +
                                      height[close_s] * sinth * binY[close_s],
                                  width[close_s] * costh * norZ[close_s] +
                                      height[close_s] * sinth * binZ[close_s]};
              const Real P2[3] = {width[secnd_s] * costh * norX[secnd_s] +
                                      height[secnd_s] * sinth * binX[secnd_s],
                                  width[secnd_s] * costh * norY[secnd_s] +
                                      height[secnd_s] * sinth * binY[secnd_s],
                                  width[secnd_s] * costh * norZ[secnd_s] +
                                      height[secnd_s] * sinth * binZ[secnd_s]};
              const Real dot1 = P1[0] * R1[0] + P1[1] * R1[1] + P1[2] * R1[2];
              const Real dot2 = P2[0] * R1[0] + P2[1] * R1[1] + P2[2] * R1[2];
              const Real base1 = dot1 * normR1;
              const Real base2 = dot2 * normR1;
              const Real radius_close = pow(width[close_s] * costh, 2) +
                                        pow(height[close_s] * sinth, 2) -
                                        base1 * base1;
              const Real radius_second = pow(width[secnd_s] * costh, 2) +
                                         pow(height[secnd_s] * sinth, 2) -
                                         base2 * base2;
              const Real center_close[3] = {rX[close_s] - nn[0] * base1,
                                            rY[close_s] - nn[1] * base1,
                                            rZ[close_s] - nn[2] * base1};
              const Real center_second[3] = {rX[secnd_s] + nn[0] * base2,
                                             rY[secnd_s] + nn[1] * base2,
                                             rZ[secnd_s] + nn[2] * base2};
              const Real dSsq = pow(center_close[0] - center_second[0], 2) +
                                pow(center_close[1] - center_second[1], 2) +
                                pow(center_close[2] - center_second[2], 2);
              const Real corr = 2 * sqrt(radius_close * radius_second);
              if (close_s == cfish->Nm - 2 || secnd_s == cfish->Nm - 2) {
                const int TT = cfish->Nm - 1, TS = cfish->Nm - 2;
                const Real PC[3] = {rX[TT], rY[TT], rZ[TT]};
                const Real PF[3] = {rX[TS], rY[TS], rZ[TS]};
                const Real DXT = p[0] - PF[0];
                const Real DYT = p[1] - PF[1];
                const Real DZT = p[2] - PF[2];
                const Real projW = (width[TS] * norX[TS]) * DXT +
                                   (width[TS] * norY[TS]) * DYT +
                                   (width[TS] * norZ[TS]) * DZT;
                const Real projH = (height[TS] * binX[TS]) * DXT +
                                   (height[TS] * binY[TS]) * DYT +
                                   (height[TS] * binZ[TS]) * DZT;
                const int signW = projW > 0 ? 1 : -1;
                const int signH = projH > 0 ? 1 : -1;
                const Real PT[3] = {rX[TS] + signH * height[TS] * binX[TS],
                                    rY[TS] + signH * height[TS] * binY[TS],
                                    rZ[TS] + signH * height[TS] * binZ[TS]};
                const Real PP[3] = {rX[TS] + signW * width[TS] * norX[TS],
                                    rY[TS] + signW * width[TS] * norY[TS],
                                    rZ[TS] + signW * width[TS] * norZ[TS]};
                defblock->sdfLab[sz][sy][sx] = dist_plane(PC, PT, PP, p, PF);
              } else if (dSsq >= radius_close + radius_second - corr) {
                const Real xMidl[3] = {rX[close_s], rY[close_s], rZ[close_s]};
                const Real grd2ML = euler_dist_sq(p, xMidl);
                const Real sign = grd2ML > radius_close ? -1 : 1;
                defblock->sdfLab[sz][sy][sx] = sign * dist1;
              } else {
                const Real Rsq = (radius_close + radius_second - corr + dSsq) *
                                 (radius_close + radius_second + corr + dSsq) /
                                 4 / dSsq;
                const Real maxAx =
                    radius_close > radius_second ? radius_close : radius_second;
                const Real d = sqrt((Rsq - maxAx) / dSsq);
                Real sign;
                if (radius_close > radius_second) {
                  const Real xMidl[3] = {
                      center_close[0] + (center_close[0] - center_second[0]) * d,
                      center_close[1] + (center_close[1] - center_second[1]) * d,
                      center_close[2] + (center_close[2] - center_second[2]) * d};
                  const Real grd2Core = euler_dist_sq(p, xMidl);
                  sign = grd2Core > Rsq ? -1 : 1;
                } else {
                  const Real xMidl[3] = {
                      center_second[0] + (center_second[0] - center_close[0]) * d,
                      center_second[1] + (center_second[1] - center_close[1]) * d,
                      center_second[2] + (center_second[2] - center_close[2]) * d};
                  const Real grd2Core = euler_dist_sq(p, xMidl);
                  sign = grd2Core > Rsq ? -1 : 1;
                }
                defblock->sdfLab[sz][sy][sx] = sign * dist1;
              }
            }
      }
    }
  }
}
static void signed_distance_sqrt(struct ObstacleBlock *defblock) {
  for (int iz = 0; iz < BS + 2; iz++)
    for (int iy = 0; iy < BS + 2; iy++)
      for (int ix = 0; ix < BS + 2; ix++) {
        if (iz < BS && iy < BS && ix < BS) {
          if (defblock->chi[iz][iy][ix] > DBL_EPSILON) {
            const Real normfac = 1.0 / defblock->chi[iz][iy][ix];
            defblock->udef[iz][iy][ix][0] *= normfac;
            defblock->udef[iz][iy][ix][1] *= normfac;
            defblock->udef[iz][iy][ix][2] *= normfac;
          }
        }
        defblock->sdfLab[iz][iy][ix] =
            defblock->sdfLab[iz][iy][ix] >= 0
                ? sqrt(defblock->sdfLab[iz][iy][ix])
                : -sqrt(-defblock->sdfLab[iz][iy][ix]);
      }
}
static void put_fish(const struct Frame *fr, Real h, Real ox, Real oy, Real oz,
                     struct ObstacleBlock *oblock,
                     const struct Segment *const *vSegments, int nseg) {
  memset(oblock->chi, 0, sizeof oblock->chi);
  memset(oblock->udef, 0, sizeof oblock->udef);
  Real *sdf = &oblock->sdfLab[0][0][0];
  for (int i = 0; i < (BS + 2) * (BS + 2) * (BS + 2); i++)
    sdf[i] = -1.;
  construct_internal(fr, h, ox, oy, oz, oblock, vSegments, nseg);
  construct_surface(fr, h, ox, oy, oz, oblock, vSegments, nseg);
  signed_distance_sqrt(oblock);
}
static void oblock_clear(struct ObstacleBlock *o) {
  o->filled = 0;
  o->nPoints = 0;
  o->CoM_x = o->CoM_y = o->CoM_z = 0;
  o->mass = 0;
  memset(o->chi, 0, sizeof o->chi);
  memset(o->udef, 0, sizeof o->udef);
  memset(o->sdfLab, 0, sizeof o->sdfLab);
}
static void fish_clear_blocks(struct Fish *f) {
  if (f->oblock)
    for (long long i = 0; i < sim.nblk; i++) {
      free(f->oblock[i]);
      f->oblock[i] = NULL;
    }
  free(f->oblock);
  f->oblock = NULL;
  free(f->myblk);
  free(f->seg_start);
  free(f->seg_idx);
  f->myblk = f->seg_start = f->seg_idx = NULL;
  f->nmyblk = f->nseg_idx = 0;
}
static void create_geometry(struct Fish *f) {
  struct Midline *m = &f->m;
  compute_midline(m, sim.time);
  integrate_linear_momentum(m);
  integrate_angular_momentum(m, sim.dt);
  const int Nm = m->Nm;
  const int Nsegments = ceil((Nm - 1.) / 8);
  struct Segment *vSegments =
      (struct Segment *)malloc(Nsegments * sizeof *vSegments);
  for (int i = 0; i < Nsegments; ++i) {
    const int nextidx = (i + 1) * (Nm - 1) / Nsegments;
    const int idx = i * (Nm - 1) / Nsegments;
    Real bbox[3][2] = {{1e9, -1e9}, {1e9, -1e9}, {1e9, -1e9}};
    for (int ss = idx; ss <= nextidx; ++ss) {
      const Real xBnd[4] = {m->rX[ss] + m->norX[ss] * m->width[ss],
                            m->rX[ss] - m->norX[ss] * m->width[ss],
                            m->rX[ss] + m->binX[ss] * m->height[ss],
                            m->rX[ss] - m->binX[ss] * m->height[ss]};
      const Real yBnd[4] = {m->rY[ss] + m->norY[ss] * m->width[ss],
                            m->rY[ss] - m->norY[ss] * m->width[ss],
                            m->rY[ss] + m->binY[ss] * m->height[ss],
                            m->rY[ss] - m->binY[ss] * m->height[ss]};
      const Real zBnd[4] = {m->rZ[ss] + m->norZ[ss] * m->width[ss],
                            m->rZ[ss] - m->norZ[ss] * m->width[ss],
                            m->rZ[ss] + m->binZ[ss] * m->height[ss],
                            m->rZ[ss] - m->binZ[ss] * m->height[ss]};
      const Real maxX = max4(xBnd[0], xBnd[1], xBnd[2], xBnd[3]);
      const Real maxY = max4(yBnd[0], yBnd[1], yBnd[2], yBnd[3]);
      const Real maxZ = max4(zBnd[0], zBnd[1], zBnd[2], zBnd[3]);
      const Real minX = min4(xBnd[0], xBnd[1], xBnd[2], xBnd[3]);
      const Real minY = min4(yBnd[0], yBnd[1], yBnd[2], yBnd[3]);
      const Real minZ = min4(zBnd[0], zBnd[1], zBnd[2], zBnd[3]);
      bbox[0][0] = minX < bbox[0][0] ? minX : bbox[0][0];
      bbox[0][1] = maxX > bbox[0][1] ? maxX : bbox[0][1];
      bbox[1][0] = minY < bbox[1][0] ? minY : bbox[1][0];
      bbox[1][1] = maxY > bbox[1][1] ? maxY : bbox[1][1];
      bbox[2][0] = minZ < bbox[2][0] ? minZ : bbox[2][0];
      bbox[2][1] = maxZ > bbox[2][1] ? maxZ : bbox[2][1];
    }
    seg_prepare(&vSegments[i], idx, nextidx, bbox, sim.hmin);
    seg_to_frame(&vSegments[i], f->position, f->quaternion);
  }
  fish_clear_blocks(f);
  f->oblock = (struct ObstacleBlock **)calloc(sim.nblk, sizeof *f->oblock);
  f->myblk = (int *)malloc(sim.nblk * sizeof *f->myblk);
  f->seg_start = (int *)malloc((sim.nblk + 1) * sizeof *f->seg_start);
  f->seg_idx = (int *)malloc(sim.nblk * Nsegments * sizeof *f->seg_idx);
  f->nmyblk = 0;
  f->nseg_idx = 0;
  for (long long i = 0; i < sim.nblk; ++i) {
    const struct Blk *b = &sim.blk[i];
    Real MINP[3], MAXP[3];
    blk_pos(b, 0, 0, 0, MINP);
    blk_pos(b, BS - 1, BS - 1, BS - 1, MAXP);
    int hasSegments = 0;
    for (int s = 0; s < Nsegments; ++s)
      if (seg_intersects(&vSegments[s], MINP, MAXP)) {
        if (!hasSegments) {
          hasSegments = 1;
          f->myblk[f->nmyblk] = i;
          f->seg_start[f->nmyblk] = f->nseg_idx;
          f->nmyblk++;
        }
        f->seg_idx[f->nseg_idx++] = s;
      }
    if (hasSegments) {
      f->oblock[i] = (struct ObstacleBlock *)malloc(sizeof(struct ObstacleBlock));
      oblock_clear(f->oblock[i]);
    }
  }
  f->seg_start[f->nmyblk] = f->nseg_idx;
  struct Frame fr;
  frame_init(&fr, f);
#pragma omp parallel for
  for (int j = 0; j < f->nmyblk; j++) {
    const int n = f->seg_start[j + 1] - f->seg_start[j];
    const struct Segment **S =
        (const struct Segment **)malloc(n * sizeof *S);
    for (int k = 0; k < n; k++)
      S[k] = &vSegments[f->seg_idx[f->seg_start[j] + k]];
    const struct Blk *b = &sim.blk[f->myblk[j]];
    put_fish(&fr, b->h, b->origin[0], b->origin[1], b->origin[2],
             f->oblock[f->myblk[j]], S, n);
    free(S);
  }
  free(vSegments);
}
static void clip_quantities(Real fmax, Real dfmax, Real dt, int zero,
                            Real fcandidate, Real dfcandidate, Real *f,
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
static void fish_create(struct Fish *f) {
  struct Midline *cFish = &f->m;
  const int Nm = cFish->Nm;
  const Real *q = f->quaternion;
  const Real Rmatrix3D[3] = {2 * (q[1] * q[3] - q[2] * q[0]),
                             2 * (q[2] * q[3] + q[1] * q[0]),
                             1 - 2 * (q[1] * q[1] + q[2] * q[2])};
  const Real d1 = cFish->rX[0] - cFish->rX[Nm / 2];
  const Real d2 = cFish->rY[0] - cFish->rY[Nm / 2];
  const Real d3 = cFish->rZ[0] - cFish->rZ[Nm / 2];
  const Real dn = pow(d1 * d1 + d2 * d2 + d3 * d3, 0.5) + 1e-21;
  const Real vx = d1 / dn;
  const Real vy = d2 / dn;
  const Real vz = d3 / dn;
  Real xx2 = Rmatrix3D[0] * vx + Rmatrix3D[1] * vy + Rmatrix3D[2] * vz;
  xx2 = xx2 > 1 ? 1 : (xx2 < -1 ? -1 : xx2);
  const Real pitch = asin(xx2);
  const Real roll = atan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                          1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
  const Real yaw = atan2(2.0 * (q[3] * q[0] + q[1] * q[2]),
                         -1.0 + 2.0 * (q[0] * q[0] + q[1] * q[1]));
  const int roll_is_small = fabs(roll) < M_PI / 9.;
  const int yaw_is_small = fabs(yaw) < M_PI / 9.;
  if (f->bCorrectPosition) {
    cFish->alpha = 1.0 + (f->position[0] - f->origC[0]) / f->length;
    cFish->dalpha = (f->transVel[0] + sim.uinf[0]) / f->length;
    if (roll_is_small == 0) {
      cFish->alpha = 1.0;
      cFish->dalpha = 0.0;
    } else if (cFish->alpha < 0.9) {
      cFish->alpha = 0.9;
      cFish->dalpha = 0.0;
    } else if (cFish->alpha > 1.1) {
      cFish->alpha = 1.1;
      cFish->dalpha = 0.0;
    }
    const Real y = f->absPos[1];
    const Real ytgt = f->origC[1];
    const Real dy = (ytgt - y) / f->length;
    const Real signY = dy > 0 ? 1 : -1;
    const Real yaw_tgt = 0;
    const Real dphi = yaw - yaw_tgt;
    const Real b = roll_is_small ? f->wyp * signY * dy * dphi : 0;
    const Real dbdt = sim.step > 1 ? (b - cFish->beta) / sim.dt : 0;
    clip_quantities(1.0, 5.0, sim.dt, 0, b, dbdt, &cFish->beta, &cFish->dbeta);
  }
  if (f->bCorrectPositionZ) {
    const Real pitch_tgt = 0;
    const Real dphi = pitch - pitch_tgt;
    const Real z = f->absPos[2];
    const Real ztgt = f->origC[2];
    const Real dz = (ztgt - z) / f->length;
    const Real signZ = dz > 0 ? 1 : -1;
    const Real g =
        (roll_is_small && yaw_is_small) ? -f->wzp * dphi * dz * signZ : 0.0;
    const Real dgdt = sim.step > 1 ? (g - cFish->gamma) / sim.dt : 0.0;
    const Real gmax = 0.10 / f->length;
    const Real dRdtmax = 0.1 * f->length / cFish->Tperiod;
    const Real dgdtmax = fabs(gmax * gmax * dRdtmax);
    clip_quantities(gmax, dgdtmax, sim.dt, 0, g, dgdt, &cFish->gamma,
                    &cFish->dgamma);
  }
  create_geometry(f);
}
static void fish_update(struct Fish *f) {
  Real *position = f->position, *absPos = f->absPos, *quaternion = f->quaternion;
  const Real *angVel = f->angVel, *transVel = f->transVel;
  const Real dqdt[4] = {
      (Real).5 * (-angVel[0] * quaternion[1] - angVel[1] * quaternion[2] -
                  angVel[2] * quaternion[3]),
      (Real).5 * (+angVel[0] * quaternion[0] + angVel[1] * quaternion[3] -
                  angVel[2] * quaternion[2]),
      (Real).5 * (-angVel[0] * quaternion[3] + angVel[1] * quaternion[0] +
                  angVel[2] * quaternion[1]),
      (Real).5 * (+angVel[0] * quaternion[2] - angVel[1] * quaternion[1] +
                  angVel[2] * quaternion[0])};
  if (sim.step < sim.step_2nd_start) {
    for (int d = 0; d < 3; d++) {
      f->old_position[d] = position[d];
      f->old_absPos[d] = absPos[d];
    }
    for (int d = 0; d < 4; d++)
      f->old_quaternion[d] = quaternion[d];
    position[0] += sim.dt * (transVel[0] + sim.uinf[0]);
    position[1] += sim.dt * (transVel[1] + sim.uinf[1]);
    position[2] += sim.dt * (transVel[2] + sim.uinf[2]);
    absPos[0] += sim.dt * transVel[0];
    absPos[1] += sim.dt * transVel[1];
    absPos[2] += sim.dt * transVel[2];
    quaternion[0] += sim.dt * dqdt[0];
    quaternion[1] += sim.dt * dqdt[1];
    quaternion[2] += sim.dt * dqdt[2];
    quaternion[3] += sim.dt * dqdt[3];
  } else {
    const Real aux = 1.0 / sim.coefU[0];
    Real temp[10] = {position[0],   position[1],  position[2],   absPos[0],
                     absPos[1],     absPos[2],    quaternion[0], quaternion[1],
                     quaternion[2], quaternion[3]};
    for (int d = 0; d < 3; d++)
      position[d] = aux * (sim.dt * (transVel[d] + sim.uinf[d]) +
                           (-sim.coefU[1] * position[d] -
                            sim.coefU[2] * f->old_position[d]));
    for (int d = 0; d < 3; d++)
      absPos[d] =
          aux * (sim.dt * (transVel[d]) +
                 (-sim.coefU[1] * absPos[d] - sim.coefU[2] * f->old_absPos[d]));
    for (int d = 0; d < 4; d++)
      quaternion[d] = aux * (sim.dt * (dqdt[d]) +
                             (-sim.coefU[1] * quaternion[d] -
                              sim.coefU[2] * f->old_quaternion[d]));
    for (int d = 0; d < 3; d++) {
      f->old_position[d] = temp[d];
      f->old_absPos[d] = temp[3 + d];
    }
    for (int d = 0; d < 4; d++)
      f->old_quaternion[d] = temp[6 + d];
  }
  const Real invD =
      1.0 / sqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                 quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
  quaternion[0] *= invD;
  quaternion[1] *= invD;
  quaternion[2] *= invD;
  quaternion[3] *= invD;
}
static void update_uinf(void) {
  int nSum[3] = {0, 0, 0};
  Real uSum[3] = {0, 0, 0};
  for (int i = 0; i < sim.nfish; i++) {
    const struct Fish *f = &sim.fish[i];
    for (int d = 0; d < 3; d++)
      if (f->bFixFrameOfRef[d]) {
        nSum[d] += 1;
        uSum[d] -= f->transVel[d];
      }
  }
  for (int d = 0; d < 3; d++)
    if (nSum[d] > 0)
      uSum[d] = uSum[d] / nSum[d];
  for (int d = 0; d < 3; d++)
    sim.uinf[d] = uSum[d];
}
static void characteristic_function(long long i) {
  const struct Blk *blk = &sim.blk[i];
  Real *b = BLK(i) + F_CHI * BS3;
  const Real h = blk->h, inv2h = .5 / h, vol = h * h * h;
  const int gp = 1;
  for (int obst_id = 0; obst_id < sim.nfish; obst_id++) {
    struct ObstacleBlock *o = sim.fish[obst_id].oblock[i];
    if (o == NULL)
      continue;
    o->CoM_x = 0;
    o->CoM_y = 0;
    o->CoM_z = 0;
    o->mass = 0;
    for (int z = 0; z < BS; ++z)
      for (int y = 0; y < BS; ++y)
        for (int x = 0; x < BS; ++x) {
          if (o->sdfLab[z + 1][y + 1][x + 1] > +gp * h ||
              o->sdfLab[z + 1][y + 1][x + 1] < -gp * h) {
            o->chi[z][y][x] = o->sdfLab[z + 1][y + 1][x + 1] > 0 ? 1 : 0;
          } else {
            const Real distPx = o->sdfLab[z + 1][y + 1][x + 1 + 1];
            const Real distMx = o->sdfLab[z + 1][y + 1][x + 1 - 1];
            const Real distPy = o->sdfLab[z + 1][y + 1 + 1][x + 1];
            const Real distMy = o->sdfLab[z + 1][y + 1 - 1][x + 1];
            const Real distPz = o->sdfLab[z + 1 + 1][y + 1][x + 1];
            const Real distMz = o->sdfLab[z + 1 - 1][y + 1][x + 1];
            const Real gradUX = inv2h * (distPx - distMx);
            const Real gradUY = inv2h * (distPy - distMy);
            const Real gradUZ = inv2h * (distPz - distMz);
            const Real gradUSq =
                gradUX * gradUX + gradUY * gradUY + gradUZ * gradUZ + DBL_EPSILON;
            const Real IplusX = distPx > 0.0 ? distPx : 0.0;
            const Real IminuX = distMx > 0.0 ? distMx : 0.0;
            const Real IplusY = distPy > 0.0 ? distPy : 0.0;
            const Real IminuY = distMy > 0.0 ? distMy : 0.0;
            const Real IplusZ = distPz > 0.0 ? distPz : 0.0;
            const Real IminuZ = distMz > 0.0 ? distMz : 0.0;
            const Real gradIX = inv2h * (IplusX - IminuX);
            const Real gradIY = inv2h * (IplusY - IminuY);
            const Real gradIZ = inv2h * (IplusZ - IminuZ);
            const Real numH = gradIX * gradUX + gradIY * gradUY + gradIZ * gradUZ;
            o->chi[z][y][x] = numH / gradUSq;
          }
          Real p[3];
          blk_pos(blk, x, y, z, p);
          const int j = z * BS * BS + y * BS + x;
          b[j] = o->chi[z][y][x] < b[j] ? b[j] : o->chi[z][y][x];
          o->CoM_x += o->chi[z][y][x] * vol * p[0];
          o->CoM_y += o->chi[z][y][x] * vol * p[1];
          o->CoM_z += o->chi[z][y][x] * vol * p[2];
          o->mass += o->chi[z][y][x] * vol;
        }
  }
}
static void invert_sym(const Real J[6], Real inv[6]) {
  const Real detJ = J[0] * (J[1] * J[2] - J[5] * J[5]) +
                    J[3] * (J[4] * J[5] - J[2] * J[3]) +
                    J[4] * (J[3] * J[5] - J[1] * J[4]);
  if (fabs(detJ) <= DBL_MIN) {
    for (int q = 0; q < 6; q++)
      inv[q] = 0;
  } else {
    inv[0] = (J[1] * J[2] - J[5] * J[5]) / detJ;
    inv[1] = (J[0] * J[2] - J[4] * J[4]) / detJ;
    inv[2] = (J[0] * J[1] - J[3] * J[3]) / detJ;
    inv[3] = (J[4] * J[5] - J[2] * J[3]) / detJ;
    inv[4] = (J[3] * J[5] - J[1] * J[4]) / detJ;
    inv[5] = (J[3] * J[4] - J[0] * J[5]) / detJ;
  }
}
static void compute_grid_com(void) {
  for (int k = 0; k < sim.nfish; k++) {
    struct Fish *f = &sim.fish[k];
    Real com[4] = {0.0, 0.0, 0.0, 0.0};
    for (long long i = 0; i < sim.nblk; i++) {
      const struct ObstacleBlock *o = f->oblock[i];
      if (o == NULL)
        continue;
      com[0] += o->mass;
      com[1] += o->CoM_x;
      com[2] += o->CoM_y;
      com[3] += o->CoM_z;
    }
    MPI_Allreduce(MPI_IN_PLACE, com, 4, MPI_Real, MPI_SUM, sim.comm);
    if (com[0] <= 0)
      continue;
    f->centerOfMass[0] = com[1] / com[0];
    f->centerOfMass[1] = com[2] / com[0];
    f->centerOfMass[2] = com[3] / com[0];
  }
}
static void integrate_udef_momenta(long long i) {
  const struct Blk *b = &sim.blk[i];
  for (int k = 0; k < sim.nfish; k++) {
    const struct Fish *f = &sim.fish[k];
    struct ObstacleBlock *o = f->oblock[i];
    if (o == NULL)
      continue;
    const Real *CM = f->centerOfMass;
    Real *M = o->mom;
    for (int q = 0; q < 13; q++)
      M[q] = 0;
    for (int z = 0; z < BS; ++z)
      for (int y = 0; y < BS; ++y)
        for (int x = 0; x < BS; ++x) {
          if (o->chi[z][y][x] <= 0)
            continue;
          Real p[3];
          blk_pos(b, x, y, z, p);
          const Real dv = b->h * b->h * b->h, X = o->chi[z][y][x];
          const Real *U = o->udef[z][y][x];
          p[0] -= CM[0];
          p[1] -= CM[1];
          p[2] -= CM[2];
          M[M_V] += X * dv;
          M[M_FX] += X * U[0] * dv;
          M[M_FY] += X * U[1] * dv;
          M[M_FZ] += X * U[2] * dv;
          M[M_TX] += X * (p[1] * U[2] - p[2] * U[1]) * dv;
          M[M_TY] += X * (p[2] * U[0] - p[0] * U[2]) * dv;
          M[M_TZ] += X * (p[0] * U[1] - p[1] * U[0]) * dv;
          M[M_J0] += X * (p[1] * p[1] + p[2] * p[2]) * dv;
          M[M_J3] -= X * p[0] * p[1] * dv;
          M[M_J1] += X * (p[0] * p[0] + p[2] * p[2]) * dv;
          M[M_J4] -= X * p[0] * p[2] * dv;
          M[M_J2] += X * (p[0] * p[0] + p[1] * p[1]) * dv;
          M[M_J5] -= X * p[1] * p[2] * dv;
        }
  }
}
static void accumulate_udef_momenta(void) {
  for (int k = 0; k < sim.nfish; k++) {
    struct Fish *f = &sim.fish[k];
    Real M[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (long long i = 0; i < sim.nblk; i++) {
      const struct ObstacleBlock *o = f->oblock[i];
      if (o == NULL)
        continue;
      for (int q = 0; q < 13; q++)
        M[q] += o->mom[q];
    }
    MPI_Allreduce(MPI_IN_PLACE, M, 13, MPI_Real, MPI_SUM, sim.comm);
    if (M[0] <= 0) {
      f->mass = 0;
      for (int d = 0; d < 3; d++)
        f->transVel_correction[d] = f->angVel_correction[d] = 0;
      for (int q = 0; q < 6; q++)
        f->J[q] = 0;
      continue;
    }
    const Real AM[3] = {M[4], M[5], M[6]};
    const Real J[6] = {M[7], M[8], M[9], M[10], M[11], M[12]};
    Real invJ[6];
    invert_sym(J, invJ);
    f->mass = M[0];
    f->transVel_correction[0] = M[1] / M[0];
    f->transVel_correction[1] = M[2] / M[0];
    f->transVel_correction[2] = M[3] / M[0];
    for (int q = 0; q < 6; q++)
      f->J[q] = J[q];
    f->angVel_correction[0] = invJ[0] * AM[0] + invJ[3] * AM[1] + invJ[4] * AM[2];
    f->angVel_correction[1] = invJ[3] * AM[0] + invJ[1] * AM[1] + invJ[5] * AM[2];
    f->angVel_correction[2] = invJ[4] * AM[0] + invJ[5] * AM[1] + invJ[2] * AM[2];
  }
}
static void remove_udef_momenta(void) {
  for (int k = 0; k < sim.nfish; k++) {
    const struct Fish *f = &sim.fish[k];
    const Real *av = f->angVel_correction;
    const Real *tv = f->transVel_correction;
    const Real *CM = f->centerOfMass;
#pragma omp parallel for schedule(dynamic, 1)
    for (long long i = 0; i < sim.nblk; i++) {
      struct ObstacleBlock *o = f->oblock[i];
      if (o == NULL)
        continue;
      const struct Blk *b = &sim.blk[i];
      for (int z = 0; z < BS; ++z)
        for (int y = 0; y < BS; ++y)
          for (int x = 0; x < BS; ++x) {
            Real p[3];
            blk_pos(b, x, y, z, p);
            p[0] -= CM[0];
            p[1] -= CM[1];
            p[2] -= CM[2];
            const Real rot[3] = {av[1] * p[2] - av[2] * p[1], av[2] * p[0] - av[0] * p[2],
                                 av[0] * p[1] - av[1] * p[0]};
            o->udef[z][y][x][0] -= tv[0] + rot[0];
            o->udef[z][y][x][1] -= tv[1] + rot[1];
            o->udef[z][y][x][2] -= tv[2] + rot[2];
          }
    }
  }
}
static void create_obstacles(Real dt) {
  if (sim.nfish == 0)
    return;
  if (sim.MeshChanged == 0 && sim.StaticObstacles)
    return;
  sim.MeshChanged = 0;
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < sim.nblk; ++i)
    memset(BLK(i) + F_CHI * BS3, 0, BS3 * sizeof(Real));
  update_uinf();
  for (int i = 0; i < sim.nfish; i++)
    fish_update(&sim.fish[i]);
  for (int i = 0; i < sim.nfish; i++)
    fish_create(&sim.fish[i]);
#pragma omp parallel for
  for (long long i = 0; i < sim.nblk; ++i)
    characteristic_function(i);
  compute_grid_com();
#pragma omp parallel for schedule(dynamic, 1)
  for (long long i = 0; i < sim.nblk; ++i)
    integrate_udef_momenta(i);
  accumulate_udef_momenta();
  remove_udef_momenta();
}

enum { Leave = 0, Refine = 1, Compress = -1 };
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
static long long node_key(int level, long long Z) {
  return nodes.level_base[level] + Z;
}
static struct Node *node_find(long long key, int create);
static void nodes_rehash(void) {
  struct Node *old = nodes.tab;
  long long oldcap = nodes.cap;
  nodes.cap = oldcap ? 2 * oldcap : 1 << 16;
  nodes.tab = (struct Node *)calloc(nodes.cap, sizeof *nodes.tab);
  nodes.n = 0;
  for (long long i = 0; i < oldcap; i++)
    if (old[i].used) {
      struct Node *nn = node_find(old[i].key, 1);
      *nn = old[i];
    }
  free(old);
}
static struct Node *node_find(long long key, int create) {
  if (nodes.cap == 0) {
    if (!create)
      return NULL;
    nodes_rehash();
  } else if (create && 2 * (nodes.n + 1) > nodes.cap)
    nodes_rehash();
  unsigned long long h = (unsigned long long)key * 0x9E3779B97F4A7C15ULL;
  long long i = (long long)(h >> 20) & (nodes.cap - 1);
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
      nd->state = Leave;
      nodes.n++;
      return nd;
    }
    if (nd->key == key)
      return nd;
    i = (i + 1) & (nodes.cap - 1);
  }
}
static struct Node *node(int level, long long Z) {
  return node_find(node_key(level, Z), 1);
}
static struct Node *node_get(int level, long long Z) {
  return node_find(node_key(level, Z), 0);
}
static void nodes_reset(void) {
  free(nodes.tab);
  nodes.tab = NULL;
  nodes.cap = nodes.n = 0;
  nodes_rehash();
}
static void nodes_init(void) {
  nodes.level_base = (long long *)malloc(sim.levelMax * sizeof(long long));
  for (int m = 0; m < sim.levelMax; m++) {
    const long long TwoPower = 1 << m;
    const long long Ntot = (long long)sim.bpdx * sim.bpdy * sim.bpdz *
                           TwoPower * TwoPower * TwoPower;
    nodes.level_base[m] = m == 0 ? Ntot : nodes.level_base[m - 1] + Ntot;
  }
  nodes.tab = NULL;
  nodes_reset();
}
static int nblocks_dim(int d, int level) {
  const int b = d == 0 ? sim.bpdx : d == 1 ? sim.bpdy : sim.bpdz;
  return b * (1 << level);
}
static long long zforward(int level, int i, int j, int k) {
  const int NX = sim.bpdx, NY = sim.bpdy, NZ = sim.bpdz;
  const int TwoPower = 1 << level;
  const int ix = (i + TwoPower * NX) % (NX * TwoPower);
  const int iy = (j + TwoPower * NY) % (NY * TwoPower);
  const int iz = (k + TwoPower * NZ) % (NZ * TwoPower);
  return sfc_forward(level, ix, iy, iz);
}
static long long znei(const struct Blk *b, int i, int j, int k) {
  return zforward(b->level, b->ix + i, b->iy + j, b->iz + k);
}
static long long zparent(const struct Blk *b) {
  return b->level == 0 ? 0 : zforward(b->level - 1, b->ix / 2, b->iy / 2, b->iz / 2);
}
static long long zchild(const struct Blk *b, int i, int j, int k) {
  return sfc_forward(b->level + 1, 2 * b->ix + i, 2 * b->iy + j, 2 * b->iz + k);
}
static long long encode(int level, long long Z, int ix, int iy, int iz) {
  const int lmax = sim.levelMax;
  long long retval = 0;
  for (int l = level; l >= 0; l--) {
    long long Zp = sfc_forward(l, ix, iy, iz);
    retval += Zp;
    ix /= 2;
    iy /= 2;
    iz /= 2;
  }
  int i, j, k;
  sfc_inverse(Z, level, &i, &j, &k);
  ix = 2 * i;
  iy = 2 * j;
  iz = 2 * k;
  for (int l = level + 1; l < lmax; l++) {
    long long Zc = sfc_forward(l, ix, iy, iz);
    Zc -= Zc % 8;
    retval += Zc;
    int ix1, iy1, iz1;
    sfc_inverse(Zc, l, &ix1, &iy1, &iz1);
    ix = 2 * ix1;
    iy = 2 * iy1;
    iz = 2 * iz1;
  }
  retval += level;
  return retval;
}
static long long blk_id(const struct Blk *b) {
  return encode(b->level, b->Z, b->ix, b->iy, b->iz);
}
static long long fld_cap;
static long long blk_alloc(int level, long long Z) {
  if (sim.nblk == fld_cap) {
    fld_cap = fld_cap ? 2 * fld_cap : 64;
    sim.blk = (struct Blk *)realloc(sim.blk, fld_cap * sizeof *sim.blk);
    Real *nf = (Real *)aligned_alloc(64, fld_cap * BLK_S * sizeof(Real));
    if (sim.fld) {
      memcpy(nf, sim.fld, sim.nblk * BLK_S * sizeof(Real));
      free(sim.fld);
    }
    sim.fld = nf;
  }
  const long long i = sim.nblk++;
  blk_fill(&sim.blk[i], level, Z);
  struct Node *nd = node(level, Z);
  nd->pos = sim.rank;
  nd->local = i;
  return i;
}
static void blk_remove(long long i) {
  struct Node *nd = node(sim.blk[i].level, sim.blk[i].Z);
  if (nd->local == i)
    nd->local = -1;
  const long long last = sim.nblk - 1;
  if (i != last) {
    sim.blk[i] = sim.blk[last];
    memcpy(BLK(i), BLK(last), BLK_S * sizeof(Real));
    node(sim.blk[i].level, sim.blk[i].Z)->local = i;
  }
  sim.nblk--;
}
static int blk_cmp(const void *a, const void *b) {
  const long long ia = *(const long long *)a, ib = *(const long long *)b;
  return (ia > ib) - (ia < ib);
}
static void blk_sort(void) {
  long long *keys = (long long *)malloc(2 * sim.nblk * sizeof *keys);
  for (long long i = 0; i < sim.nblk; i++) {
    keys[2 * i] = blk_id(&sim.blk[i]);
    keys[2 * i + 1] = i;
  }
  qsort(keys, sim.nblk, 2 * sizeof *keys, blk_cmp);
  struct Blk *nb = (struct Blk *)malloc(fld_cap * sizeof *nb);
  Real *nf = (Real *)aligned_alloc(64, fld_cap * BLK_S * sizeof(Real));
  for (long long i = 0; i < sim.nblk; i++) {
    const long long j = keys[2 * i + 1];
    nb[i] = sim.blk[j];
    memcpy(nf + i * BLK_S, BLK(j), BLK_S * sizeof(Real));
    node(nb[i].level, nb[i].Z)->local = i;
  }
  free(sim.blk);
  free(sim.fld);
  sim.blk = nb;
  sim.fld = nf;
  free(keys);
}
static int imax(int a, int b) { return a > b ? a : b; }
#define HALO_BASE (1LL << 40)
static struct {
  int nhalo, nsend, f0, nc;
  int *scnt, *sdsp, *rcnt, *rdsp;
  long long *send, *rkey;
  Real *buf;
} halo;
static long long blk_avail(int level, long long Z) {
  const struct Node *nd = node_get(level, Z);
  if (nd == NULL)
    return -1;
  if (nd->pos == sim.rank)
    return nd->local;
  if (nd->halo >= 0)
    return HALO_BASE + nd->halo;
  return -1;
}
static void tree_sync(void) {
  int *cnt = (int *)malloc(sim.size * sizeof *cnt);
  int *dsp = (int *)malloc(sim.size * sizeof *dsp);
  int n = 2 * (int)sim.nblk;
  MPI_Allgather(&n, 1, MPI_INT, cnt, 1, MPI_INT, sim.comm);
  int total = 0;
  for (int r = 0; r < sim.size; r++) {
    dsp[r] = total;
    total += cnt[r];
  }
  long long *mine = (long long *)malloc((n > 0 ? n : 1) * sizeof *mine);
  for (long long i = 0; i < sim.nblk; i++) {
    mine[2 * i] = sim.blk[i].level;
    mine[2 * i + 1] = sim.blk[i].Z;
  }
  long long *all = (long long *)malloc((total > 0 ? total : 1) * sizeof *all);
  MPI_Allgatherv(mine, n, MPI_LONG_LONG, all, cnt, dsp, MPI_LONG_LONG, sim.comm);
  nodes_reset();
  for (int r = 0; r < sim.size; r++)
    for (int j = dsp[r]; j < dsp[r] + cnt[r]; j += 2) {
      const int level = (int)all[j];
      const long long Z = all[j + 1];
      struct Blk b;
      blk_fill(&b, level, Z);
      node(level, Z)->pos = r;
      if (level < sim.levelMax - 1)
        for (int k = 0; k < 2; k++)
          for (int jj = 0; jj < 2; jj++)
            for (int i = 0; i < 2; i++)
              node(level + 1, zchild(&b, i, jj, k))->pos = -2;
      if (level > 0)
        node(level - 1, zparent(&b))->pos = -1;
    }
  for (long long i = 0; i < sim.nblk; i++)
    node(sim.blk[i].level, sim.blk[i].Z)->local = i;
  free(cnt);
  free(dsp);
  free(mine);
  free(all);
}
static int blk_remote_neighbors(const struct Blk *b, long long *keys, int *ranks) {
  const int aux = 1 << b->level;
  const int NX = sim.bpdx * aux, NY = sim.bpdy * aux, NZ = sim.bpdz * aux;
  const int xskin = b->ix == 0 || b->ix == NX - 1;
  const int yskin = b->iy == 0 || b->iy == NY - 1;
  const int zskin = b->iz == 0 || b->iz == NZ - 1;
  const int xskip = b->ix == 0 ? -1 : 1;
  const int yskip = b->iy == 0 ? -1 : 1;
  const int zskip = b->iz == 0 ? -1 : 1;
  int n = 0;
  for (int icode = 0; icode < 27; icode++) {
    if (icode == 1 * 1 + 3 * 1 + 9 * 1)
      continue;
    const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
    if (code[0] == xskip && xskin)
      continue;
    if (code[1] == yskip && yskin)
      continue;
    if (code[2] == zskip && zskin)
      continue;
    const long long zn = znei(b, code[0], code[1], code[2]);
    const struct Node *nd = node(b->level, zn);
    if (nd->pos >= 0) {
      if (nd->pos != sim.rank) {
        keys[n] = node_key(b->level, zn);
        ranks[n++] = nd->pos;
      }
    } else if (nd->pos == -2) {
      const int idx[3] = {(b->ix + code[0] + NX) % NX, (b->iy + code[1] + NY) % NY,
                          (b->iz + code[2] + NZ) % NZ};
      const long long zp = zforward(b->level - 1, idx[0] / 2, idx[1] / 2, idx[2] / 2);
      const struct Node *np = node(b->level - 1, zp);
      if (np->pos != sim.rank) {
        keys[n] = node_key(b->level - 1, zp);
        ranks[n++] = np->pos;
      }
    } else if (nd->pos == -1) {
      const int tmp = abs(code[0]) + abs(code[1]) + abs(code[2]);
      int Bstep = 1;
      if (tmp == 2)
        Bstep = 3;
      else if (tmp == 3)
        Bstep = 4;
      for (int B = 0; B <= 3; B += Bstep) {
        const int a = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
        const int ci = 2 * b->ix + imax(code[0], 0) + code[0] + (B % 2) * imax(0, 1 - abs(code[0]));
        const int cj = 2 * b->iy + imax(code[1], 0) + code[1] + a * imax(0, 1 - abs(code[1]));
        const int ck = 2 * b->iz + imax(code[2], 0) + code[2] + (B / 2) * imax(0, 1 - abs(code[2]));
        const long long zf = zforward(b->level + 1, ci, cj, ck);
        const struct Node *nf = node(b->level + 1, zf);
        if (nf->pos >= 0 && nf->pos != sim.rank) {
          keys[n] = node_key(b->level + 1, zf);
          ranks[n++] = nf->pos;
        }
      }
    }
  }
  return n;
}
static int pair_cmp(const void *a, const void *b) {
  const long long *x = (const long long *)a, *y = (const long long *)b;
  if (x[0] != y[0])
    return (x[0] > y[0]) - (x[0] < y[0]);
  return (x[1] > y[1]) - (x[1] < y[1]);
}
static long long pair_unique(long long *p, long long n) {
  long long m = 0;
  for (long long i = 0; i < n; i++)
    if (i == 0 || p[2 * i] != p[2 * (m - 1)] || p[2 * i + 1] != p[2 * (m - 1) + 1]) {
      p[2 * m] = p[2 * i];
      p[2 * m + 1] = p[2 * i + 1];
      m++;
    }
  return m;
}
static void halo_build(void) {
  for (int k = 0; k < halo.nhalo; k++) {
    struct Node *nd = node_find(halo.rkey[k], 0);
    if (nd)
      nd->halo = -1;
  }
  free(halo.scnt);
  free(halo.sdsp);
  free(halo.rcnt);
  free(halo.rdsp);
  free(halo.send);
  free(halo.rkey);
  halo.scnt = (int *)calloc(sim.size, sizeof *halo.scnt);
  halo.sdsp = (int *)calloc(sim.size, sizeof *halo.sdsp);
  halo.rcnt = (int *)calloc(sim.size, sizeof *halo.rcnt);
  halo.rdsp = (int *)calloc(sim.size, sizeof *halo.rdsp);
  const long long cap = 2 * 104 * (sim.nblk > 0 ? sim.nblk : 1);
  long long *sp = (long long *)malloc(cap * sizeof *sp);
  long long *rp = (long long *)malloc(cap * sizeof *rp);
  long long ns = 0, nr = 0;
  long long keys[104];
  int ranks[104];
  for (long long i = 0; i < sim.nblk; i++) {
    const int n = blk_remote_neighbors(&sim.blk[i], keys, ranks);
    const long long mykey = node_key(sim.blk[i].level, sim.blk[i].Z);
    for (int j = 0; j < n; j++) {
      sp[2 * ns] = ranks[j];
      sp[2 * ns + 1] = mykey;
      ns++;
      rp[2 * nr] = ranks[j];
      rp[2 * nr + 1] = keys[j];
      nr++;
    }
  }
  qsort(sp, ns, 2 * sizeof *sp, pair_cmp);
  qsort(rp, nr, 2 * sizeof *rp, pair_cmp);
  ns = pair_unique(sp, ns);
  nr = pair_unique(rp, nr);
  halo.nsend = (int)ns;
  halo.nhalo = (int)nr;
  halo.send = (long long *)malloc((ns > 0 ? ns : 1) * sizeof *halo.send);
  halo.rkey = (long long *)malloc((nr > 0 ? nr : 1) * sizeof *halo.rkey);
  for (long long k = 0; k < ns; k++) {
    halo.scnt[sp[2 * k]]++;
    halo.send[k] = node_find(sp[2 * k + 1], 1)->local;
  }
  for (long long k = 0; k < nr; k++) {
    halo.rcnt[rp[2 * k]]++;
    halo.rkey[k] = rp[2 * k + 1];
    node_find(rp[2 * k + 1], 1)->halo = (int)k;
  }
  for (int r = 1; r < sim.size; r++) {
    halo.sdsp[r] = halo.sdsp[r - 1] + halo.scnt[r - 1];
    halo.rdsp[r] = halo.rdsp[r - 1] + halo.rcnt[r - 1];
  }
  free(sp);
  free(rp);
}
static void halo_sync(int f, int nc) {
  const long long m = (long long)nc * BS3;
  halo.f0 = f;
  halo.nc = nc;
  free(halo.buf);
  halo.buf = (Real *)malloc((halo.nhalo > 0 ? halo.nhalo : 1) * m * sizeof(Real));
  Real *sbuf = (Real *)malloc((halo.nsend > 0 ? halo.nsend : 1) * m * sizeof(Real));
#pragma omp parallel for
  for (int k = 0; k < halo.nsend; k++)
    memcpy(sbuf + k * m, BLK(halo.send[k]) + f * BS3, m * sizeof(Real));
  MPI_Request *req = (MPI_Request *)malloc(2 * sim.size * sizeof *req);
  int nreq = 0;
  for (int r = 0; r < sim.size; r++) {
    if (halo.rcnt[r])
      MPI_Irecv(halo.buf + halo.rdsp[r] * m, halo.rcnt[r] * m, MPI_Real, r, 1, sim.comm, &req[nreq++]);
    if (halo.scnt[r])
      MPI_Isend(sbuf + halo.sdsp[r] * m, halo.scnt[r] * m, MPI_Real, r, 1, sim.comm, &req[nreq++]);
  }
  MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
  free(req);
  free(sbuf);
}
static void states_sync(void) {
  signed char *sb = (signed char *)malloc(halo.nsend > 0 ? halo.nsend : 1);
  signed char *rb = (signed char *)malloc(halo.nhalo > 0 ? halo.nhalo : 1);
  for (int k = 0; k < halo.nsend; k++) {
    const struct Blk *b = &sim.blk[halo.send[k]];
    sb[k] = node(b->level, b->Z)->state;
  }
  MPI_Request *req = (MPI_Request *)malloc(2 * sim.size * sizeof *req);
  int nreq = 0;
  for (int r = 0; r < sim.size; r++) {
    if (halo.rcnt[r])
      MPI_Irecv(rb + halo.rdsp[r], halo.rcnt[r], MPI_SIGNED_CHAR, r, 2, sim.comm, &req[nreq++]);
    if (halo.scnt[r])
      MPI_Isend(sb + halo.sdsp[r], halo.scnt[r], MPI_SIGNED_CHAR, r, 2, sim.comm, &req[nreq++]);
  }
  MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
  for (int k = 0; k < halo.nhalo; k++)
    node_find(halo.rkey[k], 1)->state = rb[k];
  free(req);
  free(sb);
  free(rb);
}
static Real *fld_ptr(long long i, int f, int c) {
  if (i < HALO_BASE)
    return BLK(i) + (f + c) * BS3;
  return halo.buf + ((i - HALO_BASE) * halo.nc + (f - halo.f0) + c) * BS3;
}
static struct {
  int nface;
  int *idx;
  Real *data;
  long long nsend, nrecv;
  long long *send, *recv;
  int *scnt, *sdsp, *rcnt, *rdsp;
} fc;
static Real *fc_face(long long i, int face, int c) {
  const int slot = fc.idx[6 * i + face];
  return slot < 0 ? NULL : fc.data + ((long long)slot * 3 + c) * BS * BS;
}
static int fc_cmp(const void *a, const void *b) {
  const long long *x = (const long long *)a, *y = (const long long *)b;
  for (int q = 0; q < 3; q++)
    if (x[q] != y[q])
      return (x[q] > y[q]) - (x[q] < y[q]);
  return 0;
}
static void fc_prepare(void) {
  static const int fcode[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  free(fc.idx);
  free(fc.data);
  free(fc.send);
  free(fc.recv);
  free(fc.scnt);
  free(fc.sdsp);
  free(fc.rcnt);
  free(fc.rdsp);
  fc.idx = (int *)malloc((6 * sim.nblk > 0 ? 6 * sim.nblk : 1) * sizeof *fc.idx);
  for (long long q = 0; q < 6 * sim.nblk; q++)
    fc.idx[q] = -1;
  fc.nface = 0;
  fc.send = (long long *)malloc((6 * sim.nblk > 0 ? 6 * sim.nblk : 1) * 4 * sizeof *fc.send);
  fc.recv = (long long *)malloc((24 * sim.nblk > 0 ? 24 * sim.nblk : 1) * 6 * sizeof *fc.recv);
  fc.nsend = fc.nrecv = 0;
  for (long long i = 0; i < sim.nblk; i++) {
    const struct Blk *b = &sim.blk[i];
    const int aux = 1 << b->level;
    const int NX = sim.bpdx * aux, NY = sim.bpdy * aux, NZ = sim.bpdz * aux;
    const int xskin = b->ix == 0 || b->ix == NX - 1;
    const int yskin = b->iy == 0 || b->iy == NY - 1;
    const int zskin = b->iz == 0 || b->iz == NZ - 1;
    const int xskip = b->ix == 0 ? -1 : 1;
    const int yskip = b->iy == 0 ? -1 : 1;
    const int zskip = b->iz == 0 ? -1 : 1;
    for (int f = 0; f < 6; f++) {
      const int *code = fcode[f];
      if (code[0] == xskip && xskin)
        continue;
      if (code[1] == yskip && yskin)
        continue;
      if (code[2] == zskip && zskin)
        continue;
      const int d = f / 2;
      const int face = 2 * d + (code[d] > 0);
      const struct Node *nd = node(b->level, znei(b, code[0], code[1], code[2]));
      if (nd->pos >= 0)
        continue;
      fc.idx[6 * i + face] = fc.nface++;
      if (nd->pos == -2) {
        const int idx[3] = {(b->ix + code[0] + NX) % NX, (b->iy + code[1] + NY) % NY,
                            (b->iz + code[2] + NZ) % NZ};
        const long long zp = zforward(b->level - 1, idx[0] / 2, idx[1] / 2, idx[2] / 2);
        long long *e = fc.send + 4 * fc.nsend++;
        e[0] = node(b->level - 1, zp)->pos;
        e[1] = node_key(b->level, b->Z);
        e[2] = face;
        e[3] = i;
      } else if (nd->pos == -1) {
        for (int B = 0; B <= 3; B++) {
          const int a = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
          const int ci = 2 * b->ix + imax(code[0], 0) + code[0] + (B % 2) * imax(0, 1 - abs(code[0]));
          const int cj = 2 * b->iy + imax(code[1], 0) + code[1] + a * imax(0, 1 - abs(code[1]));
          const int ck = 2 * b->iz + imax(code[2], 0) + code[2] + (B / 2) * imax(0, 1 - abs(code[2]));
          const long long zc = zforward(b->level + 1, ci, cj, ck);
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
  fc.scnt = (int *)calloc(sim.size, sizeof *fc.scnt);
  fc.sdsp = (int *)calloc(sim.size, sizeof *fc.sdsp);
  fc.rcnt = (int *)calloc(sim.size, sizeof *fc.rcnt);
  fc.rdsp = (int *)calloc(sim.size, sizeof *fc.rdsp);
  for (long long k = 0; k < fc.nsend; k++)
    fc.scnt[fc.send[4 * k]]++;
  for (long long k = 0; k < fc.nrecv; k++)
    fc.rcnt[fc.recv[6 * k]]++;
  for (int r = 1; r < sim.size; r++) {
    fc.sdsp[r] = fc.sdsp[r - 1] + fc.scnt[r - 1];
    fc.rdsp[r] = fc.rdsp[r - 1] + fc.rcnt[r - 1];
  }
  fc.data = (Real *)calloc((fc.nface > 0 ? fc.nface : 1) * 3 * BS * BS, sizeof(Real));
}
static void grid_init(void) {
  sfc_init(sim.bpdx, sim.bpdy, sim.bpdz, sim.levelMax);
  nodes_init();
  const int level = sim.levelStart;
  const long long aux = 1 << level;
  const long long total = (long long)sim.bpdx * sim.bpdy * sim.bpdz * aux * aux * aux;
  long long my_blocks = total / sim.size;
  if ((long long)sim.rank < total % sim.size)
    my_blocks++;
  long long n_start = sim.rank * (total / sim.size);
  if (total % sim.size > 0) {
    if ((long long)sim.rank < total % sim.size)
      n_start += sim.rank;
    else
      n_start += total % sim.size;
  }
  sim.nblk = 0;
  fld_cap = 0;
  sim.blk = NULL;
  sim.fld = NULL;
  for (long long Z = n_start; Z < n_start + my_blocks; Z++)
    blk_alloc(level, Z);
  memset(sim.fld, 0, sim.nblk * BLK_S * sizeof(Real));
  blk_sort();
  tree_sync();
  halo_build();
  fc_prepare();
}

enum { OP_COPY, OP_AVG8, OP_INTERP, OP_FD, OP_BC };
struct Op {
  int32_t type, bd, dst, bs, src, n, a[8];
};
struct LabTab {
  int32_t magic, ss, te, nops;
  int32_t same_copy[27][2], same_cfill[27][2], fine[27][2], coarse[27][8][2], interp[27][2],
      own_avg[2], bc[6][2][2];
  int32_t relevant[27][64];
  struct Op *ops;
};
static struct LabTab lab_tab[4];
static void lab_tables_init(void) {
  static const int cfg[4][2] = {{1, 1}, {1, 0}, {2, 1}, {3, 0}};
  for (int k = 0; k < 4; k++) {
    struct LabTab *T = &lab_tab[k];
    char name[64];
    snprintf(name, sizeof name, "lab_ss%d_t%d.bin", cfg[k][0], cfg[k][1]);
    FILE *fp = fopen(name, "rb");
    if (fp == NULL) {
      fprintf(stderr, "main.c: cannot open %s (run gen_table.py)\n", name);
      MPI_Abort(sim.comm, 1);
    }
    const size_t hdr = offsetof(struct LabTab, ops);
    if (fread(T, 1, hdr, fp) != hdr || T->magic != 0x4C414231 || T->ss != cfg[k][0] ||
        T->te != cfg[k][1]) {
      fprintf(stderr, "main.c: bad table %s\n", name);
      MPI_Abort(sim.comm, 1);
    }
    T->ops = (struct Op *)malloc(T->nops * sizeof *T->ops);
    if (fread(T->ops, sizeof *T->ops, T->nops, fp) != (size_t)T->nops) {
      fprintf(stderr, "main.c: short read from %s\n", name);
      MPI_Abort(sim.comm, 1);
    }
    fclose(fp);
  }
}
struct Lab {
  int f, nc, vflip;
  int ss[3], se[3];
  int cn[3], cc[3];
  int NX, NY, NZ;
  Real *cache, *coarse;
  const struct LabTab *tab;
};
static const double d_coef_plus[9] = {-0.09375, 0.4375,   0.15625, 0.15625, -0.5625,
                                      0.90625,  -0.09375, 0.4375,  0.15625};
static const double d_coef_minus[9] = {0.15625, -0.5625, 0.90625, -0.09375, 0.4375,
                                       0.15625, 0.15625, 0.4375,  -0.09375};
#define LAB(l, ix, iy, iz) ((l)->cache + (((iz) * (l)->cn[1] + (iy)) * (l)->cn[0] + (ix)) * (l)->nc)
#define CELL(i, f, c, x, y, z) (fld_ptr(i, f, c)[((z) * BS + (y)) * BS + (x)])
static void lab_init(struct Lab *l, int f, int nc, int ss, int te, int vflip) {
  memset(l, 0, sizeof *l);
  l->f = f;
  l->nc = nc;
  l->vflip = vflip;
  for (int k = 0; k < 4; k++)
    if (lab_tab[k].ss == ss && lab_tab[k].te == te)
      l->tab = &lab_tab[k];
  if (l->tab == NULL) {
    fprintf(stderr, "main.c: no table for ss=%d te=%d\n", ss, te);
    MPI_Abort(sim.comm, 1);
  }
  for (int d = 0; d < 3; d++) {
    l->ss[d] = -ss;
    l->se[d] = ss + 1;
    l->cn[d] = BS + 2 * ss;
    const int offset = (l->ss[d] - 1) / 2 - 1;
    const int e = l->se[d] / 2 + 2;
    l->cc[d] = BS / 2 + e - offset - 1;
  }
  l->cache = (Real *)malloc((size_t)l->cn[0] * l->cn[1] * l->cn[2] * nc * sizeof(Real));
  l->coarse = (Real *)malloc((size_t)l->cc[0] * l->cc[1] * l->cc[2] * nc * sizeof(Real));
}
static void lab_free(struct Lab *l) {
  free(l->cache);
  free(l->coarse);
}
static void lab_exec(struct Lab *l, const int32_t sec[2], Real *const *nb) {
  const struct Op *ops = l->tab->ops + sec[0];
  const int nc = l->nc;
  const int cc = l->cc[0];
  Real *buf[2] = {l->cache, l->coarse};
  Real R[8 * F_N];
  for (int k = 0; k < sec[1]; k++) {
    const struct Op *o = &ops[k];
    const int32_t *a = o->a;
    switch (o->type) {
    case OP_COPY: {
      Real *d = buf[o->bd] + (long long)o->dst * nc;
      const Real *s = nb[o->bs - 2] + o->src;
      for (int i = 0; i < o->n; i++)
        for (int c = 0; c < nc; c++)
          d[i * nc + c] = s[c * BS3 + i];
      break;
    }
    case OP_AVG8: {
      Real *d = buf[o->bd] + (long long)o->dst * nc;
      if (o->bs >= 2) {
        const Real *s = nb[o->bs - 2];
        for (int c = 0; c < nc; c++)
          d[c] = 0.125 * (s[c * BS3 + a[0]] + s[c * BS3 + a[1]] + s[c * BS3 + a[2]] + s[c * BS3 + a[3]] +
                          s[c * BS3 + a[4]] + s[c * BS3 + a[5]] + s[c * BS3 + a[6]] + s[c * BS3 + a[7]]);
      } else {
        const Real *s = buf[o->bs];
        for (int c = 0; c < nc; c++)
          d[c] = 0.125 * (s[a[0] * nc + c] + s[a[1] * nc + c] + s[a[2] * nc + c] + s[a[3] * nc + c] +
                          s[a[4] * nc + c] + s[a[5] * nc + c] + s[a[6] * nc + c] + s[a[7] * nc + c]);
      }
      break;
    }
    case OP_INTERP: {
#define C3(I, J, K) (l->coarse[(o->src + ((K) * cc + (J)) * cc + (I)) * nc + c])
      for (int c = 0; c < nc; c++) {
        const Real dudx = 0.125 * (C3(2, 1, 1) - C3(0, 1, 1));
        const Real dudy = 0.125 * (C3(1, 2, 1) - C3(1, 0, 1));
        const Real dudz = 0.125 * (C3(1, 1, 2) - C3(1, 1, 0));
        const Real dudxdy = 0.015625 * (C3(0, 0, 1) + C3(2, 2, 1) - C3(2, 0, 1) - C3(0, 2, 1));
        const Real dudxdz = 0.015625 * (C3(0, 1, 0) + C3(2, 1, 2) - C3(2, 1, 0) - C3(0, 1, 2));
        const Real dudydz = 0.015625 * (C3(1, 0, 0) + C3(1, 2, 2) - C3(1, 2, 0) - C3(1, 0, 2));
        const Real lap = C3(1, 1, 1) + 0.03125 * (C3(0, 1, 1) + C3(2, 1, 1) + C3(1, 0, 1) + C3(1, 2, 1) +
                                                  C3(1, 1, 0) + C3(1, 1, 2) + (-6.0) * C3(1, 1, 1));
        R[0 * nc + c] = lap - dudx - dudy - dudz + dudxdy + dudxdz + dudydz;
        R[1 * nc + c] = lap + dudx - dudy - dudz - dudxdy - dudxdz + dudydz;
        R[2 * nc + c] = lap - dudx + dudy - dudz - dudxdy + dudxdz - dudydz;
        R[3 * nc + c] = lap + dudx + dudy - dudz + dudxdy - dudxdz - dudydz;
        R[4 * nc + c] = lap - dudx - dudy + dudz + dudxdy - dudxdz - dudydz;
        R[5 * nc + c] = lap + dudx - dudy + dudz - dudxdy + dudxdz - dudydz;
        R[6 * nc + c] = lap - dudx + dudy + dudz - dudxdy - dudxdz + dudydz;
        R[7 * nc + c] = lap + dudx + dudy + dudz + dudxdy + dudxdz + dudydz;
      }
#undef C3
      for (int r = 0; r < 8; r++)
        if (a[r] >= 0)
          memcpy(l->cache + (long long)a[r] * nc, R + r * nc, nc * sizeof(Real));
      break;
    }
    case OP_FD: {
      static const int tang[3][2] = {{1, 2}, {0, 2}, {0, 1}};
      const int stride[3] = {1, cc, cc * cc};
      const int t1 = tang[a[0]][0], t2 = tang[a[0]][1];
      const int s1 = stride[t1], s2 = stride[t2];
      const double d1 = 0.25 * (2 * ((a[3] >> t1) & 1) - 1);
      const double d2 = 0.25 * (2 * ((a[3] >> t2) & 1) - 1);
      const double *c1 = d1 > 0 ? d_coef_plus : d_coef_minus;
      const double *c2 = d2 > 0 ? d_coef_plus : d_coef_minus;
      Real *dst = l->cache + (long long)o->dst * nc;
      const Real *bb = l->cache + (long long)a[4] * nc;
      const Real *cq = l->cache + (long long)a[5] * nc;
      for (int c = 0; c < nc; c++) {
#define CO(OFF) (l->coarse[(o->src + (OFF)) * nc + c])
        Real x1D, x2D;
        double mixed_coef = 1.0;
        int P1, M1, P2, M2;
        if (a[1] == 0) {
          x1D = (c1[6] * CO(-s1) + c1[8] * CO(s1)) + c1[7] * CO(0);
          P1 = s1;
          M1 = -s1;
          mixed_coef *= 0.5;
        } else if (a[1] == 1) {
          x1D = (c1[0] * CO(2 * s1) + c1[1] * CO(s1)) + c1[2] * CO(0);
          P1 = s1;
          M1 = 0;
        } else {
          x1D = (c1[3] * CO(-2 * s1) + c1[4] * CO(-s1)) + c1[5] * CO(0);
          P1 = 0;
          M1 = -s1;
        }
        if (a[2] == 0) {
          x2D = (c2[6] * CO(-s2) + c2[8] * CO(s2)) + c2[7] * CO(0);
          P2 = s2;
          M2 = -s2;
          mixed_coef *= 0.5;
        } else if (a[2] == 1) {
          x2D = (c2[0] * CO(2 * s2) + c2[1] * CO(s2)) + c2[2] * CO(0);
          P2 = s2;
          M2 = 0;
        } else {
          x2D = (c2[3] * CO(-2 * s2) + c2[4] * CO(-s2)) + c2[5] * CO(0);
          P2 = 0;
          M2 = -s2;
        }
        const Real mixed = mixed_coef * d1 * d2 * ((CO(M1 + M2) + CO(P1 + P2)) - (CO(P1 + M2) + CO(M1 + P2)));
#undef CO
        Real v = (x1D + x2D) + mixed;
        const int first = a[6] == 1 ? a[7] == 0 : a[7] == 1;
        v = first ? (1.0 / 15.0) * (8.0 * v + (10.0 * bb[c] - 3.0 * cq[c]))
                  : (1.0 / 15.0) * (24.0 * v + (-15.0 * bb[c] + 6 * cq[c]));
        dst[c] = v;
      }
      break;
    }
    case OP_BC: {
      Real *d = buf[o->bd] + (long long)o->dst * nc;
      const Real *s = buf[o->bs] + (long long)o->src * nc;
      memcpy(d, s, nc * sizeof(Real));
      if (l->vflip >= 0)
        d[l->vflip + a[0]] = (-1.) * s[l->vflip + a[0]];
      break;
    }
    }
  }
}
static Real *lab_block(struct Lab *l, int level, long long Z) {
  const long long i = blk_avail(level, Z);
  if (i < 0) {
    fprintf(stderr, "main.c: rank %d: block level %d Z %lld not available\n", sim.rank, level, Z);
    MPI_Abort(sim.comm, 1);
  }
  return fld_ptr(i, l->f, 0);
}
static void lab_load(struct Lab *l, long long ib) {
  const struct Blk *b = &sim.blk[ib];
  const struct LabTab *T = l->tab;
  const int aux = 1 << b->level;
  l->NX = sim.bpdx * aux;
  l->NY = sim.bpdy * aux;
  l->NZ = sim.bpdz * aux;
  for (int c = 0; c < l->nc; c++) {
    const Real *src = fld_ptr(ib, l->f, c);
    for (int iz = 0; iz < BS; iz++)
      for (int iy = 0; iy < BS; iy++)
        for (int ix = 0; ix < BS; ix++)
          LAB(l, ix - l->ss[0], iy - l->ss[1], iz - l->ss[2])[c] = src[(iz * BS + iy) * BS + ix];
  }
  const int xskin = b->ix == 0 || b->ix == l->NX - 1;
  const int yskin = b->iy == 0 || b->iy == l->NY - 1;
  const int zskin = b->iz == 0 || b->iz == l->NZ - 1;
  const int xskip = b->ix == 0 ? -1 : 1;
  const int yskip = b->iy == 0 ? -1 : 1;
  const int zskip = b->iz == 0 ? -1 : 1;
  const int wall[6] = {b->ix == 0, b->ix == l->NX - 1, b->iy == 0, b->iy == l->NY - 1, b->iz == 0,
                       b->iz == l->NZ - 1};
  const int w = (wall[0] | wall[1] << 1) | (wall[2] | wall[3] << 1) << 2 | (wall[4] | wall[5] << 1) << 4;
  const int par = (b->ix & 1) | (b->iy & 1) << 1 | (b->iz & 1) << 2;
  int same[26], coarse[26], nsame = 0, ncoarse = 0;
  unsigned coarse_mask = 0;
  for (int icode = 0; icode < 27; icode++) {
    if (icode == 1 * 1 + 3 * 1 + 9 * 1)
      continue;
    const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, icode / 9 - 1};
    if (code[0] == xskip && xskin)
      continue;
    if (code[1] == yskip && yskin)
      continue;
    if (code[2] == zskip && zskin)
      continue;
    const long long zn = znei(b, code[0], code[1], code[2]);
    const struct Node *nd = node_get(b->level, zn);
    if (nd == NULL)
      continue;
    if (nd->pos >= 0) {
      same[nsame++] = icode;
      Real *nb = lab_block(l, b->level, zn);
      lab_exec(l, T->same_copy[icode], &nb);
    } else if (nd->pos == -2) {
      coarse[ncoarse++] = icode;
      coarse_mask |= 1u << icode;
      const int idx[3] = {(b->ix + code[0] + l->NX) % l->NX, (b->iy + code[1] + l->NY) % l->NY,
                          (b->iz + code[2] + l->NZ) % l->NZ};
      Real *nb = lab_block(l, b->level - 1, zforward(b->level - 1, idx[0] / 2, idx[1] / 2, idx[2] / 2));
      lab_exec(l, T->coarse[icode][par], &nb);
    } else if (nd->pos == -1) {
      Real *nb[4] = {NULL, NULL, NULL, NULL};
      const int tmp = abs(code[0]) + abs(code[1]) + abs(code[2]);
      const int Bstep = tmp == 2 ? 3 : tmp == 3 ? 4 : 1;
      for (int B = 0; B <= 3; B += Bstep) {
        const int a = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
        const int ci = 2 * b->ix + imax(code[0], 0) + code[0] + (B % 2) * imax(0, 1 - abs(code[0]));
        const int cj = 2 * b->iy + imax(code[1], 0) + code[1] + a * imax(0, 1 - abs(code[1]));
        const int ck = 2 * b->iz + imax(code[2], 0) + code[2] + (B / 2) * imax(0, 1 - abs(code[2]));
        nb[B] = lab_block(l, b->level + 1, zforward(b->level + 1, ci, cj, ck));
      }
      lab_exec(l, T->fine[icode], nb);
    }
  }
  int coarsened = 0;
  if (ncoarse > 0)
    for (int k = 0; k < nsame; k++) {
      const int icode = same[k];
      if (T->relevant[icode][w] & coarse_mask) {
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, icode / 9 - 1};
        Real *nb = lab_block(l, b->level, znei(b, code[0], code[1], code[2]));
        lab_exec(l, T->same_cfill[icode], &nb);
        coarsened = 1;
      }
    }
  if (coarsened)
    lab_exec(l, T->own_avg, NULL);
  for (int f = 0; f < 6; f++)
    if (wall[f])
      lab_exec(l, T->bc[f][1], NULL);
  for (int k = 0; k < ncoarse; k++)
    lab_exec(l, T->interp[coarse[k]], NULL);
  for (int f = 0; f < 6; f++)
    if (wall[f])
      lab_exec(l, T->bc[f][0], NULL);
}

static void kernel_gradchi(struct Lab *l, long long ib) {
  const struct Blk *b = &sim.blk[ib];
  Real *TMP0 = BLK(ib) + F_TMP * BS3, *TMP1 = TMP0 + BS3, *TMP2 = TMP1 + BS3;
  int done = 0;
  const int offset = (b->level == sim.levelMax - 1) ? 2 : 1;
  for (int z = -offset; z < BS + offset; ++z)
    for (int y = -offset; y < BS + offset; ++y)
      for (int x = -offset; x < BS + offset; ++x) {
        if (done)
          break;
        Real *v = LAB(l, x - l->ss[0], y - l->ss[1], z - l->ss[2]);
        v[0] = (Real)1.0 < v[0] ? (Real)1.0 : v[0];
        v[0] = v[0] < (Real)0.0 ? (Real)0.0 : v[0];
        if (v[0] > 0.00001 && v[0] < 0.9) {
          const int h = BS / 2;
#define T0(X, Y, Z) TMP0[((Z) * BS + (Y)) * BS + (X)]
          T0(h - 1, h - 1, h - 1) = 1e10;
          T0(h, h - 1, h - 1) = 1e10;
          T0(h - 1, h, h - 1) = 1e10;
          T0(h, h, h - 1) = 1e10;
          T0(h - 1, h - 1, h) = 1e10;
          T0(h, h - 1, h) = 1e10;
          T0(h - 1, h, h) = 1e10;
          T0(h, h, h) = 1e10;
#undef T0
          done = 1;
          break;
        } else if (v[0] > 0.9 && z >= 0 && z < BS && y >= 0 && y < BS && x >= 0 && x < BS) {
          const int j = (z * BS + y) * BS + x;
          TMP0[j] = 0.0;
          TMP1[j] = 0.0;
          TMP2[j] = 0.0;
        }
      }
}
static void compute_gradchi(void) {
#pragma omp parallel
  {
    struct Lab l;
    lab_init(&l, F_CHI, 1, 2, 1, -1);
#pragma omp for schedule(dynamic, 1)
    for (long long i = 0; i < sim.nblk; i++) {
      lab_load(&l, i);
      kernel_gradchi(&l, i);
    }
    lab_free(&l);
  }
}
static int tag_block(long long i) {
  const Real *u0 = BLK(i) + F_TMP * BS3, *u1 = u0 + BS3, *u2 = u1 + BS3;
  double Linf = 0.0;
  for (int j = 0; j < BS3; j++) {
    const double m = fabs(sqrt(u0[j] * u0[j] + u1[j] * u1[j] + u2[j] * u2[j]));
    Linf = m > Linf ? m : Linf;
  }
  if (Linf > sim.Rtol)
    return Refine;
  else if (Linf < sim.Ctol)
    return Compress;
  return Leave;
}
static void set_state(long long i, int st) { node_get(sim.blk[i].level, sim.blk[i].Z)->state = st; }
static int get_state(long long i) { return node_get(sim.blk[i].level, sim.blk[i].Z)->state; }
static int tag_all(void) {
  int changed = 0;
#pragma omp parallel for reduction(| : changed)
  for (long long i = 0; i < sim.nblk; i++) {
    int st = tag_block(i);
    const int level = sim.blk[i].level;
    if ((st == Refine && level == sim.levelMax - 1) || (st == Compress && level == 0))
      st = Leave;
    set_state(i, st);
    if (st != Leave)
      changed = 1;
  }
  return changed;
}
static void valid_states(void) {
  const int levelMin = 0;
  const int levelMax = sim.levelMax;
  for (long long j = 0; j < sim.nblk; j++) {
    const int st = get_state(j);
    if ((st == Refine && sim.blk[j].level == levelMax - 1) || (st == Compress && sim.blk[j].level == levelMin))
      set_state(j, Leave);
  }
  for (int m = levelMax - 1; m >= levelMin; m--) {
    for (long long j = 0; j < sim.nblk; j++) {
      struct Blk *b = &sim.blk[j];
      if (b->level == m && get_state(j) != Refine && b->level != levelMax - 1) {
        const int nx = nblocks_dim(0, m), ny = nblocks_dim(1, m), nz = nblocks_dim(2, m);
        const int xskin = b->ix == 0 || b->ix == nx - 1;
        const int yskin = b->iy == 0 || b->iy == ny - 1;
        const int zskin = b->iz == 0 || b->iz == nz - 1;
        const int xskip = b->ix == 0 ? -1 : 1;
        const int yskip = b->iy == 0 ? -1 : 1;
        const int zskip = b->iz == 0 ? -1 : 1;
        for (int icode = 0; icode < 27; icode++) {
          if (get_state(j) == Refine)
            break;
          if (icode == 1 * 1 + 3 * 1 + 9 * 1)
            continue;
          const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
          if (code[0] == xskip && xskin)
            continue;
          if (code[1] == yskip && yskin)
            continue;
          if (code[2] == zskip && zskin)
            continue;
          if (node(m, znei(b, code[0], code[1], code[2]))->pos == -1) {
            if (get_state(j) == Compress)
              set_state(j, Leave);
            const int tmp = abs(code[0]) + abs(code[1]) + abs(code[2]);
            int Bstep = 1;
            if (tmp == 2)
              Bstep = 3;
            else if (tmp == 3)
              Bstep = 4;
            for (int B = 0; B <= 3; B += Bstep) {
              const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
              const int iNei = 2 * b->ix + imax(code[0], 0) + code[0] + (B % 2) * imax(0, 1 - abs(code[0]));
              const int jNei = 2 * b->iy + imax(code[1], 0) + code[1] + aux * imax(0, 1 - abs(code[1]));
              const int kNei = 2 * b->iz + imax(code[2], 0) + code[2] + (B / 2) * imax(0, 1 - abs(code[2]));
              const long long zzz = zforward(m + 1, iNei, jNei, kNei);
              if (node(m + 1, zzz)->state == Refine) {
                set_state(j, Refine);
                break;
              }
            }
          }
        }
      }
    }
    states_sync();
    if (m == levelMin)
      break;
    for (long long j = 0; j < sim.nblk; j++) {
      struct Blk *b = &sim.blk[j];
      if (b->level == m && get_state(j) == Compress) {
        const int nx = nblocks_dim(0, m), ny = nblocks_dim(1, m), nz = nblocks_dim(2, m);
        const int xskin = b->ix == 0 || b->ix == nx - 1;
        const int yskin = b->iy == 0 || b->iy == ny - 1;
        const int zskin = b->iz == 0 || b->iz == nz - 1;
        const int xskip = b->ix == 0 ? -1 : 1;
        const int yskip = b->iy == 0 ? -1 : 1;
        const int zskip = b->iz == 0 ? -1 : 1;
        for (int icode = 0; icode < 27; icode++) {
          if (icode == 1 * 1 + 3 * 1 + 9 * 1)
            continue;
          const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
          if (code[0] == xskip && xskin)
            continue;
          if (code[1] == yskip && yskin)
            continue;
          if (code[2] == zskip && zskin)
            continue;
          const struct Node *nd = node(m, znei(b, code[0], code[1], code[2]));
          if (nd->pos >= 0 && nd->state == Refine) {
            set_state(j, Leave);
            break;
          }
        }
      }
    }
  }
  for (long long jjj = 0; jjj < sim.nblk; jjj++) {
    struct Blk *b = &sim.blk[jjj];
    const int m = b->level;
    int found = 0;
    for (int i = 2 * (b->ix / 2); i <= 2 * (b->ix / 2) + 1 && !found; i++)
      for (int j = 2 * (b->iy / 2); j <= 2 * (b->iy / 2) + 1 && !found; j++)
        for (int k = 2 * (b->iz / 2); k <= 2 * (b->iz / 2) + 1; k++) {
          const struct Node *nd = node(m, zforward(m, i, j, k));
          if (nd->pos < 0 || nd->state != Compress) {
            found = 1;
            if (get_state(jjj) == Compress)
              set_state(jjj, Leave);
            break;
          }
        }
    if (found)
      for (int i = 2 * (b->ix / 2); i <= 2 * (b->ix / 2) + 1; i++)
        for (int j = 2 * (b->iy / 2); j <= 2 * (b->iy / 2) + 1; j++)
          for (int k = 2 * (b->iz / 2); k <= 2 * (b->iz / 2) + 1; k++) {
            struct Node *nd = node(m, zforward(m, i, j, k));
            if (nd->pos >= 0 && nd->state == Compress)
              nd->state = Leave;
          }
  }
}
static void refine_blocks(struct Lab *l, long long B[8], int f, int nc) {
  const int nx = BS, ny = BS, nz = BS;
  const int offsetX[2] = {0, nx / 2}, offsetY[2] = {0, ny / 2}, offsetZ[2] = {0, nz / 2};
  for (int K = 0; K < 2; K++)
    for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++) {
        const long long ib = B[K * 4 + J * 2 + I];
        for (int k = 0; k < nz; k += 2)
          for (int j = 0; j < ny; j += 2)
            for (int i = 0; i < nx; i += 2) {
              const int x = i / 2 + offsetX[I];
              const int y = j / 2 + offsetY[J];
              const int z = k / 2 + offsetZ[K];
              for (int c = 0; c < nc; c++) {
#define L(X, Y, Z) (LAB(l, (X) - l->ss[0], (Y) - l->ss[1], (Z) - l->ss[2])[c])
                const Real dudx = 0.5 * (L(x + 1, y, z) - L(x - 1, y, z));
                const Real dudy = 0.5 * (L(x, y + 1, z) - L(x, y - 1, z));
                const Real dudz = 0.5 * (L(x, y, z + 1) - L(x, y, z - 1));
                const Real dudx2 = (L(x + 1, y, z) + L(x - 1, y, z)) - 2.0 * L(x, y, z);
                const Real dudy2 = (L(x, y + 1, z) + L(x, y - 1, z)) - 2.0 * L(x, y, z);
                const Real dudz2 = (L(x, y, z + 1) + L(x, y, z - 1)) - 2.0 * L(x, y, z);
                const Real dudxdy = 0.25 * ((L(x + 1, y + 1, z) + L(x - 1, y - 1, z)) - (L(x + 1, y - 1, z) + L(x - 1, y + 1, z)));
                const Real dudxdz = 0.25 * ((L(x + 1, y, z + 1) + L(x - 1, y, z - 1)) - (L(x + 1, y, z - 1) + L(x - 1, y, z + 1)));
                const Real dudydz = 0.25 * ((L(x, y + 1, z + 1) + L(x, y - 1, z - 1)) - (L(x, y + 1, z - 1) + L(x, y - 1, z + 1)));
                const Real u = L(x, y, z);
                const Real lap = 0.03125 * (dudx2 + dudy2 + dudz2);
#undef L
                CELL(ib, f, c, i, j, k) = u + 0.25 * (-(1.0) * dudx - dudy - dudz) + lap + 0.0625 * (dudxdy + dudxdz + dudydz);
                CELL(ib, f, c, i + 1, j, k) = u + 0.25 * (dudx - dudy - dudz) + lap + 0.0625 * (-(1.0) * dudxdy - dudxdz + dudydz);
                CELL(ib, f, c, i, j + 1, k) = u + 0.25 * (-(1.0) * dudx + dudy - dudz) + lap + 0.0625 * (-(1.0) * dudxdy + dudxdz - dudydz);
                CELL(ib, f, c, i + 1, j + 1, k) = u + 0.25 * (dudx + dudy - dudz) + lap + 0.0625 * (dudxdy - dudxdz - dudydz);
                CELL(ib, f, c, i, j, k + 1) = u + 0.25 * (-(1.0) * dudx - dudy + dudz) + lap + 0.0625 * (dudxdy - dudxdz - dudydz);
                CELL(ib, f, c, i + 1, j, k + 1) = u + 0.25 * (dudx - dudy + dudz) + lap + 0.0625 * (-(1.0) * dudxdy + dudxdz - dudydz);
                CELL(ib, f, c, i, j + 1, k + 1) = u + 0.25 * (-(1.0) * dudx + dudy + dudz) + lap + 0.0625 * (-(1.0) * dudxdy - dudxdz + dudydz);
                CELL(ib, f, c, i + 1, j + 1, k + 1) = u + 0.25 * (dudx + dudy + dudz) + lap + 0.0625 * (dudxdy + dudxdz + dudydz);
              }
            }
      }
}
static void blk_pack(Real *dst, long long i) {
  dst[0] = sim.blk[i].level;
  dst[1] = (Real)sim.blk[i].Z;
  memcpy(dst + 2, BLK(i), BLK_S * sizeof(Real));
}
static void blk_unpack(const Real *src) {
  const int level = (int)src[0];
  const long long Z = (long long)src[1];
  const long long i = blk_alloc(level, Z);
  memcpy(BLK(i), src + 2, BLK_S * sizeof(Real));
}
static void blk_remove_key(int level, long long Z) {
  struct Node *nd = node(level, Z);
  if (nd->local >= 0)
    blk_remove(nd->local);
}
enum { PK = BLK_S + 2 };
static void prepare_compression(void) {
  int *scnt = (int *)calloc(sim.size, sizeof *scnt);
  int *rcnt = (int *)calloc(sim.size, sizeof *rcnt);
  int *sdsp = (int *)calloc(sim.size, sizeof *sdsp);
  int *rdsp = (int *)calloc(sim.size, sizeof *rdsp);
  long long *sp = (long long *)malloc(2 * (sim.nblk > 0 ? sim.nblk : 1) * sizeof *sp);
  long long ns = 0;
  for (long long i = 0; i < sim.nblk; i++) {
    const struct Blk *b = &sim.blk[i];
    const long long zb = zforward(b->level, 2 * (b->ix / 2), 2 * (b->iy / 2), 2 * (b->iz / 2));
    const struct Node *base = node(b->level, zb);
    if (base->pos < 0 || base->state != Compress)
      continue;
    const int baserank = base->pos;
    if (b->Z != zb) {
      if (baserank != sim.rank) {
        sp[2 * ns] = baserank;
        sp[2 * ns + 1] = i;
        ns++;
        scnt[baserank]++;
      }
    } else {
      for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
          for (int ii = 0; ii < 2; ii++) {
            const long long n = zforward(b->level, b->ix + ii, b->iy + j, b->iz + k);
            if (n == zb)
              continue;
            struct Node *nd = node(b->level, n);
            if (nd->pos != sim.rank) {
              rcnt[nd->pos]++;
              nd->pos = baserank;
            }
          }
    }
  }
  qsort(sp, ns, 2 * sizeof *sp, pair_cmp);
  int nr = 0;
  for (int r = 0; r < sim.size; r++) {
    if (r > 0) {
      sdsp[r] = sdsp[r - 1] + scnt[r - 1];
      rdsp[r] = rdsp[r - 1] + rcnt[r - 1];
    }
    nr += rcnt[r];
  }
  Real *sbuf = (Real *)malloc((ns > 0 ? ns : 1) * PK * sizeof(Real));
  Real *rbuf = (Real *)malloc((nr > 0 ? nr : 1) * PK * sizeof(Real));
  for (long long k = 0; k < ns; k++)
    blk_pack(sbuf + k * PK, sp[2 * k + 1]);
  MPI_Request *req = (MPI_Request *)malloc(2 * sim.size * sizeof *req);
  int nreq = 0;
  for (int r = 0; r < sim.size; r++)
    if (r != sim.rank) {
      if (rcnt[r])
        MPI_Irecv(rbuf + (long long)rdsp[r] * PK, rcnt[r] * PK, MPI_Real, r, 2468, sim.comm, &req[nreq++]);
      if (scnt[r])
        MPI_Isend(sbuf + (long long)sdsp[r] * PK, scnt[r] * PK, MPI_Real, r, 2468, sim.comm, &req[nreq++]);
    }
  for (long long k = 0; k < ns; k++) {
    const int level = (int)sbuf[k * PK];
    const long long Z = (long long)sbuf[k * PK + 1];
    blk_remove_key(level, Z);
    node(level, Z)->pos = -2;
  }
  MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
  for (int k = 0; k < nr; k++)
    blk_unpack(rbuf + (long long)k * PK);
  free(req);
  free(sbuf);
  free(rbuf);
  free(sp);
  free(scnt);
  free(rcnt);
  free(sdsp);
  free(rdsp);
}
static int balance_global(const long long *all_b) {
  const int size = sim.size, rank = sim.rank;
  blk_sort();
  long long total_load = 0;
  for (int r = 0; r < size; r++)
    total_load += all_b[r];
  long long my_load = total_load / size;
  if (rank < (total_load % size))
    my_load += 1;
  long long *index_start = (long long *)malloc(size * sizeof *index_start);
  index_start[0] = 0;
  for (int r = 1; r < size; r++)
    index_start[r] = index_start[r - 1] + all_b[r - 1];
  long long ideal_index = (total_load / size) * rank;
  ideal_index += (rank < (total_load % size)) ? rank : (total_load % size);
  long long *scnt = (long long *)calloc(size, sizeof *scnt);
  long long *rcnt = (long long *)calloc(size, sizeof *rcnt);
  for (int r = 0; r < size; r++)
    if (rank != r) {
      {
        const long long a1 = ideal_index;
        const long long a2 = ideal_index + my_load - 1;
        const long long b1 = index_start[r];
        const long long b2 = index_start[r] + all_b[r] - 1;
        const long long c1 = a1 > b1 ? a1 : b1;
        const long long c2 = a2 < b2 ? a2 : b2;
        if (c2 - c1 + 1 > 0)
          rcnt[r] = c2 - c1 + 1;
      }
      {
        long long other_ideal_index = (total_load / size) * r;
        other_ideal_index += (r < (total_load % size)) ? r : (total_load % size);
        long long other_load = total_load / size;
        if (r < (total_load % size))
          other_load += 1;
        const long long a1 = other_ideal_index;
        const long long a2 = other_ideal_index + other_load - 1;
        const long long b1 = index_start[rank];
        const long long b2 = index_start[rank] + all_b[rank] - 1;
        const long long c1 = a1 > b1 ? a1 : b1;
        const long long c2 = a2 < b2 ? a2 : b2;
        if (c2 - c1 + 1 > 0)
          scnt[r] = c2 - c1 + 1;
      }
    }
  long long nr = 0, ns = 0;
  long long *rdsp = (long long *)calloc(size, sizeof *rdsp);
  long long *sdsp = (long long *)calloc(size, sizeof *sdsp);
  for (int r = 0; r < size; r++) {
    rdsp[r] = nr;
    nr += rcnt[r];
    sdsp[r] = ns;
    ns += scnt[r];
  }
  Real *rbuf = (Real *)malloc((nr > 0 ? nr : 1) * PK * sizeof(Real));
  Real *sbuf = (Real *)malloc((ns > 0 ? ns : 1) * PK * sizeof(Real));
  MPI_Request *req = (MPI_Request *)malloc(2 * size * sizeof *req);
  int nreq = 0;
  for (int r = 0; r < size; r++)
    if (rcnt[r])
      MPI_Irecv(rbuf + rdsp[r] * PK, rcnt[r] * PK, MPI_Real, r, 12345, sim.comm, &req[nreq++]);
  long long counter_S = 0, counter_E = 0;
  for (int r = 0; r < rank; r++)
    if (scnt[r]) {
      for (long long i = 0; i < scnt[r]; i++)
        blk_pack(sbuf + (sdsp[r] + i) * PK, counter_S + i);
      counter_S += scnt[r];
      MPI_Isend(sbuf + sdsp[r] * PK, scnt[r] * PK, MPI_Real, r, 12345, sim.comm, &req[nreq++]);
    }
  for (int r = size - 1; r > rank; r--)
    if (scnt[r]) {
      for (long long i = 0; i < scnt[r]; i++)
        blk_pack(sbuf + (sdsp[r] + i) * PK, sim.nblk - 1 - (counter_E + i));
      counter_E += scnt[r];
      MPI_Isend(sbuf + sdsp[r] * PK, scnt[r] * PK, MPI_Real, r, 12345, sim.comm, &req[nreq++]);
    }
  for (int r = 0; r < size; r++)
    for (long long i = 0; i < scnt[r]; i++) {
      const Real *p = sbuf + (sdsp[r] + i) * PK;
      const int level = (int)p[0];
      const long long Z = (long long)p[1];
      blk_remove_key(level, Z);
      node(level, Z)->pos = r;
    }
  MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
  for (long long k = 0; k < nr; k++)
    blk_unpack(rbuf + k * PK);
  free(req);
  free(rbuf);
  free(sbuf);
  free(rdsp);
  free(sdsp);
  free(scnt);
  free(rcnt);
  free(index_start);
  return 1;
}
static int balance_diffusion(const long long *dist) {
  const int size = sim.size, rank = sim.rank;
  {
    long long max_b = dist[0], min_b = dist[0];
    for (int r = 0; r < size; r++) {
      max_b = dist[r] > max_b ? dist[r] : max_b;
      min_b = dist[r] < min_b ? dist[r] : min_b;
    }
    const double ratio = (double)max_b / min_b;
    if (ratio > 1.01 || min_b == 0)
      return balance_global(dist);
  }
  const int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
  const int left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
  const int my_blocks = (int)sim.nblk;
  int right_blocks, left_blocks;
  MPI_Request reqs[4];
  MPI_Irecv(&left_blocks, 1, MPI_INT, left, 123, sim.comm, &reqs[0]);
  MPI_Irecv(&right_blocks, 1, MPI_INT, right, 456, sim.comm, &reqs[1]);
  MPI_Isend(&my_blocks, 1, MPI_INT, left, 456, sim.comm, &reqs[2]);
  MPI_Isend(&my_blocks, 1, MPI_INT, right, 123, sim.comm, &reqs[3]);
  MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
  const int nu = 4;
  const int flux_left = (rank == 0) ? 0 : (my_blocks - left_blocks) / nu;
  const int flux_right = (rank == size - 1) ? 0 : (my_blocks - right_blocks) / nu;
  if (flux_right != 0 || flux_left != 0)
    blk_sort();
  Real *sl = NULL, *sr = NULL, *rl = NULL, *rr = NULL;
  MPI_Request req[2];
  int nreq = 0;
  if (flux_left > 0) {
    sl = (Real *)malloc((long long)flux_left * PK * sizeof(Real));
    for (int i = 0; i < flux_left; i++)
      blk_pack(sl + (long long)i * PK, i);
    MPI_Isend(sl, flux_left * PK, MPI_Real, left, 7890, sim.comm, &req[nreq++]);
  } else if (flux_left < 0) {
    rl = (Real *)malloc((long long)(-flux_left) * PK * sizeof(Real));
    MPI_Irecv(rl, -flux_left * PK, MPI_Real, left, 4560, sim.comm, &req[nreq++]);
  }
  if (flux_right > 0) {
    sr = (Real *)malloc((long long)flux_right * PK * sizeof(Real));
    for (int i = 0; i < flux_right; i++)
      blk_pack(sr + (long long)i * PK, my_blocks - i - 1);
    MPI_Isend(sr, flux_right * PK, MPI_Real, right, 4560, sim.comm, &req[nreq++]);
  } else if (flux_right < 0) {
    rr = (Real *)malloc((long long)(-flux_right) * PK * sizeof(Real));
    MPI_Irecv(rr, -flux_right * PK, MPI_Real, right, 7890, sim.comm, &req[nreq++]);
  }
  for (int i = 0; i < flux_right; i++) {
    const Real *p = sr + (long long)i * PK;
    blk_remove_key((int)p[0], (long long)p[1]);
    node((int)p[0], (long long)p[1])->pos = right;
  }
  for (int i = 0; i < flux_left; i++) {
    const Real *p = sl + (long long)i * PK;
    blk_remove_key((int)p[0], (long long)p[1]);
    node((int)p[0], (long long)p[1])->pos = left;
  }
  int moved = nreq != 0;
  if (nreq)
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
  MPI_Allreduce(MPI_IN_PLACE, &moved, 1, MPI_INT, MPI_SUM, sim.comm);
  for (int i = 0; i < -flux_left; i++)
    blk_unpack(rl + (long long)i * PK);
  for (int i = 0; i < -flux_right; i++)
    blk_unpack(rr + (long long)i * PK);
  free(sl);
  free(sr);
  free(rl);
  free(rr);
  return moved >= 1;
}
static void compute_vorticity(void);
static void adapt_mesh(void) {
  compute_vorticity();
  halo_sync(F_CHI, 1);
  compute_gradchi();
  int changed = tag_all();
  MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_SUM, sim.comm);
  if (changed)
    valid_states();
  states_sync();
  long long nref = 0, ncom = 0;
  long long *ref = (long long *)malloc((sim.nblk > 0 ? sim.nblk : 1) * sizeof *ref);
  long long *com = (long long *)malloc((sim.nblk > 0 ? sim.nblk : 1) * sizeof *com);
  long long blocks_after = sim.nblk;
  for (long long i = 0; i < sim.nblk; i++) {
    const struct Blk *b = &sim.blk[i];
    const int st = get_state(i);
    if (st == Refine) {
      ref[nref++] = node_key(b->level, b->Z);
      blocks_after += 7;
    } else if (st == Compress && b->ix % 2 == 0 && b->iy % 2 == 0 && b->iz % 2 == 0)
      com[ncom++] = node_key(b->level, b->Z);
    else if (st == Compress)
      blocks_after--;
  }
  int temp[2] = {(int)nref, (int)ncom}, result[2];
  MPI_Allreduce(temp, result, 2, MPI_INT, MPI_SUM, sim.comm);
  long long *dist = (long long *)malloc(sim.size * sizeof *dist);
  MPI_Allgather(&blocks_after, 1, MPI_LONG_LONG, dist, 1, MPI_LONG_LONG, sim.comm);
  halo_sync(F_PRES, 4);
  struct Lab lab;
  lab_init(&lab, F_PRES, 4, 1, 1, 1);
  for (long long r = 0; r < nref; r++) {
    struct Node *pn = node_find(ref[r], 1);
    const long long ip = pn->local;
    const struct Blk parent = sim.blk[ip];
    pn->state = Leave;
    lab_load(&lab, ip);
    long long B[8];
    for (int k = 0; k < 2; k++)
      for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
          const long long nc = zchild(&parent, i, j, k);
          const long long ic = blk_alloc(parent.level + 1, nc);
          struct Node *cn = node(parent.level + 1, nc);
          cn->state = Leave;
          cn->pos = -2;
          B[k * 4 + j * 2 + i] = ic;
        }
    for (int q = 0; q < 8; q++)
      memset(BLK(B[q]), 0, BLK_S * sizeof(Real));
    refine_blocks(&lab, B, F_PRES, 4);
  }
  lab_free(&lab);
  for (long long r = 0; r < nref; r++) {
    struct Node *pn = node_find(ref[r], 1);
    const struct Blk parent = sim.blk[pn->local];
    pn->pos = -1;
    pn->state = Leave;
    for (int k = 0; k < 2; k++)
      for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
          const long long nc = zchild(&parent, i, j, k);
          struct Node *cn = node(parent.level + 1, nc);
          cn->pos = sim.rank;
          if (parent.level + 2 < sim.levelMax) {
            const struct Blk cb = sim.blk[cn->local];
            for (int i0 = 0; i0 < 2; i0++)
              for (int i1 = 0; i1 < 2; i1++)
                for (int i2 = 0; i2 < 2; i2++)
                  node(parent.level + 2, zchild(&cb, i0, i1, i2))->pos = -2;
          }
        }
  }
  for (long long r = nref - 1; r >= 0; r--) {
    struct Node *nd = node_find(ref[r], 1);
    if (nd->local >= 0)
      blk_remove(nd->local);
  }
  prepare_compression();
  long long *dead = (long long *)malloc((7 * ncom > 0 ? 7 * ncom : 1) * sizeof *dead);
  long long ndead = 0;
  Real *tmp = (Real *)malloc(BLK_S * sizeof(Real));
  for (long long r = 0; r < ncom; r++) {
    struct Node *nd = node_find(com[r], 1);
    const struct Blk info = sim.blk[nd->local];
    const int level = info.level;
    long long B[8];
    for (int K = 0; K < 2; K++)
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++)
          B[K * 4 + J * 2 + I] = node(level, zforward(level, info.ix + I, info.iy + J, info.iz + K))->local;
    const int offs[2] = {0, BS / 2};
    for (int K = 0; K < 2; K++)
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          const long long ib = B[K * 4 + J * 2 + I];
          for (int f = 0; f < F_N; f++)
            for (int k = 0; k < BS; k += 2)
              for (int j = 0; j < BS; j += 2)
                for (int i = 0; i < BS; i += 2)
                  tmp[f * BS3 + ((k / 2 + offs[K]) * BS + (j / 2 + offs[J])) * BS + (i / 2 + offs[I])] =
                      0.125 * ((CELL(ib, f, 0, i, j, k) + CELL(ib, f, 0, i + 1, j + 1, k + 1)) +
                               (CELL(ib, f, 0, i + 1, j, k) + CELL(ib, f, 0, i, j + 1, k + 1)) +
                               (CELL(ib, f, 0, i, j + 1, k) + CELL(ib, f, 0, i + 1, j, k + 1)) +
                               (CELL(ib, f, 0, i + 1, j + 1, k) + CELL(ib, f, 0, i, j, k + 1)));
        }
    const long long np = zforward(level - 1, info.ix / 2, info.iy / 2, info.iz / 2);
    struct Node *pn = node(level - 1, np);
    pn->pos = sim.rank;
    pn->state = Leave;
    if (level - 2 >= 0) {
      struct Blk pb;
      blk_fill(&pb, level - 1, np);
      node(level - 2, zparent(&pb))->pos = -1;
    }
    const long long ib0 = B[0];
    blk_fill(&sim.blk[ib0], level - 1, np);
    memcpy(BLK(ib0), tmp, BLK_S * sizeof(Real));
    pn->local = ib0;
    for (int K = 0; K < 2; K++)
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          const long long n = zforward(level, info.ix + I, info.iy + J, info.iz + K);
          struct Node *cn = node(level, n);
          if (I + J + K != 0)
            dead[ndead++] = node_key(level, n);
          else
            cn->local = -1;
          cn->pos = -2;
          cn->state = Leave;
        }
  }
  free(tmp);
  for (long long d = 0; d < ndead; d++) {
    struct Node *nd = node_find(dead[d], 1);
    if (nd->local >= 0)
      blk_remove(nd->local);
  }
  const int moved = balance_diffusion(dist);
  sim.MeshChanged = result[0] > 0 || result[1] > 0 || moved;
  free(ref);
  free(com);
  free(dead);
  free(dist);
  blk_sort();
  tree_sync();
  halo_build();
  fc_prepare();
}
static void zero_fields(void) {
#pragma omp parallel for
  for (long long i = 0; i < sim.nblk; i++) {
    memset(BLK(i) + F_PRES * BS3, 0, BS3 * sizeof(Real));
    memset(BLK(i) + F_VEL * BS3, 0, 3 * BS3 * sizeof(Real));
    memset(BLK(i) + F_TMP * BS3, 0, 3 * BS3 * sizeof(Real));
    memset(BLK(i) + F_LHS * BS3, 0, BS3 * sizeof(Real));
  }
}
static void init_fields(void) {
  create_obstacles(0);
  create_obstacles(0);
  zero_fields();
  const int lmax = sim.StaticObstacles ? sim.levelMax : 3 * sim.levelMax;
  for (int l = 0; l < lmax; l++) {
    adapt_mesh();
    create_obstacles(0);
    zero_fields();
  }
}

static void fc_fill(int f, int nc) {
  const int Q = 16 * nc;
  Real *sbuf = (Real *)malloc((fc.nsend > 0 ? fc.nsend : 1) * Q * sizeof(Real));
  Real *rbuf = (Real *)malloc((fc.nrecv > 0 ? fc.nrecv : 1) * Q * sizeof(Real));
#pragma omp parallel for
  for (long long k = 0; k < fc.nsend; k++) {
    const long long *e = fc.send + 4 * k;
    for (int c = 0; c < nc; c++) {
      const Real *F = fc_face(e[3], (int)e[2], c);
      for (int i1 = 0; i1 < BS; i1 += 2)
        for (int i2 = 0; i2 < BS; i2 += 2)
          sbuf[k * Q + c * 16 + (i1 / 2) * 4 + i2 / 2] =
              ((F[i2 + i1 * BS] + F[i2 + 1 + i1 * BS]) +
               (F[i2 + (i1 + 1) * BS] + F[i2 + 1 + (i1 + 1) * BS]));
    }
  }
  MPI_Request *req = (MPI_Request *)malloc(2 * sim.size * sizeof *req);
  int nreq = 0;
  for (int r = 0; r < sim.size; r++) {
    if (r == sim.rank) {
      memcpy(rbuf + (long long)fc.rdsp[r] * Q, sbuf + (long long)fc.sdsp[r] * Q,
             (size_t)fc.rcnt[r] * Q * sizeof(Real));
      continue;
    }
    if (fc.rcnt[r])
      MPI_Irecv(rbuf + (long long)fc.rdsp[r] * Q, fc.rcnt[r] * Q, MPI_Real, r, 3, sim.comm, &req[nreq++]);
    if (fc.scnt[r])
      MPI_Isend(sbuf + (long long)fc.sdsp[r] * Q, fc.scnt[r] * Q, MPI_Real, r, 3, sim.comm, &req[nreq++]);
  }
  MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
#pragma omp parallel for
  for (long long k = 0; k < fc.nrecv; k++) {
    const long long *e = fc.recv + 6 * k;
    const int B = (int)e[5];
    const int base = B == 1 ? BS / 2 : B == 2 ? (BS / 2) * BS : B == 3 ? BS / 2 + (BS / 2) * BS : 0;
    for (int c = 0; c < nc; c++) {
      Real *F = fc_face(e[3], (int)e[4], c);
      for (int i1 = 0; i1 < BS; i1 += 2)
        for (int i2 = 0; i2 < BS; i2 += 2)
          F[base + i2 / 2 + (i1 / 2) * BS] += rbuf[k * Q + c * 16 + (i1 / 2) * 4 + i2 / 2];
    }
  }
  for (int d = 0; d < 3; d++)
    for (long long k = 0; k < fc.nrecv; k++) {
      const long long *e = fc.recv + 6 * k;
      const int face = (int)e[4];
      if (face / 2 != d)
        continue;
      const long long i = e[3];
      const int j = (face % 2 == 0) ? 0 : BS - 1;
      for (int c = 0; c < nc; c++) {
        Real *F = fc_face(i, face, c);
        for (int i1 = 0; i1 < BS; i1++)
          for (int i2 = 0; i2 < BS; i2++) {
            if (d == 0)
              CELL(i, f, c, j, i2, i1) += F[i2 + i1 * BS];
            else if (d == 1)
              CELL(i, f, c, i2, j, i1) += F[i2 + i1 * BS];
            else
              CELL(i, f, c, i2, i1, j) += F[i2 + i1 * BS];
            F[i2 + i1 * BS] = 0;
          }
      }
    }
  memset(fc.data, 0, (size_t)(fc.nface > 0 ? fc.nface : 1) * 3 * BS * BS * sizeof(Real));
  free(req);
  free(sbuf);
  free(rbuf);
}
#define L(X, Y, Z, C) (LAB(l, (X) - l->ss[0], (Y) - l->ss[1], (Z) - l->ss[2])[C])
#define L2(X, Y, Z, C) (LAB(l2, (X) - l2->ss[0], (Y) - l2->ss[1], (Z) - l2->ss[2])[C])
#define IDX(X, Y, Z) (((Z) * BS + (Y)) * BS + (X))

static void kernel_lhs(struct Lab *l, long long i) {
  const Real h = sim.blk[i].h;
  Real *o = BLK(i) + F_LHS * BS3;
  Real *F;
  for (int z = 0; z < BS; ++z)
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x)
        o[IDX(x, y, z)] = h * (L(x - 1, y, z, 0) + L(x + 1, y, z, 0) + L(x, y - 1, z, 0) +
                               L(x, y + 1, z, 0) + L(x, y, z - 1, 0) + L(x, y, z + 1, 0) -
                               6.0 * L(x, y, z, 0));
  if ((F = fc_face(i, 0, 0))) {
    const int x = 0;
    for (int z = 0; z < BS; ++z)
      for (int y = 0; y < BS; ++y)
        F[y + BS * z] = h * (L(x, y, z, 0) - L(x - 1, y, z, 0));
  }
  if ((F = fc_face(i, 1, 0))) {
    const int x = BS - 1;
    for (int z = 0; z < BS; ++z)
      for (int y = 0; y < BS; ++y)
        F[y + BS * z] = h * (L(x, y, z, 0) - L(x + 1, y, z, 0));
  }
  if ((F = fc_face(i, 2, 0))) {
    const int y = 0;
    for (int z = 0; z < BS; ++z)
      for (int x = 0; x < BS; ++x)
        F[x + BS * z] = h * (L(x, y, z, 0) - L(x, y - 1, z, 0));
  }
  if ((F = fc_face(i, 3, 0))) {
    const int y = BS - 1;
    for (int z = 0; z < BS; ++z)
      for (int x = 0; x < BS; ++x)
        F[x + BS * z] = h * (L(x, y, z, 0) - L(x, y + 1, z, 0));
  }
  if ((F = fc_face(i, 4, 0))) {
    const int z = 0;
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x)
        F[x + BS * y] = h * (L(x, y, z, 0) - L(x, y, z - 1, 0));
  }
  if ((F = fc_face(i, 5, 0))) {
    const int z = BS - 1;
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x)
        F[x + BS * y] = h * (L(x, y, z, 0) - L(x, y, z + 1, 0));
  }
}
static void compute_lhs(void) {
  Real avgP = 0;
  long long index = -1;
  if (sim.bMeanConstraint <= 2 && sim.bMeanConstraint > 0) {
    for (long long i = 0; i < sim.nblk; ++i)
      if (sim.blk[i].ix == 0 && sim.blk[i].iy == 0 && sim.blk[i].iz == 0)
        index = i;
#pragma omp parallel for reduction(+ : avgP)
    for (long long i = 0; i < sim.nblk; ++i) {
      const struct Blk *b = &sim.blk[i];
      const Real *Z = BLK(i) + F_PRES * BS3;
      const Real h3 = b->h * b->h * b->h;
      for (int j = 0; j < BS3; j++)
        avgP += Z[j] * h3;
    }
    MPI_Allreduce(MPI_IN_PLACE, &avgP, 1, MPI_Real, MPI_SUM, sim.comm);
  }
  halo_sync(F_PRES, 1);
#pragma omp parallel
  {
    struct Lab l;
    lab_init(&l, F_PRES, 1, 1, 0, -1);
#pragma omp for schedule(dynamic, 1)
    for (long long i = 0; i < sim.nblk; i++) {
      lab_load(&l, i);
      kernel_lhs(&l, i);
    }
    lab_free(&l);
  }
  fc_fill(F_LHS, 1);
  if (sim.bMeanConstraint == 0)
    return;
  if (sim.bMeanConstraint <= 2 && sim.bMeanConstraint > 0) {
    if (sim.bMeanConstraint == 1 && index != -1) {
      BLK(index)[F_LHS * BS3] = avgP;
    } else if (sim.bMeanConstraint == 2) {
#pragma omp parallel for
      for (long long i = 0; i < sim.nblk; ++i) {
        Real *LHS = BLK(i) + F_LHS * BS3;
        const Real h3 = sim.blk[i].h * sim.blk[i].h * sim.blk[i].h;
        for (int j = 0; j < BS3; j++)
          LHS[j] += avgP * h3;
      }
    }
  } else {
    for (long long i = 0; i < sim.nblk; ++i) {
      const struct Blk *b = &sim.blk[i];
      if (b->ix == 0 && b->iy == 0 && b->iz == 0)
        BLK(i)[F_LHS * BS3] = BLK(i)[F_PRES * BS3];
    }
  }
}
enum { XPAD = 4 };
static Real getz_inner(Real p[BS + 2][BS + 2][BS + 2 * XPAD], Real Ax[BS3], Real r[BS3],
                       Real *block, const Real sqrNorm0, const Real rr) {
  const Real kDivEpsilon = 1e-55;
  const Real kNormRelCriterion = 1e-7;
  const Real kNormAbsCriterion = 1e-16;
  const Real kSqrNormRelCriterion = kNormRelCriterion * kNormRelCriterion;
  const Real kSqrNormAbsCriterion = kNormAbsCriterion * kNormAbsCriterion;
  Real a2Partial[BS] = {0};
  for (int iz = 0; iz < BS; ++iz)
    for (int iy = 0; iy < BS; ++iy) {
      Real tmpAx[BS];
      for (int ix = 0; ix < BS; ++ix)
        tmpAx[ix] = p[iz + 1][iy + 1][ix + XPAD - 1] + p[iz + 1][iy + 1][ix + XPAD + 1] -
                    6 * p[iz + 1][iy + 1][ix + XPAD];
      for (int ix = 0; ix < BS; ++ix)
        tmpAx[ix] += p[iz + 1][iy][ix + XPAD];
      for (int ix = 0; ix < BS; ++ix)
        tmpAx[ix] += p[iz + 1][iy + 2][ix + XPAD];
      for (int ix = 0; ix < BS; ++ix)
        tmpAx[ix] += p[iz][iy + 1][ix + XPAD];
      for (int ix = 0; ix < BS; ++ix)
        tmpAx[ix] += p[iz + 2][iy + 1][ix + XPAD];
      for (int ix = 0; ix < BS; ++ix)
        Ax[IDX(ix, iy, iz)] = tmpAx[ix];
      for (int ix = 0; ix < BS; ++ix)
        a2Partial[ix] += p[iz + 1][iy + 1][ix + XPAD] * tmpAx[ix];
    }
  Real a2 = 0;
  for (int ix = 0; ix < BS; ++ix)
    a2 += a2Partial[ix];
  const Real a = rr / (a2 + kDivEpsilon);
  for (int iz = 0; iz < BS; ++iz)
    for (int iy = 0; iy < BS; ++iy)
      for (int ix = 0; ix < BS; ++ix)
        block[IDX(ix, iy, iz)] += a * p[iz + 1][iy + 1][ix + XPAD];
  Real s[16] = {0};
  for (int jy = 0; jy < BS3 / 16; ++jy) {
    for (int jx = 0; jx < 16; ++jx)
      r[jy * 16 + jx] -= a * Ax[jy * 16 + jx];
    for (int jx = 0; jx < 16; ++jx)
      s[jx] += r[jy * 16 + jx] * r[jy * 16 + jx];
  }
  Real sqrSum = 0;
  for (int jx = 0; jx < 16; ++jx)
    sqrSum += s[jx];
  const Real beta = sqrSum / (rr + kDivEpsilon);
  const Real sqrNorm = (Real)1 / (BS3 * BS3) * sqrSum;
  if (sqrNorm < kSqrNormRelCriterion * sqrNorm0 || sqrNorm < kSqrNormAbsCriterion)
    return -1.0;
  for (int iz = 0; iz < BS; ++iz)
    for (int iy = 0; iy < BS; ++iy)
      for (int ix = 0; ix < BS; ++ix)
        p[iz + 1][iy + 1][ix + XPAD] = r[IDX(ix, iy, iz)] + beta * p[iz + 1][iy + 1][ix + XPAD];
  return sqrSum;
}
static void getz(void) {
#pragma omp parallel
  {
  Real r[BS3], Ax[BS3], p[BS + 2][BS + 2][BS + 2 * XPAD];
  memset(p, 0, sizeof p);
#pragma omp for
  for (long long i = 0; i < sim.nblk; ++i) {
    Real *block = BLK(i) + F_PRES * BS3;
    const Real invh = 1 / sim.blk[i].h;
    Real rrPartial[BS] = {0};
    for (int iz = 0; iz < BS; ++iz)
      for (int iy = 0; iy < BS; ++iy)
        for (int ix = 0; ix < BS; ++ix) {
          r[IDX(ix, iy, iz)] = invh * block[IDX(ix, iy, iz)];
          rrPartial[ix] += r[IDX(ix, iy, iz)] * r[IDX(ix, iy, iz)];
          p[iz + 1][iy + 1][ix + XPAD] = r[IDX(ix, iy, iz)];
          block[IDX(ix, iy, iz)] = 0;
        }
    Real rr = 0;
    for (int ix = 0; ix < BS; ++ix)
      rr += rrPartial[ix];
    const Real sqrNorm0 = (Real)1 / (BS3 * BS3) * rr;
    if (sqrNorm0 < 1e-32)
      continue;
    for (int k = 0; k < 100; ++k) {
      rr = getz_inner(p, Ax, r, block, sqrNorm0, rr);
      if (rr <= 0)
        break;
    }
  }
  }
}
static void field_set(int f, const Real *in) {
#pragma omp parallel for
  for (long long i = 0; i < sim.nblk; i++)
    memcpy(BLK(i) + f * BS3, in + i * BS3, BS3 * sizeof(Real));
}
static void field_get(int f, Real *out) {
#pragma omp parallel for
  for (long long i = 0; i < sim.nblk; i++)
    memcpy(out + i * BS3, BLK(i) + f * BS3, BS3 * sizeof(Real));
}
static void poisson_precond(const Real *in, Real *out) {
  field_set(F_PRES, in);
  getz();
  field_get(F_PRES, out);
}
static void poisson_lhs(const Real *in, Real *out) {
  field_set(F_PRES, in);
  compute_lhs();
  field_get(F_LHS, out);
}
static void poisson_solve(void) {
  static Real *phat, *rhat, *shat, *what, *zhat, *qhat, *s, *w, *z, *t, *v, *q, *r, *y, *x,
      *r0, *b, *x_opt;
  static long long cap;
  const long long N = sim.nblk * BS3;
  const Real eps = 1e-100;
  const Real max_error = sim.PoissonErrorTol;
  const Real max_rel_error = sim.PoissonErrorTolRel;
  const int max_restarts = 100;
  int serious_breakdown = 0;
  int useXopt = 0;
  int restarts = 0;
  Real min_norm = 1e50;
  Real norm_1 = 0.0;
  Real norm_2 = 0.0;
  if (N > cap) {
    Real **all[18] = {&phat, &rhat, &shat, &what, &zhat, &qhat, &s, &w, &z,
                      &t,    &v,    &q,    &r,    &y,    &x,    &r0, &b, &x_opt};
    for (int k = 0; k < 18; k++) {
      free(*all[k]);
      *all[k] = (Real *)calloc(N, sizeof(Real));
    }
    cap = N;
  }
#pragma omp parallel for
  for (long long i = 0; i < sim.nblk; i++) {
    Real *rhs = BLK(i) + F_LHS * BS3;
    const Real *zz = BLK(i) + F_PRES * BS3;
    const struct Blk *bb = &sim.blk[i];
    if (sim.bMeanConstraint == 1 || sim.bMeanConstraint > 2)
      if (bb->ix == 0 && bb->iy == 0 && bb->iz == 0)
        rhs[0] = 0.0;
    for (int j = 0; j < BS3; j++) {
      b[i * BS3 + j] = rhs[j];
      r[i * BS3 + j] = rhs[j];
      x[i * BS3 + j] = zz[j];
    }
  }
  poisson_lhs(x, r0);
#pragma omp parallel for
  for (long long i = 0; i < N; i++) {
    r0[i] = r[i] - r0[i];
    r[i] = r0[i];
  }
  poisson_precond(r0, rhat);
  poisson_lhs(rhat, w);
  poisson_precond(w, what);
  poisson_lhs(what, t);
  Real alpha = 0.0;
  Real norm = 0.0;
  Real beta = 0.0;
  Real omega = 0.0;
  Real r0r_prev;
  {
    Real temp0 = 0.0;
    Real temp1 = 0.0;
#pragma omp parallel for reduction(+ : temp0, temp1, norm)
    for (long long j = 0; j < N; j++) {
      temp0 += r0[j] * r0[j];
      temp1 += r0[j] * w[j];
      norm += r0[j] * r0[j];
    }
    Real temporary[3] = {temp0, temp1, norm};
    MPI_Allreduce(MPI_IN_PLACE, temporary, 3, MPI_Real, MPI_SUM, sim.comm);
    alpha = temporary[0] / (temporary[1] + eps);
    r0r_prev = temporary[0];
    norm = sqrt(temporary[2]);
  }
  const Real init_norm = norm;
  int k;
  for (k = 0; k < 1000; k++) {
    Real qy = 0.0;
    Real yy = 0.0;
    if (k % 50 != 0) {
#pragma omp parallel for reduction(+ : qy, yy)
      for (long long j = 0; j < N; j++) {
        phat[j] = rhat[j] + beta * (phat[j] - omega * shat[j]);
        s[j] = w[j] + beta * (s[j] - omega * z[j]);
        shat[j] = what[j] + beta * (shat[j] - omega * zhat[j]);
        z[j] = t[j] + beta * (z[j] - omega * v[j]);
        q[j] = r[j] - alpha * s[j];
        qhat[j] = rhat[j] - alpha * shat[j];
        y[j] = w[j] - alpha * z[j];
        qy += q[j] * y[j];
        yy += y[j] * y[j];
      }
    } else {
#pragma omp parallel for
      for (long long j = 0; j < N; j++)
        phat[j] = rhat[j] + beta * (phat[j] - omega * shat[j]);
      poisson_lhs(phat, s);
      poisson_precond(s, shat);
      poisson_lhs(shat, z);
#pragma omp parallel for reduction(+ : qy, yy)
      for (long long j = 0; j < N; j++) {
        q[j] = r[j] - alpha * s[j];
        qhat[j] = rhat[j] - alpha * shat[j];
        y[j] = w[j] - alpha * z[j];
        qy += q[j] * y[j];
        yy += y[j] * y[j];
      }
    }
    Real quantities[7];
    quantities[0] = qy;
    quantities[1] = yy;
    MPI_Allreduce(MPI_IN_PLACE, quantities, 2, MPI_Real, MPI_SUM, sim.comm);
    poisson_precond(z, zhat);
    poisson_lhs(zhat, v);
    qy = quantities[0];
    yy = quantities[1];
    omega = qy / (yy + eps);
    Real r0r = 0.0;
    Real r0w = 0.0;
    Real r0s = 0.0;
    Real r0z = 0.0;
    norm = 0.0;
    norm_1 = 0.0;
    norm_2 = 0.0;
    if (k % 50 != 0) {
#pragma omp parallel for reduction(+ : r0r, r0w, r0s, r0z, norm_1, norm_2, norm)
      for (long long j = 0; j < N; j++) {
        x[j] = x[j] + alpha * phat[j] + omega * qhat[j];
        r[j] = q[j] - omega * y[j];
        rhat[j] = qhat[j] - omega * (what[j] - alpha * zhat[j]);
        w[j] = y[j] - omega * (t[j] - alpha * v[j]);
        r0r += r0[j] * r[j];
        r0w += r0[j] * w[j];
        r0s += r0[j] * s[j];
        r0z += r0[j] * z[j];
        norm += r[j] * r[j];
        norm_1 += r[j] * r[j];
        norm_2 += r0[j] * r0[j];
      }
    } else {
#pragma omp parallel for
      for (long long j = 0; j < N; j++)
        x[j] = x[j] + alpha * phat[j] + omega * qhat[j];
      poisson_lhs(x, r);
#pragma omp parallel for
      for (long long j = 0; j < N; j++)
        r[j] = b[j] - r[j];
      poisson_precond(r, rhat);
      poisson_lhs(rhat, w);
#pragma omp parallel for reduction(+ : r0r, r0w, r0s, r0z, norm_1, norm_2, norm)
      for (long long j = 0; j < N; j++) {
        r0r += r0[j] * r[j];
        r0w += r0[j] * w[j];
        r0s += r0[j] * s[j];
        r0z += r0[j] * z[j];
        norm += r[j] * r[j];
        norm_1 += r[j] * r[j];
        norm_2 += r0[j] * r0[j];
      }
    }
    quantities[0] = r0r;
    quantities[1] = r0w;
    quantities[2] = r0s;
    quantities[3] = r0z;
    quantities[4] = norm_1;
    quantities[5] = norm_2;
    quantities[6] = norm;
    MPI_Allreduce(MPI_IN_PLACE, quantities, 7, MPI_Real, MPI_SUM, sim.comm);
    poisson_precond(w, what);
    poisson_lhs(what, t);
    r0r = quantities[0];
    r0w = quantities[1];
    r0s = quantities[2];
    r0z = quantities[3];
    norm_1 = quantities[4];
    norm_2 = quantities[5];
    norm = sqrt(quantities[6]);
    beta = alpha / (omega + eps) * r0r / (r0r_prev + eps);
    alpha = r0r / (r0w + beta * r0s - beta * omega * r0z);
    Real alphat = 1.0 / (omega + eps) + r0w / (r0r + eps) - beta * omega * r0z / (r0r + eps);
    alphat = 1.0 / (alphat + eps);
    if (fabs(alphat) < 10 * fabs(alpha))
      alpha = alphat;
    r0r_prev = r0r;
    serious_breakdown = r0r * r0r < 1e-16 * norm_1 * norm_2;
    if (serious_breakdown && restarts < max_restarts) {
      restarts++;
#pragma omp parallel for
      for (long long i = 0; i < N; i++)
        r0[i] = r[i];
      poisson_precond(r0, rhat);
      poisson_lhs(rhat, w);
      alpha = 0.0;
      Real temp0 = 0.0;
      Real temp1 = 0.0;
#pragma omp parallel for reduction(+ : temp0, temp1)
      for (long long j = 0; j < N; j++) {
        temp0 += r0[j] * r0[j];
        temp1 += r0[j] * w[j];
      }
      Real temporary[2] = {temp0, temp1};
      MPI_Allreduce(MPI_IN_PLACE, temporary, 2, MPI_Real, MPI_SUM, sim.comm);
      poisson_precond(w, what);
      poisson_lhs(what, t);
      alpha = temporary[0] / (temporary[1] + eps);
      r0r_prev = temporary[0];
      beta = 0.0;
      omega = 0.0;
    }
    if (norm < min_norm) {
      useXopt = 1;
      min_norm = norm;
#pragma omp parallel for
      for (long long i = 0; i < N; i++)
        x_opt[i] = x[i];
    }
    if (norm < max_error || norm / (init_norm + eps) < max_rel_error)
      break;
  }
  field_set(F_PRES, useXopt ? x_opt : x);
}

static Real derivative(const Real U, const Real um3, const Real um2, const Real um1,
                       const Real u, const Real up1, const Real up2, const Real up3) {
  if (U > 0)
    return (-2 * um3 + 15 * um2 - 60 * um1 + 20 * u + 30 * up1 - 3 * up2) / 60.;
  else
    return (2 * up3 - 15 * up2 + 60 * up1 - 20 * u - 30 * um1 + 3 * um2) / 60.;
}
static void kernel_advect_diffuse(struct Lab *l, long long i) {
  const Real dt = sim.dt;
  const Real mu = sim.nu;
  const Real coef = 1.0;
  const Real *uInf = sim.uinf;
  const Real h = sim.blk[i].h;
  Real *o = BLK(i) + F_TMP * BS3;
  const Real h3 = h * h * h;
  const Real facA = -dt / h * h3 * coef;
  const Real facD = (mu / h) * (dt / h) * h3 * coef;
  Real *F;
  for (int z = 0; z < BS; ++z)
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x) {
        const Real uAbs[3] = {L(x, y, z, 0) + uInf[0], L(x, y, z, 1) + uInf[1], L(x, y, z, 2) + uInf[2]};
        const Real dudx = derivative(uAbs[0], L(x - 3, y, z, 0), L(x - 2, y, z, 0), L(x - 1, y, z, 0),
                                     L(x, y, z, 0), L(x + 1, y, z, 0), L(x + 2, y, z, 0), L(x + 3, y, z, 0));
        const Real dvdx = derivative(uAbs[0], L(x - 3, y, z, 1), L(x - 2, y, z, 1), L(x - 1, y, z, 1),
                                     L(x, y, z, 1), L(x + 1, y, z, 1), L(x + 2, y, z, 1), L(x + 3, y, z, 1));
        const Real dwdx = derivative(uAbs[0], L(x - 3, y, z, 2), L(x - 2, y, z, 2), L(x - 1, y, z, 2),
                                     L(x, y, z, 2), L(x + 1, y, z, 2), L(x + 2, y, z, 2), L(x + 3, y, z, 2));
        const Real dudy = derivative(uAbs[1], L(x, y - 3, z, 0), L(x, y - 2, z, 0), L(x, y - 1, z, 0),
                                     L(x, y, z, 0), L(x, y + 1, z, 0), L(x, y + 2, z, 0), L(x, y + 3, z, 0));
        const Real dvdy = derivative(uAbs[1], L(x, y - 3, z, 1), L(x, y - 2, z, 1), L(x, y - 1, z, 1),
                                     L(x, y, z, 1), L(x, y + 1, z, 1), L(x, y + 2, z, 1), L(x, y + 3, z, 1));
        const Real dwdy = derivative(uAbs[1], L(x, y - 3, z, 2), L(x, y - 2, z, 2), L(x, y - 1, z, 2),
                                     L(x, y, z, 2), L(x, y + 1, z, 2), L(x, y + 2, z, 2), L(x, y + 3, z, 2));
        const Real dudz = derivative(uAbs[2], L(x, y, z - 3, 0), L(x, y, z - 2, 0), L(x, y, z - 1, 0),
                                     L(x, y, z, 0), L(x, y, z + 1, 0), L(x, y, z + 2, 0), L(x, y, z + 3, 0));
        const Real dvdz = derivative(uAbs[2], L(x, y, z - 3, 1), L(x, y, z - 2, 1), L(x, y, z - 1, 1),
                                     L(x, y, z, 1), L(x, y, z + 1, 1), L(x, y, z + 2, 1), L(x, y, z + 3, 1));
        const Real dwdz = derivative(uAbs[2], L(x, y, z - 3, 2), L(x, y, z - 2, 2), L(x, y, z - 1, 2),
                                     L(x, y, z, 2), L(x, y, z + 1, 2), L(x, y, z + 2, 2), L(x, y, z + 3, 2));
        const Real duD = ((L(x + 1, y, z, 0) + L(x - 1, y, z, 0)) +
                          ((L(x, y + 1, z, 0) + L(x, y - 1, z, 0)) + (L(x, y, z + 1, 0) + L(x, y, z - 1, 0)))) -
                         6 * L(x, y, z, 0);
        const Real dvD = ((L(x, y + 1, z, 1) + L(x, y - 1, z, 1)) +
                          ((L(x, y, z + 1, 1) + L(x, y, z - 1, 1)) + (L(x + 1, y, z, 1) + L(x - 1, y, z, 1)))) -
                         6 * L(x, y, z, 1);
        const Real dwD = ((L(x, y, z + 1, 2) + L(x, y, z - 1, 2)) +
                          ((L(x + 1, y, z, 2) + L(x - 1, y, z, 2)) + (L(x, y + 1, z, 2) + L(x, y - 1, z, 2)))) -
                         6 * L(x, y, z, 2);
        const Real duA = uAbs[0] * dudx + (uAbs[1] * dudy + uAbs[2] * dudz);
        const Real dvA = uAbs[1] * dvdy + (uAbs[2] * dvdz + uAbs[0] * dvdx);
        const Real dwA = uAbs[2] * dwdz + (uAbs[0] * dwdx + uAbs[1] * dwdy);
        o[0 * BS3 + IDX(x, y, z)] += facA * duA + facD * duD;
        o[1 * BS3 + IDX(x, y, z)] += facA * dvA + facD * dvD;
        o[2 * BS3 + IDX(x, y, z)] += facA * dwA + facD * dwD;
      }
  for (int c = 0; c < 3; c++) {
    if ((F = fc_face(i, 0, c))) {
      const int x = 0;
      for (int z = 0; z < BS; ++z)
        for (int y = 0; y < BS; ++y)
          F[y + BS * z] = facD * (L(x, y, z, c) - L(x - 1, y, z, c));
    }
    if ((F = fc_face(i, 1, c))) {
      const int x = BS - 1;
      for (int z = 0; z < BS; ++z)
        for (int y = 0; y < BS; ++y)
          F[y + BS * z] = facD * (L(x, y, z, c) - L(x + 1, y, z, c));
    }
    if ((F = fc_face(i, 2, c))) {
      const int y = 0;
      for (int z = 0; z < BS; ++z)
        for (int x = 0; x < BS; ++x)
          F[x + BS * z] = facD * (L(x, y, z, c) - L(x, y - 1, z, c));
    }
    if ((F = fc_face(i, 3, c))) {
      const int y = BS - 1;
      for (int z = 0; z < BS; ++z)
        for (int x = 0; x < BS; ++x)
          F[x + BS * z] = facD * (L(x, y, z, c) - L(x, y + 1, z, c));
    }
    if ((F = fc_face(i, 4, c))) {
      const int z = 0;
      for (int y = 0; y < BS; ++y)
        for (int x = 0; x < BS; ++x)
          F[x + BS * y] = facD * (L(x, y, z, c) - L(x, y, z - 1, c));
    }
    if ((F = fc_face(i, 5, c))) {
      const int z = BS - 1;
      for (int y = 0; y < BS; ++y)
        for (int x = 0; x < BS; ++x)
          F[x + BS * y] = facD * (L(x, y, z, c) - L(x, y, z + 1, c));
    }
  }
}
static void advection_diffusion(void) {
  const Real alpha[3] = {1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0};
  const Real beta[3] = {-5.0 / 9.0, -153.0 / 128.0, 0.0};
#pragma omp parallel for
  for (long long i = 0; i < sim.nblk; i++)
    memset(BLK(i) + F_TMP * BS3, 0, 3 * BS3 * sizeof(Real));
  for (int RKstep = 0; RKstep < 3; RKstep++) {
    halo_sync(F_VEL, 3);
#pragma omp parallel
    {
      struct Lab l;
      lab_init(&l, F_VEL, 3, 3, 0, 0);
#pragma omp for schedule(dynamic, 1)
      for (long long i = 0; i < sim.nblk; i++) {
        lab_load(&l, i);
        kernel_advect_diffuse(&l, i);
      }
      lab_free(&l);
    }
    fc_fill(F_TMP, 3);
#pragma omp parallel for
    for (long long i = 0; i < sim.nblk; i++) {
      const Real h = sim.blk[i].h;
      const Real ih3 = alpha[RKstep] / (h * h * h);
      Real *tmpV = BLK(i) + F_TMP * BS3;
      Real *V = BLK(i) + F_VEL * BS3;
      for (int j = 0; j < BS3; j++) {
        V[0 * BS3 + j] += tmpV[0 * BS3 + j] * ih3;
        V[1 * BS3 + j] += tmpV[1 * BS3 + j] * ih3;
        V[2 * BS3 + j] += tmpV[2 * BS3 + j] * ih3;
        tmpV[0 * BS3 + j] *= beta[RKstep];
        tmpV[1 * BS3 + j] *= beta[RKstep];
        tmpV[2 * BS3 + j] *= beta[RKstep];
      }
    }
  }
}

static void fluid_momenta_visit(long long i, const struct Fish *f) {
  struct ObstacleBlock *o = f->oblock[i];
  if (o == NULL)
    return;
  const struct Blk *b = &sim.blk[i];
  const Real lambda = sim.lambda, dt = sim.dt;
  const Real *CM = f->centerOfMass;
  const Real *V = BLK(i) + F_VEL * BS3;
  Real *M = o->mom;
  for (int q = 0; q < M_N; q++)
    M[q] = 0;
  const Real lambdt = lambda * dt;
  for (int iz = 0; iz < BS; ++iz)
    for (int iy = 0; iy < BS; ++iy)
      for (int ix = 0; ix < BS; ++ix) {
        if (o->chi[iz][iy][ix] <= 0)
          continue;
        Real p[3];
        blk_pos(b, ix, iy, iz, p);
        const Real dv = b->h * b->h * b->h, X = o->chi[iz][iy][ix];
        const Real u0 = V[0 * BS3 + IDX(ix, iy, iz)];
        const Real u1 = V[1 * BS3 + IDX(ix, iy, iz)];
        const Real u2 = V[2 * BS3 + IDX(ix, iy, iz)];
        p[0] -= CM[0];
        p[1] -= CM[1];
        p[2] -= CM[2];
        M[M_V] += X * dv;
        M[M_J0] += X * dv * (p[1] * p[1] + p[2] * p[2]);
        M[M_J1] += X * dv * (p[0] * p[0] + p[2] * p[2]);
        M[M_J2] += X * dv * (p[0] * p[0] + p[1] * p[1]);
        M[M_J3] -= X * dv * p[0] * p[1];
        M[M_J4] -= X * dv * p[0] * p[2];
        M[M_J5] -= X * dv * p[1] * p[2];
        M[M_FX] += X * dv * u0;
        M[M_FY] += X * dv * u1;
        M[M_FZ] += X * dv * u2;
        M[M_TX] += X * dv * (p[1] * u2 - p[2] * u1);
        M[M_TY] += X * dv * (p[2] * u0 - p[0] * u2);
        M[M_TZ] += X * dv * (p[0] * u1 - p[1] * u0);
        const Real X1 = o->chi[iz][iy][ix] > 0.5 ? 1.0 : 0.0;
        const Real penalFac = dv * lambdt * X1 / (1 + X1 * lambdt);
        M[M_GfX] += penalFac;
        M[M_GpX] += penalFac * p[0];
        M[M_GpY] += penalFac * p[1];
        M[M_GpZ] += penalFac * p[2];
        M[M_Gj0] += penalFac * (p[1] * p[1] + p[2] * p[2]);
        M[M_Gj1] += penalFac * (p[0] * p[0] + p[2] * p[2]);
        M[M_Gj2] += penalFac * (p[0] * p[0] + p[1] * p[1]);
        M[M_Gj3] -= penalFac * p[0] * p[1];
        M[M_Gj4] -= penalFac * p[0] * p[2];
        M[M_Gj5] -= penalFac * p[1] * p[2];
        const Real DiffU[3] = {u0 - o->udef[iz][iy][ix][0], u1 - o->udef[iz][iy][ix][1],
                               u2 - o->udef[iz][iy][ix][2]};
        M[M_GuX] += penalFac * DiffU[0];
        M[M_GuY] += penalFac * DiffU[1];
        M[M_GuZ] += penalFac * DiffU[2];
        M[M_GaX] += penalFac * (p[1] * DiffU[2] - p[2] * DiffU[1]);
        M[M_GaY] += penalFac * (p[2] * DiffU[0] - p[0] * DiffU[2]);
        M[M_GaZ] += penalFac * (p[0] * DiffU[1] - p[1] * DiffU[0]);
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
      for (j = 0; j < N; j++) {
        const double tmp = A[k * N + j];
        A[k * N + j] = A[p * N + j];
        A[p * N + j] = tmp;
      }
      const double tmp = b[k];
      b[k] = b[p];
      b[p] = tmp;
    }
    for (i = k + 1; i < N; i++) {
      const double f = A[i * N + k] / A[k * N + k];
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
static void solve_velocities(struct Fish *f) {
  double A[36];
  const Real *penalCM = f->penalCM, *penalJ = f->penalJ;
  const Real penalM = f->penalM;
  A[0 * 6 + 0] = penalM;
  A[0 * 6 + 1] = 0.0;
  A[0 * 6 + 2] = 0.0;
  A[0 * 6 + 3] = 0.0;
  A[0 * 6 + 4] = +penalCM[2];
  A[0 * 6 + 5] = -penalCM[1];
  A[1 * 6 + 0] = 0.0;
  A[1 * 6 + 1] = penalM;
  A[1 * 6 + 2] = 0.0;
  A[1 * 6 + 3] = -penalCM[2];
  A[1 * 6 + 4] = 0.0;
  A[1 * 6 + 5] = +penalCM[0];
  A[2 * 6 + 0] = 0.0;
  A[2 * 6 + 1] = 0.0;
  A[2 * 6 + 2] = penalM;
  A[2 * 6 + 3] = +penalCM[1];
  A[2 * 6 + 4] = -penalCM[0];
  A[2 * 6 + 5] = 0.0;
  A[3 * 6 + 0] = 0.0;
  A[3 * 6 + 1] = -penalCM[2];
  A[3 * 6 + 2] = +penalCM[1];
  A[3 * 6 + 3] = penalJ[0];
  A[3 * 6 + 4] = penalJ[3];
  A[3 * 6 + 5] = penalJ[4];
  A[4 * 6 + 0] = +penalCM[2];
  A[4 * 6 + 1] = 0.0;
  A[4 * 6 + 2] = -penalCM[0];
  A[4 * 6 + 3] = penalJ[3];
  A[4 * 6 + 4] = penalJ[1];
  A[4 * 6 + 5] = penalJ[5];
  A[5 * 6 + 0] = -penalCM[1];
  A[5 * 6 + 1] = +penalCM[0];
  A[5 * 6 + 2] = 0.0;
  A[5 * 6 + 3] = penalJ[4];
  A[5 * 6 + 4] = penalJ[5];
  A[5 * 6 + 5] = penalJ[2];
  double b[6] = {f->penalLmom[0], f->penalLmom[1], f->penalLmom[2],
                 f->penalAmom[0], f->penalAmom[1], f->penalAmom[2]};
  for (int d = 0; d < 3; d++)
    if (f->bForcedInSimFrame[d]) {
      for (int j = 0; j < 6; j++)
        if (j != d)
          A[d * 6 + j] = 0;
      b[d] = penalM * f->transVel_imposed[d];
    }
  for (int d = 0; d < 3; d++)
    if (f->bBlockRotation[d]) {
      for (int j = 0; j < 6; j++)
        if (j != 3 + d)
          A[(3 + d) * 6 + j] = 0;
      b[3 + d] = 0;
    }
  double x[6] = {f->transVel[0], f->transVel[1], f->transVel[2],
                 f->angVel[0], f->angVel[1], f->angVel[2]};
  if (penalM > 0)
    solve6(A, b, x);
  f->transVel_computed[0] = x[0];
  f->transVel_computed[1] = x[1];
  f->transVel_computed[2] = x[2];
  f->angVel_computed[0] = x[3];
  f->angVel_computed[1] = x[4];
  f->angVel_computed[2] = x[5];
  for (int d = 0; d < 3; d++)
    f->transVel[d] = f->bForcedInSimFrame[d] ? f->transVel_imposed[d] : f->transVel_computed[d];
  for (int d = 0; d < 3; d++)
    f->angVel[d] = f->bBlockRotation[d] ? 0 : f->angVel_computed[d];
  if (f->collision_counter > 0) {
    f->collision_counter -= sim.dt;
    for (int d = 0; d < 3; d++) {
      f->transVel[d] = f->u_collision[d];
      f->angVel[d] = f->o_collision[d];
    }
  }
}
static void compute_velocities(struct Fish *f) {
  solve_velocities(f);
  if (f->bCorrectRoll) {
    const struct Midline *cFish = &f->m;
    const Real *q = f->quaternion;
    Real *o = f->angVel;
    const Real dq[4] = {0.5 * (-o[0] * q[1] - o[1] * q[2] - o[2] * q[3]),
                        0.5 * (+o[0] * q[0] + o[1] * q[3] - o[2] * q[2]),
                        0.5 * (-o[0] * q[3] + o[1] * q[0] + o[2] * q[1]),
                        0.5 * (+o[0] * q[2] - o[1] * q[1] + o[2] * q[0])};
    const Real nom = 2.0 * (q[3] * q[2] + q[0] * q[1]);
    const Real dnom = 2.0 * (dq[3] * q[2] + dq[0] * q[1] + q[3] * dq[2] + q[0] * dq[1]);
    const Real denom = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
    const Real ddenom = -2.0 * (2.0 * q[1] * dq[1] + 2.0 * q[2] * dq[2]);
    const Real arg = nom / denom;
    const Real darg = (dnom * denom - nom * ddenom) / denom / denom;
    const Real a = atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
    const Real da = 1.0 / (1.0 + arg * arg) * darg;
    const int Nm = cFish->Nm;
    const Real d1 = cFish->rX[0] - cFish->rX[Nm - 1];
    const Real d2 = cFish->rY[0] - cFish->rY[Nm - 1];
    const Real d3 = cFish->rZ[0] - cFish->rZ[Nm - 1];
    const Real dn = pow(d1 * d1 + d2 * d2 + d3 * d3, 0.5) + 1e-21;
    f->r_axis = (Real(*)[4])realloc(f->r_axis, (f->nr_axis + 1) * sizeof *f->r_axis);
    f->r_axis[f->nr_axis][0] = -d1 / dn;
    f->r_axis[f->nr_axis][1] = -d2 / dn;
    f->r_axis[f->nr_axis][2] = -d3 / dn;
    f->r_axis[f->nr_axis][3] = sim.dt;
    f->nr_axis++;
    Real roll_axis[3] = {0., 0., 0.};
    Real time_roll = 0.0;
    int elements_to_keep = 0;
    for (int i = f->nr_axis - 1; i >= 0; i--) {
      const Real *r = f->r_axis[i];
      const Real dt = r[3];
      if (time_roll + dt > 5.0)
        break;
      roll_axis[0] += r[0] * dt;
      roll_axis[1] += r[1] * dt;
      roll_axis[2] += r[2] * dt;
      time_roll += dt;
      elements_to_keep++;
    }
    time_roll += 1e-21;
    roll_axis[0] /= time_roll;
    roll_axis[1] /= time_roll;
    roll_axis[2] /= time_roll;
    const int elements_to_delete = f->nr_axis - elements_to_keep;
    if (elements_to_delete > 0) {
      memmove(f->r_axis, f->r_axis + elements_to_delete, elements_to_keep * sizeof *f->r_axis);
      f->nr_axis = elements_to_keep;
    }
    if (sim.time < 1.0 || time_roll < 1.0)
      return;
    const Real omega_roll = o[0] * roll_axis[0] + o[1] * roll_axis[1] + o[2] * roll_axis[2];
    o[0] += -omega_roll * roll_axis[0];
    o[1] += -omega_roll * roll_axis[1];
    o[2] += -omega_roll * roll_axis[2];
    Real correction_magnitude, dummy;
    clip_quantities(0.025, 1e4, sim.dt, 0, a + 0.05 * da, 0.0, &correction_magnitude, &dummy);
    o[0] += -correction_magnitude * roll_axis[0];
    o[1] += -correction_magnitude * roll_axis[1];
    o[2] += -correction_magnitude * roll_axis[2];
  }
}
static void update_obstacles(void) {
  if (sim.nfish == 0)
    return;
#pragma omp parallel for schedule(dynamic, 1)
  for (long long i = 0; i < sim.nblk; ++i)
    for (int k = 0; k < sim.nfish; k++)
      fluid_momenta_visit(i, &sim.fish[k]);
  for (int k = 0; k < sim.nfish; k++) {
    struct Fish *f = &sim.fish[k];
    Real M[M_N] = {0};
    for (long long i = 0; i < sim.nblk; i++) {
      const struct ObstacleBlock *o = f->oblock[i];
      if (o == NULL)
        continue;
      for (int q = 0; q < M_N; q++)
        M[q] += o->mom[q];
    }
    MPI_Allreduce(MPI_IN_PLACE, M, M_N, MPI_Real, MPI_SUM, sim.comm);
    f->penalM = M[M_GfX];
    f->penalCM[0] = M[M_GpX];
    f->penalCM[1] = M[M_GpY];
    f->penalCM[2] = M[M_GpZ];
    for (int q = 0; q < 6; q++)
      f->penalJ[q] = M[M_Gj0 + q];
    f->penalLmom[0] = M[M_GuX];
    f->penalLmom[1] = M[M_GuY];
    f->penalLmom[2] = M[M_GuZ];
    f->penalAmom[0] = M[M_GaX];
    f->penalAmom[1] = M[M_GaY];
    f->penalAmom[2] = M[M_GaZ];
    compute_velocities(f);
  }
}

static void compute_j(const Real *Rc, const Real *R, const Real *N, const Real *I, Real *J) {
  const Real m00 = I[0];
  const Real m01 = I[3];
  const Real m02 = I[4];
  const Real m11 = I[1];
  const Real m12 = I[5];
  const Real m22 = I[2];
  Real a00 = m22 * m11 - m12 * m12;
  Real a01 = m02 * m12 - m22 * m01;
  Real a02 = m01 * m12 - m02 * m11;
  Real a11 = m22 * m00 - m02 * m02;
  Real a12 = m01 * m02 - m00 * m12;
  Real a22 = m00 * m11 - m01 * m01;
  const Real determinant = 1.0 / ((m00 * a00) + (m01 * a01) + (m02 * a02));
  a00 *= determinant;
  a01 *= determinant;
  a02 *= determinant;
  a11 *= determinant;
  a12 *= determinant;
  a22 *= determinant;
  const Real aux_0 = (Rc[1] - R[1]) * N[2] - (Rc[2] - R[2]) * N[1];
  const Real aux_1 = (Rc[2] - R[2]) * N[0] - (Rc[0] - R[0]) * N[2];
  const Real aux_2 = (Rc[0] - R[0]) * N[1] - (Rc[1] - R[1]) * N[0];
  J[0] = a00 * aux_0 + a01 * aux_1 + a02 * aux_2;
  J[1] = a01 * aux_0 + a11 * aux_1 + a12 * aux_2;
  J[2] = a02 * aux_0 + a12 * aux_1 + a22 * aux_2;
}
static void elastic_collision(const Real m1, const Real m2, const Real *I1, const Real *I2,
                              const Real *v1, const Real *v2, const Real *o1, const Real *o2,
                              const Real *C1, const Real *C2, const Real NX, const Real NY,
                              const Real NZ, const Real CX, const Real CY, const Real CZ,
                              const Real *vc1, const Real *vc2, Real *hv1, Real *hv2, Real *ho1,
                              Real *ho2) {
  const Real e = 1.0;
  const Real N[3] = {NX, NY, NZ};
  const Real C[3] = {CX, CY, CZ};
  const Real k1[3] = {N[0] / m1, N[1] / m1, N[2] / m1};
  const Real k2[3] = {-N[0] / m2, -N[1] / m2, -N[2] / m2};
  Real J1[3];
  Real J2[3];
  compute_j(C, C1, N, I1, J1);
  compute_j(C, C2, N, I2, J2);
  const Real nom = (e + 1) * ((vc1[0] - vc2[0]) * N[0] + (vc1[1] - vc2[1]) * N[1] +
                              (vc1[2] - vc2[2]) * N[2]);
  const Real denom = -(1.0 / m1 + 1.0 / m2) +
                     -((J1[1] * (C[2] - C1[2]) - J1[2] * (C[1] - C1[1])) * N[0] +
                       (J1[2] * (C[0] - C1[0]) - J1[0] * (C[2] - C1[2])) * N[1] +
                       (J1[0] * (C[1] - C1[1]) - J1[1] * (C[0] - C1[0])) * N[2]) -
                     ((J2[1] * (C[2] - C2[2]) - J2[2] * (C[1] - C2[1])) * N[0] +
                      (J2[2] * (C[0] - C2[0]) - J2[0] * (C[2] - C2[2])) * N[1] +
                      (J2[0] * (C[1] - C2[1]) - J2[1] * (C[0] - C2[0])) * N[2]);
  const Real impulse = nom / (denom + 1e-21);
  hv1[0] = v1[0] + k1[0] * impulse;
  hv1[1] = v1[1] + k1[1] * impulse;
  hv1[2] = v1[2] + k1[2] * impulse;
  hv2[0] = v2[0] + k2[0] * impulse;
  hv2[1] = v2[1] + k2[1] * impulse;
  hv2[2] = v2[2] + k2[2] * impulse;
  ho1[0] = o1[0] + J1[0] * impulse;
  ho1[1] = o1[1] + J1[1] * impulse;
  ho1[2] = o1[2] + J1[2] * impulse;
  ho2[0] = o2[0] - J2[0] * impulse;
  ho2[1] = o2[1] - J2[1] * impulse;
  ho2[2] = o2[2] - J2[2] * impulse;
}
struct CollisionInfo {
  Real iM, iPosX, iPosY, iPosZ, iMomX, iMomY, iMomZ, ivecX, ivecY, ivecZ;
  Real jM, jPosX, jPosY, jPosZ, jMomX, jMomY, jMomZ, jvecX, jvecY, jvecZ;
};
static void prevent_colliding_obstacles(void) {
  const int N = sim.nfish;
  struct CollisionInfo *collisions = (struct CollisionInfo *)calloc(N, sizeof *collisions);
  for (int i = 0; i < N; ++i) {
    struct CollisionInfo *coll = &collisions[i];
    const struct Fish *fi = &sim.fish[i];
    const Real iU0 = fi->transVel[0], iU1 = fi->transVel[1], iU2 = fi->transVel[2];
    const Real iomega0 = fi->angVel[0], iomega1 = fi->angVel[1], iomega2 = fi->angVel[2];
    const Real iCx = fi->centerOfMass[0], iCy = fi->centerOfMass[1], iCz = fi->centerOfMass[2];
    for (int j = 0; j < N; ++j) {
      if (i == j)
        continue;
      const struct Fish *fj = &sim.fish[j];
      const Real jU0 = fj->transVel[0], jU1 = fj->transVel[1], jU2 = fj->transVel[2];
      const Real jomega0 = fj->angVel[0], jomega1 = fj->angVel[1], jomega2 = fj->angVel[2];
      const Real jCx = fj->centerOfMass[0], jCy = fj->centerOfMass[1], jCz = fj->centerOfMass[2];
      Real imagmax = 0.0;
      Real jmagmax = 0.0;
      for (long long k = 0; k < sim.nblk; ++k) {
        const struct ObstacleBlock *ib = fi->oblock[k], *jb = fj->oblock[k];
        if (ib == NULL || jb == NULL)
          continue;
        for (int z = 0; z < BS; ++z)
          for (int y = 0; y < BS; ++y)
            for (int x = 0; x < BS; ++x) {
              if (ib->chi[z][y][x] <= 0.0 || jb->chi[z][y][x] <= 0.0)
                continue;
              Real p[3];
              blk_pos(&sim.blk[k], x, y, z, p);
              const Real *iU = ib->udef[z][y][x], *jU = jb->udef[z][y][x];
              const Real iMomX = iU0 + iomega1 * (p[2] - iCz) - iomega2 * (p[1] - iCy) + iU[0];
              const Real iMomY = iU1 + iomega2 * (p[0] - iCx) - iomega0 * (p[2] - iCz) + iU[1];
              const Real iMomZ = iU2 + iomega0 * (p[1] - iCy) - iomega1 * (p[0] - iCx) + iU[2];
              const Real jMomX = jU0 + jomega1 * (p[2] - jCz) - jomega2 * (p[1] - jCy) + jU[0];
              const Real jMomY = jU1 + jomega2 * (p[0] - jCx) - jomega0 * (p[2] - jCz) + jU[1];
              const Real jMomZ = jU2 + jomega0 * (p[1] - jCy) - jomega1 * (p[0] - jCx) + jU[2];
              const Real imag = iMomX * iMomX + iMomY * iMomY + iMomZ * iMomZ;
              const Real jmag = jMomX * jMomX + jMomY * jMomY + jMomZ * jMomZ;
              const Real ivecX = ib->sdfLab[z + 1][y + 1][x + 2] - ib->sdfLab[z + 1][y + 1][x];
              const Real ivecY = ib->sdfLab[z + 1][y + 2][x + 1] - ib->sdfLab[z + 1][y][x + 1];
              const Real ivecZ = ib->sdfLab[z + 2][y + 1][x + 1] - ib->sdfLab[z][y + 1][x + 1];
              const Real jvecX = jb->sdfLab[z + 1][y + 1][x + 2] - jb->sdfLab[z + 1][y + 1][x];
              const Real jvecY = jb->sdfLab[z + 1][y + 2][x + 1] - jb->sdfLab[z + 1][y][x + 1];
              const Real jvecZ = jb->sdfLab[z + 2][y + 1][x + 1] - jb->sdfLab[z][y + 1][x + 1];
              const Real normi = 1.0 / (sqrt(ivecX * ivecX + ivecY * ivecY + ivecZ * ivecZ) + 1e-21);
              const Real normj = 1.0 / (sqrt(jvecX * jvecX + jvecY * jvecY + jvecZ * jvecZ) + 1e-21);
              coll->iM += 1;
              coll->iPosX += p[0];
              coll->iPosY += p[1];
              coll->iPosZ += p[2];
              coll->ivecX += ivecX * normi;
              coll->ivecY += ivecY * normi;
              coll->ivecZ += ivecZ * normi;
              if (imag > imagmax) {
                imagmax = imag;
                coll->iMomX = iMomX;
                coll->iMomY = iMomY;
                coll->iMomZ = iMomZ;
              }
              coll->jM += 1;
              coll->jPosX += p[0];
              coll->jPosY += p[1];
              coll->jPosZ += p[2];
              coll->jvecX += jvecX * normj;
              coll->jvecY += jvecY * normj;
              coll->jvecZ += jvecZ * normj;
              if (jmag > jmagmax) {
                jmagmax = jmag;
                coll->jMomX = jMomX;
                coll->jMomY = jMomY;
                coll->jMomZ = jMomZ;
              }
            }
      }
    }
  }
  Real *buffer = (Real *)malloc((20 * N > 0 ? 20 * N : 1) * sizeof(Real));
  Real *buffermax = (Real *)malloc((2 * N > 0 ? 2 * N : 1) * sizeof(Real));
  for (int i = 0; i < N; i++) {
    const struct CollisionInfo *coll = &collisions[i];
    buffermax[2 * i] = coll->iMomX * coll->iMomX + coll->iMomY * coll->iMomY + coll->iMomZ * coll->iMomZ;
    buffermax[2 * i + 1] = coll->jMomX * coll->jMomX + coll->jMomY * coll->jMomY + coll->jMomZ * coll->jMomZ;
  }
  MPI_Allreduce(MPI_IN_PLACE, buffermax, 2 * N, MPI_Real, MPI_MAX, sim.comm);
  for (int i = 0; i < N; i++) {
    const struct CollisionInfo *coll = &collisions[i];
    const Real maxi = coll->iMomX * coll->iMomX + coll->iMomY * coll->iMomY + coll->iMomZ * coll->iMomZ;
    const Real maxj = coll->jMomX * coll->jMomX + coll->jMomY * coll->jMomY + coll->jMomZ * coll->jMomZ;
    const int iok = fabs(maxi - buffermax[2 * i]) < 1e-10;
    const int jok = fabs(maxj - buffermax[2 * i + 1]) < 1e-10;
    buffer[20 * i] = coll->iM;
    buffer[20 * i + 1] = coll->iPosX;
    buffer[20 * i + 2] = coll->iPosY;
    buffer[20 * i + 3] = coll->iPosZ;
    buffer[20 * i + 4] = iok ? coll->iMomX : 0;
    buffer[20 * i + 5] = iok ? coll->iMomY : 0;
    buffer[20 * i + 6] = iok ? coll->iMomZ : 0;
    buffer[20 * i + 7] = coll->ivecX;
    buffer[20 * i + 8] = coll->ivecY;
    buffer[20 * i + 9] = coll->ivecZ;
    buffer[20 * i + 10] = coll->jM;
    buffer[20 * i + 11] = coll->jPosX;
    buffer[20 * i + 12] = coll->jPosY;
    buffer[20 * i + 13] = coll->jPosZ;
    buffer[20 * i + 14] = jok ? coll->jMomX : 0;
    buffer[20 * i + 15] = jok ? coll->jMomY : 0;
    buffer[20 * i + 16] = jok ? coll->jMomZ : 0;
    buffer[20 * i + 17] = coll->jvecX;
    buffer[20 * i + 18] = coll->jvecY;
    buffer[20 * i + 19] = coll->jvecZ;
  }
  MPI_Allreduce(MPI_IN_PLACE, buffer, 20 * N, MPI_Real, MPI_SUM, sim.comm);
  for (int i = 0; i < N; i++) {
    struct CollisionInfo *coll = &collisions[i];
    coll->iM = buffer[20 * i];
    coll->iPosX = buffer[20 * i + 1];
    coll->iPosY = buffer[20 * i + 2];
    coll->iPosZ = buffer[20 * i + 3];
    coll->iMomX = buffer[20 * i + 4];
    coll->iMomY = buffer[20 * i + 5];
    coll->iMomZ = buffer[20 * i + 6];
    coll->ivecX = buffer[20 * i + 7];
    coll->ivecY = buffer[20 * i + 8];
    coll->ivecZ = buffer[20 * i + 9];
    coll->jM = buffer[20 * i + 10];
    coll->jPosX = buffer[20 * i + 11];
    coll->jPosY = buffer[20 * i + 12];
    coll->jPosZ = buffer[20 * i + 13];
    coll->jMomX = buffer[20 * i + 14];
    coll->jMomY = buffer[20 * i + 15];
    coll->jMomZ = buffer[20 * i + 16];
    coll->jvecX = buffer[20 * i + 17];
    coll->jvecY = buffer[20 * i + 18];
    coll->jvecZ = buffer[20 * i + 19];
  }
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j) {
      struct Fish *fi = &sim.fish[i], *fj = &sim.fish[j];
      const Real m1 = fi->mass;
      const Real m2 = fj->mass;
      const Real v1[3] = {fi->transVel[0], fi->transVel[1], fi->transVel[2]};
      const Real o1[3] = {fi->angVel[0], fi->angVel[1], fi->angVel[2]};
      const Real v2[3] = {fj->transVel[0], fj->transVel[1], fj->transVel[2]};
      const Real o2[3] = {fj->angVel[0], fj->angVel[1], fj->angVel[2]};
      const Real I1[6] = {fi->J[0], fi->J[1], fi->J[2], fi->J[3], fi->J[4], fi->J[5]};
      const Real I2[6] = {fj->J[0], fj->J[1], fj->J[2], fj->J[3], fj->J[4], fj->J[5]};
      const Real C1[3] = {fi->centerOfMass[0], fi->centerOfMass[1], fi->centerOfMass[2]};
      const Real C2[3] = {fj->centerOfMass[0], fj->centerOfMass[1], fj->centerOfMass[2]};
      const struct CollisionInfo *coll = &collisions[i];
      const struct CollisionInfo *coll_other = &collisions[j];
      const Real tolerance = 0.001;
      if (coll->iM < tolerance || coll->jM < tolerance)
        continue;
      if (coll_other->iM < tolerance || coll_other->jM < tolerance)
        continue;
      if (fabs(coll->iPosX / coll->iM - coll_other->iPosX / coll_other->iM) > 0.2 ||
          fabs(coll->iPosY / coll->iM - coll_other->iPosY / coll_other->iM) > 0.2 ||
          fabs(coll->iPosZ / coll->iM - coll_other->iPosZ / coll_other->iM) > 0.2)
        continue;
      const Real norm_i = sqrt(coll->ivecX * coll->ivecX + coll->ivecY * coll->ivecY + coll->ivecZ * coll->ivecZ);
      const Real norm_j = sqrt(coll->jvecX * coll->jvecX + coll->jvecY * coll->jvecY + coll->jvecZ * coll->jvecZ);
      const Real mX = coll->ivecX / norm_i - coll->jvecX / norm_j;
      const Real mY = coll->ivecY / norm_i - coll->jvecY / norm_j;
      const Real mZ = coll->ivecZ / norm_i - coll->jvecZ / norm_j;
      const Real inorm = 1.0 / sqrt(mX * mX + mY * mY + mZ * mZ);
      const Real NX = mX * inorm;
      const Real NY = mY * inorm;
      const Real NZ = mZ * inorm;
      const Real projVel = (coll->jMomX - coll->iMomX) * NX + (coll->jMomY - coll->iMomY) * NY +
                           (coll->jMomZ - coll->iMomZ) * NZ;
      if (projVel <= 0)
        continue;
      const Real inv_iM = 1.0 / coll->iM;
      const Real inv_jM = 1.0 / coll->jM;
      const Real iPX = coll->iPosX * inv_iM;
      const Real iPY = coll->iPosY * inv_iM;
      const Real iPZ = coll->iPosZ * inv_iM;
      const Real jPX = coll->jPosX * inv_jM;
      const Real jPY = coll->jPosY * inv_jM;
      const Real jPZ = coll->jPosZ * inv_jM;
      const Real CX = 0.5 * (iPX + jPX);
      const Real CY = 0.5 * (iPY + jPY);
      const Real CZ = 0.5 * (iPZ + jPZ);
      const Real vc1[3] = {coll->iMomX, coll->iMomY, coll->iMomZ};
      const Real vc2[3] = {coll->jMomX, coll->jMomY, coll->jMomZ};
      Real ho1[3], ho2[3], hv1[3], hv2[3];
      const int iforced = fi->bForcedInSimFrame[0] || fi->bForcedInSimFrame[1] || fi->bForcedInSimFrame[2];
      const int jforced = fj->bForcedInSimFrame[0] || fj->bForcedInSimFrame[1] || fj->bForcedInSimFrame[2];
      const Real m1_i = iforced ? 1e10 * m1 : m1;
      const Real m2_j = jforced ? 1e10 * m2 : m2;
      elastic_collision(m1_i, m2_j, I1, I2, v1, v2, o1, o2, C1, C2, NX, NY, NZ, CX, CY, CZ, vc1, vc2,
                        hv1, hv2, ho1, ho2);
      for (int d = 0; d < 3; d++) {
        fi->transVel[d] = hv1[d];
        fj->transVel[d] = hv2[d];
        fi->angVel[d] = ho1[d];
        fj->angVel[d] = ho2[d];
        fi->u_collision[d] = hv1[d];
        fi->o_collision[d] = ho1[d];
        fj->u_collision[d] = hv2[d];
        fj->o_collision[d] = ho2[d];
      }
      fi->collision_counter = 0.01 * sim.dt;
      fj->collision_counter = 0.01 * sim.dt;
    }
  free(buffer);
  free(buffermax);
  free(collisions);
}
static void penalization_visit(long long i, const struct Fish *f) {
  const struct ObstacleBlock *o = f->oblock[i];
  if (o == NULL)
    return;
  const struct Blk *blk = &sim.blk[i];
  const Real dt = sim.dt, lambda = sim.lambda;
  Real *b = BLK(i) + F_VEL * BS3;
  const Real *bChi = BLK(i) + F_CHI * BS3;
  const Real *CM = f->centerOfMass;
  const Real *vel = f->transVel;
  const Real *omega = f->angVel;
  const Real lambdaFac = lambda;
  for (int iz = 0; iz < BS; ++iz)
    for (int iy = 0; iy < BS; ++iy)
      for (int ix = 0; ix < BS; ++ix) {
        if (bChi[IDX(ix, iy, iz)] > o->chi[iz][iy][ix])
          continue;
        if (o->chi[iz][iy][ix] <= 0)
          continue;
        Real p[3];
        blk_pos(blk, ix, iy, iz, p);
        p[0] -= CM[0];
        p[1] -= CM[1];
        p[2] -= CM[2];
        const Real *U = o->udef[iz][iy][ix];
        const Real U_TOT[3] = {vel[0] + omega[1] * p[2] - omega[2] * p[1] + U[0],
                               vel[1] + omega[2] * p[0] - omega[0] * p[2] + U[1],
                               vel[2] + omega[0] * p[1] - omega[1] * p[0] + U[2]};
        const Real X = o->chi[iz][iy][ix] > 0.5 ? 1.0 : 0.0;
        const Real penalFac = X * lambdaFac / (1 + X * lambdaFac * dt);
        const Real FPX = penalFac * (U_TOT[0] - b[0 * BS3 + IDX(ix, iy, iz)]);
        const Real FPY = penalFac * (U_TOT[1] - b[1 * BS3 + IDX(ix, iy, iz)]);
        const Real FPZ = penalFac * (U_TOT[2] - b[2 * BS3 + IDX(ix, iy, iz)]);
        b[0 * BS3 + IDX(ix, iy, iz)] = b[0 * BS3 + IDX(ix, iy, iz)] + dt * FPX;
        b[1 * BS3 + IDX(ix, iy, iz)] = b[1 * BS3 + IDX(ix, iy, iz)] + dt * FPY;
        b[2 * BS3 + IDX(ix, iy, iz)] = b[2 * BS3 + IDX(ix, iy, iz)] + dt * FPZ;
      }
}
static void penalization(void) {
  if (sim.nfish == 0)
    return;
  prevent_colliding_obstacles();
#pragma omp parallel for schedule(dynamic, 1)
  for (long long i = 0; i < sim.nblk; ++i)
    for (int k = 0; k < sim.nfish; k++)
      penalization_visit(i, &sim.fish[k]);
}

static void kernel_pressure_rhs(struct Lab *l, struct Lab *l2, long long i) {
  const Real dt = sim.dt;
  const Real h = sim.blk[i].h, fac = 0.5 * h * h / dt;
  const Real *c = BLK(i) + F_CHI * BS3;
  Real *p = BLK(i) + F_LHS * BS3;
  Real *F;
  for (int z = 0; z < BS; ++z)
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x) {
        p[IDX(x, y, z)] = fac * (L(x + 1, y, z, 0) - L(x - 1, y, z, 0) + L(x, y + 1, z, 1) -
                                 L(x, y - 1, z, 1) + L(x, y, z + 1, 2) - L(x, y, z - 1, 2));
        const Real divUs = L2(x + 1, y, z, 0) - L2(x - 1, y, z, 0) + L2(x, y + 1, z, 1) -
                           L2(x, y - 1, z, 1) + L2(x, y, z + 1, 2) - L2(x, y, z - 1, 2);
        p[IDX(x, y, z)] += -c[IDX(x, y, z)] * fac * divUs;
      }
  if ((F = fc_face(i, 0, 0))) {
    const int x = 0;
    for (int z = 0; z < BS; ++z)
      for (int y = 0; y < BS; ++y)
        F[y + BS * z] = fac * (L(x - 1, y, z, 0) + L(x, y, z, 0)) -
                        c[IDX(x, y, z)] * fac * (L2(x - 1, y, z, 0) + L2(x, y, z, 0));
  }
  if ((F = fc_face(i, 1, 0))) {
    const int x = BS - 1;
    for (int z = 0; z < BS; ++z)
      for (int y = 0; y < BS; ++y)
        F[y + BS * z] = -fac * (L(x + 1, y, z, 0) + L(x, y, z, 0)) +
                        c[IDX(x, y, z)] * fac * (L2(x + 1, y, z, 0) + L2(x, y, z, 0));
  }
  if ((F = fc_face(i, 2, 0))) {
    const int y = 0;
    for (int z = 0; z < BS; ++z)
      for (int x = 0; x < BS; ++x)
        F[x + BS * z] = fac * (L(x, y - 1, z, 1) + L(x, y, z, 1)) -
                        c[IDX(x, y, z)] * fac * (L2(x, y - 1, z, 1) + L2(x, y, z, 1));
  }
  if ((F = fc_face(i, 3, 0))) {
    const int y = BS - 1;
    for (int z = 0; z < BS; ++z)
      for (int x = 0; x < BS; ++x)
        F[x + BS * z] = -fac * (L(x, y + 1, z, 1) + L(x, y, z, 1)) +
                        c[IDX(x, y, z)] * fac * (L2(x, y + 1, z, 1) + L2(x, y, z, 1));
  }
  if ((F = fc_face(i, 4, 0))) {
    const int z = 0;
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x)
        F[x + BS * y] = fac * (L(x, y, z - 1, 2) + L(x, y, z, 2)) -
                        c[IDX(x, y, z)] * fac * (L2(x, y, z - 1, 2) + L2(x, y, z, 2));
  }
  if ((F = fc_face(i, 5, 0))) {
    const int z = BS - 1;
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x)
        F[x + BS * y] = -fac * (L(x, y, z + 1, 2) + L(x, y, z, 2)) +
                        c[IDX(x, y, z)] * fac * (L2(x, y, z + 1, 2) + L2(x, y, z, 2));
  }
}
static void kernel_div_pressure(struct Lab *l, long long i) {
  Real *b = BLK(i) + F_TMP * BS3;
  const Real fac = sim.blk[i].h;
  Real *F;
  for (int z = 0; z < BS; ++z)
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x)
        b[IDX(x, y, z)] = fac * (L(x + 1, y, z, 0) + L(x - 1, y, z, 0) + L(x, y + 1, z, 0) +
                                 L(x, y - 1, z, 0) + L(x, y, z + 1, 0) + L(x, y, z - 1, 0) -
                                 6.0 * L(x, y, z, 0));
  if ((F = fc_face(i, 0, 0))) {
    const int x = 0;
    for (int z = 0; z < BS; ++z)
      for (int y = 0; y < BS; ++y)
        F[y + BS * z] = fac * (L(x, y, z, 0) - L(x - 1, y, z, 0));
  }
  if ((F = fc_face(i, 1, 0))) {
    const int x = BS - 1;
    for (int z = 0; z < BS; ++z)
      for (int y = 0; y < BS; ++y)
        F[y + BS * z] = -fac * (L(x + 1, y, z, 0) - L(x, y, z, 0));
  }
  if ((F = fc_face(i, 2, 0))) {
    const int y = 0;
    for (int z = 0; z < BS; ++z)
      for (int x = 0; x < BS; ++x)
        F[x + BS * z] = fac * (L(x, y, z, 0) - L(x, y - 1, z, 0));
  }
  if ((F = fc_face(i, 3, 0))) {
    const int y = BS - 1;
    for (int z = 0; z < BS; ++z)
      for (int x = 0; x < BS; ++x)
        F[x + BS * z] = -fac * (L(x, y + 1, z, 0) - L(x, y, z, 0));
  }
  if ((F = fc_face(i, 4, 0))) {
    const int z = 0;
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x)
        F[x + BS * y] = fac * (L(x, y, z, 0) - L(x, y, z - 1, 0));
  }
  if ((F = fc_face(i, 5, 0))) {
    const int z = BS - 1;
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x)
        F[x + BS * y] = -fac * (L(x, y, z + 1, 0) - L(x, y, z, 0));
  }
}
static void kernel_gradp(struct Lab *l, long long i) {
  const Real dt = sim.dt;
  const Real h = sim.blk[i].h;
  Real *o = BLK(i) + F_TMP * BS3;
  const Real fac = -0.5 * dt * h * h;
  Real *F;
  for (int z = 0; z < BS; ++z)
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x) {
        o[0 * BS3 + IDX(x, y, z)] = fac * (L(x + 1, y, z, 0) - L(x - 1, y, z, 0));
        o[1 * BS3 + IDX(x, y, z)] = fac * (L(x, y + 1, z, 0) - L(x, y - 1, z, 0));
        o[2 * BS3 + IDX(x, y, z)] = fac * (L(x, y, z + 1, 0) - L(x, y, z - 1, 0));
      }
  if ((F = fc_face(i, 0, 0))) {
    const int x = 0;
    for (int z = 0; z < BS; ++z)
      for (int y = 0; y < BS; ++y)
        F[y + BS * z] = fac * (L(x - 1, y, z, 0) + L(x, y, z, 0));
  }
  if ((F = fc_face(i, 1, 0))) {
    const int x = BS - 1;
    for (int z = 0; z < BS; ++z)
      for (int y = 0; y < BS; ++y)
        F[y + BS * z] = -fac * (L(x + 1, y, z, 0) + L(x, y, z, 0));
  }
  if ((F = fc_face(i, 2, 1))) {
    const int y = 0;
    for (int z = 0; z < BS; ++z)
      for (int x = 0; x < BS; ++x)
        F[x + BS * z] = fac * (L(x, y - 1, z, 0) + L(x, y, z, 0));
  }
  if ((F = fc_face(i, 3, 1))) {
    const int y = BS - 1;
    for (int z = 0; z < BS; ++z)
      for (int x = 0; x < BS; ++x)
        F[x + BS * z] = -fac * (L(x, y + 1, z, 0) + L(x, y, z, 0));
  }
  if ((F = fc_face(i, 4, 2))) {
    const int z = 0;
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x)
        F[x + BS * y] = fac * (L(x, y, z - 1, 0) + L(x, y, z, 0));
  }
  if ((F = fc_face(i, 5, 2))) {
    const int z = BS - 1;
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x)
        F[x + BS * y] = -fac * (L(x, y, z + 1, 0) + L(x, y, z, 0));
  }
}
static void kernel_vorticity(struct Lab *l, long long i) {
  const Real h = sim.blk[i].h;
  const Real inv2h = .5 * h * h;
  Real *o = BLK(i) + F_TMP * BS3;
  for (int z = 0; z < BS; ++z)
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x) {
        o[0 * BS3 + IDX(x, y, z)] =
            inv2h * ((L(x, y + 1, z, 2) - L(x, y - 1, z, 2)) - (L(x, y, z + 1, 1) - L(x, y, z - 1, 1)));
        o[1 * BS3 + IDX(x, y, z)] =
            inv2h * ((L(x, y, z + 1, 0) - L(x, y, z - 1, 0)) - (L(x + 1, y, z, 2) - L(x - 1, y, z, 2)));
        o[2 * BS3 + IDX(x, y, z)] =
            inv2h * ((L(x + 1, y, z, 1) - L(x - 1, y, z, 1)) - (L(x, y + 1, z, 0) - L(x, y - 1, z, 0)));
      }
  if (fc_face(i, 0, 0)) {
    const int x = 0;
    for (int z = 0; z < BS; ++z)
      for (int y = 0; y < BS; ++y) {
        fc_face(i, 0, 1)[y + BS * z] = -inv2h * (L(x - 1, y, z, 2) + L(x, y, z, 2));
        fc_face(i, 0, 2)[y + BS * z] = +inv2h * (L(x - 1, y, z, 1) + L(x, y, z, 1));
      }
  }
  if (fc_face(i, 1, 0)) {
    const int x = BS - 1;
    for (int z = 0; z < BS; ++z)
      for (int y = 0; y < BS; ++y) {
        fc_face(i, 1, 1)[y + BS * z] = +inv2h * (L(x + 1, y, z, 2) + L(x, y, z, 2));
        fc_face(i, 1, 2)[y + BS * z] = -inv2h * (L(x + 1, y, z, 1) + L(x, y, z, 1));
      }
  }
  if (fc_face(i, 2, 0)) {
    const int y = 0;
    for (int z = 0; z < BS; ++z)
      for (int x = 0; x < BS; ++x) {
        fc_face(i, 2, 0)[x + BS * z] = +inv2h * (L(x, y - 1, z, 2) + L(x, y, z, 2));
        fc_face(i, 2, 2)[x + BS * z] = -inv2h * (L(x, y - 1, z, 0) + L(x, y, z, 0));
      }
  }
  if (fc_face(i, 3, 0)) {
    const int y = BS - 1;
    for (int z = 0; z < BS; ++z)
      for (int x = 0; x < BS; ++x) {
        fc_face(i, 3, 0)[x + BS * z] = -inv2h * (L(x, y + 1, z, 2) + L(x, y, z, 2));
        fc_face(i, 3, 2)[x + BS * z] = +inv2h * (L(x, y + 1, z, 0) + L(x, y, z, 0));
      }
  }
  if (fc_face(i, 4, 0)) {
    const int z = 0;
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x) {
        fc_face(i, 4, 0)[x + BS * y] = -inv2h * (L(x, y, z - 1, 1) + L(x, y, z, 1));
        fc_face(i, 4, 1)[x + BS * y] = +inv2h * (L(x, y, z - 1, 0) + L(x, y, z, 0));
      }
  }
  if (fc_face(i, 5, 0)) {
    const int z = BS - 1;
    for (int y = 0; y < BS; ++y)
      for (int x = 0; x < BS; ++x) {
        fc_face(i, 5, 0)[x + BS * y] = +inv2h * (L(x, y, z + 1, 1) + L(x, y, z, 1));
        fc_face(i, 5, 1)[x + BS * y] = -inv2h * (L(x, y, z + 1, 0) + L(x, y, z, 0));
      }
  }
}
static void compute_vorticity(void) {
  halo_sync(F_VEL, 3);
#pragma omp parallel
  {
    struct Lab l;
    lab_init(&l, F_VEL, 3, 1, 0, 0);
#pragma omp for schedule(dynamic, 1)
    for (long long i = 0; i < sim.nblk; i++) {
      lab_load(&l, i);
      kernel_vorticity(&l, i);
    }
    lab_free(&l);
  }
  fc_fill(F_TMP, 3);
#pragma omp parallel for
  for (long long i = 0; i < sim.nblk; i++) {
    const Real h = sim.blk[i].h;
    const Real fac = 1.0 / (h * h * h);
    Real *b = BLK(i) + F_TMP * BS3;
    for (int j = 0; j < 3 * BS3; j++)
      b[j] *= fac;
  }
}
static void update_tmpv(void) {
  for (int k = 0; k < sim.nfish; k++) {
    const struct Fish *f = &sim.fish[k];
#pragma omp parallel for schedule(dynamic, 1)
    for (long long i = 0; i < sim.nblk; ++i) {
      const struct ObstacleBlock *o = f->oblock[i];
      if (o == NULL)
        continue;
      const Real *c = BLK(i) + F_CHI * BS3;
      Real *b = BLK(i) + F_TMP * BS3;
      for (int z = 0; z < BS; ++z)
        for (int y = 0; y < BS; ++y)
          for (int x = 0; x < BS; ++x) {
            if (c[IDX(x, y, z)] > o->chi[z][y][x])
              continue;
            b[0 * BS3 + IDX(x, y, z)] += o->udef[z][y][x][0];
            b[1 * BS3 + IDX(x, y, z)] += o->udef[z][y][x][1];
            b[2 * BS3 + IDX(x, y, z)] += o->udef[z][y][x][2];
          }
    }
  }
}
static void pressure_projection(void) {
  static Real *pOld;
  static long long cap;
  const long long N = sim.nblk * BS3;
  if (N > cap) {
    free(pOld);
    pOld = (Real *)malloc(N * sizeof(Real));
    cap = N;
  }
#pragma omp parallel for
  for (long long i = 0; i < sim.nblk; i++) {
    memcpy(pOld + i * BS3, BLK(i) + F_PRES * BS3, BS3 * sizeof(Real));
    memset(BLK(i) + F_TMP * BS3, 0, 3 * BS3 * sizeof(Real));
  }
  if (sim.nfish > 0)
    update_tmpv();
  halo_sync(F_VEL, 6);
#pragma omp parallel
  {
    struct Lab l, l2;
    lab_init(&l, F_VEL, 3, 1, 0, 0);
    lab_init(&l2, F_TMP, 3, 1, 0, 0);
#pragma omp for schedule(dynamic, 1)
    for (long long i = 0; i < sim.nblk; i++) {
      lab_load(&l, i);
      lab_load(&l2, i);
      kernel_pressure_rhs(&l, &l2, i);
    }
    lab_free(&l);
    lab_free(&l2);
  }
  fc_fill(F_LHS, 1);
  if (sim.step > sim.step_2nd_start) {
    halo_sync(F_PRES, 1);
#pragma omp parallel
    {
      struct Lab l;
      lab_init(&l, F_PRES, 1, 1, 0, -1);
#pragma omp for schedule(dynamic, 1)
      for (long long i = 0; i < sim.nblk; i++) {
        lab_load(&l, i);
        kernel_div_pressure(&l, i);
      }
      lab_free(&l);
    }
    fc_fill(F_TMP, 3);
#pragma omp parallel for
    for (long long i = 0; i < sim.nblk; i++) {
      const Real *b = BLK(i) + F_TMP * BS3;
      Real *LHS = BLK(i) + F_LHS * BS3;
      Real *p = BLK(i) + F_PRES * BS3;
      for (int j = 0; j < BS3; j++) {
        LHS[j] -= b[j];
        p[j] = 0;
      }
    }
  } else {
#pragma omp parallel for
    for (long long i = 0; i < sim.nblk; i++)
      memset(BLK(i) + F_PRES * BS3, 0, BS3 * sizeof(Real));
  }
  poisson_solve();
  Real avg = 0;
  Real avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
  for (long long i = 0; i < sim.nblk; i++) {
    const Real *P = BLK(i) + F_PRES * BS3;
    const Real vv = sim.blk[i].h * sim.blk[i].h * sim.blk[i].h;
    for (int j = 0; j < BS3; j++) {
      avg += P[j] * vv;
      avg1 += vv;
    }
  }
  Real quantities[2] = {avg, avg1};
  MPI_Allreduce(MPI_IN_PLACE, quantities, 2, MPI_Real, MPI_SUM, sim.comm);
  avg = quantities[0];
  avg1 = quantities[1];
  avg = avg / avg1;
#pragma omp parallel for
  for (long long i = 0; i < sim.nblk; i++) {
    Real *P = BLK(i) + F_PRES * BS3;
    for (int j = 0; j < BS3; j++)
      P[j] -= avg;
  }
  if (sim.step > sim.step_2nd_start) {
#pragma omp parallel for
    for (long long i = 0; i < sim.nblk; i++) {
      Real *p = BLK(i) + F_PRES * BS3;
      for (int j = 0; j < BS3; j++)
        p[j] += pOld[i * BS3 + j];
    }
  }
  halo_sync(F_PRES, 1);
#pragma omp parallel
  {
    struct Lab l;
    lab_init(&l, F_PRES, 1, 1, 0, -1);
#pragma omp for schedule(dynamic, 1)
    for (long long i = 0; i < sim.nblk; i++) {
      lab_load(&l, i);
      kernel_gradp(&l, i);
    }
    lab_free(&l);
  }
  fc_fill(F_TMP, 3);
#pragma omp parallel for
  for (long long i = 0; i < sim.nblk; i++) {
    const Real h = sim.blk[i].h;
    const Real fac = 1.0 / (h * h * h);
    const Real *gradP = BLK(i) + F_TMP * BS3;
    Real *v = BLK(i) + F_VEL * BS3;
    for (int j = 0; j < 3 * BS3; j++)
      v[j] += fac * gradP[j];
  }
}

static Real find_max_u(void) {
  Real maxU = 0;
#pragma omp parallel for reduction(max : maxU)
  for (long long i = 0; i < sim.nblk; i++) {
    const Real *b = BLK(i) + F_VEL * BS3;
    for (int j = 0; j < BS3; j++) {
      const Real advu = fabs(b[0 * BS3 + j] + sim.uinf[0]);
      const Real advv = fabs(b[1 * BS3 + j] + sim.uinf[1]);
      const Real advw = fabs(b[2 * BS3 + j] + sim.uinf[2]);
      Real maxUl = advu;
      if (maxUl < advv)
        maxUl = advv;
      if (maxUl < advw)
        maxUl = advw;
      if (maxU < maxUl)
        maxU = maxUl;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &maxU, 1, MPI_Real, MPI_MAX, sim.comm);
  return maxU;
}
static Real calc_max_timestep(void) {
  const Real dt_old = sim.dt;
  sim.dt_old = sim.dt;
  const Real hMin = sim.hmin;
  Real CFL = sim.CFL;
  sim.uMax_measured = find_max_u();
  if (sim.uMax_measured > sim.uMax_allowed) {
    if (sim.rank == 0)
      fprintf(stderr, "maxU = %g exceeded uMax_allowed = %g. Aborting...\n",
              (double)sim.uMax_measured, (double)sim.uMax_allowed);
    MPI_Abort(sim.comm, 1);
  }
  if (CFL > 0) {
    const Real dtDiffusion = (1.0 / 6.0) * hMin * hMin / (sim.nu + (1.0 / 6.0) * hMin * sim.uMax_measured);
    const Real dtAdvection = hMin / (sim.uMax_measured + 1e-8);
    if (sim.step < sim.rampup) {
      const Real x = sim.step / (Real)sim.rampup;
      const Real rampCFL = exp(log(1e-3) * (1 - x) + log(CFL) * x);
      const Real b = rampCFL * dtAdvection;
      sim.dt = b < dtDiffusion ? b : dtDiffusion;
    } else {
      const Real b = CFL * dtAdvection;
      sim.dt = b < dtDiffusion ? b : dtDiffusion;
    }
  } else {
    CFL = (sim.uMax_measured + 1e-8) * sim.dt / hMin;
  }
  if (sim.dt <= 0) {
    fprintf(stderr, "dt <= 0. CFL=%f, hMin=%f, sim.uMax_measured=%f. Aborting...\n", CFL, hMin,
            sim.uMax_measured);
    MPI_Abort(sim.comm, 1);
  }
  if (sim.DLM > 0)
    sim.lambda = sim.DLM / sim.dt;
  if (sim.rank == 0)
    printf("main.c: step: %d, time: %f\n", sim.step, sim.time);
  if (sim.step >= sim.step_2nd_start) {
    const Real a = dt_old;
    const Real b = sim.dt;
    const Real c1 = -(a + b) / (a * b);
    const Real c2 = b / (a + b) / a;
    sim.coefU[0] = -b * (c1 + c2);
    sim.coefU[1] = b * c1;
    sim.coefU[2] = b * c2;
  }
  return sim.dt;
}
static int advance(const Real dt) {
  if (sim.dumpTime > 0 && sim.time >= sim.nextDumpTime) {
    sim.nextDumpTime += sim.dumpTime;
    char path[FILENAME_MAX];
    snprintf(path, sizeof path, "vel.%08d", sim.step);
    fprintf(stderr, "main.c: %s\n", path);
    dump(sim.time, path);
  }
  if (sim.step % 20 == 0 || sim.step < 10)
    adapt_mesh();
  create_obstacles(dt);
  advection_diffusion();
  update_obstacles();
  penalization();
  pressure_projection();
  sim.step++;
  sim.time += dt;
  if ((sim.endTime > 0 && sim.time > sim.endTime) || (sim.nsteps != 0 && sim.step >= sim.nsteps))
    return 1;
  return 0;
}
static void simulate(void) {
  for (;;) {
    const Real dt = calc_max_timestep();
    if (advance(dt))
      break;
  }
}

static void midline_dump(const struct Fish *f, int k, Real t, Real dt) {
  char path[FILENAME_MAX];
  snprintf(path, sizeof path, "cmidline.%d.%d.txt", f->id, k);
  FILE *fp = fopen(path, "w");
  const struct Midline *m = &f->m;
  fprintf(fp, "%.17g %.17g %d\n", t, dt, m->Nm);
  for (int i = 0; i < m->Nm; i++)
    fprintf(fp,
            "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g "
            "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
            m->rS[i], m->rX[i], m->rY[i], m->rZ[i], m->vX[i], m->vY[i], m->vZ[i],
            m->norX[i], m->norY[i], m->norZ[i], m->vNorX[i], m->vNorY[i],
            m->vNorZ[i], m->binX[i], m->binY[i], m->binZ[i], m->vBinX[i],
            m->vBinY[i], m->vBinZ[i], m->width[i], m->height[i]);
  fclose(fp);
}

static void midline_replay(const char *path) {
  FILE *fp = fopen(path, "r");
  if (fp == NULL) {
    fprintf(stderr, "main.c: cannot open %s\n", path);
    abort();
  }
  int k;
  double t, dt;
  while (fscanf(fp, "%d %lf %lf", &k, &t, &dt) == 3) {
    for (int i = 0; i < sim.nfish; i++) {
      struct Fish *f = &sim.fish[i];
      compute_midline(&f->m, t);
      integrate_linear_momentum(&f->m);
      integrate_angular_momentum(&f->m, dt);
      if (sim.rank == 0)
        midline_dump(f, k, t, dt);
    }
  }
  fclose(fp);
}

int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  sim.comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &sim.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &sim.size);
  struct Params parser;
  params_from_argv(&parser, argc, argv);
  parse_arguments(&parser);
  lab_tables_init();
  add_obstacles(&parser);
  const char *replay = param_str(&parser, "midline_replay", "");
  if (replay[0])
    midline_replay(replay);
  if (param_bool(&parser, "chi_test", 0)) {
    sfc_init(sim.bpdx, sim.bpdy, sim.bpdz, sim.levelMax);
    grid_init_uniform();
    create_obstacles(0);
    char path[FILENAME_MAX];
    snprintf(path, sizeof path, "vel.%08d", sim.step);
    dump(sim.time, path);
  } else if (!replay[0]) {
    grid_init();
    init_fields();
    simulate();
  }
  for (int i = 0; i < sim.nfish; i++) {
    fish_clear_blocks(&sim.fish[i]);
    midline_free(&sim.fish[i].m);
  }
  free(sim.fish);
  params_free(&parser);
  MPI_Finalize();
}
