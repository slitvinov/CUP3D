#define OMPI_SKIP_MPICXX 1
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <math.h>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <omp.h>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>
typedef double Real;
static const MPI_Datatype MPI_Real = MPI_DOUBLE;

static struct {
  int rank, size;
  Real h0;
} sim;

template <typename T, int kAlignment> class aligned_allocator {
public:
  typedef T *pointer;
  typedef T const *const_pointer;
  typedef T &reference;
  typedef T const &const_reference;
  typedef T value_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  template <typename U> struct rebind {
    typedef aligned_allocator<U, kAlignment> other;
  };
  aligned_allocator() noexcept {}
  aligned_allocator(aligned_allocator const &a) noexcept {}
  template <typename S>
  aligned_allocator(aligned_allocator<S, kAlignment> const &b) noexcept {}
  pointer allocate(size_type n) {
    pointer p;
    if (posix_memalign(reinterpret_cast<void **>(&p), kAlignment,
                       n * sizeof(T)))
      throw std::bad_alloc();
    return p;
  }
  void deallocate(pointer p, size_type n) noexcept { std::free(p); }
  size_type max_size() const noexcept {
    std::allocator<T> a;
    return a.max_size();
  }
  template <typename C, class... Args> void construct(C *c, Args &&...args) {
    new ((void *)c) C(std::forward<Args>(args)...);
  }
  template <typename C> void destroy(C *c) { c->~C(); }
  bool operator==(aligned_allocator const &a2) const noexcept { return 1; }
  bool operator!=(aligned_allocator const &a2) const noexcept { return 0; }
  template <typename S>
  bool operator==(aligned_allocator<S, kAlignment> const &b) const noexcept {
    return 0;
  }
  template <typename S>
  bool operator!=(aligned_allocator<S, kAlignment> const &b) const noexcept {
    return 1;
  }
};
class SpaceFillingCurve {
protected:
  int BX;
  int BY;
  int BZ;
  int levelMax;
  bool isRegular;
  int base_level;
  std::vector<std::vector<long long>> Zsave;
  std::vector<std::vector<int>> i_inverse;
  std::vector<std::vector<int>> j_inverse;
  std::vector<std::vector<int>> k_inverse;
  long long AxestoTranspose(const int *X_in, int b) const {
    if (b == 0) {
      assert(X_in[0] == 0);
      assert(X_in[1] == 0);
      assert(X_in[2] == 0);
      return 0;
    }
    const int n = 3;
    int X[3] = {X_in[0], X_in[1], X_in[2]};
    assert(b - 1 >= 0);
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
      const long long a1 =
          ((one) << (a + one)) * ((long long)X[1] >> level & one);
      const long long a2 =
          ((one) << (a + two)) * ((long long)X[0] >> level & one);
      retval += a0 + a1 + a2;
      a += 3;
    }
    return retval;
  }
  void TransposetoAxes(long long index, long long *X, int b) const {
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

public:
  SpaceFillingCurve(){};
  SpaceFillingCurve(int a_BX, int a_BY, int a_BZ, int lmax)
      : BX(a_BX), BY(a_BY), BZ(a_BZ), levelMax(lmax) {
    int n_max = std::max(std::max(BX, BY), BZ);
    base_level = (log(n_max) / log(2));
    if (base_level < (double)(log(n_max) / log(2)))
      base_level++;
    i_inverse.resize(lmax);
    j_inverse.resize(lmax);
    k_inverse.resize(lmax);
    Zsave.resize(lmax);
    {
      const int l = 0;
      int aux = pow(pow(2, l), 3);
      i_inverse[l].resize(BX * BY * BZ * aux, -1);
      j_inverse[l].resize(BX * BY * BZ * aux, -1);
      k_inverse[l].resize(BX * BY * BZ * aux, -1);
      Zsave[l].resize(BX * BY * BZ * aux, -1);
    }
    isRegular = true;
#pragma omp parallel for collapse(3)
    for (int k = 0; k < BZ; k++)
      for (int j = 0; j < BY; j++)
        for (int i = 0; i < BX; i++) {
          const int c[3] = {i, j, k};
          long long index = AxestoTranspose(c, base_level);
          long long substract = 0;
          for (long long h = 0; h < index; h++) {
            long long X[3] = {0, 0, 0};
            TransposetoAxes(h, X, base_level);
            if (X[0] >= BX || X[1] >= BY || X[2] >= BZ)
              substract++;
          }
          index -= substract;
          if (substract > 0)
            isRegular = false;
          i_inverse[0][index] = i;
          j_inverse[0][index] = j;
          k_inverse[0][index] = k;
          Zsave[0][k * BX * BY + j * BX + i] = index;
        }
  }
  long long forward(const int l, const int i, const int j, const int k) {
    const int aux = 1 << l;
    if (l >= levelMax)
      return 0;
    long long retval;
    if (!isRegular) {
      const int I = i / aux;
      const int J = j / aux;
      const int K = k / aux;
      assert(!(I >= BX || J >= BY || K >= BZ));
      const int c2_a[3] = {i - I * aux, j - J * aux, k - K * aux};
      retval = AxestoTranspose(c2_a, l);
      retval += IJK_to_index(I, J, K) * aux * aux * aux;
    } else {
      const int c2_a[3] = {i, j, k};
      retval = AxestoTranspose(c2_a, l + base_level);
    }
    return retval;
  }
  void inverse(long long Z, int l, int &i, int &j, int &k) {
    if (isRegular) {
      long long X[3] = {0, 0, 0};
      TransposetoAxes(Z, X, l + base_level);
      i = X[0];
      j = X[1];
      k = X[2];
    } else {
      long long aux = 1 << l;
      long long Zloc = Z % (aux * aux * aux);
      long long X[3] = {0, 0, 0};
      TransposetoAxes(Zloc, X, l);
      long long index = Z / (aux * aux * aux);
      int I, J, K;
      index_to_IJK(index, I, J, K);
      i = X[0] + I * aux;
      j = X[1] + J * aux;
      k = X[2] + K * aux;
    }
    return;
  }
  long long IJK_to_index(int I, int J, int K) {
    long long index = Zsave[0][(J + K * BY) * BX + I];
    return index;
  }
  void index_to_IJK(long long index, int &I, int &J, int &K) {
    I = i_inverse[0][index];
    J = j_inverse[0][index];
    K = k_inverse[0][index];
    return;
  }
  long long Encode(int level, long long Z, int index[3]) {
    int lmax = levelMax;
    long long retval = 0;
    int ix = index[0];
    int iy = index[1];
    int iz = index[2];
    for (int l = level; l >= 0; l--) {
      long long Zp = forward(l, ix, iy, iz);
      retval += Zp;
      ix /= 2;
      iy /= 2;
      iz /= 2;
    }
    ix = 2 * index[0];
    iy = 2 * index[1];
    iz = 2 * index[2];
    for (int l = level + 1; l < lmax; l++) {
      long long Zc = forward(l, ix, iy, iz);
      Zc -= Zc % 8;
      retval += Zc;
      int ix1, iy1, iz1;
      ix1 = ix;
      iy1 = iy;
      iz1 = iz;
      inverse(Zc, l, ix1, iy1, iz1);
      ix = 2 * ix1;
      iy = 2 * iy1;
      iz = 2 * iz1;
    }
    retval += level;
    return retval;
  }
};
enum State : signed char { Leave = 0, Refine = 1, Compress = -1 };
struct TreePosition {
  int position{-3};
  bool CheckCoarser() const { return position == -2; }
  bool CheckFiner() const { return position == -1; }
  bool Exists() const { return position >= 0; }
  int rank() const { return position; }
  void setrank(const int r) { position = r; }
  void setCheckCoarser() { position = -2; }
  void setCheckFiner() { position = -1; }
};
struct Info {
  long long blockID;
  long long blockID_2;
  long long Z;
  long long Znei[3][3][3];
  long long halo_block_id;
  long long Zparent;
  long long Zchild[2][2][2];
  double h;
  double origin[3];
  int index[3];
  int level;
  void *block{nullptr};
  void *auxiliary;
  bool changed2;
  State state;
  static int levelMax(int l = 0) {
    static int lmax = l;
    return lmax;
  }
  static int blocks_per_dim(int i, int nx = 0, int ny = 0, int nz = 0) {
    static int a[3] = {nx, ny, nz};
    return a[i];
  }
  static SpaceFillingCurve *SFC() {
    static SpaceFillingCurve Zcurve(blocks_per_dim(0), blocks_per_dim(1),
                                    blocks_per_dim(2), levelMax());
    return &Zcurve;
  }
  static long long forward(int level, int ix, int iy, int iz) {
    return (*SFC()).forward(level, ix, iy, iz);
  }
  static long long Encode(int level, long long Z, int index[3]) {
    return (*SFC()).Encode(level, Z, index);
  }
  static void inverse(long long Z, int l, int &i, int &j, int &k) {
    (*SFC()).inverse(Z, l, i, j, k);
  }
  template <typename T> inline void pos(T p[3], int ix, int iy, int iz) const {
    p[0] = origin[0] + h * (ix + 0.5);
    p[1] = origin[1] + h * (iy + 0.5);
    p[2] = origin[2] + h * (iz + 0.5);
  }
  template <typename T>
  inline std::array<T, 3> pos(int ix, int iy, int iz) const {
    std::array<T, 3> result;
    pos(result.data(), ix, iy, iz);
    return result;
  }
  bool operator<(const Info &other) const {
    return (blockID_2 < other.blockID_2);
  }
  Info(){};
  void setup(const int a_level, const double a_h, const double a_origin[3],
             const long long a_Z) {
    level = a_level;
    Z = a_Z;
    state = Leave;
    level = a_level;
    h = a_h;
    origin[0] = a_origin[0];
    origin[1] = a_origin[1];
    origin[2] = a_origin[2];
    changed2 = true;
    auxiliary = nullptr;
    const int TwoPower = 1 << level;
    inverse(Z, level, index[0], index[1], index[2]);
    const int Bmax[3] = {blocks_per_dim(0) * TwoPower,
                         blocks_per_dim(1) * TwoPower,
                         blocks_per_dim(2) * TwoPower};
    for (int i = -1; i < 2; i++)
      for (int j = -1; j < 2; j++)
        for (int k = -1; k < 2; k++)
          Znei[i + 1][j + 1][k + 1] =
              forward(level, (index[0] + i + Bmax[0]) % Bmax[0],
                      (index[1] + j + Bmax[1]) % Bmax[1],
                      (index[2] + k + Bmax[2]) % Bmax[2]);
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          Zchild[i][j][k] = forward(level + 1, 2 * index[0] + i,
                                    2 * index[1] + j, 2 * index[2] + k);
    Zparent = (level == 0)
                  ? 0
                  : forward(level - 1, (index[0] / 2 + Bmax[0]) % Bmax[0],
                            (index[1] / 2 + Bmax[1]) % Bmax[1],
                            (index[2] / 2 + Bmax[2]) % Bmax[2]);
    blockID_2 = Encode(level, Z, index);
    blockID = blockID_2;
  }
  long long Znei_(const int i, const int j, const int k) const {
    assert(abs(i) <= 1);
    assert(abs(j) <= 1);
    assert(abs(k) <= 1);
    return Znei[1 + i][1 + j][1 + k];
  }
};

static void dump(Real time, long nblock, Info *infos, char *path) {
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
  ncell = nblock * _BS_ * _BS_ * _BS_;
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
  for (i = 0; i < nblock; i++) {
    Info *info = &infos[i];
    Real *b = (Real *)info->block;
    j = 0;
    for (z = 0; z < _BS_; z++)
      for (y = 0; y < _BS_; y++)
        for (x = 0; x < _BS_; x++) {
          double u0, v0, w0, u1, v1, w1, h;
          h = sim.h0 / (1 << info->level);
          u0 = info->origin[0] + h * x;
          v0 = info->origin[1] + h * y;
          w0 = info->origin[2] + h * z;
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

          attr[l++] = b[j++];
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

template <typename BlockType,
          typename ElementType = typename BlockType::ElementType>
struct BlockCase {
  std::vector<std::vector<ElementType>> m_pData;
  unsigned int m_vSize[3];
  bool storedFace[6];
  int level;
  long long Z;
  BlockCase(bool _storedFace[6], unsigned int nX, unsigned int nY,
            unsigned int nZ, int _level, long long _Z) {
    m_vSize[0] = nX;
    m_vSize[1] = nY;
    m_vSize[2] = nZ;
    storedFace[0] = _storedFace[0];
    storedFace[1] = _storedFace[1];
    storedFace[2] = _storedFace[2];
    storedFace[3] = _storedFace[3];
    storedFace[4] = _storedFace[4];
    storedFace[5] = _storedFace[5];
    m_pData.resize(6);
    for (int d = 0; d < 3; d++) {
      int d1 = (d + 1) % 3;
      int d2 = (d + 2) % 3;
      if (storedFace[2 * d])
        m_pData[2 * d].resize(m_vSize[d1] * m_vSize[d2]);
      if (storedFace[2 * d + 1])
        m_pData[2 * d + 1].resize(m_vSize[d1] * m_vSize[d2]);
    }
    level = _level;
    Z = _Z;
  }
  ~BlockCase() {}
};
template <typename TGrid> class FluxCorrection {
public:
  using GridType = TGrid;
  typedef typename GridType::BlockType BlockType;
  typedef typename BlockType::ElementType ElementType;
  typedef typename ElementType::RealType Real;
  typedef BlockCase<BlockType> Case;
  int rank{0};

protected:
  std::map<std::array<long long, 2>, Case *> MapOfCases;
  TGrid *grid;
  std::vector<Case> Cases;
  void FillCase(Info &info, const int *const code) {
    const int myFace = abs(code[0]) * std::max(0, code[0]) +
                       abs(code[1]) * (std::max(0, code[1]) + 2) +
                       abs(code[2]) * (std::max(0, code[2]) + 4);
    const int otherFace = abs(-code[0]) * std::max(0, -code[0]) +
                          abs(-code[1]) * (std::max(0, -code[1]) + 2) +
                          abs(-code[2]) * (std::max(0, -code[2]) + 4);
    std::array<long long, 2> temp = {(long long)info.level, info.Z};
    auto search = MapOfCases.find(temp);
    Case &CoarseCase = (*search->second);
    std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];
    assert(myFace / 2 == otherFace / 2);
    assert(search != MapOfCases.end());
    assert(CoarseCase.Z == info.Z);
    assert(CoarseCase.level == info.level);
    for (int B = 0; B <= 3; B++) {
      const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
      const long long Z = (*grid).getZforward(
          info.level + 1,
          2 * info.index[0] + std::max(code[0], 0) + code[0] +
              (B % 2) * std::max(0, 1 - abs(code[0])),
          2 * info.index[1] + std::max(code[1], 0) + code[1] +
              aux * std::max(0, 1 - abs(code[1])),
          2 * info.index[2] + std::max(code[2], 0) + code[2] +
              (B / 2) * std::max(0, 1 - abs(code[2])));
      const int other_rank = grid->Tree(info.level + 1, Z).rank();
      if (other_rank != rank)
        continue;
      auto search1 = MapOfCases.find({info.level + 1, Z});
      Case &FineCase = (*search1->second);
      std::vector<ElementType> &FineFace = FineCase.m_pData[otherFace];
      const int d = myFace / 2;
      const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
      const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
      const int N1F = FineCase.m_vSize[d1];
      const int N2F = FineCase.m_vSize[d2];
      const int N1 = N1F;
      const int N2 = N2F;
      int base = 0;
      if (B == 1)
        base = (N2 / 2) + (0) * N2;
      else if (B == 2)
        base = (0) + (N1 / 2) * N2;
      else if (B == 3)
        base = (N2 / 2) + (N1 / 2) * N2;
      assert(search1 != MapOfCases.end());
      assert(N1F == (int)CoarseCase.m_vSize[d1]);
      assert(N2F == (int)CoarseCase.m_vSize[d2]);
      assert(FineFace.size() == CoarseFace.size());
      for (int i1 = 0; i1 < N1; i1 += 2)
        for (int i2 = 0; i2 < N2; i2 += 2) {
          CoarseFace[base + (i2 / 2) + (i1 / 2) * N2] +=
              FineFace[i2 + i1 * N2] + FineFace[i2 + 1 + i1 * N2] +
              FineFace[i2 + (i1 + 1) * N2] + FineFace[i2 + 1 + (i1 + 1) * N2];
          FineFace[i2 + i1 * N2].clear();
          FineFace[i2 + 1 + i1 * N2].clear();
          FineFace[i2 + (i1 + 1) * N2].clear();
          FineFace[i2 + 1 + (i1 + 1) * N2].clear();
        }
    }
  }

public:
  virtual void prepare(TGrid &_grid) {
    if (_grid.UpdateFluxCorrection == false)
      return;
    _grid.UpdateFluxCorrection = false;
    Cases.clear();
    MapOfCases.clear();
    grid = &_grid;
    std::vector<Info> &B = (*grid).getBlocksInfo();
    std::array<int, 3> blocksPerDim = (*grid).getMaxBlocks();
    std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1,
                                1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1,
                                1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};
    for (auto &info : B) {
      grid->getInfoAll(info.level, info.Z).auxiliary = nullptr;
      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      bool storeFace[6] = {false, false, false, false, false, false};
      bool stored = false;
      for (int f = 0; f < 6; f++) {
        const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1,
                             (icode[f] / 9) % 3 - 1};
        if (!_grid.xperiodic && code[0] == xskip && xskin)
          continue;
        if (!_grid.yperiodic && code[1] == yskip && yskin)
          continue;
        if (!_grid.zperiodic && code[2] == zskip && zskin)
          continue;
        if (!grid->Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                 .Exists()) {
          storeFace[abs(code[0]) * std::max(0, code[0]) +
                    abs(code[1]) * (std::max(0, code[1]) + 2) +
                    abs(code[2]) * (std::max(0, code[2]) + 4)] = true;
          stored = true;
        }
      }
      if (stored) {
        Cases.push_back(Case(storeFace, BlockType::sizeX, BlockType::sizeY,
                             BlockType::sizeZ, info.level, info.Z));
      }
    }
    size_t Cases_index = 0;
    if (Cases.size() > 0)
      for (auto &info : B) {
        if (Cases_index == Cases.size())
          break;
        if (Cases[Cases_index].level == info.level &&
            Cases[Cases_index].Z == info.Z) {
          MapOfCases.insert(std::pair<std::array<long long, 2>, Case *>(
              {Cases[Cases_index].level, Cases[Cases_index].Z},
              &Cases[Cases_index]));
          grid->getInfoAll(Cases[Cases_index].level, Cases[Cases_index].Z)
              .auxiliary = &Cases[Cases_index];
          info.auxiliary = &Cases[Cases_index];
          Cases_index++;
        }
      }
  }
  virtual void FillBlockCases() {
    std::vector<Info> &B = (*grid).getBlocksInfo();
    std::array<int, 3> blocksPerDim = (*grid).getMaxBlocks();
    std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1,
                                1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1,
                                1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};
#pragma omp parallel for
    for (size_t i = 0; i < B.size(); i++) {
      Info &info = B[i];
      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      for (int f = 0; f < 6; f++) {
        const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1,
                             (icode[f] / 9) % 3 - 1};
        if (!grid->xperiodic && code[0] == xskip && xskin)
          continue;
        if (!grid->yperiodic && code[1] == yskip && yskin)
          continue;
        if (!grid->zperiodic && code[2] == zskip && zskin)
          continue;
        bool checkFiner =
            grid->Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                .CheckFiner();
        if (checkFiner) {
          FillCase(info, code);
          const int myFace = abs(code[0]) * std::max(0, code[0]) +
                             abs(code[1]) * (std::max(0, code[1]) + 2) +
                             abs(code[2]) * (std::max(0, code[2]) + 4);
          std::array<long long, 2> temp = {(long long)info.level, info.Z};
          auto search = MapOfCases.find(temp);
          assert(search != MapOfCases.end());
          Case &CoarseCase = (*search->second);
          std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];
          const int d = myFace / 2;
          const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
          const int N2 = CoarseCase.m_vSize[d2];
          BlockType &block = *(BlockType *)info.block;
          const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
          const int N1 = CoarseCase.m_vSize[d1];
          if (d == 0) {
            const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeX - 1;
            for (int i1 = 0; i1 < N1; i1++)
              for (int i2 = 0; i2 < N2; i2++) {
                block(j, i2, i1) += CoarseFace[i2 + i1 * N2];
                CoarseFace[i2 + i1 * N2].clear();
              }
          } else if (d == 1) {
            const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeY - 1;
            for (int i1 = 0; i1 < N1; i1++)
              for (int i2 = 0; i2 < N2; i2++) {
                block(i2, j, i1) += CoarseFace[i2 + i1 * N2];
                CoarseFace[i2 + i1 * N2].clear();
              }
          } else {
            const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeZ - 1;
            for (int i1 = 0; i1 < N1; i1++)
              for (int i2 = 0; i2 < N2; i2++) {
                block(i2, i1, j) += CoarseFace[i2 + i1 * N2];
                CoarseFace[i2 + i1 * N2].clear();
              }
          }
        }
      }
    }
  }
};
struct BlockGroup {
  int i_min[3];
  int i_max[3];
  int level;
  std::vector<long long> Z;
  size_t ID;
  double origin[3];
  double h;
  int NXX;
  int NYY;
  int NZZ;
};
template <typename Block,
          template <typename X> class allocator = std::allocator>
class Grid {
public:
  typedef Block BlockType;
  using ElementType = typename Block::ElementType;
  typedef typename Block::RealType Real;
  std::unordered_map<long long, Info *> InfoAll;
  std::unordered_map<long long, TreePosition> Octree;
  std::vector<Info> m_vInfo;
  const int NX;
  const int NY;
  const int NZ;
  const double maxextent;
  const int levelMax;
  const int levelStart;
  const bool xperiodic;
  const bool yperiodic;
  const bool zperiodic;
  std::vector<BlockGroup> MyGroups;
  std::vector<long long> level_base;
  bool UpdateFluxCorrection{true};
  bool UpdateGroups{true};
  bool FiniteDifferences{true};
  FluxCorrection<Grid> CorrectorGrid;
  TreePosition &Tree(const int m, const long long n) {
    const long long aux = level_base[m] + n;
    const auto retval = Octree.find(aux);
    if (retval == Octree.end()) {
      {
        const auto retval1 = Octree.find(aux);
        if (retval1 == Octree.end()) {
          TreePosition dum;
          Octree[aux] = dum;
        }
      }
      return Tree(m, n);
    } else {
      return retval->second;
    }
  }
  TreePosition &Tree(Info &info) { return Tree(info.level, info.Z); }
  TreePosition &Tree(const Info &info) { return Tree(info.level, info.Z); }
  void _alloc() {
    const int m = levelStart;
    const int TwoPower = 1 << m;
    for (long long n = 0; n < NX * NY * NZ * pow(TwoPower, DIMENSION); n++) {
      Tree(m, n).setrank(0);
      _alloc(m, n);
    }
    if (m - 1 >= 0) {
      for (long long n = 0; n < NX * NY * NZ * pow((1 << (m - 1)), DIMENSION);
           n++)
        Tree(m - 1, n).setCheckFiner();
    }
    if (m + 1 < levelMax) {
      for (long long n = 0; n < NX * NY * NZ * pow((1 << (m + 1)), DIMENSION);
           n++)
        Tree(m + 1, n).setCheckCoarser();
    }
    FillPos();
  }
  void _alloc(const int m, const long long n) {
    allocator<Block> alloc;
    Info &new_info = getInfoAll(m, n);
    new_info.block = alloc.allocate(1);
#pragma omp critical
    { m_vInfo.push_back(new_info); }
    Tree(m, n).setrank(rank());
  }
  void _deallocAll() {
    allocator<Block> alloc;
    for (size_t i = 0; i < m_vInfo.size(); i++) {
      const int m = m_vInfo[i].level;
      const long long n = m_vInfo[i].Z;
      alloc.deallocate((Block *)getInfoAll(m, n).block, 1);
    }
    std::vector<long long> aux;
    for (auto &m : InfoAll)
      aux.push_back(m.first);
    for (size_t i = 0; i < aux.size(); i++) {
      const auto retval = InfoAll.find(aux[i]);
      if (retval != InfoAll.end()) {
        delete retval->second;
      }
    }
    m_vInfo.clear();
    InfoAll.clear();
    Octree.clear();
  }
  void _dealloc(const int m, const long long n) {
    allocator<Block> alloc;
    alloc.deallocate((Block *)getInfoAll(m, n).block, 1);
    for (size_t j = 0; j < m_vInfo.size(); j++) {
      if (m_vInfo[j].level == m && m_vInfo[j].Z == n) {
        m_vInfo.erase(m_vInfo.begin() + j);
        return;
      }
    }
  }
  void dealloc_many(const std::vector<long long> &dealloc_IDs) {
    for (size_t j = 0; j < m_vInfo.size(); j++)
      m_vInfo[j].changed2 = false;
    allocator<Block> alloc;
    for (size_t i = 0; i < dealloc_IDs.size(); i++)
      for (size_t j = 0; j < m_vInfo.size(); j++) {
        if (m_vInfo[j].blockID_2 == dealloc_IDs[i]) {
          const int m = m_vInfo[j].level;
          const long long n = m_vInfo[j].Z;
          m_vInfo[j].changed2 = true;
          alloc.deallocate((Block *)getInfoAll(m, n).block, 1);
          break;
        }
      }
    m_vInfo.erase(std::remove_if(m_vInfo.begin(), m_vInfo.end(),
                                 [](const Info &x) { return x.changed2; }),
                  m_vInfo.end());
  }
  void FindInfo(const int m, const long long n, const int m_new,
                const long long n_new) {
    for (size_t j = 0; j < m_vInfo.size(); j++)
      if (m == m_vInfo[j].level && n == m_vInfo[j].Z) {
        Info &correct_info = getInfoAll(m_new, n_new);
        correct_info.state = Leave;
        m_vInfo[j] = correct_info;
        return;
      }
  }
  virtual void FillPos(bool CopyInfos = true) {
    std::sort(m_vInfo.begin(), m_vInfo.end());
    if (CopyInfos)
      for (size_t j = 0; j < m_vInfo.size(); j++) {
        const int m = m_vInfo[j].level;
        const long long n = m_vInfo[j].Z;
        Info &correct_info = getInfoAll(m, n);
        correct_info.blockID = j;
        m_vInfo[j] = correct_info;
        assert(Tree(m, n).Exists());
      }
    else
      for (size_t j = 0; j < m_vInfo.size(); j++) {
        const int m = m_vInfo[j].level;
        const long long n = m_vInfo[j].Z;
        Info &correct_info = getInfoAll(m, n);
        correct_info.blockID = j;
        m_vInfo[j].blockID = j;
        m_vInfo[j].state = correct_info.state;
        assert(Tree(m, n).Exists());
      }
  }
  Grid(const unsigned int _NX, const unsigned int _NY = 1,
       const unsigned int _NZ = 1, const double _maxextent = 1,
       const unsigned int _levelStart = 0, const unsigned int _levelMax = 1,
       const bool AllocateBlocks = true, const bool a_xperiodic = true,
       const bool a_yperiodic = true, const bool a_zperiodic = true)
      : NX(_NX), NY(_NY), NZ(_NZ), maxextent(_maxextent), levelMax(_levelMax),
        levelStart(_levelStart), xperiodic(a_xperiodic), yperiodic(a_yperiodic),
        zperiodic(a_zperiodic) {
    Info dummy;
    const int nx = dummy.blocks_per_dim(0, NX, NY, NZ);
    const int ny = dummy.blocks_per_dim(1, NX, NY, NZ);
    const int nz = dummy.blocks_per_dim(2, NX, NY, NZ);
    const int lvlMax = dummy.levelMax(levelMax);
    for (int m = 0; m < lvlMax; m++) {
      const int TwoPower = 1 << m;
      const long long Ntot = nx * ny * nz * pow(TwoPower, DIMENSION);
      if (m == 0)
        level_base.push_back(Ntot);
      if (m > 0)
        level_base.push_back(level_base[m - 1] + Ntot);
    }
    if (AllocateBlocks)
      _alloc();
  }
  virtual ~Grid() { _deallocAll(); }
  virtual Block *avail(const int m, const long long n) {
    return (Block *)getInfoAll(m, n).block;
  }
  virtual int rank() const { return 0; }
  virtual void initialize_blocks(const std::vector<long long> &blocksZ,
                                 const std::vector<short int> &blockslevel) {
    _deallocAll();
    for (size_t i = 0; i < blocksZ.size(); i++) {
      const int level = blockslevel[i];
      const long long Z = blocksZ[i];
      _alloc(level, Z);
      Tree(level, Z).setrank(rank());
      int p[3];
      Info::inverse(Z, level, p[0], p[1], p[2]);
      if (level < levelMax - 1)
        for (int k1 = 0; k1 < 2; k1++)
          for (int j1 = 0; j1 < 2; j1++)
            for (int i1 = 0; i1 < 2; i1++) {
              const long long nc = getZforward(level + 1, 2 * p[0] + i1,
                                               2 * p[1] + j1, 2 * p[2] + k1);
              Tree(level + 1, nc).setCheckCoarser();
            }
      if (level > 0) {
        const long long nf =
            getZforward(level - 1, p[0] / 2, p[1] / 2, p[2] / 2);
        Tree(level - 1, nf).setCheckFiner();
      }
    }
    FillPos();
    UpdateFluxCorrection = true;
    UpdateGroups = true;
  }
  long long getZforward(const int level, const int i, const int j,
                        const int k) const {
    const int TwoPower = 1 << level;
    const int ix = (i + TwoPower * NX) % (NX * TwoPower);
    const int iy = (j + TwoPower * NY) % (NY * TwoPower);
    const int iz = (k + TwoPower * NZ) % (NZ * TwoPower);
    return Info::forward(level, ix, iy, iz);
  }
  Block *avail1(const int ix, const int iy, const int iz, const int m) {
    const long long n = getZforward(m, ix, iy, iz);
    return avail(m, n);
  }
  Block &operator()(const long long ID) { return *(Block *)m_vInfo[ID].block; }
  std::array<int, 3> getMaxBlocks() const { return {NX, NY, NZ}; }
  std::array<int, 3> getMaxMostRefinedBlocks() const {
    return {
        NX << (levelMax - 1),
        NY << (levelMax - 1),
        DIMENSION == 3 ? (NZ << (levelMax - 1)) : 1,
    };
  }
  std::array<int, 3> getMaxMostRefinedCells() const {
    const auto b = getMaxMostRefinedBlocks();
    return {b[0] * Block::sizeX, b[1] * Block::sizeY, b[2] * Block::sizeZ};
  }
  inline int getlevelMax() const { return levelMax; }
  Info &getInfoAll(const int m, const long long n) {
    const long long aux = level_base[m] + n;
    const auto retval = InfoAll.find(aux);
    if (retval != InfoAll.end()) {
      return *retval->second;
    } else {
      {
        const auto retval1 = InfoAll.find(aux);
        if (retval1 == InfoAll.end()) {
          Info *dumm = new Info();
          const int TwoPower = 1 << m;
          const double h0 = (maxextent / std::max(NX * Block::sizeX,
                                                  std::max(NY * Block::sizeY,
                                                           NZ * Block::sizeZ)));
          const double h = h0 / TwoPower;
          double origin[3];
          int i, j, k;
          Info::inverse(n, m, i, j, k);
          origin[0] = i * Block::sizeX * h;
          origin[1] = j * Block::sizeY * h;
          origin[2] = k * Block::sizeZ * h;
          dumm->setup(m, h, origin, n);
          InfoAll[aux] = dumm;
        }
      }
      return getInfoAll(m, n);
    }
  }
  std::vector<Info> &getBlocksInfo() { return m_vInfo; }
  const std::vector<Info> &getBlocksInfo() const { return m_vInfo; }
  virtual int get_world_size() const { return 1; }
  virtual void UpdateBoundary(bool clean = false) {}
};
struct StencilInfo {
  int sx;
  int sy;
  int sz;
  int ex;
  int ey;
  int ez;
  std::vector<int> selcomponents;
  bool tensorial;
  StencilInfo() {}
  StencilInfo(int _sx, int _sy, int _sz, int _ex, int _ey, int _ez,
              bool _tensorial, const std::vector<int> &components)
      : sx(_sx), sy(_sy), sz(_sz), ex(_ex), ey(_ey), ez(_ez),
        selcomponents(components), tensorial(_tensorial) {
    assert(selcomponents.size() > 0);
    if (!isvalid()) {
      std::cout << "Stencilinfo instance not valid. Aborting\n";
      abort();
    }
  }
  StencilInfo(const StencilInfo &c)
      : sx(c.sx), sy(c.sy), sz(c.sz), ex(c.ex), ey(c.ey), ez(c.ez),
        selcomponents(c.selcomponents), tensorial(c.tensorial) {}
  std::vector<int> _all() const {
    int extra[] = {sx, sy, sz, ex, ey, ez, (int)tensorial};
    std::vector<int> all(selcomponents);
    all.insert(all.end(), extra, extra + sizeof(extra) / sizeof(int));
    return all;
  }
  bool operator<(StencilInfo s) const {
    std::vector<int> me = _all(), you = s._all();
    const int N = std::min(me.size(), you.size());
    for (int i = 0; i < N; ++i)
      if (me[i] < you[i])
        return true;
      else if (me[i] > you[i])
        return false;
    return me.size() < you.size();
  }
  bool isvalid() const {
    const bool not0 = selcomponents.size() == 0;
    const bool not1 = sx > 0 || ex <= 0 || sx > ex;
    const bool not2 = sy > 0 || ey <= 0 || sy > ey;
    const bool not3 = sz > 0 || ez <= 0 || sz > ez;
    return !(not0 || not1 || not2 || not3);
  }
};
template <typename Real>
inline void pack(const Real *const srcbase, Real *const dst,
                 const unsigned int gptfloats, int *selected_components,
                 const int ncomponents, const int xstart, const int ystart,
                 const int zstart, const int xend, const int yend,
                 const int zend, const int BSX, const int BSY) {
  if (gptfloats == 1) {
    const int mod = (xend - xstart) % 4;
    for (int idst = 0, iz = zstart; iz < zend; ++iz)
      for (int iy = ystart; iy < yend; ++iy) {
        for (int ix = xstart; ix < xend - mod; ix += 4, idst += 4) {
          dst[idst + 0] = srcbase[ix + 0 + BSX * (iy + BSY * iz)];
          dst[idst + 1] = srcbase[ix + 1 + BSX * (iy + BSY * iz)];
          dst[idst + 2] = srcbase[ix + 2 + BSX * (iy + BSY * iz)];
          dst[idst + 3] = srcbase[ix + 3 + BSX * (iy + BSY * iz)];
        }
        for (int ix = xend - mod; ix < xend; ix++, idst++) {
          dst[idst] = srcbase[ix + BSX * (iy + BSY * iz)];
        }
      }
  } else {
    for (int idst = 0, iz = zstart; iz < zend; ++iz)
      for (int iy = ystart; iy < yend; ++iy)
        for (int ix = xstart; ix < xend; ++ix) {
          const Real *src = srcbase + gptfloats * (ix + BSX * (iy + BSY * iz));
          for (int ic = 0; ic < ncomponents; ic++, idst++)
            dst[idst] = src[selected_components[ic]];
        }
  }
}
template <typename Real>
inline void unpack_subregion(
    const Real *const pack, Real *const dstbase, const unsigned int gptfloats,
    const int *const selected_components, const int ncomponents,
    const int srcxstart, const int srcystart, const int srczstart, const int LX,
    const int LY, const int dstxstart, const int dstystart, const int dstzstart,
    const int dstxend, const int dstyend, const int dstzend, const int xsize,
    const int ysize, const int zsize) {
  if (gptfloats == 1) {
    const int mod = (dstxend - dstxstart) % 4;
    for (int zd = dstzstart; zd < dstzend; ++zd)
      for (int yd = dstystart; yd < dstyend; ++yd) {
        const int offset = -dstxstart + srcxstart +
                           LX * (yd - dstystart + srcystart +
                                 LY * (zd - dstzstart + srczstart));
        const int offset_dst = xsize * (yd + ysize * zd);
        for (int xd = dstxstart; xd < dstxend - mod; xd += 4) {
          dstbase[xd + 0 + offset_dst] = pack[xd + 0 + offset];
          dstbase[xd + 1 + offset_dst] = pack[xd + 1 + offset];
          dstbase[xd + 2 + offset_dst] = pack[xd + 2 + offset];
          dstbase[xd + 3 + offset_dst] = pack[xd + 3 + offset];
        }
        for (int xd = dstxend - mod; xd < dstxend; ++xd) {
          dstbase[xd + offset_dst] = pack[xd + offset];
        }
      }
  } else {
    for (int zd = dstzstart; zd < dstzend; ++zd)
      for (int yd = dstystart; yd < dstyend; ++yd)
        for (int xd = dstxstart; xd < dstxend; ++xd) {
          Real *const dst =
              dstbase + gptfloats * (xd + xsize * (yd + ysize * zd));
          const Real *src =
              pack + ncomponents * (xd - dstxstart + srcxstart +
                                    LX * (yd - dstystart + srcystart +
                                          LY * (zd - dstzstart + srczstart)));
          for (int c = 0; c < ncomponents; ++c)
            dst[selected_components[c]] = src[c];
        }
  }
}
template <typename T> class GrowingVector {
  size_t pos;
  size_t s;

public:
  std::vector<T> v;
  GrowingVector() {
    pos = 0;
    s = 0;
  }
  GrowingVector(size_t size) { resize(size); }
  GrowingVector(size_t size, T value) { resize(size, value); }
  void resize(size_t new_size, T value) {
    v.resize(new_size, value);
    s = new_size;
  }
  void resize(size_t new_size) {
    v.resize(new_size);
    s = new_size;
  }
  size_t size() { return s; }
  void clear() {
    pos = 0;
    s = 0;
  }
  void push_back(T value) {
    if (pos < v.size())
      v[pos] = value;
    else
      v.push_back(value);
    pos++;
    s++;
  }
  T *data() { return v.data(); }
  T &operator[](size_t i) { return v[i]; }
  T &back() { return v[pos - 1]; }
  typename std::vector<T>::iterator begin() { return v.begin(); }
  typename std::vector<T>::iterator end() { return v.begin() + pos; }
  void EraseAll() {
    v.clear();
    pos = 0;
    s = 0;
  }
  ~GrowingVector() { v.clear(); }
};
struct Interface {
  Info *infos[2];
  int icode[2];
  bool CoarseStencil;
  bool ToBeKept;
  int dis;
  Interface(Info &i0, Info &i1, const int a_icode0, const int a_icode1) {
    infos[0] = &i0;
    infos[1] = &i1;
    icode[0] = a_icode0;
    icode[1] = a_icode1;
    CoarseStencil = false;
    ToBeKept = true;
    dis = 0;
  }
  bool operator<(const Interface &other) const {
    if (infos[0]->blockID_2 == other.infos[0]->blockID_2) {
      if (icode[0] == other.icode[0]) {
        if (infos[1]->blockID_2 == other.infos[1]->blockID_2) {
          return (icode[1] < other.icode[1]);
        }
        return (infos[1]->blockID_2 < other.infos[1]->blockID_2);
      }
      return (icode[0] < other.icode[0]);
    }
    return (infos[0]->blockID_2 < other.infos[0]->blockID_2);
  }
};
struct MyRange {
  std::vector<int> removedIndices;
  int index;
  int sx;
  int sy;
  int sz;
  int ex;
  int ey;
  int ez;
  bool needed{true};
  bool avg_down{true};
  bool contains(MyRange &r) const {
    if (avg_down != r.avg_down)
      return false;
    int V = (ez - sz) * (ey - sy) * (ex - sx);
    int Vr = (r.ez - r.sz) * (r.ey - r.sy) * (r.ex - r.sx);
    return (sx <= r.sx && r.ex <= ex) && (sy <= r.sy && r.ey <= ey) &&
           (sz <= r.sz && r.ez <= ez) && (Vr < V);
  }
  void Remove(const MyRange &other) {
    size_t s = removedIndices.size();
    removedIndices.resize(s + other.removedIndices.size());
    for (size_t i = 0; i < other.removedIndices.size(); i++)
      removedIndices[s + i] = other.removedIndices[i];
  }
};
struct UnPackInfo {
  int offset;
  int lx;
  int ly;
  int lz;
  int srcxstart;
  int srcystart;
  int srczstart;
  int LX;
  int LY;
  int CoarseVersionOffset;
  int CoarseVersionLX;
  int CoarseVersionLY;
  int CoarseVersionsrcxstart;
  int CoarseVersionsrcystart;
  int CoarseVersionsrczstart;
  int level;
  int icode;
  int rank;
  int index_0;
  int index_1;
  int index_2;
  long long IDreceiver;
};
struct StencilManager {
  const StencilInfo stencil;
  const StencilInfo Cstencil;
  int nX;
  int nY;
  int nZ;
  int sLength[3 * 27 * 3];
  std::array<MyRange, 3 * 27> AllStencils;
  MyRange Coarse_Range;
  StencilManager(StencilInfo a_stencil, StencilInfo a_Cstencil, int a_nX,
                 int a_nY, int a_nZ)
      : stencil(a_stencil), Cstencil(a_Cstencil), nX(a_nX), nY(a_nY), nZ(a_nZ) {
    const int sC[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                       (stencil.sy - 1) / 2 + Cstencil.sy,
                       (stencil.sz - 1) / 2 + Cstencil.sz};
    const int eC[3] = {stencil.ex / 2 + Cstencil.ex,
                       stencil.ey / 2 + Cstencil.ey,
                       stencil.ez / 2 + Cstencil.ez};
    for (int icode = 0; icode < 27; icode++) {
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                           (icode / 9) % 3 - 1};
      MyRange &range0 = AllStencils[icode];
      range0.sx = code[0] < 1 ? (code[0] < 0 ? nX + stencil.sx : 0) : 0;
      range0.sy = code[1] < 1 ? (code[1] < 0 ? nY + stencil.sy : 0) : 0;
      range0.sz = code[2] < 1 ? (code[2] < 0 ? nZ + stencil.sz : 0) : 0;
      range0.ex = code[0] < 1 ? nX : stencil.ex - 1;
      range0.ey = code[1] < 1 ? nY : stencil.ey - 1;
      range0.ez = code[2] < 1 ? nZ : stencil.ez - 1;
      sLength[3 * icode + 0] = range0.ex - range0.sx;
      sLength[3 * icode + 1] = range0.ey - range0.sy;
      sLength[3 * icode + 2] = range0.ez - range0.sz;
      MyRange &range1 = AllStencils[icode + 27];
      range1.sx = code[0] < 1 ? (code[0] < 0 ? nX + 2 * stencil.sx : 0) : 0;
      range1.sy = code[1] < 1 ? (code[1] < 0 ? nY + 2 * stencil.sy : 0) : 0;
      range1.sz = code[2] < 1 ? (code[2] < 0 ? nZ + 2 * stencil.sz : 0) : 0;
      range1.ex = code[0] < 1 ? nX : 2 * (stencil.ex - 1);
      range1.ey = code[1] < 1 ? nY : 2 * (stencil.ey - 1);
      range1.ez = code[2] < 1 ? nZ : 2 * (stencil.ez - 1);
      sLength[3 * (icode + 27) + 0] = (range1.ex - range1.sx) / 2;
      sLength[3 * (icode + 27) + 1] = (range1.ey - range1.sy) / 2;
      sLength[3 * (icode + 27) + 2] = (range1.ez - range1.sz) / 2;
      MyRange &range2 = AllStencils[icode + 2 * 27];
      range2.sx = code[0] < 1 ? (code[0] < 0 ? nX / 2 + sC[0] : 0) : 0;
      range2.sy = code[1] < 1 ? (code[1] < 0 ? nY / 2 + sC[1] : 0) : 0;
      range2.ex = code[0] < 1 ? nX / 2 : eC[0] - 1;
      range2.ey = code[1] < 1 ? nY / 2 : eC[1] - 1;
      range2.sz = code[2] < 1 ? (code[2] < 0 ? nZ / 2 + sC[2] : 0) : 0;
      range2.ez = code[2] < 1 ? nZ / 2 : eC[2] - 1;
      sLength[3 * (icode + 2 * 27) + 0] = range2.ex - range2.sx;
      sLength[3 * (icode + 2 * 27) + 1] = range2.ey - range2.sy;
      sLength[3 * (icode + 2 * 27) + 2] = range2.ez - range2.sz;
    }
  }
  void CoarseStencilLength(const int icode, int *L) const {
    L[0] = sLength[3 * (icode + 2 * 27) + 0];
    L[1] = sLength[3 * (icode + 2 * 27) + 1];
    L[2] = sLength[3 * (icode + 2 * 27) + 2];
  }
  void DetermineStencilLength(const int level_sender, const int level_receiver,
                              const int icode, int *L) {
    if (level_sender == level_receiver) {
      L[0] = sLength[3 * icode + 0];
      L[1] = sLength[3 * icode + 1];
      L[2] = sLength[3 * icode + 2];
    } else if (level_sender > level_receiver) {
      L[0] = sLength[3 * (icode + 27) + 0];
      L[1] = sLength[3 * (icode + 27) + 1];
      L[2] = sLength[3 * (icode + 27) + 2];
    } else {
      L[0] = sLength[3 * (icode + 2 * 27) + 0];
      L[1] = sLength[3 * (icode + 2 * 27) + 1];
      L[2] = sLength[3 * (icode + 2 * 27) + 2];
    }
  }
  MyRange &DetermineStencil(const Interface &f, bool CoarseVersion = false) {
    if (CoarseVersion) {
      AllStencils[f.icode[1] + 2 * 27].needed = true;
      return AllStencils[f.icode[1] + 2 * 27];
    } else {
      if (f.infos[0]->level == f.infos[1]->level) {
        AllStencils[f.icode[1]].needed = true;
        return AllStencils[f.icode[1]];
      } else if (f.infos[0]->level > f.infos[1]->level) {
        AllStencils[f.icode[1] + 27].needed = true;
        return AllStencils[f.icode[1] + 27];
      } else {
        Coarse_Range.needed = true;
        const int code[3] = {f.icode[1] % 3 - 1, (f.icode[1] / 3) % 3 - 1,
                             (f.icode[1] / 9) % 3 - 1};
        const int s[3] = {
            code[0] < 1
                ? (code[0] < 0 ? ((stencil.sx - 1) / 2 + Cstencil.sx) : 0)
                : nX / 2,
            code[1] < 1
                ? (code[1] < 0 ? ((stencil.sy - 1) / 2 + Cstencil.sy) : 0)
                : nY / 2,
            code[2] < 1
                ? (code[2] < 0 ? ((stencil.sz - 1) / 2 + Cstencil.sz) : 0)
                : nZ / 2};
        const int e[3] = {
            code[0] < 1 ? (code[0] < 0 ? 0 : nX / 2)
                        : nX / 2 + stencil.ex / 2 + Cstencil.ex - 1,
            code[1] < 1 ? (code[1] < 0 ? 0 : nY / 2)
                        : nY / 2 + stencil.ey / 2 + Cstencil.ey - 1,
            code[2] < 1 ? (code[2] < 0 ? 0 : nZ / 2)
                        : nZ / 2 + stencil.ez / 2 + Cstencil.ez - 1};
        const int base[3] = {(f.infos[1]->index[0] + code[0]) % 2,
                             (f.infos[1]->index[1] + code[1]) % 2,
                             (f.infos[1]->index[2] + code[2]) % 2};
        int Cindex_true[3];
        for (int d = 0; d < 3; d++)
          Cindex_true[d] = f.infos[1]->index[d] + code[d];
        int CoarseEdge[3];
        CoarseEdge[0] = (code[0] == 0) ? 0
                        : (((f.infos[1]->index[0] % 2 == 0) &&
                            (Cindex_true[0] > f.infos[1]->index[0])) ||
                           ((f.infos[1]->index[0] % 2 == 1) &&
                            (Cindex_true[0] < f.infos[1]->index[0])))
                            ? 1
                            : 0;
        CoarseEdge[1] = (code[1] == 0) ? 0
                        : (((f.infos[1]->index[1] % 2 == 0) &&
                            (Cindex_true[1] > f.infos[1]->index[1])) ||
                           ((f.infos[1]->index[1] % 2 == 1) &&
                            (Cindex_true[1] < f.infos[1]->index[1])))
                            ? 1
                            : 0;
        CoarseEdge[2] = (code[2] == 0) ? 0
                        : (((f.infos[1]->index[2] % 2 == 0) &&
                            (Cindex_true[2] > f.infos[1]->index[2])) ||
                           ((f.infos[1]->index[2] % 2 == 1) &&
                            (Cindex_true[2] < f.infos[1]->index[2])))
                            ? 1
                            : 0;
        Coarse_Range.sx = s[0] + std::max(code[0], 0) * nX / 2 +
                          (1 - abs(code[0])) * base[0] * nX / 2 - code[0] * nX +
                          CoarseEdge[0] * code[0] * nX / 2;
        Coarse_Range.sy = s[1] + std::max(code[1], 0) * nY / 2 +
                          (1 - abs(code[1])) * base[1] * nY / 2 - code[1] * nY +
                          CoarseEdge[1] * code[1] * nY / 2;
        Coarse_Range.sz = s[2] + std::max(code[2], 0) * nZ / 2 +
                          (1 - abs(code[2])) * base[2] * nZ / 2 - code[2] * nZ +
                          CoarseEdge[2] * code[2] * nZ / 2;
        Coarse_Range.ex = e[0] + std::max(code[0], 0) * nX / 2 +
                          (1 - abs(code[0])) * base[0] * nX / 2 - code[0] * nX +
                          CoarseEdge[0] * code[0] * nX / 2;
        Coarse_Range.ey = e[1] + std::max(code[1], 0) * nY / 2 +
                          (1 - abs(code[1])) * base[1] * nY / 2 - code[1] * nY +
                          CoarseEdge[1] * code[1] * nY / 2;
        Coarse_Range.ez = e[2] + std::max(code[2], 0) * nZ / 2 +
                          (1 - abs(code[2])) * base[2] * nZ / 2 - code[2] * nZ +
                          CoarseEdge[2] * code[2] * nZ / 2;
        return Coarse_Range;
      }
    }
  }
  void __FixDuplicates(const Interface &f, const Interface &f_dup, int lx,
                       int ly, int lz, int lx_dup, int ly_dup, int lz_dup,
                       int &sx, int &sy, int &sz) {
    const Info &receiver = *f.infos[1];
    const Info &receiver_dup = *f_dup.infos[1];
    if (receiver.level >= receiver_dup.level) {
      int icode_dup = f_dup.icode[1];
      const int code_dup[3] = {icode_dup % 3 - 1, (icode_dup / 3) % 3 - 1,
                               (icode_dup / 9) % 3 - 1};
      sx = (lx == lx_dup || code_dup[0] != -1) ? 0 : lx - lx_dup;
      sy = (ly == ly_dup || code_dup[1] != -1) ? 0 : ly - ly_dup;
      sz = (lz == lz_dup || code_dup[2] != -1) ? 0 : lz - lz_dup;
    } else {
      MyRange &range = DetermineStencil(f);
      MyRange &range_dup = DetermineStencil(f_dup);
      sx = range_dup.sx - range.sx;
      sy = range_dup.sy - range.sy;
      sz = range_dup.sz - range.sz;
    }
  }
  void __FixDuplicates2(const Interface &f, const Interface &f_dup, int &sx,
                        int &sy, int &sz) {
    if (f.infos[0]->level != f.infos[1]->level ||
        f_dup.infos[0]->level != f_dup.infos[1]->level)
      return;
    MyRange &range = DetermineStencil(f, true);
    MyRange &range_dup = DetermineStencil(f_dup, true);
    sx = range_dup.sx - range.sx;
    sy = range_dup.sy - range.sy;
    sz = range_dup.sz - range.sz;
  }
};
struct HaloBlockGroup {
  std::vector<Info *> myblocks;
  std::set<int> myranks;
  bool ready = false;
};
template <typename Real, typename TGrid> class SynchronizerMPI_AMR {
  MPI_Comm comm;
  int rank;
  int size;
  StencilInfo stencil;
  StencilInfo Cstencil;
  TGrid *grid;
  int nX;
  int nY;
  int nZ;
  MPI_Datatype MPIREAL;
  std::vector<Info *> inner_blocks;
  std::vector<Info *> halo_blocks;
  std::vector<GrowingVector<Real>> send_buffer;
  std::vector<GrowingVector<Real>> recv_buffer;
  std::vector<MPI_Request> requests;
  std::vector<int> send_buffer_size;
  std::vector<int> recv_buffer_size;
  std::set<int> Neighbors;
  GrowingVector<GrowingVector<UnPackInfo>> myunpacks;
  StencilManager SM;
  const unsigned int gptfloats;
  const int NC;
  struct PackInfo {
    Real *block;
    Real *pack;
    int sx;
    int sy;
    int sz;
    int ex;
    int ey;
    int ez;
  };
  std::vector<GrowingVector<PackInfo>> send_packinfos;
  std::vector<GrowingVector<Interface>> send_interfaces;
  std::vector<GrowingVector<Interface>> recv_interfaces;
  std::vector<std::vector<int>> ToBeAveragedDown;
  bool use_averages;
  std::unordered_map<std::string, HaloBlockGroup> mapofHaloBlockGroups;
  std::unordered_map<int, MPI_Request *> mapofrequests;
  struct DuplicatesManager {
    struct cube {
      GrowingVector<MyRange> compass[27];
      void clear() {
        for (int i = 0; i < 27; i++)
          compass[i].clear();
      }
      cube() {}
      std::vector<MyRange *> keepEl() {
        std::vector<MyRange *> retval;
        for (int i = 0; i < 27; i++)
          for (size_t j = 0; j < compass[i].size(); j++)
            if (compass[i][j].needed)
              retval.push_back(&compass[i][j]);
        return retval;
      }
      void __needed(std::vector<int> &v) {
        static constexpr std::array<int, 3> faces_and_edges[18] = {
            {0, 1, 1}, {2, 1, 1}, {1, 0, 1}, {1, 2, 1}, {1, 1, 0}, {1, 1, 2},
            {0, 0, 1}, {0, 2, 1}, {2, 0, 1}, {2, 2, 1}, {1, 0, 0}, {1, 0, 2},
            {1, 2, 0}, {1, 2, 2}, {0, 1, 0}, {0, 1, 2}, {2, 1, 0}, {2, 1, 2}};
        for (auto &f : faces_and_edges)
          if (compass[f[0] + f[1] * 3 + f[2] * 9].size() != 0) {
            bool needme = false;
            auto &me = compass[f[0] + f[1] * 3 + f[2] * 9];
            for (size_t j1 = 0; j1 < me.size(); j1++)
              if (me[j1].needed) {
                needme = true;
                for (size_t j2 = 0; j2 < me.size(); j2++)
                  if (me[j2].needed && me[j2].contains(me[j1])) {
                    me[j1].needed = false;
                    me[j2].removedIndices.push_back(me[j1].index);
                    me[j2].Remove(me[j1]);
                    v.push_back(me[j1].index);
                    break;
                  }
              }
            if (!needme)
              continue;
            const int imax = (f[0] == 1) ? 2 : f[0];
            const int imin = (f[0] == 1) ? 0 : f[0];
            const int jmax = (f[1] == 1) ? 2 : f[1];
            const int jmin = (f[1] == 1) ? 0 : f[1];
            const int kmax = (f[2] == 1) ? 2 : f[2];
            const int kmin = (f[2] == 1) ? 0 : f[2];
            for (int k = kmin; k <= kmax; k++)
              for (int j = jmin; j <= jmax; j++)
                for (int i = imin; i <= imax; i++) {
                  if (i == f[0] && j == f[1] && k == f[2])
                    continue;
                  auto &other = compass[i + j * 3 + k * 9];
                  for (size_t j1 = 0; j1 < other.size(); j1++) {
                    auto &o = other[j1];
                    if (o.needed)
                      for (size_t k1 = 0; k1 < me.size(); k1++) {
                        auto &m = me[k1];
                        if (m.needed && m.contains(o)) {
                          o.needed = false;
                          m.removedIndices.push_back(o.index);
                          m.Remove(o);
                          v.push_back(o.index);
                          break;
                        }
                      }
                  }
                }
          }
      }
    };
    cube C;
    std::vector<int> offsets;
    std::vector<int> offsets_recv;
    SynchronizerMPI_AMR *Synch_ptr;
    std::vector<int> positions;
    std::vector<size_t> sizes;
    DuplicatesManager(SynchronizerMPI_AMR &Synch) {
      positions.resize(Synch.size);
      sizes.resize(Synch.size);
      offsets.resize(Synch.size, 0);
      offsets_recv.resize(Synch.size, 0);
      Synch_ptr = &Synch;
    }
    void Add(const int r, const int index) {
      if (sizes[r] == 0)
        positions[r] = index;
      sizes[r]++;
    }
    void RemoveDuplicates(const int r, std::vector<Interface> &f,
                          int &total_size) {
      if (sizes[r] == 0)
        return;
      bool skip_needed = false;
      const int nc = Synch_ptr->getstencil().selcomponents.size();
      std::sort(f.begin() + positions[r], f.begin() + sizes[r] + positions[r]);
      C.clear();
      for (size_t i = 0; i < sizes[r]; i++) {
        C.compass[f[i + positions[r]].icode[0]].push_back(
            Synch_ptr->SM.DetermineStencil(f[i + positions[r]]));
        C.compass[f[i + positions[r]].icode[0]].back().index = i + positions[r];
        C.compass[f[i + positions[r]].icode[0]].back().avg_down =
            (f[i + positions[r]].infos[0]->level >
             f[i + positions[r]].infos[1]->level);
        if (skip_needed == false)
          skip_needed = f[i + positions[r]].CoarseStencil;
      }
      if (skip_needed == false) {
        std::vector<int> remEl;
        C.__needed(remEl);
        for (size_t k = 0; k < remEl.size(); k++)
          f[remEl[k]].ToBeKept = false;
      }
      int L[3] = {0, 0, 0};
      int Lc[3] = {0, 0, 0};
      for (auto &i : C.keepEl()) {
        const int k = i->index;
        Synch_ptr->SM.DetermineStencilLength(
            f[k].infos[0]->level, f[k].infos[1]->level, f[k].icode[1], L);
        const int V = L[0] * L[1] * L[2];
        total_size += V;
        f[k].dis = offsets[r];
        if (f[k].CoarseStencil) {
          Synch_ptr->SM.CoarseStencilLength(f[k].icode[1], Lc);
          const int Vc = Lc[0] * Lc[1] * Lc[2];
          total_size += Vc;
          offsets[r] += Vc * nc;
        }
        offsets[r] += V * nc;
        for (size_t kk = 0; kk < (*i).removedIndices.size(); kk++)
          f[i->removedIndices[kk]].dis = f[k].dis;
      }
    }
    void RemoveDuplicates_recv(std::vector<Interface> &f, int &total_size,
                               const int otherrank, const size_t start,
                               const size_t finish) {
      bool skip_needed = false;
      const int nc = Synch_ptr->getstencil().selcomponents.size();
      C.clear();
      for (size_t i = start; i < finish; i++) {
        C.compass[f[i].icode[0]].push_back(
            Synch_ptr->SM.DetermineStencil(f[i]));
        C.compass[f[i].icode[0]].back().index = i;
        C.compass[f[i].icode[0]].back().avg_down =
            (f[i].infos[0]->level > f[i].infos[1]->level);
        if (skip_needed == false)
          skip_needed = f[i].CoarseStencil;
      }
      if (skip_needed == false) {
        std::vector<int> remEl;
        C.__needed(remEl);
        for (size_t k = 0; k < remEl.size(); k++)
          f[remEl[k]].ToBeKept = false;
      }
      for (auto &i : C.keepEl()) {
        const int k = i->index;
        int L[3] = {0, 0, 0};
        int Lc[3] = {0, 0, 0};
        Synch_ptr->SM.DetermineStencilLength(
            f[k].infos[0]->level, f[k].infos[1]->level, f[k].icode[1], L);
        const int V = L[0] * L[1] * L[2];
        int Vc = 0;
        total_size += V;
        f[k].dis = offsets_recv[otherrank];
        UnPackInfo info = {f[k].dis,
                           L[0],
                           L[1],
                           L[2],
                           0,
                           0,
                           0,
                           L[0],
                           L[1],
                           -1,
                           0,
                           0,
                           0,
                           0,
                           0,
                           f[k].infos[0]->level,
                           f[k].icode[1],
                           otherrank,
                           f[k].infos[0]->index[0],
                           f[k].infos[0]->index[1],
                           f[k].infos[0]->index[2],
                           f[k].infos[1]->blockID_2};
        if (f[k].CoarseStencil) {
          Synch_ptr->SM.CoarseStencilLength(f[k].icode[1], Lc);
          Vc = Lc[0] * Lc[1] * Lc[2];
          total_size += Vc;
          offsets_recv[otherrank] += Vc * nc;
          info.CoarseVersionOffset = V * nc;
          info.CoarseVersionLX = Lc[0];
          info.CoarseVersionLY = Lc[1];
        }
        offsets_recv[otherrank] += V * nc;
        Synch_ptr->myunpacks[f[k].infos[1]->halo_block_id].push_back(info);
        for (size_t kk = 0; kk < (*i).removedIndices.size(); kk++) {
          const int remEl1 = i->removedIndices[kk];
          Synch_ptr->SM.DetermineStencilLength(f[remEl1].infos[0]->level,
                                               f[remEl1].infos[1]->level,
                                               f[remEl1].icode[1], &L[0]);
          int srcx, srcy, srcz;
          Synch_ptr->SM.__FixDuplicates(f[k], f[remEl1], info.lx, info.ly,
                                        info.lz, L[0], L[1], L[2], srcx, srcy,
                                        srcz);
          int Csrcx = 0;
          int Csrcy = 0;
          int Csrcz = 0;
          if (f[k].CoarseStencil)
            Synch_ptr->SM.__FixDuplicates2(f[k], f[remEl1], Csrcx, Csrcy,
                                           Csrcz);
          Synch_ptr->myunpacks[f[remEl1].infos[1]->halo_block_id].push_back(
              {info.offset,
               L[0],
               L[1],
               L[2],
               srcx,
               srcy,
               srcz,
               info.LX,
               info.LY,
               info.CoarseVersionOffset,
               info.CoarseVersionLX,
               info.CoarseVersionLY,
               Csrcx,
               Csrcy,
               Csrcz,
               f[remEl1].infos[0]->level,
               f[remEl1].icode[1],
               otherrank,
               f[remEl1].infos[0]->index[0],
               f[remEl1].infos[0]->index[1],
               f[remEl1].infos[0]->index[2],
               f[remEl1].infos[1]->blockID_2});
          f[remEl1].dis = info.offset;
        }
      }
    }
  };
  bool UseCoarseStencil(const Interface &f) {
    Info &a = *f.infos[0];
    Info &b = *f.infos[1];
    if (a.level == 0 || (!use_averages))
      return false;
    int imin[3];
    int imax[3];
    const int aux = 1 << a.level;
    const bool periodic[3] = {grid->xperiodic, grid->yperiodic,
                              grid->zperiodic};
    const int blocks[3] = {grid->getMaxBlocks()[0] * aux - 1,
                           grid->getMaxBlocks()[1] * aux - 1,
                           grid->getMaxBlocks()[2] * aux - 1};
    for (int d = 0; d < 3; d++) {
      imin[d] = (a.index[d] < b.index[d]) ? 0 : -1;
      imax[d] = (a.index[d] > b.index[d]) ? 0 : +1;
      if (periodic[d]) {
        if (a.index[d] == 0 && b.index[d] == blocks[d])
          imin[d] = -1;
        if (b.index[d] == 0 && a.index[d] == blocks[d])
          imax[d] = +1;
      } else {
        if (a.index[d] == 0 && b.index[d] == 0)
          imin[d] = 0;
        if (a.index[d] == blocks[d] && b.index[d] == blocks[d])
          imax[d] = 0;
      }
    }
    bool retval = false;
    for (int i2 = imin[2]; i2 <= imax[2]; i2++)
      for (int i1 = imin[1]; i1 <= imax[1]; i1++)
        for (int i0 = imin[0]; i0 <= imax[0]; i0++) {
          if ((grid->Tree(a.level, a.Znei_(i0, i1, i2))).CheckCoarser()) {
            retval = true;
            break;
          }
        }
    return retval;
  }
  void AverageDownAndFill(Real *__restrict__ dst, const Info *const info,
                          const int code[3]) {
    const int s[3] = {code[0] < 1 ? (code[0] < 0 ? stencil.sx : 0) : nX,
                      code[1] < 1 ? (code[1] < 0 ? stencil.sy : 0) : nY,
                      code[2] < 1 ? (code[2] < 0 ? stencil.sz : 0) : nZ};
    const int e[3] = {
        code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + stencil.ex - 1,
        code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + stencil.ey - 1,
        code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + stencil.ez - 1};
    int pos = 0;
    const Real *src = (const Real *)(*info).block;
    const int xStep = (code[0] == 0) ? 2 : 1;
    const int yStep = (code[1] == 0) ? 2 : 1;
    const int zStep = (code[2] == 0) ? 2 : 1;
    if (gptfloats == 1) {
      for (int iz = s[2]; iz < e[2]; iz += zStep) {
        const int ZZ = (abs(code[2]) == 1)
                           ? 2 * (iz - code[2] * nZ) + std::min(0, code[2]) * nZ
                           : iz;
        for (int iy = s[1]; iy < e[1]; iy += yStep) {
          const int YY = (abs(code[1]) == 1) ? 2 * (iy - code[1] * nY) +
                                                   std::min(0, code[1]) * nY
                                             : iy;
          for (int ix = s[0]; ix < e[0]; ix += xStep) {
            const int XX = (abs(code[0]) == 1) ? 2 * (ix - code[0] * nX) +
                                                     std::min(0, code[0]) * nX
                                               : ix;
            dst[pos] = 0.125 * (src[XX + (YY + (ZZ)*nY) * nX] +
                                src[XX + (YY + (ZZ + 1) * nY) * nX] +
                                src[XX + (YY + 1 + (ZZ)*nY) * nX] +
                                src[XX + (YY + 1 + (ZZ + 1) * nY) * nX] +
                                src[XX + 1 + (YY + (ZZ)*nY) * nX] +
                                src[XX + 1 + (YY + (ZZ + 1) * nY) * nX] +
                                src[XX + 1 + (YY + 1 + (ZZ)*nY) * nX] +
                                src[XX + 1 + (YY + 1 + (ZZ + 1) * nY) * nX]);
            pos++;
          }
        }
      }
    } else {
      for (int iz = s[2]; iz < e[2]; iz += zStep) {
        const int ZZ = (abs(code[2]) == 1)
                           ? 2 * (iz - code[2] * nZ) + std::min(0, code[2]) * nZ
                           : iz;
        for (int iy = s[1]; iy < e[1]; iy += yStep) {
          const int YY = (abs(code[1]) == 1) ? 2 * (iy - code[1] * nY) +
                                                   std::min(0, code[1]) * nY
                                             : iy;
          for (int ix = s[0]; ix < e[0]; ix += xStep) {
            const int XX = (abs(code[0]) == 1) ? 2 * (ix - code[0] * nX) +
                                                     std::min(0, code[0]) * nX
                                               : ix;
            for (int c = 0; c < NC; c++) {
              int comp = stencil.selcomponents[c];
              dst[pos] =
                  0.125 *
                  ((*(src + gptfloats * ((XX) + ((YY) + (ZZ)*nY) * nX) +
                      comp)) +
                   (*(src + gptfloats * ((XX) + ((YY) + (ZZ + 1) * nY) * nX) +
                      comp)) +
                   (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ)*nY) * nX) +
                      comp)) +
                   (*(src +
                      gptfloats * ((XX) + ((YY + 1) + (ZZ + 1) * nY) * nX) +
                      comp)) +
                   (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ)*nY) * nX) +
                      comp)) +
                   (*(src +
                      gptfloats * ((XX + 1) + ((YY) + (ZZ + 1) * nY) * nX) +
                      comp)) +
                   (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ)*nY) * nX) +
                      comp)) +
                   (*(src +
                      gptfloats * ((XX + 1) + ((YY + 1) + (ZZ + 1) * nY) * nX) +
                      comp)));
              pos++;
            }
          }
        }
      }
    }
  }
  void AverageDownAndFill2(Real *dst, const Info *const info,
                           const int code[3]) {
    const int eC[3] = {(stencil.ex) / 2 + Cstencil.ex,
                       (stencil.ey) / 2 + Cstencil.ey,
                       (stencil.ez) / 2 + Cstencil.ez};
    const int sC[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                       (stencil.sy - 1) / 2 + Cstencil.sy,
                       (stencil.sz - 1) / 2 + Cstencil.sz};
    const int s[3] = {code[0] < 1 ? (code[0] < 0 ? sC[0] : 0) : nX / 2,
                      code[1] < 1 ? (code[1] < 0 ? sC[1] : 0) : nY / 2,
                      code[2] < 1 ? (code[2] < 0 ? sC[2] : 0) : nZ / 2};
    const int e[3] = {
        code[0] < 1 ? (code[0] < 0 ? 0 : nX / 2) : nX / 2 + eC[0] - 1,
        code[1] < 1 ? (code[1] < 0 ? 0 : nY / 2) : nY / 2 + eC[1] - 1,
        code[2] < 1 ? (code[2] < 0 ? 0 : nZ / 2) : nZ / 2 + eC[2] - 1};
    Real *src = (Real *)(*info).block;
    int pos = 0;
    for (int iz = s[2]; iz < e[2]; iz++) {
      const int ZZ = 2 * (iz - s[2]) + s[2] + std::max(code[2], 0) * nZ / 2 -
                     code[2] * nZ + std::min(0, code[2]) * (e[2] - s[2]);
      for (int iy = s[1]; iy < e[1]; iy++) {
        const int YY = 2 * (iy - s[1]) + s[1] + std::max(code[1], 0) * nY / 2 -
                       code[1] * nY + std::min(0, code[1]) * (e[1] - s[1]);
        for (int ix = s[0]; ix < e[0]; ix++) {
          const int XX = 2 * (ix - s[0]) + s[0] +
                         std::max(code[0], 0) * nX / 2 - code[0] * nX +
                         std::min(0, code[0]) * (e[0] - s[0]);
          for (int c = 0; c < NC; c++) {
            int comp = stencil.selcomponents[c];
            dst[pos] =
                0.125 *
                ((*(src + gptfloats * ((XX) + ((YY) + (ZZ)*nY) * nX) + comp)) +
                 (*(src + gptfloats * ((XX) + ((YY) + (ZZ + 1) * nY) * nX) +
                    comp)) +
                 (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ)*nY) * nX) +
                    comp)) +
                 (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ + 1) * nY) * nX) +
                    comp)) +
                 (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ)*nY) * nX) +
                    comp)) +
                 (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ + 1) * nY) * nX) +
                    comp)) +
                 (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ)*nY) * nX) +
                    comp)) +
                 (*(src +
                    gptfloats * ((XX + 1) + ((YY + 1) + (ZZ + 1) * nY) * nX) +
                    comp)));
            pos++;
          }
        }
      }
    }
  }
  std::string EncodeSet(const std::set<int> &ranks) {
    std::string retval;
    for (auto r : ranks) {
      std::stringstream ss;
      ss << std::setw(size) << std::setfill('0') << r;
      std::string s = ss.str();
      retval += s;
    }
    return retval;
  }

public:
  void _Setup() {
    Neighbors.clear();
    inner_blocks.clear();
    halo_blocks.clear();
    for (int r = 0; r < size; r++) {
      send_interfaces[r].clear();
      recv_interfaces[r].clear();
      send_buffer_size[r] = 0;
    }
    for (size_t i = 0; i < myunpacks.size(); i++)
      myunpacks[i].clear();
    myunpacks.clear();
    DuplicatesManager DM(*(this));
    for (Info &info : grid->getBlocksInfo()) {
      info.halo_block_id = -1;
      const bool xskin =
          info.index[0] == 0 ||
          info.index[0] == ((grid->getMaxBlocks()[0] << info.level) - 1);
      const bool yskin =
          info.index[1] == 0 ||
          info.index[1] == ((grid->getMaxBlocks()[1] << info.level) - 1);
      const bool zskin =
          info.index[2] == 0 ||
          info.index[2] == ((grid->getMaxBlocks()[2] << info.level) - 1);
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      bool isInner = true;
      std::vector<int> ToBeChecked;
      bool Coarsened = false;
      for (int icode = 0; icode < 27; icode++) {
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                             (icode / 9) % 3 - 1};
        if (!grid->xperiodic && code[0] == xskip && xskin)
          continue;
        if (!grid->yperiodic && code[1] == yskip && yskin)
          continue;
        if (!grid->zperiodic && code[2] == zskip && zskin)
          continue;
        const TreePosition &infoNeiTree =
            grid->Tree(info.level, info.Znei_(code[0], code[1], code[2]));
        if (infoNeiTree.Exists() && infoNeiTree.rank() != rank) {
          isInner = false;
          Neighbors.insert(infoNeiTree.rank());
          Info &infoNei = grid->getInfoAll(
              info.level, info.Znei_(code[0], code[1], code[2]));
          const int icode2 =
              (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
          send_interfaces[infoNeiTree.rank()].push_back(
              {info, infoNei, icode, icode2});
          recv_interfaces[infoNeiTree.rank()].push_back(
              {infoNei, info, icode2, icode});
          ToBeChecked.push_back(infoNeiTree.rank());
          ToBeChecked.push_back(
              (int)send_interfaces[infoNeiTree.rank()].size() - 1);
          ToBeChecked.push_back(
              (int)recv_interfaces[infoNeiTree.rank()].size() - 1);
          DM.Add(infoNeiTree.rank(),
                 (int)send_interfaces[infoNeiTree.rank()].size() - 1);
        } else if (infoNeiTree.CheckCoarser()) {
          Coarsened = true;
          Info &infoNei = grid->getInfoAll(
              info.level, info.Znei_(code[0], code[1], code[2]));
          const int infoNeiCoarserrank =
              grid->Tree(info.level - 1, infoNei.Zparent).rank();
          if (infoNeiCoarserrank != rank) {
            isInner = false;
            Neighbors.insert(infoNeiCoarserrank);
            Info &infoNeiCoarser =
                grid->getInfoAll(infoNei.level - 1, infoNei.Zparent);
            const int icode2 =
                (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
            const int Bmax[3] = {grid->getMaxBlocks()[0] << (info.level - 1),
                                 grid->getMaxBlocks()[1] << (info.level - 1),
                                 grid->getMaxBlocks()[2] << (info.level - 1)};
            const int test_idx[3] = {
                (infoNeiCoarser.index[0] - code[0] + Bmax[0]) % Bmax[0],
                (infoNeiCoarser.index[1] - code[1] + Bmax[1]) % Bmax[1],
                (infoNeiCoarser.index[2] - code[2] + Bmax[2]) % Bmax[2]};
            if (info.index[0] / 2 == test_idx[0] &&
                info.index[1] / 2 == test_idx[1] &&
                info.index[2] / 2 == test_idx[2]) {
              send_interfaces[infoNeiCoarserrank].push_back(
                  {info, infoNeiCoarser, icode, icode2});
              recv_interfaces[infoNeiCoarserrank].push_back(
                  {infoNeiCoarser, info, icode2, icode});
              DM.Add(infoNeiCoarserrank,
                     (int)send_interfaces[infoNeiCoarserrank].size() - 1);
              if (abs(code[0]) + abs(code[1]) + abs(code[2]) == 1) {
                const int d0 = abs(code[1] + 2 * code[2]);
                const int d1 = (d0 + 1) % 3;
                const int d2 = (d0 + 2) % 3;
                int code3[3];
                code3[d0] = code[d0];
                code3[d1] = -2 * (info.index[d1] % 2) + 1;
                code3[d2] = -2 * (info.index[d2] % 2) + 1;
                const int icode3 =
                    (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;
                int code4[3];
                code4[d0] = code[d0];
                code4[d1] = code3[d1];
                code4[d2] = 0;
                const int icode4 =
                    (code4[0] + 1) + (code4[1] + 1) * 3 + (code4[2] + 1) * 9;
                int code5[3];
                code5[d0] = code[d0];
                code5[d1] = 0;
                code5[d2] = code3[d2];
                const int icode5 =
                    (code5[0] + 1) + (code5[1] + 1) * 3 + (code5[2] + 1) * 9;
                recv_interfaces[infoNeiCoarserrank].push_back(
                    {infoNeiCoarser, info, icode2, icode3});
                recv_interfaces[infoNeiCoarserrank].push_back(
                    {infoNeiCoarser, info, icode2, icode4});
                recv_interfaces[infoNeiCoarserrank].push_back(
                    {infoNeiCoarser, info, icode2, icode5});
              } else if (abs(code[0]) + abs(code[1]) + abs(code[2]) == 2) {
                const int d0 = (1 - abs(code[1])) + 2 * (1 - abs(code[2]));
                const int d1 = (d0 + 1) % 3;
                const int d2 = (d0 + 2) % 3;
                int code3[3];
                code3[d0] = -2 * (info.index[d0] % 2) + 1;
                code3[d1] = code[d1];
                code3[d2] = code[d2];
                const int icode3 =
                    (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;
                recv_interfaces[infoNeiCoarserrank].push_back(
                    {infoNeiCoarser, info, icode2, icode3});
              }
            }
          }
        } else if (infoNeiTree.CheckFiner()) {
          Info &infoNei = grid->getInfoAll(
              info.level, info.Znei_(code[0], code[1], code[2]));
          int Bstep = 1;
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
            Bstep = 3;
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
            Bstep = 4;
          for (int B = 0; B <= 3; B += Bstep) {
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))]
                              [std::max(-code[2], 0) +
                               (B / 2) * std::max(0, 1 - abs(code[2]))];
            const int infoNeiFinerrank =
                grid->Tree(info.level + 1, nFine).rank();
            if (infoNeiFinerrank != rank) {
              isInner = false;
              Neighbors.insert(infoNeiFinerrank);
              Info &infoNeiFiner = grid->getInfoAll(info.level + 1, nFine);
              const int icode2 =
                  (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
              send_interfaces[infoNeiFinerrank].push_back(
                  {info, infoNeiFiner, icode, icode2});
              recv_interfaces[infoNeiFinerrank].push_back(
                  {infoNeiFiner, info, icode2, icode});
              DM.Add(infoNeiFinerrank,
                     (int)send_interfaces[infoNeiFinerrank].size() - 1);
              if (Bstep == 1) {
                const int d0 = abs(code[1] + 2 * code[2]);
                const int d1 = (d0 + 1) % 3;
                const int d2 = (d0 + 2) % 3;
                int code3[3];
                code3[d0] = -code[d0];
                code3[d1] = -2 * (infoNeiFiner.index[d1] % 2) + 1;
                code3[d2] = -2 * (infoNeiFiner.index[d2] % 2) + 1;
                const int icode3 =
                    (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;
                int code4[3];
                code4[d0] = -code[d0];
                code4[d1] = code3[d1];
                code4[d2] = 0;
                const int icode4 =
                    (code4[0] + 1) + (code4[1] + 1) * 3 + (code4[2] + 1) * 9;
                int code5[3];
                code5[d0] = -code[d0];
                code5[d1] = 0;
                code5[d2] = code3[d2];
                const int icode5 =
                    (code5[0] + 1) + (code5[1] + 1) * 3 + (code5[2] + 1) * 9;
                send_interfaces[infoNeiFinerrank].push_back(
                    {info, infoNeiFiner, icode, icode3});
                DM.Add(infoNeiFinerrank,
                       (int)send_interfaces[infoNeiFinerrank].size() - 1);
                send_interfaces[infoNeiFinerrank].push_back(
                    {info, infoNeiFiner, icode, icode4});
                DM.Add(infoNeiFinerrank,
                       (int)send_interfaces[infoNeiFinerrank].size() - 1);
                send_interfaces[infoNeiFinerrank].push_back(
                    {info, infoNeiFiner, icode, icode5});
                DM.Add(infoNeiFinerrank,
                       (int)send_interfaces[infoNeiFinerrank].size() - 1);
              } else if (Bstep == 3) {
                const int d0 = (1 - abs(code[1])) + 2 * (1 - abs(code[2]));
                const int d1 = (d0 + 1) % 3;
                const int d2 = (d0 + 2) % 3;
                int code3[3];
                code3[d0] = B == 0 ? 1 : -1;
                code3[d1] = -code[d1];
                code3[d2] = -code[d2];
                const int icode3 =
                    (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;
                send_interfaces[infoNeiFinerrank].push_back(
                    {info, infoNeiFiner, icode, icode3});
                DM.Add(infoNeiFinerrank,
                       (int)send_interfaces[infoNeiFinerrank].size() - 1);
              }
            }
          }
        }
      }
      if (isInner) {
        info.halo_block_id = -1;
        inner_blocks.push_back(&info);
      } else {
        info.halo_block_id = halo_blocks.size();
        halo_blocks.push_back(&info);
        if (Coarsened) {
          for (size_t j = 0; j < ToBeChecked.size(); j += 3) {
            const int r = ToBeChecked[j];
            const int send = ToBeChecked[j + 1];
            const int recv = ToBeChecked[j + 2];
            const bool tmp = UseCoarseStencil(send_interfaces[r][send]);
            send_interfaces[r][send].CoarseStencil = tmp;
            recv_interfaces[r][recv].CoarseStencil = tmp;
          }
        }
        for (int r = 0; r < size; r++)
          if (DM.sizes[r] > 0) {
            DM.RemoveDuplicates(r, send_interfaces[r].v, send_buffer_size[r]);
            DM.sizes[r] = 0;
          }
      }
      grid->getInfoAll(info.level, info.Z).halo_block_id = info.halo_block_id;
    }
    myunpacks.resize(halo_blocks.size());
    for (int r = 0; r < size; r++) {
      recv_buffer_size[r] = 0;
      std::sort(recv_interfaces[r].begin(), recv_interfaces[r].end());
      size_t counter = 0;
      while (counter < recv_interfaces[r].size()) {
        const long long ID = recv_interfaces[r][counter].infos[0]->blockID_2;
        const size_t start = counter;
        size_t finish = start + 1;
        counter++;
        size_t j;
        for (j = counter; j < recv_interfaces[r].size(); j++) {
          if (recv_interfaces[r][j].infos[0]->blockID_2 == ID)
            finish++;
          else
            break;
        }
        counter = j;
        DM.RemoveDuplicates_recv(recv_interfaces[r].v, recv_buffer_size[r], r,
                                 start, finish);
      }
      send_buffer[r].resize(send_buffer_size[r] * NC);
      recv_buffer[r].resize(recv_buffer_size[r] * NC);
      send_packinfos[r].clear();
      ToBeAveragedDown[r].clear();
      for (int i = 0; i < (int)send_interfaces[r].size(); i++) {
        const Interface &f = send_interfaces[r][i];
        if (!f.ToBeKept)
          continue;
        if (f.infos[0]->level <= f.infos[1]->level) {
          const MyRange &range = SM.DetermineStencil(f);
          send_packinfos[r].push_back(
              {(Real *)f.infos[0]->block, &send_buffer[r][f.dis], range.sx,
               range.sy, range.sz, range.ex, range.ey, range.ez});
          if (f.CoarseStencil) {
            const int V = (range.ex - range.sx) * (range.ey - range.sy) *
                          (range.ez - range.sz);
            ToBeAveragedDown[r].push_back(i);
            ToBeAveragedDown[r].push_back(f.dis + V * NC);
          }
        } else {
          ToBeAveragedDown[r].push_back(i);
          ToBeAveragedDown[r].push_back(f.dis);
        }
      }
    }
    mapofHaloBlockGroups.clear();
    for (auto &info : halo_blocks) {
      const int id = info->halo_block_id;
      UnPackInfo *unpacks = myunpacks[id].data();
      std::set<int> ranks;
      for (size_t jj = 0; jj < myunpacks[id].size(); jj++) {
        const UnPackInfo &unpack = unpacks[jj];
        ranks.insert(unpack.rank);
      }
      auto set_ID = EncodeSet(ranks);
      const auto retval = mapofHaloBlockGroups.find(set_ID);
      if (retval == mapofHaloBlockGroups.end()) {
        HaloBlockGroup temporary;
        temporary.myranks = ranks;
        temporary.myblocks.push_back(info);
        mapofHaloBlockGroups[set_ID] = temporary;
      } else {
        (retval->second).myblocks.push_back(info);
      }
    }
  }
  SynchronizerMPI_AMR(StencilInfo a_stencil, StencilInfo a_Cstencil,
                      TGrid *_grid)
      : stencil(a_stencil), Cstencil(a_Cstencil),
        SM(a_stencil, a_Cstencil, TGrid::Block::sizeX, TGrid::Block::sizeY,
           TGrid::Block::sizeZ),
        gptfloats(sizeof(typename TGrid::Block::ElementType) / sizeof(Real)),
        NC(a_stencil.selcomponents.size()) {
    grid = _grid;
    use_averages = (grid->FiniteDifferences == false || stencil.tensorial ||
                    stencil.sx < -2 || stencil.sy < -2 || stencil.sz < -2 ||
                    stencil.ex > 3 || stencil.ey > 3 || stencil.ez > 3);
    comm = grid->getWorldComm();
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    nX = TGrid::Block::sizeX;
    nY = TGrid::Block::sizeY;
    nZ = TGrid::Block::sizeZ;
    send_interfaces.resize(size);
    recv_interfaces.resize(size);
    send_packinfos.resize(size);
    send_buffer_size.resize(size);
    recv_buffer_size.resize(size);
    send_buffer.resize(size);
    recv_buffer.resize(size);
    ToBeAveragedDown.resize(size);
    std::sort(stencil.selcomponents.begin(), stencil.selcomponents.end());
    if (sizeof(Real) == sizeof(double)) {
      MPIREAL = MPI_DOUBLE;
    } else if (sizeof(Real) == sizeof(long double)) {
      MPIREAL = MPI_LONG_DOUBLE;
    } else {
      MPIREAL = MPI_FLOAT;
      assert(sizeof(Real) == sizeof(float));
    }
  }
  std::vector<Info *> &avail_inner() { return inner_blocks; }
  std::vector<Info *> &avail_halo() {
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    return halo_blocks;
  }
  std::vector<Info *> &avail_halo_nowait() { return halo_blocks; }
  std::vector<Info *> dummy_vector;
  std::vector<Info *> &avail_next() {
    bool done = false;
    auto it = mapofHaloBlockGroups.begin();
    while (done == false) {
      done = true;
      it = mapofHaloBlockGroups.begin();
      while (it != mapofHaloBlockGroups.end()) {
        if ((it->second).ready == false) {
          std::set<int> ranks = (it->second).myranks;
          int flag = 0;
          for (auto r : ranks) {
            const auto retval = mapofrequests.find(r);
            MPI_Test(retval->second, &flag, MPI_STATUS_IGNORE);
            if (flag == false)
              break;
          }
          if (flag == 1) {
            (it->second).ready = true;
            return (it->second).myblocks;
          }
        }
        done = done && (it->second).ready;
        it++;
      }
    }
    return dummy_vector;
  }
  void sync() {
    auto it = mapofHaloBlockGroups.begin();
    while (it != mapofHaloBlockGroups.end()) {
      (it->second).ready = false;
      it++;
    }
    const int timestamp = grid->getTimeStamp();
    mapofrequests.clear();
    requests.clear();
    requests.reserve(2 * size);
    for (auto r : Neighbors)
      if (recv_buffer_size[r] > 0) {
        requests.resize(requests.size() + 1);
        mapofrequests[r] = &requests.back();
        MPI_Irecv(&recv_buffer[r][0], recv_buffer_size[r] * NC, MPIREAL, r,
                  timestamp, comm, &requests.back());
      }
    for (int r = 0; r < size; r++)
      if (send_buffer_size[r] != 0) {
#pragma omp parallel
        {
#pragma omp for
          for (size_t j = 0; j < ToBeAveragedDown[r].size(); j += 2) {
            const int i = ToBeAveragedDown[r][j];
            const int d = ToBeAveragedDown[r][j + 1];
            const Interface &f = send_interfaces[r][i];
            const int code[3] = {-(f.icode[0] % 3 - 1),
                                 -((f.icode[0] / 3) % 3 - 1),
                                 -((f.icode[0] / 9) % 3 - 1)};
            if (f.CoarseStencil)
              AverageDownAndFill2(send_buffer[r].data() + d, f.infos[0], code);
            else
              AverageDownAndFill(send_buffer[r].data() + d, f.infos[0], code);
          }
#pragma omp for
          for (size_t i = 0; i < send_packinfos[r].size(); i++) {
            const PackInfo &info = send_packinfos[r][i];
            pack(info.block, info.pack, gptfloats,
                 &stencil.selcomponents.front(), NC, info.sx, info.sy, info.sz,
                 info.ex, info.ey, info.ez, nX, nY);
          }
        }
      }
    for (auto r : Neighbors)
      if (send_buffer_size[r] > 0) {
        requests.resize(requests.size() + 1);
        MPI_Isend(&send_buffer[r][0], send_buffer_size[r] * NC, MPIREAL, r,
                  timestamp, comm, &requests.back());
      }
  }
  const StencilInfo &getstencil() const { return stencil; }
  bool isready(const Info &info) {
    const int id = info.halo_block_id;
    if (id < 0)
      return true;
    UnPackInfo *unpacks = myunpacks[id].data();
    for (size_t jj = 0; jj < myunpacks[id].size(); jj++) {
      const UnPackInfo &unpack = unpacks[jj];
      const int otherrank = unpack.rank;
      int flag = 0;
      const auto retval = mapofrequests.find(otherrank);
      MPI_Test(retval->second, &flag, MPI_STATUS_IGNORE);
      if (flag == 0)
        return false;
    }
    return true;
  }
  void fetch(const Info &info, const unsigned int Length[3],
             const unsigned int CLength[3], Real *cacheBlock,
             Real *coarseBlock) {
    const int id = info.halo_block_id;
    if (id < 0)
      return;
    UnPackInfo *unpacks = myunpacks[id].data();
    for (size_t jj = 0; jj < myunpacks[id].size(); jj++) {
      const UnPackInfo &unpack = unpacks[jj];
      const int code[3] = {unpack.icode % 3 - 1, (unpack.icode / 3) % 3 - 1,
                           (unpack.icode / 9) % 3 - 1};
      const int otherrank = unpack.rank;
      const int s[3] = {code[0] < 1 ? (code[0] < 0 ? stencil.sx : 0) : nX,
                        code[1] < 1 ? (code[1] < 0 ? stencil.sy : 0) : nY,
                        code[2] < 1 ? (code[2] < 0 ? stencil.sz : 0) : nZ};
      const int e[3] = {
          code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + stencil.ex - 1,
          code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + stencil.ey - 1,
          code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + stencil.ez - 1};
      if (unpack.level == info.level) {
        Real *dst =
            cacheBlock + ((s[2] - stencil.sz) * Length[0] * Length[1] +
                          (s[1] - stencil.sy) * Length[0] + s[0] - stencil.sx) *
                             gptfloats;
        unpack_subregion(&recv_buffer[otherrank][unpack.offset], &dst[0],
                         gptfloats, &stencil.selcomponents[0],
                         stencil.selcomponents.size(), unpack.srcxstart,
                         unpack.srcystart, unpack.srczstart, unpack.LX,
                         unpack.LY, 0, 0, 0, unpack.lx, unpack.ly, unpack.lz,
                         Length[0], Length[1], Length[2]);
        if (unpack.CoarseVersionOffset >= 0) {
          const int offset[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                                 (stencil.sy - 1) / 2 + Cstencil.sy,
                                 (stencil.sz - 1) / 2 + Cstencil.sz};
          const int sC[3] = {
              code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : nX / 2,
              code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : nY / 2,
              code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : nZ / 2};
          Real *dst1 = coarseBlock +
                       ((sC[2] - offset[2]) * CLength[0] * CLength[1] +
                        (sC[1] - offset[1]) * CLength[0] + sC[0] - offset[0]) *
                           gptfloats;
          int L[3];
          SM.CoarseStencilLength(
              (-code[0] + 1) + 3 * (-code[1] + 1) + 9 * (-code[2] + 1), L);
          unpack_subregion(
              &recv_buffer[otherrank]
                          [unpack.offset + unpack.CoarseVersionOffset],
              &dst1[0], gptfloats, &stencil.selcomponents[0],
              stencil.selcomponents.size(), unpack.CoarseVersionsrcxstart,
              unpack.CoarseVersionsrcystart, unpack.CoarseVersionsrczstart,
              unpack.CoarseVersionLX, unpack.CoarseVersionLY, 0, 0, 0, L[0],
              L[1], L[2], CLength[0], CLength[1], CLength[2]);
        }
      } else if (unpack.level < info.level) {
        const int offset[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                               (stencil.sy - 1) / 2 + Cstencil.sy,
                               (stencil.sz - 1) / 2 + Cstencil.sz};
        const int sC[3] = {code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : nX / 2,
                           code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : nY / 2,
                           code[2] < 1 ? (code[2] < 0 ? offset[2] : 0)
                                       : nZ / 2};
        Real *dst = coarseBlock +
                    ((sC[2] - offset[2]) * CLength[0] * CLength[1] + sC[0] -
                     offset[0] + (sC[1] - offset[1]) * CLength[0]) *
                        gptfloats;
        unpack_subregion(&recv_buffer[otherrank][unpack.offset], &dst[0],
                         gptfloats, &stencil.selcomponents[0],
                         stencil.selcomponents.size(), unpack.srcxstart,
                         unpack.srcystart, unpack.srczstart, unpack.LX,
                         unpack.LY, 0, 0, 0, unpack.lx, unpack.ly, unpack.lz,
                         CLength[0], CLength[1], CLength[2]);
      } else {
        int B;
        if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
          B = 0;
        else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2)) {
          int t;
          if (code[0] == 0)
            t = unpack.index_0 - 2 * info.index[0];
          else if (code[1] == 0)
            t = unpack.index_1 - 2 * info.index[1];
          else
            t = unpack.index_2 - 2 * info.index[2];
          assert(t == 0 || t == 1);
          B = (t == 1) ? 3 : 0;
        } else {
          int Bmod, Bdiv;
          if (abs(code[0]) == 1) {
            Bmod = unpack.index_1 - 2 * info.index[1];
            Bdiv = unpack.index_2 - 2 * info.index[2];
          } else if (abs(code[1]) == 1) {
            Bmod = unpack.index_0 - 2 * info.index[0];
            Bdiv = unpack.index_2 - 2 * info.index[2];
          } else {
            Bmod = unpack.index_0 - 2 * info.index[0];
            Bdiv = unpack.index_1 - 2 * info.index[1];
          }
          B = 2 * Bdiv + Bmod;
        }
        const int aux1 = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
        Real *dst =
            cacheBlock +
            ((abs(code[2]) * (s[2] - stencil.sz) +
              (1 - abs(code[2])) *
                  (-stencil.sz + (B / 2) * (e[2] - s[2]) / 2)) *
                 Length[0] * Length[1] +
             (abs(code[1]) * (s[1] - stencil.sy) +
              (1 - abs(code[1])) * (-stencil.sy + aux1 * (e[1] - s[1]) / 2)) *
                 Length[0] +
             abs(code[0]) * (s[0] - stencil.sx) +
             (1 - abs(code[0])) * (-stencil.sx + (B % 2) * (e[0] - s[0]) / 2)) *
                gptfloats;
        unpack_subregion(&recv_buffer[otherrank][unpack.offset], &dst[0],
                         gptfloats, &stencil.selcomponents[0],
                         stencil.selcomponents.size(), unpack.srcxstart,
                         unpack.srcystart, unpack.srczstart, unpack.LX,
                         unpack.LY, 0, 0, 0, unpack.lx, unpack.ly, unpack.lz,
                         Length[0], Length[1], Length[2]);
      }
    }
  }
};
template <typename TFluxCorrection>
class FluxCorrectionMPI : public TFluxCorrection {
public:
  using TGrid = typename TFluxCorrection::GridType;
  typedef typename TFluxCorrection::ElementType ElementType;
  typedef typename TFluxCorrection::Real Real;
  typedef typename TFluxCorrection::BlockType BlockType;
  typedef BlockCase<BlockType> Case;
  int size;

protected:
  struct face {
    Info *infos[2];
    int icode[2];
    int offset;
    face(Info &i0, Info &i1, int a_icode0, int a_icode1) {
      infos[0] = &i0;
      infos[1] = &i1;
      icode[0] = a_icode0;
      icode[1] = a_icode1;
    }
    bool operator<(const face &other) const {
      if (infos[0]->blockID_2 == other.infos[0]->blockID_2) {
        return (icode[0] < other.icode[0]);
      } else {
        return (infos[0]->blockID_2 < other.infos[0]->blockID_2);
      }
    }
  };
  std::vector<std::vector<Real>> send_buffer;
  std::vector<std::vector<Real>> recv_buffer;
  std::vector<std::vector<face>> send_faces;
  std::vector<std::vector<face>> recv_faces;
  void FillCase(face &F) {
    Info &info = *F.infos[1];
    const int icode = F.icode[1];
    const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                         (icode / 9) % 3 - 1};
    const int myFace = abs(code[0]) * std::max(0, code[0]) +
                       abs(code[1]) * (std::max(0, code[1]) + 2) +
                       abs(code[2]) * (std::max(0, code[2]) + 4);
    std::array<long long, 2> temp = {(long long)info.level, info.Z};
    auto search = TFluxCorrection::MapOfCases.find(temp);
    assert(search != TFluxCorrection::MapOfCases.end());
    Case &CoarseCase = (*search->second);
    std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];
    for (int B = 0; B <= 3; B++) {
      const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
      const long long Z =
          (*TFluxCorrection::grid)
              .getZforward(info.level + 1,
                           2 * info.index[0] + std::max(code[0], 0) + code[0] +
                               (B % 2) * std::max(0, 1 - abs(code[0])),
                           2 * info.index[1] + std::max(code[1], 0) + code[1] +
                               aux * std::max(0, 1 - abs(code[1])),
                           2 * info.index[2] + std::max(code[2], 0) + code[2] +
                               (B / 2) * std::max(0, 1 - abs(code[2])));
      if (Z != F.infos[0]->Z)
        continue;
      const int d = myFace / 2;
      const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
      const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
      const int N1 = CoarseCase.m_vSize[d1];
      const int N2 = CoarseCase.m_vSize[d2];
      int base = 0;
      if (B == 1)
        base = (N2 / 2) + (0) * N2;
      else if (B == 2)
        base = (0) + (N1 / 2) * N2;
      else if (B == 3)
        base = (N2 / 2) + (N1 / 2) * N2;
      int r = (*TFluxCorrection::grid)
                  .Tree(F.infos[0]->level, F.infos[0]->Z)
                  .rank();
      int dis = 0;
      for (int i1 = 0; i1 < N1; i1 += 2)
        for (int i2 = 0; i2 < N2; i2 += 2) {
          for (int j = 0; j < ElementType::DIM; j++)
            CoarseFace[base + (i2 / 2) + (i1 / 2) * N2].member(j) +=
                recv_buffer[r][F.offset + dis + j];
          dis += ElementType::DIM;
        }
    }
  }
  void FillCase_2(face &F, int codex, int codey, int codez) {
    Info &info = *F.infos[1];
    const int icode = F.icode[1];
    const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                         (icode / 9) % 3 - 1};
    if (abs(code[0]) != codex)
      return;
    if (abs(code[1]) != codey)
      return;
    if (abs(code[2]) != codez)
      return;
    const int myFace = abs(code[0]) * std::max(0, code[0]) +
                       abs(code[1]) * (std::max(0, code[1]) + 2) +
                       abs(code[2]) * (std::max(0, code[2]) + 4);
    std::array<long long, 2> temp = {(long long)info.level, info.Z};
    auto search = TFluxCorrection::MapOfCases.find(temp);
    assert(search != TFluxCorrection::MapOfCases.end());
    Case &CoarseCase = (*search->second);
    std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];
    const int d = myFace / 2;
    const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
    const int N2 = CoarseCase.m_vSize[d2];
    BlockType &block = *(BlockType *)info.block;
    const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
    const int N1 = CoarseCase.m_vSize[d1];
    if (d == 0) {
      const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeX - 1;
      for (int i1 = 0; i1 < N1; i1++)
        for (int i2 = 0; i2 < N2; i2++) {
          block(j, i2, i1) += CoarseFace[i2 + i1 * N2];
          CoarseFace[i2 + i1 * N2].clear();
        }
    } else if (d == 1) {
      const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeY - 1;
      for (int i1 = 0; i1 < N1; i1++)
        for (int i2 = 0; i2 < N2; i2++) {
          block(i2, j, i1) += CoarseFace[i2 + i1 * N2];
          CoarseFace[i2 + i1 * N2].clear();
        }
    } else {
      const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeZ - 1;
      for (int i1 = 0; i1 < N1; i1++)
        for (int i2 = 0; i2 < N2; i2++) {
          block(i2, i1, j) += CoarseFace[i2 + i1 * N2];
          CoarseFace[i2 + i1 * N2].clear();
        }
    }
  }

public:
  virtual void prepare(TGrid &_grid) override {
    if (_grid.UpdateFluxCorrection == false)
      return;
    _grid.UpdateFluxCorrection = false;
    int temprank;
    MPI_Comm_size(_grid.getWorldComm(), &size);
    MPI_Comm_rank(_grid.getWorldComm(), &temprank);
    TFluxCorrection::rank = temprank;
    send_buffer.resize(size);
    recv_buffer.resize(size);
    send_faces.resize(size);
    recv_faces.resize(size);
    for (int r = 0; r < size; r++) {
      send_faces[r].clear();
      recv_faces[r].clear();
    }
    std::vector<int> send_buffer_size(size, 0);
    std::vector<int> recv_buffer_size(size, 0);
    const int NC = ElementType::DIM;
    int blocksize[3];
    blocksize[0] = BlockType::sizeX;
    blocksize[1] = BlockType::sizeY;
    blocksize[2] = BlockType::sizeZ;
    TFluxCorrection::Cases.clear();
    TFluxCorrection::MapOfCases.clear();
    TFluxCorrection::grid = &_grid;
    std::vector<Info> &BB = (*TFluxCorrection::grid).getBlocksInfo();
    std::array<int, 3> blocksPerDim = _grid.getMaxBlocks();
    std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1,
                                1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1,
                                1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};
    for (auto &info : BB) {
      (*TFluxCorrection::grid).getInfoAll(info.level, info.Z).auxiliary =
          nullptr;
      info.auxiliary = nullptr;
      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      bool storeFace[6] = {false, false, false, false, false, false};
      bool stored = false;
      for (int f = 0; f < 6; f++) {
        const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1,
                             (icode[f] / 9) % 3 - 1};
        if (!_grid.xperiodic && code[0] == xskip && xskin)
          continue;
        if (!_grid.yperiodic && code[1] == yskip && yskin)
          continue;
        if (!_grid.zperiodic && code[2] == zskip && zskin)
          continue;
        if (!(*TFluxCorrection::grid)
                 .Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                 .Exists()) {
          storeFace[abs(code[0]) * std::max(0, code[0]) +
                    abs(code[1]) * (std::max(0, code[1]) + 2) +
                    abs(code[2]) * (std::max(0, code[2]) + 4)] = true;
          stored = true;
        }
        int L[3];
        L[0] = (code[0] == 0) ? blocksize[0] / 2 : 1;
        L[1] = (code[1] == 0) ? blocksize[1] / 2 : 1;
        L[2] = (code[2] == 0) ? blocksize[2] / 2 : 1;
        int V = L[0] * L[1] * L[2];
        if ((*TFluxCorrection::grid)
                .Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                .CheckCoarser()) {
          Info &infoNei =
              (*TFluxCorrection::grid)
                  .getInfoAll(info.level,
                              info.Znei_(code[0], code[1], code[2]));
          const long long nCoarse = infoNei.Zparent;
          Info &infoNeiCoarser =
              (*TFluxCorrection::grid).getInfoAll(info.level - 1, nCoarse);
          const int infoNeiCoarserrank =
              (*TFluxCorrection::grid).Tree(info.level - 1, nCoarse).rank();
          {
            int code2[3] = {-code[0], -code[1], -code[2]};
            int icode2 =
                (code2[0] + 1) + (code2[1] + 1) * 3 + (code2[2] + 1) * 9;
            send_faces[infoNeiCoarserrank].push_back(
                face(info, infoNeiCoarser, icode[f], icode2));
            send_buffer_size[infoNeiCoarserrank] += V;
          }
        } else if ((*TFluxCorrection::grid)
                       .Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                       .CheckFiner()) {
          Info &infoNei =
              (*TFluxCorrection::grid)
                  .getInfoAll(info.level,
                              info.Znei_(code[0], code[1], code[2]));
          int Bstep = 1;
          for (int B = 0; B <= 3; B += Bstep) {
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))]
                              [std::max(-code[2], 0) +
                               (B / 2) * std::max(0, 1 - abs(code[2]))];
            const int infoNeiFinerrank =
                (*TFluxCorrection::grid).Tree(infoNei.level + 1, nFine).rank();
            {
              Info &infoNeiFiner =
                  (*TFluxCorrection::grid).getInfoAll(infoNei.level + 1, nFine);
              int icode2 =
                  (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
              recv_faces[infoNeiFinerrank].push_back(
                  face(infoNeiFiner, info, icode2, icode[f]));
              recv_buffer_size[infoNeiFinerrank] += V;
            }
          }
        }
      }
      if (stored) {
        TFluxCorrection::Cases.push_back(
            Case(storeFace, BlockType::sizeX, BlockType::sizeY,
                 BlockType::sizeZ, info.level, info.Z));
      }
    }
    size_t Cases_index = 0;
    if (TFluxCorrection::Cases.size() > 0)
      for (auto &info : BB) {
        if (Cases_index == TFluxCorrection::Cases.size())
          break;
        if (TFluxCorrection::Cases[Cases_index].level == info.level &&
            TFluxCorrection::Cases[Cases_index].Z == info.Z) {
          TFluxCorrection::MapOfCases.insert(
              std::pair<std::array<long long, 2>, Case *>(
                  {TFluxCorrection::Cases[Cases_index].level,
                   TFluxCorrection::Cases[Cases_index].Z},
                  &TFluxCorrection::Cases[Cases_index]));
          TFluxCorrection::grid
              ->getInfoAll(TFluxCorrection::Cases[Cases_index].level,
                           TFluxCorrection::Cases[Cases_index].Z)
              .auxiliary = &TFluxCorrection::Cases[Cases_index];
          info.auxiliary = &TFluxCorrection::Cases[Cases_index];
          Cases_index++;
        }
      }
    for (int r = 0; r < size; r++) {
      std::sort(send_faces[r].begin(), send_faces[r].end());
      std::sort(recv_faces[r].begin(), recv_faces[r].end());
    }
    for (int r = 0; r < size; r++) {
      send_buffer[r].resize(send_buffer_size[r] * NC);
      recv_buffer[r].resize(recv_buffer_size[r] * NC);
      int offset = 0;
      for (int k = 0; k < (int)recv_faces[r].size(); k++) {
        face &f = recv_faces[r][k];
        const int code[3] = {f.icode[1] % 3 - 1, (f.icode[1] / 3) % 3 - 1,
                             (f.icode[1] / 9) % 3 - 1};
        int L[3];
        L[0] = (code[0] == 0) ? blocksize[0] / 2 : 1;
        L[1] = (code[1] == 0) ? blocksize[1] / 2 : 1;
        L[2] = (code[2] == 0) ? blocksize[2] / 2 : 1;
        int V = L[0] * L[1] * L[2];
        f.offset = offset;
        offset += V * NC;
      }
    }
  }
  virtual void FillBlockCases() override {
    auto MPI_real =
        (sizeof(Real) == sizeof(float))
            ? MPI_FLOAT
            : ((sizeof(Real) == sizeof(double)) ? MPI_DOUBLE : MPI_LONG_DOUBLE);
    for (int r = 0; r < size; r++) {
      int displacement = 0;
      for (int k = 0; k < (int)send_faces[r].size(); k++) {
        face &f = send_faces[r][k];
        Info &info = *(f.infos[0]);
        auto search =
            TFluxCorrection::MapOfCases.find({(long long)info.level, info.Z});
        assert(search != TFluxCorrection::MapOfCases.end());
        Case &FineCase = (*search->second);
        int icode = f.icode[0];
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                             (icode / 9) % 3 - 1};
        const int myFace = abs(code[0]) * std::max(0, code[0]) +
                           abs(code[1]) * (std::max(0, code[1]) + 2) +
                           abs(code[2]) * (std::max(0, code[2]) + 4);
        std::vector<ElementType> &FineFace = FineCase.m_pData[myFace];
        const int d = myFace / 2;
        const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
        const int N2 = FineCase.m_vSize[d2];
        const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
        const int N1 = FineCase.m_vSize[d1];
        for (int i1 = 0; i1 < N1; i1 += 2)
          for (int i2 = 0; i2 < N2; i2 += 2) {
            ElementType avg =
                ((FineFace[i2 + i1 * N2] + FineFace[i2 + 1 + i1 * N2]) +
                 (FineFace[i2 + (i1 + 1) * N2] +
                  FineFace[i2 + 1 + (i1 + 1) * N2]));
            for (int j = 0; j < ElementType::DIM; j++)
              send_buffer[r][displacement + j] = avg.member(j);
            displacement += ElementType::DIM;
            FineFace[i2 + i1 * N2].clear();
            FineFace[i2 + 1 + i1 * N2].clear();
            FineFace[i2 + (i1 + 1) * N2].clear();
            FineFace[i2 + 1 + (i1 + 1) * N2].clear();
          }
      }
    }
    std::vector<MPI_Request> send_requests;
    std::vector<MPI_Request> recv_requests;
    const int me = TFluxCorrection::rank;
    for (int r = 0; r < size; r++)
      if (r != me) {
        if (recv_buffer[r].size() != 0) {
          MPI_Request req{};
          recv_requests.push_back(req);
          MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_real, r,
                    123456, (*TFluxCorrection::grid).getWorldComm(),
                    &recv_requests.back());
        }
        if (send_buffer[r].size() != 0) {
          MPI_Request req{};
          send_requests.push_back(req);
          MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_real, r,
                    123456, (*TFluxCorrection::grid).getWorldComm(),
                    &send_requests.back());
        }
      }
    MPI_Request me_send_request;
    MPI_Request me_recv_request;
    if (recv_buffer[me].size() != 0) {
      MPI_Irecv(&recv_buffer[me][0], recv_buffer[me].size(), MPI_real, me,
                123456, (*TFluxCorrection::grid).getWorldComm(),
                &me_recv_request);
    }
    if (send_buffer[me].size() != 0) {
      MPI_Isend(&send_buffer[me][0], send_buffer[me].size(), MPI_real, me,
                123456, (*TFluxCorrection::grid).getWorldComm(),
                &me_send_request);
    }
    if (recv_buffer[me].size() > 0)
      MPI_Waitall(1, &me_recv_request, MPI_STATUSES_IGNORE);
    if (send_buffer[me].size() > 0)
      MPI_Waitall(1, &me_send_request, MPI_STATUSES_IGNORE);
    for (int index = 0; index < (int)recv_faces[me].size(); index++)
      FillCase(recv_faces[me][index]);
    if (recv_requests.size() > 0)
      MPI_Waitall(recv_requests.size(), &recv_requests[0], MPI_STATUSES_IGNORE);
    for (int r = 0; r < size; r++)
      if (r != me)
        for (int index = 0; index < (int)recv_faces[r].size(); index++)
          FillCase(recv_faces[r][index]);
    for (int r = 0; r < size; r++)
      for (int index = 0; index < (int)recv_faces[r].size(); index++)
        FillCase_2(recv_faces[r][index], 1, 0, 0);
    for (int r = 0; r < size; r++)
      for (int index = 0; index < (int)recv_faces[r].size(); index++)
        FillCase_2(recv_faces[r][index], 0, 1, 0);
    for (int r = 0; r < size; r++)
      for (int index = 0; index < (int)recv_faces[r].size(); index++)
        FillCase_2(recv_faces[r][index], 0, 0, 1);
    if (send_requests.size() > 0)
      MPI_Waitall(send_requests.size(), &send_requests[0], MPI_STATUSES_IGNORE);
  }
};
template <typename TGrid> class GridMPI : public TGrid {
public:
  typedef typename TGrid::Real Real;
  typedef typename TGrid::BlockType Block;
  typedef typename TGrid::BlockType BlockType;
  typedef SynchronizerMPI_AMR<Real, GridMPI<TGrid>> SynchronizerMPIType;
  size_t timestamp;
  MPI_Comm worldcomm;
  int myrank;
  int world_size;
  std::map<StencilInfo, SynchronizerMPIType *> SynchronizerMPIs;
  FluxCorrectionMPI<FluxCorrection<GridMPI<TGrid>>> Corrector;
  std::vector<Info *> boundary;
  GridMPI(const int nX, const int nY = 1, const int nZ = 1,
          const double a_maxextent = 1, const int a_levelStart = 0,
          const int a_levelMax = 1, const MPI_Comm comm = MPI_COMM_WORLD,
          const bool a_xperiodic = true, const bool a_yperiodic = true,
          const bool a_zperiodic = true)
      : TGrid(nX, nY, nZ, a_maxextent, a_levelStart, a_levelMax, false,
              a_xperiodic, a_yperiodic, a_zperiodic),
        timestamp(0), worldcomm(comm) {
    MPI_Comm_size(worldcomm, &world_size);
    MPI_Comm_rank(worldcomm, &myrank);
    const long long total_blocks =
        nX * nY * nZ * pow(pow(2, a_levelStart), DIMENSION);
    long long my_blocks = total_blocks / world_size;
    if ((long long)myrank < total_blocks % world_size)
      my_blocks++;
    long long n_start = myrank * (total_blocks / world_size);
    if (total_blocks % world_size > 0) {
      if ((long long)myrank < total_blocks % world_size)
        n_start += myrank;
      else
        n_start += total_blocks % world_size;
    }
    std::vector<short int> levels(my_blocks, a_levelStart);
    std::vector<long long> Zs(my_blocks);
    for (long long n = n_start; n < n_start + my_blocks; n++)
      Zs[n - n_start] = n;
    initialize_blocks(Zs, levels);
    MPI_Barrier(worldcomm);
  }
  virtual ~GridMPI() override {
    for (auto it = SynchronizerMPIs.begin(); it != SynchronizerMPIs.end(); ++it)
      delete it->second;
    SynchronizerMPIs.clear();
    MPI_Barrier(worldcomm);
  }
  virtual Block *avail(const int m, const long long n) override {
    return (TGrid::Tree(m, n).rank() == myrank)
               ? (Block *)TGrid::getInfoAll(m, n).block
               : nullptr;
  }
  virtual void UpdateBoundary(bool clean = false) override {
    const auto blocksPerDim = TGrid::getMaxBlocks();
    int rank, size;
    MPI_Comm_rank(worldcomm, &rank);
    MPI_Comm_size(worldcomm, &size);
    std::vector<std::vector<long long>> send_buffer(size);
    std::vector<Info *> &bbb = boundary;
    std::set<int> Neighbors;
    for (size_t jjj = 0; jjj < bbb.size(); jjj++) {
      Info &info = *bbb[jjj];
      std::set<int> receivers;
      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      for (int icode = 0; icode < 27; icode++) {
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                             (icode / 9) % 3 - 1};
        if (!TGrid::xperiodic && code[0] == xskip && xskin)
          continue;
        if (!TGrid::yperiodic && code[1] == yskip && yskin)
          continue;
        if (!TGrid::zperiodic && code[2] == zskip && zskin)
          continue;
        Info &infoNei = TGrid::getInfoAll(
            info.level, info.Znei_(code[0], code[1], code[2]));
        const TreePosition &infoNeiTree = TGrid::Tree(infoNei.level, infoNei.Z);
        if (infoNeiTree.Exists() && infoNeiTree.rank() != rank) {
          if (infoNei.state != Refine || clean)
            infoNei.state = Leave;
          receivers.insert(infoNeiTree.rank());
          Neighbors.insert(infoNeiTree.rank());
        } else if (infoNeiTree.CheckCoarser()) {
          const long long nCoarse = infoNei.Zparent;
          Info &infoNeiCoarser = TGrid::getInfoAll(infoNei.level - 1, nCoarse);
          const int infoNeiCoarserrank =
              TGrid::Tree(infoNei.level - 1, nCoarse).rank();
          if (infoNeiCoarserrank != rank) {
            assert(infoNeiCoarserrank >= 0);
            if (infoNeiCoarser.state != Refine || clean)
              infoNeiCoarser.state = Leave;
            receivers.insert(infoNeiCoarserrank);
            Neighbors.insert(infoNeiCoarserrank);
          }
        } else if (infoNeiTree.CheckFiner()) {
          int Bstep = 1;
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
            Bstep = 3;
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
            Bstep = 4;
          for (int B = 0; B <= 3; B += Bstep) {
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))]
                              [std::max(-code[2], 0) +
                               (B / 2) * std::max(0, 1 - abs(code[2]))];
            Info &infoNeiFiner = TGrid::getInfoAll(infoNei.level + 1, nFine);
            const int infoNeiFinerrank =
                TGrid::Tree(infoNei.level + 1, nFine).rank();
            if (infoNeiFinerrank != rank) {
              if (infoNeiFiner.state != Refine || clean)
                infoNeiFiner.state = Leave;
              receivers.insert(infoNeiFinerrank);
              Neighbors.insert(infoNeiFinerrank);
            }
          }
        }
      }
      if (info.changed2 && info.state != Leave) {
        if (info.state == Refine)
          info.changed2 = false;
        std::set<int>::iterator it = receivers.begin();
        while (it != receivers.end()) {
          int temp = (info.state == Compress) ? 1 : 2;
          send_buffer[*it].push_back(info.level);
          send_buffer[*it].push_back(info.Z);
          send_buffer[*it].push_back(temp);
          it++;
        }
      }
    }
    std::vector<MPI_Request> requests;
    long long dummy = 0;
    for (int r : Neighbors)
      if (r != rank) {
        requests.resize(requests.size() + 1);
        if (send_buffer[r].size() != 0)
          MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_LONG_LONG, r,
                    123, worldcomm, &requests[requests.size() - 1]);
        else {
          MPI_Isend(&dummy, 1, MPI_LONG_LONG, r, 123, worldcomm,
                    &requests[requests.size() - 1]);
        }
      }
    std::vector<std::vector<long long>> recv_buffer(size);
    for (int r : Neighbors)
      if (r != rank) {
        int recv_size;
        MPI_Status status;
        MPI_Probe(r, 123, worldcomm, &status);
        MPI_Get_count(&status, MPI_LONG_LONG, &recv_size);
        if (recv_size > 0) {
          recv_buffer[r].resize(recv_size);
          requests.resize(requests.size() + 1);
          MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_LONG_LONG, r,
                    123, worldcomm, &requests[requests.size() - 1]);
        }
      }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    for (int r = 0; r < size; r++)
      if (recv_buffer[r].size() > 1)
        for (int index = 0; index < (int)recv_buffer[r].size(); index += 3) {
          int level = recv_buffer[r][index];
          long long Z = recv_buffer[r][index + 1];
          TGrid::getInfoAll(level, Z).state =
              (recv_buffer[r][index + 2] == 1) ? Compress : Refine;
        }
  };
  void UpdateInfoAll_States(bool UpdateIDs = false) {
    std::vector<int> myNeighbors = FindMyNeighbors();
    const auto blocksPerDim = TGrid::getMaxBlocks();
    std::vector<long long> myData;
    for (auto &info : TGrid::m_vInfo) {
      bool myflag = false;
      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      for (int icode = 0; icode < 27; icode++) {
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                             (icode / 9) % 3 - 1};
        if (!TGrid::xperiodic && code[0] == xskip && xskin)
          continue;
        if (!TGrid::yperiodic && code[1] == yskip && yskin)
          continue;
        if (!TGrid::zperiodic && code[2] == zskip && zskin)
          continue;
        Info &infoNei = TGrid::getInfoAll(
            info.level, info.Znei_(code[0], code[1], code[2]));
        const TreePosition &infoNeiTree = TGrid::Tree(infoNei.level, infoNei.Z);
        if (infoNeiTree.Exists() && infoNeiTree.rank() != myrank) {
          myflag = true;
          break;
        } else if (infoNeiTree.CheckCoarser()) {
          long long nCoarse = infoNei.Zparent;
          const int infoNeiCoarserrank =
              TGrid::Tree(infoNei.level - 1, nCoarse).rank();
          if (infoNeiCoarserrank != myrank) {
            myflag = true;
            break;
          }
        } else if (infoNeiTree.CheckFiner()) {
          int Bstep = 1;
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
            Bstep = 3;
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
            Bstep = 4;
          for (int B = 0; B <= 3; B += Bstep) {
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))]
                              [std::max(-code[2], 0) +
                               (B / 2) * std::max(0, 1 - abs(code[2]))];
            const int infoNeiFinerrank =
                TGrid::Tree(infoNei.level + 1, nFine).rank();
            if (infoNeiFinerrank != myrank) {
              myflag = true;
              break;
            }
          }
        } else if (infoNeiTree.rank() < 0) {
          myflag = true;
          break;
        }
      }
      if (myflag) {
        myData.push_back(info.level);
        myData.push_back(info.Z);
        if (UpdateIDs)
          myData.push_back(info.blockID);
      }
    }
    std::vector<std::vector<long long>> recv_buffer(myNeighbors.size());
    std::vector<std::vector<long long>> send_buffer(myNeighbors.size());
    std::vector<int> recv_size(myNeighbors.size());
    std::vector<MPI_Request> size_requests(2 * myNeighbors.size());
    int mysize = (int)myData.size();
    int kk = 0;
    for (auto r : myNeighbors) {
      MPI_Irecv(&recv_size[kk], 1, MPI_INT, r, timestamp, worldcomm,
                &size_requests[2 * kk]);
      MPI_Isend(&mysize, 1, MPI_INT, r, timestamp, worldcomm,
                &size_requests[2 * kk + 1]);
      kk++;
    }
    kk = 0;
    for (size_t j = 0; j < myNeighbors.size(); j++) {
      send_buffer[kk].resize(myData.size());
      for (size_t i = 0; i < myData.size(); i++)
        send_buffer[kk][i] = myData[i];
      kk++;
    }
    MPI_Waitall(size_requests.size(), size_requests.data(),
                MPI_STATUSES_IGNORE);
    std::vector<MPI_Request> requests(2 * myNeighbors.size());
    kk = 0;
    for (auto r : myNeighbors) {
      recv_buffer[kk].resize(recv_size[kk]);
      MPI_Irecv(recv_buffer[kk].data(), recv_buffer[kk].size(), MPI_LONG_LONG,
                r, timestamp, worldcomm, &requests[2 * kk]);
      MPI_Isend(send_buffer[kk].data(), send_buffer[kk].size(), MPI_LONG_LONG,
                r, timestamp, worldcomm, &requests[2 * kk + 1]);
      kk++;
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    kk = -1;
    const int increment = UpdateIDs ? 3 : 2;
    for (auto r : myNeighbors) {
      kk++;
      for (size_t index__ = 0; index__ < recv_buffer[kk].size();
           index__ += increment) {
        const int level = (int)recv_buffer[kk][index__];
        const long long Z = recv_buffer[kk][index__ + 1];
        TGrid::Tree(level, Z).setrank(r);
        if (UpdateIDs)
          TGrid::getInfoAll(level, Z).blockID = recv_buffer[kk][index__ + 2];
        int p[3];
        Info::inverse(Z, level, p[0], p[1], p[2]);
        if (level < TGrid::levelMax - 1)
          for (int k = 0; k < 2; k++)
            for (int j = 0; j < 2; j++)
              for (int i = 0; i < 2; i++) {
                const long long nc = TGrid::getZforward(
                    level + 1, 2 * p[0] + i, 2 * p[1] + j, 2 * p[2] + k);
                TGrid::Tree(level + 1, nc).setCheckCoarser();
              }
        if (level > 0) {
          const long long nf =
              TGrid::getZforward(level - 1, p[0] / 2, p[1] / 2, p[2] / 2);
          TGrid::Tree(level - 1, nf).setCheckFiner();
        }
      }
    }
  }
  std::vector<int> FindMyNeighbors() {
    std::vector<int> myNeighbors;
    double low[3] = {+1e20, +1e20, +1e20};
    double high[3] = {-1e20, -1e20, -1e20};
    double p_low[3];
    double p_high[3];
    for (auto &info : TGrid::m_vInfo) {
      const double h = 2 * info.h;
      info.pos(p_low, 0, 0, 0);
      info.pos(p_high, Block::sizeX - 1, Block::sizeY - 1, Block::sizeZ - 1);
      p_low[0] -= h;
      p_low[1] -= h;
      p_low[2] -= h;
      p_high[0] += h;
      p_high[1] += h;
      p_high[2] += h;
      low[0] = std::min(low[0], p_low[0]);
      low[1] = std::min(low[1], p_low[1]);
      low[2] = std::min(low[2], p_low[2]);
      high[0] = std::max(high[0], p_high[0]);
      high[1] = std::max(high[1], p_high[1]);
      high[2] = std::max(high[2], p_high[2]);
    }
    std::vector<double> all_boxes(world_size * 6);
    double my_box[6] = {low[0], low[1], low[2], high[0], high[1], high[2]};
    MPI_Allgather(my_box, 6, MPI_DOUBLE, all_boxes.data(), 6, MPI_DOUBLE,
                  worldcomm);
    for (int i = 0; i < world_size; i++) {
      if (i == myrank)
        continue;
      if (Intersect(low, high, &all_boxes[i * 6], &all_boxes[i * 6 + 3]))
        myNeighbors.push_back(i);
    }
    return myNeighbors;
  }
  bool Intersect(double *l1, double *h1, double *l2, double *h2) {
    const double h0 =
        (TGrid::maxextent / std::max(TGrid::NX * Block::sizeX,
                                     std::max(TGrid::NY * Block::sizeY,
                                              TGrid::NZ * Block::sizeZ)));
    const double extent[3] = {TGrid::NX * Block::sizeX * h0,
                              TGrid::NY * Block::sizeY * h0,
                              TGrid::NZ * Block::sizeZ * h0};
    const Real intersect[3][2] = {
        {std::max(l1[0], l2[0]), std::min(h1[0], h2[0])},
        {std::max(l1[1], l2[1]), std::min(h1[1], h2[1])},
        {std::max(l1[2], l2[2]), std::min(h1[2], h2[2])}};
    bool intersection[3];
    intersection[0] = intersect[0][1] - intersect[0][0] > 0.0;
    intersection[1] = intersect[1][1] - intersect[1][0] > 0.0;
    intersection[2] =
        DIMENSION == 3 ? (intersect[2][1] - intersect[2][0] > 0.0) : true;
    const bool isperiodic[3] = {TGrid::xperiodic, TGrid::yperiodic,
                                TGrid::zperiodic};
    for (int d = 0; d < DIMENSION; d++) {
      if (isperiodic[d]) {
        if (h2[d] > extent[d])
          intersection[d] = std::min(h1[d], h2[d] - extent[d]) -
                            std::max(l1[d], l2[d] - extent[d]);
        else if (h1[d] > extent[d])
          intersection[d] = std::min(h2[d], h1[d] - extent[d]) -
                            std::max(l2[d], l1[d] - extent[d]);
      }
      if (!intersection[d])
        return false;
    }
    return true;
  }
  SynchronizerMPIType *sync(const StencilInfo &stencil) {
    assert(stencil.isvalid());
    StencilInfo Cstencil(-1, -1, DIMENSION == 3 ? -1 : 0, 2, 2,
                         DIMENSION == 3 ? 2 : 1, true, stencil.selcomponents);
    SynchronizerMPIType *queryresult = nullptr;
    typename std::map<StencilInfo, SynchronizerMPIType *>::iterator
        itSynchronizerMPI = SynchronizerMPIs.find(stencil);
    if (itSynchronizerMPI == SynchronizerMPIs.end()) {
      queryresult = new SynchronizerMPIType(stencil, Cstencil, this);
      queryresult->_Setup();
      SynchronizerMPIs[stencil] = queryresult;
    } else {
      queryresult = itSynchronizerMPI->second;
    }
    queryresult->sync();
    timestamp = (timestamp + 1) % 32768;
    return queryresult;
  }
  virtual void
  initialize_blocks(const std::vector<long long> &blocksZ,
                    const std::vector<short int> &blockslevel) override {
    TGrid::initialize_blocks(blocksZ, blockslevel);
    UpdateInfoAll_States(false);
    for (auto it = SynchronizerMPIs.begin(); it != SynchronizerMPIs.end(); ++it)
      (*it->second)._Setup();
  }
  virtual int rank() const override { return myrank; }
  size_t getTimeStamp() const { return timestamp; }
  MPI_Comm getWorldComm() const { return worldcomm; }
  virtual int get_world_size() const override { return world_size; }
};
template <class DataType, template <typename T> class allocator>
class Matrix3D {
private:
  DataType *m_pData{nullptr};
  unsigned int m_vSize[3]{0, 0, 0};
  unsigned int m_nElements{0};
  unsigned int m_nElementsPerSlice{0};

public:
  void _Release() {
    if (m_pData != nullptr) {
      free(m_pData);
      m_pData = nullptr;
    }
  }
  void _Setup(unsigned int nSizeX, unsigned int nSizeY, unsigned int nSizeZ) {
    _Release();
    m_vSize[0] = nSizeX;
    m_vSize[1] = nSizeY;
    m_vSize[2] = nSizeZ;
    m_nElementsPerSlice = nSizeX * nSizeY;
    m_nElements = nSizeX * nSizeY * nSizeZ;
    posix_memalign((void **)&m_pData, std::max(8, CUBISM_ALIGNMENT),
                   sizeof(DataType) * m_nElements);
    assert(m_pData != nullptr);
  }
  ~Matrix3D() { _Release(); }
  Matrix3D(unsigned int nSizeX, unsigned int nSizeY, unsigned int nSizeZ)
      : m_pData(nullptr), m_nElements(0), m_nElementsPerSlice(0) {
    _Setup(nSizeX, nSizeY, nSizeZ);
  }
  Matrix3D() : m_pData(nullptr), m_nElements(-1), m_nElementsPerSlice(-1) {}
  Matrix3D(const Matrix3D &m) = delete;
  Matrix3D(Matrix3D &&m)
      : m_pData{m.m_pData}, m_vSize{m.m_vSize[0], m.m_vSize[1], m.m_vSize[2]},
        m_nElements{m.m_nElements}, m_nElementsPerSlice{m.m_nElementsPerSlice} {
    m.m_pData = nullptr;
  }
  inline Matrix3D &operator=(const Matrix3D &m) {
#ifndef NDEBUG
    assert(m_vSize[0] == m.m_vSize[0]);
    assert(m_vSize[1] == m.m_vSize[1]);
    assert(m_vSize[2] == m.m_vSize[2]);
#endif
    for (unsigned int i = 0; i < m_nElements; i++)
      m_pData[i] = m.m_pData[i];
    return *this;
  }
  inline Matrix3D &operator=(DataType d) {
    for (unsigned int i = 0; i < m_nElements; i++)
      m_pData[i] = d;
    return *this;
  }
  inline Matrix3D &operator=(const double a) {
    for (unsigned int i = 0; i < m_nElements; i++)
      m_pData[i].set(a);
    return *this;
  }
  inline DataType &Access(unsigned int ix, unsigned int iy,
                          unsigned int iz) const {
#ifndef NDEBUG
    assert(ix < m_vSize[0]);
    assert(iy < m_vSize[1]);
    assert(iz < m_vSize[2]);
#endif
    return m_pData[iz * m_nElementsPerSlice + iy * m_vSize[0] + ix];
  }
  inline const DataType &Read(unsigned int ix, unsigned int iy,
                              unsigned int iz) const {
#ifndef NDEBUG
    assert(ix < m_vSize[0]);
    assert(iy < m_vSize[1]);
    assert(iz < m_vSize[2]);
#endif
    return m_pData[iz * m_nElementsPerSlice + iy * m_vSize[0] + ix];
  }
  inline DataType &LinAccess(unsigned int i) const {
#ifndef NDEBUG
    assert(i < m_nElements);
#endif
    return m_pData[i];
  }
  inline unsigned int getNumberOfElements() const { return m_nElements; }
  inline unsigned int getNumberOfElementsPerSlice() const {
    return m_nElementsPerSlice;
  }
  inline unsigned int *getSize() const { return (unsigned int *)m_vSize; }
  inline unsigned int getSize(int dim) const { return m_vSize[dim]; }
};
#define memcpy2(a, b, c) memcpy((a), (b), (c))
constexpr int default_start[3] = {-1, -1, -1};
constexpr int default_end[3] = {2, 2, 2};
template <typename TGrid,
          template <typename X> class allocator = std::allocator>
class BlockLab {
public:
  using GridType = TGrid;
  using BlockType = typename GridType::BlockType;
  using ElementType = typename BlockType::ElementType;
  using Real = typename ElementType::RealType;

protected:
  Matrix3D<ElementType, allocator> *m_cacheBlock;
  int m_stencilStart[3];
  int m_stencilEnd[3];
  bool istensorial;
  bool use_averages;
  GridType *m_refGrid;
  int NX;
  int NY;
  int NZ;
  std::array<BlockType *, 27> myblocks;
  std::array<int, 27> coarsened_nei_codes;
  int coarsened_nei_codes_size;
  int offset[3];
  Matrix3D<ElementType, allocator> *m_CoarsenedBlock;
  int m_InterpStencilStart[3];
  int m_InterpStencilEnd[3];
  bool coarsened;
  int CoarseBlockSize[3];
  const double d_coef_plus[9] = {-0.09375, 0.4375,   0.15625, 0.15625, -0.5625,
                                 0.90625,  -0.09375, 0.4375,  0.15625};
  const double d_coef_minus[9] = {0.15625, -0.5625, 0.90625, -0.09375, 0.4375,
                                  0.15625, 0.15625, 0.4375,  -0.09375};

public:
  BlockLab()
      : m_cacheBlock(nullptr), m_refGrid(nullptr), m_CoarsenedBlock(nullptr) {
    m_stencilStart[0] = m_stencilStart[1] = m_stencilStart[2] = 0;
    m_stencilEnd[0] = m_stencilEnd[1] = m_stencilEnd[2] = 0;
    m_InterpStencilStart[0] = m_InterpStencilStart[1] =
        m_InterpStencilStart[2] = 0;
    m_InterpStencilEnd[0] = m_InterpStencilEnd[1] = m_InterpStencilEnd[2] = 0;
    CoarseBlockSize[0] = (int)BlockType::sizeX / 2;
    CoarseBlockSize[1] = (int)BlockType::sizeY / 2;
    CoarseBlockSize[2] = (int)BlockType::sizeZ / 2;
    if (CoarseBlockSize[0] == 0)
      CoarseBlockSize[0] = 1;
    if (CoarseBlockSize[1] == 0)
      CoarseBlockSize[1] = 1;
    if (CoarseBlockSize[2] == 0)
      CoarseBlockSize[2] = 1;
  }
  virtual std::string name() const { return "BlockLab"; }
  virtual bool is_xperiodic() { return true; }
  virtual bool is_yperiodic() { return true; }
  virtual bool is_zperiodic() { return true; }
  ~BlockLab() {
    _release(m_cacheBlock);
    _release(m_CoarsenedBlock);
  }
  ElementType &operator()(int ix, int iy = 0, int iz = 0) {
    assert(ix - m_stencilStart[0] >= 0 &&
           ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
    assert(iy - m_stencilStart[1] >= 0 &&
           iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
    assert(iz - m_stencilStart[2] >= 0 &&
           iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
    return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],
                                iz - m_stencilStart[2]);
  }
  const ElementType &operator()(int ix, int iy = 0, int iz = 0) const {
    assert(ix - m_stencilStart[0] >= 0 &&
           ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
    assert(iy - m_stencilStart[1] >= 0 &&
           iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
    assert(iz - m_stencilStart[2] >= 0 &&
           iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
    return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],
                                iz - m_stencilStart[2]);
  }
  const ElementType &read(int ix, int iy = 0, int iz = 0) const {
    assert(ix - m_stencilStart[0] >= 0 &&
           ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
    assert(iy - m_stencilStart[1] >= 0 &&
           iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
    assert(iz - m_stencilStart[2] >= 0 &&
           iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
    return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],
                                iz - m_stencilStart[2]);
  }
  void release() {
    _release(m_cacheBlock);
    _release(m_CoarsenedBlock);
  }
  virtual void prepare(GridType &grid, const StencilInfo &stencil,
                       const int Istencil_start[3] = default_start,
                       const int Istencil_end[3] = default_end) {
    istensorial = stencil.tensorial;
    coarsened = false;
    m_stencilStart[0] = stencil.sx;
    m_stencilStart[1] = stencil.sy;
    m_stencilStart[2] = stencil.sz;
    m_stencilEnd[0] = stencil.ex;
    m_stencilEnd[1] = stencil.ey;
    m_stencilEnd[2] = stencil.ez;
    m_InterpStencilStart[0] = Istencil_start[0];
    m_InterpStencilStart[1] = Istencil_start[1];
    m_InterpStencilStart[2] = Istencil_start[2];
    m_InterpStencilEnd[0] = Istencil_end[0];
    m_InterpStencilEnd[1] = Istencil_end[1];
    m_InterpStencilEnd[2] = Istencil_end[2];
    assert(m_InterpStencilStart[0] <= m_InterpStencilEnd[0]);
    assert(m_InterpStencilStart[1] <= m_InterpStencilEnd[1]);
    assert(m_InterpStencilStart[2] <= m_InterpStencilEnd[2]);
    assert(stencil.sx <= stencil.ex);
    assert(stencil.sy <= stencil.ey);
    assert(stencil.sz <= stencil.ez);
    assert(stencil.sx >= -BlockType::sizeX);
    assert(stencil.sy >= -BlockType::sizeY);
    assert(stencil.sz >= -BlockType::sizeZ);
    assert(stencil.ex < 2 * BlockType::sizeX);
    assert(stencil.ey < 2 * BlockType::sizeY);
    assert(stencil.ez < 2 * BlockType::sizeZ);
    m_refGrid = &grid;
    if (m_cacheBlock == NULL ||
        (int)m_cacheBlock->getSize()[0] !=
            (int)BlockType::sizeX + m_stencilEnd[0] - m_stencilStart[0] - 1 ||
        (int)m_cacheBlock->getSize()[1] !=
            (int)BlockType::sizeY + m_stencilEnd[1] - m_stencilStart[1] - 1 ||
        (int)m_cacheBlock->getSize()[2] !=
            (int)BlockType::sizeZ + m_stencilEnd[2] - m_stencilStart[2] - 1) {
      if (m_cacheBlock != NULL)
        _release(m_cacheBlock);
      m_cacheBlock = allocator<Matrix3D<ElementType, allocator>>().allocate(1);
      allocator<Matrix3D<ElementType, allocator>>().construct(m_cacheBlock);
      m_cacheBlock->_Setup(
          BlockType::sizeX + m_stencilEnd[0] - m_stencilStart[0] - 1,
          BlockType::sizeY + m_stencilEnd[1] - m_stencilStart[1] - 1,
          BlockType::sizeZ + m_stencilEnd[2] - m_stencilStart[2] - 1);
    }
    offset[0] = (m_stencilStart[0] - 1) / 2 + m_InterpStencilStart[0];
    offset[1] = (m_stencilStart[1] - 1) / 2 + m_InterpStencilStart[1];
    offset[2] = (m_stencilStart[2] - 1) / 2 + m_InterpStencilStart[2];
    const int e[3] = {(m_stencilEnd[0]) / 2 + 1 + m_InterpStencilEnd[0] - 1,
                      (m_stencilEnd[1]) / 2 + 1 + m_InterpStencilEnd[1] - 1,
                      (m_stencilEnd[2]) / 2 + 1 + m_InterpStencilEnd[2] - 1};
    if (m_CoarsenedBlock == NULL ||
        (int)m_CoarsenedBlock->getSize()[0] !=
            CoarseBlockSize[0] + e[0] - offset[0] - 1 ||
        (int)m_CoarsenedBlock->getSize()[1] !=
            CoarseBlockSize[1] + e[1] - offset[1] - 1 ||
        (int)m_CoarsenedBlock->getSize()[2] !=
            CoarseBlockSize[2] + e[2] - offset[2] - 1) {
      if (m_CoarsenedBlock != NULL)
        _release(m_CoarsenedBlock);
      m_CoarsenedBlock =
          allocator<Matrix3D<ElementType, allocator>>().allocate(1);
      allocator<Matrix3D<ElementType, allocator>>().construct(m_CoarsenedBlock);
      m_CoarsenedBlock->_Setup(CoarseBlockSize[0] + e[0] - offset[0] - 1,
                               CoarseBlockSize[1] + e[1] - offset[1] - 1,
                               CoarseBlockSize[2] + e[2] - offset[2] - 1);
    }
    use_averages = (m_refGrid->FiniteDifferences == false || istensorial ||
                    m_stencilStart[0] < -2 || m_stencilStart[1] < -2 ||
                    m_stencilStart[2] < -2 || m_stencilEnd[0] > 3 ||
                    m_stencilEnd[1] > 3 || m_stencilEnd[2] > 3);
  }
  virtual void load(const Info &info, const Real t = 0,
                    const bool applybc = true) {
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const bool xperiodic = is_xperiodic();
    const bool yperiodic = is_yperiodic();
    const bool zperiodic = is_zperiodic();
    std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
    const int aux = 1 << info.level;
    NX = blocksPerDim[0] * aux;
    NY = blocksPerDim[1] * aux;
    NZ = blocksPerDim[2] * aux;
    assert(m_cacheBlock != NULL);
    {
      BlockType &block = *(BlockType *)info.block;
      ElementType *ptrSource = &block(0);
#if 0
            for(int iz=0; iz<nZ; iz++)
            for(int iy=0; iy<nY; iy++)
            {
              ElementType * ptrDestination = &m_cacheBlock->Access(0-m_stencilStart[0], iy-m_stencilStart[1], iz-m_stencilStart[2]);
              memcpy2((char *)ptrDestination, (char *)ptrSource, sizeof(ElementType)*nX);
              ptrSource+= nX;
            }
#else
      const int nbytes = sizeof(ElementType) * nX;
      const int _iz0 = -m_stencilStart[2];
      const int _iz1 = _iz0 + nZ;
      const int _iy0 = -m_stencilStart[1];
      const int _iy1 = _iy0 + nY;
      const int m_vSize0 = m_cacheBlock->getSize(0);
      const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
      const int my_ix = -m_stencilStart[0];
#pragma GCC ivdep
      for (int iz = _iz0; iz < _iz1; iz++) {
        const int my_izx = iz * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
        for (int iy = _iy0; iy < _iy1; iy += 4) {
          ElementType *__restrict__ ptrDestination0 =
              &m_cacheBlock->LinAccess(my_izx + (iy)*m_vSize0);
          ElementType *__restrict__ ptrDestination1 =
              &m_cacheBlock->LinAccess(my_izx + (iy + 1) * m_vSize0);
          ElementType *__restrict__ ptrDestination2 =
              &m_cacheBlock->LinAccess(my_izx + (iy + 2) * m_vSize0);
          ElementType *__restrict__ ptrDestination3 =
              &m_cacheBlock->LinAccess(my_izx + (iy + 3) * m_vSize0);
          memcpy2(ptrDestination0, (ptrSource), nbytes);
          memcpy2(ptrDestination1, (ptrSource + nX), nbytes);
          memcpy2(ptrDestination2, (ptrSource + 2 * nX), nbytes);
          memcpy2(ptrDestination3, (ptrSource + 3 * nX), nbytes);
          ptrSource += 4 * nX;
        }
      }
#endif
    }
    {
      coarsened = false;
      const bool xskin = info.index[0] == 0 || info.index[0] == NX - 1;
      const bool yskin = info.index[1] == 0 || info.index[1] == NY - 1;
      const bool zskin = info.index[2] == 0 || info.index[2] == NZ - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      int icodes[DIMENSION == 2 ? 8 : 26];
      int k = 0;
      coarsened_nei_codes_size = 0;
      for (int icode = (DIMENSION == 2 ? 9 : 0);
           icode < (DIMENSION == 2 ? 18 : 27); icode++) {
        myblocks[icode] = nullptr;
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, icode / 9 - 1};
        if (!xperiodic && code[0] == xskip && xskin)
          continue;
        if (!yperiodic && code[1] == yskip && yskin)
          continue;
        if (!zperiodic && code[2] == zskip && zskin)
          continue;
        const auto &TreeNei =
            m_refGrid->Tree(info.level, info.Znei_(code[0], code[1], code[2]));
        if (TreeNei.Exists()) {
          icodes[k++] = icode;
        } else if (TreeNei.CheckCoarser()) {
          coarsened_nei_codes[coarsened_nei_codes_size++] = icode;
          CoarseFineExchange(info, code);
        }
        if (!istensorial && !use_averages &&
            abs(code[0]) + abs(code[1]) + abs(code[2]) > 1)
          continue;
        const int s[3] = {
            code[0] < 1 ? (code[0] < 0 ? m_stencilStart[0] : 0) : nX,
            code[1] < 1 ? (code[1] < 0 ? m_stencilStart[1] : 0) : nY,
            code[2] < 1 ? (code[2] < 0 ? m_stencilStart[2] : 0) : nZ};
        const int e[3] = {
            code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + m_stencilEnd[0] - 1,
            code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + m_stencilEnd[1] - 1,
            code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + m_stencilEnd[2] - 1};
        if (TreeNei.Exists())
          SameLevelExchange(info, code, s, e);
        else if (TreeNei.CheckFiner())
          FineToCoarseExchange(info, code, s, e);
      }
      if (coarsened_nei_codes_size > 0)
        for (int i = 0; i < k; ++i) {
          const int icode = icodes[i];
          const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                               icode / 9 - 1};
          const int infoNei_index[3] = {(info.index[0] + code[0] + NX) % NX,
                                        (info.index[1] + code[1] + NY) % NY,
                                        (info.index[2] + code[2] + NZ) % NZ};
          if (UseCoarseStencil(info, infoNei_index)) {
            FillCoarseVersion(info, code);
            coarsened = true;
          }
        }
      if (m_refGrid->get_world_size() == 1) {
        post_load(info, t, applybc);
      }
    }
  }

protected:
  void post_load(const Info &info, const Real t = 0, bool applybc = true) {
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    if (coarsened) {
#pragma GCC ivdep
      for (int k = 0; k < nZ / 2; k++) {
#pragma GCC ivdep
        for (int j = 0; j < nY / 2; j++) {
#pragma GCC ivdep
          for (int i = 0; i < nX / 2; i++) {
            if (i > -m_InterpStencilStart[0] &&
                i < nX / 2 - m_InterpStencilEnd[0] &&
                j > -m_InterpStencilStart[1] &&
                j < nY / 2 - m_InterpStencilEnd[1] &&
                k > -m_InterpStencilStart[2] &&
                k < nZ / 2 - m_InterpStencilEnd[2])
              continue;
            const int ix = 2 * i - m_stencilStart[0];
            const int iy = 2 * j - m_stencilStart[1];
            const int iz = 2 * k - m_stencilStart[2];
            ElementType &coarseElement = m_CoarsenedBlock->Access(
                i - offset[0], j - offset[1], k - offset[2]);
            coarseElement =
                AverageDown(m_cacheBlock->Read(ix, iy, iz),
                            m_cacheBlock->Read(ix + 1, iy, iz),
                            m_cacheBlock->Read(ix, iy + 1, iz),
                            m_cacheBlock->Read(ix + 1, iy + 1, iz),
                            m_cacheBlock->Read(ix, iy, iz + 1),
                            m_cacheBlock->Read(ix + 1, iy, iz + 1),
                            m_cacheBlock->Read(ix, iy + 1, iz + 1),
                            m_cacheBlock->Read(ix + 1, iy + 1, iz + 1));
          }
        }
      }
    }
    if (applybc)
      _apply_bc(info, t, true);
    CoarseFineInterpolation(info);
    if (applybc)
      _apply_bc(info, t);
  }
  bool UseCoarseStencil(const Info &a, const int *b_index) {
    if (a.level == 0 || (!use_averages))
      return false;
    std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
    int imin[3];
    int imax[3];
    const int aux = 1 << a.level;
    const bool periodic[3] = {is_xperiodic(), is_yperiodic(), is_zperiodic()};
    const int blocks[3] = {blocksPerDim[0] * aux - 1, blocksPerDim[1] * aux - 1,
                           blocksPerDim[2] * aux - 1};
    for (int d = 0; d < 3; d++) {
      imin[d] = (a.index[d] < b_index[d]) ? 0 : -1;
      imax[d] = (a.index[d] > b_index[d]) ? 0 : +1;
      if (periodic[d]) {
        if (a.index[d] == 0 && b_index[d] == blocks[d])
          imin[d] = -1;
        if (b_index[d] == 0 && a.index[d] == blocks[d])
          imax[d] = +1;
      } else {
        if (a.index[d] == 0 && b_index[d] == 0)
          imin[d] = 0;
        if (a.index[d] == blocks[d] && b_index[d] == blocks[d])
          imax[d] = 0;
      }
    }
    for (int itest = 0; itest < coarsened_nei_codes_size; itest++)
      for (int i2 = imin[2]; i2 <= imax[2]; i2++)
        for (int i1 = imin[1]; i1 <= imax[1]; i1++)
          for (int i0 = imin[0]; i0 <= imax[0]; i0++) {
            const int icode_test = (i0 + 1) + 3 * (i1 + 1) + 9 * (i2 + 1);
            if (coarsened_nei_codes[itest] == icode_test)
              return true;
          }
    return false;
  }
  void SameLevelExchange(const Info &info, const int *const code,
                         const int *const s, const int *const e) {
    const int bytes = (e[0] - s[0]) * sizeof(ElementType);
    if (!bytes)
      return;
    const int icode = (code[0] + 1) + 3 * (code[1] + 1) + 9 * (code[2] + 1);
    myblocks[icode] =
        m_refGrid->avail(info.level, info.Znei_(code[0], code[1], code[2]));
    if (myblocks[icode] == nullptr)
      return;
    const BlockType &b = *myblocks[icode];
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const int m_vSize0 = m_cacheBlock->getSize(0);
    const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
    const int my_ix = s[0] - m_stencilStart[0];
    const int mod = (e[1] - s[1]) % 4;
#pragma GCC ivdep
    for (int iz = s[2]; iz < e[2]; iz++) {
      const int my_izx = (iz - m_stencilStart[2]) * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
      for (int iy = s[1]; iy < e[1] - mod; iy += 4) {
        ElementType *__restrict__ ptrDest0 = &m_cacheBlock->LinAccess(
            my_izx + (iy - m_stencilStart[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest1 = &m_cacheBlock->LinAccess(
            my_izx + (iy + 1 - m_stencilStart[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest2 = &m_cacheBlock->LinAccess(
            my_izx + (iy + 2 - m_stencilStart[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest3 = &m_cacheBlock->LinAccess(
            my_izx + (iy + 3 - m_stencilStart[1]) * m_vSize0);
        const ElementType *ptrSrc0 =
            &b(s[0] - code[0] * nX, iy - code[1] * nY, iz - code[2] * nZ);
        const ElementType *ptrSrc1 =
            &b(s[0] - code[0] * nX, iy + 1 - code[1] * nY, iz - code[2] * nZ);
        const ElementType *ptrSrc2 =
            &b(s[0] - code[0] * nX, iy + 2 - code[1] * nY, iz - code[2] * nZ);
        const ElementType *ptrSrc3 =
            &b(s[0] - code[0] * nX, iy + 3 - code[1] * nY, iz - code[2] * nZ);
        memcpy2(ptrDest0, ptrSrc0, bytes);
        memcpy2(ptrDest1, ptrSrc1, bytes);
        memcpy2(ptrDest2, ptrSrc2, bytes);
        memcpy2(ptrDest3, ptrSrc3, bytes);
      }
#pragma GCC ivdep
      for (int iy = e[1] - mod; iy < e[1]; iy++) {
        ElementType *__restrict__ ptrDest = &m_cacheBlock->LinAccess(
            my_izx + (iy - m_stencilStart[1]) * m_vSize0);
        const ElementType *ptrSrc =
            &b(s[0] - code[0] * nX, iy - code[1] * nY, iz - code[2] * nZ);
        memcpy2(ptrDest, ptrSrc, bytes);
      }
    }
  }
  ElementType AverageDown(const ElementType &e0, const ElementType &e1,
                          const ElementType &e2, const ElementType &e3,
                          const ElementType &e4, const ElementType &e5,
                          const ElementType &e6, const ElementType &e7) {
    return 0.125 * (e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7);
  }
  virtual void TestInterp(ElementType *C[3][3][3], ElementType *R, int x, int y,
                          int z) {
    const ElementType dudx = 0.125 * ((*C[2][1][1]) - (*C[0][1][1]));
    const ElementType dudy = 0.125 * ((*C[1][2][1]) - (*C[1][0][1]));
    const ElementType dudz = 0.125 * ((*C[1][1][2]) - (*C[1][1][0]));
    const ElementType dudxdy = 0.015625 * ((*C[0][0][1]) + (*C[2][2][1]) -
                                           (*C[2][0][1]) - (*C[0][2][1]));
    const ElementType dudxdz = 0.015625 * ((*C[0][1][0]) + (*C[2][1][2]) -
                                           (*C[2][1][0]) - (*C[0][1][2]));
    const ElementType dudydz = 0.015625 * ((*C[1][0][0]) + (*C[1][2][2]) -
                                           (*C[1][2][0]) - (*C[1][0][2]));
    const ElementType lap =
        *C[1][1][1] + 0.03125 * ((*C[0][1][1]) + (*C[2][1][1]) + (*C[1][0][1]) +
                                 (*C[1][2][1]) + (*C[1][1][0]) + (*C[1][1][2]) +
                                 (-6.0) * (*C[1][1][1]));
    R[0] = lap - dudx - dudy - dudz + dudxdy + dudxdz + dudydz;
    R[1] = lap + dudx - dudy - dudz - dudxdy - dudxdz + dudydz;
    R[2] = lap - dudx + dudy - dudz - dudxdy + dudxdz - dudydz;
    R[3] = lap + dudx + dudy - dudz + dudxdy - dudxdz - dudydz;
    R[4] = lap - dudx - dudy + dudz + dudxdy - dudxdz - dudydz;
    R[5] = lap + dudx - dudy + dudz - dudxdy + dudxdz - dudydz;
    R[6] = lap - dudx + dudy + dudz - dudxdy - dudxdz + dudydz;
    R[7] = lap + dudx + dudy + dudz + dudxdy + dudxdz + dudydz;
  }
  void FineToCoarseExchange(const Info &info, const int *const code,
                            const int *const s, const int *const e) {
    const int bytes = (abs(code[0]) * (e[0] - s[0]) +
                       (1 - abs(code[0])) * ((e[0] - s[0]) / 2)) *
                      sizeof(ElementType);
    if (!bytes)
      return;
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const int m_vSize0 = m_cacheBlock->getSize(0);
    const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
    const int yStep = (code[1] == 0) ? 2 : 1;
    const int zStep = (code[2] == 0) ? 2 : 1;
    const int mod = ((e[1] - s[1]) / yStep) % 4;
    int Bstep = 1;
    if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
      Bstep = 3;
    else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
      Bstep = 4;
    for (int B = 0; B <= 3; B += Bstep) {
      const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
      BlockType *b_ptr =
          m_refGrid->avail1(2 * info.index[0] + std::max(code[0], 0) + code[0] +
                                (B % 2) * std::max(0, 1 - abs(code[0])),
                            2 * info.index[1] + std::max(code[1], 0) + code[1] +
                                aux * std::max(0, 1 - abs(code[1])),
                            2 * info.index[2] + std::max(code[2], 0) + code[2] +
                                (B / 2) * std::max(0, 1 - abs(code[2])),
                            info.level + 1);
      if (b_ptr == nullptr)
        continue;
      BlockType &b = *b_ptr;
      const int my_ix = abs(code[0]) * (s[0] - m_stencilStart[0]) +
                        (1 - abs(code[0])) * (s[0] - m_stencilStart[0] +
                                              (B % 2) * (e[0] - s[0]) / 2);
      const int XX = s[0] - code[0] * nX + std::min(0, code[0]) * (e[0] - s[0]);
#pragma GCC ivdep
      for (int iz = s[2]; iz < e[2]; iz += zStep) {
        const int ZZ = (abs(code[2]) == 1)
                           ? 2 * (iz - code[2] * nZ) + std::min(0, code[2]) * nZ
                           : iz;
        const int my_izx =
            (abs(code[2]) * (iz - m_stencilStart[2]) +
             (1 - abs(code[2])) *
                 (iz / 2 - m_stencilStart[2] + (B / 2) * (e[2] - s[2]) / 2)) *
                m_nElemsPerSlice +
            my_ix;
#pragma GCC ivdep
        for (int iy = s[1]; iy < e[1] - mod; iy += 4 * yStep) {
          ElementType *__restrict__ ptrDest0 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 0 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 0 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          ElementType *__restrict__ ptrDest1 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 1 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 1 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          ElementType *__restrict__ ptrDest2 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 2 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 2 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          ElementType *__restrict__ ptrDest3 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 3 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 3 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          const int YY0 = (abs(code[1]) == 1)
                              ? 2 * (iy + 0 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 0 * yStep;
          const int YY1 = (abs(code[1]) == 1)
                              ? 2 * (iy + 1 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 1 * yStep;
          const int YY2 = (abs(code[1]) == 1)
                              ? 2 * (iy + 2 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 2 * yStep;
          const int YY3 = (abs(code[1]) == 1)
                              ? 2 * (iy + 3 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 3 * yStep;
          const ElementType *ptrSrc_00 = &b(XX, YY0, ZZ);
          const ElementType *ptrSrc_10 = &b(XX, YY0, ZZ + 1);
          const ElementType *ptrSrc_20 = &b(XX, YY0 + 1, ZZ);
          const ElementType *ptrSrc_30 = &b(XX, YY0 + 1, ZZ + 1);
          const ElementType *ptrSrc_01 = &b(XX, YY1, ZZ);
          const ElementType *ptrSrc_11 = &b(XX, YY1, ZZ + 1);
          const ElementType *ptrSrc_21 = &b(XX, YY1 + 1, ZZ);
          const ElementType *ptrSrc_31 = &b(XX, YY1 + 1, ZZ + 1);
          const ElementType *ptrSrc_02 = &b(XX, YY2, ZZ);
          const ElementType *ptrSrc_12 = &b(XX, YY2, ZZ + 1);
          const ElementType *ptrSrc_22 = &b(XX, YY2 + 1, ZZ);
          const ElementType *ptrSrc_32 = &b(XX, YY2 + 1, ZZ + 1);
          const ElementType *ptrSrc_03 = &b(XX, YY3, ZZ);
          const ElementType *ptrSrc_13 = &b(XX, YY3, ZZ + 1);
          const ElementType *ptrSrc_23 = &b(XX, YY3 + 1, ZZ);
          const ElementType *ptrSrc_33 = &b(XX, YY3 + 1, ZZ + 1);
#pragma GCC ivdep
          for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) +
                                 (1 - abs(code[0])) * ((e[0] - s[0]) / 2));
               ee++) {
            ptrDest0[ee] = AverageDown(
                ptrSrc_00[2 * ee], ptrSrc_10[2 * ee], ptrSrc_20[2 * ee],
                ptrSrc_30[2 * ee], ptrSrc_00[2 * ee + 1], ptrSrc_10[2 * ee + 1],
                ptrSrc_20[2 * ee + 1], ptrSrc_30[2 * ee + 1]);
            ptrDest1[ee] = AverageDown(
                ptrSrc_01[2 * ee], ptrSrc_11[2 * ee], ptrSrc_21[2 * ee],
                ptrSrc_31[2 * ee], ptrSrc_01[2 * ee + 1], ptrSrc_11[2 * ee + 1],
                ptrSrc_21[2 * ee + 1], ptrSrc_31[2 * ee + 1]);
            ptrDest2[ee] = AverageDown(
                ptrSrc_02[2 * ee], ptrSrc_12[2 * ee], ptrSrc_22[2 * ee],
                ptrSrc_32[2 * ee], ptrSrc_02[2 * ee + 1], ptrSrc_12[2 * ee + 1],
                ptrSrc_22[2 * ee + 1], ptrSrc_32[2 * ee + 1]);
            ptrDest3[ee] = AverageDown(
                ptrSrc_03[2 * ee], ptrSrc_13[2 * ee], ptrSrc_23[2 * ee],
                ptrSrc_33[2 * ee], ptrSrc_03[2 * ee + 1], ptrSrc_13[2 * ee + 1],
                ptrSrc_23[2 * ee + 1], ptrSrc_33[2 * ee + 1]);
          }
        }
#pragma GCC ivdep
        for (int iy = e[1] - mod; iy < e[1]; iy += yStep) {
          ElementType *ptrDest = (ElementType *)&m_cacheBlock->LinAccess(
              my_izx + (abs(code[1]) * (iy - m_stencilStart[1]) +
                        (1 - abs(code[1])) * (iy / 2 - m_stencilStart[1] +
                                              aux * (e[1] - s[1]) / 2)) *
                           m_vSize0);
          const int YY = (abs(code[1]) == 1) ? 2 * (iy - code[1] * nY) +
                                                   std::min(0, code[1]) * nY
                                             : iy;
          const ElementType *ptrSrc_0 = &b(XX, YY, ZZ);
          const ElementType *ptrSrc_1 = &b(XX, YY, ZZ + 1);
          const ElementType *ptrSrc_2 = &b(XX, YY + 1, ZZ);
          const ElementType *ptrSrc_3 = &b(XX, YY + 1, ZZ + 1);
          const ElementType *ptrSrc_0_1 = &b(XX + 1, YY, ZZ);
          const ElementType *ptrSrc_1_1 = &b(XX + 1, YY, ZZ + 1);
          const ElementType *ptrSrc_2_1 = &b(XX + 1, YY + 1, ZZ);
          const ElementType *ptrSrc_3_1 = &b(XX + 1, YY + 1, ZZ + 1);
#pragma GCC ivdep
          for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) +
                                 (1 - abs(code[0])) * ((e[0] - s[0]) / 2));
               ee++) {
            ptrDest[ee] = AverageDown(ptrSrc_0[2 * ee], ptrSrc_1[2 * ee],
                                      ptrSrc_2[2 * ee], ptrSrc_3[2 * ee],
                                      ptrSrc_0_1[2 * ee], ptrSrc_1_1[2 * ee],
                                      ptrSrc_2_1[2 * ee], ptrSrc_3_1[2 * ee]);
          }
        }
      }
    }
  }
  void CoarseFineExchange(const Info &info, const int *const code) {
    const int infoNei_index[3] = {(info.index[0] + code[0] + NX) % NX,
                                  (info.index[1] + code[1] + NY) % NY,
                                  (info.index[2] + code[2] + NZ) % NZ};
    const int infoNei_index_true[3] = {(info.index[0] + code[0]),
                                       (info.index[1] + code[1]),
                                       (info.index[2] + code[2])};
    BlockType *b_ptr =
        m_refGrid->avail1((infoNei_index[0]) / 2, (infoNei_index[1]) / 2,
                          (infoNei_index[2]) / 2, info.level - 1);
    if (b_ptr == nullptr)
      return;
    const BlockType &b = *b_ptr;
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const int s[3] = {
        code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : CoarseBlockSize[0],
        code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : CoarseBlockSize[1],
        code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : CoarseBlockSize[2]};
    const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : CoarseBlockSize[0])
                                  : CoarseBlockSize[0] + (m_stencilEnd[0]) / 2 +
                                        m_InterpStencilEnd[0] - 1,
                      code[1] < 1 ? (code[1] < 0 ? 0 : CoarseBlockSize[1])
                                  : CoarseBlockSize[1] + (m_stencilEnd[1]) / 2 +
                                        m_InterpStencilEnd[1] - 1,
                      code[2] < 1 ? (code[2] < 0 ? 0 : CoarseBlockSize[2])
                                  : CoarseBlockSize[2] + (m_stencilEnd[2]) / 2 +
                                        m_InterpStencilEnd[2] - 1};
    const int bytes = (e[0] - s[0]) * sizeof(ElementType);
    if (!bytes)
      return;
    const int base[3] = {(info.index[0] + code[0]) % 2,
                         (info.index[1] + code[1]) % 2,
                         (info.index[2] + code[2]) % 2};
    int CoarseEdge[3];
    CoarseEdge[0] = (code[0] == 0) ? 0
                    : (((info.index[0] % 2 == 0) &&
                        (infoNei_index_true[0] > info.index[0])) ||
                       ((info.index[0] % 2 == 1) &&
                        (infoNei_index_true[0] < info.index[0])))
                        ? 1
                        : 0;
    CoarseEdge[1] = (code[1] == 0) ? 0
                    : (((info.index[1] % 2 == 0) &&
                        (infoNei_index_true[1] > info.index[1])) ||
                       ((info.index[1] % 2 == 1) &&
                        (infoNei_index_true[1] < info.index[1])))
                        ? 1
                        : 0;
    CoarseEdge[2] = (code[2] == 0) ? 0
                    : (((info.index[2] % 2 == 0) &&
                        (infoNei_index_true[2] > info.index[2])) ||
                       ((info.index[2] % 2 == 1) &&
                        (infoNei_index_true[2] < info.index[2])))
                        ? 1
                        : 0;
    const int start[3] = {
        std::max(code[0], 0) * nX / 2 + (1 - abs(code[0])) * base[0] * nX / 2 -
            code[0] * nX + CoarseEdge[0] * code[0] * nX / 2,
        std::max(code[1], 0) * nY / 2 + (1 - abs(code[1])) * base[1] * nY / 2 -
            code[1] * nY + CoarseEdge[1] * code[1] * nY / 2,
        std::max(code[2], 0) * nZ / 2 + (1 - abs(code[2])) * base[2] * nZ / 2 -
            code[2] * nZ + CoarseEdge[2] * code[2] * nZ / 2};
    const int m_vSize0 = m_CoarsenedBlock->getSize(0);
    const int m_nElemsPerSlice =
        m_CoarsenedBlock->getNumberOfElementsPerSlice();
    const int my_ix = s[0] - offset[0];
    const int mod = (e[1] - s[1]) % 4;
#pragma GCC ivdep
    for (int iz = s[2]; iz < e[2]; iz++) {
      const int my_izx = (iz - offset[2]) * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
      for (int iy = s[1]; iy < e[1] - mod; iy += 4) {
        ElementType *__restrict__ ptrDest0 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 0 - offset[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest1 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 1 - offset[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest2 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 2 - offset[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest3 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 3 - offset[1]) * m_vSize0);
        const ElementType *ptrSrc0 =
            &b(s[0] + start[0], iy + 0 + start[1], iz + start[2]);
        const ElementType *ptrSrc1 =
            &b(s[0] + start[0], iy + 1 + start[1], iz + start[2]);
        const ElementType *ptrSrc2 =
            &b(s[0] + start[0], iy + 2 + start[1], iz + start[2]);
        const ElementType *ptrSrc3 =
            &b(s[0] + start[0], iy + 3 + start[1], iz + start[2]);
        memcpy2(ptrDest0, ptrSrc0, bytes);
        memcpy2(ptrDest1, ptrSrc1, bytes);
        memcpy2(ptrDest2, ptrSrc2, bytes);
        memcpy2(ptrDest3, ptrSrc3, bytes);
      }
#pragma GCC ivdep
      for (int iy = e[1] - mod; iy < e[1]; iy++) {
        ElementType *ptrDest =
            &m_CoarsenedBlock->LinAccess(my_izx + (iy - offset[1]) * m_vSize0);
        const ElementType *ptrSrc =
            &b(s[0] + start[0], iy + start[1], iz + start[2]);
        memcpy2(ptrDest, ptrSrc, bytes);
      }
    }
  }
  void FillCoarseVersion(const Info &info, const int *const code) {
    const int icode = (code[0] + 1) + 3 * (code[1] + 1) + 9 * (code[2] + 1);
    if (myblocks[icode] == nullptr)
      return;
    const BlockType &b = *myblocks[icode];
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const int eC[3] = {(m_stencilEnd[0]) / 2 + m_InterpStencilEnd[0],
                       (m_stencilEnd[1]) / 2 + m_InterpStencilEnd[1],
                       (m_stencilEnd[2]) / 2 + m_InterpStencilEnd[2]};
    const int s[3] = {
        code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : CoarseBlockSize[0],
        code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : CoarseBlockSize[1],
        code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : CoarseBlockSize[2]};
    const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : CoarseBlockSize[0])
                                  : CoarseBlockSize[0] + eC[0] - 1,
                      code[1] < 1 ? (code[1] < 0 ? 0 : CoarseBlockSize[1])
                                  : CoarseBlockSize[1] + eC[1] - 1,
                      code[2] < 1 ? (code[2] < 0 ? 0 : CoarseBlockSize[2])
                                  : CoarseBlockSize[2] + eC[2] - 1};
    const int bytes = (e[0] - s[0]) * sizeof(ElementType);
    if (!bytes)
      return;
    const int start[3] = {
        s[0] + std::max(code[0], 0) * CoarseBlockSize[0] - code[0] * nX +
            std::min(0, code[0]) * (e[0] - s[0]),
        s[1] + std::max(code[1], 0) * CoarseBlockSize[1] - code[1] * nY +
            std::min(0, code[1]) * (e[1] - s[1]),
        s[2] + std::max(code[2], 0) * CoarseBlockSize[2] - code[2] * nZ +
            std::min(0, code[2]) * (e[2] - s[2])};
    const int m_vSize0 = m_CoarsenedBlock->getSize(0);
    const int m_nElemsPerSlice =
        m_CoarsenedBlock->getNumberOfElementsPerSlice();
    const int my_ix = s[0] - offset[0];
    const int XX = start[0];
#pragma GCC ivdep
    for (int iz = s[2]; iz < e[2]; iz++) {
      const int ZZ = 2 * (iz - s[2]) + start[2];
      const int my_izx = (iz - offset[2]) * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
      for (int iy = s[1]; iy < e[1]; iy++) {
        if (code[1] == 0 && code[2] == 0 && iy > -m_InterpStencilStart[1] &&
            iy < nY / 2 - m_InterpStencilEnd[1] &&
            iz > -m_InterpStencilStart[2] &&
            iz < nZ / 2 - m_InterpStencilEnd[2])
          continue;
        ElementType *__restrict__ ptrDest1 =
            &m_CoarsenedBlock->LinAccess(my_izx + (iy - offset[1]) * m_vSize0);
        const int YY = 2 * (iy - s[1]) + start[1];
        const ElementType *ptrSrc_0 = &b(XX, YY, ZZ);
        const ElementType *ptrSrc_1 = &b(XX, YY, ZZ + 1);
        const ElementType *ptrSrc_2 = &b(XX, YY + 1, ZZ);
        const ElementType *ptrSrc_3 = &b(XX, YY + 1, ZZ + 1);
#pragma GCC ivdep
        for (int ee = 0; ee < e[0] - s[0]; ee++) {
          ptrDest1[ee] =
              AverageDown(*(ptrSrc_0 + 2 * ee), *(ptrSrc_1 + 2 * ee),
                          *(ptrSrc_2 + 2 * ee), *(ptrSrc_3 + 2 * ee),
                          *(ptrSrc_0 + 2 * ee + 1), *(ptrSrc_1 + 2 * ee + 1),
                          *(ptrSrc_2 + 2 * ee + 1), *(ptrSrc_3 + 2 * ee + 1));
        }
      }
    }
  }
  void CoarseFineInterpolation(const Info &info) {
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const bool xperiodic = is_xperiodic();
    const bool yperiodic = is_yperiodic();
    const bool zperiodic = is_zperiodic();
    const std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
    const int aux = 1 << info.level;
    const bool xskin =
        info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
    const bool yskin =
        info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
    const bool zskin =
        info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
    const int xskip = info.index[0] == 0 ? -1 : 1;
    const int yskip = info.index[1] == 0 ? -1 : 1;
    const int zskip = info.index[2] == 0 ? -1 : 1;
    for (int ii = 0; ii < coarsened_nei_codes_size; ++ii) {
      const int icode = coarsened_nei_codes[ii];
      if (icode == 1 * 1 + 3 * 1 + 9 * 1)
        continue;
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                           (icode / 9) % 3 - 1};
      if (!xperiodic && code[0] == xskip && xskin)
        continue;
      if (!yperiodic && code[1] == yskip && yskin)
        continue;
      if (!zperiodic && code[2] == zskip && zskin)
        continue;
      if (!istensorial && !use_averages &&
          abs(code[0]) + abs(code[1]) + abs(code[2]) > 1)
        continue;
      const int s[3] = {
          code[0] < 1 ? (code[0] < 0 ? m_stencilStart[0] : 0) : nX,
          code[1] < 1 ? (code[1] < 0 ? m_stencilStart[1] : 0) : nY,
          code[2] < 1 ? (code[2] < 0 ? m_stencilStart[2] : 0) : nZ};
      const int e[3] = {
          code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + m_stencilEnd[0] - 1,
          code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + m_stencilEnd[1] - 1,
          code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + m_stencilEnd[2] - 1};
      const int sC[3] = {
          code[0] < 1 ? (code[0] < 0 ? ((m_stencilStart[0] - 1) / 2) : 0)
                      : CoarseBlockSize[0],
          code[1] < 1 ? (code[1] < 0 ? ((m_stencilStart[1] - 1) / 2) : 0)
                      : CoarseBlockSize[1],
          code[2] < 1 ? (code[2] < 0 ? ((m_stencilStart[2] - 1) / 2) : 0)
                      : CoarseBlockSize[2]};
      const int bytes = (e[0] - s[0]) * sizeof(ElementType);
      if (!bytes)
        continue;
      ElementType retval[8];
      if (use_averages)
        for (int iz = s[2]; iz < e[2]; iz += 2) {
          const int ZZ =
              (iz - s[2] - std::min(0, code[2]) * ((e[2] - s[2]) % 2)) / 2 +
              sC[2];
          const int z =
              abs(iz - s[2] - std::min(0, code[2]) * ((e[2] - s[2]) % 2)) % 2;
          const int izp = (abs(iz) % 2 == 1) ? -1 : 1;
          const int rzp = (izp == 1) ? 1 : 0;
          const int rz = (izp == 1) ? 0 : 1;
#pragma GCC ivdep
          for (int iy = s[1]; iy < e[1]; iy += 2) {
            const int YY =
                (iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 +
                sC[1];
            const int y =
                abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;
            const int iyp = (abs(iy) % 2 == 1) ? -1 : 1;
            const int ryp = (iyp == 1) ? 1 : 0;
            const int ry = (iyp == 1) ? 0 : 1;
#pragma GCC ivdep
            for (int ix = s[0]; ix < e[0]; ix += 2) {
              const int XX =
                  (ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 +
                  sC[0];
              const int x =
                  abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) %
                  2;
              const int ixp = (abs(ix) % 2 == 1) ? -1 : 1;
              const int rxp = (ixp == 1) ? 1 : 0;
              const int rx = (ixp == 1) ? 0 : 1;
              ElementType *Test[3][3][3];
              for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                  for (int k = 0; k < 3; k++)
                    Test[i][j][k] = &m_CoarsenedBlock->Access(
                        XX - 1 + i - offset[0], YY - 1 + j - offset[1],
                        ZZ - 1 + k - offset[2]);
              TestInterp(Test, retval, x, y, z);
              if (ix >= s[0] && ix < e[0] && iy >= s[1] && iy < e[1] &&
                  iz >= s[2] && iz < e[2])
                m_cacheBlock->Access(
                    ix - m_stencilStart[0], iy - m_stencilStart[1],
                    iz - m_stencilStart[2]) = retval[rx + 2 * ry + 4 * rz];
              if (ix + ixp >= s[0] && ix + ixp < e[0] && iy >= s[1] &&
                  iy < e[1] && iz >= s[2] && iz < e[2])
                m_cacheBlock->Access(
                    ix + ixp - m_stencilStart[0], iy - m_stencilStart[1],
                    iz - m_stencilStart[2]) = retval[rxp + 2 * ry + 4 * rz];
              if (ix >= s[0] && ix < e[0] && iy + iyp >= s[1] &&
                  iy + iyp < e[1] && iz >= s[2] && iz < e[2])
                m_cacheBlock->Access(
                    ix - m_stencilStart[0], iy + iyp - m_stencilStart[1],
                    iz - m_stencilStart[2]) = retval[rx + 2 * ryp + 4 * rz];
              if (ix + ixp >= s[0] && ix + ixp < e[0] && iy + iyp >= s[1] &&
                  iy + iyp < e[1] && iz >= s[2] && iz < e[2])
                m_cacheBlock->Access(
                    ix + ixp - m_stencilStart[0], iy + iyp - m_stencilStart[1],
                    iz - m_stencilStart[2]) = retval[rxp + 2 * ryp + 4 * rz];
              if (ix >= s[0] && ix < e[0] && iy >= s[1] && iy < e[1] &&
                  iz + izp >= s[2] && iz + izp < e[2])
                m_cacheBlock->Access(ix - m_stencilStart[0],
                                     iy - m_stencilStart[1],
                                     iz + izp - m_stencilStart[2]) =
                    retval[rx + 2 * ry + 4 * rzp];
              if (ix + ixp >= s[0] && ix + ixp < e[0] && iy >= s[1] &&
                  iy < e[1] && iz + izp >= s[2] && iz + izp < e[2])
                m_cacheBlock->Access(ix + ixp - m_stencilStart[0],
                                     iy - m_stencilStart[1],
                                     iz + izp - m_stencilStart[2]) =
                    retval[rxp + 2 * ry + 4 * rzp];
              if (ix >= s[0] && ix < e[0] && iy + iyp >= s[1] &&
                  iy + iyp < e[1] && iz + izp >= s[2] && iz + izp < e[2])
                m_cacheBlock->Access(ix - m_stencilStart[0],
                                     iy + iyp - m_stencilStart[1],
                                     iz + izp - m_stencilStart[2]) =
                    retval[rx + 2 * ryp + 4 * rzp];
              if (ix + ixp >= s[0] && ix + ixp < e[0] && iy + iyp >= s[1] &&
                  iy + iyp < e[1] && iz + izp >= s[2] && iz + izp < e[2])
                m_cacheBlock->Access(ix + ixp - m_stencilStart[0],
                                     iy + iyp - m_stencilStart[1],
                                     iz + izp - m_stencilStart[2]) =
                    retval[rxp + 2 * ryp + 4 * rzp];
            }
          }
        }
      if (m_refGrid->FiniteDifferences &&
          abs(code[0]) + abs(code[1]) + abs(code[2]) == 1) {
        const int coef_ixyz[3] = {std::min(0, code[0]) * ((e[0] - s[0]) % 2),
                                  std::min(0, code[1]) * ((e[1] - s[1]) % 2),
                                  std::min(0, code[2]) * ((e[2] - s[2]) % 2)};
        const int min_iz = std::max(s[2], -2);
        const int min_iy = std::max(s[1], -2);
        const int min_ix = std::max(s[0], -2);
        const int max_iz = std::min(e[2], nZ + 2);
        const int max_iy = std::min(e[1], nY + 2);
        const int max_ix = std::min(e[0], nX + 2);
        for (int iz = min_iz; iz < max_iz; iz++) {
          const int ZZ = (iz - s[2] - coef_ixyz[2]) / 2 + sC[2] - offset[2];
          const int z = abs(iz - s[2] - coef_ixyz[2]) % 2;
          const double dz = 0.25 * (2 * z - 1);
          const double *dz_coef = dz > 0 ? &d_coef_plus[0] : &d_coef_minus[0];
          const bool zinner = (ZZ + offset[2] != 0) &&
                              (ZZ + offset[2] != CoarseBlockSize[2] - 1);
          const bool zstart = (ZZ + offset[2] == 0);
#pragma GCC ivdep
          for (int iy = min_iy; iy < max_iy; iy++) {
            const int YY = (iy - s[1] - coef_ixyz[1]) / 2 + sC[1] - offset[1];
            const int y = abs(iy - s[1] - coef_ixyz[1]) % 2;
            const double dy = 0.25 * (2 * y - 1);
            const double *dy_coef = dy > 0 ? &d_coef_plus[0] : &d_coef_minus[0];
            const bool yinner = (YY + offset[1] != 0) &&
                                (YY + offset[1] != CoarseBlockSize[1] - 1);
            const bool ystart = (YY + offset[1] == 0);
#pragma GCC ivdep
            for (int ix = min_ix; ix < max_ix; ix++) {
              const int XX = (ix - s[0] - coef_ixyz[0]) / 2 + sC[0] - offset[0];
              const int x = abs(ix - s[0] - coef_ixyz[0]) % 2;
              const double dx = 0.25 * (2 * x - 1);
              const double *dx_coef =
                  dx > 0 ? &d_coef_plus[0] : &d_coef_minus[0];
              const bool xinner = (XX + offset[0] != 0) &&
                                  (XX + offset[0] != CoarseBlockSize[0] - 1);
              const bool xstart = (XX + offset[0] == 0);
              auto &a = m_cacheBlock->Access(ix - m_stencilStart[0],
                                             iy - m_stencilStart[1],
                                             iz - m_stencilStart[2]);
              if (code[0] != 0) {
                ElementType x1D, x2D, mixed;
                int YP, YM, ZP, ZM;
                double mixed_coef = 1.0;
                if (yinner) {
                  x1D =
                      (dy_coef[6] * m_CoarsenedBlock->Access(XX, YY - 1, ZZ) +
                       dy_coef[8] * m_CoarsenedBlock->Access(XX, YY + 1, ZZ)) +
                      dy_coef[7] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  YP = YY + 1;
                  YM = YY - 1;
                  mixed_coef *= 0.5;
                } else if (ystart) {
                  x1D =
                      (dy_coef[0] * m_CoarsenedBlock->Access(XX, YY + 2, ZZ) +
                       dy_coef[1] * m_CoarsenedBlock->Access(XX, YY + 1, ZZ)) +
                      dy_coef[2] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  YP = YY + 1;
                  YM = YY;
                } else {
                  x1D =
                      (dy_coef[3] * m_CoarsenedBlock->Access(XX, YY - 2, ZZ) +
                       dy_coef[4] * m_CoarsenedBlock->Access(XX, YY - 1, ZZ)) +
                      dy_coef[5] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  YP = YY;
                  YM = YY - 1;
                }
                if (zinner) {
                  x2D =
                      (dz_coef[6] * m_CoarsenedBlock->Access(XX, YY, ZZ - 1) +
                       dz_coef[8] * m_CoarsenedBlock->Access(XX, YY, ZZ + 1)) +
                      dz_coef[7] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  ZP = ZZ + 1;
                  ZM = ZZ - 1;
                  mixed_coef *= 0.5;
                } else if (zstart) {
                  x2D =
                      (dz_coef[0] * m_CoarsenedBlock->Access(XX, YY, ZZ + 2) +
                       dz_coef[1] * m_CoarsenedBlock->Access(XX, YY, ZZ + 1)) +
                      dz_coef[2] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  ZP = ZZ + 1;
                  ZM = ZZ;
                } else {
                  x2D =
                      (dz_coef[3] * m_CoarsenedBlock->Access(XX, YY, ZZ - 2) +
                       dz_coef[4] * m_CoarsenedBlock->Access(XX, YY, ZZ - 1)) +
                      dz_coef[5] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  ZP = ZZ;
                  ZM = ZZ - 1;
                }
                mixed = mixed_coef * dy * dz *
                        ((m_CoarsenedBlock->Access(XX, YM, ZM) +
                          m_CoarsenedBlock->Access(XX, YP, ZP)) -
                         (m_CoarsenedBlock->Access(XX, YP, ZM) +
                          m_CoarsenedBlock->Access(XX, YM, ZP)));
                a = (x1D + x2D) + mixed;
              } else if (code[1] != 0) {
                ElementType x1D, x2D, mixed;
                int XP, XM, ZP, ZM;
                double mixed_coef = 1.0;
                if (xinner) {
                  x1D =
                      (dx_coef[6] * m_CoarsenedBlock->Access(XX - 1, YY, ZZ) +
                       dx_coef[8] * m_CoarsenedBlock->Access(XX + 1, YY, ZZ)) +
                      dx_coef[7] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  XP = XX + 1;
                  XM = XX - 1;
                  mixed_coef *= 0.5;
                } else if (xstart) {
                  x1D =
                      (dx_coef[0] * m_CoarsenedBlock->Access(XX + 2, YY, ZZ) +
                       dx_coef[1] * m_CoarsenedBlock->Access(XX + 1, YY, ZZ)) +
                      dx_coef[2] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  XP = XX + 1;
                  XM = XX;
                } else {
                  x1D =
                      (dx_coef[3] * m_CoarsenedBlock->Access(XX - 2, YY, ZZ) +
                       dx_coef[4] * m_CoarsenedBlock->Access(XX - 1, YY, ZZ)) +
                      dx_coef[5] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  XP = XX;
                  XM = XX - 1;
                }
                if (zinner) {
                  x2D =
                      (dz_coef[6] * m_CoarsenedBlock->Access(XX, YY, ZZ - 1) +
                       dz_coef[8] * m_CoarsenedBlock->Access(XX, YY, ZZ + 1)) +
                      dz_coef[7] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  ZP = ZZ + 1;
                  ZM = ZZ - 1;
                  mixed_coef *= 0.5;
                } else if (zstart) {
                  x2D =
                      (dz_coef[0] * m_CoarsenedBlock->Access(XX, YY, ZZ + 2) +
                       dz_coef[1] * m_CoarsenedBlock->Access(XX, YY, ZZ + 1)) +
                      dz_coef[2] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  ZP = ZZ + 1;
                  ZM = ZZ;
                } else {
                  x2D =
                      (dz_coef[3] * m_CoarsenedBlock->Access(XX, YY, ZZ - 2) +
                       dz_coef[4] * m_CoarsenedBlock->Access(XX, YY, ZZ - 1)) +
                      dz_coef[5] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  ZP = ZZ;
                  ZM = ZZ - 1;
                }
                mixed = mixed_coef * dx * dz *
                        ((m_CoarsenedBlock->Access(XM, YY, ZM) +
                          m_CoarsenedBlock->Access(XP, YY, ZP)) -
                         (m_CoarsenedBlock->Access(XP, YY, ZM) +
                          m_CoarsenedBlock->Access(XM, YY, ZP)));
                a = (x1D + x2D) + mixed;
              } else if (code[2] != 0) {
                ElementType x1D, x2D, mixed;
                int XP, XM, YP, YM;
                double mixed_coef = 1.0;
                if (xinner) {
                  x1D =
                      (dx_coef[6] * m_CoarsenedBlock->Access(XX - 1, YY, ZZ) +
                       dx_coef[8] * m_CoarsenedBlock->Access(XX + 1, YY, ZZ)) +
                      dx_coef[7] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  XP = XX + 1;
                  XM = XX - 1;
                  mixed_coef *= 0.5;
                } else if (xstart) {
                  x1D =
                      (dx_coef[0] * m_CoarsenedBlock->Access(XX + 2, YY, ZZ) +
                       dx_coef[1] * m_CoarsenedBlock->Access(XX + 1, YY, ZZ)) +
                      dx_coef[2] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  XP = XX + 1;
                  XM = XX;
                } else {
                  x1D =
                      (dx_coef[3] * m_CoarsenedBlock->Access(XX - 2, YY, ZZ) +
                       dx_coef[4] * m_CoarsenedBlock->Access(XX - 1, YY, ZZ)) +
                      dx_coef[5] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  XP = XX;
                  XM = XX - 1;
                }
                if (yinner) {
                  x2D =
                      (dy_coef[6] * m_CoarsenedBlock->Access(XX, YY - 1, ZZ) +
                       dy_coef[8] * m_CoarsenedBlock->Access(XX, YY + 1, ZZ)) +
                      dy_coef[7] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  YP = YY + 1;
                  YM = YY - 1;
                  mixed_coef *= 0.5;
                } else if (ystart) {
                  x2D =
                      (dy_coef[0] * m_CoarsenedBlock->Access(XX, YY + 2, ZZ) +
                       dy_coef[1] * m_CoarsenedBlock->Access(XX, YY + 1, ZZ)) +
                      dy_coef[2] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  YP = YY + 1;
                  YM = YY;
                } else {
                  x2D =
                      (dy_coef[3] * m_CoarsenedBlock->Access(XX, YY - 2, ZZ) +
                       dy_coef[4] * m_CoarsenedBlock->Access(XX, YY - 1, ZZ)) +
                      dy_coef[5] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  YP = YY;
                  YM = YY - 1;
                }
                mixed = mixed_coef * dx * dy *
                        ((m_CoarsenedBlock->Access(XM, YM, ZZ) +
                          m_CoarsenedBlock->Access(XP, YP, ZZ)) -
                         (m_CoarsenedBlock->Access(XP, YM, ZZ) +
                          m_CoarsenedBlock->Access(XM, YP, ZZ)));
                a = (x1D + x2D) + mixed;
              }
              const auto &b = m_cacheBlock->Access(
                  ix - m_stencilStart[0] + (-3 * code[0] + 1) / 2 -
                      x * abs(code[0]),
                  iy - m_stencilStart[1] + (-3 * code[1] + 1) / 2 -
                      y * abs(code[1]),
                  iz - m_stencilStart[2] + (-3 * code[2] + 1) / 2 -
                      z * abs(code[2]));
              const auto &c = m_cacheBlock->Access(
                  ix - m_stencilStart[0] + (-5 * code[0] + 1) / 2 -
                      x * abs(code[0]),
                  iy - m_stencilStart[1] + (-5 * code[1] + 1) / 2 -
                      y * abs(code[1]),
                  iz - m_stencilStart[2] + (-5 * code[2] + 1) / 2 -
                      z * abs(code[2]));
              const int ccc = code[0] + code[1] + code[2];
              const int xyz =
                  abs(code[0]) * x + abs(code[1]) * y + abs(code[2]) * z;
              if (ccc == 1)
                a = (xyz == 0)
                        ? (1.0 / 15.0) * (8.0 * a + (10.0 * b - 3.0 * c))
                        : (1.0 / 15.0) * (24.0 * a + (-15.0 * b + 6 * c));
              else
                a = (xyz == 1)
                        ? (1.0 / 15.0) * (8.0 * a + (10.0 * b - 3.0 * c))
                        : (1.0 / 15.0) * (24.0 * a + (-15.0 * b + 6 * c));
            }
          }
        }
      }
    }
  }
  virtual void _apply_bc(const Info &info, const Real t = 0,
                         bool coarse = false) {}
  template <typename T> void _release(T *&t) {
    if (t != NULL) {
      allocator<T>().destroy(t);
      allocator<T>().deallocate(t, 1);
    }
    t = NULL;
  }

private:
  BlockLab(const BlockLab &) = delete;
  BlockLab &operator=(const BlockLab &) = delete;
};
template <typename MyBlockLab> class BlockLabMPI : public MyBlockLab {
public:
  using GridType = typename MyBlockLab::GridType;
  using BlockType = typename GridType::BlockType;
  using ElementType = typename BlockType::ElementType;
  using Real = typename ElementType::RealType;

private:
  typedef SynchronizerMPI_AMR<Real, GridType> SynchronizerMPIType;
  SynchronizerMPIType *refSynchronizerMPI;

public:
  virtual void prepare(GridType &grid, const StencilInfo &stencil,
                       const int Istencil_start[3] = default_start,
                       const int Istencil_end[3] = default_end) override {
    auto itSynchronizerMPI = grid.SynchronizerMPIs.find(stencil);
    refSynchronizerMPI = itSynchronizerMPI->second;
    MyBlockLab::prepare(grid, stencil);
  }
  virtual void load(const Info &info, const Real t = 0,
                    const bool applybc = true) override {
    MyBlockLab::load(info, t, applybc);
    Real *dst = (Real *)&MyBlockLab ::m_cacheBlock->LinAccess(0);
    Real *dst1 = (Real *)&MyBlockLab ::m_CoarsenedBlock->LinAccess(0);
    refSynchronizerMPI->fetch(info, MyBlockLab::m_cacheBlock->getSize(),
                              MyBlockLab::m_CoarsenedBlock->getSize(), dst,
                              dst1);
    if (MyBlockLab::m_refGrid->get_world_size() > 1)
      MyBlockLab::post_load(info, t, applybc);
  }
};
template <typename TGrid> class LoadBalancer {
public:
  typedef typename TGrid::Block BlockType;
  typedef typename TGrid::Block::ElementType ElementType;
  typedef typename TGrid::Block::ElementType::RealType Real;
  bool movedBlocks;

protected:
  TGrid *grid;
  MPI_Datatype MPI_BLOCK;
  struct MPI_Block {
    long long mn[2];
    Real data[sizeof(BlockType) / sizeof(Real)];
    MPI_Block(const Info &info, const bool Fillptr = true) {
      prepare(info, Fillptr);
    }
    void prepare(const Info &info, const bool Fillptr = true) {
      mn[0] = info.level;
      mn[1] = info.Z;
      if (Fillptr) {
        Real *aux = &((BlockType *)info.block)->data[0][0][0].member(0);
        std::memcpy(&data[0], aux, sizeof(BlockType));
      }
    }
    MPI_Block() {}
  };
  void AddBlock(const int level, const long long Z, Real *data) {
    grid->_alloc(level, Z);
    Info &info = grid->getInfoAll(level, Z);
    BlockType *b1 = (BlockType *)info.block;
    assert(b1 != NULL);
    Real *a1 = &b1->data[0][0][0].member(0);
    std::memcpy(a1, data, sizeof(BlockType));
    int p[3];
    Info::inverse(Z, level, p[0], p[1], p[2]);
    if (level < grid->getlevelMax() - 1)
      for (int k1 = 0; k1 < 2; k1++)
        for (int j1 = 0; j1 < 2; j1++)
          for (int i1 = 0; i1 < 2; i1++) {
            const long long nc = grid->getZforward(
                level + 1, 2 * p[0] + i1, 2 * p[1] + j1, 2 * p[2] + k1);
            grid->Tree(level + 1, nc).setCheckCoarser();
          }
    if (level > 0) {
      const long long nf =
          grid->getZforward(level - 1, p[0] / 2, p[1] / 2, p[2] / 2);
      grid->Tree(level - 1, nf).setCheckFiner();
    }
  }

public:
  LoadBalancer(TGrid &a_grid) {
    grid = &a_grid;
    movedBlocks = false;
    int array_of_blocklengths[2] = {2, sizeof(BlockType) / sizeof(Real)};
    MPI_Aint array_of_displacements[2] = {0, 2 * sizeof(long long)};
    MPI_Datatype array_of_types[2];
    array_of_types[0] = MPI_LONG_LONG;
    if (sizeof(Real) == sizeof(float))
      array_of_types[1] = MPI_FLOAT;
    else if (sizeof(Real) == sizeof(double))
      array_of_types[1] = MPI_DOUBLE;
    else if (sizeof(Real) == sizeof(long double))
      array_of_types[1] = MPI_LONG_DOUBLE;
    MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements,
                           array_of_types, &MPI_BLOCK);
    MPI_Type_commit(&MPI_BLOCK);
  }
  ~LoadBalancer() { MPI_Type_free(&MPI_BLOCK); }
  void PrepareCompression() {
    const int size = grid->get_world_size();
    const int rank = grid->rank();
    std::vector<Info> &I = grid->getBlocksInfo();
    std::vector<std::vector<MPI_Block>> send_blocks(size);
    std::vector<std::vector<MPI_Block>> recv_blocks(size);
    for (auto &b : I) {
      const long long nBlock =
          grid->getZforward(b.level, 2 * (b.index[0] / 2), 2 * (b.index[1] / 2),
                            2 * (b.index[2] / 2));
      const Info &base = grid->getInfoAll(b.level, nBlock);
      if (!grid->Tree(base).Exists() || base.state != Compress)
        continue;
      const Info &bCopy = grid->getInfoAll(b.level, b.Z);
      const int baserank = grid->Tree(b.level, nBlock).rank();
      const int brank = grid->Tree(b.level, b.Z).rank();
      if (b.Z != nBlock) {
        if (baserank != rank && brank == rank) {
          send_blocks[baserank].push_back({bCopy});
          grid->Tree(b.level, b.Z).setrank(baserank);
        }
      } else {
        for (int k = 0; k < 2; k++)
          for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++) {
              const long long n = grid->getZforward(
                  b.level, b.index[0] + i, b.index[1] + j, b.index[2] + k);
              if (n == nBlock)
                continue;
              Info &temp = grid->getInfoAll(b.level, n);
              const int temprank = grid->Tree(b.level, n).rank();
              if (temprank != rank) {
                recv_blocks[temprank].push_back({temp, false});
                grid->Tree(b.level, n).setrank(baserank);
              }
            }
      }
    }
    std::vector<MPI_Request> requests;
    for (int r = 0; r < size; r++)
      if (r != rank) {
        if (recv_blocks[r].size() != 0) {
          MPI_Request req{};
          requests.push_back(req);
          MPI_Irecv(&recv_blocks[r][0], recv_blocks[r].size(), MPI_BLOCK, r,
                    2468, grid->getWorldComm(), &requests.back());
        }
        if (send_blocks[r].size() != 0) {
          MPI_Request req{};
          requests.push_back(req);
          MPI_Isend(&send_blocks[r][0], send_blocks[r].size(), MPI_BLOCK, r,
                    2468, grid->getWorldComm(), &requests.back());
        }
      }
    for (int r = 0; r < size; r++)
      for (int i = 0; i < (int)send_blocks[r].size(); i++) {
        grid->_dealloc(send_blocks[r][i].mn[0], send_blocks[r][i].mn[1]);
        grid->Tree(send_blocks[r][i].mn[0], send_blocks[r][i].mn[1])
            .setCheckCoarser();
      }
    if (requests.size() != 0) {
      movedBlocks = true;
      MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
    }
    for (int r = 0; r < size; r++)
      for (int i = 0; i < (int)recv_blocks[r].size(); i++) {
        const int level = (int)recv_blocks[r][i].mn[0];
        const long long Z = recv_blocks[r][i].mn[1];
        grid->_alloc(level, Z);
        Info &info = grid->getInfoAll(level, Z);
        BlockType *b1 = (BlockType *)info.block;
        assert(b1 != NULL);
        Real *a1 = &b1->data[0][0][0].member(0);
        std::memcpy(a1, recv_blocks[r][i].data, sizeof(BlockType));
      }
  }
  void Balance_Diffusion(std::vector<long long> &block_distribution) {
    const int size = grid->get_world_size();
    const int rank = grid->rank();
    movedBlocks = false;
    {
      long long max_b = block_distribution[0];
      long long min_b = block_distribution[0];
      for (auto &b : block_distribution) {
        max_b = std::max(max_b, b);
        min_b = std::min(min_b, b);
      }
      const double ratio = static_cast<double>(max_b) / min_b;
      if (ratio > 1.01 || min_b == 0) {
        Balance_Global(block_distribution);
        return;
      }
    }
    const int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    const int left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int my_blocks = grid->getBlocksInfo().size();
    int right_blocks, left_blocks;
    MPI_Request reqs[4];
    MPI_Irecv(&left_blocks, 1, MPI_INT, left, 123, grid->getWorldComm(),
              &reqs[0]);
    MPI_Irecv(&right_blocks, 1, MPI_INT, right, 456, grid->getWorldComm(),
              &reqs[1]);
    MPI_Isend(&my_blocks, 1, MPI_INT, left, 456, grid->getWorldComm(),
              &reqs[2]);
    MPI_Isend(&my_blocks, 1, MPI_INT, right, 123, grid->getWorldComm(),
              &reqs[3]);
    MPI_Waitall(4, &reqs[0], MPI_STATUSES_IGNORE);
    const int nu = 4;
    const int flux_left = (rank == 0) ? 0 : (my_blocks - left_blocks) / nu;
    const int flux_right =
        (rank == size - 1) ? 0 : (my_blocks - right_blocks) / nu;
    std::vector<Info> SortedInfos = grid->getBlocksInfo();
    if (flux_right != 0 || flux_left != 0)
      std::sort(SortedInfos.begin(), SortedInfos.end());
    std::vector<MPI_Block> send_left;
    std::vector<MPI_Block> recv_left;
    std::vector<MPI_Block> send_right;
    std::vector<MPI_Block> recv_right;
    std::vector<MPI_Request> request;
    if (flux_left > 0) {
      send_left.resize(flux_left);
#pragma omp parallel for schedule(runtime)
      for (int i = 0; i < flux_left; i++)
        send_left[i].prepare(SortedInfos[i]);
      MPI_Request req{};
      request.push_back(req);
      MPI_Isend(&send_left[0], send_left.size(), MPI_BLOCK, left, 7890,
                grid->getWorldComm(), &request.back());
    } else if (flux_left < 0) {
      recv_left.resize(abs(flux_left));
      MPI_Request req{};
      request.push_back(req);
      MPI_Irecv(&recv_left[0], recv_left.size(), MPI_BLOCK, left, 4560,
                grid->getWorldComm(), &request.back());
    }
    if (flux_right > 0) {
      send_right.resize(flux_right);
#pragma omp parallel for schedule(runtime)
      for (int i = 0; i < flux_right; i++)
        send_right[i].prepare(SortedInfos[my_blocks - i - 1]);
      MPI_Request req{};
      request.push_back(req);
      MPI_Isend(&send_right[0], send_right.size(), MPI_BLOCK, right, 4560,
                grid->getWorldComm(), &request.back());
    } else if (flux_right < 0) {
      recv_right.resize(abs(flux_right));
      MPI_Request req{};
      request.push_back(req);
      MPI_Irecv(&recv_right[0], recv_right.size(), MPI_BLOCK, right, 7890,
                grid->getWorldComm(), &request.back());
    }
    for (int i = 0; i < flux_right; i++) {
      Info &info = SortedInfos[my_blocks - i - 1];
      grid->_dealloc(info.level, info.Z);
      grid->Tree(info.level, info.Z).setrank(right);
    }
    for (int i = 0; i < flux_left; i++) {
      Info &info = SortedInfos[i];
      grid->_dealloc(info.level, info.Z);
      grid->Tree(info.level, info.Z).setrank(left);
    }
    if (request.size() != 0) {
      movedBlocks = true;
      MPI_Waitall(request.size(), &request[0], MPI_STATUSES_IGNORE);
    }
    int temp = movedBlocks ? 1 : 0;
    MPI_Request request_reduction;
    MPI_Iallreduce(MPI_IN_PLACE, &temp, 1, MPI_INT, MPI_SUM,
                   grid->getWorldComm(), &request_reduction);
    for (int i = 0; i < -flux_left; i++)
      AddBlock(recv_left[i].mn[0], recv_left[i].mn[1], recv_left[i].data);
    for (int i = 0; i < -flux_right; i++)
      AddBlock(recv_right[i].mn[0], recv_right[i].mn[1], recv_right[i].data);
    MPI_Wait(&request_reduction, MPI_STATUS_IGNORE);
    movedBlocks = (temp >= 1);
    grid->FillPos();
  }
  void Balance_Global(std::vector<long long> &all_b) {
    const int size = grid->get_world_size();
    const int rank = grid->rank();
    std::vector<Info> SortedInfos = grid->getBlocksInfo();
    std::sort(SortedInfos.begin(), SortedInfos.end());
    long long total_load = 0;
    for (int r = 0; r < size; r++)
      total_load += all_b[r];
    long long my_load = total_load / size;
    if (rank < (total_load % size))
      my_load += 1;
    std::vector<long long> index_start(size);
    index_start[0] = 0;
    for (int r = 1; r < size; r++)
      index_start[r] = index_start[r - 1] + all_b[r - 1];
    long long ideal_index = (total_load / size) * rank;
    ideal_index += (rank < (total_load % size)) ? rank : (total_load % size);
    std::vector<std::vector<MPI_Block>> send_blocks(size);
    std::vector<std::vector<MPI_Block>> recv_blocks(size);
    for (int r = 0; r < size; r++)
      if (rank != r) {
        {
          const long long a1 = ideal_index;
          const long long a2 = ideal_index + my_load - 1;
          const long long b1 = index_start[r];
          const long long b2 = index_start[r] + all_b[r] - 1;
          const long long c1 = std::max(a1, b1);
          const long long c2 = std::min(a2, b2);
          if (c2 - c1 + 1 > 0)
            recv_blocks[r].resize(c2 - c1 + 1);
        }
        {
          long long other_ideal_index = (total_load / size) * r;
          other_ideal_index +=
              (r < (total_load % size)) ? r : (total_load % size);
          long long other_load = total_load / size;
          if (r < (total_load % size))
            other_load += 1;
          const long long a1 = other_ideal_index;
          const long long a2 = other_ideal_index + other_load - 1;
          const long long b1 = index_start[rank];
          const long long b2 = index_start[rank] + all_b[rank] - 1;
          const long long c1 = std::max(a1, b1);
          const long long c2 = std::min(a2, b2);
          if (c2 - c1 + 1 > 0)
            send_blocks[r].resize(c2 - c1 + 1);
        }
      }
    int tag = 12345;
    std::vector<MPI_Request> requests;
    for (int r = 0; r < size; r++)
      if (recv_blocks[r].size() != 0) {
        MPI_Request req{};
        requests.push_back(req);
        MPI_Irecv(recv_blocks[r].data(), recv_blocks[r].size(), MPI_BLOCK, r,
                  tag, grid->getWorldComm(), &requests.back());
      }
    long long counter_S = 0;
    long long counter_E = 0;
    for (int r = 0; r < rank; r++)
      if (send_blocks[r].size() != 0) {
        for (size_t i = 0; i < send_blocks[r].size(); i++)
          send_blocks[r][i].prepare(SortedInfos[counter_S + i]);
        counter_S += send_blocks[r].size();
        MPI_Request req{};
        requests.push_back(req);
        MPI_Isend(send_blocks[r].data(), send_blocks[r].size(), MPI_BLOCK, r,
                  tag, grid->getWorldComm(), &requests.back());
      }
    for (int r = size - 1; r > rank; r--)
      if (send_blocks[r].size() != 0) {
        for (size_t i = 0; i < send_blocks[r].size(); i++)
          send_blocks[r][i].prepare(
              SortedInfos[SortedInfos.size() - 1 - (counter_E + i)]);
        counter_E += send_blocks[r].size();
        MPI_Request req{};
        requests.push_back(req);
        MPI_Isend(send_blocks[r].data(), send_blocks[r].size(), MPI_BLOCK, r,
                  tag, grid->getWorldComm(), &requests.back());
      }
    movedBlocks = true;
    std::vector<long long> deallocIDs;
    counter_S = 0;
    counter_E = 0;
    for (int r = 0; r < size; r++)
      if (send_blocks[r].size() != 0) {
        if (r < rank) {
          for (size_t i = 0; i < send_blocks[r].size(); i++) {
            Info &info = SortedInfos[counter_S + i];
            deallocIDs.push_back(info.blockID_2);
            grid->Tree(info.level, info.Z).setrank(r);
          }
          counter_S += send_blocks[r].size();
        } else {
          for (size_t i = 0; i < send_blocks[r].size(); i++) {
            Info &info = SortedInfos[SortedInfos.size() - 1 - (counter_E + i)];
            deallocIDs.push_back(info.blockID_2);
            grid->Tree(info.level, info.Z).setrank(r);
          }
          counter_E += send_blocks[r].size();
        }
      }
    grid->dealloc_many(deallocIDs);
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
#pragma omp parallel
    {
      for (int r = 0; r < size; r++)
        if (recv_blocks[r].size() != 0) {
#pragma omp for
          for (size_t i = 0; i < recv_blocks[r].size(); i++)
            AddBlock(recv_blocks[r][i].mn[0], recv_blocks[r][i].mn[1],
                     recv_blocks[r][i].data);
        }
    }
    grid->FillPos();
  }
};
template <typename TLab> class MeshAdaptation {
protected:
  typedef typename TLab::GridType TGrid;
  typedef typename TGrid::Block BlockType;
  typedef typename TGrid::BlockType::ElementType ElementType;
  typedef typename TGrid::BlockType::ElementType::RealType Real;
  typedef SynchronizerMPI_AMR<Real, TGrid> SynchronizerMPIType;
  StencilInfo stencil;
  bool CallValidStates;
  bool boundary_needed;
  LoadBalancer<TGrid> *Balancer;
  TGrid *grid;
  double time;
  bool basic_refinement;
  double tolerance_for_refinement;
  double tolerance_for_compression;
  std::vector<long long> dealloc_IDs;

public:
  MeshAdaptation(TGrid &g, double Rtol, double Ctol) {
    grid = &g;
    tolerance_for_refinement = Rtol;
    tolerance_for_compression = Ctol;
    boundary_needed = false;
    constexpr int Gx = 1;
    constexpr int Gy = 1;
    constexpr int Gz = DIMENSION == 3 ? 1 : 0;
    stencil.sx = -Gx;
    stencil.sy = -Gy;
    stencil.sz = -Gz;
    stencil.ex = Gx + 1;
    stencil.ey = Gy + 1;
    stencil.ez = Gz + 1;
    stencil.tensorial = true;
    for (int i = 0; i < ElementType::DIM; i++)
      stencil.selcomponents.push_back(i);
    Balancer = new LoadBalancer<TGrid>(*grid);
  }
  virtual ~MeshAdaptation() { delete Balancer; }
  void Tag(double t = 0) {
    time = t;
    boundary_needed = true;
    SynchronizerMPI_AMR<Real, TGrid> *Synch = grid->sync(stencil);
    CallValidStates = false;
    bool Reduction = false;
    MPI_Request Reduction_req;
    int tmp;
    std::vector<Info *> &inner = Synch->avail_inner();
    TagBlocksVector(inner, Reduction, Reduction_req, tmp);
    std::vector<Info *> &halo = Synch->avail_halo();
    TagBlocksVector(halo, Reduction, Reduction_req, tmp);
    if (!Reduction) {
      tmp = CallValidStates ? 1 : 0;
      Reduction = true;
      MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM,
                     grid->getWorldComm(), &Reduction_req);
    }
    MPI_Wait(&Reduction_req, MPI_STATUS_IGNORE);
    CallValidStates = (tmp > 0);
    grid->boundary = halo;
    if (CallValidStates)
      ValidStates();
  }
  void Adapt(double t = 0, bool verbosity = false, bool basic = false) {
    basic_refinement = basic;
    SynchronizerMPI_AMR<Real, TGrid> *Synch = nullptr;
    if (basic == false) {
      Synch = grid->sync(stencil);
      grid->boundary = Synch->avail_halo();
      if (boundary_needed)
        grid->UpdateBoundary();
    }
    int r = 0;
    int c = 0;
    std::vector<int> m_com;
    std::vector<int> m_ref;
    std::vector<long long> n_com;
    std::vector<long long> n_ref;
    std::vector<Info> &I = grid->getBlocksInfo();
    long long blocks_after = I.size();
    for (auto &info : I) {
      if (info.state == Refine) {
        m_ref.push_back(info.level);
        n_ref.push_back(info.Z);
        blocks_after += (1 << DIMENSION) - 1;
        r++;
      } else if (info.state == Compress && info.index[0] % 2 == 0 &&
                 info.index[1] % 2 == 0 && info.index[2] % 2 == 0) {
        m_com.push_back(info.level);
        n_com.push_back(info.Z);
        c++;
      } else if (info.state == Compress) {
        blocks_after--;
      }
    }
    MPI_Request requests[2];
    int temp[2] = {r, c};
    int result[2];
    int size;
    MPI_Comm_size(grid->getWorldComm(), &size);
    std::vector<long long> block_distribution(size);
    MPI_Iallreduce(&temp, &result, 2, MPI_INT, MPI_SUM, grid->getWorldComm(),
                   &requests[0]);
    MPI_Iallgather(&blocks_after, 1, MPI_LONG_LONG, block_distribution.data(),
                   1, MPI_LONG_LONG, grid->getWorldComm(), &requests[1]);
    dealloc_IDs.clear();
    {
      TLab lab;
      if (Synch != nullptr)
        lab.prepare(*grid, Synch->getstencil());
      for (size_t i = 0; i < m_ref.size(); i++) {
        refine_1(m_ref[i], n_ref[i], lab);
      }
      for (size_t i = 0; i < m_ref.size(); i++) {
        refine_2(m_ref[i], n_ref[i]);
      }
    }
    grid->dealloc_many(dealloc_IDs);
    Balancer->PrepareCompression();
    dealloc_IDs.clear();
    for (size_t i = 0; i < m_com.size(); i++) {
      compress(m_com[i], n_com[i]);
    }
    grid->dealloc_many(dealloc_IDs);
    MPI_Waitall(2, requests, MPI_STATUS_IGNORE);
    Balancer->Balance_Diffusion(block_distribution);
    if (result[0] > 0 || result[1] > 0 || Balancer->movedBlocks) {
      grid->UpdateFluxCorrection = true;
      grid->UpdateGroups = true;
      grid->UpdateInfoAll_States(false);
      auto it = grid->SynchronizerMPIs.begin();
      while (it != grid->SynchronizerMPIs.end()) {
        (*it->second)._Setup();
        it++;
      }
    }
  }
  void TagLike(const std::vector<Info> &I1) {
    std::vector<Info> &I2 = grid->getBlocksInfo();
    for (size_t i1 = 0; i1 < I2.size(); i1++) {
      Info &ary0 = I2[i1];
      Info &info = grid->getInfoAll(ary0.level, ary0.Z);
      for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
           i++)
        for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1;
             j++)
          for (int k = 2 * (info.index[2] / 2);
               k <= 2 * (info.index[2] / 2) + 1; k++) {
            const long long n = grid->getZforward(info.level, i, j, k);
            Info &infoNei = grid->getInfoAll(info.level, n);
            infoNei.state = Leave;
          }
      info.state = Leave;
      ary0.state = Leave;
    }
#pragma omp parallel for
    for (size_t i = 0; i < I1.size(); i++) {
      const Info &info1 = I1[i];
      Info &info2 = I2[i];
      Info &info3 = grid->getInfoAll(info2.level, info2.Z);
      info2.state = info1.state;
      info3.state = info1.state;
      if (info2.state == Compress) {
        const int i2 = 2 * (info2.index[0] / 2);
        const int j2 = 2 * (info2.index[1] / 2);
        const int k2 = 2 * (info2.index[2] / 2);
        const long long n = grid->getZforward(info2.level, i2, j2, k2);
        Info &infoNei = grid->getInfoAll(info2.level, n);
        infoNei.state = Compress;
      }
    }
  }

protected:
  void TagBlocksVector(std::vector<Info *> &I, bool &Reduction,
                       MPI_Request &Reduction_req, int &tmp) {
    const int levelMax = grid->getlevelMax();
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < I.size(); i++) {
        Info &info = grid->getInfoAll(I[i]->level, I[i]->Z);
        I[i]->state = TagLoadedBlock(info);
        const bool maxLevel =
            (I[i]->state == Refine) && (I[i]->level == levelMax - 1);
        const bool minLevel = (I[i]->state == Compress) && (I[i]->level == 0);
        if (maxLevel || minLevel)
          I[i]->state = Leave;
        info.state = I[i]->state;
        if (info.state != Leave) {
#pragma omp critical
          {
            CallValidStates = true;
            if (!Reduction) {
              tmp = 1;
              Reduction = true;
              MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM,
                             grid->getWorldComm(), &Reduction_req);
            }
          }
        }
      }
    }
  }
  void refine_1(const int level, const long long Z, TLab &lab) {
    Info &parent = grid->getInfoAll(level, Z);
    parent.state = Leave;
    if (basic_refinement == false)
      lab.load(parent, time, true);
    const int p[3] = {parent.index[0], parent.index[1], parent.index[2]};
    assert(parent.block != NULL);
    assert(level <= grid->getlevelMax() - 1);
    BlockType *Blocks[8];
    for (int k = 0; k < 2; k++)
      for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
          const long long nc = grid->getZforward(level + 1, 2 * p[0] + i,
                                                 2 * p[1] + j, 2 * p[2] + k);
          Info &Child = grid->getInfoAll(level + 1, nc);
          Child.state = Leave;
          grid->_alloc(level + 1, nc);
          grid->Tree(level + 1, nc).setCheckCoarser();
          Blocks[k * 4 + j * 2 + i] = (BlockType *)Child.block;
        }
    if (basic_refinement == false)
      RefineBlocks(Blocks, lab);
  }
  void refine_2(const int level, const long long Z) {
#pragma omp critical
    { dealloc_IDs.push_back(grid->getInfoAll(level, Z).blockID_2); }
    Info &parent = grid->getInfoAll(level, Z);
    grid->Tree(parent).setCheckFiner();
    parent.state = Leave;
    int p[3] = {parent.index[0], parent.index[1], parent.index[2]};
    for (int k = 0; k < 2; k++)
      for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
          const long long nc = grid->getZforward(level + 1, 2 * p[0] + i,
                                                 2 * p[1] + j, 2 * p[2] + k);
          Info &Child = grid->getInfoAll(level + 1, nc);
          grid->Tree(Child).setrank(grid->rank());
          if (level + 2 < grid->getlevelMax())
            for (int i0 = 0; i0 < 2; i0++)
              for (int i1 = 0; i1 < 2; i1++)
                for (int i2 = 0; i2 < 2; i2++)
                  grid->Tree(level + 2, Child.Zchild[i0][i1][i2])
                      .setCheckCoarser();
        }
  }
  void compress(const int level, const long long Z) {
    assert(level > 0);
    Info &info = grid->getInfoAll(level, Z);
    assert(info.state == Compress);
    BlockType *Blocks[8];
    for (int K = 0; K < 2; K++)
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          const int blk = K * 4 + J * 2 + I;
          const long long n = grid->getZforward(
              level, info.index[0] + I, info.index[1] + J, info.index[2] + K);
          Blocks[blk] = (BlockType *)(grid->getInfoAll(level, n)).block;
        }
    const int nx = BlockType::sizeX;
    const int ny = BlockType::sizeY;
    const int nz = BlockType::sizeZ;
    const int offsetX[2] = {0, nx / 2};
    const int offsetY[2] = {0, ny / 2};
    const int offsetZ[2] = {0, nz / 2};
    if (basic_refinement == false)
      for (int K = 0; K < 2; K++)
        for (int J = 0; J < 2; J++)
          for (int I = 0; I < 2; I++) {
            BlockType &b = *Blocks[K * 4 + J * 2 + I];
            for (int k = 0; k < nz; k += 2)
              for (int j = 0; j < ny; j += 2)
                for (int i = 0; i < nx; i += 2) {
                  (*Blocks[0])(i / 2 + offsetX[I], j / 2 + offsetY[J],
                               k / 2 + offsetZ[K]) =
                      0.125 * ((b(i, j, k) + b(i + 1, j + 1, k + 1)) +
                               (b(i + 1, j, k) + b(i, j + 1, k + 1)) +
                               (b(i, j + 1, k) + b(i + 1, j, k + 1)) +
                               (b(i + 1, j + 1, k) + b(i, j, k + 1)));
                }
          }
    const long long np = grid->getZforward(
        level - 1, info.index[0] / 2, info.index[1] / 2, info.index[2] / 2);
    Info &parent = grid->getInfoAll(level - 1, np);
    grid->Tree(parent.level, parent.Z).setrank(grid->rank());
    parent.block = info.block;
    parent.state = Leave;
    if (level - 2 >= 0)
      grid->Tree(level - 2, parent.Zparent).setCheckFiner();
    for (int K = 0; K < 2; K++)
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          const long long n = grid->getZforward(
              level, info.index[0] + I, info.index[1] + J, info.index[2] + K);
          if (I + J + K == 0) {
            grid->FindInfo(level, n, level - 1, np);
          } else {
#pragma omp critical
            { dealloc_IDs.push_back(grid->getInfoAll(level, n).blockID_2); }
          }
          grid->Tree(level, n).setCheckCoarser();
          grid->getInfoAll(level, n).state = Leave;
        }
  }
  void ValidStates() {
    const std::array<int, 3> blocksPerDim = grid->getMaxBlocks();
    const int levelMin = 0;
    const int levelMax = grid->getlevelMax();
    const bool xperiodic = grid->xperiodic;
    const bool yperiodic = grid->yperiodic;
    const bool zperiodic = grid->zperiodic;
    std::vector<Info> &I = grid->getBlocksInfo();
#pragma omp parallel for
    for (size_t j = 0; j < I.size(); j++) {
      Info &info = I[j];
      if ((info.state == Refine && info.level == levelMax - 1) ||
          (info.state == Compress && info.level == levelMin)) {
        info.state = Leave;
        (grid->getInfoAll(info.level, info.Z)).state = Leave;
      }
      if (info.state != Leave) {
        info.changed2 = true;
        (grid->getInfoAll(info.level, info.Z)).changed2 = info.changed2;
      }
    }
    bool clean_boundary = true;
    for (int m = levelMax - 1; m >= levelMin; m--) {
      for (size_t j = 0; j < I.size(); j++) {
        Info &info = I[j];
        if (info.level == m && info.state != Refine &&
            info.level != levelMax - 1) {
          const int TwoPower = 1 << info.level;
          const bool xskin = info.index[0] == 0 ||
                             info.index[0] == blocksPerDim[0] * TwoPower - 1;
          const bool yskin = info.index[1] == 0 ||
                             info.index[1] == blocksPerDim[1] * TwoPower - 1;
          const bool zskin = info.index[2] == 0 ||
                             info.index[2] == blocksPerDim[2] * TwoPower - 1;
          const int xskip = info.index[0] == 0 ? -1 : 1;
          const int yskip = info.index[1] == 0 ? -1 : 1;
          const int zskip = info.index[2] == 0 ? -1 : 1;
          for (int icode = 0; icode < 27; icode++) {
            if (info.state == Refine)
              break;
            if (icode == 1 * 1 + 3 * 1 + 9 * 1)
              continue;
            const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                                 (icode / 9) % 3 - 1};
            if (!xperiodic && code[0] == xskip && xskin)
              continue;
            if (!yperiodic && code[1] == yskip && yskin)
              continue;
            if (!zperiodic && code[2] == zskip && zskin)
              continue;
            if (grid->Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                    .CheckFiner()) {
              if (info.state == Compress) {
                info.state = Leave;
                (grid->getInfoAll(info.level, info.Z)).state = Leave;
              }
              const int tmp = abs(code[0]) + abs(code[1]) + abs(code[2]);
              int Bstep = 1;
              if (tmp == 2)
                Bstep = 3;
              else if (tmp == 3)
                Bstep = 4;
              for (int B = 0; B <= 3; B += Bstep) {
                const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                const int iNei = 2 * info.index[0] + std::max(code[0], 0) +
                                 code[0] +
                                 (B % 2) * std::max(0, 1 - abs(code[0]));
                const int jNei = 2 * info.index[1] + std::max(code[1], 0) +
                                 code[1] + aux * std::max(0, 1 - abs(code[1]));
                const int kNei = 2 * info.index[2] + std::max(code[2], 0) +
                                 code[2] +
                                 (B / 2) * std::max(0, 1 - abs(code[2]));
                const long long zzz =
                    grid->getZforward(m + 1, iNei, jNei, kNei);
                Info &FinerNei = grid->getInfoAll(m + 1, zzz);
                State NeiState = FinerNei.state;
                if (NeiState == Refine) {
                  info.state = Refine;
                  (grid->getInfoAll(info.level, info.Z)).state = Refine;
                  info.changed2 = true;
                  (grid->getInfoAll(info.level, info.Z)).changed2 = true;
                  break;
                }
              }
            }
          }
        }
      }
      grid->UpdateBoundary(clean_boundary);
      clean_boundary = false;
      if (m == levelMin)
        break;
      for (size_t j = 0; j < I.size(); j++) {
        Info &info = I[j];
        if (info.level == m && info.state == Compress) {
          const int aux = 1 << info.level;
          const bool xskin =
              info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
          const bool yskin =
              info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
          const bool zskin =
              info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
          const int xskip = info.index[0] == 0 ? -1 : 1;
          const int yskip = info.index[1] == 0 ? -1 : 1;
          const int zskip = info.index[2] == 0 ? -1 : 1;
          for (int icode = 0; icode < 27; icode++) {
            if (icode == 1 * 1 + 3 * 1 + 9 * 1)
              continue;
            const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                                 (icode / 9) % 3 - 1};
            if (!xperiodic && code[0] == xskip && xskin)
              continue;
            if (!yperiodic && code[1] == yskip && yskin)
              continue;
            if (!zperiodic && code[2] == zskip && zskin)
              continue;
            Info &infoNei = grid->getInfoAll(
                info.level, info.Znei_(code[0], code[1], code[2]));
            if (grid->Tree(infoNei).Exists() && infoNei.state == Refine) {
              info.state = Leave;
              (grid->getInfoAll(info.level, info.Z)).state = Leave;
              break;
            }
          }
        }
      }
    }
    for (size_t jjj = 0; jjj < I.size(); jjj++) {
      Info &info = I[jjj];
      const int m = info.level;
      bool found = false;
      for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
           i++)
        for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1;
             j++)
          for (int k = 2 * (info.index[2] / 2);
               k <= 2 * (info.index[2] / 2) + 1; k++) {
            const long long n = grid->getZforward(m, i, j, k);
            Info &infoNei = grid->getInfoAll(m, n);
            if (grid->Tree(infoNei).Exists() == false ||
                infoNei.state != Compress) {
              found = true;
              if (info.state == Compress) {
                info.state = Leave;
                (grid->getInfoAll(info.level, info.Z)).state = Leave;
              }
              break;
            }
          }
      if (found)
        for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
             i++)
          for (int j = 2 * (info.index[1] / 2);
               j <= 2 * (info.index[1] / 2) + 1; j++)
            for (int k = 2 * (info.index[2] / 2);
                 k <= 2 * (info.index[2] / 2) + 1; k++) {
              const long long n = grid->getZforward(m, i, j, k);
              Info &infoNei = grid->getInfoAll(m, n);
              if (grid->Tree(infoNei).Exists() && infoNei.state == Compress)
                infoNei.state = Leave;
            }
    }
  }
  virtual void RefineBlocks(BlockType *B[8], TLab &Lab) {
    const int nx = BlockType::sizeX;
    const int ny = BlockType::sizeY;
    int offsetX[2] = {0, nx / 2};
    int offsetY[2] = {0, ny / 2};
    const int nz = BlockType::sizeZ;
    int offsetZ[2] = {0, nz / 2};
    for (int K = 0; K < 2; K++)
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          BlockType &b = *B[K * 4 + J * 2 + I];
          b.clear();
          for (int k = 0; k < nz; k += 2)
            for (int j = 0; j < ny; j += 2)
              for (int i = 0; i < nx; i += 2) {
                const int x = i / 2 + offsetX[I];
                const int y = j / 2 + offsetY[J];
                const int z = k / 2 + offsetZ[K];
                const ElementType dudx =
                    0.5 * (Lab(x + 1, y, z) - Lab(x - 1, y, z));
                const ElementType dudy =
                    0.5 * (Lab(x, y + 1, z) - Lab(x, y - 1, z));
                const ElementType dudz =
                    0.5 * (Lab(x, y, z + 1) - Lab(x, y, z - 1));
                const ElementType dudx2 =
                    (Lab(x + 1, y, z) + Lab(x - 1, y, z)) - 2.0 * Lab(x, y, z);
                const ElementType dudy2 =
                    (Lab(x, y + 1, z) + Lab(x, y - 1, z)) - 2.0 * Lab(x, y, z);
                const ElementType dudz2 =
                    (Lab(x, y, z + 1) + Lab(x, y, z - 1)) - 2.0 * Lab(x, y, z);
                const ElementType dudxdy =
                    0.25 * ((Lab(x + 1, y + 1, z) + Lab(x - 1, y - 1, z)) -
                            (Lab(x + 1, y - 1, z) + Lab(x - 1, y + 1, z)));
                const ElementType dudxdz =
                    0.25 * ((Lab(x + 1, y, z + 1) + Lab(x - 1, y, z - 1)) -
                            (Lab(x + 1, y, z - 1) + Lab(x - 1, y, z + 1)));
                const ElementType dudydz =
                    0.25 * ((Lab(x, y + 1, z + 1) + Lab(x, y - 1, z - 1)) -
                            (Lab(x, y + 1, z - 1) + Lab(x, y - 1, z + 1)));
                b(i, j, k) = Lab(x, y, z) +
                             0.25 * (-(1.0) * dudx - dudy - dudz) +
                             0.03125 * (dudx2 + dudy2 + dudz2) +
                             0.0625 * (dudxdy + dudxdz + dudydz);
                b(i + 1, j, k) = Lab(x, y, z) + 0.25 * (dudx - dudy - dudz) +
                                 0.03125 * (dudx2 + dudy2 + dudz2) +
                                 0.0625 * (-(1.0) * dudxdy - dudxdz + dudydz);
                b(i, j + 1, k) = Lab(x, y, z) +
                                 0.25 * (-(1.0) * dudx + dudy - dudz) +
                                 0.03125 * (dudx2 + dudy2 + dudz2) +
                                 0.0625 * (-(1.0) * dudxdy + dudxdz - dudydz);
                b(i + 1, j + 1, k) = Lab(x, y, z) +
                                     0.25 * (dudx + dudy - dudz) +
                                     0.03125 * (dudx2 + dudy2 + dudz2) +
                                     0.0625 * (dudxdy - dudxdz - dudydz);
                b(i, j, k + 1) = Lab(x, y, z) +
                                 0.25 * (-(1.0) * dudx - dudy + dudz) +
                                 0.03125 * (dudx2 + dudy2 + dudz2) +
                                 0.0625 * (dudxdy - dudxdz - dudydz);
                b(i + 1, j, k + 1) =
                    Lab(x, y, z) + 0.25 * (dudx - dudy + dudz) +
                    0.03125 * (dudx2 + dudy2 + dudz2) +
                    0.0625 * (-(1.0) * dudxdy + dudxdz - dudydz);
                b(i, j + 1, k + 1) =
                    Lab(x, y, z) + 0.25 * (-(1.0) * dudx + dudy + dudz) +
                    0.03125 * (dudx2 + dudy2 + dudz2) +
                    0.0625 * (-(1.0) * dudxdy - dudxdz + dudydz);
                b(i + 1, j + 1, k + 1) = Lab(x, y, z) +
                                         0.25 * (dudx + dudy + dudz) +
                                         0.03125 * (dudx2 + dudy2 + dudz2) +
                                         0.0625 * (dudxdy + dudxdz + dudydz);
              }
        }
  }
  virtual State TagLoadedBlock(Info &info) {
    const int nx = BlockType::sizeX;
    const int ny = BlockType::sizeY;
    BlockType &b = *(BlockType *)info.block;
    double Linf = 0.0;
    const int nz = BlockType::sizeZ;
    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++) {
          Linf = std::max(Linf, std::fabs(b(i, j, k).magnitude()));
        }
    if (Linf > tolerance_for_refinement)
      return Refine;
    else if (Linf < tolerance_for_compression)
      return Compress;
    return Leave;
  }
};
template <typename Lab, typename Kernel, typename TGrid,
          typename TGrid_corr = TGrid>
void compute(Kernel &&kernel, TGrid *g, TGrid_corr *g_corr = nullptr) {
  if (g_corr != nullptr)
    g_corr->Corrector.prepare(*g_corr);
  SynchronizerMPI_AMR<typename TGrid::Real, TGrid> &Synch =
      *(g->sync(kernel.stencil));
  std::vector<Info *> *inner = &Synch.avail_inner();
  std::vector<Info *> *halo_next;
  bool done = false;
#pragma omp parallel
  {
    Lab lab;
    lab.prepare(*g, kernel.stencil);
#pragma omp for nowait
    for (const auto &I : *inner) {
      lab.load(*I, 0);
      kernel(lab, *I);
    }
#if 1
    while (done == false) {
#pragma omp master
      halo_next = &Synch.avail_next();
#pragma omp barrier
#pragma omp for nowait
      for (const auto &I : *halo_next) {
        lab.load(*I, 0);
        kernel(lab, *I);
      }
#pragma omp single
      {
        if (halo_next->size() == 0)
          done = true;
      }
    }
#else
    std::vector<Info> &blk = g->getBlocksInfo();
    std::vector<bool> ready(blk.size(), false);
    std::vector<Info *> &avail1 = Synch.avail_halo_nowait();
    const int Nhalo = avail1.size();
    while (done == false) {
      done = true;
      for (int i = 0; i < Nhalo; i++) {
        const Info &I = *avail1[i];
        if (ready[I.blockID] == false) {
          if (Synch.isready(I)) {
            ready[I.blockID] = true;
            lab.load(I, 0);
            kernel(lab, I);
          } else {
            done = false;
          }
        }
      }
    }
#endif
  }
  Synch.avail_halo();
  if (g_corr != nullptr)
    g_corr->Corrector.FillBlockCases();
}
template <typename Kernel, typename TGrid, typename LabMPI, typename TGrid2,
          typename LabMPI2, typename TGrid_corr = TGrid>
static void compute(const Kernel &kernel, TGrid &grid, TGrid2 &grid2,
                    const bool applyFluxCorrection = false,
                    TGrid_corr *corrected_grid = nullptr) {
  if (applyFluxCorrection)
    corrected_grid->Corrector.prepare(*corrected_grid);
  SynchronizerMPI_AMR<typename TGrid::Real, TGrid> &Synch =
      *grid.sync(kernel.stencil);
  Kernel kernel2 = kernel;
  kernel2.stencil.sx = kernel2.stencil2.sx;
  kernel2.stencil.sy = kernel2.stencil2.sy;
  kernel2.stencil.sz = kernel2.stencil2.sz;
  kernel2.stencil.ex = kernel2.stencil2.ex;
  kernel2.stencil.ey = kernel2.stencil2.ey;
  kernel2.stencil.ez = kernel2.stencil2.ez;
  kernel2.stencil.tensorial = kernel2.stencil2.tensorial;
  kernel2.stencil.selcomponents.clear();
  kernel2.stencil.selcomponents = kernel2.stencil2.selcomponents;
  SynchronizerMPI_AMR<typename TGrid::Real, TGrid2> &Synch2 =
      *grid2.sync(kernel2.stencil);
  const StencilInfo &stencil = Synch.getstencil();
  const StencilInfo &stencil2 = Synch2.getstencil();
  std::vector<Info> &blk = grid.getBlocksInfo();
  std::vector<bool> ready(blk.size(), false);
  std::vector<Info *> &avail0 = Synch.avail_inner();
  std::vector<Info *> &avail02 = Synch2.avail_inner();
  const int Ninner = avail0.size();
  std::vector<Info *> avail1;
  std::vector<Info *> avail12;
#pragma omp parallel
  {
    LabMPI lab;
    LabMPI2 lab2;
    lab.prepare(grid, stencil);
    lab2.prepare(grid2, stencil2);
#pragma omp for
    for (int i = 0; i < Ninner; i++) {
      const Info &I = *avail0[i];
      const Info &I2 = *avail02[i];
      lab.load(I, 0);
      lab2.load(I2, 0);
      kernel(lab, lab2, I, I2);
      ready[I.blockID] = true;
    }
#if 1
#pragma omp master
    {
      avail1 = Synch.avail_halo();
      avail12 = Synch2.avail_halo();
    }
#pragma omp barrier
    const int Nhalo = avail1.size();
#pragma omp for
    for (int i = 0; i < Nhalo; i++) {
      const Info &I = *avail1[i];
      const Info &I2 = *avail12[i];
      lab.load(I, 0);
      lab2.load(I2, 0);
      kernel(lab, lab2, I, I2);
    }
#else
#pragma omp master
    {
      avail1 = Synch.avail_halo_nowait();
      avail12 = Synch2.avail_halo_nowait();
    }
#pragma omp barrier
    const int Nhalo = avail1.size();
    while (done == false) {
#pragma omp barrier
#pragma omp single
      done = true;
#pragma omp for
      for (int i = 0; i < Nhalo; i++) {
        const Info &I = *avail1[i];
        const Info &I2 = *avail12[i];
        if (ready[I.blockID] == false) {
          bool blockready;
#pragma omp critical
          { blockready = (Synch.isready(I) && Synch.isready(I2)); }
          if (blockready) {
            ready[I.blockID] = true;
            lab.load(I, 0);
            lab2.load(I2, 0);
            kernel(lab, lab2, I, I2);
          } else {
#pragma omp atomic write
            done = false;
          }
        }
      }
    }
    avail1 = Synch.avail_halo();
    avail12 = Synch2.avail_halo();
#endif
  }
  if (applyFluxCorrection)
    corrected_grid->Corrector.FillBlockCases();
}
struct ScalarElement {
  using RealType = Real;
  Real s = 0;
  inline void clear() { s = 0; }
  inline void set(const Real v) { s = v; }
  inline void copy(const ScalarElement &c) { s = c.s; }
  ScalarElement &operator*=(const Real a) {
    this->s *= a;
    return *this;
  }
  ScalarElement &operator+=(const ScalarElement &rhs) {
    this->s += rhs.s;
    return *this;
  }
  ScalarElement &operator-=(const ScalarElement &rhs) {
    this->s -= rhs.s;
    return *this;
  }
  ScalarElement &operator/=(const ScalarElement &rhs) {
    this->s /= rhs.s;
    return *this;
  }
  friend ScalarElement operator*(const Real a, ScalarElement el) {
    return (el *= a);
  }
  friend ScalarElement operator+(ScalarElement lhs, const ScalarElement &rhs) {
    return (lhs += rhs);
  }
  friend ScalarElement operator-(ScalarElement lhs, const ScalarElement &rhs) {
    return (lhs -= rhs);
  }
  friend ScalarElement operator/(ScalarElement lhs, const ScalarElement &rhs) {
    return (lhs /= rhs);
  }
  bool operator<(const ScalarElement &other) const { return (s < other.s); }
  bool operator>(const ScalarElement &other) const { return (s > other.s); }
  bool operator<=(const ScalarElement &other) const { return (s <= other.s); }
  bool operator>=(const ScalarElement &other) const { return (s >= other.s); }
  Real magnitude() { return s; }
  Real &member(int i) { return s; }
  static constexpr int DIM = 1;
};
struct VectorElement {
  using RealType = Real;
  static constexpr int DIM = 3;
  Real u[DIM];
  VectorElement() { clear(); }
  inline void clear() {
    for (int i = 0; i < DIM; ++i)
      u[i] = 0;
  }
  inline void set(const Real v) {
    for (int i = 0; i < DIM; ++i)
      u[i] = v;
  }
  inline void copy(const VectorElement &c) {
    for (int i = 0; i < DIM; ++i)
      u[i] = c.u[i];
  }
  VectorElement &operator=(const VectorElement &c) = default;
  VectorElement &operator*=(const Real a) {
    for (int i = 0; i < DIM; ++i)
      this->u[i] *= a;
    return *this;
  }
  VectorElement &operator+=(const VectorElement &rhs) {
    for (int i = 0; i < DIM; ++i)
      this->u[i] += rhs.u[i];
    return *this;
  }
  VectorElement &operator-=(const VectorElement &rhs) {
    for (int i = 0; i < DIM; ++i)
      this->u[i] -= rhs.u[i];
    return *this;
  }
  VectorElement &operator/=(const VectorElement &rhs) {
    for (int i = 0; i < DIM; ++i)
      this->u[i] /= rhs.u[i];
    return *this;
  }
  friend VectorElement operator*(const Real a, VectorElement el) {
    return (el *= a);
  }
  friend VectorElement operator+(VectorElement lhs, const VectorElement &rhs) {
    return (lhs += rhs);
  }
  friend VectorElement operator-(VectorElement lhs, const VectorElement &rhs) {
    return (lhs -= rhs);
  }
  friend VectorElement operator/(VectorElement lhs, const VectorElement &rhs) {
    return (lhs /= rhs);
  }
  bool operator<(const VectorElement &other) const {
    Real s1 = 0.0;
    Real s2 = 0.0;
    for (int i = 0; i < DIM; ++i) {
      s1 += u[i] * u[i];
      s2 += other.u[i] * other.u[i];
    }
    return (s1 < s2);
  }
  bool operator>(const VectorElement &other) const {
    Real s1 = 0.0;
    Real s2 = 0.0;
    for (int i = 0; i < DIM; ++i) {
      s1 += u[i] * u[i];
      s2 += other.u[i] * other.u[i];
    }
    return (s1 > s2);
  }
  bool operator<=(const VectorElement &other) const {
    Real s1 = 0.0;
    Real s2 = 0.0;
    for (int i = 0; i < DIM; ++i) {
      s1 += u[i] * u[i];
      s2 += other.u[i] * other.u[i];
    }
    return (s1 <= s2);
  }
  bool operator>=(const VectorElement &other) const {
    Real s1 = 0.0;
    Real s2 = 0.0;
    for (int i = 0; i < DIM; ++i) {
      s1 += u[i] * u[i];
      s2 += other.u[i] * other.u[i];
    }
    return (s1 >= s2);
  }
  Real magnitude() {
    Real s1 = 0.0;
    for (int i = 0; i < DIM; ++i) {
      s1 += u[i] * u[i];
    }
    return sqrt(s1);
  }
  Real &member(int i) { return u[i]; }
};
template <int blocksize, int dim, typename TElement> struct GridBlock {
  static constexpr int BS = blocksize;
  static constexpr int sizeX = blocksize;
  static constexpr int sizeY = blocksize;
  static constexpr int sizeZ = dim > 2 ? blocksize : 1;
  static constexpr std::array<int, 3> sizeArray = {sizeX, sizeY, sizeZ};
  using ElementType = TElement;
  using RealType = typename TElement::RealType;
  ElementType data[sizeZ][sizeY][sizeX];
  inline void clear() {
    ElementType *const entry = &data[0][0][0];
    for (int i = 0; i < sizeX * sizeY * sizeZ; ++i)
      entry[i].clear();
  }
  inline void set(const RealType v) {
    ElementType *const entry = &data[0][0][0];
    for (int i = 0; i < sizeX * sizeY * sizeZ; ++i)
      entry[i].set(v);
  }
  inline void copy(const GridBlock<blocksize, dim, ElementType> &c) {
    ElementType *const entry = &data[0][0][0];
    const ElementType *const source = &c.data[0][0][0];
    for (int i = 0; i < sizeX * sizeY * sizeZ; ++i)
      entry[i].copy(source[i]);
  }
  const ElementType &operator()(int ix, int iy = 0, int iz = 0) const {
    assert(ix >= 0 && iy >= 0 && iz >= 0 && ix < sizeX && iy < sizeY &&
           iz < sizeZ);
    return data[iz][iy][ix];
  }
  ElementType &operator()(int ix, int iy = 0, int iz = 0) {
    assert(ix >= 0 && iy >= 0 && iz >= 0 && ix < sizeX && iy < sizeY &&
           iz < sizeZ);
    return data[iz][iy][ix];
  }
  GridBlock(const GridBlock &) = delete;
  GridBlock &operator=(const GridBlock &) = delete;
};
template <typename TGrid, int dim,
          template <typename X> class allocator = std::allocator>
class BlockLabNeumann : public BlockLab<TGrid, allocator> {
  static constexpr int sizeX = TGrid::BlockType::sizeX;
  static constexpr int sizeY = TGrid::BlockType::sizeY;
  static constexpr int sizeZ = TGrid::BlockType::sizeZ;
  static constexpr int DIM = dim;

protected:
  template <int dir, int side> void Neumann3D(const bool coarse = false) {
    int stenBeg[3];
    int stenEnd[3];
    int bsize[3];
    if (!coarse) {
      stenEnd[0] = this->m_stencilEnd[0];
      stenEnd[1] = this->m_stencilEnd[1];
      stenEnd[2] = this->m_stencilEnd[2];
      stenBeg[0] = this->m_stencilStart[0];
      stenBeg[1] = this->m_stencilStart[1];
      stenBeg[2] = this->m_stencilStart[2];
      bsize[0] = sizeX;
      bsize[1] = sizeY;
      bsize[2] = sizeZ;
    } else {
      stenEnd[0] =
          (this->m_stencilEnd[0]) / 2 + 1 + this->m_InterpStencilEnd[0] - 1;
      stenEnd[1] =
          (this->m_stencilEnd[1]) / 2 + 1 + this->m_InterpStencilEnd[1] - 1;
      stenEnd[2] =
          (this->m_stencilEnd[2]) / 2 + 1 + this->m_InterpStencilEnd[2] - 1;
      stenBeg[0] =
          (this->m_stencilStart[0] - 1) / 2 + this->m_InterpStencilStart[0];
      stenBeg[1] =
          (this->m_stencilStart[1] - 1) / 2 + this->m_InterpStencilStart[1];
      stenBeg[2] =
          (this->m_stencilStart[2] - 1) / 2 + this->m_InterpStencilStart[2];
      bsize[0] = sizeX / 2;
      bsize[1] = sizeY / 2;
      bsize[2] = sizeZ / 2;
    }
    auto *const cb = coarse ? this->m_CoarsenedBlock : this->m_cacheBlock;
    int s[3];
    int e[3];
    s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : bsize[0]) : 0;
    s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : bsize[1]) : 0;
    s[2] = dir == 2 ? (side == 0 ? stenBeg[2] : bsize[2]) : 0;
    e[0] = dir == 0 ? (side == 0 ? 0 : bsize[0] + stenEnd[0] - 1) : bsize[0];
    e[1] = dir == 1 ? (side == 0 ? 0 : bsize[1] + stenEnd[1] - 1) : bsize[1];
    e[2] = dir == 2 ? (side == 0 ? 0 : bsize[2] + stenEnd[2] - 1) : bsize[2];
    for (int iz = s[2]; iz < e[2]; iz++)
      for (int iy = s[1]; iy < e[1]; iy++)
        for (int ix = s[0]; ix < e[0]; ix++) {
          cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2]) =
              cb->Access(
                  (dir == 0 ? (side == 0 ? 0 : bsize[0] - 1) : ix) - stenBeg[0],
                  (dir == 1 ? (side == 0 ? 0 : bsize[1] - 1) : iy) - stenBeg[1],
                  (dir == 2 ? (side == 0 ? 0 : bsize[2] - 1) : iz) -
                      stenBeg[2]);
        }
    s[dir] = stenBeg[dir] * (1 - side) + bsize[dir] * side;
    e[dir] = (bsize[dir] - 1 + stenEnd[dir]) * side;
    const int d1 = (dir + 1) % 3;
    const int d2 = (dir + 2) % 3;
    for (int b = 0; b < 2; ++b)
      for (int a = 0; a < 2; ++a) {
        s[d1] = stenBeg[d1] + a * b * (bsize[d1] - stenBeg[d1]);
        s[d2] = stenBeg[d2] + (a - a * b) * (bsize[d2] - stenBeg[d2]);
        e[d1] = (1 - b + a * b) * (bsize[d1] - 1 + stenEnd[d1]);
        e[d2] = (a + b - a * b) * (bsize[d2] - 1 + stenEnd[d2]);
        for (int iz = s[2]; iz < e[2]; iz++)
          for (int iy = s[1]; iy < e[1]; iy++)
            for (int ix = s[0]; ix < e[0]; ix++) {
              cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2]) =
                  dir == 0
                      ? cb->Access(side * (bsize[0] - 1) - stenBeg[0],
                                   iy - stenBeg[1], iz - stenBeg[2])
                      : (dir == 1
                             ? cb->Access(ix - stenBeg[0],
                                          side * (bsize[1] - 1) - stenBeg[1],
                                          iz - stenBeg[2])
                             : cb->Access(ix - stenBeg[0], iy - stenBeg[1],
                                          side * (bsize[2] - 1) - stenBeg[2]));
            }
      }
  }
  template <int dir, int side> void Neumann2D(const bool coarse = false) {
    int stenBeg[2];
    int stenEnd[2];
    int bsize[2];
    if (!coarse) {
      stenEnd[0] = this->m_stencilEnd[0];
      stenEnd[1] = this->m_stencilEnd[1];
      stenBeg[0] = this->m_stencilStart[0];
      stenBeg[1] = this->m_stencilStart[1];
      bsize[0] = sizeX;
      bsize[1] = sizeY;
    } else {
      stenEnd[0] =
          (this->m_stencilEnd[0]) / 2 + 1 + this->m_InterpStencilEnd[0] - 1;
      stenEnd[1] =
          (this->m_stencilEnd[1]) / 2 + 1 + this->m_InterpStencilEnd[1] - 1;
      stenBeg[0] =
          (this->m_stencilStart[0] - 1) / 2 + this->m_InterpStencilStart[0];
      stenBeg[1] =
          (this->m_stencilStart[1] - 1) / 2 + this->m_InterpStencilStart[1];
      bsize[0] = sizeX / 2;
      bsize[1] = sizeY / 2;
    }
    auto *const cb = coarse ? this->m_CoarsenedBlock : this->m_cacheBlock;
    int s[2];
    int e[2];
    s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : bsize[0]) : stenBeg[0];
    s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : bsize[1]) : stenBeg[1];
    e[0] = dir == 0 ? (side == 0 ? 0 : bsize[0] + stenEnd[0] - 1)
                    : bsize[0] + stenEnd[0] - 1;
    e[1] = dir == 1 ? (side == 0 ? 0 : bsize[1] + stenEnd[1] - 1)
                    : bsize[1] + stenEnd[1] - 1;
    for (int iy = s[1]; iy < e[1]; iy++)
      for (int ix = s[0]; ix < e[0]; ix++)
        cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0) = cb->Access(
            (dir == 0 ? (side == 0 ? 0 : bsize[0] - 1) : ix) - stenBeg[0],
            (dir == 1 ? (side == 0 ? 0 : bsize[1] - 1) : iy) - stenBeg[1], 0);
  }

public:
  typedef typename TGrid::BlockType::ElementType ElementTypeBlock;
  typedef typename TGrid::BlockType::ElementType ElementType;
  using Real = typename ElementType::RealType;
  virtual bool is_xperiodic() override { return false; }
  virtual bool is_yperiodic() override { return false; }
  virtual bool is_zperiodic() override { return false; }
  BlockLabNeumann() = default;
  BlockLabNeumann(const BlockLabNeumann &) = delete;
  BlockLabNeumann &operator=(const BlockLabNeumann &) = delete;
  void _apply_bc(const Info &info, const Real t = 0,
                 const bool coarse = false) override {
    if (DIM == 2) {
      if (info.index[0] == 0)
        this->template Neumann2D<0, 0>(coarse);
      if (info.index[0] == this->NX - 1)
        this->template Neumann2D<0, 1>(coarse);
      if (info.index[1] == 0)
        this->template Neumann2D<1, 0>(coarse);
      if (info.index[1] == this->NY - 1)
        this->template Neumann2D<1, 1>(coarse);
    } else if (DIM == 3) {
      if (info.index[0] == 0)
        this->template Neumann3D<0, 0>(coarse);
      if (info.index[0] == this->NX - 1)
        this->template Neumann3D<0, 1>(coarse);
      if (info.index[1] == 0)
        this->template Neumann3D<1, 0>(coarse);
      if (info.index[1] == this->NY - 1)
        this->template Neumann3D<1, 1>(coarse);
      if (info.index[2] == 0)
        this->template Neumann3D<2, 0>(coarse);
      if (info.index[2] == this->NZ - 1)
        this->template Neumann3D<2, 1>(coarse);
    }
  }
};
enum BCflag { freespace, periodic, wall };
inline BCflag string2BCflag(const std::string &strFlag) {
  if (strFlag == "periodic")
    return periodic;
  else if (strFlag == "wall")
    return wall;
  else if (strFlag == "freespace")
    return freespace;
  else {
    fprintf(stderr, "BC not recognized %s\n", strFlag.c_str());
    fflush(0);
    abort();
    return periodic;
  }
}
extern BCflag cubismBCX;
extern BCflag cubismBCY;
extern BCflag cubismBCZ;
template <typename TGrid,
          template <typename X> class allocator = std::allocator,
          int direction = 0>
class BlockLabBC : public BlockLab<TGrid, allocator> {
  static constexpr int sizeX = TGrid::BlockType::sizeX;
  static constexpr int sizeY = TGrid::BlockType::sizeY;
  static constexpr int sizeZ = TGrid::BlockType::sizeZ;
  typedef typename TGrid::BlockType::ElementType ElementTypeBlock;
  template <int dir, int side> void applyBCfaceOpen(const bool coarse = false) {
    if (!coarse) {
      auto *const cb = this->m_cacheBlock;
      int s[3] = {0, 0, 0}, e[3] = {0, 0, 0};
      const int *const stenBeg = this->m_stencilStart;
      const int *const stenEnd = this->m_stencilEnd;
      s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : sizeX) : 0;
      s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : sizeY) : 0;
      s[2] = dir == 2 ? (side == 0 ? stenBeg[2] : sizeZ) : 0;
      e[0] = dir == 0 ? (side == 0 ? 0 : sizeX + stenEnd[0] - 1) : sizeX;
      e[1] = dir == 1 ? (side == 0 ? 0 : sizeY + stenEnd[1] - 1) : sizeY;
      e[2] = dir == 2 ? (side == 0 ? 0 : sizeZ + stenEnd[2] - 1) : sizeZ;
      if (ElementTypeBlock::DIM == 1) {
        const int coef = (dir == direction) ? -1 : +1;
        for (int iz = s[2]; iz < e[2]; iz++)
          for (int iy = s[1]; iy < e[1]; iy++)
            for (int ix = s[0]; ix < e[0]; ix++) {
              cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2]) =
                  coef *
                  cb->Access((dir == 0 ? (side == 0 ? 0 : sizeX - 1) : ix) -
                                 stenBeg[0],
                             (dir == 1 ? (side == 0 ? 0 : sizeY - 1) : iy) -
                                 stenBeg[1],
                             (dir == 2 ? (side == 0 ? 0 : sizeZ - 1) : iz) -
                                 stenBeg[2]);
            }
      } else {
        for (int iz = s[2]; iz < e[2]; iz++)
          for (int iy = s[1]; iy < e[1]; iy++)
            for (int ix = s[0]; ix < e[0]; ix++) {
              cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2]) =
                  cb->Access((dir == 0 ? (side == 0 ? 0 : sizeX - 1) : ix) -
                                 stenBeg[0],
                             (dir == 1 ? (side == 0 ? 0 : sizeY - 1) : iy) -
                                 stenBeg[1],
                             (dir == 2 ? (side == 0 ? 0 : sizeZ - 1) : iz) -
                                 stenBeg[2]);
              cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2])
                  .member(dir) =
                  (-1.) *
                  cb->Access((dir == 0 ? (side == 0 ? 0 : sizeX - 1) : ix) -
                                 stenBeg[0],
                             (dir == 1 ? (side == 0 ? 0 : sizeY - 1) : iy) -
                                 stenBeg[1],
                             (dir == 2 ? (side == 0 ? 0 : sizeZ - 1) : iz) -
                                 stenBeg[2])
                      .member(dir);
            }
      }
      const int aux = coarse ? 2 : 1;
      const int bsize[3] = {sizeX / aux, sizeY / aux, sizeZ / aux};
      int s_[3], e_[3];
      s_[dir] = stenBeg[dir] * (1 - side) + bsize[dir] * side;
      e_[dir] = (bsize[dir] - 1 + stenEnd[dir]) * side;
      const int d1 = (dir + 1) % 3;
      const int d2 = (dir + 2) % 3;
      if (ElementTypeBlock::DIM == 1) {
        const int coef = (dir == direction) ? -1 : +1;
        for (int b = 0; b < 2; ++b)
          for (int a = 0; a < 2; ++a) {
            s_[d1] = stenBeg[d1] + a * b * (bsize[d1] - stenBeg[d1]);
            s_[d2] = stenBeg[d2] + (a - a * b) * (bsize[d2] - stenBeg[d2]);
            e_[d1] = (1 - b + a * b) * (bsize[d1] - 1 + stenEnd[d1]);
            e_[d2] = (a + b - a * b) * (bsize[d2] - 1 + stenEnd[d2]);
            for (int iz = s_[2]; iz < e_[2]; iz++)
              for (int iy = s_[1]; iy < e_[1]; iy++)
                for (int ix = s_[0]; ix < e_[0]; ix++) {
                  cb->Access(ix - stenBeg[0], iy - stenBeg[1],
                             iz - stenBeg[2]) =
                      coef * (dir == 0)
                          ? cb->Access(side * (bsize[0] - 1) - stenBeg[0],
                                       iy - stenBeg[1], iz - stenBeg[2])
                          : (dir == 1
                                 ? cb->Access(ix - stenBeg[0],
                                              side * (bsize[1] - 1) -
                                                  stenBeg[1],
                                              iz - stenBeg[2])
                                 : cb->Access(ix - stenBeg[0], iy - stenBeg[1],
                                              side * (bsize[2] - 1) -
                                                  stenBeg[2]));
                }
          }
      } else {
        for (int b = 0; b < 2; ++b)
          for (int a = 0; a < 2; ++a) {
            s_[d1] = stenBeg[d1] + a * b * (bsize[d1] - stenBeg[d1]);
            s_[d2] = stenBeg[d2] + (a - a * b) * (bsize[d2] - stenBeg[d2]);
            e_[d1] = (1 - b + a * b) * (bsize[d1] - 1 + stenEnd[d1]);
            e_[d2] = (a + b - a * b) * (bsize[d2] - 1 + stenEnd[d2]);
            for (int iz = s_[2]; iz < e_[2]; iz++)
              for (int iy = s_[1]; iy < e_[1]; iy++)
                for (int ix = s_[0]; ix < e_[0]; ix++) {
                  cb->Access(ix - stenBeg[0], iy - stenBeg[1],
                             iz - stenBeg[2]) =
                      dir == 0 ? cb->Access(side * (bsize[0] - 1) - stenBeg[0],
                                            iy - stenBeg[1], iz - stenBeg[2])
                               : (dir == 1 ? cb->Access(ix - stenBeg[0],
                                                        side * (bsize[1] - 1) -
                                                            stenBeg[1],
                                                        iz - stenBeg[2])
                                           : cb->Access(ix - stenBeg[0],
                                                        iy - stenBeg[1],
                                                        side * (bsize[2] - 1) -
                                                            stenBeg[2]));
                  cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2])
                      .member(dir) =
                      dir == 0
                          ? (-1.) *
                                cb->Access(side * (bsize[0] - 1) - stenBeg[0],
                                           iy - stenBeg[1], iz - stenBeg[2])
                                    .member(dir)
                          : (dir == 1
                                 ? (-1.) * cb->Access(ix - stenBeg[0],
                                                      side * (bsize[1] - 1) -
                                                          stenBeg[1],
                                                      iz - stenBeg[2])
                                               .member(dir)
                                 : (-1.) * cb->Access(ix - stenBeg[0],
                                                      iy - stenBeg[1],
                                                      side * (bsize[2] - 1) -
                                                          stenBeg[2])
                                               .member(dir));
                }
          }
      }
    } else {
      auto *const cb = this->m_CoarsenedBlock;
      int s[3] = {0, 0, 0}, e[3] = {0, 0, 0};
      const int eI[3] = {
          (this->m_stencilEnd[0]) / 2 + 1 + this->m_InterpStencilEnd[0] - 1,
          (this->m_stencilEnd[1]) / 2 + 1 + this->m_InterpStencilEnd[1] - 1,
          (this->m_stencilEnd[2]) / 2 + 1 + this->m_InterpStencilEnd[2] - 1};
      const int sI[3] = {
          (this->m_stencilStart[0] - 1) / 2 + this->m_InterpStencilStart[0],
          (this->m_stencilStart[1] - 1) / 2 + this->m_InterpStencilStart[1],
          (this->m_stencilStart[2] - 1) / 2 + this->m_InterpStencilStart[2]};
      const int *const stenBeg = sI;
      const int *const stenEnd = eI;
      s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : sizeX / 2) : 0;
      s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : sizeY / 2) : 0;
      s[2] = dir == 2 ? (side == 0 ? stenBeg[2] : sizeZ / 2) : 0;
      e[0] =
          dir == 0 ? (side == 0 ? 0 : sizeX / 2 + stenEnd[0] - 1) : sizeX / 2;
      e[1] =
          dir == 1 ? (side == 0 ? 0 : sizeY / 2 + stenEnd[1] - 1) : sizeY / 2;
      e[2] =
          dir == 2 ? (side == 0 ? 0 : sizeZ / 2 + stenEnd[2] - 1) : sizeZ / 2;
      if (ElementTypeBlock::DIM == 1) {
        const int coef = (dir == direction) ? -1 : +1;
        for (int iz = s[2]; iz < e[2]; iz++)
          for (int iy = s[1]; iy < e[1]; iy++)
            for (int ix = s[0]; ix < e[0]; ix++) {
              cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2]) =
                  coef *
                  cb->Access((dir == 0 ? (side == 0 ? 0 : sizeX / 2 - 1) : ix) -
                                 stenBeg[0],
                             (dir == 1 ? (side == 0 ? 0 : sizeY / 2 - 1) : iy) -
                                 stenBeg[1],
                             (dir == 2 ? (side == 0 ? 0 : sizeZ / 2 - 1) : iz) -
                                 stenBeg[2]);
            }
      } else {
        for (int iz = s[2]; iz < e[2]; iz++)
          for (int iy = s[1]; iy < e[1]; iy++)
            for (int ix = s[0]; ix < e[0]; ix++) {
              cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2]) =
                  cb->Access((dir == 0 ? (side == 0 ? 0 : sizeX / 2 - 1) : ix) -
                                 stenBeg[0],
                             (dir == 1 ? (side == 0 ? 0 : sizeY / 2 - 1) : iy) -
                                 stenBeg[1],
                             (dir == 2 ? (side == 0 ? 0 : sizeZ / 2 - 1) : iz) -
                                 stenBeg[2]);
              cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2])
                  .member(dir) =
                  (-1.) *
                  cb->Access((dir == 0 ? (side == 0 ? 0 : sizeX / 2 - 1) : ix) -
                                 stenBeg[0],
                             (dir == 1 ? (side == 0 ? 0 : sizeY / 2 - 1) : iy) -
                                 stenBeg[1],
                             (dir == 2 ? (side == 0 ? 0 : sizeZ / 2 - 1) : iz) -
                                 stenBeg[2])
                      .member(dir);
            }
      }
      const int aux = coarse ? 2 : 1;
      const int bsize[3] = {sizeX / aux, sizeY / aux, sizeZ / aux};
      int s_[3], e_[3];
      s_[dir] = stenBeg[dir] * (1 - side) + bsize[dir] * side;
      e_[dir] = (bsize[dir] - 1 + stenEnd[dir]) * side;
      const int d1 = (dir + 1) % 3;
      const int d2 = (dir + 2) % 3;
      if (ElementTypeBlock::DIM == 1) {
        const int coef = (dir == direction) ? -1 : +1;
        for (int b = 0; b < 2; ++b)
          for (int a = 0; a < 2; ++a) {
            s_[d1] = stenBeg[d1] + a * b * (bsize[d1] - stenBeg[d1]);
            s_[d2] = stenBeg[d2] + (a - a * b) * (bsize[d2] - stenBeg[d2]);
            e_[d1] = (1 - b + a * b) * (bsize[d1] - 1 + stenEnd[d1]);
            e_[d2] = (a + b - a * b) * (bsize[d2] - 1 + stenEnd[d2]);
            for (int iz = s_[2]; iz < e_[2]; iz++)
              for (int iy = s_[1]; iy < e_[1]; iy++)
                for (int ix = s_[0]; ix < e_[0]; ix++) {
                  cb->Access(ix - stenBeg[0], iy - stenBeg[1],
                             iz - stenBeg[2]) =
                      coef * (dir == 0)
                          ? cb->Access(side * (bsize[0] - 1) - stenBeg[0],
                                       iy - stenBeg[1], iz - stenBeg[2])
                          : (dir == 1
                                 ? cb->Access(ix - stenBeg[0],
                                              side * (bsize[1] - 1) -
                                                  stenBeg[1],
                                              iz - stenBeg[2])
                                 : cb->Access(ix - stenBeg[0], iy - stenBeg[1],
                                              side * (bsize[2] - 1) -
                                                  stenBeg[2]));
                }
          }
      } else {
        for (int b = 0; b < 2; ++b)
          for (int a = 0; a < 2; ++a) {
            s_[d1] = stenBeg[d1] + a * b * (bsize[d1] - stenBeg[d1]);
            s_[d2] = stenBeg[d2] + (a - a * b) * (bsize[d2] - stenBeg[d2]);
            e_[d1] = (1 - b + a * b) * (bsize[d1] - 1 + stenEnd[d1]);
            e_[d2] = (a + b - a * b) * (bsize[d2] - 1 + stenEnd[d2]);
            for (int iz = s_[2]; iz < e_[2]; iz++)
              for (int iy = s_[1]; iy < e_[1]; iy++)
                for (int ix = s_[0]; ix < e_[0]; ix++) {
                  cb->Access(ix - stenBeg[0], iy - stenBeg[1],
                             iz - stenBeg[2]) =
                      dir == 0 ? cb->Access(side * (bsize[0] - 1) - stenBeg[0],
                                            iy - stenBeg[1], iz - stenBeg[2])
                               : (dir == 1 ? cb->Access(ix - stenBeg[0],
                                                        side * (bsize[1] - 1) -
                                                            stenBeg[1],
                                                        iz - stenBeg[2])
                                           : cb->Access(ix - stenBeg[0],
                                                        iy - stenBeg[1],
                                                        side * (bsize[2] - 1) -
                                                            stenBeg[2]));
                  cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2])
                      .member(dir) =
                      dir == 0
                          ? (-1.) *
                                cb->Access(side * (bsize[0] - 1) - stenBeg[0],
                                           iy - stenBeg[1], iz - stenBeg[2])
                                    .member(dir)
                          : (dir == 1
                                 ? (-1.) * cb->Access(ix - stenBeg[0],
                                                      side * (bsize[1] - 1) -
                                                          stenBeg[1],
                                                      iz - stenBeg[2])
                                               .member(dir)
                                 : (-1.) * cb->Access(ix - stenBeg[0],
                                                      iy - stenBeg[1],
                                                      side * (bsize[2] - 1) -
                                                          stenBeg[2])
                                               .member(dir));
                }
          }
      }
    }
  }
  template <int dir, int side> void applyBCfaceWall(const bool coarse = false) {
    if (!coarse) {
      auto *const cb = this->m_cacheBlock;
      int s[3] = {0, 0, 0}, e[3] = {0, 0, 0};
      const int *const stenBeg = this->m_stencilStart;
      const int *const stenEnd = this->m_stencilEnd;
      s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : sizeX) : 0;
      s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : sizeY) : 0;
      s[2] = dir == 2 ? (side == 0 ? stenBeg[2] : sizeZ) : 0;
      e[0] = dir == 0 ? (side == 0 ? 0 : sizeX + stenEnd[0] - 1) : sizeX;
      e[1] = dir == 1 ? (side == 0 ? 0 : sizeY + stenEnd[1] - 1) : sizeY;
      e[2] = dir == 2 ? (side == 0 ? 0 : sizeZ + stenEnd[2] - 1) : sizeZ;
      for (int iz = s[2]; iz < e[2]; iz++)
        for (int iy = s[1]; iy < e[1]; iy++)
          for (int ix = s[0]; ix < e[0]; ix++)
            for (int k = 0; k < ElementTypeBlock::DIM; k++)
              cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2])
                  .member(k) =
                  (-1.0) *
                  cb->Access((dir == 0 ? (side == 0 ? 0 : sizeX - 1) : ix) -
                                 stenBeg[0],
                             (dir == 1 ? (side == 0 ? 0 : sizeY - 1) : iy) -
                                 stenBeg[1],
                             (dir == 2 ? (side == 0 ? 0 : sizeZ - 1) : iz) -
                                 stenBeg[2])
                      .member(k);
      const int aux = coarse ? 2 : 1;
      const int bsize[3] = {sizeX / aux, sizeY / aux, sizeZ / aux};
      int s_[3], e_[3];
      s_[dir] = stenBeg[dir] * (1 - side) + bsize[dir] * side;
      e_[dir] = (bsize[dir] - 1 + stenEnd[dir]) * side;
      const int d1 = (dir + 1) % 3;
      const int d2 = (dir + 2) % 3;
      for (int b = 0; b < 2; ++b)
        for (int a = 0; a < 2; ++a) {
          s_[d1] = stenBeg[d1] + a * b * (bsize[d1] - stenBeg[d1]);
          s_[d2] = stenBeg[d2] + (a - a * b) * (bsize[d2] - stenBeg[d2]);
          e_[d1] = (1 - b + a * b) * (bsize[d1] - 1 + stenEnd[d1]);
          e_[d2] = (a + b - a * b) * (bsize[d2] - 1 + stenEnd[d2]);
          for (int iz = s_[2]; iz < e_[2]; iz++)
            for (int iy = s_[1]; iy < e_[1]; iy++)
              for (int ix = s_[0]; ix < e_[0]; ix++)
                for (int k = 0; k < ElementTypeBlock::DIM; k++) {
                  cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2])
                      .member(k) =
                      (-1.0) *
                      (dir == 0 ? cb->Access(side * (bsize[0] - 1) - stenBeg[0],
                                             iy - stenBeg[1], iz - stenBeg[2])
                                      .member(k)
                                : (dir == 1 ? cb->Access(ix - stenBeg[0],
                                                         side * (bsize[1] - 1) -
                                                             stenBeg[1],
                                                         iz - stenBeg[2])
                                                  .member(k)
                                            : cb->Access(ix - stenBeg[0],
                                                         iy - stenBeg[1],
                                                         side * (bsize[2] - 1) -
                                                             stenBeg[2])
                                                  .member(k)));
                }
        }
    } else {
      auto *const cb = this->m_CoarsenedBlock;
      int s[3] = {0, 0, 0}, e[3] = {0, 0, 0};
      const int eI[3] = {
          (this->m_stencilEnd[0]) / 2 + 1 + this->m_InterpStencilEnd[0] - 1,
          (this->m_stencilEnd[1]) / 2 + 1 + this->m_InterpStencilEnd[1] - 1,
          (this->m_stencilEnd[2]) / 2 + 1 + this->m_InterpStencilEnd[2] - 1};
      const int sI[3] = {
          (this->m_stencilStart[0] - 1) / 2 + this->m_InterpStencilStart[0],
          (this->m_stencilStart[1] - 1) / 2 + this->m_InterpStencilStart[1],
          (this->m_stencilStart[2] - 1) / 2 + this->m_InterpStencilStart[2]};
      const int *const stenBeg = sI;
      const int *const stenEnd = eI;
      s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : sizeX / 2) : 0;
      s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : sizeY / 2) : 0;
      s[2] = dir == 2 ? (side == 0 ? stenBeg[2] : sizeZ / 2) : 0;
      e[0] =
          dir == 0 ? (side == 0 ? 0 : sizeX / 2 + stenEnd[0] - 1) : sizeX / 2;
      e[1] =
          dir == 1 ? (side == 0 ? 0 : sizeY / 2 + stenEnd[1] - 1) : sizeY / 2;
      e[2] =
          dir == 2 ? (side == 0 ? 0 : sizeZ / 2 + stenEnd[2] - 1) : sizeZ / 2;
      for (int iz = s[2]; iz < e[2]; iz++)
        for (int iy = s[1]; iy < e[1]; iy++)
          for (int ix = s[0]; ix < e[0]; ix++)
            for (int k = 0; k < ElementTypeBlock::DIM; k++) {
              cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2])
                  .member(k) =
                  (-1.0) *
                  cb->Access((dir == 0 ? (side == 0 ? 0 : sizeX / 2 - 1) : ix) -
                                 stenBeg[0],
                             (dir == 1 ? (side == 0 ? 0 : sizeY / 2 - 1) : iy) -
                                 stenBeg[1],
                             (dir == 2 ? (side == 0 ? 0 : sizeZ / 2 - 1) : iz) -
                                 stenBeg[2])
                      .member(k);
            }
      const int aux = coarse ? 2 : 1;
      const int bsize[3] = {sizeX / aux, sizeY / aux, sizeZ / aux};
      int s_[3], e_[3];
      s_[dir] = stenBeg[dir] * (1 - side) + bsize[dir] * side;
      e_[dir] = (bsize[dir] - 1 + stenEnd[dir]) * side;
      const int d1 = (dir + 1) % 3;
      const int d2 = (dir + 2) % 3;
      for (int b = 0; b < 2; ++b)
        for (int a = 0; a < 2; ++a) {
          s_[d1] = stenBeg[d1] + a * b * (bsize[d1] - stenBeg[d1]);
          s_[d2] = stenBeg[d2] + (a - a * b) * (bsize[d2] - stenBeg[d2]);
          e_[d1] = (1 - b + a * b) * (bsize[d1] - 1 + stenEnd[d1]);
          e_[d2] = (a + b - a * b) * (bsize[d2] - 1 + stenEnd[d2]);
          for (int iz = s_[2]; iz < e_[2]; iz++)
            for (int iy = s_[1]; iy < e_[1]; iy++)
              for (int ix = s_[0]; ix < e_[0]; ix++)
                for (int k = 0; k < ElementTypeBlock::DIM; k++) {
                  cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2])
                      .member(k) =
                      (-1.0) *
                      (dir == 0 ? cb->Access(side * (bsize[0] - 1) - stenBeg[0],
                                             iy - stenBeg[1], iz - stenBeg[2])
                                      .member(k)
                                : (dir == 1 ? cb->Access(ix - stenBeg[0],
                                                         side * (bsize[1] - 1) -
                                                             stenBeg[1],
                                                         iz - stenBeg[2])
                                                  .member(k)
                                            : cb->Access(ix - stenBeg[0],
                                                         iy - stenBeg[1],
                                                         side * (bsize[2] - 1) -
                                                             stenBeg[2])
                                                  .member(k)));
                }
        }
    }
  }

public:
  typedef typename TGrid::BlockType::ElementType ElementType;
  virtual bool is_xperiodic() override { return cubismBCX == periodic; }
  virtual bool is_yperiodic() override { return cubismBCY == periodic; }
  virtual bool is_zperiodic() override { return cubismBCZ == periodic; }
  BlockLabBC() = default;
  BlockLabBC(const BlockLabBC &) = delete;
  BlockLabBC &operator=(const BlockLabBC &) = delete;
  void _apply_bc(const Info &info, const Real t = 0,
                 const bool coarse = false) {
    const BCflag BCX = cubismBCX;
    const BCflag BCY = cubismBCY;
    const BCflag BCZ = cubismBCZ;
    if (BCX == wall) {
      if (info.index[0] == 0)
        this->template applyBCfaceWall<0, 0>(coarse);
      if (info.index[0] == this->NX - 1)
        this->template applyBCfaceWall<0, 1>(coarse);
    } else if (BCX != periodic) {
      if (info.index[0] == 0)
        this->template applyBCfaceOpen<0, 0>(coarse);
      if (info.index[0] == this->NX - 1)
        this->template applyBCfaceOpen<0, 1>(coarse);
    }
    if (BCY == wall) {
      if (info.index[1] == 0)
        this->template applyBCfaceWall<1, 0>(coarse);
      if (info.index[1] == this->NY - 1)
        this->template applyBCfaceWall<1, 1>(coarse);
    } else if (BCY != periodic) {
      if (info.index[1] == 0)
        this->template applyBCfaceOpen<1, 0>(coarse);
      if (info.index[1] == this->NY - 1)
        this->template applyBCfaceOpen<1, 1>(coarse);
    }
    if (BCZ == wall) {
      if (info.index[2] == 0)
        this->template applyBCfaceWall<2, 0>(coarse);
      if (info.index[2] == this->NZ - 1)
        this->template applyBCfaceWall<2, 1>(coarse);
    } else if (BCZ != periodic) {
      if (info.index[2] == 0)
        this->template applyBCfaceOpen<2, 0>(coarse);
      if (info.index[2] == this->NZ - 1)
        this->template applyBCfaceOpen<2, 1>(coarse);
    }
  }
};
template <typename TGrid,
          template <typename X> class allocator = std::allocator>
class BlockLabNeumann3D : public BlockLabNeumann<TGrid, 3, allocator> {
public:
  using cubismLab = BlockLabNeumann<TGrid, 3, allocator>;
  virtual bool is_xperiodic() override { return cubismBCX == periodic; }
  virtual bool is_yperiodic() override { return cubismBCY == periodic; }
  virtual bool is_zperiodic() override { return cubismBCZ == periodic; }
  void _apply_bc(const Info &info, const Real t = 0,
                 const bool coarse = false) override {
    if (is_xperiodic() == false) {
      if (info.index[0] == 0)
        cubismLab::template Neumann3D<0, 0>(coarse);
      if (info.index[0] == this->NX - 1)
        cubismLab::template Neumann3D<0, 1>(coarse);
    }
    if (is_yperiodic() == false) {
      if (info.index[1] == 0)
        cubismLab::template Neumann3D<1, 0>(coarse);
      if (info.index[1] == this->NY - 1)
        cubismLab::template Neumann3D<1, 1>(coarse);
    }
    if (is_zperiodic() == false) {
      if (info.index[2] == 0)
        cubismLab::template Neumann3D<2, 0>(coarse);
      if (info.index[2] == this->NZ - 1)
        cubismLab::template Neumann3D<2, 1>(coarse);
    }
  }
};
static constexpr int kBlockAlignment = 64;
template <typename T>
using aligned_block_allocator = aligned_allocator<T, kBlockAlignment>;
using ScalarBlock = GridBlock<_BS_, 3, ScalarElement>;
using ScalarGrid = GridMPI<Grid<ScalarBlock, aligned_block_allocator>>;
using ScalarLab =
    BlockLabMPI<BlockLabNeumann3D<ScalarGrid, aligned_block_allocator>>;
using VectorBlock = GridBlock<_BS_, 3, VectorElement>;
using VectorGrid = GridMPI<Grid<VectorBlock, aligned_block_allocator>>;
using VectorLab = BlockLabMPI<BlockLabBC<VectorGrid, aligned_block_allocator>>;
using ScalarAMR = MeshAdaptation<ScalarLab>;
using VectorAMR = MeshAdaptation<VectorLab>;
class ArgumentParser;
class Operator;
class Obstacle;
class ObstacleVector;
class PoissonSolverBase;
struct SimulationData {
  MPI_Comm comm;
  int rank;
  ScalarGrid *chi = nullptr;
  ScalarGrid *pres = nullptr;
  VectorGrid *vel = nullptr;
  VectorGrid *tmpV = nullptr;
  ScalarGrid *lhs = nullptr;
  ScalarAMR *chi_amr;
  ScalarAMR *pres_amr;
  VectorAMR *vel_amr;
  VectorAMR *tmpV_amr;
  ScalarAMR *lhs_amr;
  inline std::vector<Info> &chiInfo() const { return chi->getBlocksInfo(); }
  inline std::vector<Info> &presInfo() const { return pres->getBlocksInfo(); }
  inline std::vector<Info> &velInfo() const { return vel->getBlocksInfo(); }
  inline std::vector<Info> &tmpVInfo() const { return tmpV->getBlocksInfo(); }
  inline std::vector<Info> &lhsInfo() const { return lhs->getBlocksInfo(); }
  ObstacleVector *obstacle_vector = nullptr;
  std::vector<std::shared_ptr<Operator>> pipeline;
  std::shared_ptr<PoissonSolverBase> pressureSolver;
  Real dt = 0;
  Real dt_old = 0;
  Real CFL = 0;
  Real time = 0;
  int step = 0;
  Real endTime = 0;
  int nsteps = 0;
  int rampup;
  int step_2nd_start;
  Real coefU[3] = {1.5, -2.0, 0.5};
  int bpdx, bpdy, bpdz;
  int levelStart;
  int levelMax;
  Real Rtol;
  Real Ctol;
  std::array<Real, 3> extents;
  Real maxextent;
  Real hmin, hmax;
  std::array<Real, 3> uinf = {0, 0, 0};
  int levelMaxVorticity;
  Real uMax_measured = 0;
  Real uMax_allowed;
  Real nu;
  Real lambda;
  bool bImplicitPenalization = true;
  Real DLM = 0;
  Real PoissonErrorTol;
  Real PoissonErrorTolRel;
  std::string poissonSolver;
  bool bCollision = false;
  std::vector<int> bCollisionID;
  BCflag BCx_flag = freespace;
  BCflag BCy_flag = freespace;
  BCflag BCz_flag = freespace;
  int bMeanConstraint = 1;
  bool StaticObstacles = false;
  bool MeshChanged = true;
  std::string initCond = "zero";
  Real uMax_forced = 0;
  bool bFixMassFlux = false;
  int freqDiagnostics = 0;
  int saveFreq = 0;
  Real dumpTime = 0;
  Real nextDumpTime = 0;
  std::string path4serialization = "./";
  bool implicitDiffusion;
  Real DiffusionErrorTol;
  Real DiffusionErrorTolRel;
  void _preprocessArguments();
  ~SimulationData();
  SimulationData() = delete;
  SimulationData(const SimulationData &) = delete;
  SimulationData(SimulationData &&) = delete;
  SimulationData &operator=(const SimulationData &) = delete;
  SimulationData &operator=(SimulationData &&) = delete;
  SimulationData(MPI_Comm mpicomm, ArgumentParser &parser);
};
class Operator {
public:
  SimulationData &sim;
  Operator(SimulationData &s) noexcept : sim(s) {}
  virtual ~Operator() = default;
  virtual void operator()(Real dt) = 0;
};
class AdvectionDiffusion : public Operator {
  std::vector<Real> vOld;

public:
  AdvectionDiffusion(SimulationData &s) : Operator(s) {}
  ~AdvectionDiffusion() {}
  void operator()(const Real dt);
};
namespace diffusion_kernels {
static constexpr int NX = ScalarBlock::sizeX;
static constexpr int NY = ScalarBlock::sizeY;
static constexpr int NZ = ScalarBlock::sizeZ;
static constexpr int N = NX * NY * NZ;
using Block = Real[NZ][NY][NX];
static constexpr int xPad = 4;
using PaddedBlock = Real[NZ + 2][NY + 2][NX + 2 * xPad];
template <int N> static inline Real sum(const Real (&a)[N]) {
  Real s = 0;
  for (int ix = 0; ix < N; ++ix)
    s += a[ix];
  return s;
}
Real kernelDiffusionGetZInnerReference(PaddedBlock &__restrict__ p_,
                                       Block &__restrict__ Ax_,
                                       Block &__restrict__ r_,
                                       Block &__restrict__ block_,
                                       const Real sqrNorm0, const Real rr);
Real kernelDiffusionGetZInner(PaddedBlock &p, const Real *pW, const Real *pE,
                              Block &__restrict__ Ax, Block &__restrict__ r,
                              Block &__restrict__ block, Real sqrNorm0, Real rr,
                              const Real coefficient);
void getZImplParallel(const std::vector<Info> &vInfo, const Real nu,
                      const Real dt);
} // namespace diffusion_kernels
class DiffusionSolver {
public:
  int mydirection = 0;
  Real dt;

protected:
  SimulationData &sim;
  template <typename Lab> struct KernelLHSDiffusion {
    const SimulationData &sim;
    KernelLHSDiffusion(const SimulationData &s, const Real _dt)
        : sim(s), dt(_dt) {}
    const std::vector<Info> &lhsInfo = sim.lhsInfo();
    const Real dt;
    const int Nx = ScalarBlock::sizeX;
    const int Ny = ScalarBlock::sizeY;
    const int Nz = ScalarBlock::sizeZ;
    const StencilInfo stencil{-1, -1, -1, 2, 2, 2, false, {0}};
    void operator()(const Lab &lab, const Info &info) const {
      ScalarBlock &__restrict__ o = (*sim.lhs)(info.blockID);
      const Real h = info.h;
      const Real coef = -1.0 / (dt * sim.nu) * h * h * h;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x) {
            o(x, y, z) =
                h * (lab(x - 1, y, z) + lab(x + 1, y, z) + lab(x, y - 1, z) +
                     lab(x, y + 1, z) + lab(x, y, z - 1) + lab(x, y, z + 1) -
                     6.0 * lab(x, y, z)) +
                coef * lab(x, y, z);
          }
      BlockCase<ScalarBlock> *tempCase =
          (BlockCase<ScalarBlock> *)(lhsInfo[info.blockID].auxiliary);
      if (tempCase == nullptr)
        return;
      ScalarElement *const faceXm =
          tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
      ScalarElement *const faceXp =
          tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
      ScalarElement *const faceYm =
          tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
      ScalarElement *const faceYp =
          tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
      ScalarElement *const faceZm =
          tempCase->storedFace[4] ? &tempCase->m_pData[4][0] : nullptr;
      ScalarElement *const faceZp =
          tempCase->storedFace[5] ? &tempCase->m_pData[5][0] : nullptr;
      if (faceXm != nullptr) {
        const int x = 0;
        for (int z = 0; z < Nz; ++z)
          for (int y = 0; y < Ny; ++y)
            faceXm[y + Ny * z] = h * (lab(x, y, z) - lab(x - 1, y, z));
      }
      if (faceXp != nullptr) {
        const int x = Nx - 1;
        for (int z = 0; z < Nz; ++z)
          for (int y = 0; y < Ny; ++y)
            faceXp[y + Ny * z] = h * (lab(x, y, z) - lab(x + 1, y, z));
      }
      if (faceYm != nullptr) {
        const int y = 0;
        for (int z = 0; z < Nz; ++z)
          for (int x = 0; x < Nx; ++x)
            faceYm[x + Nx * z] = h * (lab(x, y, z) - lab(x, y - 1, z));
      }
      if (faceYp != nullptr) {
        const int y = Ny - 1;
        for (int z = 0; z < Nz; ++z)
          for (int x = 0; x < Nx; ++x)
            faceYp[x + Nx * z] = h * (lab(x, y, z) - lab(x, y + 1, z));
      }
      if (faceZm != nullptr) {
        const int z = 0;
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x)
            faceZm[x + Nx * y] = h * (lab(x, y, z) - lab(x, y, z - 1));
      }
      if (faceZp != nullptr) {
        const int z = Nz - 1;
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x)
            faceZp[x + Nx * y] = h * (lab(x, y, z) - lab(x, y, z + 1));
      }
    }
  };
  void _preconditioner(const std::vector<Real> &input,
                       std::vector<Real> &output) {
    auto &zInfo = sim.pres->getBlocksInfo();
    const size_t Nblocks = zInfo.size();
    const int BSX = VectorBlock::sizeX;
    const int BSY = VectorBlock::sizeY;
    const int BSZ = VectorBlock::sizeZ;
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      ScalarBlock &__restrict__ bb = (*sim.pres)(i);
      for (int iz = 0; iz < BSZ; iz++)
        for (int iy = 0; iy < BSY; iy++)
          for (int ix = 0; ix < BSX; ix++) {
            const int j = i * BSX * BSY * BSZ + iz * BSX * BSY + iy * BSX + ix;
            bb(ix, iy, iz).s = input[j];
          }
    }
#pragma omp parallel
    { diffusion_kernels::getZImplParallel(sim.presInfo(), sim.nu, dt); }
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      const ScalarBlock &__restrict__ bb = (*sim.pres)(i);
      for (int iz = 0; iz < BSZ; iz++)
        for (int iy = 0; iy < BSY; iy++)
          for (int ix = 0; ix < BSX; ix++) {
            const int j = i * BSX * BSY * BSZ + iz * BSX * BSY + iy * BSX + ix;
            output[j] = bb(ix, iy, iz).s;
            ;
          }
    }
  }
  void _lhs(std::vector<Real> &input, std::vector<Real> &output) {
    auto &zInfo = sim.pres->getBlocksInfo();
    auto &AxInfo = sim.lhs->getBlocksInfo();
    const size_t Nblocks = zInfo.size();
    const int BSX = VectorBlock::sizeX;
    const int BSY = VectorBlock::sizeY;
    const int BSZ = VectorBlock::sizeZ;
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      ScalarBlock &__restrict__ zz = *(ScalarBlock *)zInfo[i].block;
      for (int iz = 0; iz < BSZ; iz++)
        for (int iy = 0; iy < BSY; iy++)
          for (int ix = 0; ix < BSX; ix++) {
            const int j = i * BSX * BSY * BSZ + iz * BSX * BSY + iy * BSX + ix;
            zz(ix, iy, iz).s = input[j];
          }
    }
    using Lab0 =
        BlockLabMPI<BlockLabBC<ScalarGrid, aligned_block_allocator, 0>>;
    using Lab1 =
        BlockLabMPI<BlockLabBC<ScalarGrid, aligned_block_allocator, 1>>;
    using Lab2 =
        BlockLabMPI<BlockLabBC<ScalarGrid, aligned_block_allocator, 2>>;
    if (mydirection == 0)
      compute<Lab0>(KernelLHSDiffusion<Lab0>(sim, dt), sim.pres, sim.lhs);
    if (mydirection == 1)
      compute<Lab1>(KernelLHSDiffusion<Lab1>(sim, dt), sim.pres, sim.lhs);
    if (mydirection == 2)
      compute<Lab2>(KernelLHSDiffusion<Lab2>(sim, dt), sim.pres, sim.lhs);
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      ScalarBlock &__restrict__ Ax = *(ScalarBlock *)AxInfo[i].block;
      for (int iz = 0; iz < BSZ; iz++)
        for (int iy = 0; iy < BSY; iy++)
          for (int ix = 0; ix < BSX; ix++) {
            const int j = i * BSX * BSY * BSZ + iz * BSX * BSY + iy * BSX + ix;
            output[j] = Ax(ix, iy, iz).s;
          }
    }
  }
  std::vector<Real> b;
  std::vector<Real> phat;
  std::vector<Real> rhat;
  std::vector<Real> shat;
  std::vector<Real> what;
  std::vector<Real> zhat;
  std::vector<Real> qhat;
  std::vector<Real> s;
  std::vector<Real> w;
  std::vector<Real> z;
  std::vector<Real> t;
  std::vector<Real> v;
  std::vector<Real> q;
  std::vector<Real> r;
  std::vector<Real> y;
  std::vector<Real> x;
  std::vector<Real> r0;
  std::vector<Real> x_opt;

public:
  DiffusionSolver(SimulationData &ss) : sim(ss) {}
  DiffusionSolver(const DiffusionSolver &c) = delete;
  void solve() {
    const auto &AxInfo = sim.lhsInfo();
    const auto &zInfo = sim.presInfo();
    const size_t Nblocks = zInfo.size();
    const int BSX = VectorBlock::sizeX;
    const int BSY = VectorBlock::sizeY;
    const int BSZ = VectorBlock::sizeZ;
    const size_t N = BSX * BSY * BSZ * Nblocks;
    const Real eps = 1e-100;
    const Real max_error = sim.DiffusionErrorTol;
    const Real max_rel_error = sim.DiffusionErrorTolRel;
    bool serious_breakdown = false;
    bool useXopt = false;
    Real min_norm = 1e50;
    Real norm_1 = 0.0;
    Real norm_2 = 0.0;
    const MPI_Comm m_comm = sim.comm;
    const bool verbose = sim.rank == 0;
    phat.resize(N);
    rhat.resize(N);
    shat.resize(N);
    what.resize(N);
    zhat.resize(N);
    qhat.resize(N);
    s.resize(N);
    w.resize(N);
    z.resize(N);
    t.resize(N);
    v.resize(N);
    q.resize(N);
    r.resize(N);
    y.resize(N);
    x.resize(N);
    r0.resize(N);
    b.resize(N);
    x_opt.resize(N);
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      ScalarBlock &__restrict__ rhs = *(ScalarBlock *)AxInfo[i].block;
      const ScalarBlock &__restrict__ zz = *(ScalarBlock *)zInfo[i].block;
      for (int iz = 0; iz < BSZ; iz++)
        for (int iy = 0; iy < BSY; iy++)
          for (int ix = 0; ix < BSX; ix++) {
            const int j = i * BSX * BSY * BSZ + iz * BSX * BSY + iy * BSX + ix;
            b[j] = rhs(ix, iy, iz).s;
            r[j] = rhs(ix, iy, iz).s;
            x[j] = zz(ix, iy, iz).s;
          }
    }
    _lhs(x, r0);
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      r0[i] = r[i] - r0[i];
      r[i] = r0[i];
    }
    _preconditioner(r0, rhat);
    _lhs(rhat, w);
    _preconditioner(w, what);
    _lhs(what, t);
    Real alpha = 0.0;
    Real norm = 0.0;
    Real beta = 0.0;
    Real omega = 0.0;
    Real r0r_prev;
    {
      Real temp0 = 0.0;
      Real temp1 = 0.0;
#pragma omp parallel for reduction(+ : temp0, temp1, norm)
      for (size_t j = 0; j < N; j++) {
        temp0 += r0[j] * r0[j];
        temp1 += r0[j] * w[j];
        norm += r0[j] * r0[j];
      }
      Real temporary[2] = {temp0, temp1};
      MPI_Allreduce(MPI_IN_PLACE, temporary, 2, MPI_Real, MPI_SUM, m_comm);
      MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_Real, MPI_SUM, m_comm);
      alpha = temporary[0] / (temporary[1] + eps);
      r0r_prev = temporary[0];
      norm = std::sqrt(norm);
    }
    const Real init_norm = norm;
    int k;
    for (k = 0; k < 1000; k++) {
      Real qy = 0.0;
      Real yy = 0.0;
      if (k % 50 != 0) {
#pragma omp parallel for reduction(+ : qy, yy)
        for (size_t j = 0; j < N; j++) {
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
        for (size_t j = 0; j < N; j++) {
          phat[j] = rhat[j] + beta * (phat[j] - omega * shat[j]);
        }
        _lhs(phat, s);
        _preconditioner(s, shat);
        _lhs(shat, z);
#pragma omp parallel for
        for (size_t j = 0; j < N; j++) {
          q[j] = r[j] - alpha * s[j];
          qhat[j] = rhat[j] - alpha * shat[j];
          y[j] = w[j] - alpha * z[j];
          qy += q[j] * y[j];
          yy += y[j] * y[j];
        }
      }
      MPI_Request request;
      Real quantities[6];
      quantities[0] = qy;
      quantities[1] = yy;
      MPI_Iallreduce(MPI_IN_PLACE, &quantities, 2, MPI_Real, MPI_SUM, m_comm,
                     &request);
      _preconditioner(z, zhat);
      _lhs(zhat, v);
      MPI_Waitall(1, &request, MPI_STATUSES_IGNORE);
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
        for (size_t j = 0; j < N; j++) {
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
        for (size_t j = 0; j < N; j++) {
          x[j] = x[j] + alpha * phat[j] + omega * qhat[j];
        }
        _lhs(x, r);
#pragma omp parallel for
        for (size_t j = 0; j < N; j++) {
          r[j] = b[j] - r[j];
        }
        _preconditioner(r, rhat);
        _lhs(rhat, w);
#pragma omp parallel for reduction(+ : r0r, r0w, r0s, r0z, norm_1, norm_2, norm)
        for (size_t j = 0; j < N; j++) {
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
      MPI_Request request2;
      MPI_Iallreduce(MPI_IN_PLACE, &quantities, 6, MPI_Real, MPI_SUM, m_comm,
                     &request);
      MPI_Iallreduce(MPI_IN_PLACE, &norm, 1, MPI_Real, MPI_SUM, m_comm,
                     &request2);
      _preconditioner(w, what);
      _lhs(what, t);
      MPI_Waitall(1, &request, MPI_STATUSES_IGNORE);
      r0r = quantities[0];
      r0w = quantities[1];
      r0s = quantities[2];
      r0z = quantities[3];
      norm_1 = quantities[4];
      norm_2 = quantities[5];
      beta = alpha / (omega + eps) * r0r / (r0r_prev + eps);
      alpha = r0r / (r0w + beta * r0s - beta * omega * r0z);
      Real alphat = 1.0 / (omega + eps) + r0w / (r0r + eps) -
                    beta * omega * r0z / (r0r + eps);
      alphat = 1.0 / (alphat + eps);
      if (std::fabs(alphat) < 10 * std::fabs(alpha))
        alpha = alphat;
      r0r_prev = r0r;
      MPI_Waitall(1, &request2, MPI_STATUSES_IGNORE);
      norm = std::sqrt(norm);
      serious_breakdown = r0r * r0r < 1e-16 * norm_1 * norm_2;
      if (serious_breakdown) {
#pragma omp parallel for
        for (size_t i = 0; i < N; i++)
          r0[i] = r[i];
        _preconditioner(r0, rhat);
        _lhs(rhat, w);
        alpha = 0.0;
        Real temp0 = 0.0;
        Real temp1 = 0.0;
#pragma omp parallel for reduction(+ : temp0, temp1)
        for (size_t j = 0; j < N; j++) {
          temp0 += r0[j] * r0[j];
          temp1 += r0[j] * w[j];
        }
        Real temporary[2] = {temp0, temp1};
        MPI_Iallreduce(MPI_IN_PLACE, temporary, 2, MPI_Real, MPI_SUM, m_comm,
                       &request2);
        _preconditioner(w, what);
        _lhs(what, t);
        MPI_Waitall(1, &request2, MPI_STATUSES_IGNORE);
        alpha = temporary[0] / (temporary[1] + eps);
        r0r_prev = temporary[0];
        beta = 0.0;
        omega = 0.0;
      }
      if (norm < min_norm) {
        useXopt = true;
        min_norm = norm;
#pragma omp parallel for
        for (size_t i = 0; i < N; i++)
          x_opt[i] = x[i];
      }
      if (norm < max_error || norm / (init_norm + eps) < max_rel_error) {
        break;
      }
    }
    Real *xsol = useXopt ? x_opt.data() : x.data();
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      ScalarBlock &__restrict__ bb = (*sim.pres)(i);
      for (int iz = 0; iz < BSZ; iz++)
        for (int iy = 0; iy < BSY; iy++)
          for (int ix = 0; ix < BSX; ix++) {
            const int j = i * BSX * BSY * BSZ + iz * BSX * BSY + iy * BSX + ix;
            bb(ix, iy, iz).s = xsol[j];
          }
    }
  }
};
class AdvectionDiffusionImplicit : public Operator {
  std::vector<Real> pressure;
  std::vector<Real> velocity;

public:
  AdvectionDiffusionImplicit(SimulationData &s) : Operator(s) {}
  ~AdvectionDiffusionImplicit() {}
  void operator()(const Real dt);
  void euler(const Real dt);
};
class Value {
private:
  std::string content;

public:
  Value() = default;
  Value(const std::string &content_) : content(content_) {}
  Value(const Value &c) = default;
  Value &operator=(const Value &rhs) {
    if (this != &rhs)
      content = rhs.content;
    return *this;
  }
  Value &operator+=(const Value &rhs) {
    content += " " + rhs.content;
    return *this;
  }
  Value operator+(const Value &rhs) {
    return Value(content + " " + rhs.content);
  }
  double asDouble(double def = 0);
  int asInt(int def = 0);
  bool asBool(bool def = false);
  std::string asString(const std::string &def = std::string());
  friend std::ostream &operator<<(std::ostream &lhs, const Value &rhs);
};
class CommandlineParser {
private:
  const int iArgC;
  char **vArgV;
  bool bStrictMode, bVerbose;
  bool _isnumber(const std::string &s) const;

protected:
  std::map<std::string, Value> mapArguments;

public:
  CommandlineParser(int argc, char **argv);
  Value &operator()(std::string key);
  bool check(std::string key) const;
  int getargc() const { return iArgC; }
  char **getargv() const { return vArgV; }
  void set_strict_mode() { bStrictMode = true; }
  void unset_strict_mode() { bStrictMode = false; }
  void mute() { bVerbose = false; }
  void loud() { bVerbose = true; }
  void save_options(const std::string &path = ".");
};
class ArgumentParser : public CommandlineParser {
  typedef std::map<std::string, Value> ArgMap;
  typedef std::map<std::string, Value *> pArgMap;
  typedef std::map<std::string, ArgMap *> FileMap;
  const char commentStart;
  ArgMap from_commandline;
  FileMap from_files;
  pArgMap from_code;
  ArgMap mapRuntime;
  void _ignoreComments(std::istream &stream, char commentChar);
  void _parseFile(std::ifstream &stream, ArgMap &container);

public:
  ArgumentParser(const int _argc, char **_argv, const char cstart = '#')
      : CommandlineParser(_argc, _argv), commentStart(cstart) {
    from_commandline = mapArguments;
  }
  virtual ~ArgumentParser() {
    for (FileMap::iterator it = from_files.begin(); it != from_files.end();
         it++)
      delete it->second;
  }
  void readFile(const std::string &filepath);
  Value &operator()(std::string key);
  inline bool exist(const std::string &key) const { return check(key); }
};
struct BufferedLoggerImpl;
class BufferedLogger {
  BufferedLoggerImpl *const impl;

public:
  static constexpr int AUTO_FLUSH_COUNT = 100;
  BufferedLogger();
  BufferedLogger(const BufferedLogger &) = delete;
  BufferedLogger(BufferedLogger &&) = delete;
  ~BufferedLogger();
  void flush(void);
  std::stringstream &get_stream(const std::string &filename);
};
extern BufferedLogger logger;
#define SURFDH 1
struct surface_data {
  const int ix, iy, iz;
  const Real dchidx, dchidy, dchidz, delta;
  surface_data(int _ix, int _iy, int _iz, Real _dchidx, Real _dchidy,
               Real _dchidz, Real _delta)
      : ix(_ix), iy(_iy), iz(_iz), dchidx(_dchidx), dchidy(_dchidy),
        dchidz(_dchidz), delta(_delta) {}
  surface_data() = delete;
};
struct ObstacleBlock {
  static constexpr int sizeX = ScalarBlock::sizeX;
  static constexpr int sizeY = ScalarBlock::sizeY;
  static constexpr int sizeZ = ScalarBlock::sizeZ;
  Real chi[sizeZ][sizeY][sizeX];
  Real udef[sizeZ][sizeY][sizeX][3];
  Real sdfLab[sizeZ + 2][sizeY + 2][sizeX + 2];
  int nPoints = 0;
  bool filled = false;
  std::vector<surface_data *> surface;
  Real *pX = nullptr, *pY = nullptr, *pZ = nullptr, *P = nullptr;
  Real *fX = nullptr, *fY = nullptr, *fZ = nullptr;
  Real *fxP = nullptr, *fyP = nullptr, *fzP = nullptr;
  Real *fxV = nullptr, *fyV = nullptr, *fzV = nullptr;
  Real *vX = nullptr, *vY = nullptr, *vZ = nullptr;
  Real *vxDef = nullptr, *vyDef = nullptr, *vzDef = nullptr;
  Real *omegaX = nullptr, *omegaY = nullptr, *omegaZ = nullptr;
  Real CoM_x = 0, CoM_y = 0, CoM_z = 0, mass = 0;
  Real V = 0, FX = 0, FY = 0, FZ = 0, TX = 0, TY = 0, TZ = 0;
  Real J0 = 0, J1 = 0, J2 = 0, J3 = 0, J4 = 0, J5 = 0;
  Real GfX = 0, GpX = 0, GpY = 0, GpZ = 0, Gj0 = 0, Gj1 = 0, Gj2 = 0, Gj3 = 0,
       Gj4 = 0, Gj5 = 0;
  Real GuX = 0, GuY = 0, GuZ = 0, GaX = 0, GaY = 0, GaZ = 0;
  Real forcex = 0, forcey = 0, forcez = 0;
  Real forcex_P = 0, forcey_P = 0, forcez_P = 0;
  Real forcex_V = 0, forcey_V = 0, forcez_V = 0;
  Real torquex = 0, torquey = 0, torquez = 0;
  Real drag = 0, thrust = 0, Pout = 0, PoutBnd = 0, defPower = 0,
       defPowerBnd = 0, pLocom = 0;
  static const int nQoI = 19;
  virtual void sumQoI(std::vector<Real> &sum) {
    assert(sum.size() == nQoI);
    unsigned k = 0;
    sum[k++] += forcex;
    sum[k++] += forcey;
    sum[k++] += forcez;
    sum[k++] += forcex_P;
    sum[k++] += forcey_P;
    sum[k++] += forcez_P;
    sum[k++] += forcex_V;
    sum[k++] += forcey_V;
    sum[k++] += forcez_V;
    sum[k++] += torquex;
    sum[k++] += torquey;
    sum[k++] += torquez;
    sum[k++] += drag;
    sum[k++] += thrust;
    sum[k++] += Pout;
    sum[k++] += PoutBnd;
    sum[k++] += defPower;
    sum[k++] += defPowerBnd;
    sum[k++] += pLocom;
  }
  ObstacleBlock() { surface.reserve(4 * sizeX); }
  virtual ~ObstacleBlock() { clear_surface(); }
  void clear_surface() {
    filled = false;
    nPoints = 0;
    CoM_x = CoM_y = CoM_z = 0;
    forcex = forcey = forcez = 0;
    forcex_P = forcey_P = forcez_P = 0;
    forcex_V = forcey_V = forcez_V = 0;
    torquex = torquey = torquez = 0;
    mass = drag = thrust = Pout = PoutBnd = defPower = defPowerBnd = 0;
    for (auto &trash : surface) {
      if (trash == nullptr)
        continue;
      delete trash;
      trash = nullptr;
    }
    surface.clear();
    if (pX not_eq nullptr) {
      free(pX);
      pX = nullptr;
    }
    if (pY not_eq nullptr) {
      free(pY);
      pY = nullptr;
    }
    if (pZ not_eq nullptr) {
      free(pZ);
      pZ = nullptr;
    }
    if (P not_eq nullptr) {
      free(P);
      P = nullptr;
    }
    if (fX not_eq nullptr) {
      free(fX);
      fX = nullptr;
    }
    if (fY not_eq nullptr) {
      free(fY);
      fY = nullptr;
    }
    if (fZ not_eq nullptr) {
      free(fZ);
      fZ = nullptr;
    }
    if (fxP not_eq nullptr) {
      free(fxP);
      fxP = nullptr;
    }
    if (fyP not_eq nullptr) {
      free(fyP);
      fyP = nullptr;
    }
    if (fzP not_eq nullptr) {
      free(fzP);
      fzP = nullptr;
    }
    if (fxV not_eq nullptr) {
      free(fxV);
      fxV = nullptr;
    }
    if (fyV not_eq nullptr) {
      free(fyV);
      fyV = nullptr;
    }
    if (fzV not_eq nullptr) {
      free(fzV);
      fzV = nullptr;
    }
    if (vX not_eq nullptr) {
      free(vX);
      vX = nullptr;
    }
    if (vY not_eq nullptr) {
      free(vY);
      vY = nullptr;
    }
    if (vZ not_eq nullptr) {
      free(vZ);
      vZ = nullptr;
    }
    if (vxDef not_eq nullptr) {
      free(vxDef);
      vxDef = nullptr;
    }
    if (vyDef not_eq nullptr) {
      free(vyDef);
      vyDef = nullptr;
    }
    if (vzDef not_eq nullptr) {
      free(vzDef);
      vzDef = nullptr;
    }
    if (omegaX not_eq nullptr) {
      free(omegaX);
      omegaX = nullptr;
    }
    if (omegaY not_eq nullptr) {
      free(omegaY);
      omegaY = nullptr;
    }
    if (omegaZ not_eq nullptr) {
      free(omegaZ);
      omegaZ = nullptr;
    }
  }
  virtual void clear() {
    clear_surface();
    memset(chi, 0, sizeof(Real) * sizeX * sizeY * sizeZ);
    memset(udef, 0, sizeof(Real) * sizeX * sizeY * sizeZ * 3);
    memset(sdfLab, 0, sizeof(Real) * (sizeX + 2) * (sizeY + 2) * (sizeZ + 2));
  }
  inline void write(const int ix, const int iy, const int iz, const Real delta,
                    const Real gradUX, const Real gradUY, const Real gradUZ) {
    assert(!filled);
    nPoints++;
    const Real dchidx = -delta * gradUX;
    const Real dchidy = -delta * gradUY;
    const Real dchidz = -delta * gradUZ;
    surface.push_back(
        new surface_data(ix, iy, iz, dchidx, dchidy, dchidz, delta));
  }
  void allocate_surface() {
    filled = true;
    assert((int)surface.size() == nPoints);
    assert(pX == nullptr && pY == nullptr && pZ == nullptr);
    assert(vX == nullptr && vY == nullptr && vZ == nullptr);
    assert(fX == nullptr && fY == nullptr && fZ == nullptr);
    pX = init<Real>(nPoints);
    pY = init<Real>(nPoints);
    pZ = init<Real>(nPoints);
    vX = init<Real>(nPoints);
    vY = init<Real>(nPoints);
    vZ = init<Real>(nPoints);
    fX = init<Real>(nPoints);
    fY = init<Real>(nPoints);
    fZ = init<Real>(nPoints);
    fxP = init<Real>(nPoints);
    fyP = init<Real>(nPoints);
    fzP = init<Real>(nPoints);
    fxV = init<Real>(nPoints);
    fyV = init<Real>(nPoints);
    fzV = init<Real>(nPoints);
    vxDef = init<Real>(nPoints);
    vyDef = init<Real>(nPoints);
    vzDef = init<Real>(nPoints);
    P = init<Real>(nPoints);
    omegaX = init<Real>(nPoints);
    omegaY = init<Real>(nPoints);
    omegaZ = init<Real>(nPoints);
  }
  template <typename T> static inline T *init(const int N) {
    T *ptr;
    const int ret = posix_memalign((void **)&ptr, 32, N * sizeof(T));
    if (ret == EINVAL) {
      fprintf(stderr, "posix_memalign somehow returned EINVAL...\n");
      fflush(0);
      abort();
    } else if (ret == ENOMEM) {
      fprintf(stderr, "Cannot allocate %dx%d bytes with align 32!\n", N,
              (int)sizeof(T));
      fflush(0);
      abort();
    }
    assert(ptr != nullptr);
    memset(ptr, 0, N * sizeof(T));
    return ptr;
  }
};
class ArgumentParser;
class Obstacle;
class ObstacleVector;
class Obstacle {
protected:
  SimulationData &sim;
  bool printedHeaderVels = false;

public:
  std::vector<ObstacleBlock *> obstacleBlocks;
  int obstacleID = 0;
  Real absPos[3] = {0, 0, 0};
  Real position[3] = {0, 0, 0};
  Real quaternion[4] = {1, 0, 0, 0};
  Real transVel[3] = {0, 0, 0};
  Real angVel[3] = {0, 0, 0};
  Real mass;
  Real length;
  Real J[6] = {0, 0, 0, 0, 0, 0};
  std::array<bool, 3> bFixFrameOfRef = {{false, false, false}};
  std::array<bool, 3> bForcedInSimFrame = {{false, false, false}};
  std::array<bool, 3> bBlockRotation = {{false, false, false}};
  std::array<Real, 3> transVel_imposed = {{0, 0, 0}};
  Real old_position[3] = {0, 0, 0};
  Real old_absPos[3] = {0, 0, 0};
  Real old_quaternion[4] = {1, 0, 0, 0};
  Real penalM;
  std::array<Real, 3> penalLmom = {0, 0, 0};
  std::array<Real, 3> penalAmom = {0, 0, 0};
  std::array<Real, 3> penalCM = {0, 0, 0};
  std::array<Real, 6> penalJ = {0, 0, 0, 0, 0, 0};
  std::array<Real, 3> transVel_computed = {0, 0, 0};
  std::array<Real, 3> angVel_computed = {0, 0, 0};
  Real centerOfMass[3] = {0, 0, 0};
  bool bBreakSymmetry = false;
  std::array<Real, 3> force = {0, 0, 0};
  std::array<Real, 3> torque = {0, 0, 0};
  Real surfForce[3] = {0, 0, 0};
  Real presForce[3] = {0, 0, 0};
  Real viscForce[3] = {0, 0, 0};
  Real surfTorque[3] = {0, 0, 0};
  Real drag = 0, thrust = 0, Pout = 0, PoutBnd = 0, pLocom = 0;
  Real defPower = 0, defPowerBnd = 0, Pthrust = 0, Pdrag = 0, EffPDef = 0,
       EffPDefBnd = 0;
  std::array<Real, 3> transVel_correction = {0, 0, 0},
                      angVel_correction = {0, 0, 0};

protected:
public:
  Obstacle(SimulationData &s, ArgumentParser &parser);
  Obstacle(SimulationData &s) : sim(s) {}
  virtual void updateLabVelocity(int nSum[3], Real uSum[3]);
  virtual void computeVelocities();
  virtual void computeForces();
  virtual void update();
  virtual void create();
  virtual void finalize();
  std::array<Real, 3> getTranslationVelocity() const;
  std::array<Real, 3> getAngularVelocity() const;
  std::array<Real, 3> getCenterOfMass() const;
  std::array<Real, 3> getYawPitchRoll() const;
  std::vector<ObstacleBlock *> getObstacleBlocks() const {
    return obstacleBlocks;
  }
  std::vector<ObstacleBlock *> *getObstacleBlocksPtr() {
    return &obstacleBlocks;
  }
  Real collision_counter = 0;
  Real u_collision;
  Real v_collision;
  Real w_collision;
  Real ox_collision;
  Real oy_collision;
  Real oz_collision;
  virtual ~Obstacle() {
    for (auto &entry : obstacleBlocks) {
      if (entry != nullptr) {
        delete entry;
        entry = nullptr;
      }
    }
    obstacleBlocks.clear();
  }
  template <typename T> void create_base(const T &kernel) {
    for (auto &entry : obstacleBlocks) {
      if (entry == nullptr)
        continue;
      delete entry;
      entry = nullptr;
    }
    std::vector<Info> &chiInfo = sim.chiInfo();
    obstacleBlocks.resize(chiInfo.size(), nullptr);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < chiInfo.size(); i++) {
      const Info &info = chiInfo[i];
      const ScalarBlock &b = *(ScalarBlock *)info.block;
      if (kernel.isTouching(info, b)) {
        assert(obstacleBlocks[info.blockID] == nullptr);
        obstacleBlocks[info.blockID] = new ObstacleBlock();
        obstacleBlocks[info.blockID]->clear();
        kernel(info, obstacleBlocks[info.blockID]);
      }
    }
  }
};
class FishMidlineData;
struct VolumeSegment_OBB;
class Fish : public Obstacle {
protected:
  void integrateMidline();
  typedef std::vector<VolumeSegment_OBB> vecsegm_t;
  vecsegm_t prepare_vSegments();
  typedef std::vector<std::vector<VolumeSegment_OBB *>> intersect_t;
  virtual intersect_t prepare_segPerBlock(vecsegm_t &vSeg);
  virtual void writeSDFOnBlocks(std::vector<VolumeSegment_OBB> &vSegments);

public:
  Fish(SimulationData &s, ArgumentParser &p);
  ~Fish() override;
  virtual void create() override;
  FishMidlineData *myFish = nullptr;
  struct BlockID {
    Real h;
    Real origin_x;
    Real origin_y;
    Real origin_z;
    long long blockID;
  };
  std::vector<BlockID> MyBlockIDs;
  std::vector<std::vector<int>> MySegments;
  struct MPI_Obstacle {
    Real d[ScalarBlock::sizeZ * ScalarBlock::sizeY * ScalarBlock::sizeX * 3 +
           (ScalarBlock::sizeZ + 2) * (ScalarBlock::sizeY + 2) *
               (ScalarBlock::sizeX + 2)];
    int i[ScalarBlock::sizeZ * ScalarBlock::sizeY * ScalarBlock::sizeX];
  };
  MPI_Datatype MPI_BLOCKID;
  MPI_Datatype MPI_OBSTACLE;
};
struct Frenet3D {
  static void solve(const int Nm, const Real *const rS, const Real *const curv,
                    const Real *const curv_dt, const Real *const tors,
                    const Real *const tors_dt, Real *const rX, Real *const rY,
                    Real *const rZ, Real *const vX, Real *const vY,
                    Real *const vZ, Real *const norX, Real *const norY,
                    Real *const norZ, Real *const vNorX, Real *const vNorY,
                    Real *const vNorZ, Real *const binX, Real *const binY,
                    Real *const binZ, Real *const vBinX, Real *const vBinY,
                    Real *const vBinZ) {
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
                         tors_dt[i - 1] * binX[i - 1] +
                         tors[i - 1] * vBinX[i - 1];
      const Real dvNuY = -curv_dt[i - 1] * ksiY - curv[i - 1] * vKsiY +
                         tors_dt[i - 1] * binY[i - 1] +
                         tors[i - 1] * vBinY[i - 1];
      const Real dvNuZ = -curv_dt[i - 1] * ksiZ - curv[i - 1] * vKsiZ +
                         tors_dt[i - 1] * binZ[i - 1] +
                         tors[i - 1] * vBinZ[i - 1];
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
      if (d1 > std::numeric_limits<Real>::epsilon()) {
        const Real normfac = 1.0 / std::sqrt(d1);
        ksiX *= normfac;
        ksiY *= normfac;
        ksiZ *= normfac;
      }
      if (d2 > std::numeric_limits<Real>::epsilon()) {
        const Real normfac = 1.0 / std::sqrt(d2);
        norX[i] *= normfac;
        norY[i] *= normfac;
        norZ[i] *= normfac;
      }
      if (d3 > std::numeric_limits<Real>::epsilon()) {
        const Real normfac = 1.0 / std::sqrt(d3);
        binX[i] *= normfac;
        binY[i] *= normfac;
        binZ[i] *= normfac;
      }
    }
  }
};
class Interpolation1D {
public:
  template <typename T>
  static void naturalCubicSpline(const Real *x, const Real *y,
                                 const unsigned int n, const T *xx, T *yy,
                                 const unsigned int nn) {
    return naturalCubicSpline(x, y, n, xx, yy, nn, 0);
  }
  template <typename T>
  static void naturalCubicSpline(const Real *x, const Real *y,
                                 const unsigned int n, const T *xx, T *yy,
                                 const unsigned int nn, const Real offset) {
    Real *y2 = new Real[n];
    Real *u = new Real[n - 1];
    y2[0] = u[0] = 0.0;
    for (unsigned int i = 1; i < n - 1; i++) {
      const Real sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
      const Real p = sig * y2[i - 1] + 2.0;
      y2[i] = (sig - 1.0) / p;
      u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
             (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
      u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
    }
    y2[n - 1] = 0;
    for (unsigned int k = n - 2; k > 0; k--)
      y2[k] = y2[k] * y2[k + 1] + u[k];
    for (unsigned int j = 0; j < nn; j++) {
      unsigned int klo = 0;
      unsigned int khi = n - 1;
      unsigned int k = 0;
      while (khi - klo > 1) {
        k = (khi + klo) >> 1;
        if (x[k] > (xx[j] + offset))
          khi = k;
        else
          klo = k;
      }
      const Real h = x[khi] - x[klo];
      if (abs(h) < 2.2e-16) {
        printf("Interpolation points must be distinct!");
        fflush(0);
        abort();
      }
      const Real a = (x[khi] - (xx[j] + offset)) / h;
      const Real b = ((xx[j] + offset) - x[klo]) / h;
      yy[j] =
          a * y[klo] + b * y[khi] +
          ((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * (h * h) / 6;
    }
    delete[] y2;
    delete[] u;
  }
  template <typename T>
  static void cubicInterpolation(const Real x0, const Real x1, const Real x,
                                 const Real y0, const Real y1, const Real dy0,
                                 const Real dy1, T &y, T &dy) {
    const Real xrel = (x - x0);
    const Real deltax = (x1 - x0);
    const Real a = (dy0 + dy1) / (deltax * deltax) -
                   2 * (y1 - y0) / (deltax * deltax * deltax);
    const Real b =
        (-2 * dy0 - dy1) / deltax + 3 * (y1 - y0) / (deltax * deltax);
    const Real c = dy0;
    const Real d = y0;
    y = a * xrel * xrel * xrel + b * xrel * xrel + c * xrel + d;
    dy = 3 * a * xrel * xrel + 2 * b * xrel + c;
  }
  template <typename T>
  static void cubicInterpolation(const Real x0, const Real x1, const Real x,
                                 const Real y0, const Real y1, T &y, T &dy) {
    return cubicInterpolation(x0, x1, x, y0, y1, 0.0, 0.0, y, dy);
  }
};
namespace Schedulers {
template <int Npoints> struct ParameterScheduler {
  static constexpr int npoints = Npoints;
  std::array<Real, Npoints> parameters_t0;
  std::array<Real, Npoints> parameters_t1;
  std::array<Real, Npoints> dparameters_t0;
  Real t0, t1;
  void save(std::string filename) {
    std::ofstream savestream;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<Real>::digits10 + 1);
    savestream.open(filename + ".txt");
    savestream << t0 << "\t" << t1 << std::endl;
    for (int i = 0; i < Npoints; ++i)
      savestream << parameters_t0[i] << "\t" << parameters_t1[i] << "\t"
                 << dparameters_t0[i] << std::endl;
    savestream.close();
  }
  ParameterScheduler() {
    t0 = -1;
    t1 = 0;
    parameters_t0 = std::array<Real, Npoints>();
    parameters_t1 = std::array<Real, Npoints>();
    dparameters_t0 = std::array<Real, Npoints>();
  }
  void transition(const Real t, const Real tstart, const Real tend,
                  const std::array<Real, Npoints> parameters_tend,
                  const bool UseCurrentDerivative = false) {
    if (t < tstart or t > tend)
      return;
    std::array<Real, Npoints> parameters;
    std::array<Real, Npoints> dparameters;
    gimmeValues(tstart, parameters, dparameters);
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters;
    parameters_t1 = parameters_tend;
    dparameters_t0 =
        UseCurrentDerivative ? dparameters : std::array<Real, Npoints>();
  }
  void transition(const Real t, const Real tstart, const Real tend,
                  const std::array<Real, Npoints> parameters_tstart,
                  const std::array<Real, Npoints> parameters_tend) {
    if (t < tstart or t > tend)
      return;
    if (tstart < t0)
      return;
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters_tstart;
    parameters_t1 = parameters_tend;
  }
  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters,
                   std::array<Real, Npoints> &dparameters) {
    if (t < t0 or t0 < 0) {
      parameters = parameters_t0;
      dparameters = std::array<Real, Npoints>();
    } else if (t > t1) {
      parameters = parameters_t1;
      dparameters = std::array<Real, Npoints>();
    } else {
      for (int i = 0; i < Npoints; ++i)
        Interpolation1D::cubicInterpolation(t0, t1, t, parameters_t0[i],
                                            parameters_t1[i], dparameters_t0[i],
                                            0.0, parameters[i], dparameters[i]);
    }
  }
  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters) {
    std::array<Real, Npoints> dparameters_whocares;
    return gimmeValues(t, parameters, dparameters_whocares);
  }
};
struct ParameterSchedulerScalar : ParameterScheduler<1> {
  void transition(const Real t, const Real tstart, const Real tend,
                  const Real parameter_tend,
                  const bool UseCurrentDerivative = false) {
    const std::array<Real, 1> myParameter = {parameter_tend};
    return ParameterScheduler<1>::transition(t, tstart, tend, myParameter,
                                             UseCurrentDerivative);
  }
  void transition(const Real t, const Real tstart, const Real tend,
                  const Real parameter_tstart, const Real parameter_tend) {
    const std::array<Real, 1> myParameterStart = {parameter_tstart};
    const std::array<Real, 1> myParameterEnd = {parameter_tend};
    return ParameterScheduler<1>::transition(t, tstart, tend, myParameterStart,
                                             myParameterEnd);
  }
  void gimmeValues(const Real t, Real &parameter, Real &dparameter) {
    std::array<Real, 1> myParameter, mydParameter;
    ParameterScheduler<1>::gimmeValues(t, myParameter, mydParameter);
    parameter = myParameter[0];
    dparameter = mydParameter[0];
  }
  void gimmeValues(const Real t, Real &parameter) {
    std::array<Real, 1> myParameter;
    ParameterScheduler<1>::gimmeValues(t, myParameter);
    parameter = myParameter[0];
  }
};
template <int Npoints>
struct ParameterSchedulerVector : ParameterScheduler<Npoints> {
  void gimmeValues(const Real t, const std::array<Real, Npoints> &positions,
                   const int Nfine, const Real *const positions_fine,
                   Real *const parameters_fine, Real *const dparameters_fine) {
    Real *parameters_t0_fine = new Real[Nfine];
    Real *parameters_t1_fine = new Real[Nfine];
    Real *dparameters_t0_fine = new Real[Nfine];
    Interpolation1D::naturalCubicSpline(
        positions.data(), this->parameters_t0.data(), Npoints, positions_fine,
        parameters_t0_fine, Nfine);
    Interpolation1D::naturalCubicSpline(
        positions.data(), this->parameters_t1.data(), Npoints, positions_fine,
        parameters_t1_fine, Nfine);
    Interpolation1D::naturalCubicSpline(
        positions.data(), this->dparameters_t0.data(), Npoints, positions_fine,
        dparameters_t0_fine, Nfine);
    if (t < this->t0 or this->t0 < 0) {
      for (int i = 0; i < Nfine; ++i) {
        parameters_fine[i] = parameters_t0_fine[i];
        dparameters_fine[i] = 0.0;
      }
    } else if (t > this->t1) {
      for (int i = 0; i < Nfine; ++i) {
        parameters_fine[i] = parameters_t1_fine[i];
        dparameters_fine[i] = 0.0;
      }
    } else {
      for (int i = 0; i < Nfine; ++i)
        Interpolation1D::cubicInterpolation(
            this->t0, this->t1, t, parameters_t0_fine[i], parameters_t1_fine[i],
            dparameters_t0_fine[i], 0.0, parameters_fine[i],
            dparameters_fine[i]);
    }
    delete[] parameters_t0_fine;
    delete[] parameters_t1_fine;
    delete[] dparameters_t0_fine;
  }
  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters) {
    ParameterScheduler<Npoints>::gimmeValues(t, parameters);
  }
  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters,
                   std::array<Real, Npoints> &dparameters) {
    ParameterScheduler<Npoints>::gimmeValues(t, parameters, dparameters);
  }
};
template <int Npoints>
struct ParameterSchedulerLearnWave : ParameterScheduler<Npoints> {
  template <typename T>
  void gimmeValues(const Real t, const Real Twave, const Real Length,
                   const std::array<Real, Npoints> &positions, const int Nfine,
                   const T *const positions_fine, T *const parameters_fine,
                   Real *const dparameters_fine) {
    const Real _1oL = 1. / Length;
    const Real _1oT = 1. / Twave;
    for (int i = 0; i < Nfine; ++i) {
      const Real c = positions_fine[i] * _1oL - (t - this->t0) * _1oT;
      bool bCheck = true;
      if (c < positions[0]) {
        Interpolation1D::cubicInterpolation(
            c, positions[0], c, this->parameters_t0[0], this->parameters_t0[0],
            parameters_fine[i], dparameters_fine[i]);
        bCheck = false;
      } else if (c > positions[Npoints - 1]) {
        Interpolation1D::cubicInterpolation(
            positions[Npoints - 1], c, c, this->parameters_t0[Npoints - 1],
            this->parameters_t0[Npoints - 1], parameters_fine[i],
            dparameters_fine[i]);
        bCheck = false;
      } else {
        for (int j = 1; j < Npoints; ++j) {
          if ((c >= positions[j - 1]) && (c <= positions[j])) {
            Interpolation1D::cubicInterpolation(
                positions[j - 1], positions[j], c, this->parameters_t0[j - 1],
                this->parameters_t0[j], parameters_fine[i],
                dparameters_fine[i]);
            dparameters_fine[i] = -dparameters_fine[i] * _1oT;
            bCheck = false;
          }
        }
      }
      if (bCheck) {
        std::cout << "[CUP3D] Argument c=positions_fine[i]*_1oL - (t - "
                     "this->t0)*_1oT="
                  << positions_fine[i] << "*" << _1oL << "-(" << t << "-"
                  << this->t0 << ")*" << _1oT << "=" << c
                  << " could not be associated to wave nodes [Length=" << Length
                  << ", Twave=" << Twave << "]. Aborting..." << std::endl;
        abort();
      }
    }
  }
  void Turn(const Real b, const Real t_turn) {
    this->t0 = t_turn;
    for (int i = Npoints - 1; i > 1; --i)
      this->parameters_t0[i] = this->parameters_t0[i - 2];
    this->parameters_t0[1] = b;
    this->parameters_t0[0] = 0;
  }
};
} // namespace Schedulers
class FishMidlineData {
public:
  const Real length;
  const Real Tperiod;
  const Real phaseShift;
  const Real h;
  const Real waveLength = 1;
  const Real amplitudeFactor;
  const Real fracRefined = 0.1;
  const Real fracMid = 1 - 2 * fracRefined;
  const Real dSmid_tgt = h / std::sqrt(3);
  const Real dSrefine_tgt = 0.125 * h;
  const int Nmid = (int)std::ceil(length * fracMid / dSmid_tgt / 8) * 8;
  const Real dSmid = length * fracMid / Nmid;
  const int Nend =
      (int)std::ceil(fracRefined * length * 2 / (dSmid + dSrefine_tgt) / 4) * 4;
  const Real dSref = fracRefined * length * 2 / Nend - dSmid;
  const int Nm = Nmid + 2 * Nend + 1;
  Real *const rS;
  Real *const rX;
  Real *const rY;
  Real *const rZ;
  Real *const vX;
  Real *const vY;
  Real *const vZ;
  Real *const norX;
  Real *const norY;
  Real *const norZ;
  Real *const vNorX;
  Real *const vNorY;
  Real *const vNorZ;
  Real *const binX;
  Real *const binY;
  Real *const binZ;
  Real *const vBinX;
  Real *const vBinY;
  Real *const vBinZ;
  Real *const width;
  Real *const height;
  std::array<Real, 9> sensorLocation;
  Real quaternion_internal[4] = {1, 0, 0, 0};
  Real angvel_internal[3] = {0, 0, 0};

protected:
  inline Real _d_ds(const int idx, const Real *const vals,
                    const int maxidx) const {
    if (idx == 0)
      return (vals[idx + 1] - vals[idx]) / (rS[idx + 1] - rS[idx]);
    else if (idx == maxidx - 1)
      return (vals[idx] - vals[idx - 1]) / (rS[idx] - rS[idx - 1]);
    else
      return 0.5 * ((vals[idx + 1] - vals[idx]) / (rS[idx + 1] - rS[idx]) +
                    (vals[idx] - vals[idx - 1]) / (rS[idx] - rS[idx - 1]));
  }
  Real *_alloc(const int N) { return new Real[N]; }
  template <typename T> void _dealloc(T *ptr) {
    if (ptr not_eq nullptr) {
      delete[] ptr;
      ptr = nullptr;
    }
  }

public:
  FishMidlineData(Real L, Real Tp, Real phi, Real _h, Real _ampFac = 1)
      : length(L), Tperiod(Tp), phaseShift(phi), h(_h),
        amplitudeFactor(_ampFac), rS(_alloc(Nm)), rX(_alloc(Nm)),
        rY(_alloc(Nm)), rZ(_alloc(Nm)), vX(_alloc(Nm)), vY(_alloc(Nm)),
        vZ(_alloc(Nm)), norX(_alloc(Nm)), norY(_alloc(Nm)), norZ(_alloc(Nm)),
        vNorX(_alloc(Nm)), vNorY(_alloc(Nm)), vNorZ(_alloc(Nm)),
        binX(_alloc(Nm)), binY(_alloc(Nm)), binZ(_alloc(Nm)), vBinX(_alloc(Nm)),
        vBinY(_alloc(Nm)), vBinZ(_alloc(Nm)), width(_alloc(Nm)),
        height(_alloc(Nm)) {
    rS[0] = 0;
    int k = 0;
    for (int i = 0; i < Nend; ++i, k++)
      rS[k + 1] = rS[k] + dSref + (dSmid - dSref) * i / ((Real)Nend - 1.);
    for (int i = 0; i < Nmid; ++i, k++)
      rS[k + 1] = rS[k] + dSmid;
    for (int i = 0; i < Nend; ++i, k++)
      rS[k + 1] =
          rS[k] + dSref + (dSmid - dSref) * (Nend - i - 1) / ((Real)Nend - 1.);
    rS[k] = std::min(rS[k], (Real)L);
    assert(k + 1 == Nm);
  }
  virtual ~FishMidlineData() {
    _dealloc(rS);
    _dealloc(rX);
    _dealloc(rY);
    _dealloc(rZ);
    _dealloc(vX);
    _dealloc(vY);
    _dealloc(vZ);
    _dealloc(norX);
    _dealloc(norY);
    _dealloc(norZ);
    _dealloc(vNorX);
    _dealloc(vNorY);
    _dealloc(vNorZ);
    _dealloc(binX);
    _dealloc(binY);
    _dealloc(binZ);
    _dealloc(vBinX);
    _dealloc(vBinY);
    _dealloc(vBinZ);
    _dealloc(width);
    _dealloc(height);
  }
  void writeMidline2File(const int step_id, std::string filename) {
    char buf[500];
    sprintf(buf, "%s_midline_%07d.txt", filename.c_str(), step_id);
    FILE *f = fopen(buf, "a");
    fprintf(f, "s x y z vX vY vZ\n");
    for (int i = 0; i < Nm; i++)
      fprintf(f, "%g %g %g %g %g %g %g\n", rS[i], rX[i], rY[i], rZ[i], vX[i],
              vY[i], vZ[i]);
    fclose(f);
  }
  void writeMidline2File(const int step_id, std::string filename,
                         const double q[4], const double position[3]) {
    const double Rmatrix3D[3][3] = {
        {1 - 2 * (q[2] * q[2] + q[3] * q[3]), 2 * (q[1] * q[2] - q[3] * q[0]),
         2 * (q[1] * q[3] + q[2] * q[0])},
        {2 * (q[1] * q[2] + q[3] * q[0]), 1 - 2 * (q[1] * q[1] + q[3] * q[3]),
         2 * (q[2] * q[3] - q[1] * q[0])},
        {2 * (q[1] * q[3] - q[2] * q[0]), 2 * (q[2] * q[3] + q[1] * q[0]),
         1 - 2 * (q[1] * q[1] + q[2] * q[2])}};
    char buf[500];
    sprintf(buf, "%s_midline_%07d.txt", filename.c_str(), step_id);
    FILE *f = fopen(buf, "a");
    fprintf(f, "x y z s\n");
    for (int i = 0; i < Nm; i++) {
      double x[3];
      x[0] = position[0] + Rmatrix3D[0][0] * rX[i] + Rmatrix3D[0][1] * rY[i] +
             Rmatrix3D[0][2] * rZ[i];
      x[1] = position[1] + Rmatrix3D[1][0] * rX[i] + Rmatrix3D[1][1] * rY[i] +
             Rmatrix3D[1][2] * rZ[i];
      x[2] = position[2] + Rmatrix3D[2][0] * rX[i] + Rmatrix3D[2][1] * rY[i] +
             Rmatrix3D[2][2] * rZ[i];
      fprintf(f, "%g %g %g %g \n", x[0], x[1], x[2], rS[i]);
    }
    fclose(f);
  }
  void SurfaceNormal(const int idx, const Real theta, Real &nx, Real &ny,
                     Real &nz) {
    if (idx == 0) {
      nx = (rX[idx + 1] - rX[idx]) / (rS[idx + 1] - rS[idx]);
      ny = (rY[idx + 1] - rY[idx]) / (rS[idx + 1] - rS[idx]);
      nz = (rZ[idx + 1] - rZ[idx]) / (rS[idx + 1] - rS[idx]);
    } else {
      const Real costheta = cos(theta);
      const Real sintheta = sin(theta);
      Real drXds = _d_ds(idx, rX, Nm);
      Real drYds = _d_ds(idx, rY, Nm);
      Real drZds = _d_ds(idx, rZ, Nm);
      Real dnorXds = _d_ds(idx, norX, Nm);
      Real dnorYds = _d_ds(idx, norY, Nm);
      Real dnorZds = _d_ds(idx, norZ, Nm);
      Real dbinXds = _d_ds(idx, binX, Nm);
      Real dbinYds = _d_ds(idx, binY, Nm);
      Real dbinZds = _d_ds(idx, binZ, Nm);
      Real dwds = _d_ds(idx, width, Nm);
      Real dhds = _d_ds(idx, height, Nm);
      Real dxds = drXds + costheta * (dwds * norX[idx] + dnorXds * width[idx]) +
                  sintheta * (dhds * binX[idx] + dbinXds * height[idx]);
      Real dyds = drYds + costheta * (dwds * norY[idx] + dnorYds * width[idx]) +
                  sintheta * (dhds * binY[idx] + dbinYds * height[idx]);
      Real dzds = drZds + costheta * (dwds * norZ[idx] + dnorZds * width[idx]) +
                  sintheta * (dhds * binZ[idx] + dbinZds * height[idx]);
      Real dxdtheta = -sintheta * norX[idx] * width[idx] +
                      costheta * binX[idx] * height[idx];
      Real dydtheta = -sintheta * norY[idx] * width[idx] +
                      costheta * binY[idx] * height[idx];
      Real dzdtheta = -sintheta * norZ[idx] * width[idx] +
                      costheta * binZ[idx] * height[idx];
      nx = dydtheta * dzds - dzdtheta * dyds;
      ny = dzdtheta * dxds - dxdtheta * dzds;
      nz = dxdtheta * dyds - dydtheta * dxds;
    }
    const Real norm = 1.0 / (sqrt(nx * nx + ny * ny + nz * nz) +
                             std::numeric_limits<Real>::epsilon());
    nx *= norm;
    ny *= norm;
    nz *= norm;
  }
  void integrateLinearMomentum();
  void integrateAngularMomentum(const Real dt);
  virtual void computeMidline(const Real time, const Real dt) = 0;
  virtual void execute(const Real time, const Real l_tnext,
                       const std::vector<Real> &input) {}
};
struct VolumeSegment_OBB {
  Real safe_distance = 0;
  std::pair<int, int> s_range;
  Real normalI[3] = {1, 0, 0};
  Real normalJ[3] = {0, 1, 0};
  Real normalK[3] = {0, 0, 1};
  Real w[3] = {0, 0, 0}, c[3] = {0, 0, 0};
  Real objBoxLabFr[3][2] = {{0, 0}, {0, 0}, {0, 0}};
  Real objBoxObjFr[3][2] = {{0, 0}, {0, 0}, {0, 0}};
  VolumeSegment_OBB() {}
  void prepare(std::pair<int, int> _s_range, const Real bbox[3][2],
               const Real safe_dist);
  void normalizeNormals();
  void changeToComputationalFrame(const Real position[3],
                                  const Real quaternion[4]);
  bool isIntersectingWithAABB(const Real start[3], const Real end[3]) const;
};
struct PutFishOnBlocks {
  FishMidlineData *cfish;
  const Real position[3];
  const Real quaternion[4];
  const Real Rmatrix3D[3][3];
  PutFishOnBlocks(FishMidlineData *_cfish, const Real p[3], const Real q[4])
      : cfish(_cfish), position{p[0], p[1], p[2]}, quaternion{q[0], q[1], q[2],
                                                              q[3]},
        Rmatrix3D{
            {1 - 2 * (q[2] * q[2] + q[3] * q[3]),
             2 * (q[1] * q[2] - q[3] * q[0]), 2 * (q[1] * q[3] + q[2] * q[0])},
            {2 * (q[1] * q[2] + q[3] * q[0]),
             1 - 2 * (q[1] * q[1] + q[3] * q[3]),
             2 * (q[2] * q[3] - q[1] * q[0])},
            {2 * (q[1] * q[3] - q[2] * q[0]), 2 * (q[2] * q[3] + q[1] * q[0]),
             1 - 2 * (q[1] * q[1] + q[2] * q[2])}} {}
  virtual ~PutFishOnBlocks() {}
  static inline Real eulerDistSq3D(const Real a[3], const Real b[3]) {
    return std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2) +
           std::pow(a[2] - b[2], 2);
  }
  static inline Real eulerDistSq2D(const Real a[3], const Real b[3]) {
    return std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2);
  }
  void changeVelocityToComputationalFrame(Real x[3]) const {
    const Real p[3] = {x[0], x[1], x[2]};
    x[0] = Rmatrix3D[0][0] * p[0] + Rmatrix3D[0][1] * p[1] +
           Rmatrix3D[0][2] * p[2];
    x[1] = Rmatrix3D[1][0] * p[0] + Rmatrix3D[1][1] * p[1] +
           Rmatrix3D[1][2] * p[2];
    x[2] = Rmatrix3D[2][0] * p[0] + Rmatrix3D[2][1] * p[1] +
           Rmatrix3D[2][2] * p[2];
  }
  template <typename T> void changeToComputationalFrame(T x[3]) const {
    const T p[3] = {x[0], x[1], x[2]};
    x[0] = Rmatrix3D[0][0] * p[0] + Rmatrix3D[0][1] * p[1] +
           Rmatrix3D[0][2] * p[2];
    x[1] = Rmatrix3D[1][0] * p[0] + Rmatrix3D[1][1] * p[1] +
           Rmatrix3D[1][2] * p[2];
    x[2] = Rmatrix3D[2][0] * p[0] + Rmatrix3D[2][1] * p[1] +
           Rmatrix3D[2][2] * p[2];
    x[0] += position[0];
    x[1] += position[1];
    x[2] += position[2];
  }
  template <typename T> void changeFromComputationalFrame(T x[3]) const {
    const T p[3] = {x[0] - (T)position[0], x[1] - (T)position[1],
                    x[2] - (T)position[2]};
    x[0] = Rmatrix3D[0][0] * p[0] + Rmatrix3D[1][0] * p[1] +
           Rmatrix3D[2][0] * p[2];
    x[1] = Rmatrix3D[0][1] * p[0] + Rmatrix3D[1][1] * p[1] +
           Rmatrix3D[2][1] * p[2];
    x[2] = Rmatrix3D[0][2] * p[0] + Rmatrix3D[1][2] * p[1] +
           Rmatrix3D[2][2] * p[2];
  }
  void operator()(const Real, const Real, const Real, const Real,
                  ObstacleBlock *const,
                  const std::vector<VolumeSegment_OBB *> &) const;
  virtual void constructSurface(const Real, const Real, const Real, const Real,
                                ObstacleBlock *const,
                                const std::vector<VolumeSegment_OBB *> &) const;
  virtual void constructInternl(const Real, const Real, const Real, const Real,
                                ObstacleBlock *const,
                                const std::vector<VolumeSegment_OBB *> &) const;
  virtual void signedDistanceSqrt(ObstacleBlock *const) const;
};
struct PutNacaOnBlocks : public PutFishOnBlocks {
  PutNacaOnBlocks(FishMidlineData *_cfish, const Real p[3], const Real q[4])
      : PutFishOnBlocks(_cfish, p, q) {}
  Real getSmallerDistToMidLPlanar(const int start_s, const Real x[3],
                                  int &final_s) const;
  void
  constructSurface(const Real, const Real, const Real, const Real,
                   ObstacleBlock *const,
                   const std::vector<VolumeSegment_OBB *> &) const override;
  void
  constructInternl(const Real, const Real, const Real, const Real,
                   ObstacleBlock *const,
                   const std::vector<VolumeSegment_OBB *> &) const override;
};
namespace MidlineShapes {
void integrateBSpline(const Real *const xc, const Real *const yc, const int n,
                      const Real length, Real *const rS, Real *const res,
                      const int Nm);
void naca_width(const Real t_ratio, const Real L, Real *const rS,
                Real *const res, const int Nm);
void stefan_width(const Real L, Real *const rS, Real *const res, const int Nm);
void stefan_height(const Real L, Real *const rS, Real *const res, const int Nm);
void larval_width(const Real L, Real *const rS, Real *const res, const int Nm);
void larval_height(const Real L, Real *const rS, Real *const res, const int Nm);
void danio_width(const Real L, Real *const rS, Real *const res, const int Nm);
void danio_height(const Real L, Real *const rS, Real *const res, const int Nm);
void computeWidthsHeights(const std::string &heightName,
                          const std::string &widthName, Real L, Real *rS,
                          Real *height, Real *width, int nM, int mpirank);
} // namespace MidlineShapes
class ComputeDissipation : public Operator {
public:
  ComputeDissipation(SimulationData &s) : Operator(s) {}
  void operator()(const Real dt);
};
template <typename Derived> struct FillBlocksBase {
  using CHIMAT =
      Real[ScalarBlock::sizeZ][ScalarBlock::sizeY][ScalarBlock::sizeX];
  void operator()(const Info &info, ObstacleBlock *const o) const {
    ScalarBlock &b = *(ScalarBlock *)info.block;
    if (!derived()->isTouching(info, b))
      return;
    auto &SDFLAB = o->sdfLab;
    for (int iz = -1; iz < ScalarBlock::sizeZ + 1; ++iz)
      for (int iy = -1; iy < ScalarBlock::sizeY + 1; ++iy)
        for (int ix = -1; ix < ScalarBlock::sizeX + 1; ++ix) {
          Real p[3];
          info.pos(p, ix, iy, iz);
          const Real dist = derived()->signedDistance(p[0], p[1], p[2]);
          SDFLAB[iz + 1][iy + 1][ix + 1] = dist;
        }
  }

private:
  const Derived *derived() const noexcept {
    return static_cast<const Derived *>(this);
  }
};
class ExternalForcing : public Operator {
public:
  ExternalForcing(SimulationData &s) : Operator(s) {}
  void operator()(const double dt);
};
template <typename T> struct Vector3 {
  T &operator[](int k) { return x_[k]; }
  const T &operator[](int k) const { return x_[k]; }
  friend Vector3 operator+(const Vector3 &a, const Vector3 &b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
  }
  friend Vector3 operator-(const Vector3 &a, const Vector3 &b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
  }
  friend Vector3 operator*(const T &a, const Vector3 &b) {
    return {a * b[0], a * b[1], a * b[2]};
  }
  friend Vector3 cross(const Vector3 &a, const Vector3 &b) {
    return Vector3{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
  }
  friend auto dot(const Vector3 &a, const Vector3 &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }
  T x_[3] = {0};
};
inline int rayIntersectsTriangle(const Vector3<Real> &rayOrigin,
                                 const Vector3<Real> &rayVector,
                                 const Vector3<Vector3<Real>> &triangle,
                                 Vector3<Real> &intersectionPoint) {
  const Real eps = 1e-6;
  const Vector3<Real> edge1 = triangle[1] - triangle[0];
  const Vector3<Real> edge2 = triangle[2] - triangle[0];
  const Vector3<Real> h = cross(rayVector, edge2);
  const Real a = dot(edge1, h);
  if (std::abs(a * a) <= dot(edge1, edge1) * dot(h, h) * eps * eps)
    return -1;
  const Real f = 1.0 / a;
  const Vector3<Real> s = rayOrigin - triangle[0];
  const Real u = f * dot(s, h);
  if (u < 0.0 || u > 1.0)
    return 0;
  const Vector3<Real> q = cross(s, edge1);
  const Real v = f * dot(rayVector, q);
  if (v < 0.0 || u + v > 1.0)
    return 0;
  const Real w = 1.0 - u - v;
  if (u < eps || v < eps || w < eps)
    return -3;
  const Real t = f * dot(edge2, q);
  if (t < 0)
    return -2;
  intersectionPoint = rayOrigin + t * rayVector;
  return 1;
}
inline Vector3<Real> ProjectToLine(const Vector3<Real> &a,
                                   const Vector3<Real> &b,
                                   const Vector3<Real> &p) {
  const Real norm_ab = (b[0] - a[0]) * (b[0] - a[0]) +
                       (b[1] - a[1]) * (b[1] - a[1]) +
                       (b[2] - a[2]) * (b[2] - a[2]);
  const Real proj_a = std::fabs(dot(p - a, b - a));
  const Real proj_b = std::fabs(dot(p - b, b - a));
  if (proj_a <= norm_ab && proj_b <= norm_ab)
    return a + (proj_a / norm_ab) * (b - a);
  else if (proj_a < proj_b)
    return a;
  else
    return b;
}
inline Real pointTriangleSqrDistance(const Vector3<Real> &a,
                                     const Vector3<Real> &b,
                                     const Vector3<Real> &c,
                                     const Vector3<Real> &r,
                                     Vector3<Real> &rpt) {
  const auto ac = c - a;
  const auto ab = b - a;
  const auto ar = r - a;
  const auto n = cross(ab, ac);
  const Real alpha = dot(ar, n) / dot(n, n);
  const auto rp = r - alpha * n;
  Real u = 0;
  Real v = 0;
  Real w = 0;
  const Vector3<Real> temp = rp - a;
  const Real d00 = dot(ab, ab);
  const Real d01 = dot(ab, ac);
  const Real d11 = dot(ac, ac);
  const Real d20 = dot(temp, ab);
  const Real d21 = dot(temp, ac);
  const Real denom = d00 * d11 - d01 * d01;
  v = (d11 * d20 - d01 * d21) / denom;
  w = (d00 * d21 - d01 * d20) / denom;
  u = 1.0 - v - w;
  if (std::fabs(denom) < 1e-23) {
    const Real ab1 = std::fabs(dot(ab, ab));
    const Real ac1 = std::fabs(dot(ac, ac));
    const Real bc1 = std::fabs(dot(b - c, b - c));
    if (ab1 <= ac1 && ab1 <= bc1)
      rpt = ProjectToLine(a, c, rp);
    else if (ac1 <= ab1 && ac1 <= bc1)
      rpt = ProjectToLine(b, c, rp);
    else if (bc1 <= ab1 && bc1 <= ac1)
      rpt = ProjectToLine(a, b, rp);
    else
      MPI_Abort(MPI_COMM_WORLD, 666);
  } else if (u >= 0 && v >= 0 && w >= 0) {
    rpt = u * a + v * b + w * c;
  } else if (u < 0 && v < 0)
    rpt = c;
  else if (u < 0 && w < 0)
    rpt = b;
  else if (v < 0 && w < 0)
    rpt = a;
  else if (u < 0)
    rpt = ProjectToLine(b, c, rp);
  else if (v < 0)
    rpt = ProjectToLine(a, c, rp);
  else if (w < 0)
    rpt = ProjectToLine(a, b, rp);
  Real retval = (r[0] - rpt[0]) * (r[0] - rpt[0]) +
                (r[1] - rpt[1]) * (r[1] - rpt[1]) +
                (r[2] - rpt[2]) * (r[2] - rpt[2]);
  return retval;
}
class FixMassFlux : public Operator {
public:
  FixMassFlux(SimulationData &s);
  void operator()(const double dt);
};
class ObstacleVector : public Obstacle {
public:
  typedef std::vector<std::shared_ptr<Obstacle>> VectorType;
  ObstacleVector(SimulationData &s) : Obstacle(s) {}
  Obstacle *operator()(const size_t ind) const { return obstacles[ind].get(); }
  int nObstacles() const { return obstacles.size(); }
  void addObstacle(std::shared_ptr<Obstacle> obstacle) {
    obstacle->obstacleID = obstacles.size();
    obstacles.emplace_back(std::move(obstacle));
  }
  void update() override {
    for (const auto &obstacle_ptr : obstacles)
      obstacle_ptr->update();
  }
  void create() override {
    for (const auto &obstacle_ptr : obstacles)
      obstacle_ptr->create();
  }
  void finalize() override {
    for (const auto &obstacle_ptr : obstacles)
      obstacle_ptr->finalize();
  }
  void computeVelocities() override {
    for (const auto &obstacle_ptr : obstacles)
      obstacle_ptr->computeVelocities();
  }
  void computeForces() override {
    for (const auto &obstacle_ptr : obstacles)
      obstacle_ptr->computeForces();
  }
  const VectorType &getObstacleVector() const { return obstacles; }
  std::vector<std::vector<ObstacleBlock *> *> getAllObstacleBlocks() const {
    const size_t Nobs = obstacles.size();
    std::vector<std::vector<ObstacleBlock *> *> ret(Nobs, nullptr);
    for (size_t i = 0; i < Nobs; i++)
      ret[i] = obstacles[i]->getObstacleBlocksPtr();
    return ret;
  }
  std::array<Real, 3> updateUinf() const {
    std::array<int, 3> nSum = {0, 0, 0};
    std::array<Real, 3> uSum = {0, 0, 0};
    for (const auto &obstacle_ptr : obstacles)
      obstacle_ptr->updateLabVelocity(&nSum[0], &uSum[0]);
    if (nSum[0] > 0)
      uSum[0] = uSum[0] / nSum[0];
    if (nSum[1] > 0)
      uSum[1] = uSum[1] / nSum[1];
    if (nSum[2] > 0)
      uSum[2] = uSum[2] / nSum[2];
    return uSum;
  }

protected:
  VectorType obstacles;
};
class ComputeForces : public Operator {
public:
  ComputeForces(SimulationData &s) : Operator(s) {}
  void operator()(const Real dt);
};
class InitialConditions : public Operator {
public:
  InitialConditions(SimulationData &s) : Operator(s) {}
  template <typename K> inline void run(const K kernel) {
    std::vector<Info> &vInfo = sim.velInfo();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < vInfo.size(); i++)
      kernel(vInfo[i], *(VectorBlock *)vInfo[i].block);
  }
  void operator()(const Real dt);
};
struct GradChiOnTmp {
  GradChiOnTmp(const SimulationData &s) : sim(s) {}
  const SimulationData &sim;
  const StencilInfo stencil{-2, -2, -2, 3, 3, 3, true, {0}};
  void operator()(ScalarLab &lab, const Info &info) const {
    auto &__restrict__ TMP = (*sim.tmpV)(info.blockID);
    if (info.level == sim.levelMaxVorticity - 1 &&
        sim.levelMaxVorticity < sim.levelMax)
      for (int z = 0; z < VectorBlock::sizeZ; ++z)
        for (int y = 0; y < VectorBlock::sizeY; ++y)
          for (int x = 0; x < VectorBlock::sizeX; ++x) {
            if (TMP(x, y, z).magnitude() >= sim.Rtol) {
              TMP(x, y, z).u[0] = 0.5 * (sim.Rtol + sim.Ctol);
              TMP(x, y, z).u[1] = 0.0;
              TMP(x, y, z).u[2] = 0.0;
            }
          }
    bool done = false;
    const int offset = (info.level == sim.chi->getlevelMax() - 1) ? 2 : 1;
    for (int z = -offset; z < VectorBlock::sizeZ + offset; ++z)
      for (int y = -offset; y < VectorBlock::sizeY + offset; ++y)
        for (int x = -offset; x < VectorBlock::sizeX + offset; ++x) {
          if (done)
            break;
          lab(x, y, z).s = std::min(lab(x, y, z).s, (Real)1.0);
          lab(x, y, z).s = std::max(lab(x, y, z).s, (Real)0.0);
          if (lab(x, y, z).s > 0.00001 && lab(x, y, z).s < 0.9) {
            TMP(VectorBlock::sizeX / 2 - 1, VectorBlock::sizeY / 2 - 1,
                VectorBlock::sizeZ / 2 - 1)
                .u[0] = 1e10;
            TMP(VectorBlock::sizeX / 2, VectorBlock::sizeY / 2 - 1,
                VectorBlock::sizeZ / 2 - 1)
                .u[0] = 1e10;
            TMP(VectorBlock::sizeX / 2 - 1, VectorBlock::sizeY / 2,
                VectorBlock::sizeZ / 2 - 1)
                .u[0] = 1e10;
            TMP(VectorBlock::sizeX / 2, VectorBlock::sizeY / 2 - 1,
                VectorBlock::sizeZ / 2 - 1)
                .u[0] = 1e10;
            TMP(VectorBlock::sizeX / 2 - 1, VectorBlock::sizeY / 2 - 1,
                VectorBlock::sizeZ / 2)
                .u[0] = 1e10;
            TMP(VectorBlock::sizeX / 2, VectorBlock::sizeY / 2,
                VectorBlock::sizeZ / 2)
                .u[0] = 1e10;
            TMP(VectorBlock::sizeX / 2 - 1, VectorBlock::sizeY / 2 - 1,
                VectorBlock::sizeZ / 2)
                .u[0] = 1e10;
            TMP(VectorBlock::sizeX / 2, VectorBlock::sizeY / 2 - 1,
                VectorBlock::sizeZ / 2)
                .u[0] = 1e10;
            done = true;
            break;
          } else if (lab(x, y, z).s > 0.9 && z >= 0 && z < VectorBlock::sizeZ &&
                     y >= 0 && y < VectorBlock::sizeY && x >= 0 &&
                     x < VectorBlock::sizeX) {
            TMP(x, y, z).u[0] = 0.0;
            TMP(x, y, z).u[1] = 0.0;
            TMP(x, y, z).u[2] = 0.0;
          }
        }
  }
};
inline Real findMaxU(SimulationData &sim) {
  const std::vector<Info> &myInfo = sim.velInfo();
  const Real uinf[3] = {sim.uinf[0], sim.uinf[1], sim.uinf[2]};
  Real maxU = 0;
#pragma omp parallel for schedule(static) reduction(max : maxU)
  for (size_t i = 0; i < myInfo.size(); i++) {
    const VectorBlock &b = *(const VectorBlock *)myInfo[i].block;
    for (int z = 0; z < VectorBlock::sizeZ; ++z)
      for (int y = 0; y < VectorBlock::sizeY; ++y)
        for (int x = 0; x < VectorBlock::sizeX; ++x) {
          const Real advu = std::fabs(b(x, y, z).u[0] + uinf[0]);
          const Real advv = std::fabs(b(x, y, z).u[1] + uinf[1]);
          const Real advw = std::fabs(b(x, y, z).u[2] + uinf[2]);
          const Real maxUl = std::max({advu, advv, advw});
          maxU = std::max(maxU, maxUl);
        }
  }
  MPI_Allreduce(MPI_IN_PLACE, &maxU, 1, MPI_Real, MPI_MAX, sim.comm);
  assert(maxU >= 0);
  return maxU;
}
struct KernelVorticity {
  SimulationData &sim;
  KernelVorticity(SimulationData &s) : sim(s){};
  const StencilInfo stencil{-1, -1, -1, 2, 2, 2, false, {0, 1, 2}};
  const std::vector<Info> &vInfo = sim.tmpVInfo();
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  void operator()(const VectorLab &lab, const Info &info) const {
    const Info &info2 = vInfo[info.blockID];
    VectorBlock &o = *(VectorBlock *)info2.block;
    const Real inv2h = .5 * info.h * info.h;
    for (int z = 0; z < Nz; ++z)
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          const VectorElement &LW = lab(x - 1, y, z), &LE = lab(x + 1, y, z);
          const VectorElement &LS = lab(x, y - 1, z), &LN = lab(x, y + 1, z);
          const VectorElement &LF = lab(x, y, z - 1), &LB = lab(x, y, z + 1);
          o(x, y, z).u[0] = inv2h * ((LN.u[2] - LS.u[2]) - (LB.u[1] - LF.u[1]));
          o(x, y, z).u[1] = inv2h * ((LB.u[0] - LF.u[0]) - (LE.u[2] - LW.u[2]));
          o(x, y, z).u[2] = inv2h * ((LE.u[1] - LW.u[1]) - (LN.u[0] - LS.u[0]));
        }
    BlockCase<VectorBlock> *tempCase =
        (BlockCase<VectorBlock> *)(info.auxiliary);
    if (tempCase == nullptr)
      return;
    VectorElement *const faceXm =
        tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
    VectorElement *const faceXp =
        tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
    VectorElement *const faceYm =
        tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
    VectorElement *const faceYp =
        tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    VectorElement *const faceZm =
        tempCase->storedFace[4] ? &tempCase->m_pData[4][0] : nullptr;
    VectorElement *const faceZp =
        tempCase->storedFace[5] ? &tempCase->m_pData[5][0] : nullptr;
    if (faceXm != nullptr) {
      const int x = 0;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y) {
          const VectorElement &LW = lab(x - 1, y, z);
          const VectorElement &LC = lab(x, y, z);
          faceXm[y + Ny * z].u[1] = -inv2h * (LW.u[2] + LC.u[2]);
          faceXm[y + Ny * z].u[2] = +inv2h * (LW.u[1] + LC.u[1]);
        }
    }
    if (faceXp != nullptr) {
      const int x = Nx - 1;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y) {
          const VectorElement &LE = lab(x + 1, y, z);
          const VectorElement &LC = lab(x, y, z);
          faceXp[y + Ny * z].u[1] = +inv2h * (LE.u[2] + LC.u[2]);
          faceXp[y + Ny * z].u[2] = -inv2h * (LE.u[1] + LC.u[1]);
        }
    }
    if (faceYm != nullptr) {
      const int y = 0;
      for (int z = 0; z < Nz; ++z)
        for (int x = 0; x < Nx; ++x) {
          const VectorElement &LS = lab(x, y - 1, z);
          const VectorElement &LC = lab(x, y, z);
          faceYm[x + Nx * z].u[0] = +inv2h * (LS.u[2] + LC.u[2]);
          faceYm[x + Nx * z].u[2] = -inv2h * (LS.u[0] + LC.u[0]);
        }
    }
    if (faceYp != nullptr) {
      const int y = Ny - 1;
      for (int z = 0; z < Nz; ++z)
        for (int x = 0; x < Nx; ++x) {
          const VectorElement &LN = lab(x, y + 1, z);
          const VectorElement &LC = lab(x, y, z);
          faceYp[x + Nx * z].u[0] = -inv2h * (LN.u[2] + LC.u[2]);
          faceYp[x + Nx * z].u[2] = +inv2h * (LN.u[0] + LC.u[0]);
        }
    }
    if (faceZm != nullptr) {
      const int z = 0;
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          const VectorElement &LF = lab(x, y, z - 1);
          const VectorElement &LC = lab(x, y, z);
          faceZm[x + Nx * y].u[0] = -inv2h * (LF.u[1] + LC.u[1]);
          faceZm[x + Nx * y].u[1] = +inv2h * (LF.u[0] + LC.u[0]);
        }
    }
    if (faceZp != nullptr) {
      const int z = Nz - 1;
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          const VectorElement &LB = lab(x, y, z + 1);
          const VectorElement &LC = lab(x, y, z);
          faceZp[x + Nx * y].u[0] = +inv2h * (LB.u[1] + LC.u[1]);
          faceZp[x + Nx * y].u[1] = -inv2h * (LB.u[0] + LC.u[0]);
        }
    }
  }
};
class ComputeVorticity : public Operator {
public:
  ComputeVorticity(SimulationData &s) : Operator(s) {}
  void operator()(const Real dt) {
    const KernelVorticity K(sim);
    compute<VectorLab>(K, sim.vel, sim.tmpV);
    const std::vector<Info> &myInfo = sim.tmpVInfo();
#pragma omp parallel for
    for (size_t i = 0; i < myInfo.size(); i++) {
      const Info &info = myInfo[i];
      VectorBlock &b = *(VectorBlock *)info.block;
      const Real fac = 1.0 / (info.h * info.h * info.h);
      for (int z = 0; z < VectorBlock::sizeZ; ++z)
        for (int y = 0; y < VectorBlock::sizeY; ++y)
          for (int x = 0; x < VectorBlock::sizeX; ++x) {
            b(x, y, z).u[0] *= fac;
            b(x, y, z).u[1] *= fac;
            b(x, y, z).u[2] *= fac;
          }
    }
  }
};
class KernelQcriterion {
public:
  SimulationData &sim;
  KernelQcriterion(SimulationData &s) : sim(s){};
  const std::array<int, 3> stencil_start = {-1, -1, -1},
                           stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1, -1, -1, 2, 2, 2, false, {0, 1, 2}};
  const std::vector<Info> &vInfo = sim.presInfo();
  void operator()(VectorLab &lab, const Info &info) const {
    ScalarBlock &o = *(ScalarBlock *)vInfo[info.blockID].block;
    const Real inv2h = .5 / info.h;
    for (int iz = 0; iz < ScalarBlock::sizeZ; ++iz)
      for (int iy = 0; iy < ScalarBlock::sizeY; ++iy)
        for (int ix = 0; ix < ScalarBlock::sizeX; ++ix) {
          const VectorElement &LW = lab(ix - 1, iy, iz),
                              &LE = lab(ix + 1, iy, iz);
          const VectorElement &LS = lab(ix, iy - 1, iz),
                              &LN = lab(ix, iy + 1, iz);
          const VectorElement &LF = lab(ix, iy, iz - 1),
                              &LB = lab(ix, iy, iz + 1);
          const Real WX = inv2h * ((LN.u[2] - LS.u[2]) - (LB.u[1] - LF.u[1]));
          const Real WY = inv2h * ((LB.u[0] - LF.u[0]) - (LE.u[2] - LW.u[2]));
          const Real WZ = inv2h * ((LE.u[1] - LW.u[1]) - (LN.u[0] - LS.u[0]));
          const Real D11 = inv2h * (LE.u[0] - LW.u[0]);
          const Real D22 = inv2h * (LN.u[1] - LS.u[1]);
          const Real D33 = inv2h * (LB.u[2] - LF.u[2]);
          const Real D12 = inv2h * (LN.u[0] - LS.u[0] + LE.u[1] - LW.u[1]);
          const Real D13 = inv2h * (LE.u[2] - LW.u[2] + LB.u[0] - LF.u[0]);
          const Real D23 = inv2h * (LB.u[1] - LF.u[1] + LN.u[2] - LS.u[2]);
          const Real SS = D11 * D11 + D22 * D22 + D33 * D33 +
                          (D12 * D12 + D13 * D13 + D23 * D23) / 2;
          o(ix, iy, iz).s = ((WX * WX + WY * WY + WZ * WZ) / 2 - SS) / 2;
        }
  }
};
class ComputeQcriterion : public Operator {
public:
  ComputeQcriterion(SimulationData &s) : Operator(s) {}
  void operator()(const Real dt) {
    const KernelQcriterion K(sim);
    compute<VectorLab>(K, sim.vel);
  }
};
class KernelDivergence {
public:
  SimulationData &sim;
  KernelDivergence(SimulationData &s) : sim(s) {}
  const std::array<int, 3> stencil_start = {-1, -1, -1},
                           stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1, -1, -1, 2, 2, 2, false, {0, 1, 2}};
  const std::vector<Info> &vInfo = sim.tmpVInfo();
  const std::vector<Info> &chiInfo = sim.chiInfo();
  void operator()(VectorLab &lab, const Info &info) const {
    VectorBlock &o = *(VectorBlock *)vInfo[info.blockID].block;
    ScalarBlock &c = *(ScalarBlock *)chiInfo[info.blockID].block;
    const Real fac = 0.5 * info.h * info.h;
    for (int iz = 0; iz < VectorBlock::sizeZ; ++iz)
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
          o(ix, iy, iz).u[0] =
              (1.0 - c(ix, iy, iz).s) * fac *
              (lab(ix + 1, iy, iz).u[0] - lab(ix - 1, iy, iz).u[0] +
               lab(ix, iy + 1, iz).u[1] - lab(ix, iy - 1, iz).u[1] +
               lab(ix, iy, iz + 1).u[2] - lab(ix, iy, iz - 1).u[2]);
        }
    BlockCase<VectorBlock> *tempCase =
        (BlockCase<VectorBlock> *)(info.auxiliary);
    typename VectorBlock::ElementType *faceXm = nullptr;
    typename VectorBlock::ElementType *faceXp = nullptr;
    typename VectorBlock::ElementType *faceYm = nullptr;
    typename VectorBlock::ElementType *faceYp = nullptr;
    typename VectorBlock::ElementType *faceZp = nullptr;
    typename VectorBlock::ElementType *faceZm = nullptr;
    if (tempCase != nullptr) {
      faceXm = tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
      faceXp = tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
      faceYm = tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
      faceYp = tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
      faceZm = tempCase->storedFace[4] ? &tempCase->m_pData[4][0] : nullptr;
      faceZp = tempCase->storedFace[5] ? &tempCase->m_pData[5][0] : nullptr;
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iz = 0; iz < ScalarBlock::sizeZ; ++iz)
        for (int iy = 0; iy < ScalarBlock::sizeY; ++iy) {
          faceXm[iy + ScalarBlock::sizeY * iz].clear();
          faceXm[iy + ScalarBlock::sizeY * iz].u[0] =
              (1.0 - c(ix, iy, iz).s) * fac *
              (lab(ix - 1, iy, iz).u[0] + lab(ix, iy, iz).u[0]);
        }
    }
    if (faceXp != nullptr) {
      int ix = ScalarBlock::sizeX - 1;
      for (int iz = 0; iz < ScalarBlock::sizeZ; ++iz)
        for (int iy = 0; iy < ScalarBlock::sizeY; ++iy) {
          faceXp[iy + ScalarBlock::sizeY * iz].clear();
          faceXp[iy + ScalarBlock::sizeY * iz].u[0] =
              -(1.0 - c(ix, iy, iz).s) * fac *
              (lab(ix + 1, iy, iz).u[0] + lab(ix, iy, iz).u[0]);
        }
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int iz = 0; iz < ScalarBlock::sizeZ; ++iz)
        for (int ix = 0; ix < ScalarBlock::sizeX; ++ix) {
          faceYm[ix + ScalarBlock::sizeX * iz].clear();
          faceYm[ix + ScalarBlock::sizeX * iz].u[0] =
              (1.0 - c(ix, iy, iz).s) * fac *
              (lab(ix, iy - 1, iz).u[1] + lab(ix, iy, iz).u[1]);
        }
    }
    if (faceYp != nullptr) {
      int iy = ScalarBlock::sizeY - 1;
      for (int iz = 0; iz < ScalarBlock::sizeZ; ++iz)
        for (int ix = 0; ix < ScalarBlock::sizeX; ++ix) {
          faceYp[ix + ScalarBlock::sizeX * iz].clear();
          faceYp[ix + ScalarBlock::sizeX * iz].u[0] =
              -(1.0 - c(ix, iy, iz).s) * fac *
              (lab(ix, iy + 1, iz).u[1] + lab(ix, iy, iz).u[1]);
        }
    }
    if (faceZm != nullptr) {
      int iz = 0;
      for (int iy = 0; iy < ScalarBlock::sizeY; ++iy)
        for (int ix = 0; ix < ScalarBlock::sizeX; ++ix) {
          faceZm[ix + ScalarBlock::sizeX * iy].clear();
          faceZm[ix + ScalarBlock::sizeX * iy].u[0] =
              (1.0 - c(ix, iy, iz).s) * fac *
              (lab(ix, iy, iz - 1).u[2] + lab(ix, iy, iz).u[2]);
        }
    }
    if (faceZp != nullptr) {
      int iz = ScalarBlock::sizeZ - 1;
      for (int iy = 0; iy < ScalarBlock::sizeY; ++iy)
        for (int ix = 0; ix < ScalarBlock::sizeX; ++ix) {
          faceZp[ix + ScalarBlock::sizeX * iy].clear();
          faceZp[ix + ScalarBlock::sizeX * iy].u[0] =
              -(1.0 - c(ix, iy, iz).s) * fac *
              (lab(ix, iy, iz + 1).u[2] + lab(ix, iy, iz).u[2]);
        }
    }
  }
};
class ComputeDivergence : public Operator {
public:
  ComputeDivergence(SimulationData &s) : Operator(s) {}
  void operator()(const Real dt) {
    const KernelDivergence K(sim);
    compute<VectorLab>(K, sim.vel, sim.chi);
    Real div_loc = 0.0;
    const std::vector<Info> &myInfo = sim.tmpVInfo();
#pragma omp parallel for schedule(static) reduction(+ : div_loc)
    for (size_t i = 0; i < myInfo.size(); i++) {
      const Info &info = myInfo[i];
      const VectorBlock &b = *(const VectorBlock *)info.block;
      for (int iz = 0; iz < VectorBlock::sizeZ; ++iz)
        for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
          for (int ix = 0; ix < VectorBlock::sizeX; ++ix)
            div_loc += std::fabs(b(ix, iy, iz).u[0]);
    }
    Real div_tot = 0.0;
    MPI_Reduce(&div_loc, &div_tot, 1, MPI_Real, MPI_SUM, 0, sim.comm);
    size_t loc = myInfo.size();
    size_t tot;
    MPI_Reduce(&loc, &tot, 1, MPI_LONG, MPI_SUM, 0, sim.comm);
    if (sim.rank == 0) {
      std::cout << "Total div = " << div_tot << std::endl;
      std::ofstream outfile;
      outfile.open("div.txt", std::ios_base::app);
      outfile << sim.time << " " << div_tot << " " << tot << "\n";
      outfile.close();
    }
  }
};
struct SimulationData;
class PoissonSolverBase {
public:
  virtual ~PoissonSolverBase() = default;
  virtual void solve() = 0;

protected:
  typedef typename ScalarGrid::BlockType BlockType;
};
std::shared_ptr<PoissonSolverBase> makePoissonSolver(SimulationData &s);
class ArgumentParser;
class ObstacleFactory {
  SimulationData &sim;

public:
  ObstacleFactory(SimulationData &s) : sim(s) {}
  void addObstacles(ArgumentParser &parser);
  void addObstacles(const std::string &factoryContent);
};
static std::string trim(std::string str) {
  size_t i = 0, j = str.length();
  while (i < j && isspace(str[i]))
    i++;
  while (j > i && isspace(str[j - 1]))
    j--;
  return str.substr(i, j - i);
}
class FactoryFileLineParser : public ArgumentParser {
public:
  FactoryFileLineParser(std::istringstream &is_line)
      : ArgumentParser(0, NULL, '#') {
    std::string key, value;
    while (std::getline(is_line, key, '=')) {
      if (std::getline(is_line, value, ' ')) {
        mapArguments[trim(key)] = Value(trim(value));
      }
    }
    mute();
  }
};
class StefanFish : public Fish {
public:
  StefanFish(SimulationData &s, ArgumentParser &p);
  bool bCorrectPosition;
  bool bCorrectPositionZ;
  bool bCorrectRoll;
  Real origC[3];
  Real wyp;
  Real wzp;
  std::deque<std::array<Real, 4>> r_axis;
  void create() override;
  virtual void computeVelocities() override;
  void act(const Real lTact, const std::vector<Real> &a) const;
  std::vector<Real> state() const;
  Real getPhase(const Real time) const;
  Real getLearnTPeriod() const;
  ssize_t holdingBlockID(const std::array<Real, 3> pos) const;
  std::array<Real, 3> getShear(const std::array<Real, 3> pSurf) const;
};
class CurvatureDefinedFishData : public FishMidlineData {
public:
  Real lastTact = 0;
  Real lastCurv = 0;
  Real oldrCurv = 0;
  Real periodPIDval = Tperiod;
  Real periodPIDdif = 0;
  bool TperiodPID = false;
  Real lastTime = 0;
  Real time0 = 0;
  Real timeshift = 0;
  Schedulers::ParameterSchedulerVector<6> curvatureScheduler;
  Schedulers::ParameterSchedulerLearnWave<7> rlBendingScheduler;
  bool control_torsion{false};
  Schedulers::ParameterSchedulerVector<3> torsionScheduler;
  std::array<Real, 3> torsionValues = {0, 0, 0};
  std::array<Real, 3> torsionValues_previous = {0, 0, 0};
  Real Ttorsion_start = 0.0;
  Real alpha = 1;
  Real dalpha = 0;
  Real beta = 0;
  Real dbeta = 0;
  Real gamma = 0;
  Real dgamma = 0;
  Schedulers::ParameterSchedulerScalar periodScheduler;
  Real current_period = Tperiod;
  Real next_period = Tperiod;
  Real transition_start = 0.0;
  Real transition_duration = 0.1 * Tperiod;

protected:
  Real *const rK;
  Real *const vK;
  Real *const rC;
  Real *const vC;
  Real *const rB;
  Real *const vB;
  Real *const rT;
  Real *const vT;
  Real *const rC_T;
  Real *const vC_T;
  Real *const rB_T;
  Real *const vB_T;

public:
  CurvatureDefinedFishData(Real L, Real T, Real phi, Real _h,
                           const Real _ampFac)
      : FishMidlineData(L, T, phi, _h, _ampFac), rK(_alloc(Nm)), vK(_alloc(Nm)),
        rC(_alloc(Nm)), vC(_alloc(Nm)), rB(_alloc(Nm)), vB(_alloc(Nm)),
        rT(_alloc(Nm)), vT(_alloc(Nm)), rC_T(_alloc(Nm)), vC_T(_alloc(Nm)),
        rB_T(_alloc(Nm)), vB_T(_alloc(Nm)) {}
  void correctTailPeriod(const Real periodFac, const Real periodVel,
                         const Real t, const Real dt) {
    assert(periodFac > 0 && periodFac < 2);
    const Real lastArg = (lastTime - time0) / periodPIDval + timeshift;
    time0 = lastTime;
    timeshift = lastArg;
    periodPIDval = Tperiod * periodFac;
    periodPIDdif = Tperiod * periodVel;
    lastTime = t;
    TperiodPID = true;
  }
  void execute(const Real time, const Real l_tnext,
               const std::vector<Real> &input) override;
  ~CurvatureDefinedFishData() override {
    _dealloc(rK);
    _dealloc(vK);
    _dealloc(rC);
    _dealloc(vC);
    _dealloc(rB);
    _dealloc(vB);
    _dealloc(rT);
    _dealloc(vT);
    _dealloc(rC_T);
    _dealloc(vC_T);
    _dealloc(rB_T);
    _dealloc(vB_T);
  }
  void computeMidline(const Real time, const Real dt) override;
  void performPitchingMotion(const Real time);
  void recomputeNormalVectors();
  void action_curvature(const Real time, const Real l_tnext,
                        const Real action) {
    rlBendingScheduler.Turn(action, l_tnext);
  }
  void action_period(const Real time, const Real l_tnext, const Real action) {
    if (TperiodPID)
      std::cout << "Warning: PID controller should not be used with RL."
                << std::endl;
    current_period = periodPIDval;
    next_period = Tperiod * (1 + action);
    transition_start = l_tnext;
  }
  void action_torsion(const Real time, const Real l_tnext, const Real *action) {
    for (int i = 0; i < 3; i++) {
      torsionValues_previous[i] = torsionValues[i];
      torsionValues[i] = action[i];
    }
    Ttorsion_start = time;
  }
  void action_torsion_pitching_radius(const Real time, const Real l_tnext,
                                      const Real action) {
    const Real sq = 1.0 / pow(2.0, 0.5);
    const Real ar[3] = {action * (norX[0] * 0.000 + norZ[0] * 1.000),
                        action *
                            (norX[Nm / 2 - 1] * sq + norZ[Nm / 2 - 1] * sq),
                        action * (norX[Nm - 1] * 1.000 + norZ[Nm - 1] * 0.000)};
    action_torsion(time, l_tnext, ar);
  }
};
using SymM = std::array<Real, 6>;
using GenM = std::array<Real, 9>;
using GenV = std::array<Real, 3>;
static inline SymM invertSym(const SymM J) {
  const Real detJ = J[0] * (J[1] * J[2] - J[5] * J[5]) +
                    J[3] * (J[4] * J[5] - J[2] * J[3]) +
                    J[4] * (J[3] * J[5] - J[1] * J[4]);
  if (std::fabs(detJ) <= std::numeric_limits<Real>::min()) {
    return SymM{{0, 0, 0, 0, 0, 0}};
  } else {
    return SymM{
        {(J[1] * J[2] - J[5] * J[5]) / detJ, (J[0] * J[2] - J[4] * J[4]) / detJ,
         (J[0] * J[1] - J[3] * J[3]) / detJ, (J[4] * J[5] - J[2] * J[3]) / detJ,
         (J[3] * J[5] - J[1] * J[4]) / detJ,
         (J[3] * J[4] - J[0] * J[5]) / detJ}};
  }
}
static inline GenV multSymVec(const SymM J, const GenV V) {
  return GenV{{J[0] * V[0] + J[3] * V[1] + J[4] * V[2],
               J[3] * V[0] + J[1] * V[1] + J[5] * V[2],
               J[4] * V[0] + J[5] * V[1] + J[2] * V[2]}};
}
static inline GenV multGenVec(const GenM J, const GenV V) {
  return GenV{{J[0] * V[0] + J[1] * V[1] + J[2] * V[2],
               J[3] * V[0] + J[4] * V[1] + J[5] * V[2],
               J[6] * V[0] + J[7] * V[1] + J[8] * V[2]}};
}
static inline GenM multSyms(const SymM J, const SymM G) {
  return GenM{{G[0] * J[0] + G[3] * J[3] + G[4] * J[4],
               G[0] * J[3] + G[3] * J[1] + G[4] * J[5],
               G[0] * J[4] + G[4] * J[2] + G[3] * J[5],
               G[3] * J[0] + G[1] * J[3] + G[5] * J[4],
               G[1] * J[1] + G[3] * J[3] + G[5] * J[5],
               G[1] * J[5] + G[3] * J[4] + G[5] * J[2],
               G[4] * J[0] + G[2] * J[4] + G[5] * J[3],
               G[5] * J[1] + G[2] * J[5] + G[4] * J[3],
               G[2] * J[2] + G[4] * J[4] + G[5] * J[5]}};
}
static inline GenM invertGen(const GenM S) {
  const Real detS = S[0] * S[4] * S[8] - S[0] * S[5] * S[7] +
                    S[1] * S[5] * S[6] - S[1] * S[3] * S[8] +
                    S[2] * S[3] * S[7] - S[2] * S[4] * S[6];
  if (std::fabs(detS) <= std::numeric_limits<Real>::min()) {
    return GenM{{0, 0, 0, 0, 0, 0, 0, 0, 0}};
  } else {
    return GenM{{(S[4] * S[8] - S[5] * S[7]) / detS,
                 -(S[1] * S[8] - S[2] * S[7]) / detS,
                 (S[1] * S[5] - S[2] * S[4]) / detS,
                 -(S[3] * S[8] - S[5] * S[6]) / detS,
                 (S[0] * S[8] - S[2] * S[6]) / detS,
                 -(S[0] * S[5] - S[2] * S[3]) / detS,
                 (S[3] * S[7] - S[4] * S[6]) / detS,
                 -(S[0] * S[7] - S[1] * S[6]) / detS,
                 (S[0] * S[4] - S[1] * S[3]) / detS}};
  }
}
static inline GenM multGens(const GenM S, const GenM G) {
  return GenM{{G[0] * S[0] + G[3] * S[1] + G[6] * S[2],
               G[1] * S[0] + G[4] * S[1] + G[7] * S[2],
               G[2] * S[0] + G[5] * S[1] + G[8] * S[2],
               G[0] * S[3] + G[3] * S[4] + G[6] * S[5],
               G[1] * S[3] + G[4] * S[4] + G[7] * S[5],
               G[2] * S[3] + G[5] * S[4] + G[8] * S[5],
               G[0] * S[6] + G[3] * S[7] + G[6] * S[8],
               G[1] * S[6] + G[4] * S[7] + G[7] * S[8],
               G[2] * S[6] + G[5] * S[7] + G[8] * S[8]}};
}
class CreateObstacles : public Operator {
public:
  CreateObstacles(SimulationData &s) : Operator(s) {}
  void operator()(const Real dt);
};
class UpdateObstacles : public Operator {
public:
  UpdateObstacles(SimulationData &s) : Operator(s) {}
  void operator()(const Real dt);
};
class Penalization : public Operator {
public:
  Penalization(SimulationData &s);
  void operator()(const Real dt);
  void preventCollidingObstacles() const;
};
namespace poisson_kernels {
static constexpr int NX = ScalarBlock::sizeX;
static constexpr int NY = ScalarBlock::sizeY;
static constexpr int NZ = ScalarBlock::sizeZ;
static constexpr int N = NX * NY * NZ;
using Block = Real[NZ][NY][NX];
static constexpr int xPad = 4;
using PaddedBlock = Real[NZ + 2][NY + 2][NX + 2 * xPad];
template <int N> static inline Real sum(const Real (&a)[N]) {
  Real s = 0;
  for (int ix = 0; ix < N; ++ix)
    s += a[ix];
  return s;
}
Real kernelPoissonGetZInnerReference(PaddedBlock &__restrict__ p_,
                                     Block &__restrict__ Ax_,
                                     Block &__restrict__ r_,
                                     Block &__restrict__ block_,
                                     const Real sqrNorm0, const Real rr);
Real kernelPoissonGetZInner(PaddedBlock &p, const Real *pW, const Real *pE,
                            Block &__restrict__ Ax, Block &__restrict__ r,
                            Block &__restrict__ block, Real sqrNorm0, Real rr);
void getZImplParallel(const std::vector<Info> &vInfo);
} // namespace poisson_kernels
class ComputeLHS : public Operator {
  struct KernelLHSPoisson {
    const SimulationData &sim;
    KernelLHSPoisson(const SimulationData &s) : sim(s) {}
    const std::vector<Info> &lhsInfo = sim.lhsInfo();
    const int Nx = ScalarBlock::sizeX;
    const int Ny = ScalarBlock::sizeY;
    const int Nz = ScalarBlock::sizeZ;
    const StencilInfo stencil{-1, -1, -1, 2, 2, 2, false, {0}};
    void operator()(const ScalarLab &lab, const Info &info) const {
      ScalarBlock &__restrict__ o = (*sim.lhs)(info.blockID);
      const Real h = info.h;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x) {
            o(x, y, z) =
                h * (lab(x - 1, y, z) + lab(x + 1, y, z) + lab(x, y - 1, z) +
                     lab(x, y + 1, z) + lab(x, y, z - 1) + lab(x, y, z + 1) -
                     6.0 * lab(x, y, z));
          }
      BlockCase<ScalarBlock> *tempCase =
          (BlockCase<ScalarBlock> *)(lhsInfo[info.blockID].auxiliary);
      if (tempCase == nullptr)
        return;
      ScalarElement *const faceXm =
          tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
      ScalarElement *const faceXp =
          tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
      ScalarElement *const faceYm =
          tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
      ScalarElement *const faceYp =
          tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
      ScalarElement *const faceZm =
          tempCase->storedFace[4] ? &tempCase->m_pData[4][0] : nullptr;
      ScalarElement *const faceZp =
          tempCase->storedFace[5] ? &tempCase->m_pData[5][0] : nullptr;
      if (faceXm != nullptr) {
        const int x = 0;
        for (int z = 0; z < Nz; ++z)
          for (int y = 0; y < Ny; ++y)
            faceXm[y + Ny * z] = h * (lab(x, y, z) - lab(x - 1, y, z));
      }
      if (faceXp != nullptr) {
        const int x = Nx - 1;
        for (int z = 0; z < Nz; ++z)
          for (int y = 0; y < Ny; ++y)
            faceXp[y + Ny * z] = h * (lab(x, y, z) - lab(x + 1, y, z));
      }
      if (faceYm != nullptr) {
        const int y = 0;
        for (int z = 0; z < Nz; ++z)
          for (int x = 0; x < Nx; ++x)
            faceYm[x + Nx * z] = h * (lab(x, y, z) - lab(x, y - 1, z));
      }
      if (faceYp != nullptr) {
        const int y = Ny - 1;
        for (int z = 0; z < Nz; ++z)
          for (int x = 0; x < Nx; ++x)
            faceYp[x + Nx * z] = h * (lab(x, y, z) - lab(x, y + 1, z));
      }
      if (faceZm != nullptr) {
        const int z = 0;
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x)
            faceZm[x + Nx * y] = h * (lab(x, y, z) - lab(x, y, z - 1));
      }
      if (faceZp != nullptr) {
        const int z = Nz - 1;
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x)
            faceZp[x + Nx * y] = h * (lab(x, y, z) - lab(x, y, z + 1));
      }
    }
  };

public:
  ComputeLHS(SimulationData &s) : Operator(s) {}
  void operator()(const Real dt) {
    Real avgP = 0;
    int index = -1;
    MPI_Request request;
    const std::vector<Info> &vInfo_lhs = sim.lhsInfo();
    const std::vector<Info> &vInfo_z = sim.presInfo();
    const int Nx = ScalarBlock::sizeX;
    const int Ny = ScalarBlock::sizeY;
    const int Nz = ScalarBlock::sizeZ;
    if (sim.bMeanConstraint <= 2 && sim.bMeanConstraint > 0) {
#pragma omp parallel for reduction(+ : avgP)
      for (size_t i = 0; i < vInfo_z.size(); ++i) {
        const ScalarBlock &__restrict__ Z = (*sim.pres)(i);
        const Real h3 = vInfo_z[i].h * vInfo_z[i].h * vInfo_z[i].h;
        if (vInfo_z[i].index[0] == 0 && vInfo_z[i].index[1] == 0 &&
            vInfo_z[i].index[2] == 0)
          index = i;
        for (int z = 0; z < Nz; ++z)
          for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x)
              avgP += Z(x, y, z).s * h3;
      }
      MPI_Iallreduce(MPI_IN_PLACE, &avgP, 1, MPI_Real, MPI_SUM, sim.comm,
                     &request);
    }
    compute<ScalarLab>(KernelLHSPoisson(sim), sim.pres, sim.lhs);
    if (sim.bMeanConstraint == 0)
      return;
    if (sim.bMeanConstraint <= 2 && sim.bMeanConstraint > 0) {
      MPI_Waitall(1, &request, MPI_STATUSES_IGNORE);
      if (sim.bMeanConstraint == 1 && index != -1) {
        ScalarBlock &__restrict__ LHS = (*sim.lhs)(index);
        LHS(0, 0, 0).s = avgP;
      } else if (sim.bMeanConstraint == 2) {
#pragma omp parallel for
        for (size_t i = 0; i < vInfo_lhs.size(); ++i) {
          ScalarBlock &__restrict__ LHS = (*sim.lhs)(i);
          const Real h3 = vInfo_lhs[i].h * vInfo_lhs[i].h * vInfo_lhs[i].h;
          for (int z = 0; z < Nz; ++z)
            for (int y = 0; y < Ny; ++y)
              for (int x = 0; x < Nx; ++x)
                LHS(x, y, z).s += avgP * h3;
        }
      }
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < vInfo_lhs.size(); ++i) {
        ScalarBlock &__restrict__ LHS = (*sim.lhs)(i);
        const ScalarBlock &__restrict__ Z = (*sim.pres)(i);
        if (vInfo_lhs[i].index[0] == 0 && vInfo_lhs[i].index[1] == 0 &&
            vInfo_lhs[i].index[2] == 0)
          LHS(0, 0, 0).s = Z(0, 0, 0).s;
      }
    }
  }
};
class PoissonSolverAMR : public PoissonSolverBase {
protected:
  SimulationData &sim;
  ComputeLHS findLHS;
#if 1
  void _preconditioner(const std::vector<Real> &input,
                       std::vector<Real> &output) {
    auto &zInfo = sim.pres->getBlocksInfo();
    const size_t Nblocks = zInfo.size();
    const int BSX = VectorBlock::sizeX;
    const int BSY = VectorBlock::sizeY;
    const int BSZ = VectorBlock::sizeZ;
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      ScalarBlock &__restrict__ bb = (*sim.pres)(i);
      for (int iz = 0; iz < BSZ; iz++)
        for (int iy = 0; iy < BSY; iy++)
          for (int ix = 0; ix < BSX; ix++) {
            const int j = i * BSX * BSY * BSZ + iz * BSX * BSY + iy * BSX + ix;
            bb(ix, iy, iz).s = input[j];
          }
    }
#pragma omp parallel
    { poisson_kernels::getZImplParallel(sim.presInfo()); }
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      const ScalarBlock &__restrict__ bb = (*sim.pres)(i);
      for (int iz = 0; iz < BSZ; iz++)
        for (int iy = 0; iy < BSY; iy++)
          for (int ix = 0; ix < BSX; ix++) {
            const int j = i * BSX * BSY * BSZ + iz * BSX * BSY + iy * BSX + ix;
            output[j] = bb(ix, iy, iz).s;
            ;
          }
    }
  }
  void _lhs(std::vector<Real> &input, std::vector<Real> &output) {
    auto &zInfo = sim.pres->getBlocksInfo();
    auto &AxInfo = sim.lhs->getBlocksInfo();
    const size_t Nblocks = zInfo.size();
    const int BSX = VectorBlock::sizeX;
    const int BSY = VectorBlock::sizeY;
    const int BSZ = VectorBlock::sizeZ;
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      ScalarBlock &__restrict__ zz = *(ScalarBlock *)zInfo[i].block;
      for (int iz = 0; iz < BSZ; iz++)
        for (int iy = 0; iy < BSY; iy++)
          for (int ix = 0; ix < BSX; ix++) {
            const int j = i * BSX * BSY * BSZ + iz * BSX * BSY + iy * BSX + ix;
            zz(ix, iy, iz).s = input[j];
          }
    }
    findLHS(0);
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      ScalarBlock &__restrict__ Ax = *(ScalarBlock *)AxInfo[i].block;
      for (int iz = 0; iz < BSZ; iz++)
        for (int iy = 0; iy < BSY; iy++)
          for (int ix = 0; ix < BSX; ix++) {
            const int j = i * BSX * BSY * BSZ + iz * BSX * BSY + iy * BSX + ix;
            output[j] = Ax(ix, iy, iz).s;
          }
    }
  }
  std::vector<Real> b;
  std::vector<Real> phat;
  std::vector<Real> rhat;
  std::vector<Real> shat;
  std::vector<Real> what;
  std::vector<Real> zhat;
  std::vector<Real> qhat;
  std::vector<Real> s;
  std::vector<Real> w;
  std::vector<Real> z;
  std::vector<Real> t;
  std::vector<Real> v;
  std::vector<Real> q;
  std::vector<Real> r;
  std::vector<Real> y;
  std::vector<Real> x;
  std::vector<Real> r0;
  std::vector<Real> x_opt;
#else
  void getZ() {
#pragma omp parallel
    { poisson_kernels::getZImplParallel(sim.presInfo()); }
  }
  std::vector<Real> x;
  std::vector<Real> r;
  std::vector<Real> p;
  std::vector<Real> v;
  std::vector<Real> s;
  std::vector<Real> rhat;
  size_t _dest(const Info &info, const int iz, const int iy,
               const int ix) const {
    return BlockType::sizeX *
               (BlockType::sizeY * (info.blockID * BlockType::sizeZ + iz) +
                iy) +
           ix;
  }
#endif
public:
  PoissonSolverAMR(SimulationData &ss) : sim(ss), findLHS(ss) {}
  PoissonSolverAMR(const PoissonSolverAMR &c) = delete;
  void solve();
};
class PoissonSolverBase;
class PressureProjection : public Operator {
protected:
  std::shared_ptr<PoissonSolverBase> pressureSolver;
  std::vector<Real> pOld;

public:
  PressureProjection(SimulationData &s);
  ~PressureProjection() = default;
  void operator()(Real dt) override;
};
struct Simulation {
  ArgumentParser parser;
  SimulationData sim;
  void setupOperators();
  void setupGrid();
  void _ic();
  Simulation(int argc, char **argv, MPI_Comm comm);
  void init();
  void simulate();
  void adaptMesh();
  const std::vector<std::shared_ptr<Obstacle>> &getShapes() const;
  Real calcMaxTimestep();
  bool advance(Real dt);
};
struct KernelAdvectDiffuse {
  KernelAdvectDiffuse(const SimulationData &s, const Real a_coef)
      : sim(s), coef(a_coef) {}
  const SimulationData &sim;
  const Real dt = sim.dt;
  const Real mu = sim.nu;
  const Real coef;
  const std::array<Real, 3> &uInf = sim.uinf;
  const std::vector<Info> &tmpVInfo = sim.tmpVInfo();
  const StencilInfo stencil{-3, -3, -3, 4, 4, 4, false, {0, 1, 2}};
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  inline Real derivative(const Real &U, const Real &um3, const Real &um2,
                         const Real &um1, const Real &u, const Real &up1,
                         const Real &up2, const Real &up3) const {
    if (U > 0)
      return (-2 * um3 + 15 * um2 - 60 * um1 + 20 * u + 30 * up1 - 3 * up2) /
             60.;
    else
      return (2 * up3 - 15 * up2 + 60 * up1 - 20 * u - 30 * um1 + 3 * um2) /
             60.;
  }
  void operator()(const VectorLab &lab, const Info &info) const {
    VectorBlock &o = (*sim.tmpV)(info.blockID);
    const Real h3 = info.h * info.h * info.h;
    const Real facA = -dt / info.h * h3 * coef;
    const Real facD = (mu / info.h) * (dt / info.h) * h3 * coef;
    for (int z = 0; z < Nz; ++z)
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          const Real uAbs[3] = {lab(x, y, z).u[0] + uInf[0],
                                lab(x, y, z).u[1] + uInf[1],
                                lab(x, y, z).u[2] + uInf[2]};
          const Real dudx = derivative(
              uAbs[0], lab(x - 3, y, z).u[0], lab(x - 2, y, z).u[0],
              lab(x - 1, y, z).u[0], lab(x, y, z).u[0], lab(x + 1, y, z).u[0],
              lab(x + 2, y, z).u[0], lab(x + 3, y, z).u[0]);
          const Real dvdx = derivative(
              uAbs[0], lab(x - 3, y, z).u[1], lab(x - 2, y, z).u[1],
              lab(x - 1, y, z).u[1], lab(x, y, z).u[1], lab(x + 1, y, z).u[1],
              lab(x + 2, y, z).u[1], lab(x + 3, y, z).u[1]);
          const Real dwdx = derivative(
              uAbs[0], lab(x - 3, y, z).u[2], lab(x - 2, y, z).u[2],
              lab(x - 1, y, z).u[2], lab(x, y, z).u[2], lab(x + 1, y, z).u[2],
              lab(x + 2, y, z).u[2], lab(x + 3, y, z).u[2]);
          const Real dudy = derivative(
              uAbs[1], lab(x, y - 3, z).u[0], lab(x, y - 2, z).u[0],
              lab(x, y - 1, z).u[0], lab(x, y, z).u[0], lab(x, y + 1, z).u[0],
              lab(x, y + 2, z).u[0], lab(x, y + 3, z).u[0]);
          const Real dvdy = derivative(
              uAbs[1], lab(x, y - 3, z).u[1], lab(x, y - 2, z).u[1],
              lab(x, y - 1, z).u[1], lab(x, y, z).u[1], lab(x, y + 1, z).u[1],
              lab(x, y + 2, z).u[1], lab(x, y + 3, z).u[1]);
          const Real dwdy = derivative(
              uAbs[1], lab(x, y - 3, z).u[2], lab(x, y - 2, z).u[2],
              lab(x, y - 1, z).u[2], lab(x, y, z).u[2], lab(x, y + 1, z).u[2],
              lab(x, y + 2, z).u[2], lab(x, y + 3, z).u[2]);
          const Real dudz = derivative(
              uAbs[2], lab(x, y, z - 3).u[0], lab(x, y, z - 2).u[0],
              lab(x, y, z - 1).u[0], lab(x, y, z).u[0], lab(x, y, z + 1).u[0],
              lab(x, y, z + 2).u[0], lab(x, y, z + 3).u[0]);
          const Real dvdz = derivative(
              uAbs[2], lab(x, y, z - 3).u[1], lab(x, y, z - 2).u[1],
              lab(x, y, z - 1).u[1], lab(x, y, z).u[1], lab(x, y, z + 1).u[1],
              lab(x, y, z + 2).u[1], lab(x, y, z + 3).u[1]);
          const Real dwdz = derivative(
              uAbs[2], lab(x, y, z - 3).u[2], lab(x, y, z - 2).u[2],
              lab(x, y, z - 1).u[2], lab(x, y, z).u[2], lab(x, y, z + 1).u[2],
              lab(x, y, z + 2).u[2], lab(x, y, z + 3).u[2]);
          const Real duD = ((lab(x + 1, y, z).u[0] + lab(x - 1, y, z).u[0]) +
                            ((lab(x, y + 1, z).u[0] + lab(x, y - 1, z).u[0]) +
                             (lab(x, y, z + 1).u[0] + lab(x, y, z - 1).u[0]))) -
                           6 * lab(x, y, z).u[0];
          const Real dvD = ((lab(x, y + 1, z).u[1] + lab(x, y - 1, z).u[1]) +
                            ((lab(x, y, z + 1).u[1] + lab(x, y, z - 1).u[1]) +
                             (lab(x + 1, y, z).u[1] + lab(x - 1, y, z).u[1]))) -
                           6 * lab(x, y, z).u[1];
          const Real dwD = ((lab(x, y, z + 1).u[2] + lab(x, y, z - 1).u[2]) +
                            ((lab(x + 1, y, z).u[2] + lab(x - 1, y, z).u[2]) +
                             (lab(x, y + 1, z).u[2] + lab(x, y - 1, z).u[2]))) -
                           6 * lab(x, y, z).u[2];
          const Real duA = uAbs[0] * dudx + (uAbs[1] * dudy + uAbs[2] * dudz);
          const Real dvA = uAbs[1] * dvdy + (uAbs[2] * dvdz + uAbs[0] * dvdx);
          const Real dwA = uAbs[2] * dwdz + (uAbs[0] * dwdx + uAbs[1] * dwdy);
          o(x, y, z).u[0] += facA * duA + facD * duD;
          o(x, y, z).u[1] += facA * dvA + facD * dvD;
          o(x, y, z).u[2] += facA * dwA + facD * dwD;
        }
    BlockCase<VectorBlock> *tempCase =
        (BlockCase<VectorBlock> *)(tmpVInfo[info.blockID].auxiliary);
    if (tempCase == nullptr)
      return;
    VectorElement *const faceXm =
        tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
    VectorElement *const faceXp =
        tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
    VectorElement *const faceYm =
        tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
    VectorElement *const faceYp =
        tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    VectorElement *const faceZm =
        tempCase->storedFace[4] ? &tempCase->m_pData[4][0] : nullptr;
    VectorElement *const faceZp =
        tempCase->storedFace[5] ? &tempCase->m_pData[5][0] : nullptr;
    if (faceXm != nullptr) {
      const int x = 0;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y) {
          faceXm[y + Ny * z].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x - 1, y, z).u[0]);
          faceXm[y + Ny * z].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x - 1, y, z).u[1]);
          faceXm[y + Ny * z].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x - 1, y, z).u[2]);
        }
    }
    if (faceXp != nullptr) {
      const int x = Nx - 1;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y) {
          faceXp[y + Ny * z].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x + 1, y, z).u[0]);
          faceXp[y + Ny * z].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x + 1, y, z).u[1]);
          faceXp[y + Ny * z].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x + 1, y, z).u[2]);
        }
    }
    if (faceYm != nullptr) {
      const int y = 0;
      for (int z = 0; z < Nz; ++z)
        for (int x = 0; x < Nx; ++x) {
          faceYm[x + Nx * z].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x, y - 1, z).u[0]);
          faceYm[x + Nx * z].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x, y - 1, z).u[1]);
          faceYm[x + Nx * z].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x, y - 1, z).u[2]);
        }
    }
    if (faceYp != nullptr) {
      const int y = Ny - 1;
      for (int z = 0; z < Nz; ++z)
        for (int x = 0; x < Nx; ++x) {
          faceYp[x + Nx * z].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x, y + 1, z).u[0]);
          faceYp[x + Nx * z].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x, y + 1, z).u[1]);
          faceYp[x + Nx * z].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x, y + 1, z).u[2]);
        }
    }
    if (faceZm != nullptr) {
      const int z = 0;
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          faceZm[x + Nx * y].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x, y, z - 1).u[0]);
          faceZm[x + Nx * y].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x, y, z - 1).u[1]);
          faceZm[x + Nx * y].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x, y, z - 1).u[2]);
        }
    }
    if (faceZp != nullptr) {
      const int z = Nz - 1;
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          faceZp[x + Nx * y].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x, y, z + 1).u[0]);
          faceZp[x + Nx * y].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x, y, z + 1).u[1]);
          faceZp[x + Nx * y].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x, y, z + 1).u[2]);
        }
    }
  }
};
void AdvectionDiffusion::operator()(const Real dt) {
  const std::vector<Info> &velInfo = sim.velInfo();
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  const size_t Nblocks = velInfo.size();
#if 0
    vOld.resize(Nx*Ny*Nz*Nblocks*3);
#pragma omp parallel for
    for(size_t i=0; i<Nblocks; i++)
    {
        const VectorBlock & V = (*sim.vel)(i);
        for (int z=0; z<Nz; ++z)
        for (int y=0; y<Ny; ++y)
        for (int x=0; x<Nx; ++x)
        {
          const int idx = i*Nx*Ny*Nz*3+z*Ny*Nx*3+y*Nx*3+x*3;
          vOld[idx ] = V(x,y,z).u[0];
          vOld[idx+1] = V(x,y,z).u[1];
          vOld[idx+2] = V(x,y,z).u[2];
        }
    }
    const KernelAdvectDiffuse step1(sim,0.5);
    compute<VectorLab>(step1,sim.vel,sim.tmpV);
#pragma omp parallel for
    for(size_t i=0; i<Nblocks; i++)
    {
        const Real ih3 = 1.0/(velInfo[i].h*velInfo[i].h*velInfo[i].h);
        const VectorBlock & tmpV = (*sim.tmpV)(i);
        VectorBlock & V = (*sim.vel )(i);
        for (int z=0; z<Nz; ++z)
        for (int y=0; y<Ny; ++y)
        for (int x=0; x<Nx; ++x)
        {
          const int idx = i*Nx*Ny*Nz*3+z*Ny*Nx*3+y*Nx*3+x*3;
          V(x,y,z).u[0] = vOld[idx ] + tmpV(x,y,z).u[0]*ih3;
          V(x,y,z).u[1] = vOld[idx+1] + tmpV(x,y,z).u[1]*ih3;
          V(x,y,z).u[2] = vOld[idx+2] + tmpV(x,y,z).u[2]*ih3;
        }
    }
    const KernelAdvectDiffuse step2(sim,1.0);
    compute<VectorLab>(step2,sim.vel,sim.tmpV);
#pragma omp parallel for
    for(size_t i=0; i<Nblocks; i++)
    {
        const Real ih3 = 1.0/(velInfo[i].h*velInfo[i].h*velInfo[i].h);
        const VectorBlock & tmpV = (*sim.tmpV)(i);
        VectorBlock & V = (*sim.vel )(i);
        for (int z=0; z<Nz; ++z)
        for (int y=0; y<Ny; ++y)
        for (int x=0; x<Nx; ++x)
        {
          const int idx = i*Nx*Ny*Nz*3+z*Ny*Nx*3+y*Nx*3+x*3;
          V(x,y,z).u[0] = vOld[idx ] + tmpV(x,y,z).u[0]*ih3;
          V(x,y,z).u[1] = vOld[idx+1] + tmpV(x,y,z).u[1]*ih3;
          V(x,y,z).u[2] = vOld[idx+2] + tmpV(x,y,z).u[2]*ih3;
        }
    }
#else
  const KernelAdvectDiffuse step(sim, 1.0);
  const Real alpha[3] = {1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0};
  const Real beta[3] = {-5.0 / 9.0, -153.0 / 128.0, 0.0};
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    VectorBlock &tmpV = (*sim.tmpV)(i);
    tmpV.clear();
  }
  for (int RKstep = 0; RKstep < 3; RKstep++) {
    compute<VectorLab>(step, sim.vel, sim.tmpV);
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      const Real ih3 =
          alpha[RKstep] / (velInfo[i].h * velInfo[i].h * velInfo[i].h);
      VectorBlock &tmpV = (*sim.tmpV)(i);
      VectorBlock &V = (*sim.vel)(i);
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x) {
            V(x, y, z).u[0] += tmpV(x, y, z).u[0] * ih3;
            V(x, y, z).u[1] += tmpV(x, y, z).u[1] * ih3;
            V(x, y, z).u[2] += tmpV(x, y, z).u[2] * ih3;
            tmpV(x, y, z).u[0] *= beta[RKstep];
            tmpV(x, y, z).u[1] *= beta[RKstep];
            tmpV(x, y, z).u[2] *= beta[RKstep];
          }
    }
  }
#endif
}
struct KernelDiffusionRHS {
  SimulationData &sim;
  StencilInfo stencil = StencilInfo(-1, -1, -1, 2, 2, 2, false, {0, 1, 2});
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  const std::vector<Info> &tmpVInfo = sim.tmpVInfo();
  KernelDiffusionRHS(SimulationData &s) : sim(s) {}
  void operator()(const VectorLab &lab, const Info &info) const {
    VectorBlock &__restrict__ TMPV = (*sim.tmpV)(info.blockID);
    const Real facD = info.h;
    for (int z = 0; z < Nz; ++z)
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          const Real duD = ((lab(x + 1, y, z).u[0] + lab(x - 1, y, z).u[0]) +
                            ((lab(x, y + 1, z).u[0] + lab(x, y - 1, z).u[0]) +
                             (lab(x, y, z + 1).u[0] + lab(x, y, z - 1).u[0]))) -
                           6 * lab(x, y, z).u[0];
          const Real dvD = ((lab(x, y + 1, z).u[1] + lab(x, y - 1, z).u[1]) +
                            ((lab(x, y, z + 1).u[1] + lab(x, y, z - 1).u[1]) +
                             (lab(x + 1, y, z).u[1] + lab(x - 1, y, z).u[1]))) -
                           6 * lab(x, y, z).u[1];
          const Real dwD = ((lab(x, y, z + 1).u[2] + lab(x, y, z - 1).u[2]) +
                            ((lab(x + 1, y, z).u[2] + lab(x - 1, y, z).u[2]) +
                             (lab(x, y + 1, z).u[2] + lab(x, y - 1, z).u[2]))) -
                           6 * lab(x, y, z).u[2];
          TMPV(x, y, z).u[0] = facD * duD;
          TMPV(x, y, z).u[1] = facD * dvD;
          TMPV(x, y, z).u[2] = facD * dwD;
        }
    BlockCase<VectorBlock> *tempCase =
        (BlockCase<VectorBlock> *)(tmpVInfo[info.blockID].auxiliary);
    if (tempCase == nullptr)
      return;
    VectorElement *const faceXm =
        tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
    VectorElement *const faceXp =
        tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
    VectorElement *const faceYm =
        tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
    VectorElement *const faceYp =
        tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    VectorElement *const faceZm =
        tempCase->storedFace[4] ? &tempCase->m_pData[4][0] : nullptr;
    VectorElement *const faceZp =
        tempCase->storedFace[5] ? &tempCase->m_pData[5][0] : nullptr;
    if (faceXm != nullptr) {
      const int x = 0;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y) {
          faceXm[y + Ny * z].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x - 1, y, z).u[0]);
          faceXm[y + Ny * z].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x - 1, y, z).u[1]);
          faceXm[y + Ny * z].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x - 1, y, z).u[2]);
        }
    }
    if (faceXp != nullptr) {
      const int x = Nx - 1;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y) {
          faceXp[y + Ny * z].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x + 1, y, z).u[0]);
          faceXp[y + Ny * z].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x + 1, y, z).u[1]);
          faceXp[y + Ny * z].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x + 1, y, z).u[2]);
        }
    }
    if (faceYm != nullptr) {
      const int y = 0;
      for (int z = 0; z < Nz; ++z)
        for (int x = 0; x < Nx; ++x) {
          faceYm[x + Nx * z].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x, y - 1, z).u[0]);
          faceYm[x + Nx * z].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x, y - 1, z).u[1]);
          faceYm[x + Nx * z].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x, y - 1, z).u[2]);
        }
    }
    if (faceYp != nullptr) {
      const int y = Ny - 1;
      for (int z = 0; z < Nz; ++z)
        for (int x = 0; x < Nx; ++x) {
          faceYp[x + Nx * z].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x, y + 1, z).u[0]);
          faceYp[x + Nx * z].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x, y + 1, z).u[1]);
          faceYp[x + Nx * z].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x, y + 1, z).u[2]);
        }
    }
    if (faceZm != nullptr) {
      const int z = 0;
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          faceZm[x + Nx * y].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x, y, z - 1).u[0]);
          faceZm[x + Nx * y].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x, y, z - 1).u[1]);
          faceZm[x + Nx * y].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x, y, z - 1).u[2]);
        }
    }
    if (faceZp != nullptr) {
      const int z = Nz - 1;
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          faceZp[x + Nx * y].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x, y, z + 1).u[0]);
          faceZp[x + Nx * y].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x, y, z + 1).u[1]);
          faceZp[x + Nx * y].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x, y, z + 1).u[2]);
        }
    }
  }
};
struct KernelAdvect {
  KernelAdvect(const SimulationData &s, const Real _dt) : sim(s), dt(_dt) {}
  const SimulationData &sim;
  const Real dt;
  const Real mu = sim.nu;
  const std::array<Real, 3> &uInf = sim.uinf;
  const std::vector<Info> &tmpVInfo = sim.tmpVInfo();
  const StencilInfo stencil{-3, -3, -3, 4, 4, 4, false, {0, 1, 2}};
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  inline Real derivative(const Real &U, const Real &um3, const Real &um2,
                         const Real &um1, const Real &u, const Real &up1,
                         const Real &up2, const Real &up3) const {
    if (U > 0)
      return (-2 * um3 + 15 * um2 - 60 * um1 + 20 * u + 30 * up1 - 3 * up2) /
             60.;
    else
      return (2 * up3 - 15 * up2 + 60 * up1 - 20 * u - 30 * um1 + 3 * um2) /
             60.;
  }
  void operator()(const VectorLab &lab, const Info &info) const {
    VectorBlock &o = (*sim.tmpV)(info.blockID);
    VectorBlock &v = (*sim.vel)(info.blockID);
    const Real h3 = info.h * info.h * info.h;
    const Real facA = -dt / info.h * h3;
    const Real facD = (mu / info.h) * (dt / info.h) * h3;
    for (int z = 0; z < Nz; ++z)
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          const Real uAbs[3] = {lab(x, y, z).u[0] + uInf[0],
                                lab(x, y, z).u[1] + uInf[1],
                                lab(x, y, z).u[2] + uInf[2]};
          const Real dudx = derivative(
              uAbs[0], lab(x - 3, y, z).u[0], lab(x - 2, y, z).u[0],
              lab(x - 1, y, z).u[0], lab(x, y, z).u[0], lab(x + 1, y, z).u[0],
              lab(x + 2, y, z).u[0], lab(x + 3, y, z).u[0]);
          const Real dvdx = derivative(
              uAbs[0], lab(x - 3, y, z).u[1], lab(x - 2, y, z).u[1],
              lab(x - 1, y, z).u[1], lab(x, y, z).u[1], lab(x + 1, y, z).u[1],
              lab(x + 2, y, z).u[1], lab(x + 3, y, z).u[1]);
          const Real dwdx = derivative(
              uAbs[0], lab(x - 3, y, z).u[2], lab(x - 2, y, z).u[2],
              lab(x - 1, y, z).u[2], lab(x, y, z).u[2], lab(x + 1, y, z).u[2],
              lab(x + 2, y, z).u[2], lab(x + 3, y, z).u[2]);
          const Real dudy = derivative(
              uAbs[1], lab(x, y - 3, z).u[0], lab(x, y - 2, z).u[0],
              lab(x, y - 1, z).u[0], lab(x, y, z).u[0], lab(x, y + 1, z).u[0],
              lab(x, y + 2, z).u[0], lab(x, y + 3, z).u[0]);
          const Real dvdy = derivative(
              uAbs[1], lab(x, y - 3, z).u[1], lab(x, y - 2, z).u[1],
              lab(x, y - 1, z).u[1], lab(x, y, z).u[1], lab(x, y + 1, z).u[1],
              lab(x, y + 2, z).u[1], lab(x, y + 3, z).u[1]);
          const Real dwdy = derivative(
              uAbs[1], lab(x, y - 3, z).u[2], lab(x, y - 2, z).u[2],
              lab(x, y - 1, z).u[2], lab(x, y, z).u[2], lab(x, y + 1, z).u[2],
              lab(x, y + 2, z).u[2], lab(x, y + 3, z).u[2]);
          const Real dudz = derivative(
              uAbs[2], lab(x, y, z - 3).u[0], lab(x, y, z - 2).u[0],
              lab(x, y, z - 1).u[0], lab(x, y, z).u[0], lab(x, y, z + 1).u[0],
              lab(x, y, z + 2).u[0], lab(x, y, z + 3).u[0]);
          const Real dvdz = derivative(
              uAbs[2], lab(x, y, z - 3).u[1], lab(x, y, z - 2).u[1],
              lab(x, y, z - 1).u[1], lab(x, y, z).u[1], lab(x, y, z + 1).u[1],
              lab(x, y, z + 2).u[1], lab(x, y, z + 3).u[1]);
          const Real dwdz = derivative(
              uAbs[2], lab(x, y, z - 3).u[2], lab(x, y, z - 2).u[2],
              lab(x, y, z - 1).u[2], lab(x, y, z).u[2], lab(x, y, z + 1).u[2],
              lab(x, y, z + 2).u[2], lab(x, y, z + 3).u[2]);
          const Real duD = ((lab(x + 1, y, z).u[0] + lab(x - 1, y, z).u[0]) +
                            ((lab(x, y + 1, z).u[0] + lab(x, y - 1, z).u[0]) +
                             (lab(x, y, z + 1).u[0] + lab(x, y, z - 1).u[0]))) -
                           6 * lab(x, y, z).u[0];
          const Real dvD = ((lab(x, y + 1, z).u[1] + lab(x, y - 1, z).u[1]) +
                            ((lab(x, y, z + 1).u[1] + lab(x, y, z - 1).u[1]) +
                             (lab(x + 1, y, z).u[1] + lab(x - 1, y, z).u[1]))) -
                           6 * lab(x, y, z).u[1];
          const Real dwD = ((lab(x, y, z + 1).u[2] + lab(x, y, z - 1).u[2]) +
                            ((lab(x + 1, y, z).u[2] + lab(x - 1, y, z).u[2]) +
                             (lab(x, y + 1, z).u[2] + lab(x, y - 1, z).u[2]))) -
                           6 * lab(x, y, z).u[2];
          const Real duA = uAbs[0] * dudx + (uAbs[1] * dudy + uAbs[2] * dudz);
          const Real dvA = uAbs[1] * dvdy + (uAbs[2] * dvdz + uAbs[0] * dvdx);
          const Real dwA = uAbs[2] * dwdz + (uAbs[0] * dwdx + uAbs[1] * dwdy);
          o(x, y, z).u[0] = facD * duD;
          o(x, y, z).u[1] = facD * dvD;
          o(x, y, z).u[2] = facD * dwD;
          v(x, y, z).u[0] += facA * duA / h3;
          v(x, y, z).u[1] += facA * dvA / h3;
          v(x, y, z).u[2] += facA * dwA / h3;
        }
    BlockCase<VectorBlock> *tempCase =
        (BlockCase<VectorBlock> *)(tmpVInfo[info.blockID].auxiliary);
    if (tempCase == nullptr)
      return;
    VectorElement *const faceXm =
        tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
    VectorElement *const faceXp =
        tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
    VectorElement *const faceYm =
        tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
    VectorElement *const faceYp =
        tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    VectorElement *const faceZm =
        tempCase->storedFace[4] ? &tempCase->m_pData[4][0] : nullptr;
    VectorElement *const faceZp =
        tempCase->storedFace[5] ? &tempCase->m_pData[5][0] : nullptr;
    if (faceXm != nullptr) {
      const int x = 0;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y) {
          faceXm[y + Ny * z].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x - 1, y, z).u[0]);
          faceXm[y + Ny * z].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x - 1, y, z).u[1]);
          faceXm[y + Ny * z].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x - 1, y, z).u[2]);
        }
    }
    if (faceXp != nullptr) {
      const int x = Nx - 1;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y) {
          faceXp[y + Ny * z].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x + 1, y, z).u[0]);
          faceXp[y + Ny * z].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x + 1, y, z).u[1]);
          faceXp[y + Ny * z].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x + 1, y, z).u[2]);
        }
    }
    if (faceYm != nullptr) {
      const int y = 0;
      for (int z = 0; z < Nz; ++z)
        for (int x = 0; x < Nx; ++x) {
          faceYm[x + Nx * z].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x, y - 1, z).u[0]);
          faceYm[x + Nx * z].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x, y - 1, z).u[1]);
          faceYm[x + Nx * z].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x, y - 1, z).u[2]);
        }
    }
    if (faceYp != nullptr) {
      const int y = Ny - 1;
      for (int z = 0; z < Nz; ++z)
        for (int x = 0; x < Nx; ++x) {
          faceYp[x + Nx * z].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x, y + 1, z).u[0]);
          faceYp[x + Nx * z].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x, y + 1, z).u[1]);
          faceYp[x + Nx * z].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x, y + 1, z).u[2]);
        }
    }
    if (faceZm != nullptr) {
      const int z = 0;
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          faceZm[x + Nx * y].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x, y, z - 1).u[0]);
          faceZm[x + Nx * y].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x, y, z - 1).u[1]);
          faceZm[x + Nx * y].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x, y, z - 1).u[2]);
        }
    }
    if (faceZp != nullptr) {
      const int z = Nz - 1;
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          faceZp[x + Nx * y].u[0] =
              facD * (lab(x, y, z).u[0] - lab(x, y, z + 1).u[0]);
          faceZp[x + Nx * y].u[1] =
              facD * (lab(x, y, z).u[1] - lab(x, y, z + 1).u[1]);
          faceZp[x + Nx * y].u[2] =
              facD * (lab(x, y, z).u[2] - lab(x, y, z + 1).u[2]);
        }
    }
  }
};
void AdvectionDiffusionImplicit::euler(const Real dt) {
  const std::vector<Info> &velInfo = sim.velInfo();
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  const size_t Nblocks = velInfo.size();
  pressure.resize(Nblocks * Nx * Ny * Nz);
  velocity.resize(Nblocks * Nx * Ny * Nz * 3);
  compute<VectorLab>(KernelAdvect(sim, dt), sim.vel, sim.tmpV);
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    const VectorBlock &TMPV = (*sim.tmpV)(i);
    const ScalarBlock &P = (*sim.pres)(i);
    VectorBlock &V = (*sim.vel)(i);
    const Real ih3 = 1.0 / (velInfo[i].h * velInfo[i].h * velInfo[i].h);
    for (int z = 0; z < Nz; ++z)
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          const int idx = i * Nx * Ny * Nz + z * Ny * Nx + y * Nx + x;
          pressure[idx] = P(x, y, z).s;
          velocity[3 * idx + 0] = V(x, y, z).u[0];
          velocity[3 * idx + 1] = V(x, y, z).u[1];
          velocity[3 * idx + 2] = V(x, y, z).u[2];
          V(x, y, z).u[0] = TMPV(x, y, z).u[0] * ih3 + V(x, y, z).u[0];
          V(x, y, z).u[1] = TMPV(x, y, z).u[1] * ih3 + V(x, y, z).u[1];
          V(x, y, z).u[2] = TMPV(x, y, z).u[2] * ih3 + V(x, y, z).u[2];
        }
  }
  compute<VectorLab>(KernelDiffusionRHS(sim), sim.vel, sim.tmpV);
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    VectorBlock &V = (*sim.vel)(i);
    VectorBlock &TMPV = (*sim.tmpV)(i);
    const Real ih3 = 1.0 / (velInfo[i].h * velInfo[i].h * velInfo[i].h);
    for (int z = 0; z < Nz; ++z)
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          const int idx = i * Nx * Ny * Nz + z * Ny * Nx + y * Nx + x;
          TMPV(x, y, z).u[0] =
              -TMPV(x, y, z).u[0] * ih3 +
              (V(x, y, z).u[0] - velocity[3 * idx + 0]) / (dt * sim.nu);
          TMPV(x, y, z).u[1] =
              -TMPV(x, y, z).u[1] * ih3 +
              (V(x, y, z).u[1] - velocity[3 * idx + 1]) / (dt * sim.nu);
          TMPV(x, y, z).u[2] =
              -TMPV(x, y, z).u[2] * ih3 +
              (V(x, y, z).u[2] - velocity[3 * idx + 2]) / (dt * sim.nu);
        }
  }
  DiffusionSolver mmysolver(sim);
  for (int index = 0; index < 3; index++) {
    mmysolver.mydirection = index;
    mmysolver.dt = dt;
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      const Real h3 = (velInfo[i].h * velInfo[i].h * velInfo[i].h);
      ScalarBlock &RHS = (*sim.lhs)(i);
      ScalarBlock &P = (*sim.pres)(i);
      const VectorBlock &TMPV = (*sim.tmpV)(i);
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x) {
            P(x, y, z).s = 0;
            RHS(x, y, z).s = h3 * TMPV(x, y, z).u[index];
          }
    }
    mmysolver.solve();
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      ScalarBlock &P = (*sim.pres)(i);
      VectorBlock &V = (*sim.vel)(i);
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x) {
            V(x, y, z).u[index] += P(x, y, z).s;
          }
    }
  }
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    ScalarBlock &P = (*sim.pres)(i);
    for (int z = 0; z < Nz; ++z)
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          const int idx = i * Nx * Ny * Nz + z * Ny * Nx + y * Nx + x;
          P(x, y, z).s = pressure[idx];
        }
  }
}
void AdvectionDiffusionImplicit::operator()(const Real dt) { euler(sim.dt); }
double Value::asDouble(double def) {
  if (content == "") {
    std::ostringstream sbuf;
    sbuf << def;
    content = sbuf.str();
  }
  return (double)atof(content.c_str());
}
int Value::asInt(int def) {
  if (content == "") {
    std::ostringstream sbuf;
    sbuf << def;
    content = sbuf.str();
  }
  return atoi(content.c_str());
}
bool Value::asBool(bool def) {
  if (content == "") {
    if (def)
      content = "true";
    else
      content = "false";
  }
  if (content == "0")
    return false;
  if (content == "false")
    return false;
  return true;
}
std::string Value::asString(const std::string &def) {
  if (content == "")
    content = def;
  return content;
}
std::ostream &operator<<(std::ostream &lhs, const Value &rhs) {
  lhs << rhs.content;
  return lhs;
}
static inline void _normalizeKey(std::string &key) {
  if (key[0] == '-')
    key.erase(0, 1);
  if (key[0] == '+')
    key.erase(0, 1);
}
static inline bool _existKey(const std::string &key,
                             const std::map<std::string, Value> &container) {
  return container.find(key) != container.end();
}
Value &CommandlineParser::operator()(std::string key) {
  _normalizeKey(key);
  if (bStrictMode) {
    if (!_existKey(key, mapArguments)) {
      printf("Runtime option NOT SPECIFIED! ABORTING! name: %s\n", key.data());
      abort();
    }
  }
  if (bVerbose)
    printf("%s is %s\n", key.data(), mapArguments[key].asString().data());
  return mapArguments[key];
}
bool CommandlineParser::check(std::string key) const {
  _normalizeKey(key);
  return _existKey(key, mapArguments);
}
bool CommandlineParser::_isnumber(const std::string &s) const {
  char *end = NULL;
  strtod(s.c_str(), &end);
  return end != s.c_str();
}
CommandlineParser::CommandlineParser(const int argc, char **argv)
    : iArgC(argc), vArgV(argv), bStrictMode(false), bVerbose(true) {
  for (int i = 1; i < argc; i++)
    if (argv[i][0] == '-') {
      std::string values = "";
      int itemCount = 0;
      for (int j = i + 1; j < argc; j++) {
        std::string sval(argv[j]);
        const bool leadingDash = (sval[0] == '-');
        const bool isNumeric = _isnumber(sval);
        if (leadingDash && !isNumeric)
          break;
        else {
          if (std::strcmp(values.c_str(), ""))
            values += ' ';
          values += argv[j];
          itemCount++;
        }
      }
      if (itemCount == 0)
        values = "true";
      std::string key(argv[i]);
      key.erase(0, 1);
      if (key[0] == '+') {
        key.erase(0, 1);
        if (!_existKey(key, mapArguments))
          mapArguments[key] = Value(values);
        else
          mapArguments[key] += Value(values);
      } else {
        if (!_existKey(key, mapArguments))
          mapArguments[key] = Value(values);
      }
      i += itemCount;
    }
  mute();
}
void CommandlineParser::save_options(const std::string &path) {
  std::string options;
  for (std::map<std::string, Value>::iterator it = mapArguments.begin();
       it != mapArguments.end(); it++) {
    options += it->first + " " + it->second.asString() + " ";
  }
  std::string filepath = path + "/argumentparser.log";
  FILE *f = fopen(filepath.data(), "a");
  if (f == NULL) {
    fprintf(stderr, "impossible to write %s.\n", filepath.data());
    return;
  }
  fprintf(f, "%s\n", options.data());
  fclose(f);
}
void ArgumentParser::_ignoreComments(std::istream &stream,
                                     const char commentChar) {
  stream >> std::ws;
  int nextchar = stream.peek();
  while (nextchar == commentChar) {
    stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    stream >> std::ws;
    nextchar = stream.peek();
  }
}
void ArgumentParser::_parseFile(std::ifstream &stream, ArgMap &container) {
  _ignoreComments(stream, commentStart);
  while (!stream.eof()) {
    std::string line, key, val;
    std::getline(stream, line);
    std::istringstream lineStream(line);
    lineStream >> key;
    lineStream >> val;
    _ignoreComments(lineStream, commentStart);
    while (!lineStream.eof()) {
      std::string multiVal;
      lineStream >> multiVal;
      val += (" " + multiVal);
      _ignoreComments(lineStream, commentStart);
    }
    const Value V(val);
    if (key[0] == '-')
      key.erase(0, 1);
    if (key[0] == '+') {
      key.erase(0, 1);
      if (!_existKey(key, container))
        container[key] = V;
      else
        container[key] += V;
    } else if (!_existKey(key, container))
      container[key] = V;
    _ignoreComments(stream, commentStart);
  }
}
void ArgumentParser::readFile(const std::string &filepath) {
  from_files[filepath] = new ArgMap;
  ArgMap &myFMap = *(from_files[filepath]);
  std::ifstream confFile(filepath.c_str());
  if (confFile.good()) {
    _parseFile(confFile, mapArguments);
    confFile.clear();
    confFile.seekg(0, std::ios::beg);
    _parseFile(confFile, myFMap);
  }
  confFile.close();
}
Value &ArgumentParser::operator()(std::string key) {
  _normalizeKey(key);
  const bool bDefaultInCode = !_existKey(key, mapArguments);
  Value &retval = CommandlineParser::operator()(key);
  if (bDefaultInCode)
    from_code[key] = &retval;
  return retval;
}
BufferedLogger logger;
struct BufferedLoggerImpl {
  struct Stream {
    std::stringstream stream;
    int requests_since_last_flush = 0;
    Stream() = default;
    Stream(Stream &&) = default;
    Stream(const Stream &c)
        : requests_since_last_flush(c.requests_since_last_flush) {
      stream << c.stream.rdbuf();
    }
  };
  typedef std::unordered_map<std::string, Stream> container_type;
  container_type files;
  void flush(container_type::value_type &p) {
    std::ofstream savestream;
    savestream.open(p.first, std::ios::app | std::ios::out);
    savestream << p.second.stream.rdbuf();
    savestream.close();
    p.second.requests_since_last_flush = 0;
  }
  std::stringstream &get_stream(const std::string &filename) {
    auto it = files.find(filename);
    if (it != files.end()) {
      if (++it->second.requests_since_last_flush ==
          BufferedLogger::AUTO_FLUSH_COUNT)
        flush(*it);
      return it->second.stream;
    } else {
      auto new_it = files.emplace(filename, Stream()).first;
      return new_it->second.stream;
    }
  }
};
BufferedLogger::BufferedLogger() : impl(new BufferedLoggerImpl) {}
BufferedLogger::~BufferedLogger() {
  flush();
  delete impl;
}
std::stringstream &BufferedLogger::get_stream(const std::string &filename) {
  return impl->get_stream(filename);
}
void BufferedLogger::flush(void) {
  for (auto &pair : impl->files)
    impl->flush(pair);
}
namespace {
class KernelDissipation {
public:
  const Real dt, nu, center[3];
  Real *QOI;
  StencilInfo stencil{-1, -1, -1, 2, 2, 2, false, {0, 1, 2}};
  StencilInfo stencil2{-1, -1, -1, 2, 2, 2, false, {0}};
  SimulationData &sim;
  const std::vector<Info> &chiInfo = sim.chiInfo();
  KernelDissipation(Real _dt, const Real ext[3], Real _nu, Real *RDX,
                    SimulationData &s)
      : dt(_dt), nu(_nu), center{ext[0] / 2, ext[1] / 2, ext[2] / 2}, QOI(RDX),
        sim(s) {}
  void operator()(VectorLab &lab, ScalarLab &pLab, const Info &info,
                  const Info &info2) const {
    const Real h = info.h;
    const Real hCube = std::pow(h, 3), inv2h = .5 / h, invHh = 1 / (h * h);
    const ScalarBlock &chiBlock = *(ScalarBlock *)chiInfo[info.blockID].block;
    for (int iz = 0; iz < VectorBlock::sizeZ; ++iz)
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
          const VectorElement &L = lab(ix, iy, iz);
          const VectorElement &LW = lab(ix - 1, iy, iz),
                              &LE = lab(ix + 1, iy, iz);
          const VectorElement &LS = lab(ix, iy - 1, iz),
                              &LN = lab(ix, iy + 1, iz);
          const VectorElement &LF = lab(ix, iy, iz - 1),
                              &LB = lab(ix, iy, iz + 1);
          const Real X = chiBlock(ix, iy, iz).s;
          Real p[3];
          info.pos(p, ix, iy, iz);
          const Real PX = p[0] - center[0], PY = p[1] - center[1],
                     PZ = p[2] - center[2];
          const Real WX = inv2h * ((LN.u[2] - LS.u[2]) - (LB.u[1] - LF.u[1]));
          const Real WY = inv2h * ((LB.u[0] - LF.u[0]) - (LE.u[2] - LW.u[2]));
          const Real WZ = inv2h * ((LE.u[1] - LW.u[1]) - (LN.u[0] - LS.u[0]));
          const Real dPdx =
              inv2h * (pLab(ix + 1, iy, iz).s - pLab(ix - 1, iy, iz).s);
          const Real dPdy =
              inv2h * (pLab(ix, iy + 1, iz).s - pLab(ix, iy - 1, iz).s);
          const Real dPdz =
              inv2h * (pLab(ix, iy, iz + 1).s - pLab(ix, iy, iz - 1).s);
          const Real lapU = invHh * (LE.u[0] + LW.u[0] + LN.u[0] + LS.u[0] +
                                     LB.u[0] + LF.u[0] - 6 * L.u[0]);
          const Real lapV = invHh * (LE.u[1] + LW.u[1] + LN.u[1] + LS.u[1] +
                                     LB.u[1] + LF.u[1] - 6 * L.u[1]);
          const Real lapW = invHh * (LE.u[2] + LW.u[2] + LN.u[2] + LS.u[2] +
                                     LB.u[2] + LF.u[2] - 6 * L.u[2]);
          const Real V1 = lapU * L.u[0] + lapV * L.u[1] + lapW * L.u[2];
          const Real D11 = inv2h * (LE.u[0] - LW.u[0]);
          const Real D22 = inv2h * (LN.u[1] - LS.u[1]);
          const Real D33 = inv2h * (LB.u[2] - LF.u[2]);
          const Real D12 = inv2h * (LN.u[0] - LS.u[0] + LE.u[1] - LW.u[1]) / 2;
          const Real D13 = inv2h * (LB.u[0] - LF.u[0] + LE.u[2] - LW.u[2]) / 2;
          const Real D23 = inv2h * (LN.u[2] - LS.u[2] + LB.u[1] - LF.u[1]) / 2;
          const Real V2 = D11 * D11 + D22 * D22 + D33 * D33 +
                          2 * (D12 * D12 + D13 * D13 + D23 * D23);
#pragma omp critical
          {
            QOI[0] += hCube * WX;
            QOI[1] += hCube * WY;
            QOI[2] += hCube * WZ;
            QOI[3] += hCube / 2 * (PY * WZ - PZ * WY);
            QOI[4] += hCube / 2 * (PZ * WX - PX * WZ);
            QOI[5] += hCube / 2 * (PX * WY - PY * WX);
            QOI[6] += hCube * L.u[0];
            QOI[7] += hCube * L.u[1];
            QOI[8] += hCube * L.u[2];
            QOI[9] += hCube / 3 *
                      (PX * (PY * WY + PZ * WZ) - WX * (PY * PY + PZ * PZ));
            QOI[10] += hCube / 3 *
                       (PY * (PX * WX + PZ * WZ) - WY * (PX * PX + PZ * PZ));
            QOI[11] += hCube / 3 *
                       (PZ * (PX * WX + PY * WY) - WZ * (PX * PX + PY * PY));
            QOI[12] += hCube * (PY * L.u[2] - PZ * L.u[1]);
            QOI[13] += hCube * (PZ * L.u[0] - PX * L.u[2]);
            QOI[14] += hCube * (PX * L.u[1] - PY * L.u[0]);
            QOI[15] -= (1 - X) * hCube *
                       (dPdx * L.u[0] + dPdy * L.u[1] + dPdz * L.u[2]);
            QOI[16] += (1 - X) * hCube * nu * (V1 + 2 * V2);
            QOI[17] += hCube * (WX * L.u[0] + WY * L.u[1] + WZ * L.u[2]);
            QOI[18] += hCube *
                       (L.u[0] * L.u[0] + L.u[1] * L.u[1] + L.u[2] * L.u[2]) /
                       2;
            QOI[19] += hCube * std::sqrt(WX * WX + WY * WY + WZ * WZ);
          }
        }
  }
};
} // namespace
void ComputeDissipation::operator()(const Real dt) {
  if (sim.freqDiagnostics == 0 || sim.step % sim.freqDiagnostics)
    return;
  Real RDX[20] = {0.0};
  KernelDissipation diss(dt, sim.extents.data(), sim.nu, RDX, sim);
  compute<KernelDissipation, VectorGrid, VectorLab, ScalarGrid, ScalarLab>(
      diss, *sim.vel, *sim.pres);
  MPI_Allreduce(MPI_IN_PLACE, RDX, 20, MPI_Real, MPI_SUM, sim.comm);
  size_t loc = sim.velInfo().size();
  size_t tot;
  MPI_Reduce(&loc, &tot, 1, MPI_LONG, MPI_SUM, 0, sim.comm);
}
namespace diffusion_kernels {
static constexpr Real kDivEpsilon = 1e-55;
static constexpr Real kNormRelCriterion = 1e-7;
static constexpr Real kNormAbsCriterion = 1e-16;
static constexpr Real kSqrNormRelCriterion =
    kNormRelCriterion * kNormRelCriterion;
static constexpr Real kSqrNormAbsCriterion =
    kNormAbsCriterion * kNormAbsCriterion;
static inline Real subAndSumSqr(Block &__restrict__ r_,
                                const Block &__restrict__ Ax_, Real a) {
  constexpr int MX = 16;
  constexpr int MY = NX * NY * NZ / MX;
  using SquashedBlock = Real[MY][MX];
  static_assert(NX * NY % MX == 0 && sizeof(Block) == sizeof(SquashedBlock));
  SquashedBlock &__restrict__ r = (SquashedBlock &)r_;
  SquashedBlock &__restrict__ Ax = (SquashedBlock &)Ax_;
  Real s[MX] = {};
  for (int jy = 0; jy < MY; ++jy) {
    for (int jx = 0; jx < MX; ++jx)
      r[jy][jx] -= a * Ax[jy][jx];
    for (int jx = 0; jx < MX; ++jx)
      s[jx] += r[jy][jx] * r[jy][jx];
  }
  return sum(s);
}
template <typename T>
static inline T *assumeAligned(T *ptr, unsigned align, unsigned offset = 0) {
  if (sizeof(Real) == 8 || sizeof(Real) == 4) {
    assert((uintptr_t)ptr % align == offset);
    return (T *)__builtin_assume_aligned(ptr, align, offset);
  } else {
    return ptr;
  }
}
Real kernelDiffusionGetZInner(PaddedBlock &p_, const Real *pW_, const Real *pE_,
                              Block &__restrict__ Ax_, Block &__restrict__ r_,
                              Block &__restrict__ block_, const Real sqrNorm0,
                              const Real rr, const Real coefficient) {
  PaddedBlock &p = *assumeAligned(&p_, 64, 64 - xPad * sizeof(Real));
  const PaddedBlock &pW = *(PaddedBlock *)pW_;
  const PaddedBlock &pE = *(PaddedBlock *)pE_;
  Block &__restrict__ Ax = *assumeAligned(&Ax_, 64);
  Block &__restrict__ r = *assumeAligned(&r_, 64);
  Block &__restrict__ block = *assumeAligned(&block_, kBlockAlignment);
  Real a2Partial[NX] = {};
  for (int iz = 0; iz < NZ; ++iz)
    for (int iy = 0; iy < NY; ++iy) {
      Real tmpAx[NX];
      for (int ix = 0; ix < NX; ++ix) {
        tmpAx[ix] = pW[iz + 1][iy + 1][ix + xPad] +
                    pE[iz + 1][iy + 1][ix + xPad] +
                    coefficient * p[iz + 1][iy + 1][ix + xPad];
      }
      for (int ix = 0; ix < NX; ++ix)
        tmpAx[ix] += p[iz + 1][iy][ix + xPad];
      for (int ix = 0; ix < NX; ++ix)
        tmpAx[ix] += p[iz + 1][iy + 2][ix + xPad];
      for (int ix = 0; ix < NX; ++ix)
        tmpAx[ix] += p[iz][iy + 1][ix + xPad];
      for (int ix = 0; ix < NX; ++ix)
        tmpAx[ix] += p[iz + 2][iy + 1][ix + xPad];
      for (int ix = 0; ix < NX; ++ix)
        Ax[iz][iy][ix] = tmpAx[ix];
      for (int ix = 0; ix < NX; ++ix)
        a2Partial[ix] += p[iz + 1][iy + 1][ix + xPad] * tmpAx[ix];
    }
  const Real a2 = sum(a2Partial);
  const Real a = rr / (a2 + kDivEpsilon);
  for (int iz = 0; iz < NZ; ++iz)
    for (int iy = 0; iy < NY; ++iy)
      for (int ix = 0; ix < NX; ++ix)
        block[iz][iy][ix] += a * p[iz + 1][iy + 1][ix + xPad];
  const Real sqrSum = subAndSumSqr(r, Ax, a);
  const Real beta = sqrSum / (rr + kDivEpsilon);
  const Real sqrNorm = (Real)1 / (N * N) * sqrSum;
  if (sqrNorm < kSqrNormRelCriterion * sqrNorm0 ||
      sqrNorm < kSqrNormAbsCriterion)
    return -1.0;
  for (int iz = 0; iz < NZ; ++iz)
    for (int iy = 0; iy < NY; ++iy)
      for (int ix = 0; ix < NX; ++ix) {
        p[iz + 1][iy + 1][ix + xPad] =
            r[iz][iy][ix] + beta * p[iz + 1][iy + 1][ix + xPad];
      }
  const Real rrNew = sqrSum;
  return rrNew;
}
void getZImplParallel(const std::vector<Info> &vInfo, const Real nu,
                      const Real dt) {
  const size_t Nblocks = vInfo.size();
  struct Tmp {
    Block r;
    char padding1[64 - xPad * sizeof(Real)];
    PaddedBlock p;
    char padding2[xPad * sizeof(Real)];
    Block Ax;
  };
  alignas(64) Tmp tmp{};
  Block &r = tmp.r;
  Block &Ax = tmp.Ax;
  PaddedBlock &p = tmp.p;
#pragma omp for
  for (size_t i = 0; i < Nblocks; ++i) {
    static_assert(sizeof(ScalarBlock) == sizeof(Block));
    assert((uintptr_t)vInfo[i].block % kBlockAlignment == 0);
    Block &block =
        *(Block *)__builtin_assume_aligned(vInfo[i].block, kBlockAlignment);
    const Real invh = 1 / vInfo[i].h;
    Real rrPartial[NX] = {};
    for (int iz = 0; iz < NZ; ++iz)
      for (int iy = 0; iy < NY; ++iy)
        for (int ix = 0; ix < NX; ++ix) {
          r[iz][iy][ix] = invh * block[iz][iy][ix];
          rrPartial[ix] += r[iz][iy][ix] * r[iz][iy][ix];
          p[iz + 1][iy + 1][ix + xPad] = r[iz][iy][ix];
          block[iz][iy][ix] = 0;
        }
    Real rr = sum(rrPartial);
    const Real sqrNorm0 = (Real)1 / (N * N) * rr;
    if (sqrNorm0 < 1e-32)
      continue;
    const Real *pW = &p[0][0][0] - 1;
    const Real *pE = &p[0][0][0] + 1;
    const Real coefficient = -6.0 - vInfo[i].h * vInfo[i].h / nu / dt;
    for (int k = 0; k < 100; ++k) {
      rr = kernelDiffusionGetZInner(p, pW, pE, Ax, r, block, sqrNorm0, rr,
                                    coefficient);
      if (rr <= 0)
        break;
    }
  }
}
} // namespace diffusion_kernels
void ExternalForcing::operator()(const double dt) {
  const int dir = sim.BCy_flag == wall ? 1 : 2;
  const Real H = sim.extents[dir];
  const Real gradPdt = 8 * sim.uMax_forced * sim.nu / H / H * dt;
  const int DIRECTION = 0;
  const std::vector<Info> &velInfo = sim.velInfo();
#pragma omp parallel for
  for (size_t i = 0; i < velInfo.size(); i++) {
    VectorBlock &v = *(VectorBlock *)velInfo[i].block;
    for (int z = 0; z < VectorBlock::sizeZ; ++z)
      for (int y = 0; y < VectorBlock::sizeY; ++y)
        for (int x = 0; x < VectorBlock::sizeX; ++x) {
          v(x, y, z).u[DIRECTION] += gradPdt;
        }
  }
}
Fish::Fish(SimulationData &s, ArgumentParser &p) : Obstacle(s, p) {
  p.unset_strict_mode();
  int array_of_blocklengths[2] = {4, 1};
  MPI_Aint array_of_displacements[2] = {0, 4 * sizeof(Real)};
  MPI_Datatype array_of_types[2] = {MPI_Real, MPI_LONG};
  MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements,
                         array_of_types, &MPI_BLOCKID);
  MPI_Type_commit(&MPI_BLOCKID);
  const int Z = ScalarBlock::sizeZ;
  const int Y = ScalarBlock::sizeY;
  const int X = ScalarBlock::sizeX;
  int array_of_blocklengths1[2] = {Z * Y * X * 3 + (Z + 2) * (Y + 2) * (X + 2),
                                   Z * Y * X};
  MPI_Aint array_of_displacements1[2] = {
      0, (Z * Y * X * 3 + (Z + 2) * (Y + 2) * (X + 2)) * sizeof(Real)};
  MPI_Datatype array_of_types1[2] = {MPI_Real, MPI_INT};
  MPI_Type_create_struct(2, array_of_blocklengths1, array_of_displacements1,
                         array_of_types1, &MPI_OBSTACLE);
  MPI_Type_commit(&MPI_OBSTACLE);
}
Fish::~Fish() {
  if (myFish not_eq nullptr)
    delete myFish;
  MPI_Type_free(&MPI_BLOCKID);
  MPI_Type_free(&MPI_OBSTACLE);
}
void Fish::integrateMidline() {
  myFish->integrateLinearMomentum();
  myFish->integrateAngularMomentum(sim.dt);
}
std::vector<VolumeSegment_OBB> Fish::prepare_vSegments() {
  const int Nsegments = std::ceil((myFish->Nm - 1.) / 8);
  const int Nm = myFish->Nm;
  assert((Nm - 1) % Nsegments == 0);
  std::vector<VolumeSegment_OBB> vSegments(Nsegments);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < Nsegments; ++i) {
    const int nextidx = (i + 1) * (Nm - 1) / Nsegments;
    const int idx = i * (Nm - 1) / Nsegments;
    Real bbox[3][2] = {{1e9, -1e9}, {1e9, -1e9}, {1e9, -1e9}};
    for (int ss = idx; ss <= nextidx; ++ss) {
      const Real xBnd[4] = {
          myFish->rX[ss] + myFish->norX[ss] * myFish->width[ss],
          myFish->rX[ss] - myFish->norX[ss] * myFish->width[ss],
          myFish->rX[ss] + myFish->binX[ss] * myFish->height[ss],
          myFish->rX[ss] - myFish->binX[ss] * myFish->height[ss]};
      const Real yBnd[4] = {
          myFish->rY[ss] + myFish->norY[ss] * myFish->width[ss],
          myFish->rY[ss] - myFish->norY[ss] * myFish->width[ss],
          myFish->rY[ss] + myFish->binY[ss] * myFish->height[ss],
          myFish->rY[ss] - myFish->binY[ss] * myFish->height[ss]};
      const Real zBnd[4] = {
          myFish->rZ[ss] + myFish->norZ[ss] * myFish->width[ss],
          myFish->rZ[ss] - myFish->norZ[ss] * myFish->width[ss],
          myFish->rZ[ss] + myFish->binZ[ss] * myFish->height[ss],
          myFish->rZ[ss] - myFish->binZ[ss] * myFish->height[ss]};
      const Real maxX = std::max({xBnd[0], xBnd[1], xBnd[2], xBnd[3]});
      const Real maxY = std::max({yBnd[0], yBnd[1], yBnd[2], yBnd[3]});
      const Real maxZ = std::max({zBnd[0], zBnd[1], zBnd[2], zBnd[3]});
      const Real minX = std::min({xBnd[0], xBnd[1], xBnd[2], xBnd[3]});
      const Real minY = std::min({yBnd[0], yBnd[1], yBnd[2], yBnd[3]});
      const Real minZ = std::min({zBnd[0], zBnd[1], zBnd[2], zBnd[3]});
      bbox[0][0] = std::min(bbox[0][0], minX);
      bbox[0][1] = std::max(bbox[0][1], maxX);
      bbox[1][0] = std::min(bbox[1][0], minY);
      bbox[1][1] = std::max(bbox[1][1], maxY);
      bbox[2][0] = std::min(bbox[2][0], minZ);
      bbox[2][1] = std::max(bbox[2][1], maxZ);
    }
    vSegments[i].prepare(std::make_pair(idx, nextidx), bbox, sim.hmin);
    vSegments[i].changeToComputationalFrame(position, quaternion);
  }
  return vSegments;
}
using intersect_t = std::vector<std::vector<VolumeSegment_OBB *>>;
intersect_t Fish::prepare_segPerBlock(vecsegm_t &vSegments) {
  MyBlockIDs.clear();
  for (size_t j = 0; j < MySegments.size(); j++)
    MySegments[j].clear();
  MySegments.clear();
  const std::vector<Info> &chiInfo = sim.chiInfo();
  std::vector<std::vector<VolumeSegment_OBB *>> ret(chiInfo.size());
  for (auto &entry : obstacleBlocks) {
    if (entry == nullptr)
      continue;
    delete entry;
    entry = nullptr;
  }
  obstacleBlocks.resize(chiInfo.size(), nullptr);
  for (size_t i = 0; i < chiInfo.size(); ++i) {
    const Info &info = chiInfo[i];
    Real MINP[3], MAXP[3];
    info.pos(MINP, 0, 0, 0);
    info.pos(MAXP, ScalarBlock::sizeX - 1, ScalarBlock::sizeY - 1,
             ScalarBlock::sizeZ - 1);
    bool hasSegments = false;
    for (size_t s = 0; s < vSegments.size(); ++s)
      if (vSegments[s].isIntersectingWithAABB(MINP, MAXP)) {
        VolumeSegment_OBB *const ptr = &vSegments[s];
        ret[info.blockID].push_back(ptr);
        {
          if (!hasSegments) {
            hasSegments = true;
            MyBlockIDs.push_back({(Real)info.h, (Real)info.origin[0],
                                  (Real)info.origin[1], (Real)info.origin[2],
                                  info.blockID});
            MySegments.resize(MySegments.size() + 1);
          }
          MySegments.back().push_back(s);
        }
      }
    if (ret[info.blockID].size() > 0) {
      assert(obstacleBlocks[info.blockID] == nullptr);
      ObstacleBlock *const block = new ObstacleBlock();
      assert(block not_eq nullptr);
      obstacleBlocks[info.blockID] = block;
      block->clear();
    }
  }
  return ret;
}
void Fish::writeSDFOnBlocks(std::vector<VolumeSegment_OBB> &vSegments) {
#if 1
#pragma omp parallel
  {
    PutFishOnBlocks putfish(myFish, position, quaternion);
#pragma omp for
    for (size_t j = 0; j < MyBlockIDs.size(); j++) {
      std::vector<VolumeSegment_OBB *> S;
      for (size_t k = 0; k < MySegments[j].size(); k++)
        S.push_back(&vSegments[MySegments[j][k]]);
      ObstacleBlock *const block = obstacleBlocks[MyBlockIDs[j].blockID];
      putfish(MyBlockIDs[j].h, MyBlockIDs[j].origin_x, MyBlockIDs[j].origin_y,
              MyBlockIDs[j].origin_z, block, S);
    }
  }
#else
  const int tag = 34;
  MPI_Comm comm = sim.chi->getWorldComm();
  const int rank = sim.chi->rank();
  const int size = sim.chi->get_world_size();
  std::vector<std::vector<int>> OtherSegments;
  int b = (int)MyBlockIDs.size();
  std::vector<int> all_b(size);
  MPI_Allgather(&b, 1, MPI_INT, all_b.data(), 1, MPI_INT, comm);
  int total_load = 0;
  for (int r = 0; r < size; r++)
    total_load += all_b[r];
  int my_load = total_load / size;
  if (rank < (total_load % size))
    my_load += 1;
  std::vector<int> index_start(size);
  index_start[0] = 0;
  for (int r = 1; r < size; r++)
    index_start[r] = index_start[r - 1] + all_b[r - 1];
  int ideal_index = (total_load / size) * rank;
  ideal_index += (rank < (total_load % size)) ? rank : (total_load % size);
  std::vector<std::vector<BlockID>> send_blocks(size);
  std::vector<std::vector<BlockID>> recv_blocks(size);
  for (int r = 0; r < size; r++)
    if (rank != r) {
      {
        const int a1 = ideal_index;
        const int a2 = ideal_index + my_load - 1;
        const int b1 = index_start[r];
        const int b2 = index_start[r] + all_b[r] - 1;
        const int c1 = max(a1, b1);
        const int c2 = min(a2, b2);
        if (c2 - c1 + 1 > 0)
          recv_blocks[r].resize(c2 - c1 + 1);
      }
      {
        int other_ideal_index = (total_load / size) * r;
        other_ideal_index +=
            (r < (total_load % size)) ? r : (total_load % size);
        int other_load = total_load / size;
        if (r < (total_load % size))
          other_load += 1;
        const int a1 = other_ideal_index;
        const int a2 = other_ideal_index + other_load - 1;
        const int b1 = index_start[rank];
        const int b2 = index_start[rank] + all_b[rank] - 1;
        const int c1 = max(a1, b1);
        const int c2 = min(a2, b2);
        if (c2 - c1 + 1 > 0)
          send_blocks[r].resize(c2 - c1 + 1);
      }
    }
  std::vector<MPI_Request> recv_request;
  for (int r = 0; r < size; r++)
    if (recv_blocks[r].size() != 0) {
      MPI_Request req;
      recv_request.push_back(req);
      MPI_Irecv(recv_blocks[r].data(), recv_blocks[r].size(), MPI_BLOCKID, r,
                tag, comm, &recv_request.back());
    }
  std::vector<MPI_Request> send_request;
  int counter = 0;
  for (int r = 0; r < size; r++)
    if (send_blocks[r].size() != 0) {
      for (size_t i = 0; i < send_blocks[r].size(); i++) {
        send_blocks[r][i].h = MyBlockIDs[counter + i].h;
        send_blocks[r][i].origin_x = MyBlockIDs[counter + i].origin_x;
        send_blocks[r][i].origin_y = MyBlockIDs[counter + i].origin_y;
        send_blocks[r][i].origin_z = MyBlockIDs[counter + i].origin_z;
        send_blocks[r][i].blockID = MyBlockIDs[counter + i].blockID;
      }
      counter += send_blocks[r].size();
      MPI_Request req;
      send_request.push_back(req);
      MPI_Isend(send_blocks[r].data(), send_blocks[r].size(), MPI_BLOCKID, r,
                tag, comm, &send_request.back());
    }
  const int sizeZ = ScalarBlock::sizeZ;
  const int sizeY = ScalarBlock::sizeY;
  const int sizeX = ScalarBlock::sizeX;
  std::vector<std::vector<MPI_Obstacle>> send_obstacles(size);
  std::vector<std::vector<MPI_Obstacle>> recv_obstacles(size);
  for (int r = 0; r < size; r++) {
    send_obstacles[r].resize(send_blocks[r].size());
    recv_obstacles[r].resize(recv_blocks[r].size());
  }
  MPI_Waitall(send_request.size(), send_request.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(recv_request.size(), recv_request.data(), MPI_STATUSES_IGNORE);
  for (int r = 0; r < size; r++)
    if (recv_blocks[r].size() != 0) {
      for (size_t j = 0; j < OtherSegments.size(); j++)
        OtherSegments[j].clear();
      OtherSegments.clear();
      for (size_t i = 0; i < recv_blocks[r].size(); ++i) {
        const auto &info = recv_blocks[r][i];
        bool hasSegments = false;
        for (size_t s = 0; s < vSegments.size(); ++s) {
          Real min_pos[3] = {info.origin_x + 0.5 * info.h,
                             info.origin_y + 0.5 * info.h,
                             info.origin_z + 0.5 * info.h};
          Real max_pos[3] = {
              info.origin_x + (0.5 + ScalarBlock::sizeX - 1) * info.h,
              info.origin_y + (0.5 + ScalarBlock::sizeY - 1) * info.h,
              info.origin_z + (0.5 + ScalarBlock::sizeZ - 1) * info.h};
          if (vSegments[s].isIntersectingWithAABB(min_pos, max_pos)) {
            if (!hasSegments) {
              hasSegments = true;
              OtherSegments.resize(OtherSegments.size() + 1);
            }
            OtherSegments.back().push_back(s);
          }
        }
      }
#pragma omp parallel
      {
        PutFishOnBlocks putfish(myFish, position, quaternion);
#pragma omp for
        for (size_t j = 0; j < recv_blocks[r].size(); j++) {
          std::vector<VolumeSegment_OBB *> S;
          for (size_t k = 0; k < OtherSegments[j].size(); k++) {
            VolumeSegment_OBB *const ptr = &vSegments[OtherSegments[j][k]];
            S.push_back(ptr);
          }
          if (S.size() > 0) {
            ObstacleBlock block;
            block.clear();
            putfish(recv_blocks[r][j].h, recv_blocks[r][j].origin_x,
                    recv_blocks[r][j].origin_y, recv_blocks[r][j].origin_z,
                    &block, S);
            int kounter = 0;
            for (int iz = 0; iz < sizeZ; iz++)
              for (int iy = 0; iy < sizeY; iy++)
                for (int ix = 0; ix < sizeX; ix++) {
                  recv_obstacles[r][j].d[kounter] = block.udef[iz][iy][ix][0];
                  recv_obstacles[r][j].d[sizeZ * sizeY * sizeX + kounter] =
                      block.udef[iz][iy][ix][1];
                  recv_obstacles[r][j].d[sizeZ * sizeY * sizeX * 2 + kounter] =
                      block.udef[iz][iy][ix][2];
                  kounter++;
                }
            kounter = 0;
            for (int iz = 0; iz < sizeZ + 2; iz++)
              for (int iy = 0; iy < sizeY + 2; iy++)
                for (int ix = 0; ix < sizeX + 2; ix++) {
                  recv_obstacles[r][j].d[sizeZ * sizeY * sizeX * 3 + kounter] =
                      block.sdfLab[iz][iy][ix];
                  kounter++;
                }
          }
        }
      }
    }
  std::vector<MPI_Request> recv_request_obs;
  for (int r = 0; r < size; r++)
    if (send_obstacles[r].size() != 0) {
      MPI_Request req;
      recv_request_obs.push_back(req);
      MPI_Irecv(send_obstacles[r].data(), send_obstacles[r].size(),
                MPI_OBSTACLE, r, tag, comm, &recv_request_obs.back());
    }
  std::vector<MPI_Request> send_request_obs;
  for (int r = 0; r < size; r++)
    if (recv_obstacles[r].size() != 0) {
      MPI_Request req;
      send_request_obs.push_back(req);
      MPI_Isend(recv_obstacles[r].data(), recv_obstacles[r].size(),
                MPI_OBSTACLE, r, tag, comm, &send_request_obs.back());
    }
#pragma omp parallel
  {
    PutFishOnBlocks putfish(myFish, position, quaternion);
#pragma omp for
    for (size_t j = counter; j < MyBlockIDs.size(); ++j) {
      std::vector<VolumeSegment_OBB *> S;
      for (size_t k = 0; k < MySegments[j].size(); k++) {
        VolumeSegment_OBB *const ptr = &vSegments[MySegments[j][k]];
        S.push_back(ptr);
      }
      if (S.size() > 0) {
        ObstacleBlock *const block = obstacleBlocks[MyBlockIDs[j].blockID];
        putfish(MyBlockIDs[j].h, MyBlockIDs[j].origin_x, MyBlockIDs[j].origin_y,
                MyBlockIDs[j].origin_z, block, S);
      }
    }
  }
  MPI_Waitall(send_request_obs.size(), send_request_obs.data(),
              MPI_STATUSES_IGNORE);
  MPI_Waitall(recv_request_obs.size(), recv_request_obs.data(),
              MPI_STATUSES_IGNORE);
  counter = 0;
  for (int r = 0; r < size; r++)
    if (send_obstacles[r].size() != 0) {
      for (size_t i = 0; i < send_blocks[r].size(); i++) {
        ObstacleBlock *const block =
            obstacleBlocks[MyBlockIDs[counter + i].blockID];
        int kounter = 0;
        for (int iz = 0; iz < sizeZ; iz++)
          for (int iy = 0; iy < sizeY; iy++)
            for (int ix = 0; ix < sizeX; ix++) {
              block->udef[iz][iy][ix][0] = send_obstacles[r][i].d[kounter];
              block->udef[iz][iy][ix][1] =
                  send_obstacles[r][i].d[sizeZ * sizeY * sizeX + kounter];
              block->udef[iz][iy][ix][2] =
                  send_obstacles[r][i].d[sizeZ * sizeY * sizeX * 2 + kounter];
              kounter++;
            }
        kounter = 0;
        for (int iz = 0; iz < sizeZ + 2; iz++)
          for (int iy = 0; iy < sizeY + 2; iy++)
            for (int ix = 0; ix < sizeX + 2; ix++) {
              block->sdfLab[iz][iy][ix] =
                  send_obstacles[r][i].d[sizeZ * sizeY * sizeX * 3 + kounter];
              kounter++;
            }
      }
      counter += send_blocks[r].size();
    }
#endif
}
void Fish::create() {
  myFish->computeMidline(sim.time, sim.dt);
  integrateMidline();
  std::vector<VolumeSegment_OBB> vSegments = prepare_vSegments();
  const intersect_t segmPerBlock = prepare_segPerBlock(vSegments);
  writeSDFOnBlocks(vSegments);
}
using UDEFMAT = Real[_BS_][_BS_][_BS_][3];
using CHIMAT = Real[_BS_][_BS_][_BS_];
void FishMidlineData::integrateLinearMomentum() {
  Real V = 0, cmx = 0, cmy = 0, cmz = 0, lmx = 0, lmy = 0, lmz = 0;
#pragma omp parallel for schedule(static) reduction(+:V,cmx,cmy,cmz,lmx,lmy,lmz)
  for (int i = 0; i < Nm; ++i) {
    const Real ds = 0.5 * ((i == 0) ? rS[1] - rS[0]
                                    : ((i == Nm - 1) ? rS[Nm - 1] - rS[Nm - 2]
                                                     : rS[i + 1] - rS[i - 1]));
    const Real c0 = norY[i] * binZ[i] - norZ[i] * binY[i];
    const Real c1 = norZ[i] * binX[i] - norX[i] * binZ[i];
    const Real c2 = norX[i] * binY[i] - norY[i] * binX[i];
    const Real x0dot = _d_ds(i, rX, Nm);
    const Real x1dot = _d_ds(i, rY, Nm);
    const Real x2dot = _d_ds(i, rZ, Nm);
    const Real n0dot = _d_ds(i, norX, Nm);
    const Real n1dot = _d_ds(i, norY, Nm);
    const Real n2dot = _d_ds(i, norZ, Nm);
    const Real b0dot = _d_ds(i, binX, Nm);
    const Real b1dot = _d_ds(i, binY, Nm);
    const Real b2dot = _d_ds(i, binZ, Nm);
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
void FishMidlineData::integrateAngularMomentum(const Real dt) {
  Real JXX = 0;
  Real JYY = 0;
  Real JZZ = 0;
  Real JXY = 0;
  Real JYZ = 0;
  Real JZX = 0;
  Real AM_X = 0;
  Real AM_Y = 0;
  Real AM_Z = 0;
#pragma omp parallel for reduction (+:JXX,JYY,JZZ,JXY,JYZ,JZX,AM_X,AM_Y,AM_Z)
  for (int i = 0; i < Nm; ++i) {
    const Real ds = 0.5 * ((i == 0) ? rS[1] - rS[0]
                                    : ((i == Nm - 1) ? rS[Nm - 1] - rS[Nm - 2]
                                                     : rS[i + 1] - rS[i - 1]));
    const Real c0 = norY[i] * binZ[i] - norZ[i] * binY[i];
    const Real c1 = norZ[i] * binX[i] - norX[i] * binZ[i];
    const Real c2 = norX[i] * binY[i] - norY[i] * binX[i];
    const Real x0dot = _d_ds(i, rX, Nm);
    const Real x1dot = _d_ds(i, rY, Nm);
    const Real x2dot = _d_ds(i, rZ, Nm);
    const Real n0dot = _d_ds(i, norX, Nm);
    const Real n1dot = _d_ds(i, norY, Nm);
    const Real n2dot = _d_ds(i, norZ, Nm);
    const Real b0dot = _d_ds(i, binX, Nm);
    const Real b1dot = _d_ds(i, binY, Nm);
    const Real b2dot = _d_ds(i, binZ, Nm);
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
                      cN * M11 * (rX[i] * vNorY[i] + rY[i] * norX[i]) +
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
  const Real eps = std::numeric_limits<Real>::epsilon();
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
      1.0 / std::sqrt(quaternion_internal[0] * quaternion_internal[0] +
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
void VolumeSegment_OBB::prepare(std::pair<int, int> _s_range,
                                const Real bbox[3][2], const Real h) {
  safe_distance = (SURFDH + 2) * h;
  s_range.first = _s_range.first;
  s_range.second = _s_range.second;
  for (int i = 0; i < 3; ++i) {
    w[i] = (bbox[i][1] - bbox[i][0]) / 2 + safe_distance;
    c[i] = (bbox[i][1] + bbox[i][0]) / 2;
    assert(w[i] > 0);
  }
}
void VolumeSegment_OBB::normalizeNormals() {
  const Real magI =
      std::sqrt(normalI[0] * normalI[0] + normalI[1] * normalI[1] +
                normalI[2] * normalI[2]);
  const Real magJ =
      std::sqrt(normalJ[0] * normalJ[0] + normalJ[1] * normalJ[1] +
                normalJ[2] * normalJ[2]);
  const Real magK =
      std::sqrt(normalK[0] * normalK[0] + normalK[1] * normalK[1] +
                normalK[2] * normalK[2]);
  assert(magI > std::numeric_limits<Real>::epsilon());
  assert(magJ > std::numeric_limits<Real>::epsilon());
  assert(magK > std::numeric_limits<Real>::epsilon());
  const Real invMagI = Real(1) / magI;
  const Real invMagJ = Real(1) / magJ;
  const Real invMagK = Real(1) / magK;
  for (int i = 0; i < 3; ++i) {
    normalI[i] = std::fabs(normalI[i]) * invMagI;
    normalJ[i] = std::fabs(normalJ[i]) * invMagJ;
    normalK[i] = std::fabs(normalK[i]) * invMagK;
  }
}
void VolumeSegment_OBB::changeToComputationalFrame(const Real position[3],
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
  const Real p[3] = {c[0], c[1], c[2]};
  const Real nx[3] = {normalI[0], normalI[1], normalI[2]};
  const Real ny[3] = {normalJ[0], normalJ[1], normalJ[2]};
  const Real nz[3] = {normalK[0], normalK[1], normalK[2]};
  for (int i = 0; i < 3; ++i) {
    c[i] = Rmatrix[i][0] * p[0] + Rmatrix[i][1] * p[1] + Rmatrix[i][2] * p[2];
    normalI[i] =
        Rmatrix[i][0] * nx[0] + Rmatrix[i][1] * nx[1] + Rmatrix[i][2] * nx[2];
    normalJ[i] =
        Rmatrix[i][0] * ny[0] + Rmatrix[i][1] * ny[1] + Rmatrix[i][2] * ny[2];
    normalK[i] =
        Rmatrix[i][0] * nz[0] + Rmatrix[i][1] * nz[1] + Rmatrix[i][2] * nz[2];
  }
  c[0] += position[0];
  c[1] += position[1];
  c[2] += position[2];
  normalizeNormals();
  assert(normalI[0] >= 0 && normalI[1] >= 0 && normalI[2] >= 0);
  assert(normalJ[0] >= 0 && normalJ[1] >= 0 && normalJ[2] >= 0);
  assert(normalK[0] >= 0 && normalK[1] >= 0 && normalK[2] >= 0);
  const Real widthXvec[] = {w[0] * normalI[0], w[0] * normalI[1],
                            w[0] * normalI[2]};
  const Real widthYvec[] = {w[1] * normalJ[0], w[1] * normalJ[1],
                            w[1] * normalJ[2]};
  const Real widthZvec[] = {w[2] * normalK[0], w[2] * normalK[1],
                            w[2] * normalK[2]};
  for (int i = 0; i < 3; ++i) {
    objBoxLabFr[i][0] = c[i] - widthXvec[i] - widthYvec[i] - widthZvec[i];
    objBoxLabFr[i][1] = c[i] + widthXvec[i] + widthYvec[i] + widthZvec[i];
    objBoxObjFr[i][0] = c[i] - w[i];
    objBoxObjFr[i][1] = c[i] + w[i];
  }
}
#define DBLCHECK
bool VolumeSegment_OBB::isIntersectingWithAABB(const Real start[3],
                                               const Real end[3]) const {
  const Real AABB_w[3] = {(end[0] - start[0]) / 2 + safe_distance,
                          (end[1] - start[1]) / 2 + safe_distance,
                          (end[2] - start[2]) / 2 + safe_distance};
  const Real AABB_c[3] = {(end[0] + start[0]) / 2, (end[1] + start[1]) / 2,
                          (end[2] + start[2]) / 2};
  const Real AABB_box[3][2] = {{AABB_c[0] - AABB_w[0], AABB_c[0] + AABB_w[0]},
                               {AABB_c[1] - AABB_w[1], AABB_c[1] + AABB_w[1]},
                               {AABB_c[2] - AABB_w[2], AABB_c[2] + AABB_w[2]}};
  assert(AABB_w[0] > 0 && AABB_w[1] > 0 && AABB_w[2] > 0);
  using std::max;
  using std::min;
  Real intersectionLabFrame[3][2] = {{max(objBoxLabFr[0][0], AABB_box[0][0]),
                                      min(objBoxLabFr[0][1], AABB_box[0][1])},
                                     {max(objBoxLabFr[1][0], AABB_box[1][0]),
                                      min(objBoxLabFr[1][1], AABB_box[1][1])},
                                     {max(objBoxLabFr[2][0], AABB_box[2][0]),
                                      min(objBoxLabFr[2][1], AABB_box[2][1])}};
  if (intersectionLabFrame[0][1] - intersectionLabFrame[0][0] < 0 ||
      intersectionLabFrame[1][1] - intersectionLabFrame[1][0] < 0 ||
      intersectionLabFrame[2][1] - intersectionLabFrame[2][0] < 0)
    return false;
#ifdef DBLCHECK
  const Real widthXbox[3] = {AABB_w[0] * normalI[0], AABB_w[0] * normalJ[0],
                             AABB_w[0] * normalK[0]};
  const Real widthYbox[3] = {AABB_w[1] * normalI[1], AABB_w[1] * normalJ[1],
                             AABB_w[1] * normalK[1]};
  const Real widthZbox[3] = {AABB_w[2] * normalI[2], AABB_w[2] * normalJ[2],
                             AABB_w[2] * normalK[2]};
  const Real boxBox[3][2] = {
      {AABB_c[0] - widthXbox[0] - widthYbox[0] - widthZbox[0],
       AABB_c[0] + widthXbox[0] + widthYbox[0] + widthZbox[0]},
      {AABB_c[1] - widthXbox[1] - widthYbox[1] - widthZbox[1],
       AABB_c[1] + widthXbox[1] + widthYbox[1] + widthZbox[1]},
      {AABB_c[2] - widthXbox[2] - widthYbox[2] - widthZbox[2],
       AABB_c[2] + widthXbox[2] + widthYbox[2] + widthZbox[2]}};
  Real intersectionFishFrame[3][2] = {{max(boxBox[0][0], objBoxObjFr[0][0]),
                                       min(boxBox[0][1], objBoxObjFr[0][1])},
                                      {max(boxBox[1][0], objBoxObjFr[1][0]),
                                       min(boxBox[1][1], objBoxObjFr[1][1])},
                                      {max(boxBox[2][0], objBoxObjFr[2][0]),
                                       min(boxBox[2][1], objBoxObjFr[2][1])}};
  if (intersectionFishFrame[0][1] - intersectionFishFrame[0][0] < 0 ||
      intersectionFishFrame[1][1] - intersectionFishFrame[1][0] < 0 ||
      intersectionFishFrame[2][1] - intersectionFishFrame[2][0] < 0)
    return false;
#endif
  return true;
}
void PutFishOnBlocks::operator()(
    const Real h, const Real ox, const Real oy, const Real oz,
    ObstacleBlock *const oblock,
    const std::vector<VolumeSegment_OBB *> &vSegments) const {
  const int nz = ScalarBlock::sizeZ;
  const int ny = ScalarBlock::sizeY;
  const int nx = ScalarBlock::sizeX;
  Real *const sdf = &oblock->sdfLab[0][0][0];
  auto &chi = oblock->chi;
  auto &udef = oblock->udef;
  memset(chi, 0, sizeof(Real) * nx * ny * nz);
  memset(udef, 0, sizeof(Real) * nx * ny * nz * 3);
  std::fill(sdf, sdf + (nz + 2) * (ny + 2) * (nx + 2), -1.);
  constructInternl(h, ox, oy, oz, oblock, vSegments);
  constructSurface(h, ox, oy, oz, oblock, vSegments);
  signedDistanceSqrt(oblock);
}
inline Real distPlane(const Real p1[3], const Real p2[3], const Real p3[3],
                      const Real s[3], const Real IN[3]) {
  const Real t[3] = {s[0] - p1[0], s[1] - p1[1], s[2] - p1[2]};
  const Real u[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
  const Real v[3] = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};
  const Real i[3] = {IN[0] - p1[0], IN[1] - p1[1], IN[2] - p1[2]};
  const Real n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                     u[0] * v[1] - u[1] * v[0]};
  const Real projInner = i[0] * n[0] + i[1] * n[1] + i[2] * n[2];
  const Real signIn = projInner > 0 ? 1 : -1;
  const Real norm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  return signIn * (t[0] * n[0] + t[1] * n[1] + t[2] * n[2]) / norm;
}
void PutFishOnBlocks::constructSurface(
    const Real h, const Real ox, const Real oy, const Real oz,
    ObstacleBlock *const defblock,
    const std::vector<VolumeSegment_OBB *> &vSegments) const {
  const Real *const rX = cfish->rX, *const norX = cfish->norX,
                    *const vBinX = cfish->vBinX;
  const Real *const rY = cfish->rY, *const norY = cfish->norY,
                    *const vBinY = cfish->vBinY;
  const Real *const rZ = cfish->rZ, *const norZ = cfish->norZ,
                    *const vBinZ = cfish->vBinZ;
  const Real *const vX = cfish->vX, *const vNorX = cfish->vNorX,
                    *const binX = cfish->binX;
  const Real *const vY = cfish->vY, *const vNorY = cfish->vNorY,
                    *const binY = cfish->binY;
  const Real *const vZ = cfish->vZ, *const vNorZ = cfish->vNorZ,
                    *const binZ = cfish->binZ;
  Real *const width = cfish->width;
  Real *const height = cfish->height;
  CHIMAT &__restrict__ CHI = defblock->chi;
  UDEFMAT &__restrict__ UDEF = defblock->udef;
  auto &__restrict__ SDFLAB = defblock->sdfLab;
  const Real org[3] = {ox - h, oy - h, oz - h};
  const Real invh = 1.0 / h;
  const int BS[3] = {ScalarBlock::sizeX + 2, ScalarBlock::sizeY + 2,
                     ScalarBlock::sizeZ + 2};
  const Real *const rS = cfish->rS;
  const Real length = cfish->length;
  Real myP[3] = {rX[0], rY[0], rZ[0]};
  changeToComputationalFrame(myP);
  cfish->sensorLocation[0] = myP[0];
  cfish->sensorLocation[1] = myP[1];
  cfish->sensorLocation[2] = myP[2];
  for (size_t i = 0; i < vSegments.size(); ++i) {
    const int firstSegm = std::max(vSegments[i]->s_range.first, 1);
    const int lastSegm = std::min(vSegments[i]->s_range.second, cfish->Nm - 2);
    for (int ss = firstSegm; ss <= lastSegm; ++ss) {
      if (height[ss] <= 0)
        height[ss] = 1e-10;
      if (width[ss] <= 0)
        width[ss] = 1e-10;
      const Real major_axis = std::max(height[ss], width[ss]);
      const Real dtheta_tgt = std::fabs(std::asin(h / (major_axis + h) / 2));
      int Ntheta = std::ceil(2 * M_PI / dtheta_tgt);
      if (Ntheta % 2 == 1)
        Ntheta++;
      const Real dtheta = 2 * M_PI / ((Real)Ntheta);
      const Real offset = height[ss] > width[ss] ? M_PI / 2 : 0;
      for (int tt = 0; tt < Ntheta; ++tt) {
        const Real theta = tt * dtheta + offset;
        const Real sinth = std::sin(theta), costh = std::cos(theta);
        myP[0] = rX[ss] + width[ss] * costh * norX[ss] +
                 height[ss] * sinth * binX[ss];
        myP[1] = rY[ss] + width[ss] * costh * norY[ss] +
                 height[ss] * sinth * binY[ss];
        myP[2] = rZ[ss] + width[ss] * costh * norZ[ss] +
                 height[ss] * sinth * binZ[ss];
        changeToComputationalFrame(myP);
        if (rS[ss] <= 0.04 * length && rS[ss + 1] > 0.04 * length) {
          if (tt == 0) {
            cfish->sensorLocation[1 * 3 + 0] = myP[0];
            cfish->sensorLocation[1 * 3 + 1] = myP[1];
            cfish->sensorLocation[1 * 3 + 2] = myP[2];
          }
          if (tt == (int)Ntheta / 2) {
            cfish->sensorLocation[2 * 3 + 0] = myP[0];
            cfish->sensorLocation[2 * 3 + 1] = myP[1];
            cfish->sensorLocation[2 * 3 + 2] = myP[2];
          }
        }
        const int iap[3] = {(int)std::floor((myP[0] - org[0]) * invh),
                            (int)std::floor((myP[1] - org[1]) * invh),
                            (int)std::floor((myP[2] - org[2]) * invh)};
        const int nei = 3;
        const int ST[3] = {iap[0] - nei, iap[1] - nei, iap[2] - nei};
        const int EN[3] = {iap[0] + nei, iap[1] + nei, iap[2] + nei};
        if (EN[0] <= 0 || ST[0] > BS[0])
          continue;
        if (EN[1] <= 0 || ST[1] > BS[1])
          continue;
        if (EN[2] <= 0 || ST[2] > BS[2])
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
        changeToComputationalFrame(pM);
        changeToComputationalFrame(pP);
        Real udef[3] = {vX[ss] + width[ss] * costh * vNorX[ss] +
                            height[ss] * sinth * vBinX[ss],
                        vY[ss] + width[ss] * costh * vNorY[ss] +
                            height[ss] * sinth * vBinY[ss],
                        vZ[ss] + width[ss] * costh * vNorZ[ss] +
                            height[ss] * sinth * vBinZ[ss]};
        changeVelocityToComputationalFrame(udef);
        for (int sz = std::max(0, ST[2]); sz < std::min(EN[2], BS[2]); ++sz)
          for (int sy = std::max(0, ST[1]); sy < std::min(EN[1], BS[1]); ++sy)
            for (int sx = std::max(0, ST[0]); sx < std::min(EN[0], BS[0]);
                 ++sx) {
              Real p[3];
              p[0] = ox + h * (sx - 1 + 0.5);
              p[1] = oy + h * (sy - 1 + 0.5);
              p[2] = oz + h * (sz - 1 + 0.5);
              const Real dist0 = eulerDistSq3D(p, myP);
              const Real distP = eulerDistSq3D(p, pP);
              const Real distM = eulerDistSq3D(p, pM);
              if (std::fabs(SDFLAB[sz][sy][sx]) <
                  std::min({dist0, distP, distM}))
                continue;
              if (std::min({dist0, distP, distM}) > 4 * h * h)
                continue;
              changeFromComputationalFrame(p);
              int close_s = ss, secnd_s = ss + (distP < distM ? 1 : -1);
              Real dist1 = dist0, dist2 = distP < distM ? distP : distM;
              if (distP < dist0 || distM < dist0) {
                dist1 = dist2;
                dist2 = dist0;
                close_s = secnd_s;
                secnd_s = ss;
              }
              const Real W =
                  std::max(1 - std::sqrt(dist1) * (invh / 3), (Real)0);
              const bool inRange =
                  (sz - 1 >= 0 && sz - 1 < ScalarBlock::sizeZ && sy - 1 >= 0 &&
                   sy - 1 < ScalarBlock::sizeY && sx - 1 >= 0 &&
                   sx - 1 < ScalarBlock::sizeX);
              if (inRange) {
                UDEF[sz - 1][sy - 1][sx - 1][0] = W * udef[0];
                UDEF[sz - 1][sy - 1][sx - 1][1] = W * udef[1];
                UDEF[sz - 1][sy - 1][sx - 1][2] = W * udef[2];
                CHI[sz - 1][sy - 1][sx - 1] = W;
              }
              const Real R1[3] = {rX[secnd_s] - rX[close_s],
                                  rY[secnd_s] - rY[close_s],
                                  rZ[secnd_s] - rZ[close_s]};
              const Real normR1 =
                  1.0 / (1e-21 + std::sqrt(R1[0] * R1[0] + R1[1] * R1[1] +
                                           R1[2] * R1[2]));
              const Real nn[3] = {R1[0] * normR1, R1[1] * normR1,
                                  R1[2] * normR1};
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
              const Real radius_close = std::pow(width[close_s] * costh, 2) +
                                        std::pow(height[close_s] * sinth, 2) -
                                        base1 * base1;
              const Real radius_second = std::pow(width[secnd_s] * costh, 2) +
                                         std::pow(height[secnd_s] * sinth, 2) -
                                         base2 * base2;
              const Real center_close[3] = {rX[close_s] - nn[0] * base1,
                                            rY[close_s] - nn[1] * base1,
                                            rZ[close_s] - nn[2] * base1};
              const Real center_second[3] = {rX[secnd_s] + nn[0] * base2,
                                             rY[secnd_s] + nn[1] * base2,
                                             rZ[secnd_s] + nn[2] * base2};
              const Real dSsq =
                  std::pow(center_close[0] - center_second[0], 2) +
                  std::pow(center_close[1] - center_second[1], 2) +
                  std::pow(center_close[2] - center_second[2], 2);
              const Real corr = 2 * std::sqrt(radius_close * radius_second);
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
                SDFLAB[sz][sy][sx] = distPlane(PC, PT, PP, p, PF);
              } else if (dSsq >= radius_close + radius_second - corr) {
                const Real xMidl[3] = {rX[close_s], rY[close_s], rZ[close_s]};
                const Real grd2ML = eulerDistSq3D(p, xMidl);
                const Real sign = grd2ML > radius_close ? -1 : 1;
                SDFLAB[sz][sy][sx] = sign * dist1;
              } else {
                const Real Rsq = (radius_close + radius_second - corr + dSsq) *
                                 (radius_close + radius_second + corr + dSsq) /
                                 4 / dSsq;
                const Real maxAx = std::max(radius_close, radius_second);
                const Real d = std::sqrt((Rsq - maxAx) / dSsq);
                Real sign;
                if (radius_close > radius_second) {
                  const Real xMidl[3] = {
                      center_close[0] +
                          (center_close[0] - center_second[0]) * d,
                      center_close[1] +
                          (center_close[1] - center_second[1]) * d,
                      center_close[2] +
                          (center_close[2] - center_second[2]) * d};
                  const Real grd2Core = eulerDistSq3D(p, xMidl);
                  sign = grd2Core > Rsq ? -1 : 1;
                } else {
                  const Real xMidl[3] = {
                      center_second[0] +
                          (center_second[0] - center_close[0]) * d,
                      center_second[1] +
                          (center_second[1] - center_close[1]) * d,
                      center_second[2] +
                          (center_second[2] - center_close[2]) * d};
                  const Real grd2Core = eulerDistSq3D(p, xMidl);
                  sign = grd2Core > Rsq ? -1 : 1;
                }
                SDFLAB[sz][sy][sx] = sign * dist1;
              }
            }
      }
    }
  }
}
void PutFishOnBlocks::constructInternl(
    const Real h, const Real ox, const Real oy, const Real oz,
    ObstacleBlock *const defblock,
    const std::vector<VolumeSegment_OBB *> &vSegments) const {
  Real org[3] = {ox - h, oy - h, oz - h};
  const Real invh = 1.0 / h;
  CHIMAT &__restrict__ CHI = defblock->chi;
  auto &__restrict__ SDFLAB = defblock->sdfLab;
  UDEFMAT &__restrict__ UDEF = defblock->udef;
  static constexpr int BS[3] = {ScalarBlock::sizeX + 2, ScalarBlock::sizeY + 2,
                                ScalarBlock::sizeZ + 2};
  const Real *const rX = cfish->rX, *const norX = cfish->norX,
                    *const vBinX = cfish->vBinX;
  const Real *const rY = cfish->rY, *const norY = cfish->norY,
                    *const vBinY = cfish->vBinY;
  const Real *const rZ = cfish->rZ, *const norZ = cfish->norZ,
                    *const vBinZ = cfish->vBinZ;
  const Real *const vX = cfish->vX, *const vNorX = cfish->vNorX,
                    *const binX = cfish->binX;
  const Real *const vY = cfish->vY, *const vNorY = cfish->vNorY,
                    *const binY = cfish->binY;
  const Real *const vZ = cfish->vZ, *const vNorZ = cfish->vNorZ,
                    *const binZ = cfish->binZ;
  const Real *const width = cfish->width, *const height = cfish->height;
  for (size_t i = 0; i < vSegments.size(); ++i) {
    const int firstSegm = std::max(vSegments[i]->s_range.first, 1);
    const int lastSegm = std::min(vSegments[i]->s_range.second, cfish->Nm - 2);
    for (int ss = firstSegm; ss <= lastSegm; ++ss) {
      const Real myWidth = width[ss], myHeight = height[ss];
      assert(myWidth > 0 && myHeight > 0);
      const int Nh = std::floor(myHeight / h);
      for (int ih = -Nh + 1; ih < Nh; ++ih) {
        const Real offsetH = ih * h;
        const Real currWidth =
            myWidth * std::sqrt(1 - std::pow(offsetH / myHeight, 2));
        const int Nw = std::floor(currWidth / h);
        for (int iw = -Nw + 1; iw < Nw; ++iw) {
          const Real offsetW = iw * h;
          Real xp[3] = {rX[ss] + offsetW * norX[ss] + offsetH * binX[ss],
                        rY[ss] + offsetW * norY[ss] + offsetH * binY[ss],
                        rZ[ss] + offsetW * norZ[ss] + offsetH * binZ[ss]};
          changeToComputationalFrame(xp);
          xp[0] = (xp[0] - org[0]) * invh;
          xp[1] = (xp[1] - org[1]) * invh;
          xp[2] = (xp[2] - org[2]) * invh;
          const Real ap[3] = {std::floor(xp[0]), std::floor(xp[1]),
                              std::floor(xp[2])};
          const int iap[3] = {(int)ap[0], (int)ap[1], (int)ap[2]};
          if (iap[0] + 2 <= 0 || iap[0] >= BS[0])
            continue;
          if (iap[1] + 2 <= 0 || iap[1] >= BS[1])
            continue;
          if (iap[2] + 2 <= 0 || iap[2] >= BS[2])
            continue;
          Real udef[3] = {vX[ss] + offsetW * vNorX[ss] + offsetH * vBinX[ss],
                          vY[ss] + offsetW * vNorY[ss] + offsetH * vBinY[ss],
                          vZ[ss] + offsetW * vNorZ[ss] + offsetH * vBinZ[ss]};
          changeVelocityToComputationalFrame(udef);
          Real wghts[3][2];
          for (int c = 0; c < 3; ++c) {
            const Real t[2] = {std::fabs(xp[c] - ap[c]),
                               std::fabs(xp[c] - (ap[c] + 1))};
            wghts[c][0] = 1.0 - t[0];
            wghts[c][1] = 1.0 - t[1];
            assert(wghts[c][0] >= 0 && wghts[c][1] >= 0);
          }
          for (int idz = std::max(0, iap[2]); idz < std::min(iap[2] + 2, BS[2]);
               ++idz)
            for (int idy = std::max(0, iap[1]);
                 idy < std::min(iap[1] + 2, BS[1]); ++idy)
              for (int idx = std::max(0, iap[0]);
                   idx < std::min(iap[0] + 2, BS[0]); ++idx) {
                const int sx = idx - iap[0], sy = idy - iap[1],
                          sz = idz - iap[2];
                assert(sx >= 0 && sx < 2 && sy >= 0 && sy < 2 && sz >= 0 &&
                       sz < 2);
                const Real wxwywz = wghts[2][sz] * wghts[1][sy] * wghts[0][sx];
                assert(wxwywz >= 0 && wxwywz <= 1);
                if (idz - 1 >= 0 && idz - 1 < ScalarBlock::sizeZ &&
                    idy - 1 >= 0 && idy - 1 < ScalarBlock::sizeY &&
                    idx - 1 >= 0 && idx - 1 < ScalarBlock::sizeX) {
                  UDEF[idz - 1][idy - 1][idx - 1][0] += wxwywz * udef[0];
                  UDEF[idz - 1][idy - 1][idx - 1][1] += wxwywz * udef[1];
                  UDEF[idz - 1][idy - 1][idx - 1][2] += wxwywz * udef[2];
                  CHI[idz - 1][idy - 1][idx - 1] += wxwywz;
                }
                static constexpr Real eps =
                    std::numeric_limits<Real>::epsilon();
                if (std::fabs(SDFLAB[idz][idy][idx] + 1) < eps)
                  SDFLAB[idz][idy][idx] = 1;
              }
        }
      }
    }
  }
}
void PutFishOnBlocks::signedDistanceSqrt(ObstacleBlock *const defblock) const {
  static constexpr Real eps = std::numeric_limits<Real>::epsilon();
  auto &__restrict__ CHI = defblock->chi;
  auto &__restrict__ UDEF = defblock->udef;
  auto &__restrict__ SDFLAB = defblock->sdfLab;
  for (int iz = 0; iz < ScalarBlock::sizeZ + 2; iz++)
    for (int iy = 0; iy < ScalarBlock::sizeY + 2; iy++)
      for (int ix = 0; ix < ScalarBlock::sizeX + 2; ix++) {
        if (iz < ScalarBlock::sizeZ && iy < ScalarBlock::sizeY &&
            ix < ScalarBlock::sizeX) {
          if (CHI[iz][iy][ix] > eps) {
            const Real normfac = 1.0 / CHI[iz][iy][ix];
            UDEF[iz][iy][ix][0] *= normfac;
            UDEF[iz][iy][ix][1] *= normfac;
            UDEF[iz][iy][ix][2] *= normfac;
          }
        }
        SDFLAB[iz][iy][ix] = SDFLAB[iz][iy][ix] >= 0
                                 ? std::sqrt(SDFLAB[iz][iy][ix])
                                 : -std::sqrt(-SDFLAB[iz][iy][ix]);
      }
}
void PutNacaOnBlocks::constructSurface(
    const Real h, const Real ox, const Real oy, const Real oz,
    ObstacleBlock *const defblock,
    const std::vector<VolumeSegment_OBB *> &vSegments) const {
  Real org[3] = {ox - h, oy - h, oz - h};
  const Real invh = 1.0 / h;
  const Real *const rX = cfish->rX;
  const Real *const rY = cfish->rY;
  const Real *const norX = cfish->norX;
  const Real *const norY = cfish->norY;
  const Real *const vX = cfish->vX;
  const Real *const vY = cfish->vY;
  const Real *const vNorX = cfish->vNorX;
  const Real *const vNorY = cfish->vNorY;
  const Real *const width = cfish->width;
  const Real *const height = cfish->height;
  static constexpr int BS[3] = {ScalarBlock::sizeX + 2, ScalarBlock::sizeY + 2,
                                ScalarBlock::sizeZ + 2};
  CHIMAT &__restrict__ CHI = defblock->chi;
  auto &__restrict__ SDFLAB = defblock->sdfLab;
  UDEFMAT &__restrict__ UDEF = defblock->udef;
  for (size_t i = 0; i < vSegments.size(); ++i) {
    const int firstSegm = std::max(vSegments[i]->s_range.first, 1);
    const int lastSegm = std::min(vSegments[i]->s_range.second, cfish->Nm - 2);
    for (int ss = firstSegm; ss <= lastSegm; ++ss) {
      assert(height[ss] > 0 && width[ss] > 0);
      for (int signp = -1; signp <= 1; signp += 2) {
        Real myP[3] = {rX[ss + 0] + width[ss + 0] * signp * norX[ss + 0],
                       rY[ss + 0] + width[ss + 0] * signp * norY[ss + 0], 0};
        const Real pP[3] = {rX[ss + 1] + width[ss + 1] * signp * norX[ss + 1],
                            rY[ss + 1] + width[ss + 1] * signp * norY[ss + 1],
                            0};
        const Real pM[3] = {rX[ss - 1] + width[ss - 1] * signp * norX[ss - 1],
                            rY[ss - 1] + width[ss - 1] * signp * norY[ss - 1],
                            0};
        changeToComputationalFrame(myP);
        const int iap[2] = {(int)std::floor((myP[0] - org[0]) * invh),
                            (int)std::floor((myP[1] - org[1]) * invh)};
        Real udef[3] = {vX[ss + 0] + width[ss + 0] * signp * vNorX[ss + 0],
                        vY[ss + 0] + width[ss + 0] * signp * vNorY[ss + 0], 0};
        changeVelocityToComputationalFrame(udef);
        for (int sy = std::max(0, iap[1] - 1); sy < std::min(iap[1] + 3, BS[1]);
             ++sy)
          for (int sx = std::max(0, iap[0] - 1);
               sx < std::min(iap[0] + 3, BS[0]); ++sx) {
            Real p[3];
            p[0] = ox + h * (sx - 1 + 0.5);
            p[1] = oy + h * (sy - 1 + 0.5);
            p[2] = oz + h * (0 - 1 + 0.5);
            const Real dist0 = eulerDistSq2D(p, myP);
            changeFromComputationalFrame(p);
#ifndef NDEBUG
            const Real p0[3] = {rX[ss] + width[ss] * signp * norX[ss],
                                rY[ss] + width[ss] * signp * norY[ss], 0};
            const Real distC = eulerDistSq2D(p, p0);
            assert(std::fabs(distC - dist0) < 2.2e-16);
#endif
            const Real distP = eulerDistSq2D(p, pP),
                       distM = eulerDistSq2D(p, pM);
            int close_s = ss, secnd_s = ss + (distP < distM ? 1 : -1);
            Real dist1 = dist0, dist2 = distP < distM ? distP : distM;
            if (distP < dist0 || distM < dist0) {
              dist1 = dist2;
              dist2 = dist0;
              close_s = secnd_s;
              secnd_s = ss;
            }
            const Real dSsq = std::pow(rX[close_s] - rX[secnd_s], 2) +
                              std::pow(rY[close_s] - rY[secnd_s], 2);
            assert(dSsq > 2.2e-16);
            const Real cnt2ML = std::pow(width[close_s], 2);
            const Real nxt2ML = std::pow(width[secnd_s], 2);
            Real sign2d = 0;
            if (dSsq >= std::fabs(cnt2ML - nxt2ML)) {
              const Real xMidl[3] = {rX[close_s], rY[close_s], 0};
              const Real grd2ML = eulerDistSq2D(p, xMidl);
              sign2d = grd2ML > cnt2ML ? -1 : 1;
            } else {
              const Real corr = 2 * std::sqrt(cnt2ML * nxt2ML);
              const Real Rsq = (cnt2ML + nxt2ML - corr + dSsq) *
                               (cnt2ML + nxt2ML + corr + dSsq) / 4 / dSsq;
              const Real maxAx = std::max(cnt2ML, nxt2ML);
              const int idAx1 = cnt2ML > nxt2ML ? close_s : secnd_s;
              const int idAx2 = idAx1 == close_s ? secnd_s : close_s;
              const Real d = std::sqrt((Rsq - maxAx) / dSsq);
              const Real xMidl[3] = {rX[idAx1] + (rX[idAx1] - rX[idAx2]) * d,
                                     rY[idAx1] + (rY[idAx1] - rY[idAx2]) * d,
                                     0};
              const Real grd2Core = eulerDistSq2D(p, xMidl);
              sign2d = grd2Core > Rsq ? -1 : 1;
            }
            for (int sz = 0; sz < BS[2]; ++sz) {
              const Real pZ = org[2] + h * sz;
              const Real distZ = height[ss] - std::fabs(position[2] - pZ);
              const Real signZ = (0 < distZ) - (distZ < 0);
              const Real dist3D =
                  std::min(signZ * distZ * distZ, sign2d * dist1);
              if (std::fabs(SDFLAB[sz][sy][sx]) > dist3D) {
                SDFLAB[sz][sy][sx] = dist3D;
                const bool inRange =
                    (sz - 1 >= 0 && sz - 1 < ScalarBlock::sizeZ &&
                     sy - 1 >= 0 && sy - 1 < ScalarBlock::sizeY &&
                     sx - 1 >= 0 && sx - 1 < ScalarBlock::sizeX);
                if (inRange) {
                  UDEF[sz - 1][sy - 1][sx - 1][0] = udef[0];
                  UDEF[sz - 1][sy - 1][sx - 1][1] = udef[1];
                  UDEF[sz - 1][sy - 1][sx - 1][2] = udef[2];
                  CHI[sz - 1][sy - 1][sx - 1] = 1;
                }
              }
            }
          }
      }
    }
  }
}
void PutNacaOnBlocks::constructInternl(
    const Real h, const Real ox, const Real oy, const Real oz,
    ObstacleBlock *const defblock,
    const std::vector<VolumeSegment_OBB *> &vSegments) const {
  Real org[3] = {ox - h, oy - h, oz - h};
  const Real invh = 1.0 / h;
  const Real EPS = 1e-15;
  CHIMAT &__restrict__ CHI = defblock->chi;
  auto &__restrict__ SDFLAB = defblock->sdfLab;
  UDEFMAT &__restrict__ UDEF = defblock->udef;
  static constexpr int BS[3] = {ScalarBlock::sizeX + 2, ScalarBlock::sizeY + 2,
                                ScalarBlock::sizeZ + 2};
  for (size_t i = 0; i < vSegments.size(); ++i) {
    const int firstSegm = std::max(vSegments[i]->s_range.first, 1);
    const int lastSegm = std::min(vSegments[i]->s_range.second, cfish->Nm - 2);
    for (int ss = firstSegm; ss <= lastSegm; ++ss) {
      const Real myWidth = cfish->width[ss], myHeight = cfish->height[ss];
      assert(myWidth > 0 && myHeight > 0);
      const int Nw = std::floor(myWidth / h);
      for (int iw = -Nw + 1; iw < Nw; ++iw) {
        const Real offsetW = iw * h;
        Real xp[3] = {cfish->rX[ss] + offsetW * cfish->norX[ss],
                      cfish->rY[ss] + offsetW * cfish->norY[ss], 0};
        changeToComputationalFrame(xp);
        xp[0] = (xp[0] - org[0]) * invh;
        xp[1] = (xp[1] - org[1]) * invh;
        Real udef[3] = {cfish->vX[ss] + offsetW * cfish->vNorX[ss],
                        cfish->vY[ss] + offsetW * cfish->vNorY[ss], 0};
        changeVelocityToComputationalFrame(udef);
        const Real ap[2] = {std::floor(xp[0]), std::floor(xp[1])};
        const int iap[2] = {(int)ap[0], (int)ap[1]};
        Real wghts[2][2];
        for (int c = 0; c < 2; ++c) {
          const Real t[2] = {std::fabs(xp[c] - ap[c]),
                             std::fabs(xp[c] - (ap[c] + 1))};
          wghts[c][0] = 1.0 - t[0];
          wghts[c][1] = 1.0 - t[1];
        }
        for (int idz = 0; idz < BS[2]; ++idz) {
          const Real pZ = org[2] + h * idz;
          const Real distZ = myHeight - std::fabs(position[2] - pZ);
          static constexpr Real one = 1;
          const Real wz = .5 + std::min(one, std::max(distZ * invh, -one)) / 2;
          const Real signZ = (0 < distZ) - (distZ < 0);
          const Real distZsq = signZ * distZ * distZ;
          using std::max;
          using std::min;
          for (int sy = max(0, 0 - iap[1]); sy < min(2, BS[1] - iap[1]); ++sy)
            for (int sx = max(0, 0 - iap[0]); sx < min(2, BS[0] - iap[0]);
                 ++sx) {
              const Real wxwywz = wz * wghts[1][sy] * wghts[0][sx];
              const int idx = iap[0] + sx, idy = iap[1] + sy;
              assert(idx >= 0 && idx < BS[0]);
              assert(idy >= 0 && idy < BS[1]);
              assert(wxwywz >= 0 && wxwywz <= 1);
              if (idz - 1 >= 0 && idz - 1 < ScalarBlock::sizeZ &&
                  idy - 1 >= 0 && idy - 1 < ScalarBlock::sizeY &&
                  idx - 1 >= 0 && idx - 1 < ScalarBlock::sizeX) {
                UDEF[idz - 1][idy - 1][idx - 1][0] += wxwywz * udef[0];
                UDEF[idz - 1][idy - 1][idx - 1][1] += wxwywz * udef[1];
                UDEF[idz - 1][idy - 1][idx - 1][2] += wxwywz * udef[2];
                CHI[idz - 1][idy - 1][idx - 1] += wxwywz;
              }
              if (std::fabs(SDFLAB[idz][idy][idx] + 1) < EPS)
                SDFLAB[idz][idy][idx] = distZsq;
            }
        }
      }
    }
  }
}
void MidlineShapes::integrateBSpline(const Real *const xc, const Real *const yc,
                                     const int n, const Real length,
                                     Real *const rS, Real *const res,
                                     const int Nm) {
  Real len = 0;
  for (int i = 0; i < n - 1; i++) {
    len += std::sqrt(std::pow(xc[i] - xc[i + 1], 2) +
                     std::pow(yc[i] - yc[i + 1], 2));
  }
  gsl_bspline_workspace *bw;
  gsl_vector *B;
  bw = gsl_bspline_alloc(4, n - 2);
  B = gsl_vector_alloc(n);
  gsl_bspline_knots_uniform(0.0, len, bw);
  Real ti = 0;
  for (int i = 0; i < Nm; ++i) {
    res[i] = 0;
    if (rS[i] > 0 and rS[i] < length) {
      const Real dtt = (rS[i] - rS[i - 1]) / 1e3;
      while (true) {
        Real xi = 0;
        gsl_bspline_eval(ti, B, bw);
        for (int j = 0; j < n; j++)
          xi += xc[j] * gsl_vector_get(B, j);
        if (xi >= rS[i])
          break;
        if (ti + dtt > len)
          break;
        else
          ti += dtt;
      }
      for (int j = 0; j < n; j++)
        res[i] += yc[j] * gsl_vector_get(B, j);
    }
  }
  gsl_bspline_free(bw);
  gsl_vector_free(B);
}
void MidlineShapes::naca_width(const Real t_ratio, const Real L, Real *const rS,
                               Real *const res, const int Nm) {
  const Real a = 0.2969;
  const Real b = -0.1260;
  const Real c = -0.3516;
  const Real d = 0.2843;
  const Real e = -0.1015;
  const Real t = t_ratio * L;
  for (int i = 0; i < Nm; ++i) {
    if (rS[i] <= 0 or rS[i] >= L)
      res[i] = 0;
    else {
      const Real p = rS[i] / L;
      res[i] = 5 * t *
               (a * std::sqrt(p) + b * p + c * p * p + d * p * p * p +
                e * p * p * p * p);
    }
  }
}
void MidlineShapes::stefan_width(const Real L, Real *const rS, Real *const res,
                                 const int Nm) {
  const Real sb = .04 * L;
  const Real st = .95 * L;
  const Real wt = .01 * L;
  const Real wh = .04 * L;
  for (int i = 0; i < Nm; ++i) {
    if (rS[i] <= 0 or rS[i] >= L)
      res[i] = 0;
    else {
      const Real s = rS[i];
      res[i] =
          (s < sb ? std::sqrt(2.0 * wh * s - s * s)
                  : (s < st ? wh - (wh - wt) * std::pow((s - sb) / (st - sb), 2)
                            : (wt * (L - s) / (L - st))));
    }
  }
}
void MidlineShapes::stefan_height(const Real L, Real *const rS, Real *const res,
                                  const int Nm) {
  const Real a = 0.51 * L;
  const Real b = 0.08 * L;
  for (int i = 0; i < Nm; ++i) {
    if (rS[i] <= 0 or rS[i] >= L)
      res[i] = 0;
    else {
      const Real s = rS[i];
      res[i] = b * std::sqrt(1 - std::pow((s - a) / a, 2));
    }
  }
}
void MidlineShapes::larval_width(const Real L, Real *const rS, Real *const res,
                                 const int Nm) {
  const Real sb = .0862 * L;
  const Real st = .3448 * L;
  const Real wh = .0635 * L;
  const Real wt = .0254 * L;
  for (int i = 0; i < Nm; ++i) {
    if (rS[i] <= 0 or rS[i] >= L)
      res[i] = 0;
    else {
      const Real s = rS[i];
      res[i] = s < sb ? wh * std::sqrt(1 - std::pow((sb - s) / sb, 2))
                      : (s < st ? (-2 * (wt - wh) - wt * (st - sb)) *
                                          std::pow((s - sb) / (st - sb), 3) +
                                      (3 * (wt - wh) + wt * (st - sb)) *
                                          std::pow((s - sb) / (st - sb), 2) +
                                      wh
                                : (wt - wt * (s - st) / (L - st)));
    }
  }
}
void MidlineShapes::larval_height(const Real L, Real *const rS, Real *const res,
                                  const int Nm) {
  const Real s1 = 0.287 * L;
  const Real h1 = 0.072 * L;
  const Real s2 = 0.844 * L;
  const Real h2 = 0.041 * L;
  const Real s3 = 0.957 * L;
  const Real h3 = 0.071 * L;
  for (int i = 0; i < Nm; ++i) {
    if (rS[i] <= 0 or rS[i] >= L)
      res[i] = 0;
    else {
      const Real s = rS[i];
      res[i] =
          s < s1
              ? (h1 * std::sqrt(1 - std::pow((s - s1) / s1, 2)))
              : (s < s2
                     ? -2 * (h2 - h1) * std::pow((s - s1) / (s2 - s1), 3) +
                           3 * (h2 - h1) * std::pow((s - s1) / (s2 - s1), 2) +
                           h1
                     : (s < s3
                            ? -2 * (h3 - h2) *
                                      std::pow((s - s2) / (s3 - s2), 3) +
                                  3 * (h3 - h2) *
                                      std::pow((s - s2) / (s3 - s2), 2) +
                                  h2
                            : (h3 * std::sqrt(1 - std::pow((s - s3) / (L - s3),
                                                           3)))));
    }
  }
}
void MidlineShapes::danio_width(const Real L, Real *const rS, Real *const res,
                                const int Nm) {
  const int nBreaksW = 11;
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
    if (rS[i] <= 0 or rS[i] >= L)
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
void MidlineShapes::danio_height(const Real L, Real *const rS, Real *const res,
                                 const int Nm) {
  const int nBreaksH = 15;
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
    if (rS[i] <= 0 or rS[i] >= L)
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
void MidlineShapes::computeWidthsHeights(const std::string &heightName,
                                         const std::string &widthName,
                                         const Real L, Real *const rS,
                                         Real *const height, Real *const width,
                                         const int nM, const int mpirank) {
  using std::cout;
  using std::endl;
  if (!mpirank) {
    printf("height = %s, width=%s\n", heightName.c_str(), widthName.c_str());
    fflush(NULL);
  }
  {
    if (heightName.compare("largefin") == 0) {
      Real xh[8] = {0, 0, .2 * L, .4 * L, .6 * L, .8 * L, L, L};
      Real yh[8] = {0,        .055 * L, .18 * L,  .2 * L,
                    .064 * L, .002 * L, .325 * L, 0};
      integrateBSpline(xh, yh, 8, L, rS, height, nM);
    } else if (heightName.compare("tunaclone") == 0) {
      Real xh[9] = {0, 0, 0.2 * L, .4 * L, .6 * L, .9 * L, .96 * L, L, L};
      Real yh[9] = {0, .05 * L, .14 * L, .15 * L, .11 * L,
                    0, .1 * L,  .2 * L,  0};
      integrateBSpline(xh, yh, 9, L, rS, height, nM);
    } else if (heightName.compare(0, 4, "naca") == 0) {
      Real t_naca = std::stoi(heightName.substr(5), nullptr, 10) * 0.01;
      naca_width(t_naca, L, rS, height, nM);
    } else if (heightName.compare("danio") == 0) {
      danio_height(L, rS, height, nM);
    } else if (heightName.compare("stefan") == 0) {
      if (!mpirank)
        cout << "Building object's height according to Stefan profile" << endl;
      stefan_height(L, rS, height, nM);
    } else if (heightName.compare("larval") == 0) {
      if (!mpirank)
        cout << "Building object's height according to Larval profile" << endl;
      larval_height(L, rS, height, nM);
    } else {
      Real xh[8] = {0, 0, .2 * L, .4 * L, .6 * L, .8 * L, L, L};
      Real yh[8] = {0,        .055 * L,  .068 * L, .076 * L,
                    .064 * L, .0072 * L, .11 * L,  0};
      integrateBSpline(xh, yh, 8, L, rS, height, nM);
    }
  }
  {
    if (widthName.compare("fatter") == 0) {
      Real xw[6] = {0, 0, L / 3., 2 * L / 3., L, L};
      Real yw[6] = {0, 8.9e-2 * L, 7.0e-2 * L, 3.0e-2 * L, 2.0e-2 * L, 0};
      integrateBSpline(xw, yw, 6, L, rS, width, nM);
    } else if (widthName.compare(0, 4, "naca") == 0) {
      Real t_naca = std::stoi(widthName.substr(5), nullptr, 10) * 0.01;
      naca_width(t_naca, L, rS, width, nM);
    } else if (widthName.compare("danio") == 0) {
      danio_width(L, rS, width, nM);
    } else if (widthName.compare("stefan") == 0) {
      stefan_width(L, rS, width, nM);
    } else if (widthName.compare("larval") == 0) {
      larval_width(L, rS, width, nM);
    } else {
      Real xw[6] = {0, 0, L / 3., 2 * L / 3., L, L};
      Real yw[6] = {0, 8.9e-2 * L, 1.7e-2 * L, 1.6e-2 * L, 1.3e-2 * L, 0};
      integrateBSpline(xw, yw, 6, L, rS, width, nM);
    }
  }
}
static Real avgUx_nonUniform(const std::vector<Info> &myInfo,
                             const Real *const uInf, const Real volume) {
  Real avgUx = 0.;
  const int nBlocks = myInfo.size();
#pragma omp parallel for reduction(+ : avgUx)
  for (int i = 0; i < nBlocks; i++) {
    const Info &info = myInfo[i];
    const VectorBlock &b = *(const VectorBlock *)info.block;
    const Real h3 = info.h * info.h * info.h;
    for (int z = 0; z < VectorBlock::sizeZ; ++z)
      for (int y = 0; y < VectorBlock::sizeY; ++y)
        for (int x = 0; x < VectorBlock::sizeX; ++x) {
          avgUx += (b(x, y, z).u[0] + uInf[0]) * h3;
        }
  }
  avgUx = avgUx / volume;
  return avgUx;
}
FixMassFlux::FixMassFlux(SimulationData &s) : Operator(s) {}
void FixMassFlux::operator()(const double dt) {
  const std::vector<Info> &velInfo = sim.velInfo();
  const Real volume = sim.extents[0] * sim.extents[1] * sim.extents[2];
  const Real y_max = sim.extents[1];
  const Real u_avg = 2.0 / 3.0 * sim.uMax_forced;
  Real u_avg_msr = avgUx_nonUniform(velInfo, sim.uinf.data(), volume);
  MPI_Allreduce(MPI_IN_PLACE, &u_avg_msr, 1, MPI_Real, MPI_SUM, sim.comm);
  const Real delta_u = u_avg - u_avg_msr;
  const Real reTau = std::sqrt(std::fabs(delta_u / sim.dt)) / sim.nu;
  const Real scale = 6 * delta_u;
  if (sim.rank == 0) {
    printf("Measured <Ux>_V = %25.16e,\n"
           "target   <Ux>_V = %25.16e,\n"
           "delta    <Ux>_V = %25.16e,\n"
           "scale           = %25.16e,\n"
           "Re_tau          = %25.16e,\n",
           u_avg_msr, u_avg, delta_u, scale, reTau);
  }
#pragma omp parallel for
  for (size_t i = 0; i < velInfo.size(); i++) {
    VectorBlock &v = *(VectorBlock *)velInfo[i].block;
    for (int z = 0; z < VectorBlock::sizeZ; ++z)
      for (int y = 0; y < VectorBlock::sizeY; ++y) {
        Real p[3];
        velInfo[i].pos(p, 0, y, 0);
        const Real aux = 6 * scale * p[1] / y_max * (1.0 - p[1] / y_max);
        for (int x = 0; x < VectorBlock::sizeX; ++x)
          v(x, y, z).u[0] += aux;
      }
  }
}
namespace {
struct KernelComputeForces {
  const int big = 5;
  const int small = -4;
  const int bigg = ScalarBlock::sizeX + big - 1;
  const int stencil_start[3] = {small, small, small},
            stencil_end[3] = {big, big, big};
  const Real c0 = -137. / 60.;
  const Real c1 = 5.;
  const Real c2 = -5.;
  const Real c3 = 10. / 3.;
  const Real c4 = -5. / 4.;
  const Real c5 = 1. / 5.;
  inline bool inrange(const int i) const { return (i >= small && i < bigg); }
  StencilInfo stencil{small, small, small, big, big, big, true, {0, 1, 2}};
  StencilInfo stencil2{small, small, small, big, big, big, true, {0}};
  SimulationData &sim;
  const std::vector<Info> &presInfo = sim.presInfo();
  KernelComputeForces(SimulationData &s) : sim(s) {}
  void operator()(VectorLab &lab, ScalarLab &chiLab, const Info &info,
                  const Info &info2) const {
    for (const auto &obstacle : sim.obstacle_vector->getObstacleVector())
      visit(lab, chiLab, info, info2, obstacle.get());
  }
  void visit(VectorLab &l, ScalarLab &chiLab, const Info &info,
             const Info &info2, Obstacle *const op) const {
    const ScalarBlock &presBlock = *(ScalarBlock *)presInfo[info.blockID].block;
    const std::vector<ObstacleBlock *> &obstblocks = op->getObstacleBlocks();
    ObstacleBlock *const o = obstblocks[info.blockID];
    if (o == nullptr)
      return;
    if (o->nPoints == 0)
      return;
    assert(o->filled);
    o->forcex = 0;
    o->forcex_V = 0;
    o->forcex_P = 0;
    o->torquex = 0;
    o->torquey = 0;
    o->torquez = 0;
    o->thrust = 0;
    o->drag = 0;
    o->Pout = 0;
    o->defPower = 0;
    o->pLocom = 0;
    const std::array<Real, 3> CM = op->getCenterOfMass();
    const std::array<Real, 3> omega = op->getAngularVelocity();
    const std::array<Real, 3> uTrans = op->getTranslationVelocity();
    Real velUnit[3] = {0., 0., 0.};
    const Real vel_norm = std::sqrt(
        uTrans[0] * uTrans[0] + uTrans[1] * uTrans[1] + uTrans[2] * uTrans[2]);
    if (vel_norm > 1e-9) {
      velUnit[0] = uTrans[0] / vel_norm;
      velUnit[1] = uTrans[1] / vel_norm;
      velUnit[2] = uTrans[2] / vel_norm;
    }
    const Real _1oH = sim.nu / info.h;
    for (int i = 0; i < o->nPoints; i++) {
      const int ix = o->surface[i]->ix;
      const int iy = o->surface[i]->iy;
      const int iz = o->surface[i]->iz;
      Real p[3];
      info.pos(p, ix, iy, iz);
      const Real normX = o->surface[i]->dchidx;
      const Real normY = o->surface[i]->dchidy;
      const Real normZ = o->surface[i]->dchidz;
      const Real norm =
          1.0 / std::sqrt(normX * normX + normY * normY + normZ * normZ);
      const Real dx = normX * norm;
      const Real dy = normY * norm;
      const Real dz = normZ * norm;
      int x = ix;
      int y = iy;
      int z = iz;
      for (int kk = 0; kk < 5; kk++) {
        const int dxi = round(kk * dx);
        const int dyi = round(kk * dy);
        const int dzi = round(kk * dz);
        if (ix + dxi + 1 >= ScalarBlock::sizeX + big - 1 ||
            ix + dxi - 1 < small)
          continue;
        if (iy + dyi + 1 >= ScalarBlock::sizeY + big - 1 ||
            iy + dyi - 1 < small)
          continue;
        if (iz + dzi + 1 >= ScalarBlock::sizeZ + big - 1 ||
            iz + dzi - 1 < small)
          continue;
        x = ix + dxi;
        y = iy + dyi;
        z = iz + dzi;
        if (chiLab(x, y, z).s < 0.01)
          break;
      }
      const int sx = normX > 0 ? +1 : -1;
      const int sy = normY > 0 ? +1 : -1;
      const int sz = normZ > 0 ? +1 : -1;
      VectorElement dveldx;
      if (inrange(x + 5 * sx))
        dveldx = sx * (c0 * l(x, y, z) + c1 * l(x + sx, y, z) +
                       c2 * l(x + 2 * sx, y, z) + c3 * l(x + 3 * sx, y, z) +
                       c4 * l(x + 4 * sx, y, z) + c5 * l(x + 5 * sx, y, z));
      else if (inrange(x + 2 * sx))
        dveldx = sx * (-1.5 * l(x, y, z) + 2.0 * l(x + sx, y, z) -
                       0.5 * l(x + 2 * sx, y, z));
      else
        dveldx = sx * (l(x + sx, y, z) - l(x, y, z));
      VectorElement dveldy;
      if (inrange(y + 5 * sy))
        dveldy = sy * (c0 * l(x, y, z) + c1 * l(x, y + sy, z) +
                       c2 * l(x, y + 2 * sy, z) + c3 * l(x, y + 3 * sy, z) +
                       c4 * l(x, y + 4 * sy, z) + c5 * l(x, y + 5 * sy, z));
      else if (inrange(y + 2 * sy))
        dveldy = sy * (-1.5 * l(x, y, z) + 2.0 * l(x, y + sy, z) -
                       0.5 * l(x, y + 2 * sy, z));
      else
        dveldy = sx * (l(x, y + sy, z) - l(x, y, z));
      VectorElement dveldz;
      if (inrange(z + 5 * sz))
        dveldz = sz * (c0 * l(x, y, z) + c1 * l(x, y, z + sz) +
                       c2 * l(x, y, z + 2 * sz) + c3 * l(x, y, z + 3 * sz) +
                       c4 * l(x, y, z + 4 * sz) + c5 * l(x, y, z + 5 * sz));
      else if (inrange(z + 2 * sz))
        dveldz = sz * (-1.5 * l(x, y, z) + 2.0 * l(x, y, z + sz) -
                       0.5 * l(x, y, z + 2 * sz));
      else
        dveldz = sz * (l(x, y, z + sz) - l(x, y, z));
      const VectorElement dveldx2 =
          l(x - 1, y, z) - 2.0 * l(x, y, z) + l(x + 1, y, z);
      const VectorElement dveldy2 =
          l(x, y - 1, z) - 2.0 * l(x, y, z) + l(x, y + 1, z);
      const VectorElement dveldz2 =
          l(x, y, z - 1) - 2.0 * l(x, y, z) + l(x, y, z + 1);
      VectorElement dveldxdy;
      VectorElement dveldxdz;
      VectorElement dveldydz;
      if (inrange(x + 2 * sx) && inrange(y + 2 * sy))
        dveldxdy =
            sx * sy *
            (-0.5 * (-1.5 * l(x + 2 * sx, y, z) + 2 * l(x + 2 * sx, y + sy, z) -
                     0.5 * l(x + 2 * sx, y + 2 * sy, z)) +
             2 * (-1.5 * l(x + sx, y, z) + 2 * l(x + sx, y + sy, z) -
                  0.5 * l(x + sx, y + 2 * sy, z)) -
             1.5 * (-1.5 * l(x, y, z) + 2 * l(x, y + sy, z) -
                    0.5 * l(x, y + 2 * sy, z)));
      else
        dveldxdy = sx * sy * (l(x + sx, y + sy, z) - l(x + sx, y, z)) -
                   (l(x, y + sy, z) - l(x, y, z));
      if (inrange(y + 2 * sy) && inrange(z + 2 * sz))
        dveldydz =
            sy * sz *
            (-0.5 * (-1.5 * l(x, y + 2 * sy, z) + 2 * l(x, y + 2 * sy, z + sz) -
                     0.5 * l(x, y + 2 * sy, z + 2 * sz)) +
             2 * (-1.5 * l(x, y + sy, z) + 2 * l(x, y + sy, z + sz) -
                  0.5 * l(x, y + sy, z + 2 * sz)) -
             1.5 * (-1.5 * l(x, y, z) + 2 * l(x, y, z + sz) -
                    0.5 * l(x, y, z + 2 * sz)));
      else
        dveldydz = sy * sz * (l(x, y + sy, z + sz) - l(x, y + sy, z)) -
                   (l(x, y, z + sz) - l(x, y, z));
      if (inrange(x + 2 * sx) && inrange(z + 2 * sz))
        dveldxdz =
            sx * sz *
            (-0.5 * (-1.5 * l(x, y, z + 2 * sz) + 2 * l(x + sx, y, z + 2 * sz) -
                     0.5 * l(x + 2 * sx, y, z + 2 * sz)) +
             2 * (-1.5 * l(x, y, z + sz) + 2 * l(x + sx, y, z + sz) -
                  0.5 * l(x + 2 * sx, y, z + sz)) -
             1.5 * (-1.5 * l(x, y, z) + 2 * l(x + sx, y, z) -
                    0.5 * l(x + 2 * sx, y, z)));
      else
        dveldxdz = sx * sz * (l(x + sx, y, z + sz) - l(x, y, z + sz)) -
                   (l(x + sx, y, z) - l(x, y, z));
      const Real dudx = dveldx.u[0] + dveldx2.u[0] * (ix - x) +
                        dveldxdy.u[0] * (iy - y) + dveldxdz.u[0] * (iz - z);
      const Real dvdx = dveldx.u[1] + dveldx2.u[1] * (ix - x) +
                        dveldxdy.u[1] * (iy - y) + dveldxdz.u[1] * (iz - z);
      const Real dwdx = dveldx.u[2] + dveldx2.u[2] * (ix - x) +
                        dveldxdy.u[2] * (iy - y) + dveldxdz.u[2] * (iz - z);
      const Real dudy = dveldy.u[0] + dveldy2.u[0] * (iy - y) +
                        dveldydz.u[0] * (iz - z) + dveldxdy.u[0] * (ix - x);
      const Real dvdy = dveldy.u[1] + dveldy2.u[1] * (iy - y) +
                        dveldydz.u[1] * (iz - z) + dveldxdy.u[1] * (ix - x);
      const Real dwdy = dveldy.u[2] + dveldy2.u[2] * (iy - y) +
                        dveldydz.u[2] * (iz - z) + dveldxdy.u[2] * (ix - x);
      const Real dudz = dveldz.u[0] + dveldz2.u[0] * (iz - z) +
                        dveldxdz.u[0] * (ix - x) + dveldydz.u[0] * (iy - y);
      const Real dvdz = dveldz.u[1] + dveldz2.u[1] * (iz - z) +
                        dveldxdz.u[1] * (ix - x) + dveldydz.u[1] * (iy - y);
      const Real dwdz = dveldz.u[2] + dveldz2.u[2] * (iz - z) +
                        dveldxdz.u[2] * (ix - x) + dveldydz.u[2] * (iy - y);
      const Real P = presBlock(ix, iy, iz).s;
      const Real fXV = _1oH * (dudx * normX + dudy * normY + dudz * normZ);
      const Real fYV = _1oH * (dvdx * normX + dvdy * normY + dvdz * normZ);
      const Real fZV = _1oH * (dwdx * normX + dwdy * normY + dwdz * normZ);
      const Real fXP = -P * normX, fYP = -P * normY, fZP = -P * normZ;
      const Real fXT = fXV + fXP, fYT = fYV + fYP, fZT = fZV + fZP;
      o->pX[i] = p[0];
      o->pY[i] = p[1];
      o->pZ[i] = p[2];
      o->P[i] = P;
      o->fX[i] = -P * dx + _1oH * (dudx * dx + dudy * dy + dudz * dz);
      o->fY[i] = -P * dy + _1oH * (dvdx * dx + dvdy * dy + dvdz * dz);
      o->fZ[i] = -P * dz + _1oH * (dwdx * dx + dwdy * dy + dwdz * dz);
      o->fxV[i] = _1oH * (dudx * dx + dudy * dy + dudz * dz);
      o->fyV[i] = _1oH * (dvdx * dx + dvdy * dy + dvdz * dz);
      o->fzV[i] = _1oH * (dwdx * dx + dwdy * dy + dwdz * dz);
      o->omegaX[i] = (dwdy - dvdz) / info.h;
      o->omegaY[i] = (dudz - dwdx) / info.h;
      o->omegaZ[i] = (dvdx - dudy) / info.h;
      o->vxDef[i] = o->udef[iz][iy][ix][0];
      o->vX[i] = l(ix, iy, iz).u[0];
      o->vyDef[i] = o->udef[iz][iy][ix][1];
      o->vY[i] = l(ix, iy, iz).u[1];
      o->vzDef[i] = o->udef[iz][iy][ix][2];
      o->vZ[i] = l(ix, iy, iz).u[2];
      o->forcex += fXT;
      o->forcey += fYT;
      o->forcez += fZT;
      o->forcex_V += fXV;
      o->forcey_V += fYV;
      o->forcez_V += fZV;
      o->forcex_P += fXP;
      o->forcey_P += fYP;
      o->forcez_P += fZP;
      o->torquex += (p[1] - CM[1]) * fZT - (p[2] - CM[2]) * fYT;
      o->torquey += (p[2] - CM[2]) * fXT - (p[0] - CM[0]) * fZT;
      o->torquez += (p[0] - CM[0]) * fYT - (p[1] - CM[1]) * fXT;
      const Real forcePar =
          fXT * velUnit[0] + fYT * velUnit[1] + fZT * velUnit[2];
      o->thrust += .5 * (forcePar + std::fabs(forcePar));
      o->drag -= .5 * (forcePar - std::fabs(forcePar));
      const Real powOut = fXT * o->vX[i] + fYT * o->vY[i] + fZT * o->vZ[i];
      const Real powDef =
          fXT * o->vxDef[i] + fYT * o->vyDef[i] + fZT * o->vzDef[i];
      o->Pout += powOut;
      o->PoutBnd += std::min((Real)0, powOut);
      o->defPower += powDef;
      o->defPowerBnd += std::min((Real)0, powDef);
      const Real rVec[3] = {p[0] - CM[0], p[1] - CM[1], p[2] - CM[2]};
      const Real uSolid[3] = {
          uTrans[0] + omega[1] * rVec[2] - rVec[1] * omega[2],
          uTrans[1] + omega[2] * rVec[0] - rVec[2] * omega[0],
          uTrans[2] + omega[0] * rVec[1] - rVec[0] * omega[1]};
      o->pLocom += fXT * uSolid[0] + fYT * uSolid[1] + fZT * uSolid[2];
    }
  }
};
} // namespace
void ComputeForces::operator()(const Real dt) {
  if (sim.obstacle_vector->nObstacles() == 0)
    return;
  KernelComputeForces K(sim);
  compute<KernelComputeForces, VectorGrid, VectorLab, ScalarGrid, ScalarLab>(
      K, *sim.vel, *sim.chi);
  sim.obstacle_vector->computeForces();
}
class PoissonSolverBase;
namespace {
class KernelIC {
public:
  KernelIC(const Real u) {}
  void operator()(const Info &info, VectorBlock &block) const {
    for (int iz = 0; iz < VectorBlock::sizeZ; ++iz)
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        for (int ix = 0; ix < VectorBlock::sizeX; ++ix)
          block(ix, iy, iz).clear();
  }
};
class KernelIC_taylorGreen {
  const std::array<Real, 3> ext;
  const Real uMax;
  const Real a = 2 * M_PI / ext[0], b = 2 * M_PI / ext[1],
             c = 2 * M_PI / ext[2];
  const Real A = uMax, B = -uMax * ext[1] / ext[0];

public:
  KernelIC_taylorGreen(const std::array<Real, 3> &extents, const Real U)
      : ext{extents}, uMax(U) {}
  void operator()(const Info &info, VectorBlock &block) const {
    for (int iz = 0; iz < VectorBlock::sizeZ; ++iz)
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
          block(ix, iy, iz).clear();
          Real p[3];
          info.pos(p, ix, iy, iz);
          block(ix, iy, iz).u[0] =
              A * std::cos(a * p[0]) * std::sin(b * p[1]) * std::sin(c * p[2]);
          block(ix, iy, iz).u[1] =
              B * std::sin(a * p[0]) * std::cos(b * p[1]) * std::sin(c * p[2]);
        }
  }
};
class IC_vorticity {
public:
  SimulationData &sim;
  const int Ncoil = 90;
  std::vector<Real> phi_coil;
  std::vector<Real> x_coil;
  std::vector<Real> y_coil;
  std::vector<Real> z_coil;
  std::vector<Real> dx_coil;
  std::vector<Real> dy_coil;
  std::vector<Real> dz_coil;
  IC_vorticity(SimulationData &s) : sim(s) {
    phi_coil.resize(Ncoil);
    x_coil.resize(Ncoil);
    y_coil.resize(Ncoil);
    z_coil.resize(Ncoil);
    const int m = 2;
    const Real dphi = 2.0 * M_PI / Ncoil;
    for (int i = 0; i < Ncoil; i++) {
      const Real phi = i * dphi;
      phi_coil[i] = phi;
      const Real R = 0.05 * sin(m * phi);
      x_coil[i] = R * cos(phi) + 1.0;
      y_coil[i] = R * sin(phi) + 1.0;
      z_coil[i] = R * cos(m * phi) + 1.0;
    }
    dx_coil.resize(Ncoil);
    dy_coil.resize(Ncoil);
    dz_coil.resize(Ncoil);
    for (int i = 0; i < Ncoil; i++) {
      const Real phi = i * dphi;
      phi_coil[i] = phi;
      const Real R = 0.05 * sin(m * phi);
      const Real dR = 0.05 * m * cos(m * phi);
      const Real sinphi = sin(phi);
      const Real cosphi = cos(phi);
      dx_coil[i] = dR * cosphi - R * sinphi;
      dy_coil[i] = dR * sinphi + R * cosphi;
      dz_coil[i] = dR * cos(m * phi) - m * R * sin(m * phi);
      const Real norm =
          1.0 / pow(dx_coil[i] * dx_coil[i] + dy_coil[i] * dy_coil[i] +
                        dz_coil[i] * dz_coil[i] + 1e-21,
                    0.5);
      dx_coil[i] *= norm;
      dy_coil[i] *= norm;
      dz_coil[i] *= norm;
    }
  }
  int nearestCoil(const Real x, const Real y, const Real z) {
    int retval = -1;
    Real d = 1e10;
    for (int i = 0; i < Ncoil; i++) {
      const Real dtest = (x_coil[i] - x) * (x_coil[i] - x) +
                         (y_coil[i] - y) * (y_coil[i] - y) +
                         (z_coil[i] - z) * (z_coil[i] - z);
      if (dtest < d) {
        retval = i;
        d = dtest;
      }
    }
    return retval;
  }
  ~IC_vorticity() = default;
  void vort(const Real x, const Real y, const Real z, Real &omega_x,
            Real &omega_y, Real &omega_z) {
    const int idx = nearestCoil(x, y, z);
    const Real r2 = (x_coil[idx] - x) * (x_coil[idx] - x) +
                    (y_coil[idx] - y) * (y_coil[idx] - y) +
                    (z_coil[idx] - z) * (z_coil[idx] - z);
    const Real mag = 1.0 / (r2 + 1) / (r2 + 1);
    omega_x = mag * dx_coil[idx];
    omega_y = mag * dy_coil[idx];
    omega_z = mag * dz_coil[idx];
  }
  void run() {
    const int nz = VectorBlock::sizeZ;
    const int ny = VectorBlock::sizeY;
    const int nx = VectorBlock::sizeX;
    std::vector<Info> &velInfo = sim.velInfo();
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      Real p[3];
      VectorBlock &VEL = (*sim.vel)(i);
      for (int iz = 0; iz < nz; ++iz)
        for (int iy = 0; iy < ny; ++iy)
          for (int ix = 0; ix < nx; ++ix) {
            velInfo[i].pos(p, ix, iy, iz);
            vort(p[0], p[1], p[2], VEL(ix, iy, iz).u[0], VEL(ix, iy, iz).u[1],
                 VEL(ix, iy, iz).u[2]);
          }
    }
    {
      ComputeVorticity findOmega(sim);
      findOmega(0);
    }
    std::shared_ptr<PoissonSolverBase> pressureSolver;
    pressureSolver = makePoissonSolver(sim);
    Real PoissonErrorTol = sim.PoissonErrorTol;
    Real PoissonErrorTolRel = sim.PoissonErrorTolRel;
    sim.PoissonErrorTol = 0;
    sim.PoissonErrorTolRel = 0;
    for (int d = 0; d < 3; d++) {
#pragma omp parallel for
      for (size_t i = 0; i < velInfo.size(); i++) {
        const VectorBlock &TMPV = (*sim.tmpV)(i);
        ScalarBlock &PRES = (*sim.pres)(i);
        ScalarBlock &LHS = (*sim.lhs)(i);
        for (int iz = 0; iz < nz; ++iz)
          for (int iy = 0; iy < ny; ++iy)
            for (int ix = 0; ix < nx; ++ix) {
              PRES(ix, iy, iz).s = 0.0;
              LHS(ix, iy, iz).s = -TMPV(ix, iy, iz).u[d];
            }
      }
      pressureSolver->solve();
#pragma omp parallel for
      for (size_t i = 0; i < velInfo.size(); i++) {
        VectorBlock &VEL = (*sim.vel)(i);
        ScalarBlock &PRES = (*sim.pres)(i);
        for (int iz = 0; iz < nz; ++iz)
          for (int iy = 0; iy < ny; ++iy)
            for (int ix = 0; ix < nx; ++ix) {
              VEL(ix, iy, iz).u[d] = PRES(ix, iy, iz).s;
            }
      }
    }
    sim.PoissonErrorTol = PoissonErrorTol;
    sim.PoissonErrorTolRel = PoissonErrorTolRel;
  }
};
} // namespace
static void initialPenalization(SimulationData &sim, const Real dt) {
  const std::vector<Info> &velInfo = sim.velInfo();
  for (const auto &obstacle : sim.obstacle_vector->getObstacleVector()) {
    using CHI_MAT = Real[_BS_][_BS_][_BS_];
    using UDEFMAT = Real[_BS_][_BS_][_BS_][3];
#pragma omp parallel
    {
      const auto &obstblocks = obstacle->getObstacleBlocks();
      const std::array<Real, 3> centerOfMass = obstacle->getCenterOfMass();
      const std::array<Real, 3> uBody = obstacle->getTranslationVelocity();
      const std::array<Real, 3> omegaBody = obstacle->getAngularVelocity();
#pragma omp for schedule(dynamic)
      for (size_t i = 0; i < velInfo.size(); ++i) {
        const Info &info = velInfo[i];
        const auto pos = obstblocks[info.blockID];
        if (pos == nullptr)
          continue;
        VectorBlock &b = (*sim.vel)(i);
        CHI_MAT &__restrict__ CHI = pos->chi;
        UDEFMAT &__restrict__ UDEF = pos->udef;
        for (int iz = 0; iz < VectorBlock::sizeZ; ++iz)
          for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
            for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
              Real p[3];
              info.pos(p, ix, iy, iz);
              p[0] -= centerOfMass[0];
              p[1] -= centerOfMass[1];
              p[2] -= centerOfMass[2];
              const Real object_UR[3] = {
                  (Real)omegaBody[1] * p[2] - (Real)omegaBody[2] * p[1],
                  (Real)omegaBody[2] * p[0] - (Real)omegaBody[0] * p[2],
                  (Real)omegaBody[0] * p[1] - (Real)omegaBody[1] * p[0]};
              const Real U_TOT[3] = {
                  (Real)uBody[0] + object_UR[0] + UDEF[iz][iy][ix][0],
                  (Real)uBody[1] + object_UR[1] + UDEF[iz][iy][ix][1],
                  (Real)uBody[2] + object_UR[2] + UDEF[iz][iy][ix][2]};
              b(ix, iy, iz).u[0] +=
                  CHI[iz][iy][ix] * (U_TOT[0] - b(ix, iy, iz).u[0]);
              b(ix, iy, iz).u[1] +=
                  CHI[iz][iy][ix] * (U_TOT[1] - b(ix, iy, iz).u[1]);
              b(ix, iy, iz).u[2] +=
                  CHI[iz][iy][ix] * (U_TOT[2] - b(ix, iy, iz).u[2]);
            }
      }
    }
  }
}
void InitialConditions::operator()(const Real dt) {
  if (sim.initCond == "zero") {
    run(KernelIC(0));
  }
  if (sim.initCond == "taylorGreen") {
    run(KernelIC_taylorGreen(sim.extents, sim.uMax_forced));
  }
  if (sim.initCond == "vorticity") {
    IC_vorticity ic_vorticity(sim);
    ic_vorticity.run();
  }
  {
    std::vector<Info> &chiInfo = sim.chiInfo();
#pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < chiInfo.size(); i++) {
      ScalarBlock &PRES = (*sim.pres)(i);
      ScalarBlock &LHS = (*sim.lhs)(i);
      VectorBlock &TMPV = (*sim.tmpV)(i);
      for (int iz = 0; iz < ScalarBlock::sizeZ; ++iz)
        for (int iy = 0; iy < ScalarBlock::sizeY; ++iy)
          for (int ix = 0; ix < ScalarBlock::sizeX; ++ix) {
            PRES(ix, iy, iz).s = 0;
            LHS(ix, iy, iz).s = 0;
            TMPV(ix, iy, iz).u[0] = 0;
            TMPV(ix, iy, iz).u[1] = 0;
            TMPV(ix, iy, iz).u[2] = 0;
          }
    }
    initialPenalization(sim, dt);
  }
}
class NacaMidlineData : public FishMidlineData {
  Real *const rK;
  Real *const vK;
  Real *const rC;
  Real *const vC;

public:
  NacaMidlineData(const Real L, const Real _h, Real zExtent, Real t_ratio,
                  Real HoverL = 1)
      : FishMidlineData(L, 1, 0, _h), rK(_alloc(Nm)), vK(_alloc(Nm)),
        rC(_alloc(Nm)), vC(_alloc(Nm)) {
    for (int i = 0; i < Nm; ++i)
      height[i] = length * HoverL / 2;
    MidlineShapes::naca_width(t_ratio, length, rS, width, Nm);
    computeMidline(0.0, 0.0);
  }
  void computeMidline(const Real time, const Real dt) override {
#if 1
    rX[0] = rY[0] = rZ[0] = 0.0;
    vX[0] = vY[0] = vZ[0] = 0.0;
    norX[0] = 0.0;
    norY[0] = 1.0;
    norZ[0] = 0.0;
    binX[0] = 0.0;
    binY[0] = 0.0;
    binZ[0] = 1.0;
    vNorX[0] = vNorY[0] = vNorZ[0] = 0.0;
    vBinX[0] = vBinY[0] = vBinZ[0] = 0.0;
    for (int i = 1; i < Nm; ++i) {
      rY[i] = rZ[i] = 0.0;
      vX[i] = vY[i] = vZ[i] = 0.0;
      rX[i] = rX[i - 1] + std::fabs(rS[i] - rS[i - 1]);
      norX[i] = 0.0;
      norY[i] = 1.0;
      norZ[i] = 0.0;
      binX[i] = 0.0;
      binY[i] = 0.0;
      binZ[i] = 1.0;
      vNorX[i] = vNorY[i] = vNorZ[i] = 0.0;
      vBinX[i] = vBinY[i] = vBinZ[i] = 0.0;
    }
#else
    const std::array<Real, 6> curvature_points = {
        0, .15 * length, .4 * length, .65 * length, .9 * length, length};
    const std::array<Real, 6> curvature_values = {
        0.82014 / length, 1.46515 / length, 2.57136 / length,
        3.75425 / length, 5.09147 / length, 5.70449 / length};
    curvScheduler.transition(time, 0, 1, curvature_values, curvature_values);
    curvScheduler.gimmeValues(time, curvature_points, Nm, rS, rC, vC);
    for (int i = 0; i < Nm; i++) {
      const Real darg = 2. * M_PI;
      const Real arg = 2. * M_PI * (time - rS[i] / length) + M_PI * phaseShift;
      rK[i] = rC[i] * std::sin(arg);
      vK[i] = vC[i] * std::sin(arg) + rC[i] * std::cos(arg) * darg;
    }
    IF2D_Frenet2D::solve(Nm, rS, rK, vK, rX, rY, vX, vY, norX, norY, vNorX,
                         vNorY);
#endif
  }
};
using UDEFMAT = Real[_BS_][_BS_][_BS_][3];
using CHIMAT = Real[_BS_][_BS_][_BS_];
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
Obstacle::Obstacle(SimulationData &s, ArgumentParser &parser) : sim(s) {
  length = parser("-L").asDouble();
  position[0] = parser("-xpos").asDouble();
  position[1] = parser("-ypos").asDouble(sim.extents[1] / 2);
  position[2] = parser("-zpos").asDouble(sim.extents[2] / 2);
  quaternion[0] = parser("-quat0").asDouble(0.0);
  quaternion[1] = parser("-quat1").asDouble(0.0);
  quaternion[2] = parser("-quat2").asDouble(0.0);
  quaternion[3] = parser("-quat3").asDouble(0.0);
  Real planarAngle = parser("-planarAngle").asDouble(0.0) / 180 * M_PI;
  const Real q_length =
      std::sqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
  quaternion[0] /= q_length;
  quaternion[1] /= q_length;
  quaternion[2] /= q_length;
  quaternion[3] /= q_length;
  if (std::fabs(q_length - 1.0) > 100 * EPS) {
    quaternion[0] = std::cos(0.5 * planarAngle);
    quaternion[1] = 0;
    quaternion[2] = 0;
    quaternion[3] = std::sin(0.5 * planarAngle);
  } else {
    if (std::fabs(planarAngle) > 0 && sim.rank == 0)
      std::cout << "WARNING: Obstacle arguments include both quaternions and "
                   "planarAngle."
                << "Quaterion arguments have priority and therefore "
                   "planarAngle will be ignored.\n";
    planarAngle = 2 * std::atan2(quaternion[3], quaternion[0]);
  }
  bool bFSM_alldir = parser("-bForcedInSimFrame").asBool(false);
  bForcedInSimFrame[0] =
      bFSM_alldir || parser("-bForcedInSimFrame_x").asBool(false);
  bForcedInSimFrame[1] =
      bFSM_alldir || parser("-bForcedInSimFrame_y").asBool(false);
  bForcedInSimFrame[2] =
      bFSM_alldir || parser("-bForcedInSimFrame_z").asBool(false);
  Real enforcedVelocity[3];
  enforcedVelocity[0] = -parser("-xvel").asDouble(0.0);
  enforcedVelocity[1] = -parser("-yvel").asDouble(0.0);
  enforcedVelocity[2] = -parser("-zvel").asDouble(0.0);
  const bool bFixToPlanar = parser("-bFixToPlanar").asBool(false);
  bool bFOR_alldir = parser("-bFixFrameOfRef").asBool(false);
  bFixFrameOfRef[0] = bFOR_alldir || parser("-bFixFrameOfRef_x").asBool(false);
  bFixFrameOfRef[1] = bFOR_alldir || parser("-bFixFrameOfRef_y").asBool(false);
  bFixFrameOfRef[2] = bFOR_alldir || parser("-bFixFrameOfRef_z").asBool(false);
  bBreakSymmetry = parser("-bBreakSymmetry").asBool(false);
  absPos[0] = position[0];
  absPos[1] = position[1];
  absPos[2] = position[2];
  const Real one =
      std::sqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
  if (std::fabs(one - 1.0) > 5 * EPS) {
    printf("Parsed quaternion length is not equal to one. It really ought to "
           "be.\n");
    fflush(0);
    abort();
  }
  if (length < 5 * EPS) {
    printf("Parsed length is equal to zero. It really ought not to be.\n");
    fflush(0);
    abort();
  }
  for (int d = 0; d < 3; ++d) {
    bForcedInSimFrame[d] = bForcedInSimFrame[d];
    if (bForcedInSimFrame[d]) {
      transVel_imposed[d] = transVel[d] = enforcedVelocity[d];
      if (!sim.rank)
        printf("Obstacle forced to move relative to sim domain with constant "
               "%c-vel: %f\n",
               "xyz"[d], transVel[d]);
    }
  }
  const bool anyVelForced =
      bForcedInSimFrame[0] || bForcedInSimFrame[1] || bForcedInSimFrame[2];
  if (anyVelForced) {
    if (!sim.rank)
      printf("Obstacle has no angular velocity.\n");
    bBlockRotation[0] = true;
    bBlockRotation[1] = true;
    bBlockRotation[2] = true;
  }
  if (bFixToPlanar) {
    if (!sim.rank)
      printf("Obstacle motion restricted to constant Z-plane.\n");
    bForcedInSimFrame[2] = true;
    transVel_imposed[2] = 0;
    bBlockRotation[1] = true;
    bBlockRotation[0] = true;
  }
  if (bBreakSymmetry)
    if (!sim.rank)
      printf("Symmetry broken by imposing sinusodial y-velocity in t=[1,2].\n");
}
void Obstacle::updateLabVelocity(int nSum[3], Real uSum[3]) {
  if (bFixFrameOfRef[0]) {
    nSum[0] += 1;
    uSum[0] -= transVel[0];
  }
  if (bFixFrameOfRef[1]) {
    nSum[1] += 1;
    uSum[1] -= transVel[1];
  }
  if (bFixFrameOfRef[2]) {
    nSum[2] += 1;
    uSum[2] -= transVel[2];
  }
}
void Obstacle::computeVelocities() {
  std::vector<double> A(36);
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
  double b[6] = {penalLmom[0], penalLmom[1], penalLmom[2],
                 penalAmom[0], penalAmom[1], penalAmom[2]};
  if (bBreakSymmetry) {
    if (sim.time > 3.0 && sim.time < 4.0)
      transVel_imposed[1] = 0.1 * length * std::sin(M_PI * (sim.time - 3.0));
    else
      transVel_imposed[1] = 0.0;
  }
  if (bForcedInSimFrame[0]) {
    A[0 * 6 + 1] = 0;
    A[0 * 6 + 2] = 0;
    A[0 * 6 + 3] = 0;
    A[0 * 6 + 4] = 0;
    A[0 * 6 + 5] = 0;
    b[0] = penalM * transVel_imposed[0];
  }
  if (bForcedInSimFrame[1]) {
    A[1 * 6 + 0] = 0;
    A[1 * 6 + 2] = 0;
    A[1 * 6 + 3] = 0;
    A[1 * 6 + 4] = 0;
    A[1 * 6 + 5] = 0;
    b[1] = penalM * transVel_imposed[1];
  }
  if (bForcedInSimFrame[2]) {
    A[2 * 6 + 0] = 0;
    A[2 * 6 + 1] = 0;
    A[2 * 6 + 3] = 0;
    A[2 * 6 + 4] = 0;
    A[2 * 6 + 5] = 0;
    b[2] = penalM * transVel_imposed[2];
  }
  if (bBlockRotation[0]) {
    A[3 * 6 + 0] = 0;
    A[3 * 6 + 1] = 0;
    A[3 * 6 + 2] = 0;
    A[3 * 6 + 4] = 0;
    A[3 * 6 + 5] = 0;
    b[3] = 0;
  }
  if (bBlockRotation[1]) {
    A[4 * 6 + 0] = 0;
    A[4 * 6 + 1] = 0;
    A[4 * 6 + 2] = 0;
    A[4 * 6 + 3] = 0;
    A[4 * 6 + 5] = 0;
    b[4] = 0;
  }
  if (bBlockRotation[2]) {
    A[5 * 6 + 0] = 0;
    A[5 * 6 + 1] = 0;
    A[5 * 6 + 2] = 0;
    A[5 * 6 + 3] = 0;
    A[5 * 6 + 4] = 0;
    b[5] = 0;
  }
  gsl_matrix_view Agsl = gsl_matrix_view_array(A.data(), 6, 6);
  gsl_vector_view bgsl = gsl_vector_view_array(b, 6);
  gsl_vector *xgsl = gsl_vector_alloc(6);
  int sgsl;
  gsl_permutation *permgsl = gsl_permutation_alloc(6);
  gsl_linalg_LU_decomp(&Agsl.matrix, permgsl, &sgsl);
  gsl_linalg_LU_solve(&Agsl.matrix, permgsl, &bgsl.vector, xgsl);
  transVel_computed[0] = gsl_vector_get(xgsl, 0);
  transVel_computed[1] = gsl_vector_get(xgsl, 1);
  transVel_computed[2] = gsl_vector_get(xgsl, 2);
  angVel_computed[0] = gsl_vector_get(xgsl, 3);
  angVel_computed[1] = gsl_vector_get(xgsl, 4);
  angVel_computed[2] = gsl_vector_get(xgsl, 5);
  gsl_permutation_free(permgsl);
  gsl_vector_free(xgsl);
  force[0] = mass * (transVel_computed[0] - transVel[0]) / sim.dt;
  force[1] = mass * (transVel_computed[1] - transVel[1]) / sim.dt;
  force[2] = mass * (transVel_computed[2] - transVel[2]) / sim.dt;
  const std::array<Real, 3> dAv = {(angVel_computed[0] - angVel[0]) / sim.dt,
                                   (angVel_computed[1] - angVel[1]) / sim.dt,
                                   (angVel_computed[2] - angVel[2]) / sim.dt};
  torque[0] = J[0] * dAv[0] + J[3] * dAv[1] + J[4] * dAv[2];
  torque[1] = J[3] * dAv[0] + J[1] * dAv[1] + J[5] * dAv[2];
  torque[2] = J[4] * dAv[0] + J[5] * dAv[1] + J[2] * dAv[2];
  if (bForcedInSimFrame[0]) {
    assert(std::fabs(transVel[0] - transVel_imposed[0]) < 1e-12);
    transVel[0] = transVel_imposed[0];
  } else
    transVel[0] = transVel_computed[0];
  if (bForcedInSimFrame[1]) {
    assert(std::fabs(transVel[1] - transVel_imposed[1]) < 1e-12);
    transVel[1] = transVel_imposed[1];
  } else
    transVel[1] = transVel_computed[1];
  if (bForcedInSimFrame[2]) {
    assert(std::fabs(transVel[2] - transVel_imposed[2]) < 1e-12);
    transVel[2] = transVel_imposed[2];
  } else
    transVel[2] = transVel_computed[2];
  if (bBlockRotation[0]) {
    assert(std::fabs(angVel[0] - 0) < 1e-12);
    angVel[0] = 0;
  } else
    angVel[0] = angVel_computed[0];
  if (bBlockRotation[1]) {
    assert(std::fabs(angVel[1] - 0) < 1e-12);
    angVel[1] = 0;
  } else
    angVel[1] = angVel_computed[1];
  if (bBlockRotation[2]) {
    assert(std::fabs(angVel[2] - 0) < 1e-12);
    angVel[2] = 0;
  } else
    angVel[2] = angVel_computed[2];
  if (collision_counter > 0) {
    collision_counter -= sim.dt;
    transVel[0] = u_collision;
    transVel[1] = v_collision;
    transVel[2] = w_collision;
    angVel[0] = ox_collision;
    angVel[1] = oy_collision;
    angVel[2] = oz_collision;
  }
}
void Obstacle::computeForces() {
  static const int nQoI = ObstacleBlock::nQoI;
  std::vector<Real> sum = std::vector<Real>(nQoI, 0);
  for (auto &block : obstacleBlocks) {
    if (block == nullptr)
      continue;
    block->sumQoI(sum);
  }
  MPI_Allreduce(MPI_IN_PLACE, sum.data(), nQoI, MPI_Real, MPI_SUM, sim.comm);
  unsigned k = 0;
  surfForce[0] = sum[k++];
  surfForce[1] = sum[k++];
  surfForce[2] = sum[k++];
  presForce[0] = sum[k++];
  presForce[1] = sum[k++];
  presForce[2] = sum[k++];
  viscForce[0] = sum[k++];
  viscForce[1] = sum[k++];
  viscForce[2] = sum[k++];
  surfTorque[0] = sum[k++];
  surfTorque[1] = sum[k++];
  surfTorque[2] = sum[k++];
  drag = sum[k++];
  thrust = sum[k++];
  Pout = sum[k++];
  PoutBnd = sum[k++];
  defPower = sum[k++];
  defPowerBnd = sum[k++];
  pLocom = sum[k++];
  const Real vel_norm =
      std::sqrt(transVel[0] * transVel[0] + transVel[1] * transVel[1] +
                transVel[2] * transVel[2]);
  Pthrust = thrust * vel_norm;
  Pdrag = drag * vel_norm;
  EffPDef = Pthrust / (Pthrust - std::min(defPower, (Real)0) + EPS);
  EffPDefBnd = Pthrust / (Pthrust - defPowerBnd + EPS);
}
void Obstacle::update() {
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
    old_position[0] = position[0];
    old_position[1] = position[1];
    old_position[2] = position[2];
    old_absPos[0] = absPos[0];
    old_absPos[1] = absPos[1];
    old_absPos[2] = absPos[2];
    old_quaternion[0] = quaternion[0];
    old_quaternion[1] = quaternion[1];
    old_quaternion[2] = quaternion[2];
    old_quaternion[3] = quaternion[3];
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
    position[0] =
        aux * (sim.dt * (transVel[0] + sim.uinf[0]) +
               (-sim.coefU[1] * position[0] - sim.coefU[2] * old_position[0]));
    position[1] =
        aux * (sim.dt * (transVel[1] + sim.uinf[1]) +
               (-sim.coefU[1] * position[1] - sim.coefU[2] * old_position[1]));
    position[2] =
        aux * (sim.dt * (transVel[2] + sim.uinf[2]) +
               (-sim.coefU[1] * position[2] - sim.coefU[2] * old_position[2]));
    absPos[0] = aux * (sim.dt * (transVel[0]) + (-sim.coefU[1] * absPos[0] -
                                                 sim.coefU[2] * old_absPos[0]));
    absPos[1] = aux * (sim.dt * (transVel[1]) + (-sim.coefU[1] * absPos[1] -
                                                 sim.coefU[2] * old_absPos[1]));
    absPos[2] = aux * (sim.dt * (transVel[2]) + (-sim.coefU[1] * absPos[2] -
                                                 sim.coefU[2] * old_absPos[2]));
    quaternion[0] =
        aux * (sim.dt * (dqdt[0]) + (-sim.coefU[1] * quaternion[0] -
                                     sim.coefU[2] * old_quaternion[0]));
    quaternion[1] =
        aux * (sim.dt * (dqdt[1]) + (-sim.coefU[1] * quaternion[1] -
                                     sim.coefU[2] * old_quaternion[1]));
    quaternion[2] =
        aux * (sim.dt * (dqdt[2]) + (-sim.coefU[1] * quaternion[2] -
                                     sim.coefU[2] * old_quaternion[2]));
    quaternion[3] =
        aux * (sim.dt * (dqdt[3]) + (-sim.coefU[1] * quaternion[3] -
                                     sim.coefU[2] * old_quaternion[3]));
    old_position[0] = temp[0];
    old_position[1] = temp[1];
    old_position[2] = temp[2];
    old_absPos[0] = temp[3];
    old_absPos[1] = temp[4];
    old_absPos[2] = temp[5];
    old_quaternion[0] = temp[6];
    old_quaternion[1] = temp[7];
    old_quaternion[2] = temp[8];
    old_quaternion[3] = temp[9];
  }
  const Real invD =
      1.0 /
      std::sqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
  quaternion[0] *= invD;
  quaternion[1] *= invD;
  quaternion[2] *= invD;
  quaternion[3] *= invD;
#ifndef NDEBUG
  const Real q_length =
      std::sqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
  assert(std::abs(q_length - 1.0) < 5 * EPS);
#endif
}
void Obstacle::create() {
  printf("Entered the wrong create operator\n");
  fflush(0);
  exit(1);
}
void Obstacle::finalize() {}
std::array<Real, 3> Obstacle::getTranslationVelocity() const {
  return std::array<Real, 3>{{transVel[0], transVel[1], transVel[2]}};
}
std::array<Real, 3> Obstacle::getAngularVelocity() const {
  return std::array<Real, 3>{{angVel[0], angVel[1], angVel[2]}};
}
std::array<Real, 3> Obstacle::getCenterOfMass() const {
  return std::array<Real, 3>{
      {centerOfMass[0], centerOfMass[1], centerOfMass[2]}};
}
std::array<Real, 3> Obstacle::getYawPitchRoll() const {
  const Real roll = atan2(
      2.0 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1]),
      1.0 - 2.0 * (quaternion[1] * quaternion[1] +
                   quaternion[2] * quaternion[2]));
  const Real pitch = asin(
      2.0 * (quaternion[2] * quaternion[0] - quaternion[3] * quaternion[1]));
  const Real yaw = atan2(
      2.0 * (quaternion[3] * quaternion[0] + quaternion[1] * quaternion[2]),
      -1.0 + 2.0 * (quaternion[0] * quaternion[0] +
                    quaternion[1] * quaternion[1]));
  return std::array<Real, 3>{{yaw, pitch, roll}};
}
using VectorType = ObstacleVector::VectorType;
static std::shared_ptr<Obstacle>
_createObstacle(SimulationData &sim, const std::string &objectName,
                FactoryFileLineParser &lineParser) {
  if (objectName == "StefanFish" || objectName == "stefanfish")
    return std::make_shared<StefanFish>(sim, lineParser);
  else if (sim.rank == 0) {
    std::cout << "[CUP3D] Case " << objectName << " is not defined: aborting\n"
              << std::flush;
    abort();
  }
  return {};
}
static void _addObstacles(SimulationData &sim, std::stringstream &stream) {
  std::vector<std::pair<std::string, FactoryFileLineParser>> factoryLines;
  std::string line;
  while (std::getline(stream, line)) {
    std::istringstream line_stream(line);
    std::string ID;
    line_stream >> ID;
    if (ID.empty() || ID[0] == '#')
      continue;
    factoryLines.emplace_back(ID, FactoryFileLineParser(line_stream));
  }
  if (factoryLines.empty()) {
    if (sim.rank == 0)
      std::cout << "[CUP3D] OBSTACLE FACTORY did not create any obstacles.\n";
    return;
  }
  for (auto &l : factoryLines) {
    sim.obstacle_vector->addObstacle(_createObstacle(sim, l.first, l.second));
  }
}
void ObstacleFactory::addObstacles(ArgumentParser &parser) {
  parser.unset_strict_mode();
  const std::string factory_filename = parser("-factory").asString("factory");
  std::string factory_content = parser("-factory-content").asString("");
  if (factory_content.compare("") == 0)
    factory_content = parser("-shapes").asString("");
  std::stringstream stream(factory_content);
  if (!factory_filename.empty()) {
    std::ifstream file(factory_filename);
    if (file.is_open()) {
      stream << '\n';
      stream << file.rdbuf();
    }
  }
  _addObstacles(sim, stream);
}
void ObstacleFactory::addObstacles(const std::string &factoryContent) {
  std::stringstream stream(factoryContent);
  _addObstacles(sim, stream);
}
namespace {
using CHIMAT = Real[_BS_][_BS_][_BS_];
using UDEFMAT = Real[_BS_][_BS_][_BS_][3];
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
struct KernelCharacteristicFunction {
  using v_v_ob = std::vector<std::vector<ObstacleBlock *> *>;
  const v_v_ob &vec_obstacleBlocks;
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  KernelCharacteristicFunction(const v_v_ob &v) : vec_obstacleBlocks(v) {}
  void operate(const Info &info, ScalarBlock &b) const {
    const Real h = info.h, inv2h = .5 / h, fac1 = .5 * h * h, vol = h * h * h;
    const int gp = 1;
    for (size_t obst_id = 0; obst_id < vec_obstacleBlocks.size(); obst_id++) {
      const auto &obstacleBlocks = *vec_obstacleBlocks[obst_id];
      ObstacleBlock *const o = obstacleBlocks[info.blockID];
      if (o == nullptr)
        continue;
      CHIMAT &__restrict__ CHI = o->chi;
      o->CoM_x = 0;
      o->CoM_y = 0;
      o->CoM_z = 0;
      o->mass = 0;
      const auto &SDFLAB = o->sdfLab;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x) {
#if 1
            if (SDFLAB[z + 1][y + 1][x + 1] > +gp * h ||
                SDFLAB[z + 1][y + 1][x + 1] < -gp * h) {
              CHI[z][y][x] = SDFLAB[z + 1][y + 1][x + 1] > 0 ? 1 : 0;
            } else {
              const Real distPx = SDFLAB[z + 1][y + 1][x + 1 + 1];
              const Real distMx = SDFLAB[z + 1][y + 1][x + 1 - 1];
              const Real distPy = SDFLAB[z + 1][y + 1 + 1][x + 1];
              const Real distMy = SDFLAB[z + 1][y + 1 - 1][x + 1];
              const Real distPz = SDFLAB[z + 1 + 1][y + 1][x + 1];
              const Real distMz = SDFLAB[z + 1 - 1][y + 1][x + 1];
              const Real gradUX = inv2h * (distPx - distMx);
              const Real gradUY = inv2h * (distPy - distMy);
              const Real gradUZ = inv2h * (distPz - distMz);
              const Real gradUSq =
                  gradUX * gradUX + gradUY * gradUY + gradUZ * gradUZ + EPS;
              const Real IplusX = std::max((Real)0.0, distPx);
              const Real IminuX = std::max((Real)0.0, distMx);
              const Real IplusY = std::max((Real)0.0, distPy);
              const Real IminuY = std::max((Real)0.0, distMy);
              const Real IplusZ = std::max((Real)0.0, distPz);
              const Real IminuZ = std::max((Real)0.0, distMz);
              const Real gradIX = inv2h * (IplusX - IminuX);
              const Real gradIY = inv2h * (IplusY - IminuY);
              const Real gradIZ = inv2h * (IplusZ - IminuZ);
              const Real numH =
                  gradIX * gradUX + gradIY * gradUY + gradIZ * gradUZ;
              CHI[z][y][x] = numH / gradUSq;
            }
#else
            CHI[z][y][x] = SDFLAB[z + 1][y + 1][x + 1] > 0 ? 1 : 0;
#endif
            Real p[3];
            info.pos(p, x, y, z);
            b(x, y, z).s = std::max(CHI[z][y][x], b(x, y, z).s);
            o->CoM_x += CHI[z][y][x] * vol * p[0];
            o->CoM_y += CHI[z][y][x] * vol * p[1];
            o->CoM_z += CHI[z][y][x] * vol * p[2];
            o->mass += CHI[z][y][x] * vol;
          }
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x) {
            const Real distPx = SDFLAB[z + 1][y + 1][x + 1 + 1];
            const Real distMx = SDFLAB[z + 1][y + 1][x + 1 - 1];
            const Real distPy = SDFLAB[z + 1][y + 1 + 1][x + 1];
            const Real distMy = SDFLAB[z + 1][y + 1 - 1][x + 1];
            const Real distPz = SDFLAB[z + 1 + 1][y + 1][x + 1];
            const Real distMz = SDFLAB[z + 1 - 1][y + 1][x + 1];
            const Real gradUX = inv2h * (distPx - distMx);
            const Real gradUY = inv2h * (distPy - distMy);
            const Real gradUZ = inv2h * (distPz - distMz);
            const Real gradUSq =
                gradUX * gradUX + gradUY * gradUY + gradUZ * gradUZ + EPS;
            const Real gradHX =
                (x == 0)
                    ? 2.0 * (-0.5 * CHI[z][y][x + 2] + 2.0 * CHI[z][y][x + 1] -
                             1.5 * CHI[z][y][x])
                    : ((x == Nx - 1) ? 2.0 * (1.5 * CHI[z][y][x] -
                                              2.0 * CHI[z][y][x - 1] +
                                              0.5 * CHI[z][y][x - 2])
                                     : (CHI[z][y][x + 1] - CHI[z][y][x - 1]));
            const Real gradHY =
                (y == 0)
                    ? 2.0 * (-0.5 * CHI[z][y + 2][x] + 2.0 * CHI[z][y + 1][x] -
                             1.5 * CHI[z][y][x])
                    : ((y == Ny - 1) ? 2.0 * (1.5 * CHI[z][y][x] -
                                              2.0 * CHI[z][y - 1][x] +
                                              0.5 * CHI[z][y - 2][x])
                                     : (CHI[z][y + 1][x] - CHI[z][y - 1][x]));
            const Real gradHZ =
                (z == 0)
                    ? 2.0 * (-0.5 * CHI[z + 2][y][x] + 2.0 * CHI[z + 1][y][x] -
                             1.5 * CHI[z][y][x])
                    : ((z == Nz - 1) ? 2.0 * (1.5 * CHI[z][y][x] -
                                              2.0 * CHI[z - 1][y][x] +
                                              0.5 * CHI[z - 2][y][x])
                                     : (CHI[z + 1][y][x] - CHI[z - 1][y][x]));
            if (gradHX * gradHX + gradHY * gradHY + gradHZ * gradHZ < 1e-12)
              continue;
            const Real numD =
                gradHX * gradUX + gradHY * gradUY + gradHZ * gradUZ;
            const Real Delta = fac1 * numD / gradUSq;
            if (Delta > EPS)
              o->write(x, y, z, Delta, gradUX, gradUY, gradUZ);
          }
      o->allocate_surface();
    }
  }
};
} // namespace
static void kernelComputeGridCoM(SimulationData &sim) {
  for (const auto &obstacle : sim.obstacle_vector->getObstacleVector()) {
    Real com[4] = {0.0, 0.0, 0.0, 0.0};
    const auto &obstblocks = obstacle->getObstacleBlocks();
#pragma omp parallel for schedule(static, 1) reduction(+ : com[:4])
    for (size_t i = 0; i < obstblocks.size(); i++) {
      if (obstblocks[i] == nullptr)
        continue;
      com[0] += obstblocks[i]->mass;
      com[1] += obstblocks[i]->CoM_x;
      com[2] += obstblocks[i]->CoM_y;
      com[3] += obstblocks[i]->CoM_z;
    }
    MPI_Allreduce(MPI_IN_PLACE, com, 4, MPI_Real, MPI_SUM, sim.comm);
    assert(com[0] > std::numeric_limits<Real>::epsilon());
    obstacle->centerOfMass[0] = com[1] / com[0];
    obstacle->centerOfMass[1] = com[2] / com[0];
    obstacle->centerOfMass[2] = com[3] / com[0];
  }
}
static void _kernelIntegrateUdefMomenta(SimulationData &sim, const Info &info) {
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  for (const auto &obstacle : sim.obstacle_vector->getObstacleVector()) {
    const auto &obstblocks = obstacle->getObstacleBlocks();
    ObstacleBlock *const o = obstblocks[info.blockID];
    if (o == nullptr)
      continue;
    const std::array<Real, 3> CM = obstacle->getCenterOfMass();
    const std::array<Real, 3> oldCorrVel = {{obstacle->transVel_correction[0],
                                             obstacle->transVel_correction[1],
                                             obstacle->transVel_correction[2]}};
    const CHIMAT &__restrict__ CHI = o->chi;
    const UDEFMAT &__restrict__ UDEF = o->udef;
    Real &VV = o->V;
    Real &FX = o->FX, &FY = o->FY, &FZ = o->FZ;
    Real &TX = o->TX, &TY = o->TY, &TZ = o->TZ;
    Real &J0 = o->J0, &J1 = o->J1, &J2 = o->J2;
    Real &J3 = o->J3, &J4 = o->J4, &J5 = o->J5;
    VV = 0;
    FX = 0;
    FY = 0;
    FZ = 0;
    TX = 0;
    TY = 0;
    TZ = 0;
    J0 = 0;
    J1 = 0;
    J2 = 0;
    J3 = 0;
    J4 = 0;
    J5 = 0;
    for (int z = 0; z < Nz; ++z)
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          if (CHI[z][y][x] <= 0)
            continue;
          Real p[3];
          info.pos(p, x, y, z);
          const Real dv = info.h * info.h * info.h, X = CHI[z][y][x];
          p[0] -= CM[0];
          p[1] -= CM[1];
          p[2] -= CM[2];
          const Real dUs = UDEF[z][y][x][0] - oldCorrVel[0];
          const Real dVs = UDEF[z][y][x][1] - oldCorrVel[1];
          const Real dWs = UDEF[z][y][x][2] - oldCorrVel[2];
          VV += X * dv;
          FX += X * UDEF[z][y][x][0] * dv;
          FY += X * UDEF[z][y][x][1] * dv;
          FZ += X * UDEF[z][y][x][2] * dv;
          TX += X * (p[1] * dWs - p[2] * dVs) * dv;
          TY += X * (p[2] * dUs - p[0] * dWs) * dv;
          TZ += X * (p[0] * dVs - p[1] * dUs) * dv;
          J0 += X * (p[1] * p[1] + p[2] * p[2]) * dv;
          J3 -= X * p[0] * p[1] * dv;
          J1 += X * (p[0] * p[0] + p[2] * p[2]) * dv;
          J4 -= X * p[0] * p[2] * dv;
          J2 += X * (p[0] * p[0] + p[1] * p[1]) * dv;
          J5 -= X * p[1] * p[2] * dv;
        }
  }
}
static void kernelIntegrateUdefMomenta(SimulationData &sim) {
  const std::vector<Info> &chiInfo = sim.chiInfo();
#pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < chiInfo.size(); ++i)
    _kernelIntegrateUdefMomenta(sim, chiInfo[i]);
}
static void kernelAccumulateUdefMomenta(SimulationData &sim,
                                        bool justDebug = false) {
  for (const auto &obst : sim.obstacle_vector->getObstacleVector()) {
    Real M[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const auto &oBlock = obst->getObstacleBlocks();
#pragma omp parallel for schedule(static, 1) reduction(+ : M[:13])
    for (size_t i = 0; i < oBlock.size(); i++) {
      if (oBlock[i] == nullptr)
        continue;
      M[0] += oBlock[i]->V;
      M[1] += oBlock[i]->FX;
      M[2] += oBlock[i]->FY;
      M[3] += oBlock[i]->FZ;
      M[4] += oBlock[i]->TX;
      M[5] += oBlock[i]->TY;
      M[6] += oBlock[i]->TZ;
      M[7] += oBlock[i]->J0;
      M[8] += oBlock[i]->J1;
      M[9] += oBlock[i]->J2;
      M[10] += oBlock[i]->J3;
      M[11] += oBlock[i]->J4;
      M[12] += oBlock[i]->J5;
    }
    const auto comm = sim.comm;
    MPI_Allreduce(MPI_IN_PLACE, M, 13, MPI_Real, MPI_SUM, comm);
    assert(M[0] > EPS);
    const GenV AM = {{M[4], M[5], M[6]}};
    const SymM J = {{M[7], M[8], M[9], M[10], M[11], M[12]}};
    const SymM invJ = invertSym(J);
    if (justDebug) {
      assert(std::fabs(M[1]) < 100 * EPS);
      assert(std::fabs(M[2]) < 100 * EPS);
      assert(std::fabs(M[3]) < 100 * EPS);
      assert(std::fabs(AM[0]) < 100 * EPS);
      assert(std::fabs(AM[1]) < 100 * EPS);
      assert(std::fabs(AM[2]) < 100 * EPS);
    } else {
      obst->mass = M[0];
      obst->transVel_correction[0] = M[1] / M[0];
      obst->transVel_correction[1] = M[2] / M[0];
      obst->transVel_correction[2] = M[3] / M[0];
      obst->J[0] = M[7];
      obst->J[1] = M[8];
      obst->J[2] = M[9];
      obst->J[3] = M[10];
      obst->J[4] = M[11];
      obst->J[5] = M[12];
      obst->angVel_correction[0] =
          invJ[0] * AM[0] + invJ[3] * AM[1] + invJ[4] * AM[2];
      obst->angVel_correction[1] =
          invJ[3] * AM[0] + invJ[1] * AM[1] + invJ[5] * AM[2];
      obst->angVel_correction[2] =
          invJ[4] * AM[0] + invJ[5] * AM[1] + invJ[2] * AM[2];
    }
  }
}
static void kernelRemoveUdefMomenta(SimulationData &sim,
                                    bool justDebug = false) {
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  const std::vector<Info> &chiInfo = sim.chiInfo();
  for (const auto &obstacle : sim.obstacle_vector->getObstacleVector()) {
    const std::array<Real, 3> angVel_correction = obstacle->angVel_correction;
    const std::array<Real, 3> transVel_correction =
        obstacle->transVel_correction;
    const std::array<Real, 3> CM = obstacle->getCenterOfMass();
    const auto &obstacleBlocks = obstacle->getObstacleBlocks();
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < chiInfo.size(); i++) {
      const Info &info = chiInfo[i];
      const auto pos = obstacleBlocks[info.blockID];
      if (pos == nullptr)
        continue;
      UDEFMAT &__restrict__ UDEF = pos->udef;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x) {
            Real p[3];
            info.pos(p, x, y, z);
            p[0] -= CM[0];
            p[1] -= CM[1];
            p[2] -= CM[2];
            const Real rotVel_correction[3] = {
                angVel_correction[1] * p[2] - angVel_correction[2] * p[1],
                angVel_correction[2] * p[0] - angVel_correction[0] * p[2],
                angVel_correction[0] * p[1] - angVel_correction[1] * p[0]};
            UDEF[z][y][x][0] -= transVel_correction[0] + rotVel_correction[0];
            UDEF[z][y][x][1] -= transVel_correction[1] + rotVel_correction[1];
            UDEF[z][y][x][2] -= transVel_correction[2] + rotVel_correction[2];
          }
    }
  }
}
void CreateObstacles::operator()(const Real dt) {
  if (sim.obstacle_vector->nObstacles() == 0)
    return;
  if (sim.MeshChanged == false && sim.StaticObstacles)
    return;
  sim.MeshChanged = false;
  std::vector<Info> &chiInfo = sim.chiInfo();
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < chiInfo.size(); ++i) {
    ScalarBlock &CHI = *(ScalarBlock *)chiInfo[i].block;
    CHI.clear();
  }
  sim.uinf = sim.obstacle_vector->updateUinf();
  sim.obstacle_vector->update();
  { sim.obstacle_vector->create(); }
  {
#pragma omp parallel
    {
      auto vecOB = sim.obstacle_vector->getAllObstacleBlocks();
      const KernelCharacteristicFunction K(vecOB);
#pragma omp for
      for (size_t i = 0; i < chiInfo.size(); ++i) {
        ScalarBlock &CHI = *(ScalarBlock *)chiInfo[i].block;
        K.operate(chiInfo[i], CHI);
      }
    }
  }
  kernelComputeGridCoM(sim);
  kernelIntegrateUdefMomenta(sim);
  kernelAccumulateUdefMomenta(sim);
  kernelRemoveUdefMomenta(sim);
  sim.obstacle_vector->finalize();
}
namespace {
using CHIMAT = Real[_BS_][_BS_][_BS_];
using UDEFMAT = Real[_BS_][_BS_][_BS_][3];
template <bool implicitPenalization> struct KernelIntegrateFluidMomenta {
  const Real lambda, dt;
  ObstacleVector *const obstacle_vector;
  Real dvol(const Info &I, const int x, const int y, const int z) const {
    return I.h * I.h * I.h;
  }
  KernelIntegrateFluidMomenta(Real _dt, Real _lambda, ObstacleVector *ov)
      : lambda(_lambda), dt(_dt), obstacle_vector(ov) {}
  void operator()(const Info &info) const {
    for (const auto &obstacle : obstacle_vector->getObstacleVector())
      visit(info, obstacle.get());
  }
  void visit(const Info &info, Obstacle *const op) const {
    const std::vector<ObstacleBlock *> &obstblocks = op->getObstacleBlocks();
    ObstacleBlock *const o = obstblocks[info.blockID];
    if (o == nullptr)
      return;
    const std::array<Real, 3> CM = op->getCenterOfMass();
    const VectorBlock &b = *(VectorBlock *)info.block;
    const CHIMAT &__restrict__ CHI = o->chi;
    Real &VV = o->V;
    Real &FX = o->FX, &FY = o->FY, &FZ = o->FZ;
    Real &TX = o->TX, &TY = o->TY, &TZ = o->TZ;
    VV = 0;
    FX = 0;
    FY = 0;
    FZ = 0;
    TX = 0;
    TY = 0;
    TZ = 0;
    Real &J0 = o->J0, &J1 = o->J1, &J2 = o->J2;
    Real &J3 = o->J3, &J4 = o->J4, &J5 = o->J5;
    J0 = 0;
    J1 = 0;
    J2 = 0;
    J3 = 0;
    J4 = 0;
    J5 = 0;
    const UDEFMAT &__restrict__ UDEF = o->udef;
    const Real lambdt = lambda * dt;
    if (implicitPenalization) {
      o->GfX = 0;
      o->GpX = 0;
      o->GpY = 0;
      o->GpZ = 0;
      o->Gj0 = 0;
      o->Gj1 = 0;
      o->Gj2 = 0;
      o->Gj3 = 0;
      o->Gj4 = 0;
      o->Gj5 = 0;
      o->GuX = 0;
      o->GuY = 0;
      o->GuZ = 0;
      o->GaX = 0;
      o->GaY = 0;
      o->GaZ = 0;
    }
    for (int iz = 0; iz < VectorBlock::sizeZ; ++iz)
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
          if (CHI[iz][iy][ix] <= 0)
            continue;
          Real p[3];
          info.pos(p, ix, iy, iz);
          const Real dv = dvol(info, ix, iy, iz), X = CHI[iz][iy][ix];
          p[0] -= CM[0];
          p[1] -= CM[1];
          p[2] -= CM[2];
          VV += X * dv;
          J0 += X * dv * (p[1] * p[1] + p[2] * p[2]);
          J1 += X * dv * (p[0] * p[0] + p[2] * p[2]);
          J2 += X * dv * (p[0] * p[0] + p[1] * p[1]);
          J3 -= X * dv * p[0] * p[1];
          J4 -= X * dv * p[0] * p[2];
          J5 -= X * dv * p[1] * p[2];
          FX += X * dv * b(ix, iy, iz).u[0];
          FY += X * dv * b(ix, iy, iz).u[1];
          FZ += X * dv * b(ix, iy, iz).u[2];
          TX +=
              X * dv * (p[1] * b(ix, iy, iz).u[2] - p[2] * b(ix, iy, iz).u[1]);
          TY +=
              X * dv * (p[2] * b(ix, iy, iz).u[0] - p[0] * b(ix, iy, iz).u[2]);
          TZ +=
              X * dv * (p[0] * b(ix, iy, iz).u[1] - p[1] * b(ix, iy, iz).u[0]);
          if (implicitPenalization) {
            const Real X1 = CHI[iz][iy][ix] > 0.5 ? 1.0 : 0.0;
            const Real penalFac = dv * lambdt * X1 / (1 + X1 * lambdt);
            o->GfX += penalFac;
            o->GpX += penalFac * p[0];
            o->GpY += penalFac * p[1];
            o->GpZ += penalFac * p[2];
            o->Gj0 += penalFac * (p[1] * p[1] + p[2] * p[2]);
            o->Gj1 += penalFac * (p[0] * p[0] + p[2] * p[2]);
            o->Gj2 += penalFac * (p[0] * p[0] + p[1] * p[1]);
            o->Gj3 -= penalFac * p[0] * p[1];
            o->Gj4 -= penalFac * p[0] * p[2];
            o->Gj5 -= penalFac * p[1] * p[2];
            const Real DiffU[3] = {b(ix, iy, iz).u[0] - UDEF[iz][iy][ix][0],
                                   b(ix, iy, iz).u[1] - UDEF[iz][iy][ix][1],
                                   b(ix, iy, iz).u[2] - UDEF[iz][iy][ix][2]};
            o->GuX += penalFac * DiffU[0];
            o->GuY += penalFac * DiffU[1];
            o->GuZ += penalFac * DiffU[2];
            o->GaX += penalFac * (p[1] * DiffU[2] - p[2] * DiffU[1]);
            o->GaY += penalFac * (p[2] * DiffU[0] - p[0] * DiffU[2]);
            o->GaZ += penalFac * (p[0] * DiffU[1] - p[1] * DiffU[0]);
          }
        }
  }
};
} // namespace
template <bool implicitPenalization>
static void kernelFinalizeObstacleVel(SimulationData &sim, const Real dt) {
  for (const auto &obst : sim.obstacle_vector->getObstacleVector()) {
    static constexpr int nQoI = 29;
    Real M[nQoI] = {0};
    const auto &oBlock = obst->getObstacleBlocks();
#pragma omp parallel for schedule(static, 1) reduction(+ : M[:nQoI])
    for (size_t i = 0; i < oBlock.size(); i++) {
      if (oBlock[i] == nullptr)
        continue;
      int k = 0;
      M[k++] += oBlock[i]->V;
      M[k++] += oBlock[i]->FX;
      M[k++] += oBlock[i]->FY;
      M[k++] += oBlock[i]->FZ;
      M[k++] += oBlock[i]->TX;
      M[k++] += oBlock[i]->TY;
      M[k++] += oBlock[i]->TZ;
      M[k++] += oBlock[i]->J0;
      M[k++] += oBlock[i]->J1;
      M[k++] += oBlock[i]->J2;
      M[k++] += oBlock[i]->J3;
      M[k++] += oBlock[i]->J4;
      M[k++] += oBlock[i]->J5;
      if (implicitPenalization) {
        M[k++] += oBlock[i]->GfX;
        M[k++] += oBlock[i]->GpX;
        M[k++] += oBlock[i]->GpY;
        M[k++] += oBlock[i]->GpZ;
        M[k++] += oBlock[i]->Gj0;
        M[k++] += oBlock[i]->Gj1;
        M[k++] += oBlock[i]->Gj2;
        M[k++] += oBlock[i]->Gj3;
        M[k++] += oBlock[i]->Gj4;
        M[k++] += oBlock[i]->Gj5;
        M[k++] += oBlock[i]->GuX;
        M[k++] += oBlock[i]->GuY;
        M[k++] += oBlock[i]->GuZ;
        M[k++] += oBlock[i]->GaX;
        M[k++] += oBlock[i]->GaY;
        M[k++] += oBlock[i]->GaZ;
        assert(k == 29);
      } else
        assert(k == 13);
    }
    const auto comm = sim.comm;
    MPI_Allreduce(MPI_IN_PLACE, M, nQoI, MPI_Real, MPI_SUM, comm);
#ifndef NDEBUG
    const Real J_magnitude = obst->J[0] + obst->J[1] + obst->J[2];
    static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
#endif
    assert(std::fabs(obst->mass - M[0]) < 10 * EPS * obst->mass);
    assert(std::fabs(obst->J[0] - M[7]) < 10 * EPS * J_magnitude);
    assert(std::fabs(obst->J[1] - M[8]) < 10 * EPS * J_magnitude);
    assert(std::fabs(obst->J[2] - M[9]) < 10 * EPS * J_magnitude);
    assert(std::fabs(obst->J[3] - M[10]) < 10 * EPS * J_magnitude);
    assert(std::fabs(obst->J[4] - M[11]) < 10 * EPS * J_magnitude);
    assert(std::fabs(obst->J[5] - M[12]) < 10 * EPS * J_magnitude);
    assert(M[0] > EPS);
    if (implicitPenalization) {
      obst->penalM = M[13];
      obst->penalCM = {M[14], M[15], M[16]};
      obst->penalJ = {M[17], M[18], M[19], M[20], M[21], M[22]};
      obst->penalLmom = {M[23], M[24], M[25]};
      obst->penalAmom = {M[26], M[27], M[28]};
    } else {
      obst->penalM = M[0];
      obst->penalCM = {0, 0, 0};
      obst->penalJ = {M[7], M[8], M[9], M[10], M[11], M[12]};
      obst->penalLmom = {M[1], M[2], M[3]};
      obst->penalAmom = {M[4], M[5], M[6]};
    }
    obst->computeVelocities();
  }
}
void UpdateObstacles::operator()(const Real dt) {
  if (sim.obstacle_vector->nObstacles() == 0)
    return;
  {
    std::vector<Info> &velInfo = sim.velInfo();
#pragma omp parallel
    {
      if (sim.bImplicitPenalization) {
        KernelIntegrateFluidMomenta<1> K(dt, sim.lambda, sim.obstacle_vector);
#pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < velInfo.size(); ++i)
          K(velInfo[i]);
      } else {
        KernelIntegrateFluidMomenta<0> K(dt, sim.lambda, sim.obstacle_vector);
#pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < velInfo.size(); ++i)
          K(velInfo[i]);
      }
    }
  }
  if (sim.bImplicitPenalization) {
    kernelFinalizeObstacleVel<1>(sim, dt);
  } else {
    kernelFinalizeObstacleVel<0>(sim, dt);
  }
}
namespace {
using CHIMAT = Real[_BS_][_BS_][_BS_];
using UDEFMAT = Real[_BS_][_BS_][_BS_][3];
struct KernelPenalization {
  const Real dt, invdt = 1.0 / dt, lambda;
  const bool implicitPenalization;
  ObstacleVector *const obstacle_vector;
  KernelPenalization(const Real _dt, const Real _lambda,
                     const bool _implicitPenalization, ObstacleVector *ov)
      : dt(_dt), lambda(_lambda), implicitPenalization(_implicitPenalization),
        obstacle_vector(ov) {}
  void operator()(const Info &info, const Info &ChiInfo) const {
    for (const auto &obstacle : obstacle_vector->getObstacleVector())
      visit(info, ChiInfo, obstacle.get());
  }
  void visit(const Info &info, const Info &ChiInfo,
             Obstacle *const obstacle) const {
    const auto &obstblocks = obstacle->getObstacleBlocks();
    ObstacleBlock *const o = obstblocks[info.blockID];
    if (o == nullptr)
      return;
    const CHIMAT &__restrict__ CHI = o->chi;
    const UDEFMAT &__restrict__ UDEF = o->udef;
    VectorBlock &b = *(VectorBlock *)info.block;
    ScalarBlock &bChi = *(ScalarBlock *)ChiInfo.block;
    const std::array<Real, 3> CM = obstacle->getCenterOfMass();
    const std::array<Real, 3> vel = obstacle->getTranslationVelocity();
    const std::array<Real, 3> omega = obstacle->getAngularVelocity();
    const Real dv = std::pow(info.h, 3);
    const Real lambdaFac = implicitPenalization ? lambda : invdt;
    Real &FX = o->FX, &FY = o->FY, &FZ = o->FZ;
    Real &TX = o->TX, &TY = o->TY, &TZ = o->TZ;
    FX = 0;
    FY = 0;
    FZ = 0;
    TX = 0;
    TY = 0;
    TZ = 0;
    for (int iz = 0; iz < VectorBlock::sizeZ; ++iz)
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
          if (bChi(ix, iy, iz).s > CHI[iz][iy][ix])
            continue;
          if (CHI[iz][iy][ix] <= 0)
            continue;
          Real p[3];
          info.pos(p, ix, iy, iz);
          p[0] -= CM[0];
          p[1] -= CM[1];
          p[2] -= CM[2];
          const Real U_TOT[3] = {
              vel[0] + omega[1] * p[2] - omega[2] * p[1] + UDEF[iz][iy][ix][0],
              vel[1] + omega[2] * p[0] - omega[0] * p[2] + UDEF[iz][iy][ix][1],
              vel[2] + omega[0] * p[1] - omega[1] * p[0] + UDEF[iz][iy][ix][2]};
          const Real X = implicitPenalization
                             ? (CHI[iz][iy][ix] > 0.5 ? 1.0 : 0.0)
                             : CHI[iz][iy][ix];
          const Real penalFac = implicitPenalization
                                    ? X * lambdaFac / (1 + X * lambdaFac * dt)
                                    : X * lambdaFac;
          const Real FPX = penalFac * (U_TOT[0] - b(ix, iy, iz).u[0]);
          const Real FPY = penalFac * (U_TOT[1] - b(ix, iy, iz).u[1]);
          const Real FPZ = penalFac * (U_TOT[2] - b(ix, iy, iz).u[2]);
          b(ix, iy, iz).u[0] = b(ix, iy, iz).u[0] + dt * FPX;
          b(ix, iy, iz).u[1] = b(ix, iy, iz).u[1] + dt * FPY;
          b(ix, iy, iz).u[2] = b(ix, iy, iz).u[2] + dt * FPZ;
          FX += dv * FPX;
          FY += dv * FPY;
          FZ += dv * FPZ;
          TX += dv * (p[1] * FPZ - p[2] * FPY);
          TY += dv * (p[2] * FPX - p[0] * FPZ);
          TZ += dv * (p[0] * FPY - p[1] * FPX);
        }
  }
};
static void kernelFinalizePenalizationForce(SimulationData &sim) {
  for (const auto &obst : sim.obstacle_vector->getObstacleVector()) {
    static constexpr int nQoI = 6;
    Real M[nQoI] = {0};
    const auto &oBlock = obst->getObstacleBlocks();
#pragma omp parallel for schedule(static) reduction(+ : M[:nQoI])
    for (size_t i = 0; i < oBlock.size(); ++i) {
      if (oBlock[i] == nullptr)
        continue;
      M[0] += oBlock[i]->FX;
      M[1] += oBlock[i]->FY;
      M[2] += oBlock[i]->FZ;
      M[3] += oBlock[i]->TX;
      M[4] += oBlock[i]->TY;
      M[5] += oBlock[i]->TZ;
    }
    const auto comm = sim.comm;
    MPI_Allreduce(MPI_IN_PLACE, M, nQoI, MPI_Real, MPI_SUM, comm);
    obst->force[0] = M[0];
    obst->force[1] = M[1];
    obst->force[2] = M[2];
    obst->torque[0] = M[3];
    obst->torque[1] = M[4];
    obst->torque[2] = M[5];
  }
}
void ComputeJ(const Real *Rc, const Real *R, const Real *N, const Real *I,
              Real *J) {
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
void ElasticCollision(const Real m1, const Real m2, const Real *I1,
                      const Real *I2, const Real *v1, const Real *v2,
                      const Real *o1, const Real *o2, const Real *C1,
                      const Real *C2, const Real NX, const Real NY,
                      const Real NZ, const Real CX, const Real CY,
                      const Real CZ, const Real *vc1, const Real *vc2,
                      Real *hv1, Real *hv2, Real *ho1, Real *ho2) {
  const Real e = 1.0;
  const Real N[3] = {NX, NY, NZ};
  const Real C[3] = {CX, CY, CZ};
  const Real k1[3] = {N[0] / m1, N[1] / m1, N[2] / m1};
  const Real k2[3] = {-N[0] / m2, -N[1] / m2, -N[2] / m2};
  Real J1[3];
  Real J2[3];
  ComputeJ(C, C1, N, I1, J1);
  ComputeJ(C, C2, N, I2, J2);
  const Real nom =
      (e + 1) * ((vc1[0] - vc2[0]) * N[0] + (vc1[1] - vc2[1]) * N[1] +
                 (vc1[2] - vc2[2]) * N[2]);
  const Real denom =
      -(1.0 / m1 + 1.0 / m2) +
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
} // namespace
void Penalization::preventCollidingObstacles() const {
  const auto &shapes = sim.obstacle_vector->getObstacleVector();
  const auto &infos = sim.chiInfo();
  const size_t N = sim.obstacle_vector->nObstacles();
  sim.bCollisionID.clear();
  struct CollisionInfo {
    Real iM = 0;
    Real iPosX = 0;
    Real iPosY = 0;
    Real iPosZ = 0;
    Real iMomX = 0;
    Real iMomY = 0;
    Real iMomZ = 0;
    Real ivecX = 0;
    Real ivecY = 0;
    Real ivecZ = 0;
    Real jM = 0;
    Real jPosX = 0;
    Real jPosY = 0;
    Real jPosZ = 0;
    Real jMomX = 0;
    Real jMomY = 0;
    Real jMomZ = 0;
    Real jvecX = 0;
    Real jvecY = 0;
    Real jvecZ = 0;
  };
  std::vector<CollisionInfo> collisions(N);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N; ++i) {
    auto &coll = collisions[i];
    const auto &iBlocks = shapes[i]->obstacleBlocks;
    const Real iU0 = shapes[i]->transVel[0];
    const Real iU1 = shapes[i]->transVel[1];
    const Real iU2 = shapes[i]->transVel[2];
    const Real iomega0 = shapes[i]->angVel[0];
    const Real iomega1 = shapes[i]->angVel[1];
    const Real iomega2 = shapes[i]->angVel[2];
    const Real iCx = shapes[i]->centerOfMass[0];
    const Real iCy = shapes[i]->centerOfMass[1];
    const Real iCz = shapes[i]->centerOfMass[2];
    for (size_t j = 0; j < N; ++j) {
      if (i == j)
        continue;
      const auto &jBlocks = shapes[j]->obstacleBlocks;
      const Real jU0 = shapes[j]->transVel[0];
      const Real jU1 = shapes[j]->transVel[1];
      const Real jU2 = shapes[j]->transVel[2];
      const Real jomega0 = shapes[j]->angVel[0];
      const Real jomega1 = shapes[j]->angVel[1];
      const Real jomega2 = shapes[j]->angVel[2];
      const Real jCx = shapes[j]->centerOfMass[0];
      const Real jCy = shapes[j]->centerOfMass[1];
      const Real jCz = shapes[j]->centerOfMass[2];
      Real imagmax = 0.0;
      Real jmagmax = 0.0;
      assert(iBlocks.size() == jBlocks.size());
      for (size_t k = 0; k < iBlocks.size(); ++k) {
        if (iBlocks[k] == nullptr || jBlocks[k] == nullptr)
          continue;
        const auto &iSDF = iBlocks[k]->sdfLab;
        const auto &jSDF = jBlocks[k]->sdfLab;
        const auto &iChi = iBlocks[k]->chi;
        const auto &jChi = jBlocks[k]->chi;
        const auto &iUDEF = iBlocks[k]->udef;
        const auto &jUDEF = jBlocks[k]->udef;
        for (int z = 0; z < VectorBlock::sizeZ; ++z)
          for (int y = 0; y < VectorBlock::sizeY; ++y)
            for (int x = 0; x < VectorBlock::sizeX; ++x) {
              if (iChi[z][y][x] <= 0.0 || jChi[z][y][x] <= 0.0)
                continue;
              const auto p = infos[k].pos<Real>(x, y, z);
              const Real iMomX = iU0 + iomega1 * (p[2] - iCz) -
                                 iomega2 * (p[1] - iCy) + iUDEF[z][y][x][0];
              const Real iMomY = iU1 + iomega2 * (p[0] - iCx) -
                                 iomega0 * (p[2] - iCz) + iUDEF[z][y][x][1];
              const Real iMomZ = iU2 + iomega0 * (p[1] - iCy) -
                                 iomega1 * (p[0] - iCx) + iUDEF[z][y][x][2];
              const Real jMomX = jU0 + jomega1 * (p[2] - jCz) -
                                 jomega2 * (p[1] - jCy) + jUDEF[z][y][x][0];
              const Real jMomY = jU1 + jomega2 * (p[0] - jCx) -
                                 jomega0 * (p[2] - jCz) + jUDEF[z][y][x][1];
              const Real jMomZ = jU2 + jomega0 * (p[1] - jCy) -
                                 jomega1 * (p[0] - jCx) + jUDEF[z][y][x][2];
              const Real imag = iMomX * iMomX + iMomY * iMomY + iMomZ * iMomZ;
              const Real jmag = jMomX * jMomX + jMomY * jMomY + jMomZ * jMomZ;
              const Real ivecX =
                  iSDF[z + 1][y + 1][x + 2] - iSDF[z + 1][y + 1][x];
              const Real ivecY =
                  iSDF[z + 1][y + 2][x + 1] - iSDF[z + 1][y][x + 1];
              const Real ivecZ =
                  iSDF[z + 2][y + 1][x + 1] - iSDF[z][y + 1][x + 1];
              const Real jvecX =
                  jSDF[z + 1][y + 1][x + 2] - jSDF[z + 1][y + 1][x];
              const Real jvecY =
                  jSDF[z + 1][y + 2][x + 1] - jSDF[z + 1][y][x + 1];
              const Real jvecZ =
                  jSDF[z + 2][y + 1][x + 1] - jSDF[z][y + 1][x + 1];
              const Real normi =
                  1.0 /
                  (sqrt(ivecX * ivecX + ivecY * ivecY + ivecZ * ivecZ) + 1e-21);
              const Real normj =
                  1.0 /
                  (sqrt(jvecX * jvecX + jvecY * jvecY + jvecZ * jvecZ) + 1e-21);
              coll.iM += 1;
              coll.iPosX += p[0];
              coll.iPosY += p[1];
              coll.iPosZ += p[2];
              coll.ivecX += ivecX * normi;
              coll.ivecY += ivecY * normi;
              coll.ivecZ += ivecZ * normi;
              if (imag > imagmax) {
                imagmax = imag;
                coll.iMomX = iMomX;
                coll.iMomY = iMomY;
                coll.iMomZ = iMomZ;
              }
              coll.jM += 1;
              coll.jPosX += p[0];
              coll.jPosY += p[1];
              coll.jPosZ += p[2];
              coll.jvecX += jvecX * normj;
              coll.jvecY += jvecY * normj;
              coll.jvecZ += jvecZ * normj;
              if (jmag > jmagmax) {
                jmagmax = jmag;
                coll.jMomX = jMomX;
                coll.jMomY = jMomY;
                coll.jMomZ = jMomZ;
              }
            }
      }
    }
  }
  std::vector<Real> buffer(20 * N);
  std::vector<Real> buffermax(2 * N);
  for (size_t i = 0; i < N; i++) {
    const auto &coll = collisions[i];
    buffermax[2 * i] = coll.iMomX * coll.iMomX + coll.iMomY * coll.iMomY +
                       coll.iMomZ * coll.iMomZ;
    buffermax[2 * i + 1] = coll.jMomX * coll.jMomX + coll.jMomY * coll.jMomY +
                           coll.jMomZ * coll.jMomZ;
  }
  MPI_Allreduce(MPI_IN_PLACE, buffermax.data(), buffermax.size(), MPI_Real,
                MPI_MAX, sim.comm);
  for (size_t i = 0; i < N; i++) {
    const auto &coll = collisions[i];
    const Real maxi = coll.iMomX * coll.iMomX + coll.iMomY * coll.iMomY +
                      coll.iMomZ * coll.iMomZ;
    const Real maxj = coll.jMomX * coll.jMomX + coll.jMomY * coll.jMomY +
                      coll.jMomZ * coll.jMomZ;
    const bool iok = std::fabs(maxi - buffermax[2 * i]) < 1e-10;
    const bool jok = std::fabs(maxj - buffermax[2 * i + 1]) < 1e-10;
    buffer[20 * i] = coll.iM;
    buffer[20 * i + 1] = coll.iPosX;
    buffer[20 * i + 2] = coll.iPosY;
    buffer[20 * i + 3] = coll.iPosZ;
    buffer[20 * i + 4] = iok ? coll.iMomX : 0;
    buffer[20 * i + 5] = iok ? coll.iMomY : 0;
    buffer[20 * i + 6] = iok ? coll.iMomZ : 0;
    buffer[20 * i + 7] = coll.ivecX;
    buffer[20 * i + 8] = coll.ivecY;
    buffer[20 * i + 9] = coll.ivecZ;
    buffer[20 * i + 10] = coll.jM;
    buffer[20 * i + 11] = coll.jPosX;
    buffer[20 * i + 12] = coll.jPosY;
    buffer[20 * i + 13] = coll.jPosZ;
    buffer[20 * i + 14] = jok ? coll.jMomX : 0;
    buffer[20 * i + 15] = jok ? coll.jMomY : 0;
    buffer[20 * i + 16] = jok ? coll.jMomZ : 0;
    buffer[20 * i + 17] = coll.jvecX;
    buffer[20 * i + 18] = coll.jvecY;
    buffer[20 * i + 19] = coll.jvecZ;
  }
  MPI_Allreduce(MPI_IN_PLACE, buffer.data(), buffer.size(), MPI_Real, MPI_SUM,
                sim.comm);
  for (size_t i = 0; i < N; i++) {
    auto &coll = collisions[i];
    coll.iM = buffer[20 * i];
    coll.iPosX = buffer[20 * i + 1];
    coll.iPosY = buffer[20 * i + 2];
    coll.iPosZ = buffer[20 * i + 3];
    coll.iMomX = buffer[20 * i + 4];
    coll.iMomY = buffer[20 * i + 5];
    coll.iMomZ = buffer[20 * i + 6];
    coll.ivecX = buffer[20 * i + 7];
    coll.ivecY = buffer[20 * i + 8];
    coll.ivecZ = buffer[20 * i + 9];
    coll.jM = buffer[20 * i + 10];
    coll.jPosX = buffer[20 * i + 11];
    coll.jPosY = buffer[20 * i + 12];
    coll.jPosZ = buffer[20 * i + 13];
    coll.jMomX = buffer[20 * i + 14];
    coll.jMomY = buffer[20 * i + 15];
    coll.jMomZ = buffer[20 * i + 16];
    coll.jvecX = buffer[20 * i + 17];
    coll.jvecY = buffer[20 * i + 18];
    coll.jvecZ = buffer[20 * i + 19];
  }
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i + 1; j < N; ++j) {
      const Real m1 = shapes[i]->mass;
      const Real m2 = shapes[j]->mass;
      const Real v1[3] = {shapes[i]->transVel[0], shapes[i]->transVel[1],
                          shapes[i]->transVel[2]};
      const Real o1[3] = {shapes[i]->angVel[0], shapes[i]->angVel[1],
                          shapes[i]->angVel[2]};
      const Real v2[3] = {shapes[j]->transVel[0], shapes[j]->transVel[1],
                          shapes[j]->transVel[2]};
      const Real o2[3] = {shapes[j]->angVel[0], shapes[j]->angVel[1],
                          shapes[j]->angVel[2]};
      const Real I1[6] = {shapes[i]->J[0], shapes[i]->J[1], shapes[i]->J[2],
                          shapes[i]->J[3], shapes[i]->J[4], shapes[i]->J[5]};
      const Real I2[6] = {shapes[j]->J[0], shapes[j]->J[1], shapes[j]->J[2],
                          shapes[j]->J[3], shapes[j]->J[4], shapes[j]->J[5]};
      const Real C1[3] = {shapes[i]->centerOfMass[0],
                          shapes[i]->centerOfMass[1],
                          shapes[i]->centerOfMass[2]};
      const Real C2[3] = {shapes[j]->centerOfMass[0],
                          shapes[j]->centerOfMass[1],
                          shapes[j]->centerOfMass[2]};
      auto &coll = collisions[i];
      auto &coll_other = collisions[j];
      const Real tolerance = 0.001;
      if (coll.iM < tolerance || coll.jM < tolerance)
        continue;
      if (coll_other.iM < tolerance || coll_other.jM < tolerance)
        continue;
      if (std::fabs(coll.iPosX / coll.iM - coll_other.iPosX / coll_other.iM) >
              0.2 ||
          std::fabs(coll.iPosY / coll.iM - coll_other.iPosY / coll_other.iM) >
              0.2 ||
          std::fabs(coll.iPosZ / coll.iM - coll_other.iPosZ / coll_other.iM) >
              0.2) {
        continue;
      }
      sim.bCollision = true;
#pragma omp critical
      {
        sim.bCollisionID.push_back(i);
        sim.bCollisionID.push_back(j);
      }
      const Real norm_i =
          std::sqrt(coll.ivecX * coll.ivecX + coll.ivecY * coll.ivecY +
                    coll.ivecZ * coll.ivecZ);
      const Real norm_j =
          std::sqrt(coll.jvecX * coll.jvecX + coll.jvecY * coll.jvecY +
                    coll.jvecZ * coll.jvecZ);
      const Real mX = coll.ivecX / norm_i - coll.jvecX / norm_j;
      const Real mY = coll.ivecY / norm_i - coll.jvecY / norm_j;
      const Real mZ = coll.ivecZ / norm_i - coll.jvecZ / norm_j;
      const Real inorm = 1.0 / std::sqrt(mX * mX + mY * mY + mZ * mZ);
      const Real NX = mX * inorm;
      const Real NY = mY * inorm;
      const Real NZ = mZ * inorm;
      const Real projVel = (coll.jMomX - coll.iMomX) * NX +
                           (coll.jMomY - coll.iMomY) * NY +
                           (coll.jMomZ - coll.iMomZ) * NZ;
      if (projVel <= 0)
        continue;
      const Real inv_iM = 1.0 / coll.iM;
      const Real inv_jM = 1.0 / coll.jM;
      const Real iPX = coll.iPosX * inv_iM;
      const Real iPY = coll.iPosY * inv_iM;
      const Real iPZ = coll.iPosZ * inv_iM;
      const Real jPX = coll.jPosX * inv_jM;
      const Real jPY = coll.jPosY * inv_jM;
      const Real jPZ = coll.jPosZ * inv_jM;
      const Real CX = 0.5 * (iPX + jPX);
      const Real CY = 0.5 * (iPY + jPY);
      const Real CZ = 0.5 * (iPZ + jPZ);
      const Real vc1[3] = {coll.iMomX, coll.iMomY, coll.iMomZ};
      const Real vc2[3] = {coll.jMomX, coll.jMomY, coll.jMomZ};
      Real ho1[3];
      Real ho2[3];
      Real hv1[3];
      Real hv2[3];
      const bool iforced = shapes[i]->bForcedInSimFrame[0] ||
                           shapes[i]->bForcedInSimFrame[1] ||
                           shapes[i]->bForcedInSimFrame[2];
      const bool jforced = shapes[j]->bForcedInSimFrame[0] ||
                           shapes[j]->bForcedInSimFrame[1] ||
                           shapes[j]->bForcedInSimFrame[2];
      const Real m1_i = iforced ? 1e10 * m1 : m1;
      const Real m2_j = jforced ? 1e10 * m2 : m2;
      ElasticCollision(m1_i, m2_j, I1, I2, v1, v2, o1, o2, C1, C2, NX, NY, NZ,
                       CX, CY, CZ, vc1, vc2, hv1, hv2, ho1, ho2);
      shapes[i]->transVel[0] = hv1[0];
      shapes[i]->transVel[1] = hv1[1];
      shapes[i]->transVel[2] = hv1[2];
      shapes[j]->transVel[0] = hv2[0];
      shapes[j]->transVel[1] = hv2[1];
      shapes[j]->transVel[2] = hv2[2];
      shapes[i]->angVel[0] = ho1[0];
      shapes[i]->angVel[1] = ho1[1];
      shapes[i]->angVel[2] = ho1[2];
      shapes[j]->angVel[0] = ho2[0];
      shapes[j]->angVel[1] = ho2[1];
      shapes[j]->angVel[2] = ho2[2];
      shapes[i]->u_collision = hv1[0];
      shapes[i]->v_collision = hv1[1];
      shapes[i]->w_collision = hv1[2];
      shapes[i]->ox_collision = ho1[0];
      shapes[i]->oy_collision = ho1[1];
      shapes[i]->oz_collision = ho1[2];
      shapes[j]->u_collision = hv2[0];
      shapes[j]->v_collision = hv2[1];
      shapes[j]->w_collision = hv2[2];
      shapes[j]->ox_collision = ho2[0];
      shapes[j]->oy_collision = ho2[1];
      shapes[j]->oz_collision = ho2[2];
      shapes[i]->collision_counter = 0.01 * sim.dt;
      shapes[j]->collision_counter = 0.01 * sim.dt;
    }
}
Penalization::Penalization(SimulationData &s) : Operator(s) {}
void Penalization::operator()(const Real dt) {
  if (sim.obstacle_vector->nObstacles() == 0)
    return;
  preventCollidingObstacles();
  std::vector<Info> &chiInfo = sim.chiInfo();
  std::vector<Info> &velInfo = sim.velInfo();
#pragma omp parallel
  {
    KernelPenalization K(dt, sim.lambda, sim.bImplicitPenalization,
                         sim.obstacle_vector);
#pragma omp for schedule(dynamic, 1)
    for (size_t i = 0; i < chiInfo.size(); ++i)
      K(velInfo[i], chiInfo[i]);
  }
  kernelFinalizePenalizationForce(sim);
}
static void _normalize(Real *const x, Real *const y, Real *const z) {
  const Real norm = std::sqrt(*x * *x + *y * *y + *z * *z);
  assert(norm > 1e-9);
  const Real inv = 1.0 / norm;
  *x = inv * *x;
  *y = inv * *y;
  *z = inv * *z;
}
static void _normalized_cross(const Real ax, const Real ay, const Real az,
                              const Real bx, const Real by, const Real bz,
                              Real *const cx, Real *const cy, Real *const cz) {
  const Real x = ay * bz - az * by;
  const Real y = az * bx - ax * bz;
  const Real z = ax * by - ay * bx;
  const Real norm = std::sqrt(x * x + y * y + z * z);
  assert(norm > 1e-9);
  const Real inv = 1.0 / norm;
  *cx = inv * x;
  *cy = inv * y;
  *cz = inv * z;
}
void PoissonSolverAMR::solve() {
  const auto &AxInfo = sim.lhsInfo();
  const auto &zInfo = sim.presInfo();
  const size_t Nblocks = zInfo.size();
  const int BSX = VectorBlock::sizeX;
  const int BSY = VectorBlock::sizeY;
  const int BSZ = VectorBlock::sizeZ;
  const size_t N = BSX * BSY * BSZ * Nblocks;
  const Real eps = 1e-100;
  const Real max_error = sim.PoissonErrorTol;
  const Real max_rel_error = sim.PoissonErrorTolRel;
  const int max_restarts = 100;
  bool serious_breakdown = false;
  bool useXopt = false;
  int restarts = 0;
  Real min_norm = 1e50;
  Real norm_1 = 0.0;
  Real norm_2 = 0.0;
  const MPI_Comm m_comm = sim.comm;
  phat.resize(N);
  rhat.resize(N);
  shat.resize(N);
  what.resize(N);
  zhat.resize(N);
  qhat.resize(N);
  s.resize(N);
  w.resize(N);
  z.resize(N);
  t.resize(N);
  v.resize(N);
  q.resize(N);
  r.resize(N);
  y.resize(N);
  x.resize(N);
  r0.resize(N);
  b.resize(N);
  x_opt.resize(N);
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    ScalarBlock &__restrict__ rhs = *(ScalarBlock *)AxInfo[i].block;
    const ScalarBlock &__restrict__ zz = *(ScalarBlock *)zInfo[i].block;
    if (sim.bMeanConstraint == 1 || sim.bMeanConstraint > 2)
      if (AxInfo[i].index[0] == 0 && AxInfo[i].index[1] == 0 &&
          AxInfo[i].index[2] == 0)
        rhs(0, 0, 0).s = 0.0;
    for (int iz = 0; iz < BSZ; iz++)
      for (int iy = 0; iy < BSY; iy++)
        for (int ix = 0; ix < BSX; ix++) {
          const int j = i * BSX * BSY * BSZ + iz * BSX * BSY + iy * BSX + ix;
          b[j] = rhs(ix, iy, iz).s;
          r[j] = rhs(ix, iy, iz).s;
          x[j] = zz(ix, iy, iz).s;
        }
  }
  _lhs(x, r0);
#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    r0[i] = r[i] - r0[i];
    r[i] = r0[i];
  }
  _preconditioner(r0, rhat);
  _lhs(rhat, w);
  _preconditioner(w, what);
  _lhs(what, t);
  Real alpha = 0.0;
  Real norm = 0.0;
  Real beta = 0.0;
  Real omega = 0.0;
  Real r0r_prev;
  {
    Real temp0 = 0.0;
    Real temp1 = 0.0;
#pragma omp parallel for reduction(+ : temp0, temp1, norm)
    for (size_t j = 0; j < N; j++) {
      temp0 += r0[j] * r0[j];
      temp1 += r0[j] * w[j];
      norm += r0[j] * r0[j];
    }
    Real temporary[3] = {temp0, temp1, norm};
    MPI_Allreduce(MPI_IN_PLACE, temporary, 3, MPI_Real, MPI_SUM, m_comm);
    alpha = temporary[0] / (temporary[1] + eps);
    r0r_prev = temporary[0];
    norm = std::sqrt(temporary[2]);
  }
  const Real init_norm = norm;
  int k;
  for (k = 0; k < 1000; k++) {
    Real qy = 0.0;
    Real yy = 0.0;
    if (k % 50 != 0) {
#pragma omp parallel for reduction(+ : qy, yy)
      for (size_t j = 0; j < N; j++) {
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
      for (size_t j = 0; j < N; j++) {
        phat[j] = rhat[j] + beta * (phat[j] - omega * shat[j]);
      }
      _lhs(phat, s);
      _preconditioner(s, shat);
      _lhs(shat, z);
#pragma omp parallel for reduction(+ : qy, yy)
      for (size_t j = 0; j < N; j++) {
        q[j] = r[j] - alpha * s[j];
        qhat[j] = rhat[j] - alpha * shat[j];
        y[j] = w[j] - alpha * z[j];
        qy += q[j] * y[j];
        yy += y[j] * y[j];
      }
    }
    MPI_Request request;
    Real quantities[7];
    quantities[0] = qy;
    quantities[1] = yy;
    MPI_Iallreduce(MPI_IN_PLACE, &quantities, 2, MPI_Real, MPI_SUM, m_comm,
                   &request);
    _preconditioner(z, zhat);
    _lhs(zhat, v);
    MPI_Waitall(1, &request, MPI_STATUSES_IGNORE);
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
      for (size_t j = 0; j < N; j++) {
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
      for (size_t j = 0; j < N; j++) {
        x[j] = x[j] + alpha * phat[j] + omega * qhat[j];
      }
      _lhs(x, r);
#pragma omp parallel for
      for (size_t j = 0; j < N; j++) {
        r[j] = b[j] - r[j];
      }
      _preconditioner(r, rhat);
      _lhs(rhat, w);
#pragma omp parallel for reduction(+ : r0r, r0w, r0s, r0z, norm_1, norm_2, norm)
      for (size_t j = 0; j < N; j++) {
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
    MPI_Iallreduce(MPI_IN_PLACE, &quantities, 7, MPI_Real, MPI_SUM, m_comm,
                   &request);
    _preconditioner(w, what);
    _lhs(what, t);
    MPI_Waitall(1, &request, MPI_STATUSES_IGNORE);
    r0r = quantities[0];
    r0w = quantities[1];
    r0s = quantities[2];
    r0z = quantities[3];
    norm_1 = quantities[4];
    norm_2 = quantities[5];
    norm = std::sqrt(quantities[6]);
    beta = alpha / (omega + eps) * r0r / (r0r_prev + eps);
    alpha = r0r / (r0w + beta * r0s - beta * omega * r0z);
    Real alphat = 1.0 / (omega + eps) + r0w / (r0r + eps) -
                  beta * omega * r0z / (r0r + eps);
    alphat = 1.0 / (alphat + eps);
    if (std::fabs(alphat) < 10 * std::fabs(alpha))
      alpha = alphat;
    r0r_prev = r0r;
    serious_breakdown = r0r * r0r < 1e-16 * norm_1 * norm_2;
    if (serious_breakdown && restarts < max_restarts) {
      restarts++;
#pragma omp parallel for
      for (size_t i = 0; i < N; i++)
        r0[i] = r[i];
      _preconditioner(r0, rhat);
      _lhs(rhat, w);
      alpha = 0.0;
      Real temp0 = 0.0;
      Real temp1 = 0.0;
#pragma omp parallel for reduction(+ : temp0, temp1)
      for (size_t j = 0; j < N; j++) {
        temp0 += r0[j] * r0[j];
        temp1 += r0[j] * w[j];
      }
      MPI_Request request2;
      Real temporary[2] = {temp0, temp1};
      MPI_Iallreduce(MPI_IN_PLACE, temporary, 2, MPI_Real, MPI_SUM, m_comm,
                     &request2);
      _preconditioner(w, what);
      _lhs(what, t);
      MPI_Waitall(1, &request2, MPI_STATUSES_IGNORE);
      alpha = temporary[0] / (temporary[1] + eps);
      r0r_prev = temporary[0];
      beta = 0.0;
      omega = 0.0;
    }
    if (norm < min_norm) {
      useXopt = true;
      min_norm = norm;
#pragma omp parallel for
      for (size_t i = 0; i < N; i++)
        x_opt[i] = x[i];
    }
    if (norm < max_error || norm / (init_norm + eps) < max_rel_error) {
      break;
    }
  }
  Real *solution = useXopt ? x_opt.data() : x.data();
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    ScalarBlock &P = (*sim.pres)(i);
    for (int iz = 0; iz < BSZ; iz++)
      for (int iy = 0; iy < BSY; iy++)
        for (int ix = 0; ix < BSX; ix++) {
          const int j = i * BSX * BSY * BSZ + iz * BSX * BSY + iy * BSX + ix;
          P(ix, iy, iz).s = solution[j];
        }
  }
}
namespace poisson_kernels {
static constexpr Real kDivEpsilon = 1e-55;
static constexpr Real kNormRelCriterion = 1e-7;
static constexpr Real kNormAbsCriterion = 1e-16;
static constexpr Real kSqrNormRelCriterion =
    kNormRelCriterion * kNormRelCriterion;
static constexpr Real kSqrNormAbsCriterion =
    kNormAbsCriterion * kNormAbsCriterion;
static inline Real subAndSumSqr(Block &__restrict__ r_,
                                const Block &__restrict__ Ax_, Real a) {
  constexpr int MX = 16;
  constexpr int MY = NX * NY * NZ / MX;
  using SquashedBlock = Real[MY][MX];
  static_assert(NX * NY % MX == 0 && sizeof(Block) == sizeof(SquashedBlock));
  SquashedBlock &__restrict__ r = (SquashedBlock &)r_;
  SquashedBlock &__restrict__ Ax = (SquashedBlock &)Ax_;
  Real s[MX] = {};
  for (int jy = 0; jy < MY; ++jy) {
    for (int jx = 0; jx < MX; ++jx)
      r[jy][jx] -= a * Ax[jy][jx];
    for (int jx = 0; jx < MX; ++jx)
      s[jx] += r[jy][jx] * r[jy][jx];
  }
  return sum(s);
}
template <typename T>
static inline T *assumeAligned(T *ptr, unsigned align, unsigned offset = 0) {
  if (sizeof(Real) == 8 || sizeof(Real) == 4) {
    assert((uintptr_t)ptr % align == offset);
    return (T *)__builtin_assume_aligned(ptr, align, offset);
  } else {
    return ptr;
  }
}
Real kernelPoissonGetZInner(PaddedBlock &p_, const Real *pW_, const Real *pE_,
                            Block &__restrict__ Ax_, Block &__restrict__ r_,
                            Block &__restrict__ block_, const Real sqrNorm0,
                            const Real rr) {
  PaddedBlock &p = *assumeAligned(&p_, 64, 64 - xPad * sizeof(Real));
  const PaddedBlock &pW = *(PaddedBlock *)pW_;
  const PaddedBlock &pE = *(PaddedBlock *)pE_;
  Block &__restrict__ Ax = *assumeAligned(&Ax_, 64);
  Block &__restrict__ r = *assumeAligned(&r_, 64);
  Block &__restrict__ block = *assumeAligned(&block_, kBlockAlignment);
  Real a2Partial[NX] = {};
  for (int iz = 0; iz < NZ; ++iz)
    for (int iy = 0; iy < NY; ++iy) {
      Real tmpAx[NX];
      for (int ix = 0; ix < NX; ++ix) {
        tmpAx[ix] = pW[iz + 1][iy + 1][ix + xPad] +
                    pE[iz + 1][iy + 1][ix + xPad] -
                    6 * p[iz + 1][iy + 1][ix + xPad];
      }
      for (int ix = 0; ix < NX; ++ix)
        tmpAx[ix] += p[iz + 1][iy][ix + xPad];
      for (int ix = 0; ix < NX; ++ix)
        tmpAx[ix] += p[iz + 1][iy + 2][ix + xPad];
      for (int ix = 0; ix < NX; ++ix)
        tmpAx[ix] += p[iz][iy + 1][ix + xPad];
      for (int ix = 0; ix < NX; ++ix)
        tmpAx[ix] += p[iz + 2][iy + 1][ix + xPad];
      for (int ix = 0; ix < NX; ++ix)
        Ax[iz][iy][ix] = tmpAx[ix];
      for (int ix = 0; ix < NX; ++ix)
        a2Partial[ix] += p[iz + 1][iy + 1][ix + xPad] * tmpAx[ix];
    }
  const Real a2 = sum(a2Partial);
  const Real a = rr / (a2 + kDivEpsilon);
  for (int iz = 0; iz < NZ; ++iz)
    for (int iy = 0; iy < NY; ++iy)
      for (int ix = 0; ix < NX; ++ix)
        block[iz][iy][ix] += a * p[iz + 1][iy + 1][ix + xPad];
  const Real sqrSum = subAndSumSqr(r, Ax, a);
  const Real beta = sqrSum / (rr + kDivEpsilon);
  const Real sqrNorm = (Real)1 / (N * N) * sqrSum;
  if (sqrNorm < kSqrNormRelCriterion * sqrNorm0 ||
      sqrNorm < kSqrNormAbsCriterion)
    return -1.0;
  for (int iz = 0; iz < NZ; ++iz)
    for (int iy = 0; iy < NY; ++iy)
      for (int ix = 0; ix < NX; ++ix) {
        p[iz + 1][iy + 1][ix + xPad] =
            r[iz][iy][ix] + beta * p[iz + 1][iy + 1][ix + xPad];
      }
  const Real rrNew = sqrSum;
  return rrNew;
}
void getZImplParallel(const std::vector<Info> &vInfo) {
  const size_t Nblocks = vInfo.size();
  struct Tmp {
    Block r;
    char padding1[64 - xPad * sizeof(Real)];
    PaddedBlock p;
    char padding2[xPad * sizeof(Real)];
    Block Ax;
  };
  alignas(64) Tmp tmp{};
  Block &r = tmp.r;
  Block &Ax = tmp.Ax;
  PaddedBlock &p = tmp.p;
#pragma omp for
  for (size_t i = 0; i < Nblocks; ++i) {
    static_assert(sizeof(ScalarBlock) == sizeof(Block));
    assert((uintptr_t)vInfo[i].block % kBlockAlignment == 0);
    Block &block =
        *(Block *)__builtin_assume_aligned(vInfo[i].block, kBlockAlignment);
    const Real invh = 1 / vInfo[i].h;
    Real rrPartial[NX] = {};
    for (int iz = 0; iz < NZ; ++iz)
      for (int iy = 0; iy < NY; ++iy)
        for (int ix = 0; ix < NX; ++ix) {
          r[iz][iy][ix] = invh * block[iz][iy][ix];
          rrPartial[ix] += r[iz][iy][ix] * r[iz][iy][ix];
          p[iz + 1][iy + 1][ix + xPad] = r[iz][iy][ix];
          block[iz][iy][ix] = 0;
        }
    Real rr = sum(rrPartial);
    const Real sqrNorm0 = (Real)1 / (N * N) * rr;
    if (sqrNorm0 < 1e-32)
      continue;
    const Real *pW = &p[0][0][0] - 1;
    const Real *pE = &p[0][0][0] + 1;
    for (int k = 0; k < 100; ++k) {
      rr = kernelPoissonGetZInner(p, pW, pE, Ax, r, block, sqrNorm0, rr);
      if (rr <= 0)
        break;
    }
  }
}
} // namespace poisson_kernels
std::shared_ptr<PoissonSolverBase> makePoissonSolver(SimulationData &s) {
  if (s.poissonSolver == "iterative") {
    return std::make_shared<PoissonSolverAMR>(s);
  } else if (s.poissonSolver == "cuda_iterative") {
    throw std::runtime_error(
        "Poisson solver: \"" + s.poissonSolver +
        "\" must be compiled with the -DGPU_POISSON flag!");
  } else {
    throw std::invalid_argument("Poisson solver: \"" + s.poissonSolver +
                                "\" unrecognized!");
  }
}
using CHIMAT = Real[_BS_][_BS_][_BS_];
using UDEFMAT = Real[_BS_][_BS_][_BS_][3];
struct KernelDivPressure {
  const SimulationData &sim;
  const StencilInfo stencil = StencilInfo(-1, -1, -1, 2, 2, 2, false, {0});
  const std::vector<Info> &tmpVInfo = sim.tmpVInfo();
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  KernelDivPressure(const SimulationData &s) : sim(s) {}
  void operator()(const ScalarLab &lab, const Info &info) const {
    VectorBlock &__restrict__ b = (*sim.tmpV)(info.blockID);
    const Real fac = info.h;
    for (int z = 0; z < Nz; ++z)
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x)
          b(x, y, z).u[0] = fac * (lab(x + 1, y, z).s + lab(x - 1, y, z).s +
                                   lab(x, y + 1, z).s + lab(x, y - 1, z).s +
                                   lab(x, y, z + 1).s + lab(x, y, z - 1).s -
                                   6.0 * lab(x, y, z).s);
    BlockCase<VectorBlock> *tempCase =
        (BlockCase<VectorBlock> *)(tmpVInfo[info.blockID].auxiliary);
    if (tempCase == nullptr)
      return;
    VectorElement *const faceXm =
        tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
    VectorElement *const faceXp =
        tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
    VectorElement *const faceYm =
        tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
    VectorElement *const faceYp =
        tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    VectorElement *const faceZm =
        tempCase->storedFace[4] ? &tempCase->m_pData[4][0] : nullptr;
    VectorElement *const faceZp =
        tempCase->storedFace[5] ? &tempCase->m_pData[5][0] : nullptr;
    if (faceXm != nullptr) {
      const int x = 0;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          faceXm[y + Ny * z].u[0] = fac * (lab(x, y, z).s - lab(x - 1, y, z).s);
    }
    if (faceXp != nullptr) {
      const int x = Nx - 1;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          faceXp[y + Ny * z].u[0] =
              -fac * (lab(x + 1, y, z).s - lab(x, y, z).s);
    }
    if (faceYm != nullptr) {
      const int y = 0;
      for (int z = 0; z < Nz; ++z)
        for (int x = 0; x < Nx; ++x)
          faceYm[x + Nx * z].u[0] = fac * (lab(x, y, z).s - lab(x, y - 1, z).s);
    }
    if (faceYp != nullptr) {
      const int y = Ny - 1;
      for (int z = 0; z < Nz; ++z)
        for (int x = 0; x < Nx; ++x)
          faceYp[x + Nx * z].u[0] =
              -fac * (lab(x, y + 1, z).s - lab(x, y, z).s);
    }
    if (faceZm != nullptr) {
      const int z = 0;
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x)
          faceZm[x + Nx * y].u[0] = fac * (lab(x, y, z).s - lab(x, y, z - 1).s);
    }
    if (faceZp != nullptr) {
      const int z = Nz - 1;
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x)
          faceZp[x + Nx * y].u[0] =
              -fac * (lab(x, y, z + 1).s - lab(x, y, z).s);
    }
  }
};
struct KernelPressureRHS {
  SimulationData &sim;
  const Real dt;
  ObstacleVector *const obstacle_vector = sim.obstacle_vector;
  const int nShapes = obstacle_vector->nObstacles();
  StencilInfo stencil = StencilInfo(-1, -1, -1, 2, 2, 2, false, {0, 1, 2});
  StencilInfo stencil2 = StencilInfo(-1, -1, -1, 2, 2, 2, false, {0, 1, 2});
  const std::vector<Info> &lhsInfo = sim.lhsInfo();
  const std::vector<Info> &chiInfo = sim.chiInfo();
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  KernelPressureRHS(SimulationData &s, const Real a_dt) : sim(s), dt(a_dt) {}
  void operator()(const VectorLab &lab, const VectorLab &uDefLab,
                  const Info &info, const Info &info2) const {
    const Real h = info.h, fac = 0.5 * h * h / dt;
    const ScalarBlock &__restrict__ c = (*sim.chi)(info2.blockID);
    ScalarBlock &__restrict__ p = (*sim.lhs)(info2.blockID);
    for (int z = 0; z < Nz; ++z)
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          {
            const VectorElement &LW = lab(x - 1, y, z), &LE = lab(x + 1, y, z);
            const VectorElement &LS = lab(x, y - 1, z), &LN = lab(x, y + 1, z);
            const VectorElement &LF = lab(x, y, z - 1), &LB = lab(x, y, z + 1);
            p(x, y, z).s = fac * (LE.u[0] - LW.u[0] + LN.u[1] - LS.u[1] +
                                  LB.u[2] - LF.u[2]);
          }
          {
            const VectorElement &LW = uDefLab(x - 1, y, z),
                                &LE = uDefLab(x + 1, y, z);
            const VectorElement &LS = uDefLab(x, y - 1, z),
                                &LN = uDefLab(x, y + 1, z);
            const VectorElement &LF = uDefLab(x, y, z - 1),
                                &LB = uDefLab(x, y, z + 1);
            const Real divUs =
                LE.u[0] - LW.u[0] + LN.u[1] - LS.u[1] + LB.u[2] - LF.u[2];
            p(x, y, z).s += -c(x, y, z).s * fac * divUs;
          }
        }
    BlockCase<ScalarBlock> *tempCase =
        (BlockCase<ScalarBlock> *)(lhsInfo[info2.blockID].auxiliary);
    if (tempCase == nullptr)
      return;
    ScalarElement *const faceXm =
        tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
    ScalarElement *const faceXp =
        tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
    ScalarElement *const faceYm =
        tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
    ScalarElement *const faceYp =
        tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    ScalarElement *const faceZm =
        tempCase->storedFace[4] ? &tempCase->m_pData[4][0] : nullptr;
    ScalarElement *const faceZp =
        tempCase->storedFace[5] ? &tempCase->m_pData[5][0] : nullptr;
    if (faceXm != nullptr) {
      const int x = 0;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          faceXm[y + Ny * z].s =
              fac * (lab(x - 1, y, z).u[0] + lab(x, y, z).u[0]) -
              c(x, y, z).s * fac *
                  (uDefLab(x - 1, y, z).u[0] + uDefLab(x, y, z).u[0]);
    }
    if (faceXp != nullptr) {
      const int x = Nx - 1;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          faceXp[y + Ny * z].s =
              -fac * (lab(x + 1, y, z).u[0] + lab(x, y, z).u[0]) +
              c(x, y, z).s * fac *
                  (uDefLab(x + 1, y, z).u[0] + uDefLab(x, y, z).u[0]);
    }
    if (faceYm != nullptr) {
      const int y = 0;
      for (int z = 0; z < Nz; ++z)
        for (int x = 0; x < Nx; ++x)
          faceYm[x + Nx * z].s =
              fac * (lab(x, y - 1, z).u[1] + lab(x, y, z).u[1]) -
              c(x, y, z).s * fac *
                  (uDefLab(x, y - 1, z).u[1] + uDefLab(x, y, z).u[1]);
    }
    if (faceYp != nullptr) {
      const int y = Ny - 1;
      for (int z = 0; z < Nz; ++z)
        for (int x = 0; x < Nx; ++x)
          faceYp[x + Nx * z].s =
              -fac * (lab(x, y + 1, z).u[1] + lab(x, y, z).u[1]) +
              c(x, y, z).s * fac *
                  (uDefLab(x, y + 1, z).u[1] + uDefLab(x, y, z).u[1]);
    }
    if (faceZm != nullptr) {
      const int z = 0;
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x)
          faceZm[x + Nx * y].s =
              fac * (lab(x, y, z - 1).u[2] + lab(x, y, z).u[2]) -
              c(x, y, z).s * fac *
                  (uDefLab(x, y, z - 1).u[2] + uDefLab(x, y, z).u[2]);
    }
    if (faceZp != nullptr) {
      const int z = Nz - 1;
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x)
          faceZp[x + Nx * y].s =
              -fac * (lab(x, y, z + 1).u[2] + lab(x, y, z).u[2]) +
              c(x, y, z).s * fac *
                  (uDefLab(x, y, z + 1).u[2] + uDefLab(x, y, z).u[2]);
    }
  }
};
static void kernelUpdateTmpV(SimulationData &sim) {
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  const std::vector<Info> &chiInfo = sim.chiInfo();
#pragma omp parallel
  {
    for (const auto &obstacle : sim.obstacle_vector->getObstacleVector()) {
      const auto &obstblocks = obstacle->getObstacleBlocks();
#pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < chiInfo.size(); ++i) {
        const Info &info = chiInfo[i];
        const auto pos = obstblocks[info.blockID];
        if (pos == nullptr)
          continue;
        const ScalarBlock &c = (*sim.chi)(i);
        VectorBlock &b = (*sim.tmpV)(i);
        const UDEFMAT &__restrict__ UDEF = pos->udef;
        const CHIMAT &__restrict__ CHI = pos->chi;
        for (int z = 0; z < Nz; ++z)
          for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x) {
              if (c(x, y, z).s > CHI[z][y][x])
                continue;
              b(x, y, z).u[0] += UDEF[z][y][x][0];
              b(x, y, z).u[1] += UDEF[z][y][x][1];
              b(x, y, z).u[2] += UDEF[z][y][x][2];
            }
      }
    }
  }
}
struct KernelGradP {
  const StencilInfo stencil{-1, -1, -1, 2, 2, 2, false, {0}};
  SimulationData &sim;
  const std::vector<Info> &tmpVInfo = sim.tmpVInfo();
  const Real dt;
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  KernelGradP(SimulationData &s, const Real a_dt) : sim(s), dt(a_dt) {}
  ~KernelGradP() {}
  void operator()(const ScalarLab &lab, const Info &info) const {
    VectorBlock &o = (*sim.tmpV)(info.blockID);
    const Real fac = -0.5 * dt * info.h * info.h;
    for (int z = 0; z < Nz; ++z)
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          o(x, y, z).u[0] = fac * (lab(x + 1, y, z).s - lab(x - 1, y, z).s);
          o(x, y, z).u[1] = fac * (lab(x, y + 1, z).s - lab(x, y - 1, z).s);
          o(x, y, z).u[2] = fac * (lab(x, y, z + 1).s - lab(x, y, z - 1).s);
        }
    BlockCase<VectorBlock> *tempCase =
        (BlockCase<VectorBlock> *)(tmpVInfo[info.blockID].auxiliary);
    if (tempCase == nullptr)
      return;
    VectorElement *faceXm =
        tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
    VectorElement *faceXp =
        tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
    VectorElement *faceYm =
        tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
    VectorElement *faceYp =
        tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    VectorElement *faceZm =
        tempCase->storedFace[4] ? &tempCase->m_pData[4][0] : nullptr;
    VectorElement *faceZp =
        tempCase->storedFace[5] ? &tempCase->m_pData[5][0] : nullptr;
    if (faceXm != nullptr) {
      const int x = 0;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          faceXm[y + Ny * z].u[0] = fac * (lab(x - 1, y, z).s + lab(x, y, z).s);
    }
    if (faceXp != nullptr) {
      const int x = Nx - 1;
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          faceXp[y + Ny * z].u[0] =
              -fac * (lab(x + 1, y, z).s + lab(x, y, z).s);
    }
    if (faceYm != nullptr) {
      const int y = 0;
      for (int z = 0; z < Nz; ++z)
        for (int x = 0; x < Nx; ++x)
          faceYm[x + Nx * z].u[1] = fac * (lab(x, y - 1, z).s + lab(x, y, z).s);
    }
    if (faceYp != nullptr) {
      const int y = Ny - 1;
      for (int z = 0; z < Nz; ++z)
        for (int x = 0; x < Nx; ++x)
          faceYp[x + Nx * z].u[1] =
              -fac * (lab(x, y + 1, z).s + lab(x, y, z).s);
    }
    if (faceZm != nullptr) {
      const int z = 0;
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x)
          faceZm[x + Nx * y].u[2] = fac * (lab(x, y, z - 1).s + lab(x, y, z).s);
    }
    if (faceZp != nullptr) {
      const int z = Nz - 1;
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x)
          faceZp[x + Nx * y].u[2] =
              -fac * (lab(x, y, z + 1).s + lab(x, y, z).s);
    }
  }
};
PressureProjection::PressureProjection(SimulationData &s) : Operator(s) {
  pressureSolver = makePoissonSolver(s);
  sim.pressureSolver = pressureSolver;
}
void PressureProjection::operator()(const Real dt) {
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  const std::vector<Info> &presInfo = sim.presInfo();
  pOld.resize(Nx * Ny * Nz * presInfo.size());
  {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < presInfo.size(); i++) {
      const ScalarBlock &p = (*sim.pres)(i);
      VectorBlock &tmpV = (*sim.tmpV)(i);
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x) {
            pOld[i * Nx * Ny * Nz + z * Ny * Nx + y * Nx + x] = p(x, y, z).s;
            tmpV(x, y, z).u[0] = 0;
            tmpV(x, y, z).u[1] = 0;
            tmpV(x, y, z).u[2] = 0;
          }
    }
    if (sim.obstacle_vector->nObstacles() > 0)
      kernelUpdateTmpV(sim);
    KernelPressureRHS K(sim, dt);
    compute<KernelPressureRHS, VectorGrid, VectorLab, VectorGrid, VectorLab,
            ScalarGrid>(K, *sim.vel, *sim.tmpV, true, sim.lhs);
  }
  if (sim.step > sim.step_2nd_start) {
    compute<ScalarLab>(KernelDivPressure(sim), sim.pres, sim.tmpV);
#pragma omp parallel for
    for (size_t i = 0; i < presInfo.size(); i++) {
      const VectorBlock &b = (*sim.tmpV)(i);
      ScalarBlock &LHS = (*sim.lhs)(i);
      ScalarBlock &p = (*sim.pres)(i);
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x) {
            LHS(x, y, z).s -= b(x, y, z).u[0];
            p(x, y, z).s = 0;
          }
    }
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < presInfo.size(); i++) {
      ScalarBlock &p = (*sim.pres)(i);
      p.clear();
    }
  }
  pressureSolver->solve();
  Real avg = 0;
  Real avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
  for (size_t i = 0; i < presInfo.size(); i++) {
    ScalarBlock &P = (*sim.pres)(i);
    const Real vv = presInfo[i].h * presInfo[i].h * presInfo[i].h;
    for (int iz = 0; iz < Nz; iz++)
      for (int iy = 0; iy < Ny; iy++)
        for (int ix = 0; ix < Nx; ix++) {
          avg += P(ix, iy, iz).s * vv;
          avg1 += vv;
        }
  }
  Real quantities[2] = {avg, avg1};
  MPI_Allreduce(MPI_IN_PLACE, &quantities, 2, MPI_Real, MPI_SUM, sim.comm);
  avg = quantities[0];
  avg1 = quantities[1];
  avg = avg / avg1;
#pragma omp parallel for
  for (size_t i = 0; i < presInfo.size(); i++) {
    ScalarBlock &__restrict__ P = (*sim.pres)(i);
    for (int iz = 0; iz < Nz; iz++)
      for (int iy = 0; iy < Ny; iy++)
        for (int ix = 0; ix < Nx; ix++)
          P(ix, iy, iz).s -= avg;
  }
  const std::vector<Info> &velInfo = sim.velInfo();
  if (sim.step > sim.step_2nd_start) {
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      ScalarBlock &p = (*sim.pres)(i);
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x)
            p(x, y, z).s += pOld[i * Nx * Ny * Nz + z * Ny * Nx + y * Nx + x];
    }
  }
  compute<ScalarLab>(KernelGradP(sim, dt), sim.pres, sim.tmpV);
#pragma omp parallel for
  for (size_t i = 0; i < velInfo.size(); i++) {
    const Real fac = 1.0 / (velInfo[i].h * velInfo[i].h * velInfo[i].h);
    const VectorBlock &gradP = (*sim.tmpV)(i);
    VectorBlock &v = (*sim.vel)(i);
    for (int z = 0; z < Nz; ++z)
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
          v(x, y, z).u[0] += fac * gradP(x, y, z).u[0];
          v(x, y, z).u[1] += fac * gradP(x, y, z).u[1];
          v(x, y, z).u[2] += fac * gradP(x, y, z).u[2];
        }
  }
}
Simulation::Simulation(int argc, char **argv, MPI_Comm comm)
    : parser(argc, argv), sim(comm, parser) {}
void Simulation::init() {
  sim._preprocessArguments();
  setupGrid();
  setupOperators();
  sim.obstacle_vector = new ObstacleVector(sim);
  ObstacleFactory(sim).addObstacles(parser);
  (*sim.pipeline[0])(0);
  (*sim.pipeline[0])(0);
  _ic();
  const int lmax = sim.StaticObstacles ? sim.levelMax : 3 * sim.levelMax;
  for (int l = 0; l < lmax; l++) {
    adaptMesh();
    (*sim.pipeline[0])(0);
    _ic();
  }
}
void Simulation::adaptMesh() {
  ComputeVorticity findOmega(sim);
  findOmega(0);
  compute<ScalarLab>(GradChiOnTmp(sim), sim.chi);
  sim.tmpV_amr->Tag();
  sim.lhs_amr->TagLike(sim.tmpVInfo());
  sim.vel_amr->TagLike(sim.tmpVInfo());
  sim.chi_amr->TagLike(sim.tmpVInfo());
  sim.pres_amr->TagLike(sim.tmpVInfo());
  sim.chi_amr->Adapt(sim.time, false, true);
  sim.lhs_amr->Adapt(sim.time, false, true);
  sim.tmpV_amr->Adapt(sim.time, false, true);
  sim.pres_amr->Adapt(sim.time, false, false);
  sim.vel_amr->Adapt(sim.time, false, false);
  sim.MeshChanged = sim.pres->UpdateFluxCorrection;
}
const std::vector<std::shared_ptr<Obstacle>> &Simulation::getShapes() const {
  return sim.obstacle_vector->getObstacleVector();
}
void Simulation::_ic() {
  InitialConditions coordIC(sim);
  coordIC(0);
}
void Simulation::setupGrid() {
  sim.chi = new ScalarGrid(
      sim.bpdx, sim.bpdy, sim.bpdz, sim.maxextent, sim.levelStart, sim.levelMax,
      sim.comm, (sim.BCx_flag == periodic), (sim.BCy_flag == periodic),
      (sim.BCz_flag == periodic));
  sim.lhs = new ScalarGrid(
      sim.bpdx, sim.bpdy, sim.bpdz, sim.maxextent, sim.levelStart, sim.levelMax,
      sim.comm, (sim.BCx_flag == periodic), (sim.BCy_flag == periodic),
      (sim.BCz_flag == periodic));
  sim.pres = new ScalarGrid(
      sim.bpdx, sim.bpdy, sim.bpdz, sim.maxextent, sim.levelStart, sim.levelMax,
      sim.comm, (sim.BCx_flag == periodic), (sim.BCy_flag == periodic),
      (sim.BCz_flag == periodic));
  sim.vel = new VectorGrid(
      sim.bpdx, sim.bpdy, sim.bpdz, sim.maxextent, sim.levelStart, sim.levelMax,
      sim.comm, (sim.BCx_flag == periodic), (sim.BCy_flag == periodic),
      (sim.BCz_flag == periodic));
  sim.tmpV = new VectorGrid(
      sim.bpdx, sim.bpdy, sim.bpdz, sim.maxextent, sim.levelStart, sim.levelMax,
      sim.comm, (sim.BCx_flag == periodic), (sim.BCy_flag == periodic),
      (sim.BCz_flag == periodic));
  sim.chi_amr = new ScalarAMR(*(sim.chi), sim.Rtol, sim.Ctol);
  sim.lhs_amr = new ScalarAMR(*(sim.lhs), sim.Rtol, sim.Ctol);
  sim.pres_amr = new ScalarAMR(*(sim.pres), sim.Rtol, sim.Ctol);
  sim.vel_amr = new VectorAMR(*(sim.vel), sim.Rtol, sim.Ctol);
  sim.tmpV_amr = new VectorAMR(*(sim.tmpV), sim.Rtol, sim.Ctol);
}
void Simulation::setupOperators() {
  sim.pipeline.push_back(std::make_shared<CreateObstacles>(sim));
  if (sim.implicitDiffusion)
    sim.pipeline.push_back(std::make_shared<AdvectionDiffusionImplicit>(sim));
  else
    sim.pipeline.push_back(std::make_shared<AdvectionDiffusion>(sim));
  if (sim.uMax_forced > 0) {
    if (sim.bFixMassFlux)
      sim.pipeline.push_back(std::make_shared<FixMassFlux>(sim));
    else
      sim.pipeline.push_back(std::make_shared<ExternalForcing>(sim));
  }
  sim.pipeline.push_back(std::make_shared<UpdateObstacles>(sim));
  sim.pipeline.push_back(std::make_shared<Penalization>(sim));
  sim.pipeline.push_back(std::make_shared<PressureProjection>(sim));
  sim.pipeline.push_back(std::make_shared<ComputeForces>(sim));
  sim.pipeline.push_back(std::make_shared<ComputeDissipation>(sim));
}
void Simulation::simulate() {
  for (;;) {
    const Real dt = calcMaxTimestep();
    if (advance(dt))
      break;
  }
}
Real Simulation::calcMaxTimestep() {
  const Real dt_old = sim.dt;
  sim.dt_old = sim.dt;
  const Real hMin = sim.hmin;
  Real CFL = sim.CFL;
  sim.uMax_measured = findMaxU(sim);
  if (sim.uMax_measured > sim.uMax_allowed) {
    if (sim.rank == 0) {
      std::cerr << "maxU = " << sim.uMax_measured
                << " exceeded uMax_allowed = " << sim.uMax_allowed
                << ". Aborting...\n";
      MPI_Abort(sim.comm, 1);
    }
  }
  if (CFL > 0) {
    const Real dtDiffusion =
        (sim.implicitDiffusion && sim.step > 10)
            ? 0.1
            : (1.0 / 6.0) * hMin * hMin /
                  (sim.nu + (1.0 / 6.0) * hMin * sim.uMax_measured);
    const Real dtAdvection = hMin / (sim.uMax_measured + 1e-8);
    if (sim.step < sim.rampup) {
      const Real x = sim.step / (Real)sim.rampup;
      const Real rampCFL =
          std::exp(std::log(1e-3) * (1 - x) + std::log(CFL) * x);
      sim.dt = std::min(dtDiffusion, rampCFL * dtAdvection);
    } else
      sim.dt = std::min(dtDiffusion, CFL * dtAdvection);
  } else {
    CFL = (sim.uMax_measured + 1e-8) * sim.dt / hMin;
  }
  if (sim.dt <= 0) {
    fprintf(stderr,
            "dt <= 0. CFL=%f, hMin=%f, sim.uMax_measured=%f. Aborting...\n",
            CFL, hMin, sim.uMax_measured);
    MPI_Abort(sim.comm, 1);
  }
  if (sim.DLM > 0)
    sim.lambda = sim.DLM / sim.dt;
  if (sim.rank == 0)
    printf("main.cpp: step: %d, time: %f\n", sim.step, sim.time);
  if (sim.step > sim.step_2nd_start) {
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
bool Simulation::advance(const Real dt) {
  if (sim.dumpTime > 0 && sim.time >= sim.nextDumpTime) {
    sim.nextDumpTime += sim.dumpTime;
    char path[FILENAME_MAX];
    snprintf(path, sizeof path, "vel.%08d", sim.step);
    fprintf(stderr, "main.cpp: %s\n", path);
    dump(sim.time, sim.chi->m_vInfo.size(), sim.chi->m_vInfo.data(), path);
  }
  if (sim.step % 20 == 0 || sim.step < 10)
    adaptMesh();
  for (size_t c = 0; c < sim.pipeline.size(); c++) {
    (*sim.pipeline[c])(dt);
  }
  sim.step++;
  sim.time += dt;
  if ((sim.endTime > 0 && sim.time > sim.endTime) ||
      (sim.nsteps != 0 && sim.step >= sim.nsteps)) {
    return true;
  }
  return false;
}
BCflag cubismBCX;
BCflag cubismBCY;
BCflag cubismBCZ;
SimulationData::SimulationData(MPI_Comm mpicomm, ArgumentParser &parser)
    : comm(mpicomm) {
  MPI_Comm_rank(comm, &rank);
  bpdx = parser("-bpdx").asInt();
  bpdy = parser("-bpdy").asInt();
  bpdz = parser("-bpdz").asInt();
  levelMax = parser("-levelMax").asInt();
  levelStart = parser("-levelStart").asInt(levelMax - 1);
  Rtol = parser("-Rtol").asDouble();
  Ctol = parser("-Ctol").asDouble();
  levelMaxVorticity = parser("-levelMaxVorticity").asInt(levelMax);
  StaticObstacles = parser("-StaticObstacles").asBool(false);
  extents[0] = parser("extentx").asDouble(0);
  extents[1] = parser("extenty").asDouble(0);
  extents[2] = parser("extentz").asDouble(0);
  if (extents[0] + extents[1] + extents[2] < 1e-21)
    extents[0] = parser("extent").asDouble(1);
  uinf[0] = parser("-uinfx").asDouble(0.0);
  uinf[1] = parser("-uinfy").asDouble(0.0);
  uinf[2] = parser("-uinfz").asDouble(0.0);
  CFL = parser("-CFL").asDouble(.1);
  dt = parser("-dt").asDouble(0);
  rampup = parser("-rampup").asInt(100);
  nsteps = parser("-nsteps").asInt(0);
  endTime = parser("-tend").asDouble(0);
  step_2nd_start = 2;
  nu = parser("-nu").asDouble();
  initCond = parser("-initCond").asString("zero");
  uMax_forced = parser("-uMax_forced").asDouble(0.0);
  bFixMassFlux = parser("-bFixMassFlux").asBool(false);
  bImplicitPenalization = parser("-implicitPenalization").asBool(true);
  lambda = parser("-lambda").asDouble(1e6);
  DLM = parser("-use-dlm").asDouble(0);
  freqDiagnostics = parser("-freqDiagnostics").asInt(100);
  PoissonErrorTol = parser("-poissonTol").asDouble(1e-6);
  PoissonErrorTolRel = parser("-poissonTolRel").asDouble(1e-4);
  bMeanConstraint = parser("-bMeanConstraint").asInt(1);
  poissonSolver = parser("-poissonSolver").asString("iterative");
  implicitDiffusion = parser("-implicitDiffusion").asBool(false);
  DiffusionErrorTol = parser("-diffusionTol").asDouble(1e-6);
  DiffusionErrorTolRel = parser("diffusionTolRel").asDouble(1e-4);
  uMax_allowed = parser("-umax").asDouble(10.0);
  std::string BC_x = parser("-BC_x").asString("freespace");
  std::string BC_y = parser("-BC_y").asString("freespace");
  std::string BC_z = parser("-BC_z").asString("freespace");
  BCx_flag = string2BCflag(BC_x);
  BCy_flag = string2BCflag(BC_y);
  BCz_flag = string2BCflag(BC_z);
  cubismBCX = BCx_flag;
  cubismBCY = BCy_flag;
  cubismBCZ = BCz_flag;
  int dumpFreq = parser("-fdump").asDouble(0);
  dumpTime = parser("-tdump").asDouble(0.0);
  saveFreq = parser("-fsave").asInt(0);
  if (saveFreq <= 0 && dumpFreq > 0)
    saveFreq = dumpFreq;
  path4serialization = parser("-serialization").asString("./");
}
void SimulationData::_preprocessArguments() {
  if (bpdx < 1 || bpdy < 1 || bpdz < 1) {
    fprintf(stderr, "Invalid bpd: %d x %d x %d\n", bpdx, bpdy, bpdz);
    fflush(0);
    abort();
  }
  const int aux = 1 << (levelMax - 1);
  const Real NFE[3] = {
      (Real)bpdx * aux * ScalarBlock::sizeX,
      (Real)bpdy * aux * ScalarBlock::sizeY,
      (Real)bpdz * aux * ScalarBlock::sizeZ,
  };
  const Real maxbpd = std::max({NFE[0], NFE[1], NFE[2]});
  maxextent = std::max({extents[0], extents[1], extents[2]});
  sim.h0 = maxextent / std::max({bpdx, bpdy, bpdz}) / _BS_;

  if (extents[0] <= 0 || extents[1] <= 0 || extents[2] <= 0) {
    extents[0] = (NFE[0] / maxbpd) * maxextent;
    extents[1] = (NFE[1] / maxbpd) * maxextent;
    extents[2] = (NFE[2] / maxbpd) * maxextent;
  } else {
    fprintf(stderr, "Invalid extent: %f x %f x %f\n", extents[0], extents[1],
            extents[2]);
    fflush(0);
    abort();
  }
  hmin = extents[0] / NFE[0];
  hmax = extents[0] * aux / NFE[0];
  assert(nu >= 0);
  assert(lambda > 0 || DLM > 0);
  assert(saveFreq >= 0.0);
  assert(dumpTime >= 0.0);
}
SimulationData::~SimulationData() {
  delete obstacle_vector;
  delete chi;
  delete vel;
  delete lhs;
  delete tmpV;
  delete pres;
  delete chi_amr;
  delete vel_amr;
  delete lhs_amr;
  delete tmpV_amr;
  delete pres_amr;
}
void CurvatureDefinedFishData::execute(const Real time, const Real l_tnext,
                                       const std::vector<Real> &input) {
  if (input.size() == 1) {
    rlBendingScheduler.Turn(input[0], l_tnext);
  } else if (input.size() == 3) {
    assert(control_torsion == false);
    rlBendingScheduler.Turn(input[0], l_tnext);
    if (TperiodPID)
      std::cout << "Warning: PID controller should not be used with RL."
                << std::endl;
    current_period = periodPIDval;
    next_period = Tperiod * (1 + input[1]);
    transition_start = l_tnext;
  } else if (input.size() == 5) {
    assert(control_torsion == true);
    rlBendingScheduler.Turn(input[0], l_tnext);
    if (TperiodPID)
      std::cout << "Warning: PID controller should not be used with RL."
                << std::endl;
    current_period = periodPIDval;
    next_period = Tperiod * (1 + input[1]);
    transition_start = l_tnext;
    for (int i = 0; i < 3; i++) {
      torsionValues_previous[i] = torsionValues[i];
      torsionValues[i] = input[i + 2];
    }
    Ttorsion_start = time;
  }
}
void CurvatureDefinedFishData::computeMidline(const Real t, const Real dt) {
  periodScheduler.transition(t, transition_start,
                             transition_start + transition_duration,
                             current_period, next_period);
  periodScheduler.gimmeValues(t, periodPIDval, periodPIDdif);
  if (transition_start < t && t < transition_start + transition_duration) {
    timeshift = (t - time0) / periodPIDval + timeshift;
    time0 = t;
  }
  const std::array<Real, 6> curvaturePoints = {
      0.0, 0.15 * length, 0.4 * length, 0.65 * length, 0.9 * length, length};
  const std::array<Real, 7> bendPoints = {-0.5, -0.25, 0.0, 0.25,
                                          0.5,  0.75,  1.0};
  const std::array<Real, 6> curvatureValues = {
      0.82014 / length, 1.46515 / length, 2.57136 / length,
      3.75425 / length, 5.09147 / length, 5.70449 / length};
#if 1
  const std::array<Real, 6> curvatureZeros = std::array<Real, 6>();
  curvatureScheduler.transition(0, 0, Tperiod, curvatureZeros, curvatureValues);
#else
  curvatureScheduler.transition(t, 0, Tperiod, curvatureValues,
                                curvatureValues);
#endif
  curvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rC, vC);
  rlBendingScheduler.gimmeValues(t, periodPIDval, length, bendPoints, Nm, rS,
                                 rB, vB);
  const Real diffT =
      TperiodPID ? 1 - (t - time0) * periodPIDdif / periodPIDval : 1;
  const Real darg = 2 * M_PI / periodPIDval * diffT;
  const Real arg0 =
      2 * M_PI * ((t - time0) / periodPIDval + timeshift) + M_PI * phaseShift;
#pragma omp parallel for
  for (int i = 0; i < Nm; ++i) {
    const Real arg = arg0 - 2 * M_PI * rS[i] / length / waveLength;
    const Real curv = std::sin(arg) + rB[i] + beta;
    const Real dcurv = std::cos(arg) * darg + vB[i] + dbeta;
    rK[i] = alpha * amplitudeFactor * rC[i] * curv;
    vK[i] = alpha * amplitudeFactor * (vC[i] * curv + rC[i] * dcurv) +
            dalpha * amplitudeFactor * rC[i] * curv;
    rT[i] = 0;
    vT[i] = 0;
    assert(!std::isnan(rK[i]));
    assert(!std::isinf(rK[i]));
    assert(!std::isnan(vK[i]));
    assert(!std::isinf(vK[i]));
  }
  if (control_torsion) {
    const std::array<Real, 3> torsionPoints = {0.0, 0.5 * length, length};
    torsionScheduler.transition(t, Ttorsion_start,
                                Ttorsion_start + 0.5 * Tperiod,
                                torsionValues_previous, torsionValues);
    torsionScheduler.gimmeValues(t, torsionPoints, Nm, rS, rT, vT);
  }
  Frenet3D::solve(Nm, rS, rK, vK, rT, vT, rX, rY, rZ, vX, vY, vZ, norX, norY,
                  norZ, vNorX, vNorY, vNorZ, binX, binY, binZ, vBinX, vBinY,
                  vBinZ);
  performPitchingMotion(t);
}
void CurvatureDefinedFishData::performPitchingMotion(const Real t) {
  Real R, Rdot;
  if (std::fabs(gamma) > 1e-10) {
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
  recomputeNormalVectors();
}
void CurvatureDefinedFishData::recomputeNormalVectors() {
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
StefanFish::StefanFish(SimulationData &s, ArgumentParser &p) : Fish(s, p) {
  const Real Tperiod = p("-T").asDouble(1.0);
  const Real phaseShift = p("-phi").asDouble(0.0);
  const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
  bCorrectPosition = p("-CorrectPosition").asBool(false);
  bCorrectPositionZ = p("-CorrectPositionZ").asBool(false);
  bCorrectRoll = p("-CorrectRoll").asBool(false);
  std::string heightName = p("-heightProfile").asString("baseline");
  std::string widthName = p("-widthProfile").asString("baseline");
  if ((bCorrectPosition || bCorrectPositionZ || bCorrectRoll) &&
      std::fabs(quaternion[0] - 1) > 1e-6) {
    std::cout << "PID controller only works for zero initial angles."
              << std::endl;
    MPI_Abort(sim.comm, 1);
  }
  myFish = new CurvatureDefinedFishData(length, Tperiod, phaseShift, sim.hmin,
                                        ampFac);
  MidlineShapes::computeWidthsHeights(heightName, widthName, length, myFish->rS,
                                      myFish->height, myFish->width, myFish->Nm,
                                      sim.rank);
  origC[0] = position[0];
  origC[1] = position[1];
  origC[2] = position[2];
  if (sim.rank == 0)
    printf("nMidline=%d, length=%f, Tperiod=%f, phaseShift=%f\n", myFish->Nm,
           length, Tperiod, phaseShift);
  wyp = p("-wyp").asDouble(1.0);
  wzp = p("-wzp").asDouble(1.0);
}
static void clip_quantities(const Real fmax, const Real dfmax, const Real dt,
                            const bool zero, const Real fcandidate,
                            const Real dfcandidate, Real &f, Real &df) {
  if (zero) {
    f = 0;
    df = 0;
  } else if (std::fabs(dfcandidate) > dfmax) {
    df = dfcandidate > 0 ? +dfmax : -dfmax;
    f = f + dt * df;
  } else if (std::fabs(fcandidate) < fmax) {
    f = fcandidate;
    df = dfcandidate;
  } else {
    f = fcandidate > 0 ? fmax : -fmax;
    df = 0;
  }
}
void StefanFish::create() {
  auto *const cFish = dynamic_cast<CurvatureDefinedFishData *>(myFish);
  const int Nm = cFish->Nm;
  const Real q[4] = {quaternion[0], quaternion[1], quaternion[2],
                     quaternion[3]};
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
  const Real xx2 = Rmatrix3D[0] * vx + Rmatrix3D[1] * vy + Rmatrix3D[2] * vz;
  const Real pitch = asin(xx2);
  const Real roll = atan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                          1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
  const Real yaw = atan2(2.0 * (q[3] * q[0] + q[1] * q[2]),
                         -1.0 + 2.0 * (q[0] * q[0] + q[1] * q[1]));
  const bool roll_is_small = std::fabs(roll) < M_PI / 9.;
  const bool yaw_is_small = std::fabs(yaw) < M_PI / 9.;
  if (bCorrectPosition) {
    cFish->alpha = 1.0 + (position[0] - origC[0]) / length;
    cFish->dalpha = (transVel[0] + sim.uinf[0]) / length;
    if (roll_is_small == false) {
      cFish->alpha = 1.0;
      cFish->dalpha = 0.0;
    } else if (cFish->alpha < 0.9) {
      cFish->alpha = 0.9;
      cFish->dalpha = 0.0;
    } else if (cFish->alpha > 1.1) {
      cFish->alpha = 1.1;
      cFish->dalpha = 0.0;
    }
    const Real y = absPos[1];
    const Real ytgt = origC[1];
    const Real dy = (ytgt - y) / length;
    const Real signY = dy > 0 ? 1 : -1;
    const Real yaw_tgt = 0;
    const Real dphi = yaw - yaw_tgt;
    const Real b = roll_is_small ? wyp * signY * dy * dphi : 0;
    const Real dbdt = sim.step > 1 ? (b - cFish->beta) / sim.dt : 0;
    clip_quantities(1.0, 5.0, sim.dt, false, b, dbdt, cFish->beta,
                    cFish->dbeta);
  }
  if (bCorrectPositionZ) {
    const Real pitch_tgt = 0;
    const Real dphi = pitch - pitch_tgt;
    const Real z = absPos[2];
    const Real ztgt = origC[2];
    const Real dz = (ztgt - z) / length;
    const Real signZ = dz > 0 ? 1 : -1;
    const Real g =
        (roll_is_small && yaw_is_small) ? -wzp * dphi * dz * signZ : 0.0;
    const Real dgdt = sim.step > 1 ? (g - cFish->gamma) / sim.dt : 0.0;
    const Real gmax = 0.10 / length;
    const Real dRdtmax = 0.1 * length / cFish->Tperiod;
    const Real dgdtmax = std::fabs(gmax * gmax * dRdtmax);
    clip_quantities(gmax, dgdtmax, sim.dt, false, g, dgdt, cFish->gamma,
                    cFish->dgamma);
  }
  Fish::create();
}
void StefanFish::computeVelocities() {
  Obstacle::computeVelocities();
  if (bCorrectRoll) {
    auto *const cFish = dynamic_cast<CurvatureDefinedFishData *>(myFish);
    const Real *const q = quaternion;
    Real *const o = angVel;
    const Real dq[4] = {0.5 * (-o[0] * q[1] - o[1] * q[2] - o[2] * q[3]),
                        0.5 * (+o[0] * q[0] + o[1] * q[3] - o[2] * q[2]),
                        0.5 * (-o[0] * q[3] + o[1] * q[0] + o[2] * q[1]),
                        0.5 * (+o[0] * q[2] - o[1] * q[1] + o[2] * q[0])};
    const Real nom = 2.0 * (q[3] * q[2] + q[0] * q[1]);
    const Real dnom =
        2.0 * (dq[3] * q[2] + dq[0] * q[1] + q[3] * dq[2] + q[0] * dq[1]);
    const Real denom = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
    const Real ddenom = -2.0 * (2.0 * q[1] * dq[1] + 2.0 * q[2] * dq[2]);
    const Real arg = nom / denom;
    const Real darg = (dnom * denom - nom * ddenom) / denom / denom;
    const Real a = atan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                         1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
    const Real da = 1.0 / (1.0 + arg * arg) * darg;
    const int ss = cFish->Nm / 2;
    const Real offset = cFish->height[ss] > cFish->width[ss] ? M_PI / 2 : 0;
    const Real theta = offset;
    const Real sinth = std::sin(theta), costh = std::cos(theta);
    Real ax = cFish->width[ss] * costh * cFish->norX[ss] +
              cFish->height[ss] * sinth * cFish->binX[ss];
    Real ay = cFish->width[ss] * costh * cFish->norY[ss] +
              cFish->height[ss] * sinth * cFish->binY[ss];
    Real az = cFish->width[ss] * costh * cFish->norZ[ss] +
              cFish->height[ss] * sinth * cFish->binZ[ss];
    const Real inorm = 1.0 / sqrt(ax * ax + ay * ay + az * az + 1e-21);
    ax *= inorm;
    ay *= inorm;
    az *= inorm;
    std::array<Real, 4> roll_axis_temp;
    const int Nm = cFish->Nm;
    const Real d1 = cFish->rX[0] - cFish->rX[Nm - 1];
    const Real d2 = cFish->rY[0] - cFish->rY[Nm - 1];
    const Real d3 = cFish->rZ[0] - cFish->rZ[Nm - 1];
    const Real dn = pow(d1 * d1 + d2 * d2 + d3 * d3, 0.5) + 1e-21;
    roll_axis_temp[0] = -d1 / dn;
    roll_axis_temp[1] = -d2 / dn;
    roll_axis_temp[2] = -d3 / dn;
    roll_axis_temp[3] = sim.dt;
    r_axis.push_back(roll_axis_temp);
    std::array<Real, 3> roll_axis = {0., 0., 0.};
    Real time_roll = 0.0;
    int elements_to_keep = 0;
    for (int i = r_axis.size() - 1; i >= 0; i--) {
      const auto &r = r_axis[i];
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
    const int elements_to_delete = r_axis.size() - elements_to_keep;
    for (int i = 0; i < elements_to_delete; i++)
      r_axis.pop_front();
    if (sim.time < 1.0 || time_roll < 1.0)
      return;
    const Real omega_roll =
        o[0] * roll_axis[0] + o[1] * roll_axis[1] + o[2] * roll_axis[2];
    o[0] += -omega_roll * roll_axis[0];
    o[1] += -omega_roll * roll_axis[1];
    o[2] += -omega_roll * roll_axis[2];
    Real correction_magnitude, dummy;
    clip_quantities(0.025, 1e4, sim.dt, false, a + 0.05 * da, 0.0,
                    correction_magnitude, dummy);
    o[0] += -correction_magnitude * roll_axis[0];
    o[1] += -correction_magnitude * roll_axis[1];
    o[2] += -correction_magnitude * roll_axis[2];
  }
}
void StefanFish::act(const Real t_rlAction, const std::vector<Real> &a) const {
  auto *const cFish = dynamic_cast<CurvatureDefinedFishData *>(myFish);
  if (cFish == nullptr) {
    printf("Someone touched my fish\n");
    abort();
  }
  std::vector<Real> actions = a;
  if (actions.size() == 0) {
    std::cerr << "No actions given to CurvatureDefinedFishData::execute\n";
    MPI_Abort(sim.comm, 1);
  }
  if (bForcedInSimFrame[2] == true && a.size() > 1)
    actions[1] = 0;
  cFish->execute(sim.time, t_rlAction, actions);
}
Real StefanFish::getLearnTPeriod() const {
  auto *const cFish = dynamic_cast<CurvatureDefinedFishData *>(myFish);
  assert(cFish != nullptr);
  return cFish->next_period;
}
Real StefanFish::getPhase(const Real t) const {
  auto *const cFish = dynamic_cast<CurvatureDefinedFishData *>(myFish);
  const Real T0 = cFish->time0;
  const Real Ts = cFish->timeshift;
  const Real Tp = cFish->periodPIDval;
  const Real arg = 2 * M_PI * ((t - T0) / Tp + Ts) + M_PI * cFish->phaseShift;
  const Real phase = std::fmod(arg, 2 * M_PI);
  return (phase < 0) ? 2 * M_PI + phase : phase;
}
std::vector<Real> StefanFish::state() const {
  auto *const cFish = dynamic_cast<CurvatureDefinedFishData *>(myFish);
  assert(cFish != nullptr);
  const Real Tperiod = cFish->Tperiod;
  std::vector<Real> S(25);
  S[0] = position[0];
  S[1] = position[1];
  S[2] = position[2];
  S[3] = quaternion[0];
  S[4] = quaternion[1];
  S[5] = quaternion[2];
  S[6] = quaternion[3];
  S[7] = getPhase(sim.time);
  S[8] = transVel[0] * Tperiod / length;
  S[9] = transVel[1] * Tperiod / length;
  S[10] = transVel[2] * Tperiod / length;
  S[11] = angVel[0] * Tperiod;
  S[12] = angVel[1] * Tperiod;
  S[13] = angVel[2] * Tperiod;
  S[14] = cFish->lastCurv;
  S[15] = cFish->oldrCurv;
  const std::array<Real, 3> locFront = {cFish->sensorLocation[0 * 3 + 0],
                                        cFish->sensorLocation[0 * 3 + 1],
                                        cFish->sensorLocation[0 * 3 + 2]};
  const std::array<Real, 3> locUpper = {cFish->sensorLocation[1 * 3 + 0],
                                        cFish->sensorLocation[1 * 3 + 1],
                                        cFish->sensorLocation[1 * 3 + 2]};
  const std::array<Real, 3> locLower = {cFish->sensorLocation[2 * 3 + 0],
                                        cFish->sensorLocation[2 * 3 + 1],
                                        cFish->sensorLocation[2 * 3 + 2]};
  std::array<Real, 3> shearFront = getShear(locFront);
  std::array<Real, 3> shearUpper = getShear(locLower);
  std::array<Real, 3> shearLower = getShear(locUpper);
  S[16] = shearFront[0] * Tperiod / length;
  S[17] = shearFront[1] * Tperiod / length;
  S[18] = shearFront[2] * Tperiod / length;
  S[19] = shearUpper[0] * Tperiod / length;
  S[20] = shearUpper[1] * Tperiod / length;
  S[21] = shearUpper[2] * Tperiod / length;
  S[22] = shearLower[0] * Tperiod / length;
  S[23] = shearLower[1] * Tperiod / length;
  S[24] = shearLower[2] * Tperiod / length;
  return S;
}
ssize_t StefanFish::holdingBlockID(const std::array<Real, 3> pos) const {
  const std::vector<Info> &velInfo = sim.velInfo();
  for (size_t i = 0; i < velInfo.size(); ++i) {
    std::array<Real, 3> MIN = velInfo[i].pos<Real>(0, 0, 0);
    std::array<Real, 3> MAX = velInfo[i].pos<Real>(
        ScalarBlock::sizeX - 1, ScalarBlock::sizeY - 1, ScalarBlock::sizeZ - 1);
    MIN[0] -= 0.5 * velInfo[i].h;
    MIN[1] -= 0.5 * velInfo[i].h;
    MIN[2] -= 0.5 * velInfo[i].h;
    MAX[0] += 0.5 * velInfo[i].h;
    MAX[1] += 0.5 * velInfo[i].h;
    MAX[2] += 0.5 * velInfo[i].h;
    if (pos[0] >= MIN[0] && pos[1] >= MIN[1] && pos[2] >= MIN[2] &&
        pos[0] <= MAX[0] && pos[1] <= MAX[1] && pos[2] <= MAX[2]) {
      return i;
    }
  }
  return -1;
};
std::array<Real, 3>
StefanFish::getShear(const std::array<Real, 3> pSurf) const {
  const std::vector<Info> &velInfo = sim.velInfo();
  Real myF[3] = {0, 0, 0};
  ssize_t blockIdSurf = holdingBlockID(pSurf);
  if (blockIdSurf >= 0) {
    const auto &skinBinfo = velInfo[blockIdSurf];
    if (obstacleBlocks[blockIdSurf] != nullptr) {
      Real dmin = 1e10;
      ObstacleBlock *const O = obstacleBlocks[blockIdSurf];
      for (int k = 0; k < O->nPoints; ++k) {
        const int ix = O->surface[k]->ix;
        const int iy = O->surface[k]->iy;
        const int iz = O->surface[k]->iz;
        const std::array<Real, 3> p = skinBinfo.pos<Real>(ix, iy, iz);
        const Real d = (p[0] - pSurf[0]) * (p[0] - pSurf[0]) +
                       (p[1] - pSurf[1]) * (p[1] - pSurf[1]) +
                       (p[2] - pSurf[2]) * (p[2] - pSurf[2]);
        if (d < dmin) {
          dmin = d;
          myF[0] = O->fxV[k];
          myF[1] = O->fyV[k];
          myF[2] = O->fzV[k];
        }
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, myF, 3, MPI_Real, MPI_SUM, sim.comm);
  return std::array<Real, 3>{{myF[0], myF[1], myF[2]}};
};
int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &sim.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &sim.size);
  MPI_Barrier(MPI_COMM_WORLD);
  Simulation *sim0 = new Simulation(argc, argv, MPI_COMM_WORLD);
  sim0->init();
  sim0->simulate();
  delete sim0;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
