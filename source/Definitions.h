//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_DataStructures_h
#define CubismUP_3D_DataStructures_h

#include "Base.h"

#include "utils/AlignedAllocator.h"
#include "utils/FDcoeffs.h"

// Cubism dependencies.
#include <Cubism/Grid.h>
#include <Cubism/GridMPI.h>
#include <Cubism/BlockInfo.h>
#ifdef _VTK_
#include <Cubism/SerializerIO_ImageVTK.h>
#endif
#include <Cubism/BlockLab.h>
#include <Cubism/BlockLabMPI.h>

#ifndef CUP_BLOCK_SIZE
#define CUP_BLOCK_SIZE 16
#endif

#include <array>
#include <cassert>
#include <cstddef>   // For `offsetof()`.
#include <iosfwd>
#include <string>

#include "MeshAdaptation_CUP.h"

CubismUP_3D_NAMESPACE_BEGIN

//#define ENERGY_FLUX_SPECTRUM 1

enum { FE_CHI = 0, FE_U, FE_V, FE_W, FE_P, FE_TMPU, FE_TMPV, FE_TMPW };
struct FluidElement
{
  static constexpr int DIM = 8;
  typedef Real RealType;
  Real chi=0, u=0, v=0, w=0, p=0, tmpU=0, tmpV=0, tmpW=0;
  void clear() { chi =0; u =0; v =0; w =0; p =0; tmpU =0; tmpV =0; tmpW =0; }
  ~FluidElement() {}
  FluidElement& operator=(const FluidElement& c) {
    chi = c.chi; u = c.u; v = c.v; w = c.w; p = c.p;
    tmpU = c.tmpU; tmpV = c.tmpV; tmpW = c.tmpW;
    return *this;
  }
  Real & member(int i)
  {
    Real * tmp = & this->chi;
    return *(tmp + i);
  }
  Real magnitude()//used in TagLoadedBlock, to adapt the mesh
  {
      return chi;
  }
  FluidElement &operator*=(const Real a)
  {
    this->chi  *= a;
    this->u    *= a;
    this->v    *= a;
    this->w    *= a;
    this->p    *= a;
    this->tmpU *= a;
    this->tmpV *= a;
    this->tmpW *= a;
    return *this;
  }
  FluidElement &operator+=(const FluidElement &rhs)
  {
    this->chi  += rhs.chi ;
    this->u    += rhs.u   ;
    this->v    += rhs.v   ;
    this->w    += rhs.w   ;
    this->p    += rhs.p   ;
    this->tmpU += rhs.tmpU;
    this->tmpV += rhs.tmpV;
    this->tmpW += rhs.tmpW;
    return *this;
  }
  FluidElement &operator-=(const FluidElement &rhs)
  {
    this->chi  -= rhs.chi ;
    this->u    -= rhs.u   ;
    this->v    -= rhs.v   ;
    this->w    -= rhs.w   ;
    this->p    -= rhs.p   ;
    this->tmpU -= rhs.tmpU;
    this->tmpV -= rhs.tmpV;
    this->tmpW -= rhs.tmpW;
    return *this;
  }
  //only for debug
  FluidElement &operator=(const double a)
  {
    this->chi  = a;
    this->u    = a;
    this->v    = a;
    this->w    = a;
    this->p    = a;
    this->tmpU = a;
    this->tmpV = a;
    this->tmpW = a;
    return *this;
  }
  friend FluidElement operator*(const Real a, FluidElement el)
  {
    return (el *= a);
  }
  friend FluidElement operator+(FluidElement lhs, const FluidElement &rhs)
  {
    return (lhs += rhs);
  }
  friend FluidElement operator-(FluidElement lhs, const FluidElement &rhs)
  {
    return (lhs -= rhs);
  }
};

struct PenalizationHelperElement
{
  static constexpr int DIM = 4;
  typedef Real RealType;
  // Vel b4 pressure projection and after. These quantitites are not penalized.
  Real uPres=0, vPres=0, wPres=0, rhs0=0;
  void clear() { uPres=0; vPres=0; wPres=0; rhs0=0; }
  ~PenalizationHelperElement() {}
  PenalizationHelperElement& operator=(const PenalizationHelperElement& c) {
    uPres = c.uPres; vPres = c.vPres; wPres = c.wPres; rhs0 = c.rhs0;
    return *this;
  }
  Real & member(int i)
  {
    Real * tmp = & this->uPres;
    return *(tmp + i);
  }
  Real magnitude()//used in TagLoadedBlock, to adapt the mesh
  {
    return rhs0;
  }
  PenalizationHelperElement &operator*=(const Real a)
  {
    this->uPres *= a;
    this->vPres *= a;
    this->wPres *= a;
    this->rhs0  *= a;
    return *this;
  }
  PenalizationHelperElement &operator+=(const PenalizationHelperElement &rhs)
  {
    this->uPres+= rhs.uPres;
    this->vPres+= rhs.vPres;
    this->wPres+= rhs.wPres;
    this->rhs0 += rhs.rhs0 ;
    return *this;
  }
  PenalizationHelperElement &operator-=(const PenalizationHelperElement &rhs)
  {
    this->uPres-= rhs.uPres;
    this->vPres-= rhs.vPres;
    this->wPres-= rhs.wPres;
    this->rhs0 -= rhs.rhs0 ;
    return *this;
  }
  //only for debug
  PenalizationHelperElement &operator=(const double a)
  {
    this->uPres = a;
    this->vPres = a;
    this->wPres = a;
    this->rhs0  = a;
    return *this;
  }
  friend PenalizationHelperElement operator*(const Real a, PenalizationHelperElement el)
  {
      return (el *= a);
  }
  friend PenalizationHelperElement operator+(PenalizationHelperElement lhs, const PenalizationHelperElement &rhs)
  {
      return (lhs += rhs);
  }
  friend PenalizationHelperElement operator-(PenalizationHelperElement lhs, const PenalizationHelperElement &rhs)
  {
      return (lhs -= rhs);
  }
};

struct DumpElement {
    DumpReal u, v, w, chi, p;
    DumpElement() : u(0), v(0), w(0), chi(0), p(0) {}
    void clear() { u = v = w = chi = p = 0; }
};

enum BCflag {dirichlet, periodic, wall, freespace};
inline BCflag string2BCflag(const std::string &strFlag)
{
  if (strFlag == "periodic") return periodic;
  else
  if (strFlag == "dirichlet") return dirichlet;
  else
  if (strFlag == "wall") return wall;
  else
  if (strFlag == "freespace") return freespace;
  else {
    fprintf(stderr,"BC not recognized %s\n",strFlag.c_str()); fflush(0);abort();
    return periodic; // dummy
  }
}

struct StreamerDiv
{
  static const int channels = 1;
  template <typename T>
  static void operate(const FluidElement& input, T output[1])
  { output[0] = input.p; }

  template <typename T>
  static void operate(const T input[1], FluidElement& output)
  { output.p = input[0]; }
};

template <typename TElement>
struct BaseBlock
{
  //these identifiers are required by cubism!
  static constexpr int BS = CUP_BLOCK_SIZE;
  static constexpr int sizeX = BS;
  static constexpr int sizeY = BS;
  static constexpr int sizeZ = BS;
  static constexpr std::array<int, 3> sizeArray = {BS, BS, BS};
  typedef TElement ElementType;
  typedef TElement element_type;
  typedef Real   RealType;
  //__attribute__((aligned(32)))
  TElement data[sizeZ][sizeY][sizeX];

  FDBlockCoeffs_x __attribute__((__aligned__(32))) fd_cx; // finite-difference single coefficients
  FDBlockCoeffs_y __attribute__((__aligned__(32))) fd_cy; // finite-difference single coefficients
  FDBlockCoeffs_z __attribute__((__aligned__(32))) fd_cz; // finite-difference single coefficients
  //Real __attribute__((__aligned__(32))) invh_x[sizeX]; // pre-compute inverse mesh-spacings
  //Real __attribute__((__aligned__(32))) invh_y[sizeY]; // pre-compute inverse mesh-spacings
  //Real __attribute__((__aligned__(32))) invh_z[sizeZ]; // pre-compute inverse mesh-spacings
  std::array<Real, 3> min_pos;
  std::array<Real, 3> max_pos;

  //required from Grid.h
  void clear()
  {
      TElement * entry = &data[0][0][0];
      const int N = sizeX*sizeY*sizeZ;
      for(int i=0; i<N; ++i) entry[i].clear();
  }

  TElement& operator()(int ix, int iy=0, int iz=0)
  {
    assert(ix>=0); assert(ix<sizeX);
    assert(iy>=0); assert(iy<sizeY);
    assert(iz>=0); assert(iz<sizeZ);
    return data[iz][iy][ix];
  }

  const TElement& operator()(int ix, int iy = 0, int iz = 0) const
  {
    assert(ix>=0); assert(ix<sizeX);
    assert(iy>=0); assert(iy<sizeY);
    assert(iz>=0); assert(iz<sizeZ);
    return data[iz][iy][ix];
  }

  template <typename Streamer>
  inline void Write(std::ofstream& output, Streamer streamer) const
  {
    for(int iz=0; iz<sizeZ; iz++)
      for(int iy=0; iy<sizeY; iy++)
        for(int ix=0; ix<sizeX; ix++)
          streamer.operate(data[iz][iy][ix], output);
  }

  template <typename Streamer>
  inline void Read(std::ifstream& input, Streamer streamer)
  {
    for(int iz=0; iz<sizeZ; iz++)
      for(int iy=0; iy<sizeY; iy++)
        for(int ix=0; ix<sizeX; ix++)
          streamer.operate(input, data[iz][iy][ix]);
  }
};

struct StreamerChi
{
    static const int NCHANNELS = 1;
    static const int CLASS = 0;

    template <typename TBlock, typename T>
    static inline void operate(const TBlock& b, const int ix, const int iy, const int iz, T output[NCHANNELS])
    {
      output[0] = b(ix,iy,iz).chi;
    }
    static std::string prefix()
    {
      return std::string("chi_");
    }

    static const char * getAttributeName() { return "Scalar"; }
};

struct StreamerVelocityVector
{
    static const int NCHANNELS = 3;
    static const int CLASS = 0;

    // Write
    template <typename TBlock, typename T>
    static inline void operate(const TBlock& b, const int ix, const int iy, const int iz, T output[NCHANNELS])
    {
        output[0] = b(ix,iy,iz).u;
        output[1] = b(ix,iy,iz).v;
        output[2] = b(ix,iy,iz).w;
    }

    // Read
    template <typename TBlock, typename T>
    static inline void operate(TBlock& b, const T input[NCHANNELS], const int ix, const int iy, const int iz)
    {
        b(ix,iy,iz).u = input[0];
        b(ix,iy,iz).v = input[1];
        b(ix,iy,iz).w = input[2];
    }

    static std::string prefix()
    {
      return std::string("vel_");
    }

    static const char * getAttributeName() { return "Vector"; }
};

struct StreamerTmpVector
{
    static const int NCHANNELS = 3;
    static const int CLASS = 0;

    // Write
    template <typename TBlock, typename T>
    static inline void operate(const TBlock& b, const int ix, const int iy, const int iz, T output[NCHANNELS])
    {
        output[0] = b(ix,iy,iz).tmpU;
        output[1] = b(ix,iy,iz).tmpV;
        output[2] = b(ix,iy,iz).tmpW;
    }

    static std::string prefix()
    {
      return std::string("tmp_");
    }

    static const char * getAttributeName() { return "Vector"; }
};

struct StreamerPressure
{
    static const int NCHANNELS = 1;
    static const int CLASS = 0;

    template <typename TBlock, typename T>
    static inline void operate(const TBlock& b, const int ix, const int iy, const int iz, T output[NCHANNELS])
    {
      output[0] = b(ix,iy,iz).p;
    }

    static std::string prefix()
    {
      return std::string("pres_");
    }

    static const char * getAttributeName() { return "Scalar"; }
};


template<typename BlockType, template<typename X> class allocator=std::allocator>
class BlockLabBC: public cubism::BlockLab<BlockType,allocator>
{
  typedef typename BlockType::ElementType ElementTypeBlock;
  static constexpr int sizeX = BlockType::sizeX;
  static constexpr int sizeY = BlockType::sizeY;
  static constexpr int sizeZ = BlockType::sizeZ;

  // Each of these flags for now supports either 1 or anything else
  // If 1 then periodic, if 2 then else dirichlet==absorbing==freespace
  // In reality these 3 BC should be different, but since we only use second ord
  // finite differences on cell centers in practice they are the same.
  // (this does not equally hold for the Poisson solver)
  // In the future we might have to support more general ways to define BC
  BCflag BCX = dirichlet, BCY = dirichlet, BCZ = dirichlet;

  // Used for Boundary Conditions:
  // Apply bc on face of direction dir and side side (0 or 1):
  template<int dir, int side> void applyBCfaceOpen(const bool coarse = false)
  {
    if (!coarse)
    {
      auto * const cb = this->m_cacheBlock;
  
      int s[3] = {0,0,0}, e[3] = {0,0,0};
      const int* const stenBeg = this->m_stencilStart;
      const int* const stenEnd = this->m_stencilEnd;
      s[0] =  dir==0 ? (side==0 ? stenBeg[0] : sizeX ) : 0;
      s[1] =  dir==1 ? (side==0 ? stenBeg[1] : sizeY ) : 0;
      s[2] =  dir==2 ? (side==0 ? stenBeg[2] : sizeZ ) : 0;
      e[0] =  dir==0 ? (side==0 ? 0 : sizeX + stenEnd[0]-1 ) : sizeX;
      e[1] =  dir==1 ? (side==0 ? 0 : sizeY + stenEnd[1]-1 ) : sizeY;
      e[2] =  dir==2 ? (side==0 ? 0 : sizeZ + stenEnd[2]-1 ) : sizeZ;
      for(int iz=s[2]; iz<e[2]; iz++)
      for(int iy=s[1]; iy<e[1]; iy++)
      for(int ix=s[0]; ix<e[0]; ix++)
        cb->Access(ix-stenBeg[0], iy-stenBeg[1], iz-stenBeg[2]) = cb->Access
          (
            ( dir==0 ? (side==0 ? 0 : sizeX-1 ) : ix ) - stenBeg[0],
            ( dir==1 ? (side==0 ? 0 : sizeY-1 ) : iy ) - stenBeg[1],
            ( dir==2 ? (side==0 ? 0 : sizeZ-1 ) : iz ) - stenBeg[2]
          );      
    }
    else
    {
      auto * const cb = this->m_CoarsenedBlock;
  
      int s[3] = {0,0,0}, e[3] = {0,0,0};
      const int eI[3] = {(this->m_stencilEnd[0])/2 + 1 + this->m_InterpStencilEnd[0] -1,
                         (this->m_stencilEnd[1])/2 + 1 + this->m_InterpStencilEnd[1] -1,
                         (this->m_stencilEnd[2])/2 + 1 + this->m_InterpStencilEnd[2] -1};
      const int sI[3] = {(this->m_stencilStart[0]-1)/2+  this->m_InterpStencilStart[0],
                         (this->m_stencilStart[1]-1)/2+  this->m_InterpStencilStart[1],
                         (this->m_stencilStart[2]-1)/2+  this->m_InterpStencilStart[2]};

      const int* const stenBeg = sI;
      const int* const stenEnd = eI;
      s[0] =  dir==0 ? (side==0 ? stenBeg[0] : sizeX/2 ) : 0;
      s[1] =  dir==1 ? (side==0 ? stenBeg[1] : sizeY/2 ) : 0;
      s[2] =  dir==2 ? (side==0 ? stenBeg[2] : sizeZ/2 ) : 0;
      e[0] =  dir==0 ? (side==0 ? 0 : sizeX/2 + stenEnd[0]-1 ) : sizeX/2;
      e[1] =  dir==1 ? (side==0 ? 0 : sizeY/2 + stenEnd[1]-1 ) : sizeY/2;
      e[2] =  dir==2 ? (side==0 ? 0 : sizeZ/2 + stenEnd[2]-1 ) : sizeZ/2;
      for(int iz=s[2]; iz<e[2]; iz++)
      for(int iy=s[1]; iy<e[1]; iy++)
      for(int ix=s[0]; ix<e[0]; ix++)
      {
        cb->Access(ix-stenBeg[0], iy-stenBeg[1], iz-stenBeg[2]) = cb->Access
          ( ( dir==0 ? (side==0 ? 0 : sizeX/2-1 ) : ix ) - stenBeg[0],
            ( dir==1 ? (side==0 ? 0 : sizeY/2-1 ) : iy ) - stenBeg[1],
            ( dir==2 ? (side==0 ? 0 : sizeZ/2-1 ) : iz ) - stenBeg[2]);
      }

      //tensorial edges and corners also filled (this is necessary for the coarse block!)
      int aux = coarse ? 2:1;
      const int bsize[3] = {sizeX/aux, sizeY/aux, sizeZ/aux};
      int s_[3], e_[3];
      s_[dir] = stenBeg[dir]*(1-side) + bsize[dir]*side;
      e_[dir] = (bsize[dir]-1+stenEnd[dir])*side;
      const int d1 = (dir + 1) % 3;
      const int d2 = (dir + 2) % 3;
      for(int b=0; b<2; ++b)
      for(int a=0; a<2; ++a)
      {
        s_[d1] = stenBeg[d1] + a*b*(bsize[d1] - stenBeg[d1]);
        s_[d2] = stenBeg[d2] + (a-a*b)*(bsize[d2] - stenBeg[d2]);
        e_[d1] = (1-b+a*b)*(bsize[d1] - 1 + stenEnd[d1]);
        e_[d2] = (a+b-a*b)*(bsize[d2] - 1 + stenEnd[d2]);
        for(int iz=s_[2]; iz<e_[2]; iz++)
        for(int iy=s_[1]; iy<e_[1]; iy++)
        for(int ix=s_[0]; ix<e_[0]; ix++)
        {
          cb->Access(ix-stenBeg[0], iy-stenBeg[1], iz-stenBeg[2]) = dir==0?
          cb->Access(side*(bsize[0]-1)-stenBeg[0], iy-stenBeg[1], iz-stenBeg[2]) : (dir==1?
          cb->Access(ix-stenBeg[0], side*(bsize[1]-1)-stenBeg[1], iz-stenBeg[2]) :
          cb->Access(ix-stenBeg[0], iy-stenBeg[1], side*(bsize[2]-1)-stenBeg[2]));
        }
      }
    }
  }
  template<int dir, int side> void applyBCfaceWall(const bool coarse=false)
  {
    std::cout << "applyBCfaceWall not ready for coarse block" << std::endl; abort();
    if (!coarse)
    {
      auto * const cb = this->m_cacheBlock;
  
      int s[3] = {0,0,0}, e[3] = {0,0,0};
      const int* const stenBeg = this->m_stencilStart;
      const int* const stenEnd = this->m_stencilEnd;
      s[0] =  dir==0 ? (side==0 ? stenBeg[0] : sizeX ) : 0;
      s[1] =  dir==1 ? (side==0 ? stenBeg[1] : sizeY ) : 0;
      s[2] =  dir==2 ? (side==0 ? stenBeg[2] : sizeZ ) : 0;
      e[0] =  dir==0 ? (side==0 ? 0 : sizeX + stenEnd[0]-1 ) : sizeX;
      e[1] =  dir==1 ? (side==0 ? 0 : sizeY + stenEnd[1]-1 ) : sizeY;
      e[2] =  dir==2 ? (side==0 ? 0 : sizeZ + stenEnd[2]-1 ) : sizeZ;
      for(int iz=s[2]; iz<e[2]; iz++)
      for(int iy=s[1]; iy<e[1]; iy++)
      for(int ix=s[0]; ix<e[0]; ix++)
      {
        auto& DST = cb->Access(ix-stenBeg[0], iy-stenBeg[1], iz-stenBeg[2]);
        const auto& SRCV = cb->Access (
            ( dir==0 ? (side==0 ? -1 -ix : 2*sizeX -1 -ix ) : ix ) - stenBeg[0],
            ( dir==1 ? (side==0 ? -1 -iy : 2*sizeY -1 -iy ) : iy ) - stenBeg[1],
            ( dir==2 ? (side==0 ? -1 -iz : 2*sizeZ -1 -iz ) : iz ) - stenBeg[2]
          );
        const auto& SRCP = cb->Access (
            ( dir==0 ? (side==0 ? 0 : sizeX-1 ) : ix ) - stenBeg[0],
            ( dir==1 ? (side==0 ? 0 : sizeY-1 ) : iy ) - stenBeg[1],
            ( dir==2 ? (side==0 ? 0 : sizeZ-1 ) : iz ) - stenBeg[2]
          );
        DST.p =    SRCP.p; DST.chi  =  0;
        DST.u =  - SRCV.u; DST.tmpU =  - SRCV.tmpU;
        DST.v =  - SRCV.v; DST.tmpV =  - SRCV.tmpV;
        DST.w =  - SRCV.w; DST.tmpW =  - SRCV.tmpW;
      }
    }
    else
    {
      auto * const cb = this->m_CoarsenedBlock;
  
      int s[3] = {0,0,0}, e[3] = {0,0,0};

      const int eI[3] = {(this->m_stencilEnd[0])/2 + 1 + this->m_InterpStencilEnd[0] -1,
                         (this->m_stencilEnd[1])/2 + 1 + this->m_InterpStencilEnd[1] -1,
                         (this->m_stencilEnd[2])/2 + 1 + this->m_InterpStencilEnd[2] -1};
      const int sI[3] = {(this->m_stencilStart[0]-1)/2+  this->m_InterpStencilStart[0],
                         (this->m_stencilStart[1]-1)/2+  this->m_InterpStencilStart[1],
                         (this->m_stencilStart[2]-1)/2+  this->m_InterpStencilStart[2]};

      const int* const stenBeg = sI;
      const int* const stenEnd = eI;

      s[0] =  dir==0 ? (side==0 ? stenBeg[0] : sizeX/2 ) : 0;
      s[1] =  dir==1 ? (side==0 ? stenBeg[1] : sizeY/2 ) : 0;
      s[2] =  dir==2 ? (side==0 ? stenBeg[2] : sizeZ/2 ) : 0;
      e[0] =  dir==0 ? (side==0 ? 0 : sizeX/2 + stenEnd[0]-1 ) : sizeX/2;
      e[1] =  dir==1 ? (side==0 ? 0 : sizeY/2 + stenEnd[1]-1 ) : sizeY/2;
      e[2] =  dir==2 ? (side==0 ? 0 : sizeZ/2 + stenEnd[2]-1 ) : sizeZ/2;
      for(int iz=s[2]; iz<e[2]; iz++)
      for(int iy=s[1]; iy<e[1]; iy++)
      for(int ix=s[0]; ix<e[0]; ix++)
      {
        auto& DST = cb->Access(ix-stenBeg[0], iy-stenBeg[1], iz-stenBeg[2]);
        const auto& SRCV = cb->Access (
            ( dir==0 ? (side==0 ? -1 -ix : 2*sizeX/2 -1 -ix ) : ix ) - stenBeg[0],
            ( dir==1 ? (side==0 ? -1 -iy : 2*sizeY/2 -1 -iy ) : iy ) - stenBeg[1],
            ( dir==2 ? (side==0 ? -1 -iz : 2*sizeZ/2 -1 -iz ) : iz ) - stenBeg[2]
          );
        const auto& SRCP = cb->Access (
            ( dir==0 ? (side==0 ? 0 : sizeX/2-1 ) : ix ) - stenBeg[0],
            ( dir==1 ? (side==0 ? 0 : sizeY/2-1 ) : iy ) - stenBeg[1],
            ( dir==2 ? (side==0 ? 0 : sizeZ/2-1 ) : iz ) - stenBeg[2]
          );
        DST.p =    SRCP.p; DST.chi  =  0;
        DST.u =  - SRCV.u; DST.tmpU =  - SRCV.tmpU;
        DST.v =  - SRCV.v; DST.tmpV =  - SRCV.tmpV;
        DST.w =  - SRCV.w; DST.tmpW =  - SRCV.tmpW;
      }

    }

  }

 public:

  typedef typename BlockType::ElementType ElementType;
  virtual void TestInterp(ElementType *C[3][3][3], ElementType &R, int x, int y, int z, const std::vector<int> & selcomponents) override
  {
     cubism::BlockLab<BlockType,allocator>::TestInterp(C,R,x,y,z,selcomponents);
     if (R.chi < 0.0) R.chi = 0.0;
     if (R.chi > 1.0) R.chi = 1.0;
  }

  void setBC(const BCflag _BCX, const BCflag _BCY, const BCflag _BCZ) {
    BCX=_BCX; BCY=_BCY; BCZ=_BCZ;
  }
  bool is_xperiodic() { return BCX == periodic; }
  bool is_yperiodic() { return BCY == periodic; }
  bool is_zperiodic() { return BCZ == periodic; }

  BlockLabBC() = default;
  BlockLabBC(const BlockLabBC&) = delete;
  BlockLabBC& operator=(const BlockLabBC&) = delete;

  // Called by Cubism:
  void _apply_bc(const cubism::BlockInfo& info, const Real t=0, const bool coarse = false)
  {
    if(BCX == periodic) {   /* PERIODIC */ }
    else if (BCX == wall) { /* WALL */
      if(info.index[0]==0 )          this->template applyBCfaceWall<0,0>(coarse);
      if(info.index[0]==this->NX-1 ) this->template applyBCfaceWall<0,1>(coarse);
    } else { /* dirichlet==absorbing==freespace */
      if(info.index[0]==0 )          this->template applyBCfaceOpen<0,0>(coarse);
      if(info.index[0]==this->NX-1 ) this->template applyBCfaceOpen<0,1>(coarse);
    }

    if(BCY == periodic) {   /* PERIODIC */ }
    else if (BCY == wall) { /* WALL */
      if(info.index[1]==0 )          this->template applyBCfaceWall<1,0>(coarse);
      if(info.index[1]==this->NY-1 ) this->template applyBCfaceWall<1,1>(coarse);
    } else { /* dirichlet==absorbing==freespace */
      if(info.index[1]==0 )          this->template applyBCfaceOpen<1,0>(coarse);
      if(info.index[1]==this->NY-1 ) this->template applyBCfaceOpen<1,1>(coarse);
    }

    if(BCZ == periodic) {   /* PERIODIC */ }
    else if (BCZ == wall) { /* WALL */
      if(info.index[2]==0 )          this->template applyBCfaceWall<2,0>(coarse);
      if(info.index[2]==this->NZ-1 ) this->template applyBCfaceWall<2,1>(coarse);
    } else { /* dirichlet==absorbing==freespace */
      if(info.index[2]==0 )          this->template applyBCfaceOpen<2,0>(coarse);
      if(info.index[2]==this->NZ-1 ) this->template applyBCfaceOpen<2,1>(coarse);
    }
  }
};

using FluidBlock = BaseBlock<FluidElement>;
using FluidGrid    = cubism::Grid<FluidBlock, aligned_allocator>;
using FluidGridMPI = cubism::GridMPI<FluidGrid>;

using PenalizationBlock   = BaseBlock<PenalizationHelperElement>;
using PenalizationGrid    = cubism::Grid<PenalizationBlock, aligned_allocator>;
using PenalizationGridMPI = cubism::GridMPI<PenalizationGrid>;

using Lab          = BlockLabBC<FluidBlock, aligned_allocator>;
using LabMPI       = cubism::BlockLabMPI<Lab,FluidGridMPI>;

using AMR = MeshAdaptation_CUP<FluidGridMPI,LabMPI>;

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_DataStructures_h
