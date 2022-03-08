#pragma once

#include "Cubism/AMR_MeshAdaptationMPI.h"

using namespace cubism;

template <typename TGrid, typename TLab>
class MeshAdaptation_CUP : public MeshAdaptationMPI<TGrid,TLab>
{
 public:
   Real Rtol_chi;
   Real Ctol_chi;
   typedef typename TGrid::Block BlockType;
   typedef typename TGrid::BlockType::ElementType ElementType;
   typedef typename TGrid::BlockType Block;

   MeshAdaptation_CUP(TGrid &grid, Real Rtol, Real Ctol): MeshAdaptationMPI<TGrid,TLab>(grid,Rtol,Ctol)
   {
    Rtol_chi = 1e-6;
    Ctol_chi = 1e-9;
   }

   virtual void RefineBlocks(BlockType *B[8], BlockInfo parent) override
   {
      static const int nx = BlockType::sizeX;
      static const int ny = BlockType::sizeY;
      static const int nz = BlockType::sizeZ;

      #if 1 //Taylor expansion
            MeshAdaptationMPI<TGrid,TLab>::RefineBlocks(B,parent);
      
      #else //WENO3
            int tid = omp_get_thread_num();
            int offsetX[2] = {0, nx / 2};
            int offsetY[2] = {0, ny / 2};
            int offsetZ[2] = {0, nz / 2};
            TLab &Lab_ = MeshAdaptationMPI<TGrid,TLab>::labs[tid];
      
            for (int K = 0; K < 2; K++)
            for (int J = 0; J < 2; J++)
            for (int I = 0; I < 2; I++)
            {
              BlockType &b = *B[K * 4 + J * 2 + I];
              for (int k = 0; k < nz; k += 2)
              for (int j = 0; j < ny; j += 2)
              for (int i = 0; i < nx; i += 2)
              {
                 const int Nweno = WENOWAVELET;
                 ElementType El[Nweno][Nweno][Nweno];
                 for (int i0 = -Nweno / 2; i0 <= Nweno / 2; i0++)
                 for (int i1 = -Nweno / 2; i1 <= Nweno / 2; i1++)
                 for (int i2 = -Nweno / 2; i2 <= Nweno / 2; i2++)
                     El[i0 + Nweno / 2][i1 + Nweno / 2][i2 + Nweno / 2] = Lab_(i / 2 + offsetX[I] + i0,
                                                                               j / 2 + offsetY[J] + i1,
                                                                               k / 2 + offsetZ[K] + i2);
      
                 ElementType Lines[Nweno][Nweno][2];
                 ElementType Planes[Nweno][4];
                 ElementType Ref[8];
      
                 for (int i2 = -Nweno / 2; i2 <= Nweno / 2; i2++)
                 for (int i1 = -Nweno / 2; i1 <= Nweno / 2; i1++)
                      Kernel_1D(El[0][i1 + Nweno / 2][i2 + Nweno / 2],
                                El[1][i1 + Nweno / 2][i2 + Nweno / 2],
                                El[2][i1 + Nweno / 2][i2 + Nweno / 2],
                                Lines[i1 + Nweno / 2][i2 + Nweno / 2][0],
                                Lines[i1 + Nweno / 2][i2 + Nweno / 2][1]);
                 for (int i2 = -Nweno / 2; i2 <= Nweno / 2; i2++)
                 {
                      Kernel_1D(Lines[0][i2 + Nweno / 2][0], Lines[1][i2 + Nweno / 2][0],
                                Lines[2][i2 + Nweno / 2][0], Planes[i2 + Nweno / 2][0],
                                Planes[i2 + Nweno / 2][1]);
                      Kernel_1D(Lines[0][i2 + Nweno / 2][1], Lines[1][i2 + Nweno / 2][1],
                                Lines[2][i2 + Nweno / 2][1], Planes[i2 + Nweno / 2][2],
                                Planes[i2 + Nweno / 2][3]);
                 }
                 Kernel_1D(Planes[0][0], Planes[1][0], Planes[2][0], Ref[0], Ref[1]);
                 Kernel_1D(Planes[0][1], Planes[1][1], Planes[2][1], Ref[2], Ref[3]);
                 Kernel_1D(Planes[0][2], Planes[1][2], Planes[2][2], Ref[4], Ref[5]);
                 Kernel_1D(Planes[0][3], Planes[1][3], Planes[2][3], Ref[6], Ref[7]);
      
                 b(i, j, k)             = Ref[0];
                 b(i, j, k + 1)         = Ref[1];
                 b(i, j + 1, k)         = Ref[2];
                 b(i, j + 1, k + 1)     = Ref[3];
                 b(i + 1, j, k)         = Ref[4];
                 b(i + 1, j, k + 1)     = Ref[5];
                 b(i + 1, j + 1, k)     = Ref[6];
                 b(i + 1, j + 1, k + 1) = Ref[7];
              }
            }
      #endif
      //clipping for chi field
      for (int K = 0; K < 2; K++)
      for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++)
      {
        BlockType &b = *B[K * 4 + J * 2 + I];
        for (int iz = 0; iz < nz; iz ++)
        for (int iy = 0; iy < ny; iy ++)
        for (int ix = 0; ix < nx; ix ++)
        {
          b(ix  ,iy  ,iz  ).chi = std::max( b(ix  ,iy  ,iz  ).chi,(Real) 0.0);
          b(ix  ,iy  ,iz  ).chi = std::min( b(ix  ,iy  ,iz  ).chi,(Real) 1.0);
        }
      }
   }
   #if 0
       void Kernel_1D(ElementType E0, ElementType E1, ElementType E2, ElementType &left, ElementType &right)
       {
          for (int i = 0 ; i < ElementType::DIM ; i++)
            WENOWavelets3(E0.member(i), E1.member(i), E2.member(i), left.member(i), right.member(i));
       }
       void WENOWavelets3(Real cm, Real c, Real cp, Real &left, Real &right)
       {
          const Real b1  = (c - cm) * (c - cm);
          const Real b2  = (c - cp) * (c - cp);
          Real w1  = (1e-6 + b2) * (1e-6 + b2); // yes, 2 goes to 1 and 1 goes to 2
          Real w2  = (1e-6 + b1) * (1e-6 + b1);
          const Real aux = 1.0 / (w1 + w2);
          w1 *= aux;
          w2 *= aux;
          Real g1, g2;
          g1    = 0.75 * c + 0.25 * cm;
          g2    = 1.25 * c - 0.25 * cp;
          left  = g1 * w1 + g2 * w2;
          g1    = 1.25 * c - 0.25 * cm;
          g2    = 0.75 * c + 0.25 * cp;
          right = g1 * w1 + g2 * w2;
       }
   #endif

   virtual State TagLoadedBlock(TLab &Lab_, BlockInfo & info) override
   {
      //This assumes zero Neumann BCs for velocity
      static const int nx = BlockType::sizeX;
      static const int ny = BlockType::sizeY;
      static const int nz = BlockType::sizeZ;

      const Real inv2h = .5/info.h;  
      Real Linf_2 = 0.0; //vorticity

      //Loop over block and halo cells and refine mesh
      //if any of the cells have chi > 0 and chi < 0.9 (equivalent to grad(chi) != 0)
      bool hasChi = false;
      const int offset = (info.level == MeshAdaptationMPI<TGrid,TLab>::m_refGrid->getlevelMax()-1) ? 2 : 1;
      for(int z=-offset; z<nz+offset; ++z)
      for(int y=-offset; y<ny+offset; ++y)
      for(int x=-offset; x<nx+offset; ++x)
      {
        Lab_(x,y,z).chi = std::min(Lab_(x,y,z).chi,(Real)1.0);
        Lab_(x,y,z).chi = std::max(Lab_(x,y,z).chi,(Real)0.0);
        if (Lab_(x,y,z).chi > 0.0 && Lab_(x,y,z).chi < 0.9)
        {
          hasChi = true;
          break;
        }
      }
      for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
      {
        const ElementType &LW=Lab_(i-1,j  ,k  ), &LE=Lab_(i+1,j  ,k  );
        const ElementType &LS=Lab_(i  ,j-1,k  ), &LN=Lab_(i  ,j+1,k  );
        const ElementType &LF=Lab_(i  ,j  ,k-1), &LB=Lab_(i  ,j  ,k+1);
        const Real omega_x = (LN.w-LS.w) - (LB.v-LF.v);
        const Real omega_y = (LB.u-LF.u) - (LE.w-LW.w);
        const Real omega_z = (LE.v-LW.v) - (LN.u-LS.u);
        const Real omega = omega_x*omega_x + omega_y*omega_y + omega_z*omega_z;
        Linf_2 = max(Linf_2,omega);
      }
      Linf_2 = inv2h * sqrt(Linf_2);

      if ( Linf_2 > MeshAdaptationMPI<TGrid,TLab>::tolerance_for_refinement || hasChi) return Refine;
      if ( Linf_2 < MeshAdaptationMPI<TGrid,TLab>::tolerance_for_compression && hasChi == false) return Compress;
      return Leave;
   }
};
