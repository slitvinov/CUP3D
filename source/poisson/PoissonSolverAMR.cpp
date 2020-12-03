//
//  CubismUP_3D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

#include "PoissonSolverAMR.h"

namespace cubismup3d {

Real PoissonSolverAMR::computeRelativeCorrection() const
{
  Real sumRHS = 0, sumABS = 0;
  const std::vector<cubism::BlockInfo>& vInfo = grid.getBlocksInfo();
  #pragma omp parallel for schedule(static) reduction(+ : sumRHS, sumABS)
  for(size_t i=0; i<vInfo.size(); ++i)
  {
    BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
    const double h3 = vInfo[i].h_gridpoint * vInfo[i].h_gridpoint * vInfo[i].h_gridpoint;
    for(int iz=0; iz<BlockType::sizeZ; iz++)
    for(int iy=0; iy<BlockType::sizeY; iy++)
    for(int ix=0; ix<BlockType::sizeX; ix++)
    {
      sumABS += h3 * std::fabs( b.tmp[iz][iy][ix] );
      sumRHS += h3 * b.tmp[iz][iy][ix];
    }
  }
  double sums[2] = {sumRHS, sumABS};
  MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE,MPI_SUM, m_comm);
  sums[1] = std::max(std::numeric_limits<double>::epsilon(), sums[1]);
  const Real correction = sums[0] / sums[1];
  if(sim.verbose)
    printf("Relative RHS correction:%e / %e\n", sums[0], sums[1]);
  return correction;
}

Real PoissonSolverAMR::computeAverage() const
{
  Real avgP = 0;
  const std::vector<cubism::BlockInfo>& vInfo = grid.getBlocksInfo();
  #pragma omp parallel for schedule(static) reduction(+ : avgP)
  for(size_t i=0; i<vInfo.size(); ++i)
  {
    const double h3 = vInfo[i].h_gridpoint * vInfo[i].h_gridpoint * vInfo[i].h_gridpoint;
    BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
    for(int iz=0; iz<BlockType::sizeZ; iz++)
    for(int iy=0; iy<BlockType::sizeY; iy++)
    for(int ix=0; ix<BlockType::sizeX; ix++)
      avgP += h3 * b.tmp[iz][iy][ix];
  }
  MPI_Allreduce(MPI_IN_PLACE, &avgP, 1, MPIREAL, MPI_SUM, m_comm);
  avgP /= sim.extent[0] * sim.extent[1] * sim.extent[2];
  return avgP;
}


#ifdef PRECOND
double PoissonSolverAMR::getA_local(int I1,int I2)
{
  static constexpr int BSX = BlockType::sizeX;
  static constexpr int BSY = BlockType::sizeY;
  
  int k1 =  I1 / (BSX*BSY);
  int j1 = (I1 - k1*(BSX*BSY) ) / BSX;
  int i1 =  I1 - k1*(BSX*BSY) - j1*BSX;

  int k2 =  I2 / (BSX*BSY);
  int j2 = (I2 - k2*(BSX*BSY) ) / BSX;
  int i2 =  I2 - k2*(BSX*BSY) - j2*BSX;

  if (i1==i2 && j1==j2 && k1==k2)
  {
    return 6.0;
  }
  else if (abs(i1-i2) + abs(j1-j2) + abs(k1-k2) == 1)
  {
    return -1.0;
  }
  else
  {
    return 0.0;
  }
}

PoissonSolverAMR::PoissonSolverAMR(SimulationData& s): sim(s),findLHS(s)
{
  if (StreamerDiv::channels != 1) {
    fprintf(stderr, "PoissonSolverScalar_MPI(): Error: StreamerDiv::channels is %d (should be 1)\n",
            StreamerDiv::channels);
    fflush(0); exit(1);
  }

  std::vector<std::vector<double>> L;

  int BSX = BlockType::sizeX;
  int BSY = BlockType::sizeY;
  int BSZ = BlockType::sizeZ;
  int N = BSX*BSY*BSZ;
  L.resize(N);

  for (int i = 0 ; i<N ; i++)
  {
    L[i].resize(i+1);
  }
  for (int i = 0 ; i<N ; i++)
  {
    double s1=0;
    for (int k=0; k<=i-1; k++)
      s1 += L[i][k]*L[i][k];
    L[i][i] = sqrt(getA_local(i,i) - s1);
    for (int j=i+1; j<N; j++)
    {
      double s2 = 0;
      for (int k=0; k<=i-1; k++)
        s2 += L[i][k]*L[j][k];
      L[j][i] = (getA_local(j,i)-s2) / L[i][i];
    }
  }
  L_row.resize(omp_get_max_threads());
  L_col.resize(omp_get_max_threads());
  Ld.resize(omp_get_max_threads());
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    L_row[tid].resize(N);
    L_col[tid].resize(N);

    for (int i = 0 ; i<N ; i++)
    {
      Ld[tid].push_back(1.0/L[i][i]);
      for (int j = 0 ; j < i ; j++)
      {
        if ( abs(L[i][j]) > 1e-10 ) L_row[tid][i].push_back({j,L[i][j]});
      }
    }    
    for (int j = 0 ; j<N ; j++)
    {
      for (int i = j ; i < N ; i++)
      {
        if ( abs(L[i][j]) > 1e-10 ) L_col[tid][j].push_back({i,L[i][j]});
      }
    }
  }
}

void PoissonSolverAMR::FindZ()
{
  static constexpr int BSX = BlockType::sizeX;
  static constexpr int BSY = BlockType::sizeY;
  static constexpr int BSZ = BlockType::sizeZ;
  static constexpr int N   = BSX*BSY*BSZ;

  std::vector<cubism::BlockInfo>& vInfo = grid.getBlocksInfo();
  const size_t Nblocks = vInfo.size();

  #pragma omp parallel
  { 
    int tid = omp_get_thread_num();
    #pragma omp for schedule(runtime)
    for (size_t i=0; i < Nblocks; i++)
    {
      BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
      const double invh = 1.0/vInfo[i].h_gridpoint;
  
      //1. z = L^{-1}r
      for (int I = 0; I < N ; I++)
      {
        double rhs = 0.0;
        for (size_t jj = 0 ; jj < L_row[tid][I].size(); jj++)
        {
          int J = L_row[tid][I][jj].first;
          double LIJ = L_row[tid][I][jj].second;
          int iz =  J / (BSX*BSY);
          int iy = (J - iz*(BSX*BSY) )/ BSX;
          int ix =  J - iz*(BSX*BSY) - iy * BSX;
          rhs += LIJ*b(ix,iy,iz).zVector;
        }    
        int iz =  I / (BSX*BSY);
        int iy = (I - iz*(BSX*BSY) )/ BSX;
        int ix =  I - iz*(BSX*BSY) - iy * BSX;
        b(ix,iy,iz).zVector = (b(ix,iy,iz).rVector - rhs) *Ld[tid][I];
      }
  

      //2. z = L^T{-1}r
      for (int I = N-1; I >= 0 ; I--)
      {
        double rhs = 0.0;
        for (size_t jj = 0 ; jj < L_col[tid][I].size(); jj++)
        {
          int J = L_col[tid][I][jj].first;
          double LJI = L_col[tid][I][jj].second;

          int iz =  J / (BSX*BSY);
          int iy = (J - iz*(BSX*BSY) )/ BSX;
          int ix =  J - iz*(BSX*BSY) - iy * BSX;
          rhs += LJI*b(ix,iy,iz).zVector;
        }
        int iz =  I / (BSX*BSY);
        int iy = (I - iz*(BSX*BSY) )/ BSX;
        int ix =  I - iz*(BSX*BSY) - iy * BSX;
        b(ix,iy,iz).zVector = (b(ix,iy,iz).zVector - rhs) *Ld[tid][I];
      }

      for (int iz=0;iz<BSZ;iz++)
      for (int iy=0;iy<BSY;iy++)
      for (int ix=0;ix<BSX;ix++)
        b(ix,iy,iz).zVector = -invh*b(ix,iy,iz).zVector;
    }
  }
}
#else
PoissonSolverAMR::PoissonSolverAMR(SimulationData& s): sim(s),findLHS(s)
{
  if (StreamerDiv::channels != 1) {
    fprintf(stderr, "PoissonSolverScalar_MPI(): Error: StreamerDiv::channels is %d (should be 1)\n",
            StreamerDiv::channels);
    fflush(0); exit(1);
  }
}
#endif


void PoissonSolverAMR::getZ()
{
  static constexpr int BSX = BlockType::sizeX;
  static constexpr int BSY = BlockType::sizeY;
  static constexpr int BSZ = BlockType::sizeZ;
  static constexpr int N   = BSX*BSY*BSZ;

  std::vector<cubism::BlockInfo>& vInfo = grid.getBlocksInfo();
  const size_t Nblocks = vInfo.size();

  #pragma omp parallel
  { 
    int tid = omp_get_thread_num();
    #pragma omp for schedule(runtime)
    for (size_t i=0; i < Nblocks; i++)
    {
      BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
      const double invh = 1.0/vInfo[i].h_gridpoint;
  
      std::vector <double> tempZ (BSX*BSY*BSZ);
      for(int iz=0; iz<BlockType::sizeZ; iz++)
      for(int iy=0; iy<BlockType::sizeY; iy++)
      for(int ix=0; ix<BlockType::sizeX; ix++)
      {
        tempZ[iz*BSX*BSY + iy*BSX + ix ] = b(ix,iy,iz).zVector;
      }

      //1. z = L^{-1}r
      for (int I = 0; I < N ; I++)
      {
        double rhs = 0.0;
        for (size_t jj = 0 ; jj < L_row[tid][I].size(); jj++)
        {
          int J = L_row[tid][I][jj].first;
          double LIJ = L_row[tid][I][jj].second;
          int iz =  J / (BSX*BSY);
          int iy = (J - iz*(BSX*BSY) )/ BSX;
          int ix =  J - iz*(BSX*BSY) - iy * BSX;
          rhs += LIJ*b(ix,iy,iz).zVector;
        }    
        int iz =  I / (BSX*BSY);
        int iy = (I - iz*(BSX*BSY) )/ BSX;
        int ix =  I - iz*(BSX*BSY) - iy * BSX;
        b(ix,iy,iz).zVector = (tempZ[iz*BSX*BSY + iy*BSX + ix ] - rhs) *Ld[tid][I];
      }
  

      //2. z = L^T{-1}r
      for (int I = N-1; I >= 0 ; I--)
      {
        double rhs = 0.0;
        for (size_t jj = 0 ; jj < L_col[tid][I].size(); jj++)
        {
          int J = L_col[tid][I][jj].first;
          double LJI = L_col[tid][I][jj].second;

          int iz =  J / (BSX*BSY);
          int iy = (J - iz*(BSX*BSY) )/ BSX;
          int ix =  J - iz*(BSX*BSY) - iy * BSX;
          rhs += LJI*b(ix,iy,iz).zVector;
        }
        int iz =  I / (BSX*BSY);
        int iy = (I - iz*(BSX*BSY) )/ BSX;
        int ix =  I - iz*(BSX*BSY) - iy * BSX;
        b(ix,iy,iz).zVector = (b(ix,iy,iz).zVector - rhs) *Ld[tid][I];
      }

      for (int iz=0;iz<BSZ;iz++)
      for (int iy=0;iy<BSY;iy++)
      for (int ix=0;ix<BSX;ix++)
        b(ix,iy,iz).zVector = -invh*b(ix,iy,iz).zVector;
    }
  }
}


#if 0
void PoissonSolverAMR::solve()
{
  sim.startProfiler("PoissonSolverAMR :: step one");
  
  static constexpr int BSX = BlockType::sizeX;
  static constexpr int BSY = BlockType::sizeY;
  static constexpr int BSZ = BlockType::sizeZ;

  std::vector<cubism::BlockInfo>& vInfo = grid.getBlocksInfo();

  const size_t Nblocks = vInfo.size();
  const size_t N = BSX*BSY*BSZ*Nblocks;
  size_t Nsystem;
  MPI_Allreduce(&N,&Nsystem,1,MPI_LONG,MPI_SUM,m_comm);

  /*
  The following vectors are needed:

  p: conjugate vector
  r: residual vector
  x: current solution estimate
  s: best solution estimate
  data: rhs of system, or the current Ax (LHS * x)

  They are stored in tmpU,tmpV,tmpW, u and v. 

  */


  double rMAX = 1e50;
  std::vector <Real> storeGridElements (6*N);

  #pragma omp parallel for schedule(runtime)
  for (size_t i=0; i < Nblocks; i++)
  {
    BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
    const size_t offset = _offset( vInfo[i] );
    for(int iz=0; iz<BlockType::sizeZ; iz++)
    for(int iy=0; iy<BlockType::sizeY; iy++)
    for(int ix=0; ix<BlockType::sizeX; ix++)
    {
      const size_t src_index = _dest(offset, iz, iy, ix);     
      storeGridElements[6*src_index  ] = b(ix,iy,iz).pVector ;
      storeGridElements[6*src_index+1] = b(ix,iy,iz).rVector ;
      storeGridElements[6*src_index+2] = b(ix,iy,iz).xVector ;
      storeGridElements[6*src_index+3] = b(ix,iy,iz).sVector ;
      storeGridElements[6*src_index+4] = b(ix,iy,iz).AxVector;
      storeGridElements[6*src_index+5] = b(ix,iy,iz).zVector;
    }
  }

  //const double max_error = 1e-12;
  //const double max_rel_error = 1e-3;
  const double max_error = 1e-9;
  const double max_rel_error = 1e-2;
  double err_min         = 1e50;
  double err             = 1;
  double alpha           = 1.0;
  double beta            = 0.0;
  double rk_rk,rkp1_rkp1,pAp;
  #ifdef PRECOND
  double rk_zk;
  double rkp1_zkp1;
  #endif

  //Copy system RHS (stored in data) to rVector
  #pragma omp parallel for schedule(runtime)
  for (size_t i=0; i < Nblocks; i++)
  {
    BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
    for(int iz=0; iz<BlockType::sizeZ; iz++)
    for(int iy=0; iy<BlockType::sizeY; iy++)
    for(int ix=0; ix<BlockType::sizeX; ix++)
    {
      b(ix,iy,iz).rVector = b.tmp[iz][iy][ix];
      b(ix,iy,iz).pVector = b(ix,iy,iz).xVector;//this is done because Get_LHS works with pVector
    }
  }
  findLHS(0); // AxVector <-- A*x_{0}, x_0 = pressure

  //rVector <-- rVector - alpha * AxVector
  //rk_rk = rVector /cdot rVector
  rk_rk = 0.0;
  #pragma omp parallel for reduction(+:rk_rk)
  for(size_t i=0; i< Nblocks; i++)
  {
    BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
    for(int iz=0; iz<BlockType::sizeZ; iz++)
    for(int iy=0; iy<BlockType::sizeY; iy++)
    for(int ix=0; ix<BlockType::sizeX; ix++)
    {
      b(ix,iy,iz).rVector -= alpha * b(ix,iy,iz).AxVector;
      rk_rk += b(ix,iy,iz).rVector * b(ix,iy,iz).rVector;     
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,&rk_rk,1,MPI_DOUBLE,MPI_SUM,m_comm);
  
  #ifdef PRECOND
    FindZ();
    //rk_zk = rVector /cdot rVector
    //pVector = zVector
    rk_zk = 0.0;
    #pragma omp parallel for reduction(+:rk_zk)
    for(size_t i=0; i< Nblocks; i++)
    {
      BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
      for(int iz=0; iz<BlockType::sizeZ; iz++)
      for(int iy=0; iy<BlockType::sizeY; iy++)
      for(int ix=0; ix<BlockType::sizeX; ix++)
      {
        rk_zk += b(ix,iy,iz).rVector * b(ix,iy,iz).zVector;
        b(ix,iy,iz).pVector = b(ix,iy,iz).zVector;       
      }
    }
    MPI_Allreduce(MPI_IN_PLACE,&rk_zk,1,MPI_DOUBLE,MPI_SUM,m_comm);
  #else
    //pVector = rVector
    #pragma omp parallel for schedule(runtime)
    for (size_t i=0; i < Nblocks; i++)
    {
      BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
      for(int iz=0; iz<BlockType::sizeZ; iz++)
      for(int iy=0; iy<BlockType::sizeY; iy++)
      for(int ix=0; ix<BlockType::sizeX; ix++)
        b(ix,iy,iz).pVector = b(ix,iy,iz).rVector;
    }
  #endif
 
  bool flag = false;
  int count = 0;
  int err_init = std::sqrt(rk_rk)/Nsystem;

  sim.stopProfiler();//step one

  //for (size_t k = 1; k < 10000; k++)
  for (size_t k = 1; k < 4000; k++)
  {    
    count++;
    err = std::sqrt(rk_rk)/Nsystem;

    if (m_rank == 0 && (k % 100 == 0 || k < 20) ) std::cout << "k=" << k<<" err="<<err << " " << rMAX << std::endl;

    if (err < err_min && k >=2)
    {
      flag = true;
      err_min = err;
      #pragma omp parallel for schedule(runtime)
      for (size_t i=0; i < Nblocks; i++)
      {
        BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
        for(int iz=0; iz<BlockType::sizeZ; iz++)
        for(int iy=0; iy<BlockType::sizeY; iy++)
        for(int ix=0; ix<BlockType::sizeX; ix++)
          b(ix,iy,iz).sVector = b(ix,iy,iz).xVector;
      }
    }

    if ( (err < max_error || err/err_init < max_rel_error ) && k > 3) break;
    //if (  err/(err_min+1e-21) > 10.0 && k > 20) break; //error grows, stop iterations!
    if (  err/(err_min+1e-21) > 2.0 && k > 20) break; //error grows, stop iterations!

    sim.startProfiler("PoissonSolverAMR :: LHS");
    findLHS(0);// AxVector <-- A*pVector
    sim.stopProfiler();

    sim.startProfiler("PoissonSolverAMR :: loop");
    //pAp = pVector /cdot AxVector
    pAp = 0.0;
    #pragma omp parallel for reduction(+:pAp)
    for(size_t i=0; i< Nblocks; i++)
    {
      BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
      for(int iz=0; iz<BlockType::sizeZ; iz++)
      for(int iy=0; iy<BlockType::sizeY; iy++)
      for(int ix=0; ix<BlockType::sizeX; ix++)
        pAp += b(ix,iy,iz).pVector * b(ix,iy,iz).AxVector;
    }
    MPI_Allreduce(MPI_IN_PLACE,&pAp,1,MPI_DOUBLE,MPI_SUM,m_comm);

    if ( std::fabs(pAp) < 1e-21) {std::cout << "CG:pAp is small"<< std::endl; break;}

    #ifdef PRECOND
       alpha = rk_zk/pAp;
    #else
       alpha = rk_rk/pAp;
    #endif

    //x_{k+1} <-- x_{k} + alpha * p_{k}
    //r_{k+1} <-- r_{k} - alpha * Ax
    //rkp1_rkp1 = rVector /cdot rVector
    rkp1_rkp1 = 0.0;
    rMAX = -1e50;
    #pragma omp parallel for reduction(+:rkp1_rkp1)
    for(size_t i=0; i< Nblocks; i++)
    {
      double rLOC = -1e50;
      BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
      for(int iz=0; iz<BlockType::sizeZ; iz++)
      for(int iy=0; iy<BlockType::sizeY; iy++)
      for(int ix=0; ix<BlockType::sizeX; ix++)
      {
        b(ix,iy,iz).xVector += alpha * b(ix,iy,iz).pVector;
        b(ix,iy,iz).rVector -= alpha * b(ix,iy,iz).AxVector;       
        rkp1_rkp1 += b(ix,iy,iz).rVector * b(ix,iy,iz).rVector;
        rLOC = std::max(rLOC, std::fabs(b(ix,iy,iz).rVector));
      }
      #pragma omp critical
      {
          rMAX = std::max(rLOC, rMAX);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE,&rkp1_rkp1,1,MPI_DOUBLE,MPI_SUM,m_comm);
    MPI_Allreduce(MPI_IN_PLACE,&rMAX     ,1,MPI_DOUBLE,MPI_MAX,m_comm);

    #ifdef PRECOND
      FindZ();
      rkp1_zkp1 = 0.0;
      #pragma omp parallel for reduction(+:rkp1_zkp1)
      for(size_t i=0; i< Nblocks; i++)
      {
        BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
        for(int iz=0; iz<BlockType::sizeZ; iz++)
        for(int iy=0; iy<BlockType::sizeY; iy++)
        for(int ix=0; ix<BlockType::sizeX; ix++)
          rkp1_zkp1 += b(ix,iy,iz).rVector * b(ix,iy,iz).zVector;
      }
      MPI_Allreduce(MPI_IN_PLACE,&rkp1_zkp1,1,MPI_DOUBLE,MPI_SUM,m_comm);
      beta = rkp1_zkp1 / rk_zk;
      rk_zk = rkp1_zkp1;  
      
      //p_{k+1} <-- z_{k+1} + beta * p_{k}   
      #pragma omp parallel for schedule(runtime)
      for(size_t i=0; i< Nblocks; i++)
      {
        BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
        for(int iz=0; iz<BlockType::sizeZ; iz++)
        for(int iy=0; iy<BlockType::sizeY; iy++)
        for(int ix=0; ix<BlockType::sizeX; ix++)
          b(ix,iy,iz).pVector = beta * b(ix,iy,iz).pVector + b(ix,iy,iz).zVector;
      }
    #else
      beta = rkp1_rkp1 / rk_rk;
      //p_{k+1} <-- r_{k+1} + beta * p_{k}    
      #pragma omp parallel for schedule(runtime)
      for(size_t i=0; i< Nblocks; i++)
      {
        BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
        for(int iz=0; iz<BlockType::sizeZ; iz++)
        for(int iy=0; iy<BlockType::sizeY; iy++)
        for(int ix=0; ix<BlockType::sizeX; ix++)
          b(ix,iy,iz).pVector = beta * b(ix,iy,iz).pVector + b(ix,iy,iz).rVector;
      }
    #endif

    rk_rk = rkp1_rkp1;
    sim.stopProfiler();
  }

  sim.startProfiler("PoissonSolverAMR :: step two");

  if (flag)
  {
    #pragma omp parallel for schedule(runtime)
    for(size_t i=0; i< Nblocks; i++)
    {
      BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
      for(int iz=0; iz<BlockType::sizeZ; iz++)
      for(int iy=0; iy<BlockType::sizeY; iy++)
      for(int ix=0; ix<BlockType::sizeX; ix++)
        b(ix,iy,iz).xVector = b(ix,iy,iz).sVector;
    }
  }
  else
  {
    err_min = err;
  }

  if (m_rank==0) std::cout << "CG Poisson solver took "<<count << " iterations. Final residual norm = "<< err_min << "  MAX=" << rMAX << std::endl;


  #pragma omp parallel for schedule(runtime)
  for (size_t i=0; i < Nblocks; i++)
  {
    BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
    for(int iz=0; iz<BlockType::sizeZ; iz++)
    for(int iy=0; iy<BlockType::sizeY; iy++)
    for(int ix=0; ix<BlockType::sizeX; ix++)
    {
      b.tmp[iz][iy][ix] = b(ix,iy,iz).xVector;
    }
  }

  const Real avgP = computeAverage();
  // Subtract average pressure from all gridpoints
  #pragma omp parallel for schedule(runtime)
  for (size_t i=0; i < Nblocks; i++)
  {
    BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
    for(int iz=0; iz<BlockType::sizeZ; iz++)
    for(int iy=0; iy<BlockType::sizeY; iy++)
    for(int ix=0; ix<BlockType::sizeX; ix++)
    {
      b.tmp[iz][iy][ix] -= avgP;
    }
  }
  //std::cout <<"average = " << avgP << std::endl;
  //NOT DONE YET!!!!
  //// // Set this new mean-0 pressure as next guess
  //// // Save pressure of a corner of the grid so that it can be imposed next time
  //// pLast = data[fixed_idx];
  //// if(sim.verbose) printf("Avg Pressure:%f\n", avgP);


  #pragma omp parallel for schedule(runtime)
  for (size_t i=0; i < Nblocks; i++)
  {
    BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
    const size_t offset = _offset( vInfo[i] );
    for(int iz=0; iz<BlockType::sizeZ; iz++)
    for(int iy=0; iy<BlockType::sizeY; iy++)
    for(int ix=0; ix<BlockType::sizeX; ix++)
    {
      const size_t src_index = _dest(offset, iz, iy, ix);     
      b(ix,iy,iz).pVector  = storeGridElements[6*src_index  ];
      b(ix,iy,iz).rVector  = storeGridElements[6*src_index+1];
      b(ix,iy,iz).xVector  = storeGridElements[6*src_index+2];
      b(ix,iy,iz).sVector  = storeGridElements[6*src_index+3];
      b(ix,iy,iz).AxVector = storeGridElements[6*src_index+4];
      b(ix,iy,iz).zVector  = storeGridElements[6*src_index+5];
    }
  }

  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<vInfo.size(); ++i)
  {
    BlockType& b = *(BlockType*) vInfo[i].ptrBlock;
    for(int iz=0; iz<BlockType::sizeZ; iz++)
    for(int iy=0; iy<BlockType::sizeY; iy++)
    for(int ix=0; ix<BlockType::sizeX; ix++)
      b(ix,iy,iz).p = b.tmp[iz][iy][ix];
  }

  sim.stopProfiler();
}

#else
void PoissonSolverAMR::solve()
{
    sim.startProfiler("PoissonSolverAMR :: step one");
  
    static constexpr int BSX = BlockType::sizeX;
    static constexpr int BSY = BlockType::sizeY;
    static constexpr int BSZ = BlockType::sizeZ;

    std::vector<cubism::BlockInfo>& vInfo = grid.getBlocksInfo();

    const double eps = 1e-21;
    const size_t Nblocks = vInfo.size();
    const size_t N = BSX*BSY*BSZ*Nblocks;
    size_t Nsystem;
    MPI_Allreduce(&N,&Nsystem,1,MPI_LONG,MPI_SUM,m_comm);

    std::vector <Real> storeGridElements (8*N);
    #pragma omp parallel for schedule(runtime)
    for (size_t i=0; i < Nblocks; i++)
    {
        BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
        const size_t offset = _offset( vInfo[i] );
        for(int iz=0; iz<BlockType::sizeZ; iz++)
        for(int iy=0; iy<BlockType::sizeY; iy++)
        for(int ix=0; ix<BlockType::sizeX; ix++)
        {
            const size_t src_index = _dest(offset, iz, iy, ix);
            storeGridElements[8*src_index  ] = b(ix,iy,iz).chi;
            storeGridElements[8*src_index+1] = b(ix,iy,iz).u;
            storeGridElements[8*src_index+2] = b(ix,iy,iz).v;
            storeGridElements[8*src_index+3] = b(ix,iy,iz).w;
            storeGridElements[8*src_index+4] = b(ix,iy,iz).p;
            storeGridElements[8*src_index+5] = b(ix,iy,iz).tmpU;
            storeGridElements[8*src_index+6] = b(ix,iy,iz).tmpV;
            storeGridElements[8*src_index+7] = b(ix,iy,iz).tmpW;
        }
    }

    //1. rVector <-- b - AxVector
    //2. rhatVector = rVector
    //3. rho = a = omega = 1.0
    //4. vVector = pVector = 0
    #pragma omp parallel for schedule(runtime)
    for (size_t i=0; i < Nblocks; i++)
    {
        BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
        for(int iz=0; iz<BlockType::sizeZ; iz++)
        for(int iy=0; iy<BlockType::sizeY; iy++)
        for(int ix=0; ix<BlockType::sizeX; ix++)
            b(ix,iy,iz).zVector = b(ix,iy,iz).xVector;//this is done because Get_LHS works with zVector
    }
    findLHS(0); // AxVector <-- A*x_{0}, x_0 = pressure
    #pragma omp parallel for
    for(size_t i=0; i< Nblocks; i++)
    {
        BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
        for(int iz=0; iz<BlockType::sizeZ; iz++)
        for(int iy=0; iy<BlockType::sizeY; iy++)
        for(int ix=0; ix<BlockType::sizeX; ix++)
        {
            b(ix,iy,iz).rVector = b.tmp[iz][iy][ix] - b(ix,iy,iz).AxVector;
            b(ix,iy,iz).rhatVector = b(ix,iy,iz).rVector;
            b(ix,iy,iz).pVector = 0.0;
            b(ix,iy,iz).vVector = 0.0;
        }
    }
    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    double rho_m1;

    std::vector <Real> x_opt (N);
    bool useXopt = false;
    double min_norm = 1e50;
    double init_norm;
    double norm;
    const double max_error = 1e-9;
    const double max_rel_error = 1e-2;

    //5. for k = 1,2,...
    for (size_t k = 1; k < 4000; k++)
    {
        //1. rho_i = (rhat_0,r_{k-1})
        //2. beta = rho_{i}/rho_{i-1} * alpha/omega_{i-1}
        rho_m1 = rho;
        rho = 0.0;
        #pragma omp parallel for reduction(+:rho)
        for(size_t i=0; i< Nblocks; i++)
        {
            BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
            for(int iz=0; iz<BlockType::sizeZ; iz++)
            for(int iy=0; iy<BlockType::sizeY; iy++)
            for(int ix=0; ix<BlockType::sizeX; ix++)
                rho += b(ix,iy,iz).rVector * b(ix,iy,iz).rhatVector;
        }
        MPI_Allreduce(MPI_IN_PLACE,&rho,1,MPI_DOUBLE,MPI_SUM,m_comm);
        double beta = rho / (rho_m1+eps) * alpha / (omega+eps) ;

        //3. p_i = r_{i-1} + beta*(p_{i-1}-omega *v_{i-1})
        #pragma omp parallel for schedule(runtime)
        for (size_t i=0; i < Nblocks; i++)
        {
            BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
            for(int iz=0; iz<BlockType::sizeZ; iz++)
            for(int iy=0; iy<BlockType::sizeY; iy++)
            for(int ix=0; ix<BlockType::sizeX; ix++)
                b(ix,iy,iz).pVector = b(ix,iy,iz).rVector + beta* ( b(ix,iy,iz).pVector - omega * b(ix,iy,iz).vVector);
        }

        //4. z = K_2^{-1} p
        #pragma omp parallel for schedule(runtime)
        for (size_t i=0; i < Nblocks; i++)
        {
            BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
            for(int iz=0; iz<BlockType::sizeZ; iz++)
            for(int iy=0; iy<BlockType::sizeY; iy++)
            for(int ix=0; ix<BlockType::sizeX; ix++)
                b(ix,iy,iz).zVector = b(ix,iy,iz).pVector;
        }
        getZ();

        //5. v = A z
        //6. alpha = rho_i / (rhat_0,v_i)
        alpha = 0.0;
        findLHS(0); //v stored in AxVector
        #pragma omp parallel for reduction(+:alpha)
        for (size_t i=0; i < Nblocks; i++)
        {
            BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
            for(int iz=0; iz<BlockType::sizeZ; iz++)
            for(int iy=0; iy<BlockType::sizeY; iy++)
            for(int ix=0; ix<BlockType::sizeX; ix++)
            {
                b(ix,iy,iz).vVector = b(ix,iy,iz).AxVector;
                alpha += b(ix,iy,iz).rhatVector * b(ix,iy,iz).vVector;
            }
        }  
        MPI_Allreduce(MPI_IN_PLACE,&alpha,1,MPI_DOUBLE,MPI_SUM,m_comm);
        alpha = rho / (alpha + eps);

        //7. x += a z
        //8. 
        //9. s = r_{i-1}-alpha * v_i
        //10. z = K_2^{-1} s
        #pragma omp parallel for schedule(runtime)
        for (size_t i=0; i < Nblocks; i++)
        {
            BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
            for(int iz=0; iz<BlockType::sizeZ; iz++)
            for(int iy=0; iy<BlockType::sizeY; iy++)
            for(int ix=0; ix<BlockType::sizeX; ix++)
            {
                b(ix,iy,iz).xVector += alpha * b(ix,iy,iz).zVector;
                b(ix,iy,iz).sVector = b(ix,iy,iz).rVector - alpha * b(ix,iy,iz).vVector;
                b(ix,iy,iz).zVector = b(ix,iy,iz).sVector;
            }
        }
        getZ();

        //11. t = Az
        findLHS(0); // t stored in AxVector

        //12. omega = ...
        double aux1 = 0;
        double aux2 = 0;
        #pragma omp parallel for reduction (+:aux1,aux2)
        for (size_t i=0; i < Nblocks; i++)
        {
            BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
            for(int iz=0; iz<BlockType::sizeZ; iz++)
            for(int iy=0; iy<BlockType::sizeY; iy++)
            for(int ix=0; ix<BlockType::sizeX; ix++)
            {
                aux1 += b(ix,iy,iz).AxVector * b(ix,iy,iz).sVector;
                aux2 += b(ix,iy,iz).AxVector * b(ix,iy,iz).AxVector;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE,&aux1,1,MPI_DOUBLE,MPI_SUM,m_comm);
        MPI_Allreduce(MPI_IN_PLACE,&aux2,1,MPI_DOUBLE,MPI_SUM,m_comm);
        omega = aux1 / (aux2+eps); 

        //13. x += omega * z
        //14.
        //15. r = s - omega * t
        norm = 0.0; 
        #pragma omp parallel for reduction(+:norm)
        for (size_t i=0; i < Nblocks; i++)
        {
            BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
            for(int iz=0; iz<BlockType::sizeZ; iz++)
            for(int iy=0; iy<BlockType::sizeY; iy++)
            for(int ix=0; ix<BlockType::sizeX; ix++)
            {
                b(ix,iy,iz).xVector += omega * b(ix,iy,iz).zVector;
                b(ix,iy,iz).rVector  = b(ix,iy,iz).sVector- omega * b(ix,iy,iz).AxVector;
                norm+= b(ix,iy,iz).rVector*b(ix,iy,iz).rVector; 
            }
        }
        MPI_Allreduce(MPI_IN_PLACE,&norm,1,MPI_DOUBLE,MPI_SUM,m_comm);
        norm = std::sqrt(norm) / Nsystem;

        if (norm < min_norm)
        {
            min_norm = norm;
            #pragma omp parallel for schedule(runtime)
            for (size_t i=0; i < Nblocks; i++)
            {
                BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
                const size_t offset = _offset( vInfo[i] );
                for(int iz=0; iz<BlockType::sizeZ; iz++)
                for(int iy=0; iy<BlockType::sizeY; iy++)
                for(int ix=0; ix<BlockType::sizeX; ix++)
                {
                    const size_t src_index = _dest(offset, iz, iy, ix);
                    x_opt[src_index] = b(ix,iy,iz).xVector;
                }
            }
        }
    
        if (k==1) init_norm = norm;



        if (norm / (init_norm+eps) > 2.0 && k > 10)
        {
            useXopt = true;
            if (m_rank==0)
                std::cout <<  "Poisson solver converged after " <<  k << " iterations. Error norm = " << norm << std::endl;
            break;
        }

        if ( (norm < max_error || norm/init_norm < max_rel_error ) && k > 1 )
        {
            if (m_rank==0)
                std::cout <<  "Poisson solver converged after " <<  k << " iterations. Error norm = " << norm << std::endl;
            break;
        }

    }//k-loop


    if (useXopt)
    {
        #pragma omp parallel for schedule(runtime)
        for (size_t i=0; i < Nblocks; i++)
        {
            BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
            const size_t offset = _offset( vInfo[i] );
            for(int iz=0; iz<BlockType::sizeZ; iz++)
            for(int iy=0; iy<BlockType::sizeY; iy++)
            for(int ix=0; ix<BlockType::sizeX; ix++)
            {
                const size_t src_index = _dest(offset, iz, iy, ix);
                b(ix,iy,iz).xVector = x_opt[src_index];
            }
        }
    }

    #pragma omp parallel for schedule(runtime)
    for (size_t i=0; i < Nblocks; i++)
    {
        BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
        for(int iz=0; iz<BlockType::sizeZ; iz++)
        for(int iy=0; iy<BlockType::sizeY; iy++)
        for(int ix=0; ix<BlockType::sizeX; ix++)
            b.tmp[iz][iy][ix] = b(ix,iy,iz).xVector;
    }
  
    // Subtract average pressure from all gridpoints and copy back stuff
    const Real avgP = computeAverage();
    if (m_rank == 0) 
      std::cout << "avgP = " << avgP << std::endl; 
    #pragma omp parallel for schedule(runtime)
    for (size_t i=0; i < Nblocks; i++)
    {
        BlockType & __restrict__ b  = *(BlockType*) vInfo[i].ptrBlock;
        const size_t offset = _offset( vInfo[i] );
        for(int iz=0; iz<BlockType::sizeZ; iz++)
        for(int iy=0; iy<BlockType::sizeY; iy++)
        for(int ix=0; ix<BlockType::sizeX; ix++)
        {
            b.tmp[iz][iy][ix] -= avgP;
            const size_t src_index = _dest(offset, iz, iy, ix);     
            b(ix,iy,iz).chi  = storeGridElements[8*src_index  ];
            b(ix,iy,iz).u    = storeGridElements[8*src_index+1];
            b(ix,iy,iz).v    = storeGridElements[8*src_index+2];
            b(ix,iy,iz).w    = storeGridElements[8*src_index+3];
            b(ix,iy,iz).p    = storeGridElements[8*src_index+4];
            b(ix,iy,iz).tmpU = storeGridElements[8*src_index+5];
            b(ix,iy,iz).tmpV = storeGridElements[8*src_index+6];
            b(ix,iy,iz).tmpW = storeGridElements[8*src_index+7];
            b(ix,iy,iz).p = b.tmp[iz][iy][ix];
        }
    } 

    sim.stopProfiler();
}

#endif

}//namespace cubismup3d