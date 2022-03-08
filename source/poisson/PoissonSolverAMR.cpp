//
//  CubismUP_3D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

#include "PoissonSolverAMR.h"

namespace cubismup3d {

PoissonSolverAMR::PoissonSolverAMR(SimulationData& s): sim(s),findLHS(s)
{
}

void PoissonSolverAMR::getZ()
{
  const std::vector<cubism::BlockInfo>& vInfo = gridPoisson.getBlocksInfo();
  const size_t Nblocks = vInfo.size();

  #pragma omp parallel
  {
    const int nx = BlockType::sizeX;
    const int ny = BlockType::sizeY;
    const int nz = BlockType::sizeZ;
    const int nx2 = nx + 2;
    const int ny2 = ny + 2;
    const int nz2 = nz + 2;
    const int N = nx*ny*nz;
    const int N2 = nx2*ny2*nz2;
    std::vector<Real> p (N2,0.0);
    std::vector<Real> r (N ,0.0);
    std::vector<Real> Ax(N ,0.0);

    #pragma omp for
    for (size_t i=0; i < Nblocks; i++)
    {
        BlockTypePoisson & __restrict__ b  = *(BlockTypePoisson*) vInfo[i].ptrBlock;
        const Real invh = 1.0/vInfo[i].h;
        Real norm0 = 0;
        Real rr = 0;
        Real a2 = 0;
        Real beta = 0;
        for(int iz=0; iz<BlockType::sizeZ; iz++)
        for(int iy=0; iy<BlockType::sizeY; iy++)
        for(int ix=0; ix<BlockType::sizeX; ix++)
        {
            r[ix+iy*nx+iz*nx*ny] = invh*b(ix,iy,iz).s;
            norm0 += r[ix+iy*nx+iz*nx*ny]*r[ix+iy*nx+iz*nx*ny];
            p[(ix+1)+(iy+1)*(nx+2)+(iz+1)*(nx+2)*(ny+2)] = r[ix+iy*nx+iz*nx*ny];
            rr += r[ix+iy*nx+iz*nx*ny]*r[ix+iy*nx+iz*nx*ny];
	    b(ix,iy,iz).s = 0.0;
        }
        norm0 = sqrt(norm0)/N;
        Real norm = 0;
        if (norm0 > 1e-16)
        for (int k = 0 ; k < 100 ; k ++)
        {
            a2 = 0;
            for(int iz=0; iz<nz; iz++)
            for(int iy=0; iy<ny; iy++)
            for(int ix=0; ix<nx; ix++)
            {
                const int index1 = ix+iy*nx+iz*nx*ny;
                const int index2 = (ix+1)+(iy+1)*(nx+2)+(iz+1)*(nx+2)*(ny+2);
                Ax[index1] = p[index2 + 1] + p[index2 - 1] + p[index2 + nx2] + p[index2 - nx2] + p[index2 + nx2*ny2] + p[index2 - nx2*ny2] - 6.0*p[index2];
                a2 += p[index2]*Ax[index1];
            }
            const Real a = rr/(a2+1e-55);
            Real norm_new = 0;
            beta = 0;
            for(int iz=0; iz<nz; iz++)
            for(int iy=0; iy<ny; iy++)
            for(int ix=0; ix<nx; ix++)
            {
                const int index1 = ix+iy*nx+iz*nx*ny;
                const int index2 = (ix+1)+(iy+1)*(nx+2)+(iz+1)*(nx+2)*(ny+2);
	        b(ix,iy,iz).s += a*p [index2];
                r[index1] -= a*Ax[index1];
                norm_new += r[index1]*r[index1];
            }
            beta = norm_new;
            norm_new = sqrt(norm_new)/N;
            //if (k>max(max(nx,ny),nz) && std::fabs(norm/norm_new - 1.0) < 1e-7) break;
            norm = norm_new;
            if (norm/norm0< 1e-7) break;
            Real temp = rr;
            rr = beta;
            beta /= (temp+1e-55);
            for(int iz=0; iz<nz; iz++)
            for(int iy=0; iy<ny; iy++)
            for(int ix=0; ix<nx; ix++)
            {
                const int index1 = ix+iy*nx+iz*nx*ny;
                const int index2 = (ix+1)+(iy+1)*(nx+2)+(iz+1)*(nx+2)*(ny+2);
                p[index2] =r[index1] + beta*p[index2];
            }
        }
    }    
  }
}

void PoissonSolverAMR::solve()
{
    std::vector<cubism::BlockInfo>& vInfo = grid.getBlocksInfo();
    const Real eps = 1e-21;
    const size_t Nblocks = vInfo.size();
    const size_t N = BlockType::sizeX*BlockType::sizeY*BlockType::sizeZ*Nblocks;
    std::vector<Real> x   (N,0.0);
    std::vector<Real> r   (N,0.0);
    std::vector<Real> p   (N,0.0);
    std::vector<Real> v   (N,0.0);
    std::vector<Real> s   (N,0.0);
    std::vector<Real> rhat(N,0.0);
    std::vector<cubism::BlockInfo>& vInfoPoisson = gridPoisson.getBlocksInfo();

    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
        BlockTypePoisson & __restrict__ bPoisson  = *(BlockTypePoisson*) vInfoPoisson[i].ptrBlock;
	if (sim.bMeanConstraint == 1 || sim.bMeanConstraint > 2)
        if (vInfoPoisson[i].index[0] == 0 && 
            vInfoPoisson[i].index[1] == 0 && 
            vInfoPoisson[i].index[2] == 0)
            bPoisson(0,0,0).lhs = 0.0;

        for(int iz=0; iz<BlockType::sizeZ; iz++)
        for(int iy=0; iy<BlockType::sizeY; iy++)
        for(int ix=0; ix<BlockType::sizeX; ix++)
        {
            const size_t src_index = _dest(vInfoPoisson[i], iz, iy, ix);
            x[src_index] = bPoisson(ix,iy,iz).s;
            r[src_index] = bPoisson(ix,iy,iz).lhs;//this is the rhs now
        }
    }


    //1. rVector <-- b - AxVector
    //2. rhatVector = rVector
    //3. rho = a = omega = 1.0
    //4. vVector = pVector = 0

    Real norm = 0.0; 
    findLHS(0); // AxVector <-- A*x_{0}, x_0 = pressure
    #pragma omp parallel for reduction (+:norm)
    for(size_t i=0; i< Nblocks; i++)
    {
        BlockTypePoisson & __restrict__ bPoisson  = *(BlockTypePoisson*) vInfoPoisson[i].ptrBlock;
        for(int iz=0; iz<BlockType::sizeZ; iz++)
        for(int iy=0; iy<BlockType::sizeY; iy++)
        for(int ix=0; ix<BlockType::sizeX; ix++)
        {
            const size_t src_index = _dest(vInfoPoisson[i], iz, iy, ix);
            r[src_index] -= bPoisson(ix,iy,iz).lhs;
            rhat[src_index] = r[src_index];
            norm+= r[src_index]*r[src_index];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE,&norm,1,MPI_Real,MPI_SUM,grid.getCartComm());
    norm = std::sqrt(norm);

    Real rho = 1.0;
    Real alpha = 1.0;
    Real omega = 1.0;
    Real rho_m1;

    std::vector <Real> x_opt (N);
    bool useXopt = false;
    Real min_norm = 1e50;
    Real init_norm=norm;
    const Real max_error     = sim.PoissonErrorTol;
    const Real max_rel_error = sim.PoissonErrorTolRel;
    bool serious_breakdown = false;
    int iter_opt = 0;

    //5. for k = 1,2,...
    for (size_t k = 1; k < 1000; k++)
    {
        //1. rho_i = (rhat_0,r_{k-1})
        //2. beta = rho_{i}/rho_{i-1} * alpha/omega_{i-1}
        rho_m1 = rho;
        rho = 0.0;
        Real norm_1 = 0.0;
        Real norm_2 = 0.0;
        #pragma omp parallel for reduction(+:rho,norm_1,norm_2)
        for(size_t i=0; i< N; i++)
        {
            rho += r[i]*rhat[i];
            norm_1 += r[i]*r[i];
            norm_2 += rhat[i]*rhat[i];
        }
        Real aux_norm [3] = {rho,norm_1,norm_2};
        MPI_Allreduce(MPI_IN_PLACE,&aux_norm,3,MPI_Real,MPI_SUM,grid.getCartComm());
        rho = aux_norm[0];
        norm_1 = aux_norm[1];
        norm_2 = aux_norm[2];
        Real beta = rho / (rho_m1+eps) * alpha / (omega+eps) ;

        norm_1 = sqrt(norm_1);
        norm_2 = sqrt(norm_2);
        Real cosTheta = rho/norm_1/norm_2; 
        serious_breakdown = std::fabs(cosTheta) < 1e-8;
        if (serious_breakdown)
        {
            beta = 0.0;
            rho = 0.0;
            #pragma omp parallel for reduction(+:rho)
            for(size_t i=0; i< N; i++)
            {
                rhat[i] = r[i];
                rho += r[i]*rhat[i];
                p[i] = 0.0;
                v[i] = 0.0;
            }
            if (sim.rank==0) 
                std::cout << "  [Poisson solver]: restart at iteration:" << k << 
                             "  norm:"<< norm <<" init_norm:" << init_norm << std::endl;
            MPI_Allreduce(MPI_IN_PLACE,&rho,1,MPI_Real,MPI_SUM,grid.getCartComm());
            alpha = 1.;
            omega = 1.;
            rho_m1 = 1.;
            beta = rho / (rho_m1+eps) * alpha / (omega+eps) ;
        }

        //3. p_i = r_{i-1} + beta*(p_{i-1}-omega *v_{i-1})
        //4. z = K_2^{-1} p
        #pragma omp parallel for
        for (size_t i=0; i < Nblocks; i++)
        {
            BlockTypePoisson & __restrict__ bPoisson  = *(BlockTypePoisson*) vInfoPoisson[i].ptrBlock;
            for(int iz=0; iz<BlockType::sizeZ; iz++)
            for(int iy=0; iy<BlockType::sizeY; iy++)
            for(int ix=0; ix<BlockType::sizeX; ix++)
            {
                const size_t src_index = _dest(vInfoPoisson[i], iz, iy, ix);
                p[src_index] = r[src_index] + beta*(p[src_index]-omega*v[src_index]);
                bPoisson(ix,iy,iz).s = p[src_index];
            }
        }
        getZ();

        //5. v = A z
        //6. alpha = rho_i / (rhat_0,v_i)
        alpha = 0.0;
        findLHS(0); //v stored in AxVector
        #pragma omp parallel for reduction(+:alpha)
        for (size_t i=0; i < Nblocks; i++)
        {
            BlockTypePoisson & __restrict__ bPoisson  = *(BlockTypePoisson*) vInfoPoisson[i].ptrBlock;
            for(int iz=0; iz<BlockType::sizeZ; iz++)
            for(int iy=0; iy<BlockType::sizeY; iy++)
            for(int ix=0; ix<BlockType::sizeX; ix++)
            {
                const size_t src_index = _dest(vInfoPoisson[i], iz, iy, ix);
                v[src_index] = bPoisson(ix,iy,iz).lhs;
                alpha += rhat[src_index] * v[src_index];
            }
        }  
        MPI_Allreduce(MPI_IN_PLACE,&alpha,1,MPI_Real,MPI_SUM,grid.getCartComm());
        alpha = rho / (alpha + eps);
        //7. x += a z
        //8. 
        //9. s = r_{i-1}-alpha * v_i
        //10. z = K_2^{-1} s
        #pragma omp parallel for
        for (size_t i=0; i < Nblocks; i++)
        {
            BlockTypePoisson & __restrict__ bPoisson  = *(BlockTypePoisson*) vInfoPoisson[i].ptrBlock;
            for(int iz=0; iz<BlockType::sizeZ; iz++)
            for(int iy=0; iy<BlockType::sizeY; iy++)
            for(int ix=0; ix<BlockType::sizeX; ix++)
            {
                const size_t src_index = _dest(vInfoPoisson[i], iz, iy, ix);
                x[src_index] += alpha * bPoisson(ix,iy,iz).s;
                s[src_index] = r[src_index] - alpha * v[src_index];
                bPoisson(ix,iy,iz).s = s[src_index];
            }
        }
        getZ();

        //11. t = Az
        findLHS(0); // t stored in AxVector

        //12. omega = ...
        Real aux1 = 0;
        Real aux2 = 0;
        #pragma omp parallel for reduction (+:aux1,aux2)
        for (size_t i=0; i < Nblocks; i++)
        {
            BlockTypePoisson & __restrict__ bPoisson  = *(BlockTypePoisson*) vInfoPoisson[i].ptrBlock;
            for(int iz=0; iz<BlockType::sizeZ; iz++)
            for(int iy=0; iy<BlockType::sizeY; iy++)
            for(int ix=0; ix<BlockType::sizeX; ix++)
            {
                const size_t src_index = _dest(vInfoPoisson[i], iz, iy, ix);
                aux1 += bPoisson(ix,iy,iz).lhs * s[src_index];
                aux2 += bPoisson(ix,iy,iz).lhs * bPoisson(ix,iy,iz).lhs;
            }
        }
        Real aux_12[2] = {aux1,aux2};
        MPI_Allreduce(MPI_IN_PLACE,&aux_12,2,MPI_Real,MPI_SUM,grid.getCartComm());
        aux1 = aux_12[0];
        aux2 = aux_12[1];
        omega = aux1 / (aux2+eps); 

        //13. x += omega * z
        //14.
        //15. r = s - omega * t
        norm = 0.0; 
        #pragma omp parallel for reduction(+:norm)
        for (size_t i=0; i < Nblocks; i++)
        {
            BlockTypePoisson & __restrict__ bPoisson  = *(BlockTypePoisson*) vInfoPoisson[i].ptrBlock;
            for(int iz=0; iz<BlockType::sizeZ; iz++)
            for(int iy=0; iy<BlockType::sizeY; iy++)
            for(int ix=0; ix<BlockType::sizeX; ix++)
            {
                const size_t src_index = _dest(vInfoPoisson[i], iz, iy, ix);
                x[src_index] += omega * bPoisson(ix,iy,iz).s;
                r[src_index] = s[src_index]-omega*bPoisson(ix,iy,iz).lhs;
                norm+= r[src_index]*r[src_index];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE,&norm,1,MPI_Real,MPI_SUM,grid.getCartComm());
        norm = std::sqrt(norm);

        if (norm < min_norm)
        {
            useXopt = true;
            iter_opt = k;
            min_norm = norm;
            #pragma omp parallel for
            for (size_t i=0; i < N; i++)
            {
                x_opt[i] = x[i];
            }
        }
  
        if (k==1) init_norm = norm;

        if (norm / (init_norm+eps) > 100000.0 && k > 10)
        {
            useXopt = true;
            if (sim.rank==0)
                std::cout <<  "XOPT Poisson solver converged after " <<  k << " iterations. Error norm = " << norm << "  iter_opt="<< iter_opt << std::endl;
            break;
        }

        if ( (norm < max_error || norm/init_norm < max_rel_error ) )
        {
            if (sim.rank==0)
                std::cout <<  "  [Poisson solver]: Converged after " <<  k << " iterations. Error norm = " << norm << std::endl;
            break;
        }

    }//k-loop

    Real * xsol = useXopt ? x_opt.data():x.data();
    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
        const int m = vInfoPoisson[i].level;
        const long long n = vInfoPoisson[i].Z;
        const BlockInfo & info = grid.getBlockInfoAll(m,n);
        BlockType & __restrict__ b  = *(BlockType*) info.ptrBlock;
        for(int iz=0; iz<BlockType::sizeZ; iz++)
        for(int iy=0; iy<BlockType::sizeY; iy++)
        for(int ix=0; ix<BlockType::sizeX; ix++)
        {
            const size_t src_index = _dest(vInfoPoisson[i], iz, iy, ix);
            b(ix,iy,iz).p = xsol[src_index];
        }
    }
  
}
}//namespace cubismup3d
