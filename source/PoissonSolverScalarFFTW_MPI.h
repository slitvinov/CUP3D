#pragma once

//#include <mpi.h>
//#include <vector>
//#include <cassert>
//#include <cmath>
//#include <iostream>


#include "Definitions.h"

#include <fftw3.h>
#include <fftw3-mpi.h>

#ifndef _SP_COMP_
typedef fftw_complex mycomplex;
typedef fftw_plan myplan;
#else
typedef fftwf_complex mycomplex;
typedef fftwf_plan myplan;
#endif

using namespace std;

#include <BlockInfo.h>

class FFTWBase_MPI
{
	static int registered_objects;
	static bool initialized; //a la singleton
	
	static void _setup(const int desired_threads)
	{
		if (!initialized) {
			initialized = true;
			const int supported_threads = MPI::Query_thread();
			if (supported_threads>=MPI_THREAD_FUNNELED) {
#ifndef _SP_COMP_
				const int retval = fftw_init_threads();
#else
				const int retval = fftwf_init_threads();
#endif
				if(retval==0) {
					cout << "FFTWBase::setup(): Oops the call to fftw_init_threads() returned zero. Aborting\n";
					abort();
				}
				else {
#ifndef _SP_COMP_
					fftw_plan_with_nthreads(desired_threads);
#else
					fftwf_plan_with_nthreads(desired_threads);
#endif
				}
			}
			
#ifndef _SP_COMP_
			fftw_mpi_init();
#else
			fftwf_mpi_init();
#endif
		}
		
		registered_objects++;
	}
	
public:
	
	FFTWBase_MPI(const int desired_threads) { _setup(desired_threads); }
	
	static void dispose()
	{
		registered_objects--;
		
		if (registered_objects == 0) {
#ifndef _SP_COMP_
			fftw_mpi_cleanup();
#else
			fftwf_mpi_cleanup();
#endif
		}
	}
};

template<typename TGrid, typename TStreamer>
class PoissonSolverScalarFFTW_MPI : FFTWBase_MPI
{
	typedef typename TGrid::BlockType BlockType;
	
	bool initialized;
	
	mycomplex local_rhs, local_work;
	myplan fwd, bwd;
	ptrdiff_t alloc_local, local_n0, local_0_start, local_n1, local_1_start;
	
protected:
	
	void _setup(Real *& rhs , const size_t nx, const size_t ny, const size_t nz)
	{
		if (!initialized) {
			initialized = true;
#ifndef _SP_COMP_
			alloc_local = fftw_mpi_local_size_3d_transposed(nx, ny, nz/2+1, MPI::COMM_WORLD, &local_n0, &local_0_start, &local_n1, &local_1_start);
			rhs = fftw_alloc_real(2*alloc_local);
			fwd = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, rhs, (mycomplex *)rhs, MPI::COMM_WORLD, FFTW_MPI_TRANSPOSED_OUT | FFTW_MEASURE);
			bwd = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, (mycomplex *)rhs, rhs, MPI::COMM_WORLD, FFTW_MPI_TRANSPOSED_IN | FFTW_MEASURE);
#else
			alloc_local = fftwf_mpi_local_size_3d_transposed(nx, ny, nz/2+1, MPI::COMM_WORLD, &local_n0, &local_0_start, &local_n1, &local_1_start);
			rhs = fftwf_alloc_real(2*alloc_local);
			fwd = fftwf_mpi_plan_dft_r2c_3d(nx, ny, nz, rhs, (mycomplex *)rhs, MPI::COMM_WORLD, FFTW_MPI_TRANSPOSED_OUT | FFTW_MEASURE);
			bwd = fftwf_mpi_plan_dft_c2r_3d(nx, ny, nz, (mycomplex *)rhs, rhs, MPI::COMM_WORLD, FFTW_MPI_TRANSPOSED_IN | FFTW_MEASURE);
#endif
		}
	}
	
	void _cub2fftw(TGrid& grid, Real * out, const size_t nx, const size_t ny, const size_t nz, const size_t nz_hat) const
	{
		const int bs[3] = {BlockType::sizeX, BlockType::sizeY, BlockType::sizeZ};
		vector<BlockInfo> infos = grid.getBlocksInfo();
		const size_t N = infos.size();
		const size_t mybpd[3] = {
				grid.getResidentBlocksPerDimension(0),
				grid.getResidentBlocksPerDimension(1),
				grid.getResidentBlocksPerDimension(2)
		};
		
#pragma omp parallel for
		for(int i=0; i<N; ++i) {
			const BlockInfo info = infos[i];
			BlockType& b = *(BlockType*)info.ptrBlock;
			
			const size_t offset = bs[2]*info.indexLocal[2]+nz_hat*2*(bs[1]*info.indexLocal[1]+mybpd[1]*bs[1]*bs[0]*info.indexLocal[0]);
			
			for(int iz=0; iz<BlockType::sizeZ; iz++)
			for(int iy=0; iy<BlockType::sizeY; iy++)
			for(int ix=0; ix<BlockType::sizeX; ix++) {
				const size_t dest_index = offset + iz + 2*nz_hat*( iy + BlockType::sizeY*mybpd[1]*ix );
				assert(dest_index>=0 && dest_index<nx*ny*nz_hat*2);
				TStreamer::operate(b.data[iz][iy][ix], &out[dest_index]);
			}
		}
	}
	
	void _solve(mycomplex * in_out, const size_t nx, const size_t ny, const size_t nz, const size_t nz_hat, const Real norm_factor, const Real h)
	{
#if 0
		const Real h2 = h*h;
		const Real factor = h2*norm_factor;
		
#pragma omp parallel for
		for(int j=0; j<local_n1; ++j)
		for(int i=0; i<nx; ++i)
		for(int k = 0; k<nz_hat; ++k) {
			const int linidx = (j*nx+i)*nz_hat + k;
			assert(linidx >=0 && linidx<nx*local_n1*nz_hat);
			assert(linidx < alloc_local);

			const Real denom = 32.*(cos(2.*M_PI*i/nx) + cos(2.*M_PI*(local_1_start+j)/ny) + cos(2.*M_PI*k/nz)) -
								2.*(cos(4.*M_PI*i/nx) + cos(4.*M_PI*(local_1_start+j)/ny) + cos(4.*M_PI*k/nz)) - 90.;

			const Real inv_denom = (denom==0)? 0.:1./denom;
			const Real fatfactor = 12. * inv_denom * factor;

			in_out[linidx][0] *= fatfactor;
			in_out[linidx][1] *= fatfactor;
		}
#else
        const Real waveFactX = 2.0*M_PI/(nx*h);
        const Real waveFactY = 2.0*M_PI/(ny*h);
        const Real waveFactZ = 2.0*M_PI/(nz*h);

#pragma omp parallel for
		for(int j = 0; j<local_n1; ++j)
		for(int i=0; i<nx; ++i)
		for(int k = 0; k<nz_hat; ++k) {
			const int linidx = (j*nx+i)*nz_hat + k;
			//assert(linidx >=0 && linidx<nx*ny*nz_hat); // linking error with openmp

			const int kx = (i <= nx/2) ? i : -(nx-i);
			const int ky = (local_1_start+j <= ny/2) ? local_1_start+j : -(ny-(local_1_start+j));
			const int kz = (k <= nz/2) ? k : -(nz-k);
			const Real rkx = kx*waveFactX;
			const Real rky = ky*waveFactY;
			const Real rkz = kz*waveFactZ;

			const Real kinv = (kx==0 && ky==0 && kz==0) ? 0.0 : -1.0/(rkx*rkx+rky*rky+rkz*rkz);
			in_out[linidx][0] *= kinv*norm_factor;
			in_out[linidx][1] *= kinv*norm_factor;
		}
#endif
		
		//this is sparta!
		if (local_1_start == 0) in_out[0][0] = in_out[0][1] = 0;
	}
	
	void _fftw2cub(Real * out, TGrid& grid, const size_t nx, const size_t ny, const size_t nz, const size_t nz_hat) const
	{
		const int bs[3] = {BlockType::sizeX, BlockType::sizeY, BlockType::sizeZ};
		vector<BlockInfo> infos = grid.getBlocksInfo();
		const size_t N = infos.size();
		const size_t mybpd[3] = {
				grid.getResidentBlocksPerDimension(0),
				grid.getResidentBlocksPerDimension(1),
				grid.getResidentBlocksPerDimension(2)
		};
		
#pragma omp parallel for
		for(int i=0; i<N; ++i) {
			const BlockInfo info = infos[i];
			BlockType& b = *(BlockType*)infos[i].ptrBlock;
			
			const size_t offset = bs[2]*info.indexLocal[2]+nz_hat*2*(bs[1]*info.indexLocal[1]+mybpd[1]*bs[1]*bs[0]*info.indexLocal[0]);

			for(int iz=0; iz<BlockType::sizeZ; iz++)
			for(int iy=0; iy<BlockType::sizeY; iy++)
			for(int ix=0; ix<BlockType::sizeX; ix++) {
				const size_t src_index = offset + iz + 2*nz_hat*( iy + BlockType::sizeY*mybpd[1]*ix );
				assert(src_index>=0 && src_index<nx*ny*nz_hat*2);
				TStreamer::operate(&out[src_index], b.data[iz][iy][ix]);
			}
		}
	}
	
public:
	Real * data;
	
	PoissonSolverScalarFFTW_MPI(const int desired_threads, TGrid& grid): FFTWBase_MPI(desired_threads), initialized(false)
	{
		if (TStreamer::channels != 1) {
			cout << "PoissonSolverScalar_MPI(): Error: TStreamer::channels is " << TStreamer::channels << " (should be 1).\n";
			abort();
		}
		
		const int bs[3] = {BlockType::sizeX, BlockType::sizeY, BlockType::sizeZ};
		const size_t gsize[3] = {
				grid.getBlocksPerDimension(0)*bs[0],
				grid.getBlocksPerDimension(1)*bs[1],
				grid.getBlocksPerDimension(2)*bs[2]
		};
		_setup(data, gsize[0], gsize[1], gsize[2]);
	}
	
	void solve(TGrid& grid)
	{
		const int bs[3] = {BlockType::sizeX, BlockType::sizeY, BlockType::sizeZ};
		const size_t gsize[3] = {
				grid.getBlocksPerDimension(0)*bs[0],
				grid.getBlocksPerDimension(1)*bs[1],
				grid.getBlocksPerDimension(2)*bs[2]
		};
		
		_cub2fftw(grid, data, local_n0, gsize[1], gsize[2], gsize[2]/2+1);
		
#ifndef _SP_COMP_
		fftw_execute(fwd);
#else
		fftwf_execute(fwd);
#endif
		
		const Real norm_factor = 1./(gsize[0]*gsize[1]*gsize[2]);
		const Real h = grid.getBlocksInfo().front().h_gridpoint;
		_solve((mycomplex *)data, gsize[0], gsize[1], gsize[2], gsize[2]/2+1, norm_factor, h);
		
#ifndef _SP_COMP_
		fftw_execute(bwd);
#else
		fftwf_execute(bwd);
#endif
		
		_fftw2cub(data, grid, gsize[0], gsize[1], gsize[2], gsize[2]/2+1);
	}
	
	void dispose()
	{
		if (initialized) {
			initialized = false;
			
#ifndef _SP_COMP_
			fftw_destroy_plan(fwd);
			fftw_destroy_plan(bwd);
			fftw_free(data);
#else
			fftwf_destroy_plan(fwd);
			fftwf_destroy_plan(bwd);
			fftwf_free(data);
#endif
			FFTWBase_MPI::dispose();
		}
	}
};

// 2D, FFT+DCT in 1 step
template<typename TGrid, typename TStreamer>
class PoissonSolverScalarFFTW_MPI_DCT : FFTWBase_MPI
{
	Profiler profiler;
	
protected:
	typedef typename TGrid::BlockType BlockType;
	
	bool initialized;
	
	myplan fwd, bwd;
	ptrdiff_t alloc_local, local_n0, local_0_start, local_n1, local_1_start;
	
protected:
	
	virtual void _setup(Real *& rhs, const size_t nx, const size_t ny, const size_t nz)
	{
		if (!initialized) {
			initialized = true;
			
			//  Forward X:  FFT
			//  Backward X: FFT
			//  Forward Y:  DST-III
			//  Backward Y: DST-II
#ifndef _SP_COMP_
			alloc_local = fftw_mpi_local_size_3d_transposed(nx, ny, nz, MPI::COMM_WORLD, &local_n0, &local_0_start, &local_n1, &local_1_start);
			rhs = fftw_alloc_real(alloc_local);
			fwd = fftw_mpi_plan_r2r_3d(nx, ny, nz, rhs, rhs, MPI::COMM_WORLD, FFTW_R2HC, FFTW_REDFT11, FFTW_R2HC, FFTW_MEASURE);
			bwd = fftw_mpi_plan_r2r_3d(nx, ny, nz, rhs, rhs, MPI::COMM_WORLD, FFTW_HC2R, FFTW_REDFT11, FFTW_HC2R, FFTW_MEASURE);
#else // _SP_COMP_
			alloc_local = fftwf_mpi_local_size_3d_transposed(nx, ny, nz, MPI::COMM_WORLD, &local_n0, &local_0_start, &local_n1, &local_1_start);
			rhs = fftwf_alloc_real(alloc_local);
			fwd = fftwf_mpi_plan_r2r_3d(nx, ny, nz, rhs, rhs, MPI::COMM_WORLD, FFTW_R2HC, FFTW_REDFT11, FFTW_R2HC, FFTW_MEASURE);
			bwd = fftwf_mpi_plan_r2r_3d(nx, ny, nz, rhs, rhs, MPI::COMM_WORLD, FFTW_HC2R, FFTW_REDFT11, FFTW_HC2R, FFTW_MEASURE);
#endif // _SP_COMP_
		}
	}
	
	void _cub2fftw(TGrid& grid, Real * out, const size_t nx, const size_t ny, const size_t nz) const
	{
		vector<BlockInfo> infos = grid.getBlocksInfo();
		const size_t N = infos.size();
		const size_t mybpd[3] = {
				grid.getResidentBlocksPerDimension(0),
				grid.getResidentBlocksPerDimension(1),
				grid.getResidentBlocksPerDimension(2)
		};
		
#pragma omp parallel for
		for(int i=0; i<N; ++i) {
			const BlockInfo info = infos[i];
			BlockType& b = *(BlockType*)infos[i].ptrBlock;
			
			// indexing for fftw - [x][y][z] (z is the fastest running index), block part of the index
			const size_t offset = BlockType::sizeZ*info.indexLocal[2] + nz*BlockType::sizeY*info.indexLocal[1] + nz*mybpd[1]*BlockType::sizeY*BlockType::sizeX*info.indexLocal[0];
			
			for(int iz=0; iz<BlockType::sizeZ; iz++)
			for(int iy=0; iy<BlockType::sizeY; iy++)
			for(int ix=0; ix<BlockType::sizeX; ix++) {
				// indexing for fftw - [x][y] (y is the fastest running index), local part of the index
				const size_t dest_index = offset + iz + nz*iy +  BlockType::sizeY*mybpd[1]*nz*ix;

				assert(dest_index>=0 && dest_index<nx*ny*nz); // linking error with openmp: http://lists.apple.com/archives/xcode-users/2011/Oct/msg00252.html
				TStreamer::operate(b(ix,iy,iz), &out[dest_index]);
			}
		}
	}
	
	virtual void _solve(Real * in_out, const size_t nx, const size_t ny, const size_t nz, const Real norm_factor, const Real h)
	{
		if (TStreamer::channels != 1) {
			cout << "PoissonSolverScalar::PoissonSolverScalar(): Error: TStreamer::channels is " << TStreamer::channels << " and it should be 1. Aborting\n";
			abort();
		}
		
		const Real h2 = h*h;
		const Real factor = h2*norm_factor;
		
		const Real fx =    M_PI/nx;
		const Real fy = 2.*M_PI/ny;
		const Real fz = 2.*M_PI/nz;
		
#pragma omp parallel for
		for(int j = 0; j<local_n1; ++j) {
			const Real cosj = cos(fy*(local_1_start+j));
			const Real cos2j = cos(2*fy*(local_1_start+j));
			
			for(int i=0; i<nx; ++i) {
				const Real cosi = cos(fx*(i+.5));
				const Real cos2i = cos(2.*fx*(i+.5));
				
				for(int k=0; k<nz; ++k) {
					const int linidx = i*nz+j*nz*nx+k;
					assert(linidx >=0 && linidx<nx*local_n1*nz); // linking error with openmp
					assert(linidx < alloc_local);
					
					// based on the 5 point stencil in 1D (h^4 error)
					/*
					 const Real denomY = 32.*cos(fy*(j+.5)) - 2.*cos(2*fy*(j+.5)) - 30.;
					 const Real denomX = 32.*cos(fx*i)      - 2.*cos(2.*fx*i)     - 30.;
					 const Real denomZ = 32.*cos(fz*k)      - 2.*cos(2.*fz*k)     - 30.;
					 const Real denom = denomX + denomY + denomZ;
					 */
					const Real denom = 32. * (cosj + cosi + cos(fz*k)) - 2. * (cos2j + cos2i + cos(2.*fz*k)) - 90.;
					const Real inv_denom = (denom==0)? 0.:1./denom;
					const Real fatfactor = 12. * inv_denom * factor;
					
					// based on the 3 point stencil in 1D (h^2 error)
					//const Real denom = 2.*(cos(2.*M_PI*i/nx) + cos(2.*M_PI*j/ny)) - 4.;
					//const Real inv_denom = (denom==0)? 0.:1./denom;
					//const Real fatfactor = inv_denom * factor;
					
					// this is to check the transform only
					//const Real fatfactor = norm_factor;
					in_out[linidx] *= fatfactor;
				}
			}
		}
		//this is sparta!
		//in_out[0] = 0; // WTF? this prevents correct fw/bw if not transformations happen, but has no influence on the 2nd derivative
		// this element should be removed for the FFT but not for the DCT! how to solve this issue?
	}
	
	void _fftw2cub(Real * out, TGrid& grid, const size_t nx, const size_t ny, const size_t nz) const
	{
		vector<BlockInfo> infos = grid.getBlocksInfo();
		const size_t N = infos.size();
		const size_t mybpd[3] = {
				grid.getResidentBlocksPerDimension(0),
				grid.getResidentBlocksPerDimension(1),
				grid.getResidentBlocksPerDimension(2)
		};
		
#pragma omp parallel for
		for(int i=0; i<N; ++i) {
			const BlockInfo info = infos[i];
			BlockType& b = *(BlockType*)infos[i].ptrBlock;
			
			const size_t offset = BlockType::sizeZ*info.indexLocal[2] + nz*BlockType::sizeY*info.indexLocal[1] + mybpd[1]*BlockType::sizeY*nz*BlockType::sizeX*info.indexLocal[0];
			
			for(int iz=0; iz<BlockType::sizeZ; iz++)
			for(int iy=0; iy<BlockType::sizeY; iy++)
			for(int ix=0; ix<BlockType::sizeX; ix++) {
				const size_t src_index = offset + iz + nz*iy + BlockType::sizeY*mybpd[1]*nz*ix;

				assert(src_index>=0 && src_index<nx*ny*nz); // linking error with openmp
				TStreamer::operate(&out[src_index], b(ix,iy,iz));
			}
		}
	}
	
public:
	Real * data; // rhs in _setup, out in cub2fftw and fftw2cub
	
	PoissonSolverScalarFFTW_MPI_DCT(const int desired_threads, TGrid& grid): FFTWBase_MPI(desired_threads), initialized(false)
	{
		// dimensions of blocks and whole grid
		const int bs[3] = {BlockType::sizeX, BlockType::sizeY, BlockType::sizeZ};
		const size_t gsize[3] = {grid.getBlocksPerDimension(0)*bs[0], grid.getBlocksPerDimension(1)*bs[1], grid.getBlocksPerDimension(2)*bs[2]};
		
		profiler.push_start("SETUP");
		_setup(data, gsize[0], gsize[1], gsize[2]);
		profiler.pop_stop();
	}
	
	void solve(TGrid& grid)
	{
		// dimensions of blocks and whole grid
		const int bs[3] = {BlockType::sizeX, BlockType::sizeY, BlockType::sizeZ};
		const size_t gsize[3] = {grid.getBlocksPerDimension(0)*bs[0], grid.getBlocksPerDimension(1)*bs[1], grid.getBlocksPerDimension(2)*bs[2]};
		
		profiler.push_start("CUB2FFTW");
		_cub2fftw(grid, data, local_n0, gsize[1], gsize[2]);
		profiler.pop_stop();
		
		profiler.push_start("FFTW FORWARD");
#ifndef _SP_COMP_
		fftw_execute(fwd);
#else // _SP_COMP_
		fftwf_execute(fwd);
#endif // _SP_COMP_
		profiler.pop_stop();
		
		const Real norm_factor = .5/(gsize[0]*gsize[1]*gsize[2]); // needs to be checked!
		const Real h = grid.getBlocksInfo().front().h_gridpoint;
		assert(1./gsize[0]==h);
		
		profiler.push_start("SOLVE");
		_solve(data, gsize[0], gsize[1], gsize[2], norm_factor, h);
		profiler.pop_stop();

		profiler.push_start("FFTW INVERSE");
#ifndef _SP_COMP_
		fftw_execute(bwd);
#else // _SP_COMP_
		fftwf_execute(bwd);
#endif // _SP_COMP_
		profiler.pop_stop();
		
		profiler.push_start("FFTW2CUB");
		_fftw2cub(data, grid, gsize[0], gsize[1], gsize[2]);
		profiler.pop_stop();
		
		//profiler.printSummary();
	}
	
	void dispose()
	{
		if (initialized) {
			initialized = false;
			
#ifndef _SP_COMP_
			fftw_destroy_plan(fwd);
			fftw_destroy_plan(bwd);
			
			fftw_free(data);
#else // _SP_COMP_
			fftwf_destroy_plan(fwd);
			fftwf_destroy_plan(bwd);
			
			fftwf_free(data);
#endif // _SP_COMP_
			FFTWBase_MPI::dispose();
		}
	}
};
