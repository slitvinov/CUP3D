#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mpi.h>
#include <omp.h>
#include <random>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "Base.h"
#include "AlignedAllocator.h"
#include "Grid.h"
#include "GridMPI.h"
#include "BlockInfo.h"
#include "BlockLab.h"
#include "BlockLabMPI.h"
#include "Definitions.h"
#include "AMR_MeshAdaptation.h"
#include "LDefinitions.h"
#include "SimulationData.h"
#include "Operator.h"
#include "AdvectionDiffusion.h"
#include "AdvectionDiffusionImplicit.h"
#include "ArgumentParser.h"
#include "BufferedLogger.h"
#include "CarlingFish.h"
#include "FishLibrary.h"
#include "FishShapes.h"
#include "ComputeDissipation.h"
#include "Cylinder.h"
#include "ObstacleLibrary.h"
#include "CylinderNozzle.h"
#include "ArgumentParser.h"
#include "DiffusionSolverAMRKernels.h"
#include "Ellipsoid.h"
#include "ExternalForcing.h"
#include "ExternalObstacle.h"
#include "happly.h"
#include "Fish.h"
#include "HDF5Dumper.h"
#include "FixMassFlux.h"
#include "FluidSolidForces.h"
#include "InitialConditions.h"
#include "ProcessHelpers.h"
#include "ObstacleVector.h"
#include "PoissonSolverBase.h"
#include "Pipe.h"
#include "Naca.h"
#include "Obstacle.h"
#include "ObstacleFactory.h"
#include "FactoryFileLineParser.h"
#include "SmartNaca.h"
#include "Plate.h"
#include "Sphere.h"
#include "StefanFish.h"
#include "ObstaclesCreate.h"
#include "ObstaclesUpdate.h"
#include "Penalization.h"
#include "PoissonSolverAMR.h"
#include "PoissonSolverAMRKernels.h"
#include "PressureProjection.h"
#include "Profiler.h"

CubismUP_3D_NAMESPACE_BEGIN

    class Simulation {
protected:
  cubism::ArgumentParser parser;

public:
  SimulationData sim;

  void initialGridRefinement();
  void serialize(const std::string append = std::string());
  void deserialize();
  void setupOperators();
  void setupGrid();
  void _ic();

  // Simulation(MPI_Comm mpicomm, cubism::ArgumentParser &parser);
  Simulation(int argc, char **argv, MPI_Comm comm);

  void init();

  void simulate();

  /// Manually trigger mesh adaptation.
  void adaptMesh();

  /* Get reference to the obstacle container. */
  const std::vector<std::shared_ptr<Obstacle>> &getShapes() const;

  /* Calculate maximum allowed time step, including CFL and ramp-up. */
  Real calcMaxTimestep();

  /*
   * Perform one timestep of the simulation.
   *
   * Returns true if the simulation is finished.
   */
  bool advance(Real dt);

  /// Compute vorticity and store to tmpU, tmpV and tmpW.
  void computeVorticity();

  /// Insert the operator at the end of the pipeline.
  void insertOperator(std::shared_ptr<Operator> op);
};

/** Create a Simulation object from a vector of command-line arguments.

    The argv vector should NOT contain the argv[0] argument, it is filled with
    a dummy value instead.
*/
std::shared_ptr<Simulation>
createSimulation(MPI_Comm comm, const std::vector<std::string> &argv);

CubismUP_3D_NAMESPACE_END

CubismUP_3D_NAMESPACE_BEGIN

//#define WENO
#ifdef PRESERVE_SYMMETRY
#define DISABLE_OPTIMIZATIONS __attribute__((optimize("-O1")))
#else
#define DISABLE_OPTIMIZATIONS
#endif

    struct KernelAdvectDiffuse {
  KernelAdvectDiffuse(const SimulationData &s, const Real a_coef)
      : sim(s), coef(a_coef) {}
  const SimulationData &sim;
  const Real dt = sim.dt;
  const Real mu = sim.nu;
  const Real coef;
  const std::array<Real, 3> &uInf = sim.uinf;
  const std::vector<BlockInfo> &tmpVInfo = sim.tmpVInfo();
  const StencilInfo stencil{-3, -3, -3, 4, 4, 4, false, {0, 1, 2}};
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
#ifdef WENO
  DISABLE_OPTIMIZATIONS
  inline Real weno5_plus(const Real &um2, const Real &um1, const Real &u,
                         const Real &up1, const Real &up2) const {
    const Real exponent = 2;
    const Real e = 1e-6;
    const Real b1 = 13.0 / 12.0 * pow((um2 + u) - 2 * um1, 2) +
                    0.25 * pow((um2 + 3 * u) - 4 * um1, 2);
    const Real b2 =
        13.0 / 12.0 * pow((um1 + up1) - 2 * u, 2) + 0.25 * pow(um1 - up1, 2);
    const Real b3 = 13.0 / 12.0 * pow((u + up2) - 2 * up1, 2) +
                    0.25 * pow((3 * u + up2) - 4 * up1, 2);
    const Real g1 = 0.1;
    const Real g2 = 0.6;
    const Real g3 = 0.3;
    const Real what1 = g1 / pow(b1 + e, exponent);
    const Real what2 = g2 / pow(b2 + e, exponent);
    const Real what3 = g3 / pow(b3 + e, exponent);
    const Real aux = 1.0 / ((what1 + what3) + what2);
    const Real w1 = what1 * aux;
    const Real w2 = what2 * aux;
    const Real w3 = what3 * aux;
    const Real f1 = (11.0 / 6.0) * u + ((1.0 / 3.0) * um2 - (7.0 / 6.0) * um1);
    const Real f2 = (5.0 / 6.0) * u + ((-1.0 / 6.0) * um1 + (1.0 / 3.0) * up1);
    const Real f3 = (1.0 / 3.0) * u + ((+5.0 / 6.0) * up1 - (1.0 / 6.0) * up2);
    return (w1 * f1 + w3 * f3) + w2 * f2;
  }
  DISABLE_OPTIMIZATIONS
  inline Real weno5_minus(const Real &um2, const Real &um1, const Real &u,
                          const Real &up1, const Real &up2) const {
    const Real exponent = 2;
    const Real e = 1e-6;
    const Real b1 = 13.0 / 12.0 * pow((um2 + u) - 2 * um1, 2) +
                    0.25 * pow((um2 + 3 * u) - 4 * um1, 2);
    const Real b2 =
        13.0 / 12.0 * pow((um1 + up1) - 2 * u, 2) + 0.25 * pow(um1 - up1, 2);
    const Real b3 = 13.0 / 12.0 * pow((u + up2) - 2 * up1, 2) +
                    0.25 * pow((3 * u + up2) - 4 * up1, 2);
    const Real g1 = 0.3;
    const Real g2 = 0.6;
    const Real g3 = 0.1;
    const Real what1 = g1 / pow(b1 + e, exponent);
    const Real what2 = g2 / pow(b2 + e, exponent);
    const Real what3 = g3 / pow(b3 + e, exponent);
    const Real aux = 1.0 / ((what1 + what3) + what2);
    const Real w1 = what1 * aux;
    const Real w2 = what2 * aux;
    const Real w3 = what3 * aux;
    const Real f1 = (1.0 / 3.0) * u + ((-1.0 / 6.0) * um2 + (5.0 / 6.0) * um1);
    const Real f2 = (5.0 / 6.0) * u + ((1.0 / 3.0) * um1 - (1.0 / 6.0) * up1);
    const Real f3 = (11.0 / 6.0) * u + ((-7.0 / 6.0) * up1 + (1.0 / 3.0) * up2);
    return (w1 * f1 + w3 * f3) + w2 * f2;
  }
  inline Real derivative(const Real &U, const Real &um3, const Real &um2,
                         const Real &um1, const Real &u, const Real &up1,
                         const Real &up2, const Real &up3) const {
    Real fp = 0.0;
    Real fm = 0.0;
    if (U > 0) {
      fp = weno5_plus(um2, um1, u, up1, up2);
      fm = weno5_plus(um3, um2, um1, u, up1);
    } else {
      fp = weno5_minus(um1, u, up1, up2, up3);
      fm = weno5_minus(um2, um1, u, up1, up2);
    }
    return (fp - fm);
  }
#else
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
#endif

  void operator()(const VectorLab &lab, const BlockInfo &info) const {
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
      return; // no flux corrections needed for this block

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
  const std::vector<BlockInfo> &velInfo = sim.velInfo();
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  const size_t Nblocks = velInfo.size();

#if 0
    //Perform midpoint integration of equation: du/dt = - (u * nabla) u + nu Delta u

    vOld.resize(Nx*Ny*Nz*Nblocks*3);

    //1.Save u^{n} to Vold
#pragma omp parallel for
    for(size_t i=0; i<Nblocks; i++)
    {
        const VectorBlock & V = (*sim.vel)(i);
        for (int z=0; z<Nz; ++z)
        for (int y=0; y<Ny; ++y)
        for (int x=0; x<Nx; ++x)
        {
          const int idx = i*Nx*Ny*Nz*3+z*Ny*Nx*3+y*Nx*3+x*3;
          vOld[idx  ] = V(x,y,z).u[0];
          vOld[idx+1] = V(x,y,z).u[1];
          vOld[idx+2] = V(x,y,z).u[2];
        }
    }

    // 2. Set u^{n+1/2} = u^{n} + 0.5*dt*RHS(u^{n})
    const KernelAdvectDiffuse step1(sim,0.5);
    compute<VectorLab>(step1,sim.vel,sim.tmpV); //Store 0.5*dt*RHS(u^{n}) to tmpV
#pragma omp parallel for
    for(size_t i=0; i<Nblocks; i++)//Set u^{n+1/2} = u^{n} + 0.5*dt*RHS(u^{n})
    {
        const Real ih3 = 1.0/(velInfo[i].h*velInfo[i].h*velInfo[i].h);
        const VectorBlock & tmpV = (*sim.tmpV)(i);
        VectorBlock & V          = (*sim.vel )(i);

        for (int z=0; z<Nz; ++z)
        for (int y=0; y<Ny; ++y)
        for (int x=0; x<Nx; ++x)
        {
          const int idx = i*Nx*Ny*Nz*3+z*Ny*Nx*3+y*Nx*3+x*3;
          V(x,y,z).u[0] = vOld[idx  ] + tmpV(x,y,z).u[0]*ih3;
          V(x,y,z).u[1] = vOld[idx+1] + tmpV(x,y,z).u[1]*ih3;
          V(x,y,z).u[2] = vOld[idx+2] + tmpV(x,y,z).u[2]*ih3;
        }
    }

    // 3. Set u^{n+1} = u^{n} + dt*RHS(u^{n+1/2})
    const KernelAdvectDiffuse step2(sim,1.0);
    compute<VectorLab>(step2,sim.vel,sim.tmpV);//Store dt*RHS(u^{n+1/2}) to tmpV
#pragma omp parallel for
    for(size_t i=0; i<Nblocks; i++)//Set u^{n+1} = u^{n} + dt*RHS(u^{n+1/2})
    {
        const Real ih3 = 1.0/(velInfo[i].h*velInfo[i].h*velInfo[i].h);
        const VectorBlock & tmpV = (*sim.tmpV)(i);
        VectorBlock & V          = (*sim.vel )(i);
        for (int z=0; z<Nz; ++z)
        for (int y=0; y<Ny; ++y)
        for (int x=0; x<Nx; ++x)
        {
          const int idx = i*Nx*Ny*Nz*3+z*Ny*Nx*3+y*Nx*3+x*3;
          V(x,y,z).u[0] = vOld[idx  ] + tmpV(x,y,z).u[0]*ih3;
          V(x,y,z).u[1] = vOld[idx+1] + tmpV(x,y,z).u[1]*ih3;
          V(x,y,z).u[2] = vOld[idx+2] + tmpV(x,y,z).u[2]*ih3;
        }
    }
#else
  // Low-storage 3rd-order Runge Kutta
  const KernelAdvectDiffuse step(sim, 1.0);
  const Real alpha[3] = {1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0};
  const Real beta[3] = {-5.0 / 9.0, -153.0 / 128.0, 0.0};

#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    VectorBlock &tmpV = (*sim.tmpV)(i);
    tmpV.clear();
  }

  for (int RKstep = 0; RKstep < 3; RKstep++) {
    compute<VectorLab>(step, sim.vel, sim.tmpV); // Store dt*RHS(u) to tmpV
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

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN

//#define WENO
#ifdef PRESERVE_SYMMETRY
#define DISABLE_OPTIMIZATIONS __attribute__((optimize("-O1")))
#else
#define DISABLE_OPTIMIZATIONS
#endif

    struct KernelDiffusionRHS {
  SimulationData &sim;
  StencilInfo stencil = StencilInfo(-1, -1, -1, 2, 2, 2, false, {0, 1, 2});
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  const std::vector<BlockInfo> &tmpVInfo = sim.tmpVInfo();

  KernelDiffusionRHS(SimulationData &s) : sim(s) {}

  void operator()(const VectorLab &lab, const BlockInfo &info) const {
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
      return; // no flux corrections needed for this block

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
  const std::vector<BlockInfo> &tmpVInfo = sim.tmpVInfo();
  const StencilInfo stencil{-3, -3, -3, 4, 4, 4, false, {0, 1, 2}};
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
#ifdef WENO
  DISABLE_OPTIMIZATIONS
  inline Real weno5_plus(const Real &um2, const Real &um1, const Real &u,
                         const Real &up1, const Real &up2) const {
    const Real exponent = 2;
    const Real e = 1e-6;
    const Real b1 = 13.0 / 12.0 * pow((um2 + u) - 2 * um1, 2) +
                    0.25 * pow((um2 + 3 * u) - 4 * um1, 2);
    const Real b2 =
        13.0 / 12.0 * pow((um1 + up1) - 2 * u, 2) + 0.25 * pow(um1 - up1, 2);
    const Real b3 = 13.0 / 12.0 * pow((u + up2) - 2 * up1, 2) +
                    0.25 * pow((3 * u + up2) - 4 * up1, 2);
    const Real g1 = 0.1;
    const Real g2 = 0.6;
    const Real g3 = 0.3;
    const Real what1 = g1 / pow(b1 + e, exponent);
    const Real what2 = g2 / pow(b2 + e, exponent);
    const Real what3 = g3 / pow(b3 + e, exponent);
    const Real aux = 1.0 / ((what1 + what3) + what2);
    const Real w1 = what1 * aux;
    const Real w2 = what2 * aux;
    const Real w3 = what3 * aux;
    const Real f1 = (11.0 / 6.0) * u + ((1.0 / 3.0) * um2 - (7.0 / 6.0) * um1);
    const Real f2 = (5.0 / 6.0) * u + ((-1.0 / 6.0) * um1 + (1.0 / 3.0) * up1);
    const Real f3 = (1.0 / 3.0) * u + ((+5.0 / 6.0) * up1 - (1.0 / 6.0) * up2);
    return (w1 * f1 + w3 * f3) + w2 * f2;
  }
  DISABLE_OPTIMIZATIONS
  inline Real weno5_minus(const Real &um2, const Real &um1, const Real &u,
                          const Real &up1, const Real &up2) const {
    const Real exponent = 2;
    const Real e = 1e-6;
    const Real b1 = 13.0 / 12.0 * pow((um2 + u) - 2 * um1, 2) +
                    0.25 * pow((um2 + 3 * u) - 4 * um1, 2);
    const Real b2 =
        13.0 / 12.0 * pow((um1 + up1) - 2 * u, 2) + 0.25 * pow(um1 - up1, 2);
    const Real b3 = 13.0 / 12.0 * pow((u + up2) - 2 * up1, 2) +
                    0.25 * pow((3 * u + up2) - 4 * up1, 2);
    const Real g1 = 0.3;
    const Real g2 = 0.6;
    const Real g3 = 0.1;
    const Real what1 = g1 / pow(b1 + e, exponent);
    const Real what2 = g2 / pow(b2 + e, exponent);
    const Real what3 = g3 / pow(b3 + e, exponent);
    const Real aux = 1.0 / ((what1 + what3) + what2);
    const Real w1 = what1 * aux;
    const Real w2 = what2 * aux;
    const Real w3 = what3 * aux;
    const Real f1 = (1.0 / 3.0) * u + ((-1.0 / 6.0) * um2 + (5.0 / 6.0) * um1);
    const Real f2 = (5.0 / 6.0) * u + ((1.0 / 3.0) * um1 - (1.0 / 6.0) * up1);
    const Real f3 = (11.0 / 6.0) * u + ((-7.0 / 6.0) * up1 + (1.0 / 3.0) * up2);
    return (w1 * f1 + w3 * f3) + w2 * f2;
  }
  inline Real derivative(const Real &U, const Real &um3, const Real &um2,
                         const Real &um1, const Real &u, const Real &up1,
                         const Real &up2, const Real &up3) const {
    Real fp = 0.0;
    Real fm = 0.0;
    if (U > 0) {
      fp = weno5_plus(um2, um1, u, up1, up2);
      fm = weno5_plus(um3, um2, um1, u, up1);
    } else {
      fp = weno5_minus(um1, u, up1, up2, up3);
      fm = weno5_minus(um2, um1, u, up1, up2);
    }
    return (fp - fm);
  }
#else
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
#endif

  void operator()(const VectorLab &lab, const BlockInfo &info) const {
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
      return; // no flux corrections needed for this block

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
  const std::vector<BlockInfo> &velInfo = sim.velInfo();
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  const size_t Nblocks = velInfo.size();

  pressure.resize(Nblocks * Nx * Ny * Nz);
  velocity.resize(Nblocks * Nx * Ny * Nz * 3);

  // Explicit Euler timestep for advection terms. We also store pressure field.
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
    for (size_t i = 0; i < Nblocks;
         i++) // Set u^{n+1/2} = u^{n} + 0.5*dt*RHS(u^{n})
    {
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
    for (size_t i = 0; i < Nblocks;
         i++) // Set u^{n+1/2} = u^{n} + 0.5*dt*RHS(u^{n})
    {
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

CubismUP_3D_NAMESPACE_END

    namespace cubism {

  ///////////////////////////////////////////////////////////
  // Value
  ///////////////////////////////////////////////////////////
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

  ///////////////////////////////////////////////////////////
  // CommandlineParser
  ///////////////////////////////////////////////////////////
  static inline void _normalizeKey(std::string & key) {
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
        printf("Runtime option NOT SPECIFIED! ABORTING! name: %s\n",
               key.data());
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
    return end != s.c_str(); // only care if the number is numeric or not.  This
                             // includes nan and inf
  }

  CommandlineParser::CommandlineParser(const int argc, char **argv)
      : iArgC(argc), vArgV(argv), bStrictMode(false), bVerbose(true) {
    // parse commandline <key> <value> pairs.  Key passed on the command
    // line must start with a leading dash (-). For example:
    // -mykey myvalue0 [myvalue1 ...]
    for (int i = 1; i < argc; i++)
      if (argv[i][0] == '-') {
        std::string values = "";
        int itemCount = 0;

        // check if the current key i is a list of values. If yes,
        // concatenate them into a string
        for (int j = i + 1; j < argc; j++) {
          // if the current value is numeric and (possibly) negative,
          // do not interpret it as a key.
          // XXX: [fabianw@mavt.ethz.ch; 2019-03-28] WARNING:
          // This will treat -nan as a NUMBER and not as a KEY
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
        key.erase(0, 1);   // remove leading '-'
        if (key[0] == '+') // for key concatenation
        {
          key.erase(0, 1);
          if (!_existKey(key, mapArguments))
            mapArguments[key] = Value(values); // skip leading white space
          else
            mapArguments[key] += Value(values);
        } else // regular key
        {
          if (!_existKey(key, mapArguments))
            mapArguments[key] = Value(values);
        }

        i += itemCount;
      }

    mute();
    // printf("found %ld arguments of %d\n",mapArguments.size(),argc);
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

  void CommandlineParser::print_args() {
    for (std::map<std::string, Value>::iterator it = mapArguments.begin();
         it != mapArguments.end(); it++) {
      std::cout.width(50);
      std::cout.fill('.');
      std::cout << std::left << it->first;
      std::cout << ": " << it->second.asString() << std::endl;
    }
  }

  ///////////////////////////////////////////////////////////
  // ArgumentParser
  ///////////////////////////////////////////////////////////
  void ArgumentParser::_ignoreComments(std::istream & stream,
                                       const char commentChar) {
    stream >> std::ws;
    int nextchar = stream.peek();
    while (nextchar == commentChar) {
      stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      stream >> std::ws;
      nextchar = stream.peek();
    }
  }

  void ArgumentParser::_parseFile(std::ifstream & stream, ArgMap & container) {
    // read (key value) pairs from input file, ignore comments
    // beginning with commentStart
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
        if (!_existKey(key, container)) // skip leading white space
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
      _parseFile(confFile,
                 myFMap); // we keep a reference for each separate file read
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

  void ArgumentParser::write_runtime_environment() const {
    time_t rawtime;
    std::time(&rawtime);
    struct tm *timeinfo = std::localtime(&rawtime);
    char buf[256];
    std::strftime(buf, 256, "%A, %h %d %Y, %r", timeinfo);

    std::ofstream runtime("runtime_environment.conf");
    runtime << commentStart << " RUNTIME ENVIRONMENT SETTINGS" << std::endl;
    runtime << commentStart << " ============================" << std::endl;
    runtime << commentStart << " " << buf << std::endl;
    runtime << commentStart
            << " Use this file to set runtime parameter interactively."
            << std::endl;
    runtime << commentStart
            << " The parameter are read every \"refreshperiod\" steps."
            << std::endl;
    runtime << commentStart
            << " When editing this file, you may use comments and string "
               "concatenation."
            << std::endl;
    runtime
        << commentStart
        << " The simulation can be terminated without killing it by setting "
           "\"exit\" to true."
        << std::endl;
    runtime << commentStart
            << " (This will write a serialized restart state. Set \"exitsave\" "
               "to false if not desired.)"
            << std::endl;
    runtime << commentStart << std::endl;
    runtime << commentStart
            << " !!! WARNING !!! EDITING THIS FILE CAN POTENTIALLY CRASH YOUR "
               "SIMULATION !!! WARNING !!!"
            << std::endl;
    for (typename std::map<std::string, Value>::const_iterator it =
             mapArguments.begin();
         it != mapArguments.end(); ++it)
      runtime << it->first << '\t' << it->second << std::endl;
  }

  void ArgumentParser::read_runtime_environment() {
    mapRuntime.clear();
    std::ifstream runtime("runtime_environment.conf");
    if (runtime.good())
      _parseFile(runtime, mapRuntime);
    runtime.close();
  }

  Value &ArgumentParser::parseRuntime(std::string key) {
    _normalizeKey(key);
    if (!_existKey(key, mapRuntime)) {
      printf("ERROR: Runtime parsing for key %s NOT FOUND!! Check your "
             "runtime_environment.conf file\n",
             key.data());
      abort();
    }
    return mapRuntime[key];
  }

  void ArgumentParser::print_args() {
    std::cout
        << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
           "~~~~~~~"
        << std::endl;
    std::cout << "* Summary:" << std::endl;
    std::cout << "*    Parameter read from command line:                "
              << from_commandline.size() << std::endl;
    size_t nFiles = 0;
    size_t nFileParameter = 0;
    for (FileMap::const_iterator it = from_files.begin();
         it != from_files.end(); ++it) {
      if (it->second->size() > 0) {
        ++nFiles;
        nFileParameter += it->second->size();
      }
    }
    std::cout << "*    Parameter read from " << std::setw(3) << std::right
              << nFiles << " file(s):                 " << nFileParameter
              << std::endl;
    std::cout << "*    Parameter read from defaults in code:            "
              << from_code.size() << std::endl;
    std::cout << "*    Total number of parameter read from all sources: "
              << mapArguments.size() << std::endl;
    std::cout
        << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
           "~~~~~~~"
        << std::endl;

    // command line given arguments
    if (!from_commandline.empty()) {
      std::cout << "* Command Line:" << std::endl;
      std::cout
          << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
             "~~~~~~~~~"
          << std::endl;
      for (ArgMap::iterator it = from_commandline.begin();
           it != from_commandline.end(); it++) {
        std::cout.width(50);
        std::cout.fill('.');
        std::cout << std::left << it->first;
        std::cout << ": " << it->second.asString() << std::endl;
      }
      std::cout
          << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
             "~~~~~~~~~"
          << std::endl;
    }

    // options read from input files
    if (!from_files.empty()) {
      for (FileMap::iterator itFile = from_files.begin();
           itFile != from_files.end(); itFile++) {
        if (!itFile->second->empty()) {
          std::cout << "* File: " << itFile->first << std::endl;
          std::cout
              << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                 "~~~~~~~~~~~~~"
              << std::endl;
          ArgMap &fileArgs = *(itFile->second);
          for (ArgMap::iterator it = fileArgs.begin(); it != fileArgs.end();
               it++) {
            std::cout.width(50);
            std::cout.fill('.');
            std::cout << std::left << it->first;
            std::cout << ": " << it->second.asString() << std::endl;
          }
          std::cout
              << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                 "~~~~~~~~~~~~~"
              << std::endl;
        }
      }
    }

    // defaults defined in code
    if (!from_code.empty()) {
      std::cout << "* Defaults in Code:" << std::endl;
      std::cout
          << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
             "~~~~~~~~~"
          << std::endl;
      for (pArgMap::iterator it = from_code.begin(); it != from_code.end();
           it++) {
        std::cout.width(50);
        std::cout.fill('.');
        std::cout << std::left << it->first;
        std::cout << ": " << it->second->asString() << std::endl;
      }
      std::cout
          << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
             "~~~~~~~~~"
          << std::endl;
    }
  }

} // namespace cubism

namespace cubismup3d {

BufferedLogger logger;

struct BufferedLoggerImpl {
  struct Stream {
    std::stringstream stream;
    int requests_since_last_flush = 0;

    // GN: otherwise icpc complains
    Stream() = default;
    Stream(Stream &&) = default;
    Stream(const Stream &c)
        : requests_since_last_flush(c.requests_since_last_flush) {
      stream << c.stream.rdbuf();
    }
  };
  typedef std::unordered_map<std::string, Stream> container_type;
  container_type files;

  /*
   * Flush a single stream and reset the counter.
   */
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
      // With request_since_last_flush == 0,
      // the first flush will have AUTO_FLUSH_COUNT frames.
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

} // namespace cubismup3d

CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

class CarlingFishMidlineData : public FishMidlineData {
public:
  bool quadraticAmplitude = false;

protected:
  const Real carlingAmp;
  static constexpr Real carlingInv = 0.03125;

  const Real quadraticFactor; // Should be set to 0.1, which gives peak-to-peak
                              // amp of 0.2L (this is physically observed in
                              // most fish species)

  inline Real rampFactorSine(const Real t, const Real T) const {
    // return (t<T ? ( 1 - std::cos(M_PI*t/T) )/2 : 1.0);
    return (t < T ? std::sin(0.5 * M_PI * t / T) : 1.0);
  }

  inline Real rampFactorVelSine(const Real t, const Real T) const {
    // return (t<T ? 0.5*M_PI/T * std::sin(M_PI*t/T) : 0.0);
    return (t < T ? 0.5 * M_PI / T * std::cos(0.5 * M_PI * t / T) : 0.0);
  }

  inline Real getQuadAmp(const Real s) const {
    // Maertens et al. JFM 2017:
    return quadraticFactor *
           (length - .825 * (s - length) + 1.625 * (s * s / length - length));
    // return s*s*quadraticFactor/length;
  }
  inline Real getLinAmp(const Real s) const {
    return carlingAmp * (s + carlingInv * length);
  }

  inline Real getArg(const Real s, const Real t) const {
    return 2.0 * M_PI * (s / (waveLength * length) - t / Tperiod + phaseShift);
  }

public:
  // L=length, T=period, phi=phase shift, _h=grid size, A=amplitude modulation
  CarlingFishMidlineData(Real L, Real T, Real phi, Real _h, Real A)
      : FishMidlineData(L, T, phi, _h, A), carlingAmp(.1212121212 * A),
        quadraticFactor(.1 * A) {
    // FinSize has now been updated with value read from text file. Recompute
    // heights to over-write with updated values
    // printf("Overwriting default tail-fin size for Plain Carling:\n");
    //_computeWidthsHeights();
  }

  virtual void computeMidline(const Real t, const Real dt) override;

  template <bool bQuadratic> void _computeMidlinePosVel(const Real t) {
    const Real rampFac = rampFactorSine(t, Tperiod), dArg = -2 * M_PI / Tperiod;
    const Real rampFacVel = rampFactorVelSine(t, Tperiod);
    {
      const Real arg = getArg(rS[0], t);
      const Real cosa = std::cos(arg), sina = std::sin(arg);
      const Real amp = bQuadratic ? getQuadAmp(rS[0]) : getLinAmp(rS[0]);
      const Real Y = sina * amp, VY = cosa * dArg * amp;
      rX[0] = 0.0;
      vX[0] = 0.0; // rX[0] is constant
      rY[0] = rampFac * Y;
      vY[0] = rampFac * VY + rampFacVel * Y;
      rZ[0] = 0.0;
      vZ[0] = 0.0;
    }
    for (int i = 1; i < Nm; ++i) {
      const Real arg = getArg(rS[i], t);
      const Real cosa = std::cos(arg), sina = std::sin(arg);
      const Real amp = bQuadratic ? getQuadAmp(rS[i]) : getLinAmp(rS[i]);
      const Real Y = sina * amp, VY = cosa * dArg * amp;
      rY[i] = rampFac * Y;
      vY[i] = rampFac * VY + rampFacVel * Y;
      const Real dy = rY[i] - rY[i - 1], ds = rS[i] - rS[i - 1],
                 dVy = vY[i] - vY[i - 1];
      const Real dx = std::sqrt(ds * ds - dy * dy);
      assert(dx > 0);
      rX[i] = rX[i - 1] + dx;
      vX[i] =
          vX[i - 1] - dy / dx * dVy; // use ds^2 = dx^2+dy^2 -> ddx = -dy/dx*ddy
      rZ[i] = 0.0;
      vZ[i] = 0.0;
    }
  }
};

void CarlingFishMidlineData::computeMidline(const Real t, const Real dt) {
  if (quadraticAmplitude)
    _computeMidlinePosVel<true>(t);
  else
    _computeMidlinePosVel<false>(t);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < Nm - 1; i++) {
    const Real ds = rS[i + 1] - rS[i];
    const Real tX = rX[i + 1] - rX[i];
    const Real tY = rY[i + 1] - rY[i];
    const Real tVX = vX[i + 1] - vX[i];
    const Real tVY = vY[i + 1] - vY[i];
    norX[i] = -tY / ds;
    norY[i] = tX / ds;
    norZ[i] = 0.0;
    vNorX[i] = -tVY / ds;
    vNorY[i] = tVX / ds;
    vNorZ[i] = 0.0;
    binX[i] = 0.0;
    binY[i] = 0.0;
    binZ[i] = 1.0;
    vBinX[i] = 0.0;
    vBinY[i] = 0.0;
    vBinZ[i] = 0.0;
  }
  norX[Nm - 1] = norX[Nm - 2];
  norY[Nm - 1] = norY[Nm - 2];
  norZ[Nm - 1] = norZ[Nm - 2];
  vNorX[Nm - 1] = vNorX[Nm - 2];
  vNorY[Nm - 1] = vNorY[Nm - 2];
  vNorZ[Nm - 1] = vNorZ[Nm - 2];
  binX[Nm - 1] = binX[Nm - 2];
  binY[Nm - 1] = binY[Nm - 2];
  binZ[Nm - 1] = binZ[Nm - 2];
  vBinX[Nm - 1] = vBinX[Nm - 2];
  vBinY[Nm - 1] = vBinY[Nm - 2];
  vBinZ[Nm - 1] = vBinZ[Nm - 2];
}

CarlingFish::CarlingFish(SimulationData &s, ArgumentParser &p) : Fish(s, p) {
  // _ampFac=0.0 for towed fish :
  const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
  const bool bQuadratic = p("-bQuadratic").asBool(false);
  const Real Tperiod = p("-T").asDouble(1.0);
  const Real phaseShift = p("-phi").asDouble(0.0);

  CarlingFishMidlineData *localFish =
      new CarlingFishMidlineData(length, Tperiod, phaseShift, sim.hmin, ampFac);

  // generic copy for base class:
  assert(myFish == nullptr);
  myFish = (FishMidlineData *)localFish;

  localFish->quadraticAmplitude = bQuadratic;
  std::string heightName = p("-heightProfile").asString("baseline");
  std::string widthName = p("-widthProfile").asString("baseline");
  MidlineShapes::computeWidthsHeights(heightName, widthName, length, myFish->rS,
                                      myFish->height, myFish->width, myFish->Nm,
                                      sim.rank);

  if (!sim.rank)
    printf("CarlingFish: N:%d, L:%f, T:%f, phi:%f, amplitude:%f\n", myFish->Nm,
           length, Tperiod, phaseShift, ampFac);
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

namespace {

class KernelDissipation {
public:
  const Real dt, nu, center[3];
  Real *QOI;
  StencilInfo stencil{-1, -1, -1, 2, 2, 2, false, {0, 1, 2}};
  StencilInfo stencil2{-1, -1, -1, 2, 2, 2, false, {0}};
  SimulationData &sim;

  const std::vector<cubism::BlockInfo> &chiInfo = sim.chiInfo();

  KernelDissipation(Real _dt, const Real ext[3], Real _nu, Real *RDX,
                    SimulationData &s)
      : dt(_dt), nu(_nu), center{ext[0] / 2, ext[1] / 2, ext[2] / 2}, QOI(RDX),
        sim(s) {}

  void operator()(VectorLab &lab, ScalarLab &pLab, const BlockInfo &info,
                  const BlockInfo &info2) const {
    const Real h = info.h;
    const Real hCube = std::pow(h, 3), inv2h = .5 / h, invHh = 1 / (h * h);
    const ScalarBlock &chiBlock =
        *(ScalarBlock *)chiInfo[info.blockID].ptrBlock;

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
          // vorticity
          const Real WX = inv2h * ((LN.u[2] - LS.u[2]) - (LB.u[1] - LF.u[1]));
          const Real WY = inv2h * ((LB.u[0] - LF.u[0]) - (LE.u[2] - LW.u[2]));
          const Real WZ = inv2h * ((LE.u[1] - LW.u[1]) - (LN.u[0] - LS.u[0]));
          //  - \nabla P \cdot \bm{u}
          const Real dPdx =
              inv2h * (pLab(ix + 1, iy, iz).s - pLab(ix - 1, iy, iz).s);
          const Real dPdy =
              inv2h * (pLab(ix, iy + 1, iz).s - pLab(ix, iy - 1, iz).s);
          const Real dPdz =
              inv2h * (pLab(ix, iy, iz + 1).s - pLab(ix, iy, iz - 1).s);
          //  + \mu \bm{u} \cdot \nabla^2 \bm{u}
          const Real lapU = invHh * (LE.u[0] + LW.u[0] + LN.u[0] + LS.u[0] +
                                     LB.u[0] + LF.u[0] - 6 * L.u[0]);
          const Real lapV = invHh * (LE.u[1] + LW.u[1] + LN.u[1] + LS.u[1] +
                                     LB.u[1] + LF.u[1] - 6 * L.u[1]);
          const Real lapW = invHh * (LE.u[2] + LW.u[2] + LN.u[2] + LS.u[2] +
                                     LB.u[2] + LF.u[2] - 6 * L.u[2]);
          const Real V1 = lapU * L.u[0] + lapV * L.u[1] + lapW * L.u[2];
          // + 2 \mu \bm{D} : \bm{D}
          const Real D11 = inv2h * (LE.u[0] - LW.u[0]); // shear stresses
          const Real D22 = inv2h * (LN.u[1] - LS.u[1]); // shear stresses
          const Real D33 = inv2h * (LB.u[2] - LF.u[2]); // shear stresses
          const Real D12 = inv2h * (LN.u[0] - LS.u[0] + LE.u[1] - LW.u[1]) /
                           2; // shear stresses
          const Real D13 = inv2h * (LB.u[0] - LF.u[0] + LE.u[2] - LW.u[2]) /
                           2; // shear stresses
          const Real D23 = inv2h * (LN.u[2] - LS.u[2] + LB.u[1] - LF.u[1]) /
                           2; // shear stresses
          const Real V2 = D11 * D11 + D22 * D22 + D33 * D33 +
                          2 * (D12 * D12 + D13 * D13 + D23 * D23);

#pragma omp critical
          {
            // three linear invariants (conserved in inviscid and viscous flows)
            // conservation of vorticity: int w dx = 0
            QOI[0] += hCube * WX;
            QOI[1] += hCube * WY;
            QOI[2] += hCube * WZ;
            // conservation of linear impulse: int u dx = 0.5 int (x cross w) dx
            QOI[3] += hCube / 2 * (PY * WZ - PZ * WY);
            QOI[4] += hCube / 2 * (PZ * WX - PX * WZ);
            QOI[5] += hCube / 2 * (PX * WY - PY * WX);
            QOI[6] += hCube * L.u[0];
            QOI[7] += hCube * L.u[1];
            QOI[8] += hCube * L.u[2];
            // conserve ang imp.: int (x cross u)dx = 1/3 int (x cross (x cross
            // w) )dx
            // = 1/3 int x (w \cdot x) - w (x \cdot x) ) dx (some terms cancel)
            QOI[9] += hCube / 3 *
                      (PX * (PY * WY + PZ * WZ) - WX * (PY * PY + PZ * PZ));
            QOI[10] += hCube / 3 *
                       (PY * (PX * WX + PZ * WZ) - WY * (PX * PX + PZ * PZ));
            QOI[11] += hCube / 3 *
                       (PZ * (PX * WX + PY * WY) - WZ * (PX * PX + PY * PY));
            QOI[12] += hCube * (PY * L.u[2] - PZ * L.u[1]);
            QOI[13] += hCube * (PZ * L.u[0] - PX * L.u[2]);
            QOI[14] += hCube * (PX * L.u[1] - PY * L.u[0]);

            // presPow
            // viscPow
            QOI[15] -= (1 - X) * hCube *
                       (dPdx * L.u[0] + dPdy * L.u[1] + dPdz * L.u[2]);
            QOI[16] += (1 - X) * hCube * nu * (V1 + 2 * V2);

            // two quadratic invariants: kinetic energy (from solver) and
            // helicity (conserved in inviscid flows)
            // helicity
            // kineticEn
            // enstrophy
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
  cubism::compute<KernelDissipation, VectorGrid, VectorLab, ScalarGrid,
                  ScalarLab>(diss, *sim.vel, *sim.pres);

  MPI_Allreduce(MPI_IN_PLACE, RDX, 20, MPI_Real, MPI_SUM, sim.comm);

  size_t loc = sim.velInfo().size();
  size_t tot;
  MPI_Reduce(&loc, &tot, 1, MPI_LONG, MPI_SUM, 0, sim.comm);
  if (sim.rank == 0 && sim.muteAll == false) {
    std::ofstream outfile;
    outfile.open("diagnostics.dat", std::ios_base::app);
    if (sim.step == 0)
      outfile << "step_id time circ_x circ_y circ_z linImp_x linImp_y linImp_z "
                 "linMom_x linMom_y linMom_z angImp_x angImp_y angImp_z "
                 "angMom_x angMom_y "
                 "angMom_z presPow viscPow helicity kineticEn enstrophy blocks"
              << std::endl;
    outfile << sim.step << " " << sim.time << " " << RDX[0] << " " << RDX[1]
            << " " << RDX[2] << " " << RDX[3] << " " << RDX[4] << " " << RDX[5]
            << " " << RDX[6] << " " << RDX[7] << " " << RDX[8] << " " << RDX[9]
            << " " << RDX[10] << " " << RDX[11] << " " << RDX[12] << " "
            << RDX[13] << " " << RDX[14] << " " << RDX[15] << " " << RDX[16]
            << " " << RDX[17] << " " << RDX[18] << " " << RDX[19] << " " << tot
            << std::endl;
    outfile.close();
  }
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

namespace DCylinderObstacle {
struct FillBlocks : FillBlocksBase<FillBlocks> {
  const Real radius, halflength, angle, h, safety = (2 + SURFDH) * h;
  const Real cosang = std::cos(angle), sinang = std::sin(angle);
  const Real position[3];
  const Real box[3][2] = {{(Real)position[0] - radius - safety,
                           (Real)position[0] + radius + safety},
                          {(Real)position[1] - radius - safety,
                           (Real)position[1] + radius + safety},
                          {(Real)position[2] - halflength - safety,
                           (Real)position[2] + halflength + safety}};

  FillBlocks(const Real r, const Real halfl, const Real ang, const Real _h,
             const Real p[3])
      : radius(r), halflength(halfl), angle(ang),
        h(_h), position{p[0], p[1], p[2]} {}

  inline bool isTouching(const BlockInfo &info, const ScalarBlock &b) const {
    Real MINP[3], MAXP[3];
    info.pos(MINP, 0, 0, 0);
    info.pos(MAXP, ScalarBlock::sizeX - 1, ScalarBlock::sizeY - 1,
             ScalarBlock::sizeZ - 1);
    const Real intersect[3][2] = {
        {std::max(MINP[0], box[0][0]), std::min(MAXP[0], box[0][1])},
        {std::max(MINP[1], box[1][0]), std::min(MAXP[1], box[1][1])},
        {std::max(MINP[2], box[2][0]), std::min(MAXP[2], box[2][1])}};
    return intersect[0][1] - intersect[0][0] > 0 &&
           intersect[1][1] - intersect[1][0] > 0 &&
           intersect[2][1] - intersect[2][0] > 0;
  }

  inline Real signedDistance(const Real xo, const Real yo,
                             const Real zo) const {
    const Real x = xo - position[0], y = yo - position[1], z = zo - position[2];
    const Real x_rotated = x * cosang + y * sinang;
    const Real planeDist =
        std::min(-x_rotated, radius - std::sqrt(x * x + y * y));
    const Real vertiDist = halflength - std::fabs(z);
    return std::min(planeDist, vertiDist);
  }
};
} // namespace DCylinderObstacle

namespace CylinderObstacle {
struct FillBlocks : FillBlocksBase<FillBlocks> {
  const Real radius, halflength, h, safety = (2 + SURFDH) * h;
  const Real position[3];
  const Real box[3][2] = {{(Real)position[0] - radius - safety,
                           (Real)position[0] + radius + safety},
                          {(Real)position[1] - radius - safety,
                           (Real)position[1] + radius + safety},
                          {(Real)position[2] - halflength - safety,
                           (Real)position[2] + halflength + safety}};

  FillBlocks(const Real r, const Real halfl, const Real _h, const Real p[3])
      : radius(r), halflength(halfl), h(_h), position{p[0], p[1], p[2]} {}

  inline bool isTouching(const BlockInfo &info, const ScalarBlock &b) const {
    Real MINP[3], MAXP[3];
    info.pos(MINP, 0, 0, 0);
    info.pos(MAXP, ScalarBlock::sizeX - 1, ScalarBlock::sizeY - 1,
             ScalarBlock::sizeZ - 1);
    const Real intersect[3][2] = {
        {std::max(MINP[0], box[0][0]), std::min(MAXP[0], box[0][1])},
        {std::max(MINP[1], box[1][0]), std::min(MAXP[1], box[1][1])},
        {std::max(MINP[2], box[2][0]), std::min(MAXP[2], box[2][1])}};
    return intersect[0][1] - intersect[0][0] > 0 &&
           intersect[1][1] - intersect[1][0] > 0 &&
           intersect[2][1] - intersect[2][0] > 0;
  }

  inline Real signedDistance(const Real xo, const Real yo,
                             const Real zo) const {
    const Real x = xo - position[0], y = yo - position[1], z = zo - position[2];
    const Real planeDist = radius - std::sqrt(x * x + y * y);
    const Real vertiDist = halflength - std::fabs(z);
    return std::min(planeDist, vertiDist);
  }
};
} // namespace CylinderObstacle

Cylinder::Cylinder(SimulationData &s, ArgumentParser &p)
    : Obstacle(s, p), radius(.5 * length),
      halflength(p("-halflength").asDouble(.5 * sim.extents[2])) {
  section = p("-section").asString("circular");
  accel = p("-accel").asBool(false);
  if (accel) {
    if (not bForcedInSimFrame[0]) {
      printf("Warning: Cylinder was not set to be forced in x-dir, yet the "
             "accel pattern is active.\n");
    }
    umax = -p("-xvel").asDouble(0.0);
    vmax = -p("-yvel").asDouble(0.0);
    wmax = -p("-zvel").asDouble(0.0);
    tmax = p("-T").asDouble(1.0);
  }
  _init();
}

void Cylinder::_init(void) {
  if (sim.verbose)
    printf("Created Cylinder with radius %f and halflength %f\n", radius,
           halflength);

  // D-cyl can float around the domain, but does not support rotation. TODO
  bBlockRotation[0] = true;
  bBlockRotation[1] = true;
  bBlockRotation[2] = true;
}

void Cylinder::create() {
  const Real h = sim.hmin;
  if (section == "D") {
    const Real angle =
        2 * std::atan2(quaternion[3], quaternion[0]); // planar angle (xy plane)
    const DCylinderObstacle::FillBlocks kernel(radius, halflength, angle, h,
                                               position);
    create_base<DCylinderObstacle::FillBlocks>(kernel);
  } else /* else do square section, but figure how to make code smaller */
  {      /* else normal cylinder */
    const CylinderObstacle::FillBlocks kernel(radius, halflength, h, position);
    create_base<CylinderObstacle::FillBlocks>(kernel);
  }
}

void Cylinder::computeVelocities() {
  if (accel) {
    if (sim.time < tmax)
      transVel_imposed[0] = umax * sim.time / tmax;
    else {
      transVel_imposed[0] = umax;
      transVel_imposed[1] = vmax;
      transVel_imposed[2] = wmax;
    }
  }

  Obstacle::computeVelocities();
}

void Cylinder::finalize() {
  // this method allows any computation that requires the char function
  // to be computed. E.g. compute the effective center of mass or removing
  // momenta from udef
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

CylinderNozzle::CylinderNozzle(SimulationData &s, ArgumentParser &p)
    : Cylinder(s, p), Nactuators(p("-Nactuators").asInt(2)),
      actuator_theta(p("-actuator_theta").asDouble(10.) * M_PI / 180.),
      regularizer(p("-regularizer").asDouble(1.0)),
      ccoef(p("-ccoef").asDouble(0.1)) {
  actuators.resize(Nactuators, 0.);
  actuatorSchedulers.resize(Nactuators);
  actuators_prev_value.resize(Nactuators);
  actuators_next_value.resize(Nactuators);
}

void CylinderNozzle::finalize() {
  const double cd = force[0] / (0.5 * transVel[0] * transVel[0] * 2 * radius *
                                2 * halflength);
  fx_integral += -std::fabs(cd) * sim.dt;

  const Real transition_duration = 0.1;
  for (size_t idx = 0; idx < actuators.size(); idx++) {
    Real dummy;
    actuatorSchedulers[idx].transition(
        sim.time, t_change, t_change + transition_duration,
        actuators_prev_value[idx], actuators_next_value[idx]);
    actuatorSchedulers[idx].gimmeValues(sim.time, actuators[idx], dummy);
  }

  const auto &vInfo = sim.vel->getBlocksInfo();
  const Real dtheta = 2 * M_PI / Nactuators;
  const Real Cx = position[0];
  const Real Cy = position[1];
  const Real Cz = position[2];
  const Real Uact_max =
      ccoef * pow(transVel[0] * transVel[0] + transVel[1] * transVel[1] +
                      transVel[2] * transVel[2],
                  0.5);
  for (size_t i = 0; i < vInfo.size(); i++) {
    const auto &info = vInfo[i];
    if (obstacleBlocks[info.blockID] == nullptr)
      continue; // obst not in block
    ObstacleBlock &o = *obstacleBlocks[info.blockID];
    auto &__restrict__ UDEF = o.udef;

    for (int iz = 0; iz < ScalarBlock::sizeZ; iz++)
      for (int iy = 0; iy < ScalarBlock::sizeY; iy++)
        for (int ix = 0; ix < ScalarBlock::sizeX; ix++) {
          UDEF[iz][iy][ix][0] = 0.0;
          UDEF[iz][iy][ix][1] = 0.0;
          UDEF[iz][iy][ix][2] = 0.0;
          Real p[3];
          info.pos(p, ix, iy, iz);
          const Real x = p[0] - Cx;
          const Real y = p[1] - Cy;
          const Real z = p[2] - Cz;
          const Real r = x * x + y * y;
          if (r > (radius + 2 * info.h) * (radius + 2 * info.h) ||
              r < (radius - 2 * info.h) * (radius - 2 * info.h))
            continue;
          if (std::fabs(z) > 0.99 * halflength)
            continue;
          // if (std::fabs(z) > 0.75*halflength) continue;

          Real theta = atan2(y, x);
          if (theta < 0)
            theta += 2. * M_PI;

          int idx = round(theta / dtheta); // this is the closest actuator
          if (idx == Nactuators)
            idx = 0; // periodic around the cylinder

          const Real theta0 = idx * dtheta;

          const Real phi = theta - theta0;
          if (std::fabs(phi) < 0.5 * actuator_theta ||
              (idx == 0 && std::fabs(phi - 2 * M_PI) < 0.5 * actuator_theta)) {
            const Real rr = radius / pow(r, 0.5);
            const Real ur = Uact_max * rr * actuators[idx] *
                            cos(M_PI * phi / actuator_theta);
            UDEF[iz][iy][ix][0] = ur * cos(theta);
            UDEF[iz][iy][ix][1] = ur * sin(theta);
            UDEF[iz][iy][ix][2] = 0.0;
          }
        }
  }
}

void CylinderNozzle::act(std::vector<Real> action, const int agentID) {
  t_change = sim.time;
  if (action.size() != actuators.size()) {
    std::cerr << "action size needs to be equal to actuators\n";
    fflush(0);
    abort();
  }

  bool bounded = false;
  while (bounded == false) {
    bounded = true;
    Real Q = 0;
    for (size_t i = 0; i < action.size(); i++) {
      Q += action[i];
    }
    Q /= action.size();
    for (size_t i = 0; i < action.size(); i++) {
      action[i] -= Q;
      if (std::fabs(action[i]) > 1.0)
        bounded = false;
      action[i] = std::max(action[i], -1.0);
      action[i] = std::min(action[i], +1.0);
    }
  }

  for (size_t i = 0; i < action.size(); i++) {
    actuators_prev_value[i] = actuators[i];
    actuators_next_value[i] = action[i];
  }
}

Real CylinderNozzle::reward(const int agentID) {
  Real retval = fx_integral / 0.1; // 0.1 is the action times
  fx_integral = 0;
  Real regularizer_sum = 0.0;
  for (size_t idx = 0; idx < actuators.size(); idx++) {
    regularizer_sum += actuators[idx] * actuators[idx];
  }
  regularizer_sum = pow(regularizer_sum, 0.5) / actuators.size(); // O(1)
  const double c = -regularizer;
  return retval + c * regularizer_sum;
}

std::vector<Real> CylinderNozzle::state(const int agentID) {
  std::vector<Real> S;
  const int bins = 16;
  const Real bins_theta = 10 * M_PI / 180.0;
  const Real dtheta = 2. * M_PI / bins;
  std::vector<int> n_s(bins, 0.0);
  std::vector<Real> p_s(bins, 0.0);
  std::vector<Real> o_s(bins, 0.0);
  for (auto &block : obstacleBlocks)
    if (block not_eq nullptr) {
      for (int i = 0; i < block->nPoints; i++) {
        const Real x = block->pX[i] - position[0];
        const Real y = block->pY[i] - position[1];
        const Real z = block->pZ[i] - position[2];
        if (std::fabs(z) > 0.99 * halflength)
          continue;

        Real theta = atan2(y, x);
        if (theta < 0)
          theta += 2. * M_PI;
        int idx = round(theta / dtheta); // this is the closest actuator
        if (idx == bins)
          idx = 0; // periodic around the cylinder
        const Real theta0 = idx * dtheta;
        const Real phi = theta - theta0;
        if (std::fabs(phi) < 0.5 * bins_theta ||
            (idx == 0 && std::fabs(phi - 2 * M_PI) < 0.5 * bins_theta)) {
          const Real p = block->P[i];
          const Real om = block->omegaZ[i];
          n_s[idx]++;
          p_s[idx] += p;
          o_s[idx] += om;
        }
      }
    }

  MPI_Allreduce(MPI_IN_PLACE, n_s.data(), n_s.size(), MPI_INT, MPI_SUM,
                sim.comm);
  for (int idx = 0; idx < bins; idx++) {
    if (n_s[idx] == 0)
      continue;
    p_s[idx] /= n_s[idx];
    o_s[idx] /= n_s[idx];
  }

  for (int idx = 0; idx < bins; idx++)
    S.push_back(p_s[idx]);
  for (int idx = 0; idx < bins; idx++)
    S.push_back(o_s[idx]);
  MPI_Allreduce(MPI_IN_PLACE, S.data(), S.size(), MPI_Real, MPI_SUM, sim.comm);
  S.push_back(-force[0] / (2 * halflength));
  S.push_back(-force[1] / (2 * halflength));
  const Real Re = std::fabs(transVel[0]) * (2 * radius) / sim.nu;
  S.push_back(Re);
  S.push_back(ccoef);

  if (sim.rank == 0)
    for (size_t i = 0; i < S.size(); i++)
      std::cout << S[i] << " ";

  return S;
}

CubismUP_3D_NAMESPACE_END

    /*
    Optimization comments:
      - The innermost loop has to be very simple in order for the compiler to
        optimize it. Temporary accumulators and storage arrays have to be used
    to enable vectorization.

      - In order to vectorize the stencil, the shifted west and east pointers to
    p have to be provided separately, without the compiler knowing that they are
        related to the same buffer.

      - The same would be true for south, north, back and front shifts, but we
    pad the p block not with +/-1 padding but with +/-4, and put a proper offset
        (depending on sizeof(Real)) to have everything nicely aligned with
    respect to the 32B boundary. This was tested only on AVX-256, but should
    work for AVX-512 as well.

      - For correctness, the p pointers must not have __restrict__, since p,
        pW and pE do overlap. (Not important here though, since we removed the
        convergence for loop, see below). All other arrays do have __restrict__,
    so this does not affect vectorization anyway.

      - The outer for loop that repeats the kernel until convergence breaks the
        vectorization of the stencil in gcc, hence it was removed from this
    file.

      - Putting this loop into another function does not help, since the
    compiler merges the two functions and breaks the vectorization. This can be
    fixed by adding `static __attribute__((noinline))` to the kernel function,
    but it is a bit risky, and doesn't seem to improve the generated code. The
    cost of the bare function call here is about 3ns.

      - Not tested here, but unaligned access can be up to 2x slower than
    aligned, so it is important to ensure alignment.
        https://www.agner.org/optimize/blog/read.php?i=423


    Compilation hints:
      - If gcc is used, the Ax--p stencil won't be vectorized unless version 11
    or later is used.

      - -ffast-math might affect ILP and reductions. Not verified.

      - Changing the order of operations may cause the compiler to produce
        different operation order and hence cause the number of convergence
        iterations to change.

      - To show the assembly, use e.g.
          objdump -dS -Mintel --no-show-raw-insn DiffusionSolverAMRKernels.cpp.o
    > DiffusionSolverAMRKErnels.cpp.lst

      - With gcc 11, it might be necessary to use "-g -gdwarf-4" instead of "-g"
        for objdump to work. For more information look here:
        https://gcc.gnu.org/gcc-11/changes.html


    Benchmarks info for Broadwell CPU with AVX2 (256-bit):
      - Computational limit is 2x 256-bit SIMD FMAs per cycle == 16 FLOPs/cycle.

      - Memory limit for L1 cache is 2x256-bit reads and 1x256-bit write per
    cycle. See "Haswell and Broadwell pipeline", section "Read and write
    bandwidth": https://www.agner.org/optimize/microarchitecture.pdf

        These amount to 64B reads and 32B writes per cycle, however we get about
        80% of that, consistent with benchmarks here:
        https://www.agner.org/optimize/blog/read.php?i=423

      - The kernels below are memory bound.
    */

    namespace cubismup3d {
  namespace diffusion_kernels {

  // Note: kDivEpsilon is too small for single precision!
  static constexpr Real kDivEpsilon = 1e-55;
  static constexpr Real kNormRelCriterion = 1e-7;
  static constexpr Real kNormAbsCriterion = 1e-16;
  static constexpr Real kSqrNormRelCriterion =
      kNormRelCriterion * kNormRelCriterion;
  static constexpr Real kSqrNormAbsCriterion =
      kNormAbsCriterion * kNormAbsCriterion;

  /*
  // Reference non-vectorized implementation of the kernel.
  Real kernelDiffusionGetZInnerReference(
      PaddedBlock & __restrict__ p,
      Block & __restrict__ Ax,
      Block & __restrict__ r,
      Block & __restrict__ block,
      const Real sqrNorm0,
      const Real rr)
  {
    Real a2 = 0;
    for (int iz = 0; iz < NZ; ++iz)
    for (int iy = 0; iy < NY; ++iy)
    for (int ix = 0; ix < NX; ++ix) {
      Ax[iz][iy][ix] = p[iz + 1][iy + 1][ix + xPad - 1]
                     + p[iz + 1][iy + 1][ix + xPad + 1]
                     + p[iz + 1][iy + 0][ix + xPad]
                     + p[iz + 1][iy + 2][ix + xPad]
                     + p[iz + 0][iy + 1][ix + xPad]
                     + p[iz + 2][iy + 1][ix + xPad]
                     - 6 * p[iz + 1][iy + 1][ix + xPad];
      a2 += p[iz + 1][iy + 1][ix + xPad] * Ax[iz][iy][ix];
    }

    const Real a = rr / (a2 + kDivEpsilon);
    Real sqrNorm = 0;
    for (int iz = 0; iz < NZ; ++iz)
    for (int iy = 0; iy < NY; ++iy)
    for (int ix = 0; ix < NX; ++ix) {
      block[iz][iy][ix] += a * p[iz + 1][iy + 1][ix + xPad];
      r[iz][iy][ix] -= a * Ax[iz][iy][ix];
      sqrNorm += r[iz][iy][ix] * r[iz][iy][ix];
    }

    const Real beta = sqrNorm / (rr + kDivEpsilon);
    const Real rrNew = sqrNorm;
    const Real norm = std::sqrt(sqrNorm) / N;

    if (norm / std::sqrt(sqrNorm0) < kNormRelCriterion || norm <
  kNormAbsCriterion) return 0;

    for (int iz = 0; iz < NZ; ++iz)
    for (int iy = 0; iy < NY; ++iy)
    for (int ix = 0; ix < NX; ++ix) {
      p[iz + 1][iy + 1][ix + xPad] =
          r[iz][iy][ix] + beta * p[iz + 1][iy + 1][ix + xPad];
    }

    return rrNew;
  }
  */

  /// Update `r -= a * Ax` and return `sum(r^2)`.
  static inline Real subAndSumSqr(Block &__restrict__ r_,
                                  const Block &__restrict__ Ax_, Real a) {
    // The block structure is not important here, we can treat it as a
    // contiguous array. However, we group into groups of length 16, to help
    // with ILP and vectorization.
    constexpr int MX = 16;
    constexpr int MY = NX * NY * NZ / MX;
    using SquashedBlock = Real[MY][MX];
    static_assert(NX * NY % MX == 0 && sizeof(Block) == sizeof(SquashedBlock));
    SquashedBlock &__restrict__ r = (SquashedBlock &)r_;
    SquashedBlock &__restrict__ Ax = (SquashedBlock &)Ax_;

    // This kernel reaches neither the compute nor the memory bound.
    // The problem could be high latency of FMA instructions.
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
      // if ((uintptr_t)ptr % align != offset)
      //   throw std::runtime_error("wrong alignment");
      assert((uintptr_t)ptr % align == offset);

      // Works with gcc, clang and icc.
      return (T *)__builtin_assume_aligned(ptr, align, offset);
    } else {
      return ptr; // No alignment assumptions for long double.
    }
  }

  Real kernelDiffusionGetZInner(PaddedBlock &p_, const Real *pW_,
                                const Real *pE_, Block &__restrict__ Ax_,
                                Block &__restrict__ r_,
                                Block &__restrict__ block_, const Real sqrNorm0,
                                const Real rr, const Real coefficient) {
    PaddedBlock &p = *assumeAligned(&p_, 64, 64 - xPad * sizeof(Real));
    const PaddedBlock &pW =
        *(PaddedBlock *)pW_; // Aligned to 64B + 24 (for doubles).
    const PaddedBlock &pE =
        *(PaddedBlock *)pE_; // Aligned to 64B + 40 (for doubles).
    Block &__restrict__ Ax = *assumeAligned(&Ax_, 64);
    Block &__restrict__ r = *assumeAligned(&r_, 64);
    Block &__restrict__ block = *assumeAligned(&block_, kBlockAlignment);

    // Broadwell: 6.0-6.6 FLOP/cycle, depending probably on array alignments.
    Real a2Partial[NX] = {};
    for (int iz = 0; iz < NZ; ++iz)
      for (int iy = 0; iy < NY; ++iy) {
        // On Broadwell and earlier it might be beneficial to turn some of these
        // a+b additions into FMAs of form 1*a+b, because those CPUs can do 2
        // FMAs/cycle and only 1 ADD/cycle. However, it wouldn't be simple to
        // convience the compiler to do so, and it wouldn't matter from Skylake
        // on. https://www.agner.org/optimize/blog/read.php?i=415

        Real tmpAx[NX];
        for (int ix = 0; ix < NX; ++ix) {
          tmpAx[ix] = pW[iz + 1][iy + 1][ix + xPad] +
                      pE[iz + 1][iy + 1][ix + xPad] +
                      coefficient * p[iz + 1][iy + 1][ix + xPad];
        }

        // This kernel is memory bound. The compiler should figure out that some
        // loads can be reused between consecutive iy.

        // Merging the following two loops (i.e. to ensure symmetry preservation
        // when there is no -ffast-math) kills vectorization in gcc 11.
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

    // Interleaving this kernel with the next one seems to improve the
    // maximum performance by 5-10% (after fine-tuning MX in the subAndSumSqr
    // part), but it increases the variance a lot so it is not clear whether it
    // is faster on average. For now, keeping it separate.
    for (int iz = 0; iz < NZ; ++iz)
      for (int iy = 0; iy < NY; ++iy)
        for (int ix = 0; ix < NX; ++ix)
          block[iz][iy][ix] += a * p[iz + 1][iy + 1][ix + xPad];

    // Kernel: 2 reads + 1 write + 4 FLOPs/cycle -> should be memory bound.
    // Broadwell: 9.2 FLOP/cycle, 37+18.5 B/cycle -> latency bound?
    // r -= a * Ax, sqrSum = sum(r^2)
    const Real sqrSum = subAndSumSqr(r, Ax, a);

    const Real beta = sqrSum / (rr + kDivEpsilon);
    const Real sqrNorm = (Real)1 / (N * N) * sqrSum;

    if (sqrNorm < kSqrNormRelCriterion * sqrNorm0 ||
        sqrNorm < kSqrNormAbsCriterion)
      return -1.0;

    // Kernel: 2 reads + 1 write + 2 FLOPs per cell -> limit is L1 cache.
    // Broadwell: 6.5 FLOP/cycle, 52+26 B/cycle
    for (int iz = 0; iz < NZ; ++iz)
      for (int iy = 0; iy < NY; ++iy)
        for (int ix = 0; ix < NX; ++ix) {
          p[iz + 1][iy + 1][ix + xPad] =
              r[iz][iy][ix] + beta * p[iz + 1][iy + 1][ix + xPad];
        }

    const Real rrNew = sqrSum;
    return rrNew;
  }

  void getZImplParallel(const std::vector<cubism::BlockInfo> &vInfo,
                        const Real nu, const Real dt) {
    const size_t Nblocks = vInfo.size();

    // We could enable this, we don't really care about denormals.
    // _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    // A struct to enforce relative alignment between matrices. The relative
    // alignment of Ax and r MUST NOT be a multiple of 4KB due to cache bank
    // conflicts. See "Haswell and Broadwell pipeline", section
    // "Cache and memory access" here:
    // https://www.agner.org/optimize/microarchitecture.pdf
    struct Tmp {
      // It seems like some offsets with respect to the page boundary of 4KB are
      // faster than the others. (This is accomplished by adding an offset here
      // and using alignas(4096) below). However, this is likely CPU-dependent,
      // so we don't hardcode such fine-tunings here.
      // char offset[0xec0];
      Block r;
      // Ensure p[0+1][0+1][0+xPad] is 64B-aligned for AVX-512 to work.
      char padding1[64 - xPad * sizeof(Real)];
      PaddedBlock p;
      char padding2[xPad * sizeof(Real)];
      Block Ax;
    };
    alignas(64) Tmp tmp{}; // See the kernels cpp file for required alignments.
    Block &r = tmp.r;
    Block &Ax = tmp.Ax;
    PaddedBlock &p = tmp.p;

#pragma omp for
    for (size_t i = 0; i < Nblocks; ++i) {
      static_assert(sizeof(ScalarBlock) == sizeof(Block));
      assert((uintptr_t)vInfo[i].ptrBlock % kBlockAlignment == 0);
      Block &block = *(Block *)__builtin_assume_aligned(vInfo[i].ptrBlock,
                                                        kBlockAlignment);

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
        // rr = kernelDiffusionGetZInnerReference(p,Ax, r, block, sqrNorm0, rr);
        rr = kernelDiffusionGetZInner(p, pW, pE, Ax, r, block, sqrNorm0, rr,
                                      coefficient);
        if (rr <= 0)
          break;
      }
    }
  }

  } // namespace diffusion_kernels
} // namespace cubismup3d

CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

namespace EllipsoidObstacle {
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
static constexpr int IMAX = 2 * std::numeric_limits<Real>::max_exponent;

static Real distPointEllipseSpecial(const Real e[2], const Real y[2],
                                    Real x[2]) {
  if (y[1] > 0) {
    if (y[0] > 0) {
      // Bisect to compute the root of F(t) for t >= -e1*e1.
      const Real esqr[2] = {e[0] * e[0], e[1] * e[1]};
      const Real ey[2] = {e[0] * y[0], e[1] * y[1]};
      Real t0 = -esqr[1] + ey[1];
      Real t1 = -esqr[1] + std::sqrt(ey[0] * ey[0] + ey[1] * ey[1]);
      Real t = t0;
      for (int i = 0; i < IMAX; ++i) {
        t = 0.5 * (t0 + t1);
        if (std::fabs(t - t0) < EPS || std::fabs(t - t1) < EPS)
          break;
        const Real r[2] = {ey[0] / (t + esqr[0]), ey[1] / (t + esqr[1])};
        const Real f = r[0] * r[0] + r[1] * r[1] - 1;
        if (f > 0)
          t0 = t;
        else if (f < 0)
          t1 = t;
        else
          break;
      }
      x[0] = esqr[0] * y[0] / (t + esqr[0]);
      x[1] = esqr[1] * y[1] / (t + esqr[1]);
      const Real d[2] = {x[0] - y[0], x[1] - y[1]};
      return std::sqrt(d[0] * d[0] + d[1] * d[1]);
    } else { // y0 == 0
      x[0] = 0;
      x[1] = e[1];
      return std::fabs(y[1] - e[1]);
    }
  } else { // y1 == 0
    const Real denom0 = e[0] * e[0] - e[1] * e[1];
    const Real e0y0 = e[0] * y[0];
    if (e0y0 < denom0) {
      // y0 is inside the subinterval.
      const Real x0de0 = e0y0 / denom0;
      const Real x0de0sqr = x0de0 * x0de0;
      x[0] = e[0] * x0de0;
      x[1] = e[1] * std::sqrt(std::fabs((Real)1 - x0de0sqr));
      const Real d0 = x[0] - y[0];
      return std::sqrt(d0 * d0 + x[1] * x[1]);
    } else {
      // y0 is outside the subinterval.  The closest ellipse point has
      // x1 == 0 and is on the domain-boundary interval (x0/e0)^2 = 1.
      x[0] = e[0];
      x[1] = 0;
      return std::fabs(y[0] - e[0]);
    }
  }
}

//----------------------------------------------------------------------------
// The ellipse is (x0/e0)^2 + (x1/e1)^2 = 1.  The query point is (y0,y1).
// The function returns the distance from the query point to the ellipse.
// It also computes the ellipse point (x0,x1) that is closest to (y0,y1).
//----------------------------------------------------------------------------
/*
static Real distancePointEllipse (const Real e[2], const Real y[2], Real x[2])
{
  // Determine reflections for y to the first quadrant.
  const bool reflect[2] = {y[0] < 0, y[1] < 0};
  // Determine the axis order for decreasing extents.
  const int permute[2] = {e[0]<e[1] ? 1 : 0, e[0]<e[1] ? 0 : 1};
  int invpermute[2];
  for (int i = 0; i < 2; ++i) invpermute[permute[i]] = i;
  Real locE[2], locY[2];
  for (int i = 0; i < 2; ++i) {
    const int j = permute[i];
    locE[i] = e[j];
    locY[i] = y[j];
    if (reflect[j]) locY[i] = -locY[i];
  }

  Real locX[2];
  const Real distance = distPointEllipseSpecial(locE, locY, locX);

  // Restore the axis order and reflections.
  for (int i = 0; i < 2; ++i) {
    const int j = invpermute[i];
    if (reflect[j]) locX[j] = -locX[j];
    x[i] = locX[j];
  }
  return distance;
}
*/
// code from http://www.geometrictools.com/
//----------------------------------------------------------------------------
// The ellipsoid is (x0/e0)^2 + (x1/e1)^2 + (x2/e2)^2 = 1 with e0 >= e1 >= e2.
// The query point is (y0,y1,y2) with y0 >= 0, y1 >= 0, and y2 >= 0.  The
// function returns the distance from the query point to the ellipsoid.  It
// also computes the ellipsoid point (x0,x1,x2) in the first octant that is
// closest to (y0,y1,y2).
//----------------------------------------------------------------------------

static Real distPointEllipsoidSpecial(const Real e[3], const Real y[3],
                                      Real x[3]) {
  if (y[2] > 0) {
    if (y[1] > 0) {
      if (y[0] > 0) {
        // Bisect to compute the root of F(t) for t >= -e2*e2.
        const Real esq[3] = {e[0] * e[0], e[1] * e[1], e[2] * e[2]};
        const Real ey[3] = {e[0] * y[0], e[1] * y[1], e[2] * y[2]};
        Real t0 = -esq[2] + ey[2];
        Real t1 =
            -esq[2] + std::sqrt(ey[0] * ey[0] + ey[1] * ey[1] + ey[2] * ey[2]);
        Real t = t0;
        for (int i = 0; i < IMAX; ++i) {
          t = 0.5 * (t0 + t1);
          if (std::fabs(t - t0) < EPS || std::fabs(t - t1) < EPS)
            break;
          const Real r[3] = {ey[0] / (t + esq[0]), ey[1] / (t + esq[1]),
                             ey[2] / (t + esq[2])};
          const Real f = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] - 1;
          if (f > 0)
            t0 = t;
          else if (f < 0)
            t1 = t;
          else
            break;
        }
        x[0] = esq[0] * y[0] / (t + esq[0]);
        x[1] = esq[1] * y[1] / (t + esq[1]);
        x[2] = esq[2] * y[2] / (t + esq[2]);
        const Real d[3] = {x[0] - y[0], x[1] - y[1], x[2] - y[2]};
        return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      } else { // y0 == 0
        x[0] = 0;
        const Real etmp[2] = {e[1], e[2]};
        const Real ytmp[2] = {y[1], y[2]};
        Real xtmp[2];
        const Real distance = distPointEllipseSpecial(etmp, ytmp, xtmp);
        x[1] = xtmp[0];
        x[2] = xtmp[1];
        return distance;
      }
    } else { // y1 == 0
      x[1] = 0;
      if (y[0] > 0) {
        const Real etmp[2] = {e[0], e[2]};
        const Real ytmp[2] = {y[0], y[2]};
        Real xtmp[2];
        const Real distance = distPointEllipseSpecial(etmp, ytmp, xtmp);
        x[0] = xtmp[0];
        x[2] = xtmp[1];
        return distance;
      } else { // y0 == 0
        x[0] = 0;
        x[2] = e[2];
        return std::fabs(y[2] - e[2]);
      }
    }
  } else { // y2 == 0
    const Real denom[2] = {e[0] * e[0] - e[2] * e[2],
                           e[1] * e[1] - e[2] * e[2]};
    const Real ey[2] = {e[0] * y[0], e[1] * y[1]};
    if (ey[0] < denom[0] && ey[1] < denom[1]) {
      // (y0,y1) is inside the axis-aligned bounding rectangle of the
      // subellipse.  This intermediate test is designed to guard
      // against the division by zero when e0 == e2 or e1 == e2.
      const Real xde[2] = {ey[0] / denom[0], ey[1] / denom[1]};
      const Real xdesqr[2] = {xde[0] * xde[0], xde[1] * xde[1]};
      const Real discr = 1 - xdesqr[0] - xdesqr[1];
      if (discr > 0) {
        // (y0,y1) is inside the subellipse.  The closest ellipsoid
        // point has x2 > 0.
        x[0] = e[0] * xde[0];
        x[1] = e[1] * xde[1];
        x[2] = e[2] * std::sqrt(discr);
        const Real d[2] = {x[0] - y[0], x[1] - y[1]};
        return std::sqrt(d[0] * d[0] + d[1] * d[1] + x[2] * x[2]);
      } else {
        // (y0,y1) is outside the subellipse.  The closest ellipsoid
        // point has x2 == 0 and is on the domain-boundary ellipse
        // (x0/e0)^2 + (x1/e1)^2 = 1.
        x[2] = 0;
        return distPointEllipseSpecial(e, y, x);
      }
    } else {
      // (y0,y1) is outside the subellipse.  The closest ellipsoid
      // point has x2 == 0 and is on the domain-boundary ellipse
      // (x0/e0)^2 + (x1/e1)^2 = 1.
      x[2] = 0;
      return distPointEllipseSpecial(e, y, x);
    }
  }
}

//----------------------------------------------------------------------------
// The ellipsoid is (x0/e0)^2 + (x1/e1)^2 + (x2/e2)^2 = 1.  The query point is
// (y0,y1,y2).  The function returns the distance from the query point to the
// ellipsoid.   It also computes the ellipsoid point (x0,x1,x2) that is
// closest to (y0,y1,y2).
//----------------------------------------------------------------------------
static Real distancePointEllipsoid(const Real e[3], const Real y[3],
                                   Real x[3]) {
  // Determine reflections for y to the first octant.
  const bool reflect[3] = {y[0] < 0, y[1] < 0, y[2] < 0};

  // Determine the axis order for decreasing extents.
  int permute[3];
  if (e[0] < e[1]) {
    if (e[2] < e[0]) {
      permute[0] = 1;
      permute[1] = 0;
      permute[2] = 2;
    } else if (e[2] < e[1]) {
      permute[0] = 1;
      permute[1] = 2;
      permute[2] = 0;
    } else {
      permute[0] = 2;
      permute[1] = 1;
      permute[2] = 0;
    }
  } else {
    if (e[2] < e[1]) {
      permute[0] = 0;
      permute[1] = 1;
      permute[2] = 2;
    } else if (e[2] < e[0]) {
      permute[0] = 0;
      permute[1] = 2;
      permute[2] = 1;
    } else {
      permute[0] = 2;
      permute[1] = 0;
      permute[2] = 1;
    }
  }

  int invpermute[3];
  for (int i = 0; i < 3; ++i)
    invpermute[permute[i]] = i;

  Real locE[3], locY[3];
  for (int i = 0; i < 3; ++i) {
    const int j = permute[i];
    locE[i] = e[j];
    locY[i] = y[j];
    if (reflect[j])
      locY[i] = -locY[i];
  }

  Real locX[3];
  const Real distance = distPointEllipsoidSpecial(locE, locY, locX);

  // Restore the axis order and reflections.
  for (int i = 0; i < 3; ++i) {
    const int j = invpermute[i];
    if (reflect[j])
      locX[j] = -locX[j];
    x[i] = locX[j];
  }

  return distance;
}

struct FillBlocks : FillBlocksBase<FillBlocks> {
  const Real e0, e1, e2, h, safety = (2 + SURFDH) * h;
  const Real maxAxis = std::max({e0, e1, e2});
  const Real position[3], quaternion[4];
  const Real box[3][2] = {{(Real)position[0] - 2 * (maxAxis + safety),
                           (Real)position[0] + 2 * (maxAxis + safety)},
                          {(Real)position[1] - 2 * (maxAxis + safety),
                           (Real)position[1] + 2 * (maxAxis + safety)},
                          {(Real)position[2] - 2 * (maxAxis + safety),
                           (Real)position[2] + 2 * (maxAxis + safety)}};
  const Real w = quaternion[0], x = quaternion[1], y = quaternion[2],
             z = quaternion[3];
  const Real Rmatrix[3][3] = {
      {1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)},
      {2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)},
      {2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)}};

  FillBlocks(const Real _e0, const Real _e1, const Real _e2, const Real _h,
             const Real p[3], const Real q[4])
      : e0(_e0), e1(_e1), e2(_e2),
        h(_h), position{p[0], p[1], p[2]}, quaternion{q[0], q[1], q[2], q[3]} {}

  inline bool isTouching(const BlockInfo &info, const ScalarBlock &b) const {
    Real MINP[3], MAXP[3];
    info.pos(MINP, 0, 0, 0);
    info.pos(MAXP, ScalarBlock::sizeX - 1, ScalarBlock::sizeY - 1,
             ScalarBlock::sizeZ - 1);
    const Real intersect[3][2] = {
        {std::max(MINP[0], box[0][0]), std::min(MAXP[0], box[0][1])},
        {std::max(MINP[1], box[1][0]), std::min(MAXP[1], box[1][1])},
        {std::max(MINP[2], box[2][0]), std::min(MAXP[2], box[2][1])}};
    return intersect[0][1] - intersect[0][0] > 0 &&
           intersect[1][1] - intersect[1][0] > 0 &&
           intersect[2][1] - intersect[2][0] > 0;
  }

  inline Real signedDistance(const Real xo, const Real yo,
                             const Real zo) const {
    const Real p[3] = {xo - (Real)position[0], yo - (Real)position[1],
                       zo - (Real)position[2]};
    // rotate
    const Real t[3] = {
        Rmatrix[0][0] * p[0] + Rmatrix[0][1] * p[1] + Rmatrix[0][2] * p[2],
        Rmatrix[1][0] * p[0] + Rmatrix[1][1] * p[1] + Rmatrix[1][2] * p[2],
        Rmatrix[2][0] * p[0] + Rmatrix[2][1] * p[1] + Rmatrix[2][2] * p[2]};
    // find distance
    const Real e[3] = {e0, e1, e2};
    Real xs[3];
    const Real dist = distancePointEllipsoid(e, t, xs);
    const Real Dcentre = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
    const Real Dsurf = xs[0] * xs[0] + xs[1] * xs[1] + xs[2] * xs[2];
    const int sign = Dcentre > Dsurf ? 1 : -1;
    return dist * sign;
  }
};
} // namespace EllipsoidObstacle

Ellipsoid::Ellipsoid(SimulationData &s, ArgumentParser &p)
    : Obstacle(s, p), radius(0.5 * length) {
  e0 = p("-aspectRatioX").asDouble(1) * radius;
  e1 = p("-aspectRatioY").asDouble(1) * radius;
  e2 = p("-aspectRatioZ").asDouble(1) * radius;
  accel_decel = p("-accel").asBool(false);
  if (accel_decel) {
    if (not bForcedInSimFrame[0]) {
      printf("Warning: sphere was not set to be forced in x-dir, yet the "
             "accel_decel pattern is active.\n");
    }
    umax = p("-xvel").asDouble(0.0);
    tmax = p("-T").asDouble(1.);
  }
}

void Ellipsoid::create() {
  const Real h = sim.hmin;
  const EllipsoidObstacle::FillBlocks K(e0, e1, e2, h, position, quaternion);

  create_base<EllipsoidObstacle::FillBlocks>(K);
}

void Ellipsoid::finalize() {
  // this method allows any computation that requires the char function
  // to be computed. E.g. compute the effective center of mass or removing
  // momenta from udef
}

void Ellipsoid::computeVelocities() {
  Obstacle::computeVelocities();

  if (accel_decel) {
    if (sim.time < tmax)
      transVel[0] = umax * sim.time / tmax;
    else if (sim.time < 2 * tmax)
      transVel[0] = umax * (2 * tmax - sim.time) / tmax;
    else
      transVel[0] = 0;
  }
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

void ExternalForcing::operator()(const double dt) {
  sim.startProfiler("Forcing Kernel");
  const int dir = sim.BCy_flag == wall ? 1 : 2;
  const Real H = sim.extents[dir];
  const Real gradPdt = 8 * sim.uMax_forced * sim.nu / H / H * dt;
  const int DIRECTION = 0;
  const std::vector<BlockInfo> &velInfo = sim.velInfo();
#pragma omp parallel for
  for (size_t i = 0; i < velInfo.size(); i++) {
    VectorBlock &v = *(VectorBlock *)velInfo[i].ptrBlock;
    for (int z = 0; z < VectorBlock::sizeZ; ++z)
      for (int y = 0; y < VectorBlock::sizeY; ++y)
        for (int x = 0; x < VectorBlock::sizeX; ++x) {
          v(x, y, z).u[DIRECTION] += gradPdt;
        }
  }

  sim.stopProfiler();
}

CubismUP_3D_NAMESPACE_END
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

namespace ExternalObstacleObstacle {

inline bool exists(const std::string &name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

struct FillBlocks {
  ExternalObstacle *obstacle;
  std::array<std::array<Real, 2>, 3> box;

  FillBlocks(ExternalObstacle *_obstacle) {
    obstacle = _obstacle;

    // Compute maximal extents
    Real MIN = std::numeric_limits<Real>::min();
    Real MAX = std::numeric_limits<Real>::max();
    Vector3<Real> min = {MAX, MAX, MAX};
    Vector3<Real> max = {MIN, MIN, MIN};
    for (const auto &point : obstacle->x_)
      for (size_t i = 0; i < 3; i++) {
        if (point[i] < min[i])
          min[i] = point[i];
        if (point[i] > max[i])
          max[i] = point[i];
      }
    box = {{{min[0], max[0]}, {min[1], max[1]}, {min[2], max[2]}}};
  }

  inline bool isTouching(const BlockInfo &info, const ScalarBlock &b) const {
    Real MINP[3], MAXP[3];
    info.pos(MINP, 0, 0, 0);
    info.pos(MAXP, ScalarBlock::sizeX - 1, ScalarBlock::sizeY - 1,
             ScalarBlock::sizeZ - 1);
    const Real intersect[3][2] = {
        {std::max(MINP[0], box[0][0]), std::min(MAXP[0], box[0][1])},
        {std::max(MINP[1], box[1][0]), std::min(MAXP[1], box[1][1])},
        {std::max(MINP[2], box[2][0]), std::min(MAXP[2], box[2][1])}};
    return intersect[0][1] - intersect[0][0] > 0 &&
           intersect[1][1] - intersect[1][0] > 0 &&
           intersect[2][1] - intersect[2][0] > 0;
  }

  inline Real signedDistance(const Real px, const Real py, const Real pz,
                             const int id) const {
    const std::vector<int> &myTriangles = obstacle->BlocksToTriangles[id];

    if (myTriangles.size() == 0)
      return -1; // very far

    const auto &x_ = obstacle->x_;
    const auto &tri_ = obstacle->tri_;
    const auto &length = obstacle->length;

    // Find the closest triangles and the distance to them.
    Vector3<Real> p = {px, py, pz};
    Vector3<Real> dummy;
    Real minSqrDist = 1e10;
    std::vector<int> closest;
    for (size_t index = 0; index < myTriangles.size(); ++index) {
      const int i = myTriangles[index];
      Vector3<Vector3<Real>> t{
          x_[tri_[i][0]],
          x_[tri_[i][1]],
          x_[tri_[i][2]],
      };
      const Real sqrDist = pointTriangleSqrDistance(t[0], t[1], t[2], p, dummy);
      if (std::fabs(sqrDist - minSqrDist) < length * 0.01 * length * 0.01) {
        if (sqrDist < minSqrDist)
          minSqrDist = sqrDist;
        closest.push_back(i);
      } else if (sqrDist < minSqrDist) {
        minSqrDist = sqrDist;
        closest.clear();
        closest.push_back(i);
      }
    }

    const Real dist = std::sqrt(minSqrDist);

    bool trust = true;
    Real side = -1;
    for (size_t c = 0; c < closest.size(); c++) {
      const int i = closest[c];
      Vector3<Vector3<Real>> t{
          x_[tri_[i][0]],
          x_[tri_[i][1]],
          x_[tri_[i][2]],
      };
      Vector3<Real> n{};
      n = cross(t[1] - t[0], t[2] - t[0]);
      const Real delta0 = n[0] * t[0][0] + n[1] * t[0][1] + n[2] * t[0][2];
      const Real delta1 = n[0] * t[1][0] + n[1] * t[1][1] + n[2] * t[1][2];
      const Real delta2 = n[0] * t[2][0] + n[1] * t[2][1] + n[2] * t[2][2];
      const Real delta_max = std::max({delta0, delta1, delta2});
      const Real delta_min = std::min({delta0, delta1, delta2});
      const Real delta =
          std::fabs(delta_max) > std::fabs(delta_min) ? delta_max : delta_min;
      const Real dot_prod = n[0] * p[0] + n[1] * p[1] + n[2] * p[2];
      const Real newside = -(dot_prod - delta);
      if (c > 0 && newside * side < 0)
        trust = false;
      side = newside;
      if (!trust)
        break;
    }

    if (trust) {
      return std::copysign(dist, side);
    } else {
      return isInner(p) ? dist : -dist;
    }
  }

  inline bool isInner(const Vector3<Real> &p) const {
    const auto &x_ = obstacle->x_;
    const auto &tri_ = obstacle->tri_;
    const auto &randomNormals = obstacle->randomNormals;

    for (const auto &randomNormal : randomNormals) {
      size_t numIntersections = 0;
      bool validRay = true;
#if 1
      for (const auto &tri : tri_) {
        Vector3<Vector3<Real>> t{x_[tri[0]], x_[tri[1]], x_[tri[2]]};

        // Send ray. Return 0 for miss, 1 for hit, -1 for parallel triangle, -2
        // for line intersection
        Vector3<Real> intersectionPoint{};
        const int intersection =
            rayIntersectsTriangle(p, randomNormal, t, intersectionPoint);

        if (intersection > 0) {
          numIntersections += intersection;
        } else if (intersection == -3) // check whether ray is invalid (corner
                                       // or edge intersection)
        {
          validRay = false;
          break;
        }
      }
#else
      const auto &position = obstacle->position;
      const auto &IJKToTriangles = obstacle->IJKToTriangles;
      const int n = obstacle->nIJK;
      const Real h = obstacle->hIJK;
      const Real extent = 0.5 * n * h;
      const Real h2 = 0.5 * h;

      Vector3<Real> Ray = {p[0] - position[0], p[1] - position[1],
                           p[2] - position[2]};
      std::vector<bool> block_visited(n * n * n, false);
      int i = -1;
      int j = -1;
      int k = -1;
      bool done = false;
      while (done == false) {
        Ray = Ray + (0.10 * h) * randomNormal;
        i = round((extent - h2 + Ray[0]) / h);
        j = round((extent - h2 + Ray[1]) / h);
        k = round((extent - h2 + Ray[2]) / h);
        i = std::min(i, n - 1);
        j = std::min(j, n - 1);
        k = std::min(k, n - 1);
        i = std::max(i, 0);
        j = std::max(j, 0);
        k = std::max(k, 0);
        const int b = i + j * n + k * n * n;
        if (!block_visited[b]) {
          block_visited[b] = true;
          Vector3<Real> centerV;
          centerV[0] = -extent + h2 + i * h + position[0];
          centerV[1] = -extent + h2 + j * h + position[1];
          centerV[2] = -extent + h2 + k * h + position[2];
          for (auto &tr : IJKToTriangles[b]) {
            Vector3<Vector3<Real>> t{
                x_[tri_[tr][0]],
                x_[tri_[tr][1]],
                x_[tri_[tr][2]],
            };
            Vector3<Real> intersectionPoint{};
            const int intersection =
                rayIntersectsTriangle(p, randomNormal, t, intersectionPoint);
            if (intersection > 0) {
              if ((std::fabs(intersectionPoint[0] - centerV[0]) <= h2) &&
                  (std::fabs(intersectionPoint[1] - centerV[1]) <= h2) &&
                  (std::fabs(intersectionPoint[2] - centerV[2]) <= h2)) {
                numIntersections += intersection;
              }
            } else if (intersection == -3) {
              validRay = false;
              done = true;
              break;
            }
          }
        }
        done = (std::fabs(Ray[0]) > extent || std::fabs(Ray[1]) > extent ||
                std::fabs(Ray[2]) > extent);
      }

#endif

      if (validRay)
        return (numIntersections % 2 == 1);
    }

    std::cout << "Point " << p[0] << " " << p[1] << " " << p[2]
              << " has no valid rays!" << std::endl;
    for (size_t i = 0; i < randomNormals.size(); i++) {
      std::cout << p[0] + randomNormals[i][0] << " "
                << p[1] + randomNormals[i][1] << " "
                << p[2] + randomNormals[i][2] << std::endl;
    }
    abort();
  }

  using CHIMAT =
      Real[ScalarBlock::sizeZ][ScalarBlock::sizeY][ScalarBlock::sizeX];
  void operator()(const cubism::BlockInfo &info, ObstacleBlock *const o) const {
    for (int iz = -1; iz < ScalarBlock::sizeZ + 1; ++iz)
      for (int iy = -1; iy < ScalarBlock::sizeY + 1; ++iy)
        for (int ix = -1; ix < ScalarBlock::sizeX + 1; ++ix) {
          Real p[3];
          info.pos(p, ix, iy, iz);
          const Real dist = signedDistance(p[0], p[1], p[2], info.blockID);
          o->sdfLab[iz + 1][iy + 1][ix + 1] = dist;
        }
  }
};
} // namespace ExternalObstacleObstacle

ExternalObstacle::ExternalObstacle(SimulationData &s, ArgumentParser &p)
    : Obstacle(s, p) {
  path = p("-externalObstaclePath").asString();
  if (ExternalObstacleObstacle::exists(path)) {
    if (sim.rank == 0)
      std::cout << "[ExternalObstacle] Reading mesh from " << path << std::endl;

    // 1.Construct the data object by reading from file and get mesh-style data
    // from the object read
    happly::PLYData plyIn(path);
    std::vector<std::array<Real, 3>> vPos = plyIn.getVertexPositions();
    std::vector<std::vector<int>> fInd = plyIn.getFaceIndices<int>();

    // 2.Compute maximal extent and ExternalObstacle's center of mass
    Real MIN = std::numeric_limits<Real>::min();
    Real MAX = std::numeric_limits<Real>::max();
    Vector3<Real> min = {MAX, MAX, MAX};
    Vector3<Real> max = {MIN, MIN, MIN};
    Vector3<Real> mean = {0, 0, 0};
    for (const auto &point : vPos)
      for (size_t i = 0; i < 3; i++) {
        mean[i] += point[i];
        if (point[i] < min[i])
          min[i] = point[i];
        if (point[i] > max[i])
          max[i] = point[i];
      }
    mean[0] /= vPos.size();
    mean[1] /= vPos.size();
    mean[2] /= vPos.size();
    Vector3<Real> diff = max - min;
    const Real maxSize = std::max({diff[0], diff[1], diff[2]});
    const Real scalingFac = length / maxSize;

    // 3.Initialize vectors of Vector3 required by triangleMeshSDF
    for (const auto &point : vPos) {
      Vector3<Real> pt = {scalingFac * (point[0] - mean[0]),
                          scalingFac * (point[1] - mean[1]),
                          scalingFac * (point[2] - mean[2])};
      x_.push_back(pt);
    }
    for (const auto &indx : fInd) {
      Vector3<int> id = {indx[0], indx[1], indx[2]};
      tri_.push_back(id);
    }

    if (sim.rank == 0) {
      std::cout << "[ExternalObstacle] Largest extent = " << maxSize
                << ", target length = " << length
                << ", scaling factor = " << scalingFac << std::endl;
      std::cout << "[ExternalObstacle] Read grid with nPoints = " << vPos.size()
                << ", nTriangles = " << fInd.size() << std::endl;
    }
  } else {
    fprintf(stderr, "[ExternalObstacle] ERROR: Unable to find %s file\n",
            path.c_str());
    fflush(0);
    abort();
  }

  // create 10 random vectors to determine if point is inside obstacle
  gen = std::mt19937();
  normalDistribution = std::normal_distribution<Real>(0.0, 1.0);
  for (size_t i = 0; i < 10; i++) {
    Real normRandomNormal = 0.0;
    Vector3<Real> randomNormal;
    while (std::fabs(normRandomNormal) < 1e-7) {
      randomNormal = {normalDistribution(gen), normalDistribution(gen),
                      normalDistribution(gen)};
      normRandomNormal =
          std::sqrt(dot(randomNormal, randomNormal)); // norm of the vector
    }
    randomNormal = (1 / normRandomNormal) * randomNormal;
    randomNormals.push_back(randomNormal);
  }

  rotate();
}

void ExternalObstacle::create() {
  const std::vector<cubism::BlockInfo> &chiInfo = sim.chiInfo();
  BlocksToTriangles.clear();
  BlocksToTriangles.resize(
      chiInfo.size()); // each block has a set of indices(triangles) that are
                       // inside it
  const int BS =
      std::max({ScalarBlock::sizeX, ScalarBlock::sizeY, ScalarBlock::sizeZ});

#pragma omp parallel for
  for (size_t b = 0; b < chiInfo.size(); b++) {
    Vector3<Real> dummy;
    const cubism::BlockInfo &info = chiInfo[b];
    Real center[3];
    info.pos(center, ScalarBlock::sizeX / 2, ScalarBlock::sizeY / 2,
             ScalarBlock::sizeZ / 2);
    Vector3<Real> centerV;
    centerV[0] = center[0] - 0.5 * info.h;
    centerV[1] = center[1] - 0.5 * info.h;
    centerV[2] = center[2] - 0.5 * info.h;

    for (size_t tr = 0; tr < tri_.size(); tr++) // loop over all triangles
    {
      const int v0 = tri_[tr][0];
      const int v1 = tri_[tr][1];
      const int v2 = tri_[tr][2];
      const Vector3<Real> &t0 = x_[v0];
      const Vector3<Real> &t1 = x_[v1];
      const Vector3<Real> &t2 = x_[v2];
      const Real sqrDist = pointTriangleSqrDistance(t0, t1, t2, centerV, dummy);
      if (sqrDist <
          BS * info.h * BS * info.h * 0.75) // = (info.h * BS/2 * sqrt(3))^2
      {
#pragma omp critical
        { BlocksToTriangles[b].push_back(tr); }
      }
    }
  }
  /*
  nIJK = 32;
  hIJK = 1.1*length/nIJK; // about 3% of the object's length, arbitrary choice
  const Real extent = nIJK*hIJK;
  #if 0 //serial implementation
  if (IJKToTriangles.size() == 0)
  {
    IJKToTriangles.resize(nIJK*nIJK*nIJK);
    #pragma omp parallel for collapse (3)
    for (int k = 0 ; k < nIJK; k++)
    for (int j = 0 ; j < nIJK; j++)
    for (int i = 0 ; i < nIJK; i++)
    {
      const int idx = i + nIJK*(j + nIJK*k);
      Vector3<Real> centerV;
      Vector3<Real> rpt;
      rpt[0]=0;
      rpt[1]=0;
      rpt[2]=0;
      centerV[0] = -0.5*extent + 0.5*hIJK + i*hIJK + position[0];
      centerV[1] = -0.5*extent + 0.5*hIJK + j*hIJK + position[1];
      centerV[2] = -0.5*extent + 0.5*hIJK + k*hIJK + position[2];
      for (size_t tr = 0 ; tr < tri_.size() ; tr++) //loop over all triangles
      {
        const int v0 = tri_[tr][0];
        const int v1 = tri_[tr][1];
        const int v2 = tri_[tr][2];
        const Vector3<Real> & t0 = x_[v0];
        const Vector3<Real> & t1 = x_[v1];
        const Vector3<Real> & t2 = x_[v2];
        pointTriangleSqrDistance(t0,t1,t2,centerV,rpt);
        if((std::fabs(rpt[0]-centerV[0]) < 0.51*hIJK) &&
           (std::fabs(rpt[1]-centerV[1]) < 0.51*hIJK) &&
           (std::fabs(rpt[2]-centerV[2]) < 0.51*hIJK) )
        {
          #pragma omp critical
          {
            IJKToTriangles[idx].push_back(tr);
          }
        }
      }
    }
  }
  #else //MPI implementation
  if (IJKToTriangles.size() == 0)
  {
    int size;
    MPI_Comm_size(sim.comm,&size);
    int rank = sim.rank;

    const int total_load = nIJK*nIJK*nIJK;
    const int my_load = (rank < total_load % size) ? (total_load / size + 1) :
  (total_load / size); int mystart = (total_load / size) * rank; mystart +=
  (rank < (total_load % size)) ? rank : (total_load % size); const int myend =
  mystart + my_load;

    //Create a local vector of vectors
    std::vector<std::vector<int>> local_data(my_load);

    #pragma omp parallel for
    for (int idx = mystart ; idx < myend ; idx++)
    {
      const int k = idx/(nIJK*nIJK);
      const int j = (idx - k*nIJK*nIJK)/nIJK;
      const int i = (idx - k*nIJK*nIJK-j*nIJK)%nIJK;
      Vector3<Real> centerV;
      Vector3<Real> rpt;
      rpt[0]=0;
      rpt[1]=0;
      rpt[2]=0;
      centerV[0] = -0.5*extent + 0.5*hIJK + i*hIJK + position[0];
      centerV[1] = -0.5*extent + 0.5*hIJK + j*hIJK + position[1];
      centerV[2] = -0.5*extent + 0.5*hIJK + k*hIJK + position[2];
      for (size_t tr = 0 ; tr < tri_.size() ; tr++) //loop over all triangles
      {
        const int v0 = tri_[tr][0];
        const int v1 = tri_[tr][1];
        const int v2 = tri_[tr][2];
        const Vector3<Real> & t0 = x_[v0];
        const Vector3<Real> & t1 = x_[v1];
        const Vector3<Real> & t2 = x_[v2];
        pointTriangleSqrDistance(t0,t1,t2,centerV,rpt);
        if((std::fabs(rpt[0]-centerV[0]) < 0.51*hIJK) &&
           (std::fabs(rpt[1]-centerV[1]) < 0.51*hIJK) &&
           (std::fabs(rpt[2]-centerV[2]) < 0.51*hIJK) )
        {
          #pragma omp critical
          {
            local_data[idx-mystart].push_back(tr);
          }
        }
      }
    }

    // Flatten the local vectors for communication
    std::vector<int> send_data;
    for (const auto& vec : local_data)
      send_data.insert(send_data.end(), vec.begin(), vec.end());

    // Communicate the local vectors among ranks
    std::vector<int> recv_counts(size);
    const int send_count = send_data.size();
    MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
  sim.comm);

    std::vector<int> displacements(size);
    int total_size = 0;
    for (int i = 0; i < size; i++)
    {
      displacements[i] = total_size;
      total_size += recv_counts[i];
    }

    // Allocate memory for the received global vectors
    std::vector<int> recv_data(total_size);

    // Communicate the local vectors and receive the global vectors
    MPI_Allgatherv(send_data.data(), send_data.size(), MPI_INT,
  recv_data.data(), recv_counts.data(), displacements.data(), MPI_INT,
  sim.comm);

    std::vector<int> vector_displacements(total_load,0);
    for (int idx = mystart ; idx < myend ; idx++)
    {
      vector_displacements[idx] = local_data[idx-mystart].size();
    }
    MPI_Allreduce(MPI_IN_PLACE, vector_displacements.data(),
  vector_displacements.size(), MPI_INT, MPI_SUM, sim.comm);

    //Reconstruct the global vector of vectors
    size_t current_pos = 0;
    for (int idx = 0; idx < total_load; idx++)
    {
      int count = vector_displacements[idx];
      std::vector<int> vec(recv_data.begin() + current_pos, recv_data.begin() +
  current_pos + count); IJKToTriangles.push_back(vec); current_pos += count;
    }
  }
  #endif
  */
  ExternalObstacleObstacle::FillBlocks K(this);

  create_base<ExternalObstacleObstacle::FillBlocks>(K);
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

Fish::Fish(SimulationData &s, ArgumentParser &p) : Obstacle(s, p) {
  p.unset_strict_mode();
#if 1
  // MPI datatypes (used for load-balancing when creating the fish surface)
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
#endif
}

Fish::~Fish() {
  if (myFish not_eq nullptr)
    delete myFish;
#if 1
  MPI_Type_free(&MPI_BLOCKID);
  MPI_Type_free(&MPI_OBSTACLE);
#endif
}

void Fish::integrateMidline() {
  myFish->integrateLinearMomentum();
  myFish->integrateAngularMomentum(sim.dt);
}

std::vector<VolumeSegment_OBB> Fish::prepare_vSegments() {
  /*
    - VolumeSegment_OBB's volume cannot be zero
    - therefore no VolumeSegment_OBB can be only occupied by extension midline
      points (which have width and height = 0)
    - performance of create seems to decrease if VolumeSegment_OBB are bigger
    - this is the smallest number of VolumeSegment_OBB (Nsegments) and points in
      the midline (Nm) to ensure at least one non ext. point inside all segments
   */
  const int Nsegments = std::ceil((myFish->Nm - 1.) / 8);
  const int Nm = myFish->Nm;
  assert((Nm - 1) % Nsegments == 0);

  std::vector<VolumeSegment_OBB> vSegments(Nsegments);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < Nsegments; ++i) {
    const int nextidx = (i + 1) * (Nm - 1) / Nsegments;
    const int idx = i * (Nm - 1) / Nsegments;
    // find bounding box based on this
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

  const std::vector<cubism::BlockInfo> &chiInfo = sim.chiInfo();
  std::vector<std::vector<VolumeSegment_OBB *>> ret(chiInfo.size());

  // clear deformation velocities
  for (auto &entry : obstacleBlocks) {
    if (entry == nullptr)
      continue;
    delete entry;
    entry = nullptr;
  }
  obstacleBlocks.resize(chiInfo.size(), nullptr);

  //#pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < chiInfo.size(); ++i) {
    const BlockInfo &info = chiInfo[i];
    Real MINP[3], MAXP[3];
    info.pos(MINP, 0, 0, 0);
    info.pos(MAXP, ScalarBlock::sizeX - 1, ScalarBlock::sizeY - 1,
             ScalarBlock::sizeZ - 1);

    bool hasSegments = false;
    for (size_t s = 0; s < vSegments.size(); ++s)
      if (vSegments[s].isIntersectingWithAABB(MINP, MAXP)) {
        VolumeSegment_OBB *const ptr = &vSegments[s];
        ret[info.blockID].push_back(ptr);

        //#pragma omp critical
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

    // allocate new blocks if necessary
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
#if 1 // no load-balancing here
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
#else // load-balancing

  const int tag = 34;
  MPI_Comm comm = sim.chi->getWorldComm();
  const int rank = sim.chi->rank();
  const int size = sim.chi->get_world_size();
  std::vector<std::vector<int>> OtherSegments;

  // Each MPI rank owns two arrays:
  // MyBlockIDs[]: a list of blocks that have at least one segment
  // MySegments[i][j] : a list of integers j=0,... for MyBlockIDs[i]

  // Determine the total load and how much load corresponds to each rank:
  // All N blocks are indexed for 0 to N-1. Each rank computes the range of
  // indices that it must have, in order to have an equal load distribution.
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

  // Now that each rank knows what range it should have, it will determine where
  // to send/receive
  std::vector<std::vector<BlockID>> send_blocks(size);
  std::vector<std::vector<BlockID>> recv_blocks(size);
  for (int r = 0; r < size; r++)
    if (rank != r) {
      { // check if I need to receive blocks
        const int a1 = ideal_index;
        const int a2 = ideal_index + my_load - 1;
        const int b1 = index_start[r];
        const int b2 = index_start[r] + all_b[r] - 1;
        const int c1 = max(a1, b1);
        const int c2 = min(a2, b2);
        if (c2 - c1 + 1 > 0)
          recv_blocks[r].resize(c2 - c1 + 1);
      }
      { // check if I need to send blocks
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

  // Send and receive the information needed to create the obstacle blocks.
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

  // allocate buffers for the actual data that will be sent/received
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

  // Do the blocks I received
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

  // Send and receive data (yes, we receive send_obstacles and send
  // recv_obstacles)
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

// Compute my own blocks (that I did not send), while waiting for communication
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

#endif // load-balancing
}

void Fish::create() {
  // STRATEGY
  // we need some things already
  // - the internal angle at the previous timestep, obtained from integrating
  // the actual def velocities (not the imposed deformation velocies, because
  // they dont have zero ang mom)
  // - the internal angular velocity at previous timestep

  // 1. create midline
  // 2. integrate to find CoM, angular velocity, etc
  // 3. shift midline to CoM frame: zero internal linear momentum and angular
  // momentum

  // 4. split the fish into segments (according to s)
  // 5. rotate the segments to computational frame (comp CoM and angle)
  // 6. for each Block in the domain, find those segments that intersect it
  // 7. for each of those blocks, allocate an ObstacleBlock

  // 8. put the 3D shape on the grid: SDF-P2M for sdf, normal P2M for udef
  // apply_pid_corrections();

  // 1.
  myFish->computeMidline(sim.time, sim.dt);

  // 2. & 3.
  integrateMidline();

  // CAREFUL: this func assumes everything is already centered around CM to
  // start with, which is true (see steps 2. & 3. ...) for rX, rY: they are zero
  // at CM, negative before and + after

  // 4. & 5.
  std::vector<VolumeSegment_OBB> vSegments = prepare_vSegments();

  // 6. & 7.
  const intersect_t segmPerBlock = prepare_segPerBlock(vSegments);

  // 8.
  writeSDFOnBlocks(vSegments);
}

void Fish::saveRestart(FILE *f) {
  assert(f != NULL);
  Obstacle::saveRestart(f);

  fprintf(f, "angvel_internal_x: %20.20e\n",
          (double)myFish->angvel_internal[0]);
  fprintf(f, "angvel_internal_y: %20.20e\n",
          (double)myFish->angvel_internal[1]);
  fprintf(f, "angvel_internal_z: %20.20e\n",
          (double)myFish->angvel_internal[2]);
  fprintf(f, "quaternion_internal_0: %20.20e\n",
          (double)myFish->quaternion_internal[0]);
  fprintf(f, "quaternion_internal_1: %20.20e\n",
          (double)myFish->quaternion_internal[1]);
  fprintf(f, "quaternion_internal_2: %20.20e\n",
          (double)myFish->quaternion_internal[2]);
  fprintf(f, "quaternion_internal_3: %20.20e\n",
          (double)myFish->quaternion_internal[3]);
}

void Fish::loadRestart(FILE *f) {
  assert(f != NULL);
  Obstacle::loadRestart(f);
  bool ret = true;
  double temp;
  ret = ret && 1 == fscanf(f, "angvel_internal_x: %le\n", &temp);
  myFish->angvel_internal[0] = temp;
  ret = ret && 1 == fscanf(f, "angvel_internal_y: %le\n", &temp);
  myFish->angvel_internal[1] = temp;
  ret = ret && 1 == fscanf(f, "angvel_internal_z: %le\n", &temp);
  myFish->angvel_internal[2] = temp;
  ret = ret && 1 == fscanf(f, "quaternion_internal_0: %le\n", &temp);
  myFish->quaternion_internal[0] = temp;
  ret = ret && 1 == fscanf(f, "quaternion_internal_1: %le\n", &temp);
  myFish->quaternion_internal[1] = temp;
  ret = ret && 1 == fscanf(f, "quaternion_internal_2: %le\n", &temp);
  myFish->quaternion_internal[2] = temp;
  ret = ret && 1 == fscanf(f, "quaternion_internal_3: %le\n", &temp);
  myFish->quaternion_internal[3] = temp;
  if ((not ret)) {
    printf("Error reading restart file. Aborting...\n");
    fflush(0);
    abort();
  }
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

using UDEFMAT = Real[CUP_BLOCK_SIZEZ][CUP_BLOCK_SIZEY][CUP_BLOCK_SIZEX][3];
using CHIMAT = Real[CUP_BLOCK_SIZEZ][CUP_BLOCK_SIZEY][CUP_BLOCK_SIZEX];

void FishMidlineData::integrateLinearMomentum() {
  // Compute the center of mass and center of mass velocities of the fish.
  // Then change midline frame of reference to that (new origin is the fish
  // center of mass). Mass is computed as: M = int_{x} int_{y} int_{z} dx dy dz
  // = int_{s} int_{E} |Jacobian| ds dE where E is an elliptic cross-section of
  // the fish. The coordinate transformation that gives us the Jacobian is:
  // x(s,h1,h2) = rS(s) + nor(s)*width(s)*h1 + bin(s)*height(s)*h2
  // where h1,h2 are the coordinates in the ellipse
  // Center of mass (and its velocity) is computed as:
  // C_{x} = 1/M * int_{x} int_{y} int_{z} x dx dy dz = int_{s} int_{E} x
  // |Jacobian| ds dE

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
  // Compute the moments of inertia and angular velocities of the fish.
  // See comments in FishMidlineData::integrateLinearMomentum.
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

  // Invert I
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

  // now we do the rotation
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
    // rotation position and velocity
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
    // rotation normal vector
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
    // rotation binormal vector
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
  safe_distance = (SURFDH + 2) * h; // two points on each side for Towers
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
    // also take absolute value since thats what we need when doing intersection
    // checks later
    normalI[i] = std::fabs(normalI[i]) * invMagI;
    normalJ[i] = std::fabs(normalJ[i]) * invMagJ;
    normalK[i] = std::fabs(normalK[i]) * invMagK;
  }
}

void VolumeSegment_OBB::changeToComputationalFrame(const Real position[3],
                                                   const Real quaternion[4]) {
  // we are in CoM frame and change to comp frame --> first rotate around CoM
  // (which is at (0,0) in CoM frame), then update center
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

  // Find the x,y,z max extents in lab frame ( exploit normal(I,J,K)[:] >=0 )
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
  // Remember: Incoming coordinates are cell centers, not cell faces
  // start and end are two diagonally opposed corners of grid block
  // GN halved the safety here but added it back to w[] in prepare
  const Real AABB_w[3] = {// half block width + safe distance
                          (end[0] - start[0]) / 2 + safe_distance,
                          (end[1] - start[1]) / 2 + safe_distance,
                          (end[2] - start[2]) / 2 + safe_distance};

  const Real AABB_c[3] = {// block center
                          (end[0] + start[0]) / 2, (end[1] + start[1]) / 2,
                          (end[2] + start[2]) / 2};

  const Real AABB_box[3][2] = {{AABB_c[0] - AABB_w[0], AABB_c[0] + AABB_w[0]},
                               {AABB_c[1] - AABB_w[1], AABB_c[1] + AABB_w[1]},
                               {AABB_c[2] - AABB_w[2], AABB_c[2] + AABB_w[2]}};

  assert(AABB_w[0] > 0 && AABB_w[1] > 0 && AABB_w[2] > 0);

  // Now Identify the ones that do not intersect
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
  const Real widthXbox[3] = {
      AABB_w[0] * normalI[0], AABB_w[0] * normalJ[0],
      AABB_w[0] *
          normalK[0]}; // This is x-width of box, expressed in fish frame
  const Real widthYbox[3] = {
      AABB_w[1] * normalI[1], AABB_w[1] * normalJ[1],
      AABB_w[1] *
          normalK[1]}; // This is y-width of box, expressed in fish frame
  const Real widthZbox[3] = {
      AABB_w[2] * normalI[2], AABB_w[2] * normalJ[2],
      AABB_w[2] *
          normalK[2]}; // This is z-height of box, expressed in fish frame

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
  // make p1 origin of a frame of ref
  const Real t[3] = {s[0] - p1[0], s[1] - p1[1], s[2] - p1[2]};
  const Real u[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
  const Real v[3] = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};
  const Real i[3] = {IN[0] - p1[0], IN[1] - p1[1], IN[2] - p1[2]};
  // normal to the plane:
  const Real n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                     u[0] * v[1] - u[1] * v[0]};
  // if normal points inside then this is going to be positive:
  const Real projInner = i[0] * n[0] + i[1] * n[1] + i[2] * n[2];
  // if normal points outside we need to change sign of result:
  const Real signIn = projInner > 0 ? 1 : -1;
  // every point of the plane will have no projection onto n
  // therefore, distance of t from plane is:
  const Real norm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  return signIn * (t[0] * n[0] + t[1] * n[1] + t[2] * n[2]) / norm;
}

void PutFishOnBlocks::constructSurface(
    const Real h, const Real ox, const Real oy, const Real oz,
    ObstacleBlock *const defblock,
    const std::vector<VolumeSegment_OBB *> &vSegments) const {
  // Construct the surface of the fish.
  // By definition, this is where SDF = 0.
  // Here, we know an analytical expression for the fish surface:
  // x(s,theta) = r(s) + width(s)*cos(theta)*nor(s) +
  // height(s)*sin(theta)*bin(s)
  // We loop over the discretized version of this equation and for each point of
  // the surface we find the grid point that is the closest. The SDF is computed
  // for that grid point plus the grid points that are within a distance of 2h
  // of that point. This is done because we need the SDF in a zone of +-2h of
  // the actual surface, so that we can use the mollified Heaviside function
  // from Towers for chi.

  // Pointers to discrete values that describe the surface.
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

  // These are typically 8x8x8 (blocksize=8) matrices that are filled here.
  CHIMAT &__restrict__ CHI = defblock->chi;
  UDEFMAT &__restrict__ UDEF = defblock->udef;

  // This is an (8+2)x(8+2)x(8+2) matrix with the SDF. We compute the sdf for
  // the 8x8x8 block but we also add +-1 grid points of ghost values. That way
  // there is no need for communication later on. We do this because an
  // analytical expression is available and the extra cost for this computation
  // is small.
  auto &__restrict__ SDFLAB = defblock->sdfLab;

  // Origin of the block, displaced by (-h,-h,-h). This is where the first ghost
  // cell is.
  const Real org[3] = {ox - h, oy - h, oz - h};

  const Real invh = 1.0 / h;
  const int BS[3] = {ScalarBlock::sizeX + 2, ScalarBlock::sizeY + 2,
                     ScalarBlock::sizeZ + 2};

  // needed variables to store sensor location
  const Real *const rS = cfish->rS;
  const Real length = cfish->length;

  // save location for tip of head
  Real myP[3] = {rX[0], rY[0], rZ[0]};
  changeToComputationalFrame(myP);
  cfish->sensorLocation[0] = myP[0];
  cfish->sensorLocation[1] = myP[1];
  cfish->sensorLocation[2] = myP[2];

  // Loop over vSegments of this block.
  for (size_t i = 0; i < vSegments.size(); ++i) {
    // Each segment has a range of s for the surface that intersects this block.
    const int firstSegm = std::max(vSegments[i]->s_range.first, 1);
    const int lastSegm = std::min(vSegments[i]->s_range.second, cfish->Nm - 2);

    // Loop over discrete s of surface x(s,theta)
    for (int ss = firstSegm; ss <= lastSegm; ++ss) {
      if (height[ss] <= 0)
        height[ss] = 1e-10;
      if (width[ss] <= 0)
        width[ss] = 1e-10;

      // Here we discretize theta for the given s (for given s, the
      // cross-section is an ellipse) This is done by setting the maximum
      // arclength to h/2. Note that maximum arclength = sin(dtheta) *
      // (major_axis+h) = h/2
      const Real major_axis = std::max(height[ss], width[ss]);
      const Real dtheta_tgt = std::fabs(std::asin(h / (major_axis + h) / 2));
      int Ntheta = std::ceil(2 * M_PI / dtheta_tgt);
      if (Ntheta % 2 == 1)
        Ntheta++; // force Ntheta to be even so that the fish is symmetric
      const Real dtheta = 2 * M_PI / ((Real)Ntheta);

      // theta = 0 at the major axis, this variable takes care of that
      const Real offset = height[ss] > width[ss] ? M_PI / 2 : 0;

      // Loop over discrete theta of surface x(s,theta)
      for (int tt = 0; tt < Ntheta; ++tt) {
        const Real theta = tt * dtheta + offset;
        const Real sinth = std::sin(theta), costh = std::cos(theta);

        // Current surface point (in frame of reference of fish)
        myP[0] = rX[ss] + width[ss] * costh * norX[ss] +
                 height[ss] * sinth * binX[ss];
        myP[1] = rY[ss] + width[ss] * costh * norY[ss] +
                 height[ss] * sinth * binY[ss];
        myP[2] = rZ[ss] + width[ss] * costh * norZ[ss] +
                 height[ss] * sinth * binZ[ss];

        changeToComputationalFrame(myP);

        // save location for side of head; for angle = 0 and angle = pi this is
        // a sensor location
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

        // Find index of nearest grid point to myP
        const int iap[3] = {(int)std::floor((myP[0] - org[0]) * invh),
                            (int)std::floor((myP[1] - org[1]) * invh),
                            (int)std::floor((myP[2] - org[2]) * invh)};

        // Loop over that point and a neighborhood of +-3 points, to compute the
        // SDF near the surface
        const int nei = 3;
        const int ST[3] = {iap[0] - nei, iap[1] - nei, iap[2] - nei};
        const int EN[3] = {iap[0] + nei, iap[1] + nei, iap[2] + nei};

        if (EN[0] <= 0 || ST[0] > BS[0])
          continue; // NearNeigh loop
        if (EN[1] <= 0 || ST[1] > BS[1])
          continue; // does not intersect
        if (EN[2] <= 0 || ST[2] > BS[2])
          continue; // with this block

        // Store the surface point at the next and the previous cross-sections.
        // They will be used to compute the SDF later.
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

        // Deformation velocity of surface point.
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
              // Grid point in the neighborhood near surface
              Real p[3];
              p[0] = ox + h * (sx - 1 + 0.5);
              p[1] = oy + h * (sy - 1 + 0.5);
              p[2] = oz + h * (sz - 1 + 0.5);

              const Real dist0 = eulerDistSq3D(p, myP);
              const Real distP = eulerDistSq3D(p, pP);
              const Real distM = eulerDistSq3D(p, pM);

              // check if this grid point has already found a closer surf-point:
              if (std::fabs(SDFLAB[sz][sy][sx]) <
                  std::min({dist0, distP, distM}))
                continue;

              // if this grid point is > 2h distance of the grid point that is
              // nearest to the surface don't compute the sdf
              if (std::min({dist0, distP, distM}) > 4 * h * h)
                continue;

              changeFromComputationalFrame(p);

              // among the three points myP, pP, pM find the two that are the
              // closest to this grid point
              int close_s = ss, secnd_s = ss + (distP < distM ? 1 : -1);
              Real dist1 = dist0, dist2 = distP < distM ? distP : distM;
              if (distP < dist0 || distM < dist0) {
                dist1 = dist2;
                dist2 = dist0;
                close_s = secnd_s;
                secnd_s = ss;
              }

              // Interpolate the surface deformation velocity to this grid
              // point. W behaves like hat interpolation kernel that is used for
              // internal fish points. Introducing W (used to be W=1) smoothens
              // transition from surface to internal points. In fact, later we
              // plus equal udef*hat of internal points. If hat>0, point should
              // behave like internal point, meaning that fish-section udef
              // rotation should multiply distance from midline instead of
              // entire half-width. Remember that uder will become udef / chi,
              // so W simplifies out.
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
                CHI[sz - 1][sy - 1][sx - 1] =
                    W; // Not chi, just used to interpolate udef!
              }

              // Now we compute the SDF of that point.
              // If that point is close to the tail, we project is onto the
              // plane defined by the tail and then compute the sign of the
              // distance based on the side of the plane the point lies on.

              // Else, we model the span between two ellipses (cross-sections)
              // as a spherical segment. See also:
              // http://mathworld.wolfram.com/SphericalSegment.html The
              // spherical segment is defined by two circles. To define those
              // cyrcles, we need their centers and a point on them. The points
              // myP,pM,pP (we are using two of those three here) are the cyrcle
              // points we need. The centers are found by taking the normal
              // starting from myP/pM/pP to the line defined by the vector R1:
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

              // These are a^2 and b^2 in
              // http://mathworld.wolfram.com/SphericalSegment.html
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

              // This is h in http://mathworld.wolfram.com/SphericalSegment.html
              const Real dSsq =
                  std::pow(center_close[0] - center_second[0], 2) +
                  std::pow(center_close[1] - center_second[1], 2) +
                  std::pow(center_close[2] - center_second[2], 2);

              const Real corr = 2 * std::sqrt(radius_close * radius_second);

              if (close_s == cfish->Nm - 2 ||
                  secnd_s == cfish->Nm - 2) // point is close to tail
              {
                // compute the 5 corners of the pyramid around tail last point
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
              } else if (dSsq >= radius_close + radius_second -
                                     corr) // if ds > delta radius
              {
                // if the two cross-sections are close and have axis that do not
                // differ much, we just use the nearest neighbor to compute the
                // sdf (no need for spherical segment model)
                const Real xMidl[3] = {rX[close_s], rY[close_s], rZ[close_s]};
                const Real grd2ML = eulerDistSq3D(p, xMidl);
                const Real sign = grd2ML > radius_close ? -1 : 1;
                SDFLAB[sz][sy][sx] = sign * dist1;
              } else // here we use the spherical segment model
              {
                const Real Rsq =
                    (radius_close + radius_second - corr +
                     dSsq) // radius of the spere
                    * (radius_close + radius_second + corr + dSsq) / 4 / dSsq;
                const Real maxAx = std::max(radius_close, radius_second);

                // 'submerged' fraction of radius:
                const Real d =
                    std::sqrt((Rsq - maxAx) / dSsq); //(divided by ds)
                // position of the centre of the sphere:
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

  // construct the deformation velocities (P2M with hat function as kernel)
  for (size_t i = 0; i < vSegments.size(); ++i) {
    const int firstSegm = std::max(vSegments[i]->s_range.first, 1);
    const int lastSegm = std::min(vSegments[i]->s_range.second, cfish->Nm - 2);
    for (int ss = firstSegm; ss <= lastSegm; ++ss) {
      // P2M udef of a slice at this s
      const Real myWidth = width[ss], myHeight = height[ss];
      assert(myWidth > 0 && myHeight > 0);
      const int Nh =
          std::floor(myHeight / h); // floor bcz we already did interior
      for (int ih = -Nh + 1; ih < Nh; ++ih) {
        const Real offsetH = ih * h;
        const Real currWidth =
            myWidth * std::sqrt(1 - std::pow(offsetH / myHeight, 2));
        const int Nw =
            std::floor(currWidth / h); // floor bcz we already did interior
        for (int iw = -Nw + 1; iw < Nw; ++iw) {
          const Real offsetW = iw * h;
          Real xp[3] = {rX[ss] + offsetW * norX[ss] + offsetH * binX[ss],
                        rY[ss] + offsetW * norY[ss] + offsetH * binY[ss],
                        rZ[ss] + offsetW * norZ[ss] + offsetH * binZ[ss]};
          changeToComputationalFrame(xp);
          xp[0] = (xp[0] - org[0]) * invh; // how many grid points
          xp[1] = (xp[1] - org[1]) * invh; // from this block origin
          xp[2] = (xp[2] - org[2]) * invh; // is this fishpoint located at?
          const Real ap[3] = {std::floor(xp[0]), std::floor(xp[1]),
                              std::floor(xp[2])};
          const int iap[3] = {(int)ap[0], (int)ap[1], (int)ap[2]};
          if (iap[0] + 2 <= 0 || iap[0] >= BS[0])
            continue; // hatP2M loop
          if (iap[1] + 2 <= 0 || iap[1] >= BS[1])
            continue; // does not intersect
          if (iap[2] + 2 <= 0 || iap[2] >= BS[2])
            continue; // with this block

          Real udef[3] = {vX[ss] + offsetW * vNorX[ss] + offsetH * vBinX[ss],
                          vY[ss] + offsetW * vNorY[ss] + offsetH * vBinY[ss],
                          vZ[ss] + offsetW * vNorZ[ss] + offsetH * vBinZ[ss]};
          changeVelocityToComputationalFrame(udef);
          Real wghts[3][2]; // P2M weights
          for (int c = 0; c < 3; ++c) {
            const Real t[2] = {// we floored, hat between xp and grid point +-1
                               std::fabs(xp[c] - ap[c]),
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
                // set sign for all interior points
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

  // construct the shape (P2M with min(distance) as kernel) onto defblocks
  for (size_t i = 0; i < vSegments.size(); ++i) {
    // iterate over segments contained in the vSegm intersecting this block:
    const int firstSegm = std::max(vSegments[i]->s_range.first, 1);
    const int lastSegm = std::min(vSegments[i]->s_range.second, cfish->Nm - 2);
    for (int ss = firstSegm; ss <= lastSegm; ++ss) {
      assert(height[ss] > 0 && width[ss] > 0);
      // for each segment, we have one point to left and right of midl
      for (int signp = -1; signp <= 1; signp += 2) {
        // create a surface point
        // special treatment of tail (width = 0 --> no ellipse, just line)
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
        // support is two points left, two points right --> Towers Chi will be
        // one point left, one point right, but needs SDF wider
        for (int sy = std::max(0, iap[1] - 1); sy < std::min(iap[1] + 3, BS[1]);
             ++sy)
          for (int sx = std::max(0, iap[0] - 1);
               sx < std::min(iap[0] + 3, BS[0]); ++sx) {
            Real p[3]; // info.pos(p, sx-1, sy-1, 0-1);
            p[0] = ox + h * (sx - 1 + 0.5);
            p[1] = oy + h * (sy - 1 + 0.5);
            p[2] = oz + h * (0 - 1 + 0.5);

            const Real dist0 = eulerDistSq2D(p, myP);

            changeFromComputationalFrame(p);
#ifndef NDEBUG // check that change of ref frame does not affect dist
            const Real p0[3] = {rX[ss] + width[ss] * signp * norX[ss],
                                rY[ss] + width[ss] * signp * norY[ss], 0};
            const Real distC = eulerDistSq2D(p, p0);
            assert(std::fabs(distC - dist0) < 2.2e-16);
#endif
            const Real distP = eulerDistSq2D(p, pP),
                       distM = eulerDistSq2D(p, pM);

            int close_s = ss, secnd_s = ss + (distP < distM ? 1 : -1);
            Real dist1 = dist0, dist2 = distP < distM ? distP : distM;
            if (distP < dist0 || distM < dist0) { // switch nearest surf point
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
            if (dSsq >=
                std::fabs(cnt2ML - nxt2ML)) { // if no abrupt changes in width
                                              // we use nearest neighbour
              const Real xMidl[3] = {rX[close_s], rY[close_s], 0};
              const Real grd2ML = eulerDistSq2D(p, xMidl);
              sign2d = grd2ML > cnt2ML ? -1 : 1;
            } else {
              // else we model the span between ellipses as a spherical segment
              // http://mathworld.wolfram.com/SphericalSegment.html
              const Real corr = 2 * std::sqrt(cnt2ML * nxt2ML);
              const Real Rsq =
                  (cnt2ML + nxt2ML - corr + dSsq) // radius of the sphere
                  * (cnt2ML + nxt2ML + corr + dSsq) / 4 / dSsq;
              const Real maxAx = std::max(cnt2ML, nxt2ML);
              const int idAx1 = cnt2ML > nxt2ML ? close_s : secnd_s;
              const int idAx2 = idAx1 == close_s ? secnd_s : close_s;
              // 'submerged' fraction of radius:
              const Real d = std::sqrt((Rsq - maxAx) / dSsq); // (divided by ds)
              // position of the centre of the sphere:
              const Real xMidl[3] = {rX[idAx1] + (rX[idAx1] - rX[idAx2]) * d,
                                     rY[idAx1] + (rY[idAx1] - rY[idAx2]) * d,
                                     0};
              const Real grd2Core = eulerDistSq2D(p, xMidl);
              sign2d = grd2Core > Rsq ? -1 : 1; // as always, neg outside
            }

            // since naca extends over z axis, loop over all block
            for (int sz = 0; sz < BS[2]; ++sz) {
              const Real pZ = org[2] + h * sz;
              // positive inside negative outside ... as usual
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
                  // not chi yet, just used for interpolating udef:
                  CHI[sz - 1][sy - 1][sx - 1] = 1;
                }
              }
            }
            // Not chi yet, I stored squared distance from analytical boundary
            // distSq is updated only if curr value is smaller than the old one
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

  // construct the deformation velocities (P2M with hat function as kernel)
  for (size_t i = 0; i < vSegments.size(); ++i) {
    const int firstSegm = std::max(vSegments[i]->s_range.first, 1);
    const int lastSegm = std::min(vSegments[i]->s_range.second, cfish->Nm - 2);
    for (int ss = firstSegm; ss <= lastSegm; ++ss) {
      // P2M udef of a slice at this s
      const Real myWidth = cfish->width[ss], myHeight = cfish->height[ss];
      assert(myWidth > 0 && myHeight > 0);
      // here we process also all inner points. Nw to the left and right of midl
      // add xtension here to make sure we have it in each direction:
      const int Nw =
          std::floor(myWidth / h); // floor bcz we already did interior
      for (int iw = -Nw + 1; iw < Nw; ++iw) {
        const Real offsetW = iw * h;
        Real xp[3] = {cfish->rX[ss] + offsetW * cfish->norX[ss],
                      cfish->rY[ss] + offsetW * cfish->norY[ss], 0};
        changeToComputationalFrame(xp);
        xp[0] = (xp[0] - org[0]) * invh; // how many grid points from this block
        xp[1] = (xp[1] - org[1]) * invh; // origin is this fishpoint located at?
        Real udef[3] = {cfish->vX[ss] + offsetW * cfish->vNorX[ss],
                        cfish->vY[ss] + offsetW * cfish->vNorY[ss], 0};
        changeVelocityToComputationalFrame(udef);
        const Real ap[2] = {std::floor(xp[0]), std::floor(xp[1])};
        const int iap[2] = {(int)ap[0], (int)ap[1]};
        Real wghts[2][2]; // P2M weights
        for (int c = 0; c < 2; ++c) {
          const Real t[2] = {// we floored, hat between xp and grid point +-1
                             std::fabs(xp[c] - ap[c]),
                             std::fabs(xp[c] - (ap[c] + 1))};
          wghts[c][0] = 1.0 - t[0];
          wghts[c][1] = 1.0 - t[1];
        }

        for (int idz = 0; idz < BS[2]; ++idz) {
          const Real pZ = org[2] + h * idz;
          // positive inside negative outside ... as usual
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
              // set sign for all interior points:
              if (std::fabs(SDFLAB[idz][idy][idx] + 1) < EPS)
                SDFLAB[idz][idy][idx] = distZsq;
            }
        }
      }
    }
  }
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

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
  // allocate a cubic bspline workspace (k = 4)
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
      /*
      if(s>0.99*L){ // Go linear, otherwise trailing edge is not closed - NACA
      analytical's fault const Real temp = 0.99; const Real y1 = 5*t*
      (a*std::sqrt(temp) +b*temp +c*temp*temp +d*temp*temp*temp +
      e*temp*temp*temp*temp); const Real dydx = (0-y1)/(L-0.99*L); return y1 +
      dydx * (s - 0.99*L); }else{ // NACA analytical return 5*t* (a*std::sqrt(p)
      +b*p +c*p*p +d*p*p*p + e*p*p*p*p);
      }
      */
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

      // Find current segment
      int currentSegW = 1;
      while (sNormalized >= breaksW[currentSegW])
        currentSegW++;
      currentSegW--; // Shift back to the correct segment
      // if(rS[i]==L) currentSegW = nBreaksW-2; Not necessary - excluded by the
      // if conditional

      // Declare pointer for easy access
      const Real *paramsW = coeffsW[currentSegW];
      // Reconstruct cubic
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

      // Find current segment
      int currentSegH = 1;
      while (sNormalized >= breaksH[currentSegH])
        currentSegH++;
      currentSegH--; // Shift back to the correct segment
      // if(rS[i]==L) currentSegH = nBreaksH-2; Not necessary - excluded by the
      // if conditional

      // Declare pointer for easy access
      const Real *paramsH = coeffsH[currentSegH];
      // Reconstruct cubic
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
      if (!mpirank)
        cout << "Building object's height according to 'largefin' profile."
             << endl;
      Real xh[8] = {0, 0, .2 * L, .4 * L, .6 * L, .8 * L, L, L};
      Real yh[8] = {0,        .055 * L, .18 * L,  .2 * L,
                    .064 * L, .002 * L, .325 * L, 0};
      // TODO read second to last number from factory
      integrateBSpline(xh, yh, 8, L, rS, height, nM);
    } else if (heightName.compare("tunaclone") == 0) {
      if (!mpirank)
        cout << "Building object's height according to 'tunaclone' profile."
             << endl;
      Real xh[9] = {0, 0, 0.2 * L, .4 * L, .6 * L, .9 * L, .96 * L, L, L};
      Real yh[9] = {0, .05 * L, .14 * L, .15 * L, .11 * L,
                    0, .1 * L,  .2 * L,  0};
      integrateBSpline(xh, yh, 9, L, rS, height, nM);
    } else if (heightName.compare(0, 4, "naca") == 0) {
      Real t_naca = std::stoi(heightName.substr(5), nullptr, 10) * 0.01;
      if (!mpirank)
        cout << "Building object's height according to naca profile with adim. "
                "thickness param set to "
             << t_naca << " ." << endl;
      naca_width(t_naca, L, rS, height, nM);
    } else if (heightName.compare("danio") == 0) {
      if (!mpirank)
        cout << "Building object's height according to Danio (zebrafish) "
                "profile from Maertens2017 (JFM)"
             << endl;
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
      if (!mpirank)
        cout << "Building object's height according to baseline profile."
             << endl;
      Real xh[8] = {0, 0, .2 * L, .4 * L, .6 * L, .8 * L, L, L};
      Real yh[8] = {0,        .055 * L,  .068 * L, .076 * L,
                    .064 * L, .0072 * L, .11 * L,  0};
      integrateBSpline(xh, yh, 8, L, rS, height, nM);
    }
  }

  {
    if (widthName.compare("fatter") == 0) {
      if (!mpirank)
        cout << "Building object's width according to 'fatter' profile."
             << endl;
      Real xw[6] = {0, 0, L / 3., 2 * L / 3., L, L};
      Real yw[6] = {0, 8.9e-2 * L, 7.0e-2 * L, 3.0e-2 * L, 2.0e-2 * L, 0};
      integrateBSpline(xw, yw, 6, L, rS, width, nM);
    } else if (widthName.compare(0, 4, "naca") == 0) {
      Real t_naca = std::stoi(widthName.substr(5), nullptr, 10) * 0.01;
      if (!mpirank)
        cout << "Building object's width according to naca profile with adim. "
                "thickness param set to "
             << t_naca << " ." << endl;
      naca_width(t_naca, L, rS, width, nM);
    } else if (widthName.compare("danio") == 0) {
      if (!mpirank)
        cout << "Building object's width according to Danio (zebrafish) "
                "profile from Maertens2017 (JFM)"
             << endl;
      danio_width(L, rS, width, nM);
    } else if (widthName.compare("stefan") == 0) {
      if (!mpirank)
        cout << "Building object's width according to Stefan profile" << endl;
      stefan_width(L, rS, width, nM);
    } else if (widthName.compare("larval") == 0) {
      if (!mpirank)
        cout << "Building object's width according to Larval profile" << endl;
      larval_width(L, rS, width, nM);
    } else {
      if (!mpirank)
        cout << "Building object's width according to baseline profile."
             << endl;
      Real xw[6] = {0, 0, L / 3., 2 * L / 3., L, L};
      Real yw[6] = {0, 8.9e-2 * L, 1.7e-2 * L, 1.6e-2 * L, 1.3e-2 * L, 0};
      integrateBSpline(xw, yw, 6, L, rS, width, nM);
    }
  }
#if 0
  if(!mpirank) {
    FILE * heightWidth;
    heightWidth = fopen("widthHeight.dat","w");
    for(int i=0; i<nM; ++i)
      fprintf(heightWidth,"%.8e \t %.8e \t %.8e \n", rS[i], width[i], height[i]);
    fclose(heightWidth);
  }
#endif
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

static Real avgUx_nonUniform(const std::vector<BlockInfo> &myInfo,
                             const Real *const uInf, const Real volume) {
  // Average Ux on the simulation volume :
  //   Sum on the xz-plane (uniform)
  //   Integral along Y    (non-uniform)
  //
  // <Ux>_{xz} (iy) = 1/(Nx.Ny) . \Sum_{ix, iz} u(ix,iy,iz)
  //
  // <Ux>_{xyz} = 1/Ly . \Sum_{iy} <Ux>_{xz} (iy) . h_y(*,iy,*)
  //            = /1(Nx.Ny.Ly) . \Sum_{ix,iy,iz} u(ix,iy,iz).h_y(ix,iy,iz)
  Real avgUx = 0.;
  const int nBlocks = myInfo.size();

#pragma omp parallel for reduction(+ : avgUx)
  for (int i = 0; i < nBlocks; i++) {
    const BlockInfo &info = myInfo[i];
    const VectorBlock &b = *(const VectorBlock *)info.ptrBlock;
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
  sim.startProfiler("FixedMassFlux");

  const std::vector<BlockInfo> &velInfo = sim.velInfo();

  // fix base_u_avg and y_max AD HOC for channel flow
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
    VectorBlock &v = *(VectorBlock *)velInfo[i].ptrBlock;
    for (int z = 0; z < VectorBlock::sizeZ; ++z)
      for (int y = 0; y < VectorBlock::sizeY; ++y) {
        Real p[3];
        velInfo[i].pos(p, 0, y, 0);
        const Real aux = 6 * scale * p[1] / y_max * (1.0 - p[1] / y_max);
        for (int x = 0; x < VectorBlock::sizeX; ++x)
          v(x, y, z).u[0] += aux;
      }
  }

  sim.stopProfiler();
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

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

  const std::vector<cubism::BlockInfo> &presInfo = sim.presInfo();

  KernelComputeForces(SimulationData &s) : sim(s) {}

  void operator()(VectorLab &lab, ScalarLab &chiLab, const BlockInfo &info,
                  const BlockInfo &info2) const {
    for (const auto &obstacle : sim.obstacle_vector->getObstacleVector())
      visit(lab, chiLab, info, info2, obstacle.get());
  }

  void visit(VectorLab &l, ScalarLab &chiLab, const BlockInfo &info,
             const BlockInfo &info2, Obstacle *const op) const {
    const ScalarBlock &presBlock =
        *(ScalarBlock *)presInfo[info.blockID].ptrBlock;
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

    // loop over elements of block info that have nonzero gradChi
    for (int i = 0; i < o->nPoints; i++) {
      const int ix = o->surface[i]->ix;
      const int iy = o->surface[i]->iy;
      const int iz = o->surface[i]->iz;

      Real p[3];
      info.pos(p, ix, iy, iz);

      // shear stresses
      const Real normX = o->surface[i]->dchidx; //*h^3 (multiplied in dchidx)
      const Real normY = o->surface[i]->dchidy; //*h^3 (multiplied in dchidy)
      const Real normZ = o->surface[i]->dchidz; //*h^3 (multiplied in dchidz)
      const Real norm =
          1.0 / std::sqrt(normX * normX + normY * normY + normZ * normZ);
      const Real dx = normX * norm;
      const Real dy = normY * norm;
      const Real dz = normZ * norm;

      int x = ix;
      int y = iy;
      int z = iz;
      for (int kk = 0; kk < 5; kk++) // 5 is arbitrary
      {
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

      // normals computed with Towers 2009
      // Actually using the volume integral, since (\iint -P \hat{n} dS) =
      // (\iiint -\nabla P dV). Also, P*\nabla\Chi = \nabla P penalty-accel and
      // surf-force match up if resolution is high enough (200 points per fish)
      const Real P = presBlock(ix, iy, iz).s;
      const Real fXV = _1oH * (dudx * normX + dudy * normY + dudz * normZ);
      const Real fYV = _1oH * (dvdx * normX + dvdy * normY + dvdz * normZ);
      const Real fZV = _1oH * (dwdx * normX + dwdy * normY + dwdz * normZ);

      const Real fXP = -P * normX, fYP = -P * normY, fZP = -P * normZ;
      const Real fXT = fXV + fXP, fYT = fYV + fYP, fZT = fZV + fZP;

      // store:
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

      // forces (total, visc, pressure):
      o->forcex += fXT;
      o->forcey += fYT;
      o->forcez += fZT;
      o->forcex_V += fXV;
      o->forcey_V += fYV;
      o->forcez_V += fZV;
      o->forcex_P += fXP;
      o->forcey_P += fYP;
      o->forcez_P += fZP;
      // torque:
      o->torquex += (p[1] - CM[1]) * fZT - (p[2] - CM[2]) * fYT;
      o->torquey += (p[2] - CM[2]) * fXT - (p[0] - CM[0]) * fZT;
      o->torquez += (p[0] - CM[0]) * fYT - (p[1] - CM[1]) * fXT;
      // thrust, drag:
      const Real forcePar =
          fXT * velUnit[0] + fYT * velUnit[1] + fZT * velUnit[2];
      o->thrust += .5 * (forcePar + std::fabs(forcePar));
      o->drag -= .5 * (forcePar - std::fabs(forcePar));

      // power output (and negative definite variant which ensures no elastic
      // energy absorption)
      // This is total power, for overcoming not only deformation, but also the
      // oncoming velocity. Work done by fluid, not by the object (for that,
      // just take -ve)
      const Real powOut = fXT * o->vX[i] + fYT * o->vY[i] + fZT * o->vZ[i];
      // deformation power output (and negative definite variant which ensures
      // no elastic energy absorption)
      const Real powDef =
          fXT * o->vxDef[i] + fYT * o->vyDef[i] + fZT * o->vzDef[i];
      o->Pout += powOut;
      o->PoutBnd += std::min((Real)0, powOut);
      o->defPower += powDef;
      o->defPowerBnd += std::min((Real)0, powDef);

      // Compute P_locomotion = Force*(uTrans + uRot)
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
  cubism::compute<KernelComputeForces, VectorGrid, VectorLab, ScalarGrid,
                  ScalarLab>(K, *sim.vel, *sim.chi);
  // do the final reductions and so on
  sim.obstacle_vector->computeForces();
}

CubismUP_3D_NAMESPACE_END

    class PoissonSolverBase;

CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

namespace {

class KernelIC {
public:
  KernelIC(const Real u) {}
  void operator()(const BlockInfo &info, VectorBlock &block) const {
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
  void operator()(const BlockInfo &info, VectorBlock &block) const {
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

class KernelIC_channel {
  const int dir;
  const std::array<Real, 3> ext;
  const Real uMax, H = ext[dir], FAC = 4 * uMax / H / H; // FAC = 0.5*G/mu
  // umax =  0.5*G/mu * 0.25*H*H
public:
  KernelIC_channel(const std::array<Real, 3> &extents, const Real U,
                   const int _dir)
      : dir(_dir), ext{extents}, uMax(U) {}

  void operator()(const BlockInfo &info, VectorBlock &block) const {
    for (int iz = 0; iz < VectorBlock::sizeZ; ++iz)
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
          block(ix, iy, iz).clear();
          Real p[3];
          info.pos(p, ix, iy, iz);
          block(ix, iy, iz).u[0] = FAC * p[dir] * (H - p[dir]);
        }
  }
};

class KernelIC_channelrandom {
  const int dir;
  const std::array<Real, 3> ext;
  const Real uMax, H = ext[dir], FAC = 4 * uMax / H / H; // FAC = 0.5*G/mu
  // const Real delta_tau = 5.0/180;
  // umax =  0.5*G/mu * 0.25*H*H
public:
  KernelIC_channelrandom(const std::array<Real, 3> &extents, const Real U,
                         const int _dir)
      : dir(_dir), ext{extents}, uMax(U) {}

  void operator()(const BlockInfo &info, VectorBlock &block) const {
    std::random_device seed;
    std::mt19937 gen(seed());
    std::normal_distribution<Real> dist(0.0, 0.01);
    for (int iz = 0; iz < VectorBlock::sizeZ; ++iz)
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
          block(ix, iy, iz).clear();
          block(ix, iy, iz).u[0] = dist(gen);
        }
    // If we set block(ix,iy,iz).u = U*(1.0+dist(gen)) the compiler gives
    // an annoying warning. Doing this slower approach with two loops makes
    // the warning disappear. This won't impact performance as it's done
    // onle once per simulation (initial conditions).
    for (int iz = 0; iz < ScalarBlock::sizeZ; ++iz)
      for (int iy = 0; iy < ScalarBlock::sizeY; ++iy)
        for (int ix = 0; ix < ScalarBlock::sizeX; ++ix) {
          Real p[3];
          info.pos(p, ix, iy, iz);
          const Real U = FAC * p[dir] * (H - p[dir]);
          block(ix, iy, iz).u[0] = U * (block(ix, iy, iz).u[0] + 1.0);
        }
  }
};

class KernelIC_pipe {
  const Real mu, R, uMax, G = (16 * mu * uMax) / (3 * R * R);
  Real position[3] = {0, 0, 0};

public:
  KernelIC_pipe(const Real _mu, const Real _R, const int _uMax, Real _pos[])
      : mu(_mu), R(_R), uMax(_uMax) {
    for (size_t i = 0; i < 3; i++)
      position[i] = _pos[i];
  }

  void operator()(const BlockInfo &info, VectorBlock &block) const {
    for (int iz = 0; iz < VectorBlock::sizeZ; ++iz)
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
          // Compute
          Real p[3];
          info.pos(p, ix, iy, iz);
          const Real dxSq = (position[0] - p[0]) * (position[0] - p[0]);
          const Real dySq = (position[1] - p[1]) * (position[1] - p[1]);
          const Real rSq = dxSq + dySq;

          // Set Poiseuille flow
          block(ix, iy, iz).clear();
          const Real RSq = R * R;
          if (rSq < RSq)
            block(ix, iy, iz).u[2] = G / (4 * mu) * (RSq - rSq);
          else
            block(ix, iy, iz).u[2] = 0.0;
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

  std::vector<Real> dx_coil; // tangent vector
  std::vector<Real> dy_coil; // tangent vector
  std::vector<Real> dz_coil; // tangent vector

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
    // 1. Fill vel with vorticity values
    const int nz = VectorBlock::sizeZ;
    const int ny = VectorBlock::sizeY;
    const int nx = VectorBlock::sizeX;
    std::vector<BlockInfo> &velInfo = sim.velInfo();
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

    // 2. Compute curl(omega)
    //   Here we use the "ComputeVorticity" function from ProcessHelpers.h
    //   This computes the curl of whatever is stored in sim.vel (currently the
    //   vorticity field) and saves it to tmpV.
    {
      ComputeVorticity findOmega(sim);
      findOmega(0);
    }

    // 3. Solve nabla^2 u = - curl(omega)
    std::shared_ptr<PoissonSolverBase> pressureSolver;
    pressureSolver = makePoissonSolver(sim);

    Real PoissonErrorTol = sim.PoissonErrorTol;
    Real PoissonErrorTolRel = sim.PoissonErrorTolRel;
    sim.PoissonErrorTol = 0; // we are solving this once, so we set tolerance =
                             // 0
    sim.PoissonErrorTolRel =
        0; // we are solving this once, so we set tolerance = 0
    for (int d = 0; d < 3; d++) {
// 3a. Fill RHS with -omega[d] and set initial guess to zero.
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

      // 3b. solve poisson equation
      pressureSolver->solve();

// 3c. Fill vel
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
    sim.PoissonErrorTol =
        PoissonErrorTol; // recover tolerance for pressure projection
    sim.PoissonErrorTolRel =
        PoissonErrorTolRel; // recover tolerance for pressure projection
  }
};

} // anonymous namespace

static void initialPenalization(SimulationData &sim, const Real dt) {
  const std::vector<BlockInfo> &velInfo = sim.velInfo();
  for (const auto &obstacle : sim.obstacle_vector->getObstacleVector()) {
    using CHI_MAT = Real[CUP_BLOCK_SIZEZ][CUP_BLOCK_SIZEY][CUP_BLOCK_SIZEX];
    using UDEFMAT = Real[CUP_BLOCK_SIZEZ][CUP_BLOCK_SIZEY][CUP_BLOCK_SIZEX][3];
// TODO: Refactor to use only one omp parallel.
#pragma omp parallel
    {
      const auto &obstblocks = obstacle->getObstacleBlocks();
      const std::array<Real, 3> centerOfMass = obstacle->getCenterOfMass();
      const std::array<Real, 3> uBody = obstacle->getTranslationVelocity();
      const std::array<Real, 3> omegaBody = obstacle->getAngularVelocity();

#pragma omp for schedule(dynamic)
      for (size_t i = 0; i < velInfo.size(); ++i) {
        const BlockInfo &info = velInfo[i];
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
              // what if multiple obstacles share a block??
              // let's plus equal and wake up during the night to stress about
              // it
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
    if (sim.verbose)
      printf("[CUP3D] - Zero-values initial conditions.\n");
    run(KernelIC(0));
  }
  if (sim.initCond == "taylorGreen") {
    if (sim.verbose)
      printf("[CUP3D] - Taylor Green vortex initial conditions.\n");
    run(KernelIC_taylorGreen(sim.extents, sim.uMax_forced));
  }
  if (sim.initCond == "channelRandom") {
    if (sim.verbose)
      printf("[CUP3D] - Channel flow random initial conditions.\n");
    if (sim.BCx_flag == wall) {
      printf("ERROR: channel flow must be periodic or dirichlet in x.\n");
      fflush(0);
      abort();
    }
    const bool channelY = sim.BCy_flag == wall, channelZ = sim.BCz_flag == wall;
    if ((channelY && channelZ) or (!channelY && !channelZ)) {
      printf("ERROR: wrong channel flow BC in y or z.\n");
      fflush(0);
      abort();
    }
    const int dir = channelY ? 1 : 2;
    run(KernelIC_channelrandom(sim.extents, sim.uMax_forced, dir));
  }
  if (sim.initCond == "channel") {
    if (sim.verbose)
      printf("[CUP3D] - Channel flow initial conditions.\n");
    if (sim.BCx_flag == wall) {
      printf("ERROR: channel flow must be periodic or dirichlet in x.\n");
      fflush(0);
      abort();
    }
    const bool channelY = sim.BCy_flag == wall, channelZ = sim.BCz_flag == wall;
    if ((channelY && channelZ) or (!channelY && !channelZ)) {
      printf("ERROR: wrong channel flow BC in y or z.\n");
      fflush(0);
      abort();
    }
    const int dir = channelY ? 1 : 2;
    run(KernelIC_channel(sim.extents, sim.uMax_forced, dir));
  }
  if (sim.initCond == "pipe") {
    // Make sure periodic boundary conditions are set in z-direction
    if (sim.verbose)
      printf("[CUP3D] - Channel flow initial conditions.\n");
    if (sim.BCz_flag != periodic) {
      printf("ERROR: pipe flow must be periodic in z.\n");
      fflush(0);
      abort();
    }

    // Get obstacle and need quantities from pipe
    const auto &obstacles = sim.obstacle_vector->getObstacleVector();
    Pipe *pipe = dynamic_cast<Pipe *>(obstacles[0].get());

    // Set initial conditions
    run(KernelIC_pipe(sim.nu, pipe->length / 2.0, sim.uMax_forced,
                      pipe->position));
  }
  if (sim.initCond == "vorticity") {
    if (sim.verbose)
      printf("[CUP3D] - Vorticity initial conditions.\n");
    IC_vorticity ic_vorticity(sim);
    ic_vorticity.run();
  }

  {
    std::vector<cubism::BlockInfo> &chiInfo = sim.chiInfo();
// zero fields, going to contain Udef:
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
    // store deformation velocities onto tmp fields:
    initialPenalization(sim, dt);
  }
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

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
#else // 2d stefan swimmer
    const std::array<Real, 6> curvature_points = {
        0, .15 * length, .4 * length, .65 * length, .9 * length, length};
    const std::array<Real, 6> curvature_values = {
        0.82014 / length, 1.46515 / length, 2.57136 / length,
        3.75425 / length, 5.09147 / length, 5.70449 / length};
    curvScheduler.transition(time, 0, 1, curvature_values, curvature_values);
    // query the schedulers for current values
    curvScheduler.gimmeValues(time, curvature_points, Nm, rS, rC, vC);
    // construct the curvature
    for (int i = 0; i < Nm; i++) {
      const Real darg = 2. * M_PI;
      const Real arg = 2. * M_PI * (time - rS[i] / length) + M_PI * phaseShift;
      rK[i] = rC[i] * std::sin(arg);
      vK[i] = vC[i] * std::sin(arg) + rC[i] * std::cos(arg) * darg;
    }

    // solve frenet to compute midline parameters
    IF2D_Frenet2D::solve(Nm, rS, rK, vK, rX, rY, vX, vY, norX, norY, vNorX,
                         vNorY);
#endif
  }
};

Naca::Naca(SimulationData &s, ArgumentParser &p) : Fish(s, p) {
  Apitch = p("-Apitch").asDouble(0.0) * M_PI /
           180;                        // aplitude of sinusoidal pitch angle
  Fpitch = p("-Fpitch").asDouble(0.0); // frequency
  Mpitch = p("-Mpitch").asDouble(0.0) * M_PI / 180; // mean angle
  Fheave = p("-Fheave").asDouble(0.0);          // frequency of rowing motion
  Aheave = p("-Aheave").asDouble(0.0) * length; // amplitude (NON DIMENSIONAL)
  tAccel = p("-tAccel").asDouble(-1);
  fixedCenterDist = p("-fixedCenterDist").asDouble(0);
  const Real thickness = p("-tRatio").asDouble(0.12);
  myFish = new NacaMidlineData(length, sim.hmin, sim.extents[2], thickness);
  if (sim.rank == 0 && sim.verbose)
    printf("[CUP3D] - NacaData Nm=%d L=%f t=%f A=%f w=%f xvel=%f yvel=%f "
           "tAccel=%f fixedCenterDist=%f\n",
           myFish->Nm, (double)length, (double)thickness, (double)Apitch,
           (double)Fpitch, (double)transVel_imposed[0],
           (double)transVel_imposed[1], (double)tAccel,
           (double)fixedCenterDist);
  // only allow rotation around z-axis and translation in xy-plane
  bBlockRotation[0] = true;
  bBlockRotation[1] = true;
  bForcedInSimFrame[2] = true;
}

void Naca::computeVelocities() {
  const Real omegaAngle = 2 * M_PI * Fpitch;
  const Real angle = Mpitch + Apitch * std::sin(omegaAngle * sim.time);
  const Real omega = Apitch * omegaAngle * std::cos(omegaAngle * sim.time);
  // angular velocity
  angVel[0] = 0;
  angVel[1] = 0;
  angVel[2] = omega;

  // heaving motion
  const Real v_heave =
      -2.0 * M_PI * Fheave * Aheave * std::sin(2 * M_PI * Fheave * sim.time);
  if (sim.time < tAccel) {
    // linear velocity (due to rotation-axis != CoM)
    transVel[0] = (1.0 - sim.time / tAccel) * 0.01 * transVel_imposed[0] +
                  (sim.time / tAccel) * transVel_imposed[0] -
                  fixedCenterDist * length * omega * std::sin(angle);
    transVel[1] = (1.0 - sim.time / tAccel) * 0.01 * transVel_imposed[1] +
                  (sim.time / tAccel) * transVel_imposed[1] +
                  fixedCenterDist * length * omega * std::cos(angle) + v_heave;
    transVel[2] = 0.0;
  } else {
    // linear velocity (due to rotation-axis != CoM)
    transVel[0] = transVel_imposed[0] -
                  fixedCenterDist * length * omega * std::sin(angle);
    transVel[1] = transVel_imposed[1] +
                  fixedCenterDist * length * omega * std::cos(angle) + v_heave;
    transVel[2] = 0.0;
  }
}

using intersect_t = std::vector<std::vector<VolumeSegment_OBB *>>;
void Naca::writeSDFOnBlocks(std::vector<VolumeSegment_OBB> &vSegments) {
#pragma omp parallel
  {
    PutNacaOnBlocks putfish(myFish, position, quaternion);
#pragma omp for
    for (size_t j = 0; j < MyBlockIDs.size(); j++) {
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
}

void Naca::updateLabVelocity(int nSum[3], Real uSum[3]) {
  // heaving motion
  const Real v_heave =
      -2.0 * M_PI * Fheave * Aheave * std::sin(2 * M_PI * Fheave * sim.time);

  if (bFixFrameOfRef[0]) {
    (nSum[0])++;
    if (sim.time < tAccel)
      uSum[0] -= (1.0 - sim.time / tAccel) * 0.01 * transVel_imposed[0] +
                 (sim.time / tAccel) * transVel_imposed[0];
    else
      uSum[0] -= transVel_imposed[0];
  }
  if (bFixFrameOfRef[1]) {
    (nSum[1])++;
    if (sim.time < tAccel)
      uSum[1] -= (1.0 - sim.time / tAccel) * 0.01 * transVel_imposed[1] +
                 (sim.time / tAccel) * transVel_imposed[1] + v_heave;
    else
      uSum[1] -= transVel_imposed[1] + v_heave;
  }
  if (bFixFrameOfRef[2]) {
    (nSum[2])++;
    uSum[2] -= transVel[2];
  }
}

void Naca::update() {
  const Real angle_2D =
      Mpitch + Apitch * std::cos(2 * M_PI * (Fpitch * sim.time));
  quaternion[0] = std::cos(0.5 * angle_2D);
  quaternion[1] = 0;
  quaternion[2] = 0;
  quaternion[3] = std::sin(0.5 * angle_2D);

  absPos[0] += sim.dt * transVel[0];
  absPos[1] += sim.dt * transVel[1];
  absPos[2] += sim.dt * transVel[2];

  position[0] += sim.dt * (transVel[0] + sim.uinf[0]);
  // if user wants to keep airfoil in the mid plane then we just integrate
  // relative velocity (should be 0), otherwise we know that y velocity
  // is sinusoidal, therefore we can just use analytical form
  if (bFixFrameOfRef[1])
    position[1] += sim.dt * (transVel[1] + sim.uinf[1]);
  else
    position[1] =
        sim.extents[1] / 2 + Aheave * std::cos(2 * M_PI * Fheave * sim.time);
  position[2] += sim.dt * (transVel[2] + sim.uinf[2]);
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

using UDEFMAT = Real[CUP_BLOCK_SIZEZ][CUP_BLOCK_SIZEY][CUP_BLOCK_SIZEX][3];
using CHIMAT = Real[CUP_BLOCK_SIZEZ][CUP_BLOCK_SIZEY][CUP_BLOCK_SIZEX];
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();

Obstacle::Obstacle(SimulationData &s, ArgumentParser &parser) : sim(s) {
  length = parser("-L").asDouble();         // Mandatory.
  position[0] = parser("-xpos").asDouble(); // Mandatory.
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

  // if true, obstacle will never change its velocity:
  bool bFSM_alldir = parser("-bForcedInSimFrame").asBool(false);
  bForcedInSimFrame[0] =
      bFSM_alldir || parser("-bForcedInSimFrame_x").asBool(false);
  bForcedInSimFrame[1] =
      bFSM_alldir || parser("-bForcedInSimFrame_y").asBool(false);
  bForcedInSimFrame[2] =
      bFSM_alldir || parser("-bForcedInSimFrame_z").asBool(false);
  // only active if corresponding bForcedInLabFrame is true:
  Real enforcedVelocity[3];
  enforcedVelocity[0] = -parser("-xvel").asDouble(0.0);
  enforcedVelocity[1] = -parser("-yvel").asDouble(0.0);
  enforcedVelocity[2] = -parser("-zvel").asDouble(0.0);
  const bool bFixToPlanar = parser("-bFixToPlanar").asBool(false);
  // this is different, obstacle can change the velocity, but sim frame will
  // follow:
  bool bFOR_alldir = parser("-bFixFrameOfRef").asBool(false);
  bFixFrameOfRef[0] = bFOR_alldir || parser("-bFixFrameOfRef_x").asBool(false);
  bFixFrameOfRef[1] = bFOR_alldir || parser("-bFixFrameOfRef_y").asBool(false);
  bFixFrameOfRef[2] = bFOR_alldir || parser("-bFixFrameOfRef_z").asBool(false);
  // boolean to break symmetry to trigger vortex shedding
  bBreakSymmetry = parser("-bBreakSymmetry").asBool(false);

  absPos[0] = position[0];
  absPos[1] = position[1];
  absPos[2] = position[2];

  if (!sim.rank) {
    printf("Obstacle L=%g, pos=[%g %g %g], q=[%g %g %g %g]\n", length,
           position[0], position[1], position[2], quaternion[0], quaternion[1],
           quaternion[2], quaternion[3]);
  }

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
  std::vector<double> A(36); // need to use double (not Real) for GSL
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

  // TODO here we can add dt * appliedForce/Torque[i]
  double b[6] = {// need to use double (not Real) for GSL
                 penalLmom[0], penalLmom[1], penalLmom[2],
                 penalAmom[0], penalAmom[1], penalAmom[2]};

  // modify y-velocity for symmetry breaking
  if (bBreakSymmetry) {
    if (sim.time > 3.0 && sim.time < 4.0)
      // transVel_imposed[1] = length*std::sin(M_PI*(sim.time-3.0)); // for
      // Re=300
      transVel_imposed[1] =
          0.1 * length * std::sin(M_PI * (sim.time - 3.0)); // for Re=1000
    else
      transVel_imposed[1] = 0.0;
  }

  // Momenta are conserved if a dof (a row of mat A) is not externally forced
  // This means that if obstacle is free to move according to fluid forces,
  // momenta after penal should be equal to moments before penal!
  // If dof is forced, change in momt. assumed to be entirely due to forcing.
  // In this case, leave row diagonal to compute change in momt for post/dbg.
  // If dof (row) is free then i need to fill the non-diagonal terms.
  if (bForcedInSimFrame[0]) { // then momenta not conserved in this dof
    A[0 * 6 + 1] = 0;
    A[0 * 6 + 2] = 0;
    A[0 * 6 + 3] = 0;
    A[0 * 6 + 4] = 0;
    A[0 * 6 + 5] = 0;
    b[0] = penalM * transVel_imposed[0]; // multply by penalM for conditioning
  }
  if (bForcedInSimFrame[1]) { // then momenta not conserved in this dof
    A[1 * 6 + 0] = 0;
    A[1 * 6 + 2] = 0;
    A[1 * 6 + 3] = 0;
    A[1 * 6 + 4] = 0;
    A[1 * 6 + 5] = 0;
    b[1] = penalM * transVel_imposed[1];
  }
  if (bForcedInSimFrame[2]) { // then momenta not conserved in this dof
    A[2 * 6 + 0] = 0;
    A[2 * 6 + 1] = 0;
    A[2 * 6 + 3] = 0;
    A[2 * 6 + 4] = 0;
    A[2 * 6 + 5] = 0;
    b[2] = penalM * transVel_imposed[2];
  }
  if (bBlockRotation[0]) { // then momenta not conserved in this dof
    A[3 * 6 + 0] = 0;
    A[3 * 6 + 1] = 0;
    A[3 * 6 + 2] = 0;
    A[3 * 6 + 4] = 0;
    A[3 * 6 + 5] = 0;
    b[3] = 0; // TODO IMPOSED ANG VEL?
  }
  if (bBlockRotation[1]) { // then momenta not conserved in this dof
    A[4 * 6 + 0] = 0;
    A[4 * 6 + 1] = 0;
    A[4 * 6 + 2] = 0;
    A[4 * 6 + 3] = 0;
    A[4 * 6 + 5] = 0;
    b[4] = 0; // TODO IMPOSED ANG VEL?
  }
  if (bBlockRotation[2]) { // then momenta not conserved in this dof
    A[5 * 6 + 0] = 0;
    A[5 * 6 + 1] = 0;
    A[5 * 6 + 2] = 0;
    A[5 * 6 + 3] = 0;
    A[5 * 6 + 4] = 0;
    b[5] = 0; // TODO IMPOSED ANG VEL?
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

  // additive quantities: (check against order in sumQoI of ObstacleBlocks.h )
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
  // derived quantities:
  Pthrust = thrust * vel_norm;
  Pdrag = drag * vel_norm;
  EffPDef = Pthrust / (Pthrust - std::min(defPower, (Real)0) + EPS);
  EffPDefBnd = Pthrust / (Pthrust - defPowerBnd + EPS);

  _writeSurfForcesToFile();
  _writeDiagForcesToFile();
  _writeComputedVelToFile();
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

  /*
    // normality preserving advection (Simulation of colliding constrained rigid
    bodies - Kleppmann 2007 Cambridge University, p51)
    // move the correct distance on the quaternion unit ball surface, end up
    with normalized quaternion const Real DQ[4] = { dqdt[0]*dt, dqdt[1]*dt,
    dqdt[2]*dt, dqdt[3]*dt }; const Real DQn =
    std::sqrt(DQ[0]*DQ[0]+DQ[1]*DQ[1]+DQ[2]*DQ[2]+DQ[3]*DQ[3]);

    if(DQn>EPS)// && currentRKstep == 0)
    {
      const Real tanF = std::tan(DQn)/DQn;
      const Real D[4] = {
        Q[0] +tanF*DQ[0], Q[1] +tanF*DQ[1], Q[2] +tanF*DQ[2], Q[3] +tanF*DQ[3],
      };
      const Real invD = 1/std::sqrt(D[0]*D[0]+D[1]*D[1]+D[2]*D[2]+D[3]*D[3]);
      quaternion[0] = D[0] * invD; quaternion[1] = D[1] * invD;
      quaternion[2] = D[2] * invD; quaternion[3] = D[3] * invD;
    }
  */

  if (sim.verbose && sim.time > 0) {
    const Real rad2deg = 180. / M_PI;
    std::array<Real, 3> ypr = getYawPitchRoll();
    ypr[0] *= rad2deg;
    ypr[1] *= rad2deg;
    ypr[2] *= rad2deg;
    printf("pos:[%+.2f %+.2f %+.2f], u:[%+.2f %+.2f %+.2f], omega:[%+.2f %+.2f "
           "%+.2f], yaw: %+.1f, pitch: %+.1f, roll: %+.1f \n",
           absPos[0], absPos[1], absPos[2], transVel[0], transVel[1],
           transVel[2], angVel[0], angVel[1], angVel[2], ypr[0], ypr[1],
           ypr[2]);
  }
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

void Obstacle::saveRestart(FILE *f) {
  assert(f != NULL);
  fprintf(f, "x:       %20.20e\n", (double)position[0]);
  fprintf(f, "y:       %20.20e\n", (double)position[1]);
  fprintf(f, "z:       %20.20e\n", (double)position[2]);
  fprintf(f, "xAbs:    %20.20e\n", (double)absPos[0]);
  fprintf(f, "yAbs:    %20.20e\n", (double)absPos[1]);
  fprintf(f, "zAbs:    %20.20e\n", (double)absPos[2]);
  fprintf(f, "quat_0:  %20.20e\n", (double)quaternion[0]);
  fprintf(f, "quat_1:  %20.20e\n", (double)quaternion[1]);
  fprintf(f, "quat_2:  %20.20e\n", (double)quaternion[2]);
  fprintf(f, "quat_3:  %20.20e\n", (double)quaternion[3]);
  fprintf(f, "u_x:     %20.20e\n", (double)transVel[0]);
  fprintf(f, "u_y:     %20.20e\n", (double)transVel[1]);
  fprintf(f, "u_z:     %20.20e\n", (double)transVel[2]);
  fprintf(f, "omega_x: %20.20e\n", (double)angVel[0]);
  fprintf(f, "omega_y: %20.20e\n", (double)angVel[1]);
  fprintf(f, "omega_z: %20.20e\n", (double)angVel[2]);
  fprintf(f, "old_position_0:    %20.20e\n", (double)old_position[0]);
  fprintf(f, "old_position_1:    %20.20e\n", (double)old_position[1]);
  fprintf(f, "old_position_2:    %20.20e\n", (double)old_position[2]);
  fprintf(f, "old_absPos_0:      %20.20e\n", (double)old_absPos[0]);
  fprintf(f, "old_absPos_1:      %20.20e\n", (double)old_absPos[1]);
  fprintf(f, "old_absPos_2:      %20.20e\n", (double)old_absPos[2]);
  fprintf(f, "old_quaternion_0:  %20.20e\n", (double)old_quaternion[0]);
  fprintf(f, "old_quaternion_1:  %20.20e\n", (double)old_quaternion[1]);
  fprintf(f, "old_quaternion_2:  %20.20e\n", (double)old_quaternion[2]);
  fprintf(f, "old_quaternion_3:  %20.20e\n", (double)old_quaternion[3]);
}

void Obstacle::loadRestart(FILE *f) {
  assert(f != NULL);
  bool ret = true;
  double temp;
  ret = ret && 1 == fscanf(f, "x:       %le\n", &temp);
  position[0] = temp;
  ret = ret && 1 == fscanf(f, "y:       %le\n", &temp);
  position[1] = temp;
  ret = ret && 1 == fscanf(f, "z:       %le\n", &temp);
  position[2] = temp;
  ret = ret && 1 == fscanf(f, "xAbs:    %le\n", &temp);
  absPos[0] = temp;
  ret = ret && 1 == fscanf(f, "yAbs:    %le\n", &temp);
  absPos[1] = temp;
  ret = ret && 1 == fscanf(f, "zAbs:    %le\n", &temp);
  absPos[2] = temp;
  ret = ret && 1 == fscanf(f, "quat_0:  %le\n", &temp);
  quaternion[0] = temp;
  ret = ret && 1 == fscanf(f, "quat_1:  %le\n", &temp);
  quaternion[1] = temp;
  ret = ret && 1 == fscanf(f, "quat_2:  %le\n", &temp);
  quaternion[2] = temp;
  ret = ret && 1 == fscanf(f, "quat_3:  %le\n", &temp);
  quaternion[3] = temp;
  ret = ret && 1 == fscanf(f, "u_x:     %le\n", &temp);
  transVel[0] = temp;
  ret = ret && 1 == fscanf(f, "u_y:     %le\n", &temp);
  transVel[1] = temp;
  ret = ret && 1 == fscanf(f, "u_z:     %le\n", &temp);
  transVel[2] = temp;
  ret = ret && 1 == fscanf(f, "omega_x: %le\n", &temp);
  angVel[0] = temp;
  ret = ret && 1 == fscanf(f, "omega_y: %le\n", &temp);
  angVel[1] = temp;
  ret = ret && 1 == fscanf(f, "omega_z: %le\n", &temp);
  angVel[2] = temp;
  ret = ret && 1 == fscanf(f, "old_position_0:    %le\n", &temp);
  old_position[0] = temp;
  ret = ret && 1 == fscanf(f, "old_position_1:    %le\n", &temp);
  old_position[1] = temp;
  ret = ret && 1 == fscanf(f, "old_position_2:    %le\n", &temp);
  old_position[2] = temp;
  ret = ret && 1 == fscanf(f, "old_absPos_0:      %le\n", &temp);
  old_absPos[0] = temp;
  ret = ret && 1 == fscanf(f, "old_absPos_1:      %le\n", &temp);
  old_absPos[1] = temp;
  ret = ret && 1 == fscanf(f, "old_absPos_2:      %le\n", &temp);
  old_absPos[2] = temp;
  ret = ret && 1 == fscanf(f, "old_quaternion_0:  %le\n", &temp);
  old_quaternion[0] = temp;
  ret = ret && 1 == fscanf(f, "old_quaternion_1:  %le\n", &temp);
  old_quaternion[1] = temp;
  ret = ret && 1 == fscanf(f, "old_quaternion_2:  %le\n", &temp);
  old_quaternion[2] = temp;
  ret = ret && 1 == fscanf(f, "old_quaternion_3:  %le\n", &temp);
  old_quaternion[3] = temp;

  if ((not ret)) {
    printf("Error reading restart file. Aborting...\n");
    fflush(0);
    abort();
  }
  if (sim.verbose == 0 && sim.rank == 0)
    printf("Restarting Object.. x: %le, y: %le, z: %le, u_x: %le, u_y: %le, "
           "u_z: %le\n",
           (double)absPos[0], (double)absPos[1], (double)absPos[2], transVel[0],
           (double)transVel[1], (double)transVel[2]);
}

void Obstacle::_writeComputedVelToFile() {
  if (sim.rank != 0 || sim.muteAll)
    return;
  std::stringstream ssR;
  ssR << "velocity_" << obstacleID << ".dat";
  std::stringstream &savestream = logger.get_stream(ssR.str());

  if (sim.step == 0 && not printedHeaderVels) {
    printedHeaderVels = true;
    savestream << "step time CMx CMy CMz quat_0 quat_1 quat_2 quat_3 vel_x "
                  "vel_y vel_z angvel_x angvel_y angvel_z mass J0 J1 J2 J3 J4 "
                  "J5 yaw pitch roll"
               << std::endl;
  }

  const Real rad2deg = 180. / M_PI;
  std::array<Real, 3> ypr = getYawPitchRoll();
  ypr[0] *= rad2deg;
  ypr[1] *= rad2deg;
  ypr[2] *= rad2deg;

  savestream << sim.step << " ";
  savestream.setf(std::ios::scientific);
  savestream.precision(std::numeric_limits<float>::digits10 + 1);
  savestream << sim.time << " " << absPos[0] << " " << absPos[1] << " "
             << absPos[2] << " " << quaternion[0] << " " << quaternion[1] << " "
             << quaternion[2] << " " << quaternion[3] << " " << transVel[0]
             << " " << transVel[1] << " " << transVel[2] << " " << angVel[0]
             << " " << angVel[1] << " " << angVel[2] << " " << mass << " "
             << J[0] << " " << J[1] << " " << J[2] << " " << J[3] << " " << J[4]
             << " " << J[5] << " " << ypr[0] << " " << ypr[1] << " " << ypr[2]
             << std::endl;
}

void Obstacle::_writeSurfForcesToFile() {
  if (sim.rank != 0 || sim.muteAll)
    return;
  std::stringstream fnameF, fnameP;
  fnameF << "forceValues_" << obstacleID << ".dat";
  std::stringstream &ssF = logger.get_stream(fnameF.str());
  if (sim.step == 0) {
    ssF << "step time Fx Fy Fz torque_x torque_y torque_z FxPres FyPres FzPres "
           "FxVisc FyVisc FzVisc drag thrust"
        << std::endl;
  }

  ssF << sim.step << " ";
  ssF.setf(std::ios::scientific);
  ssF.precision(std::numeric_limits<float>::digits10 + 1);
  ssF << sim.time << " " << surfForce[0] << " " << surfForce[1] << " "
      << surfForce[2] << " " << surfTorque[0] << " " << surfTorque[1] << " "
      << surfTorque[2] << " " << presForce[0] << " " << presForce[1] << " "
      << presForce[2] << " " << viscForce[0] << " " << viscForce[1] << " "
      << viscForce[2] << " " << drag << " " << thrust << std::endl;

  fnameP << "powerValues_" << obstacleID << ".dat";
  std::stringstream &ssP = logger.get_stream(fnameP.str());
  if (sim.step == 0) {
    ssP << "time Pthrust Pdrag PoutBnd Pout PoutNew defPowerBnd defPower "
           "EffPDefBnd EffPDef"
        << std::endl;
  }
  ssP.setf(std::ios::scientific);
  ssP.precision(std::numeric_limits<float>::digits10 + 1);
  ssP << sim.time << " " << Pthrust << " " << Pdrag << " " << PoutBnd << " "
      << Pout << " " << pLocom << " " << defPowerBnd << " " << defPower << " "
      << EffPDefBnd << " " << EffPDef << std::endl;
}

void Obstacle::_writeDiagForcesToFile() {
  if (sim.rank != 0 || sim.muteAll)
    return;
  std::stringstream fnameF;
  fnameF << "forceValues_penalization_" << obstacleID << ".dat";
  std::stringstream &ssF = logger.get_stream(fnameF.str());
  if (sim.step == 0) {
    ssF << "step time mass force_x force_y force_z torque_x torque_y torque_z "
           "penalLmom_x penalLmom_y penalLmom_z penalAmom_x penalAmom_y "
           "penalAmom_z penalCM_x penalCM_y penalCM_z linVel_comp_x "
           "linVel_comp_y linVel_comp_z angVel_comp_x angVel_comp_y "
           "angVel_comp_z penalM penalJ0 penalJ1 penalJ2 penalJ3 penalJ4 "
           "penalJ5"
        << std::endl;
  }

  ssF << sim.step << " ";
  ssF.setf(std::ios::scientific);
  ssF.precision(std::numeric_limits<float>::digits10 + 1);
  ssF << sim.time << " " << mass << " " << force[0] << " " << force[1] << " "
      << force[2] << " " << torque[0] << " " << torque[1] << " " << torque[2]
      << " " << penalLmom[0] << " " << penalLmom[1] << " " << penalLmom[2]
      << " " << penalAmom[0] << " " << penalAmom[1] << " " << penalAmom[2]
      << " " << penalCM[0] << " " << penalCM[1] << " " << penalCM[2] << " "
      << transVel_computed[0] << " " << transVel_computed[1] << " "
      << transVel_computed[2] << " " << angVel_computed[0] << " "
      << angVel_computed[1] << " " << angVel_computed[2] << " " << penalM << " "
      << penalJ[0] << " " << penalJ[1] << " " << penalJ[2] << " " << penalJ[3]
      << " " << penalJ[4] << " " << penalJ[5] << std::endl;
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;
using VectorType = ObstacleVector::VectorType;

/*
 * Create an obstacle instance given its name and arguments.
 */
static std::shared_ptr<Obstacle>
_createObstacle(SimulationData &sim, const std::string &objectName,
                FactoryFileLineParser &lineParser) {
  if (objectName == "Sphere")
    return std::make_shared<Sphere>(sim, lineParser);
  if (objectName == "StefanFish" || objectName == "stefanfish")
    return std::make_shared<StefanFish>(sim, lineParser);
  if (objectName == "CarlingFish")
    return std::make_shared<CarlingFish>(sim, lineParser);
  if (objectName == "Naca")
    return std::make_shared<Naca>(sim, lineParser);
  if (objectName == "SmartNaca")
    return std::make_shared<SmartNaca>(sim, lineParser);
  if (objectName == "Cylinder")
    return std::make_shared<Cylinder>(sim, lineParser);
  if (objectName == "CylinderNozzle")
    return std::make_shared<CylinderNozzle>(sim, lineParser);
  if (objectName == "Plate")
    return std::make_shared<Plate>(sim, lineParser);
  if (objectName == "Pipe")
    return std::make_shared<Pipe>(sim, lineParser);
  if (objectName == "Ellipsoid")
    return std::make_shared<Ellipsoid>(sim, lineParser);
  if (objectName == "ExternalObstacle")
    return std::make_shared<ExternalObstacle>(sim, lineParser);

  if (sim.rank == 0) {
    std::cout << "[CUP3D] Case " << objectName << " is not defined: aborting\n"
              << std::flush;
    abort();
  }

  return {};
}

/*
 * Add one obstacle per non-empty non-comment line of the given stream.
 */
static void _addObstacles(SimulationData &sim, std::stringstream &stream) {
  // if (sim.rank == 0)
  //   printf("[CUP3D] Factory content:\n%s\n\n", stream.str().c_str());
  // here we store the data per object
  std::vector<std::pair<std::string, FactoryFileLineParser>> factoryLines;
  std::string line;

  while (std::getline(stream, line)) {
    std::istringstream line_stream(line);
    std::string ID;
    line_stream >> ID;
    if (ID.empty() || ID[0] == '#')
      continue; // Comments and empty lines ignored.
    factoryLines.emplace_back(ID, FactoryFileLineParser(line_stream));
  }
  if (factoryLines.empty()) {
    if (sim.rank == 0)
      std::cout << "[CUP3D] OBSTACLE FACTORY did not create any obstacles.\n";
    return;
  }
  if (sim.rank == 0) {
    std::cout << "-------------   OBSTACLE FACTORY : START ("
              << factoryLines.size() << " objects)   ------------\n";
  }

  for (auto &l : factoryLines) {
    sim.obstacle_vector->addObstacle(_createObstacle(sim, l.first, l.second));
    if (sim.rank == 0)
      std::cout << "-----------------------------------------------------------"
                   "---------"
                << std::endl;
  }
}

void ObstacleFactory::addObstacles(cubism::ArgumentParser &parser) {
  // Read parser information
  parser.unset_strict_mode();
  const std::string factory_filename = parser("-factory").asString("factory");
  std::string factory_content = parser("-factory-content").asString("");
  if (factory_content.compare("") == 0)
    factory_content = parser("-shapes").asString("");

  std::stringstream stream(factory_content);
  if (!factory_filename.empty()) {
    // https://stackoverflow.com/questions/132358/how-to-read-file-content-into-istringstream
    // Good enough solution.
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

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

namespace {

using CHIMAT = Real[CUP_BLOCK_SIZEZ][CUP_BLOCK_SIZEY][CUP_BLOCK_SIZEX];
using UDEFMAT = Real[CUP_BLOCK_SIZEZ][CUP_BLOCK_SIZEY][CUP_BLOCK_SIZEX][3];
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();

struct KernelCharacteristicFunction {
  using v_v_ob = std::vector<std::vector<ObstacleBlock *> *>;
  const v_v_ob &vec_obstacleBlocks;
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;

  KernelCharacteristicFunction(const v_v_ob &v) : vec_obstacleBlocks(v) {}

  void operate(const BlockInfo &info, ScalarBlock &b) const {
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
      //////////////////////////
      // FDMH_1 computation to approximate Heaviside function H(SDF(x,y,z))
      // Reference: John D.Towers, "Finite difference methods for approximating
      // Heaviside functions", eq.(14)
      //////////////////////////
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x) {
#if 1
            // here I read fist from SDF to deal with obstacles sharing block
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
              // gradU
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
              // gradI: first primitive of H(x): I(x) = int_0^x H(y) dy
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
            // gradU
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
            const Real Delta = fac1 * numD / gradUSq; // h^3 * Delta
            if (Delta > EPS)
              o->write(x, y, z, Delta, gradUX, gradUY, gradUZ);
          }
      o->allocate_surface();
    }
  }
};

} // anonymous namespace

/// Compute chi-based center of mass for each obstacle.
static void kernelComputeGridCoM(SimulationData &sim) {
  // TODO: Refactor to use only one omp parallel and only one MPI_Allreduce.
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

static void _kernelIntegrateUdefMomenta(SimulationData &sim,
                                        const BlockInfo &info) {
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  for (const auto &obstacle : sim.obstacle_vector->getObstacleVector()) {
    const auto &obstblocks = obstacle->getObstacleBlocks();
    ObstacleBlock *const o = obstblocks[info.blockID];
    if (o == nullptr)
      continue;

    const std::array<Real, 3> CM = obstacle->getCenterOfMass();
    // We use last momentum computed by this method to stabilize the computation
    // of the ang vel. This is because J is really tiny.
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

/// Integrate momenta over the grid.
static void kernelIntegrateUdefMomenta(SimulationData &sim) {
  const std::vector<cubism::BlockInfo> &chiInfo = sim.chiInfo();
#pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < chiInfo.size(); ++i)
    _kernelIntegrateUdefMomenta(sim, chiInfo[i]);
}

/// Reduce momenta across blocks and MPI.
static void kernelAccumulateUdefMomenta(SimulationData &sim,
                                        bool justDebug = false) {
  // TODO: Refactor to use only one omp parallel and one MPI_Allreduce.
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
      // solve avel = invJ \dot angMomentum
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

/// Remove momenta from udef.
static void kernelRemoveUdefMomenta(SimulationData &sim,
                                    bool justDebug = false) {
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  const std::vector<BlockInfo> &chiInfo = sim.chiInfo();

  // TODO: Refactor to use only one omp parallel.
  for (const auto &obstacle : sim.obstacle_vector->getObstacleVector()) {
    const std::array<Real, 3> angVel_correction = obstacle->angVel_correction;
    const std::array<Real, 3> transVel_correction =
        obstacle->transVel_correction;

    const std::array<Real, 3> CM = obstacle->getCenterOfMass();
    const auto &obstacleBlocks = obstacle->getObstacleBlocks();

#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < chiInfo.size(); i++) {
      const BlockInfo &info = chiInfo[i];
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

  std::vector<BlockInfo> &chiInfo = sim.chiInfo();
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < chiInfo.size(); ++i) {
    ScalarBlock &CHI = *(ScalarBlock *)chiInfo[i].ptrBlock;
    CHI.clear();
  }

  // Obstacles' advection must be done after we perform penalization:
  sim.uinf = sim.obstacle_vector->updateUinf();
  sim.obstacle_vector->update();

  { // put signed distance function on the grid
    sim.obstacle_vector->create();
  }

  {
#pragma omp parallel
    {
      auto vecOB = sim.obstacle_vector->getAllObstacleBlocks();
      const KernelCharacteristicFunction K(vecOB);
#pragma omp for
      for (size_t i = 0; i < chiInfo.size(); ++i) {
        ScalarBlock &CHI = *(ScalarBlock *)chiInfo[i].ptrBlock;
        K.operate(chiInfo[i], CHI);
      }
    }
  }

  // compute actual CoM given the CHI on the grid
  kernelComputeGridCoM(sim);
  kernelIntegrateUdefMomenta(sim);
  kernelAccumulateUdefMomenta(sim);
  kernelRemoveUdefMomenta(sim);
  sim.obstacle_vector->finalize(); // whatever else the obstacle needs
}

CubismUP_3D_NAMESPACE_END

    // define this to update obstacles with old (mrag-like) approach of
    // integrating momenta contained in chi before the penalization step:

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

namespace {

using CHIMAT = Real[CUP_BLOCK_SIZEZ][CUP_BLOCK_SIZEY][CUP_BLOCK_SIZEX];
using UDEFMAT = Real[CUP_BLOCK_SIZEZ][CUP_BLOCK_SIZEY][CUP_BLOCK_SIZEX][3];

template <bool implicitPenalization> struct KernelIntegrateFluidMomenta {
  const Real lambda, dt;
  ObstacleVector *const obstacle_vector;

  Real dvol(const BlockInfo &I, const int x, const int y, const int z) const {
    return I.h * I.h * I.h;
  }

  KernelIntegrateFluidMomenta(Real _dt, Real _lambda, ObstacleVector *ov)
      : lambda(_lambda), dt(_dt), obstacle_vector(ov) {}

  void operator()(const cubism::BlockInfo &info) const {
    for (const auto &obstacle : obstacle_vector->getObstacleVector())
      visit(info, obstacle.get());
  }

  void visit(const BlockInfo &info, Obstacle *const op) const {
    const std::vector<ObstacleBlock *> &obstblocks = op->getObstacleBlocks();
    ObstacleBlock *const o = obstblocks[info.blockID];
    if (o == nullptr)
      return;

    const std::array<Real, 3> CM = op->getCenterOfMass();
    const VectorBlock &b = *(VectorBlock *)info.ptrBlock;
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

} // Anonymous namespace.

template <bool implicitPenalization>
static void kernelFinalizeObstacleVel(SimulationData &sim, const Real dt) {
  // TODO: Refactor to use only one omp parallel and one MPI_Allreduce.
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

  { // integrate momenta by looping over grid
    std::vector<cubism::BlockInfo> &velInfo = sim.velInfo();
#pragma omp parallel
    {
      // if(0) {
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

  // if(0) {
  if (sim.bImplicitPenalization) {
    kernelFinalizeObstacleVel<1>(sim, dt);
  } else {
    kernelFinalizeObstacleVel<0>(sim, dt);
  }
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

namespace {

using CHIMAT = Real[CUP_BLOCK_SIZEZ][CUP_BLOCK_SIZEY][CUP_BLOCK_SIZEX];
using UDEFMAT = Real[CUP_BLOCK_SIZEZ][CUP_BLOCK_SIZEY][CUP_BLOCK_SIZEX][3];

struct KernelPenalization {
  const Real dt, invdt = 1.0 / dt, lambda;
  const bool implicitPenalization;
  ObstacleVector *const obstacle_vector;

  KernelPenalization(const Real _dt, const Real _lambda,
                     const bool _implicitPenalization, ObstacleVector *ov)
      : dt(_dt), lambda(_lambda), implicitPenalization(_implicitPenalization),
        obstacle_vector(ov) {}

  void operator()(const cubism::BlockInfo &info,
                  const BlockInfo &ChiInfo) const {
    for (const auto &obstacle : obstacle_vector->getObstacleVector())
      visit(info, ChiInfo, obstacle.get());
  }

  void visit(const BlockInfo &info, const BlockInfo &ChiInfo,
             Obstacle *const obstacle) const {
    const auto &obstblocks = obstacle->getObstacleBlocks();
    ObstacleBlock *const o = obstblocks[info.blockID];
    if (o == nullptr)
      return;

    const CHIMAT &__restrict__ CHI = o->chi;
    const UDEFMAT &__restrict__ UDEF = o->udef;
    VectorBlock &b = *(VectorBlock *)info.ptrBlock;
    ScalarBlock &bChi = *(ScalarBlock *)ChiInfo.ptrBlock;
    const std::array<Real, 3> CM = obstacle->getCenterOfMass();
    const std::array<Real, 3> vel = obstacle->getTranslationVelocity();
    const std::array<Real, 3> omega = obstacle->getAngularVelocity();
    const Real dv = std::pow(info.h, 3);

    // Obstacle-specific lambda, useful for gradually adding an obstacle to the
    // flow. lambda = 1/dt hardcoded for expl time int, other options are wrong.
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
          // What if multiple obstacles share a block? Do not write udef onto
          // grid if CHI stored on the grid is greater than obst's CHI.
          if (bChi(ix, iy, iz).s > CHI[iz][iy][ix])
            continue;
          if (CHI[iz][iy][ix] <= 0)
            continue; // no need to do anything
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
          // What if two obstacles overlap? Let's plus equal. We will need a
          // repulsion term of the velocity at some point in the code.
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
  // TODO: Refactor to use only one omp parallel and MPI_Allreduce.
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
  // Invert I
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
  const Real e = 1.0; // coefficient of restitution
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

  struct CollisionInfo // hitter and hittee, symmetry but we do things twice
  {
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

  std::vector<Real> buffer(20 * N); // CollisionInfo holds 20 Reals
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

      // less than 'tolerance' fluid element(s) of overlap: wait to get closer.
      // no hit
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
              0.2) // used 0.2 because fish lenght is 0.2 usually!
      {
        continue; // then both objects i and j collided, but not with each
                  // other!
      }

      // A collision happened!
      sim.bCollision = true;
#pragma omp critical
      {
        sim.bCollisionID.push_back(i);
        sim.bCollisionID.push_back(j);
      }

      // 1. Compute collision normal vector (NX,NY,NZ)
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
        continue; // vel goes away from collision: no need to bounce

      // 2. Compute collision location
      const Real inv_iM = 1.0 / coll.iM;
      const Real inv_jM = 1.0 / coll.jM;
      const Real iPX = coll.iPosX * inv_iM; // object i collision location
      const Real iPY = coll.iPosY * inv_iM;
      const Real iPZ = coll.iPosZ * inv_iM;
      const Real jPX = coll.jPosX * inv_jM; // object j collision location
      const Real jPY = coll.jPosY * inv_jM;
      const Real jPZ = coll.jPosZ * inv_jM;
      const Real CX = 0.5 * (iPX + jPX);
      const Real CY = 0.5 * (iPY + jPY);
      const Real CZ = 0.5 * (iPZ + jPZ);

      // 3. Take care of the collision. Assume elastic collision (kinetic energy
      // is conserved)
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

      if (sim.verbose) {
#pragma omp critical
        {
          std::cout << "Collision between objects " << i << " and " << j
                    << std::endl;
          std::cout << " iM   (0) = " << collisions[i].iM
                    << " jM   (1) = " << collisions[j].jM << std::endl;
          std::cout << " jM   (0) = " << collisions[i].jM
                    << " jM   (1) = " << collisions[j].iM << std::endl;
          std::cout << " Normal vector = (" << NX << "," << NY << "," << NZ
                    << ")" << std::endl;
          std::cout << " Location      = (" << CX << "," << CY << "," << CZ
                    << ")" << std::endl;
          std::cout << " Shape " << i << " before collision u    =(" << v1[0]
                    << "," << v1[1] << "," << v1[2] << ")" << std::endl;
          std::cout << " Shape " << i << " after  collision u    =(" << hv1[0]
                    << "," << hv1[1] << "," << hv1[2] << ")" << std::endl;
          std::cout << " Shape " << j << " before collision u    =(" << v2[0]
                    << "," << v2[1] << "," << v2[2] << ")" << std::endl;
          std::cout << " Shape " << j << " after  collision u    =(" << hv2[0]
                    << "," << hv2[1] << "," << hv2[2] << ")" << std::endl;
          std::cout << " Shape " << i << " before collision omega=(" << o1[0]
                    << "," << o1[1] << "," << o1[2] << ")" << std::endl;
          std::cout << " Shape " << i << " after  collision omega=(" << ho1[0]
                    << "," << ho1[1] << "," << ho1[2] << ")" << std::endl;
          std::cout << " Shape " << j << " before collision omega=(" << o2[0]
                    << "," << o2[1] << "," << o2[2] << ")" << std::endl;
          std::cout << " Shape " << j << " after  collision omega=(" << ho2[0]
                    << "," << ho2[1] << "," << ho2[2] << ")" << std::endl;
        }
      }
    }
}

Penalization::Penalization(SimulationData &s) : Operator(s) {}

void Penalization::operator()(const Real dt) {
  if (sim.obstacle_vector->nObstacles() == 0)
    return;

  preventCollidingObstacles();

  std::vector<cubism::BlockInfo> &chiInfo = sim.chiInfo();
  std::vector<cubism::BlockInfo> &velInfo = sim.velInfo();
#pragma omp parallel
  {
    // each thread needs to call its own non-const operator() function
    KernelPenalization K(dt, sim.lambda, sim.bImplicitPenalization,
                         sim.obstacle_vector);
#pragma omp for schedule(dynamic, 1)
    for (size_t i = 0; i < chiInfo.size(); ++i)
      K(velInfo[i], chiInfo[i]);
  }

  kernelFinalizePenalizationForce(sim);
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

namespace PipeObstacle {
struct FillBlocks : FillBlocksBase<FillBlocks> {
  const Real radius, halflength, h, safety = (8 + SURFDH) * h;
  const Real position[3];
  const Real box[3][2] = {
      {(Real)position[0] - (std::sqrt(2) / 2 * radius) + safety,
       (Real)position[0] + (std::sqrt(2) / 2 * radius) - safety},
      {(Real)position[1] - (std::sqrt(2) / 2 * radius) + safety,
       (Real)position[1] + (std::sqrt(2) / 2 * radius) - safety},
      {(Real)position[2] - halflength - safety,
       (Real)position[2] + halflength + safety}};

  FillBlocks(const Real r, const Real halfl, const Real _h, const Real p[3])
      : radius(r), halflength(halfl), h(_h), position{p[0], p[1], p[2]} {}

  inline bool isTouching(const BlockInfo &info, const ScalarBlock &b) const {
    Real MINP[3], MAXP[3];
    info.pos(MINP, 0, 0, 0);
    info.pos(MAXP, ScalarBlock::sizeX - 1, ScalarBlock::sizeY - 1,
             ScalarBlock::sizeZ - 1);
    const Real intersect[3][2] = {
        {std::max(MINP[0], box[0][0]), std::min(MAXP[0], box[0][1])},
        {std::max(MINP[1], box[1][0]), std::min(MAXP[1], box[1][1])},
        {std::max(MINP[2], box[2][0]), std::min(MAXP[2], box[2][1])}};
    return not(intersect[0][1] - intersect[0][0] > 0 &&
               intersect[1][1] - intersect[1][0] > 0 &&
               intersect[2][1] - intersect[2][0] > 0);
  }

  inline Real signedDistance(const Real xo, const Real yo,
                             const Real zo) const {
    const Real x = xo - position[0], y = yo - position[1], z = zo - position[2];
    const Real planeDist = radius - std::sqrt(x * x + y * y);
    const Real vertiDist = halflength - std::fabs(z);
    return -std::min(planeDist, vertiDist);
  }
};
} // namespace PipeObstacle

Pipe::Pipe(SimulationData &s, ArgumentParser &p)
    : Obstacle(s, p), radius(.5 * length),
      halflength(p("-halflength").asDouble(.5 * sim.extents[2])) {
  section = p("-section").asString("circular");
  accel = p("-accel").asBool(false);
  if (accel) {
    if (not bForcedInSimFrame[0]) {
      printf("Warning: Pipe was not set to be forced in x-dir, yet the accel "
             "pattern is active.\n");
    }
    umax = -p("-xvel").asDouble(0.0);
    vmax = -p("-yvel").asDouble(0.0);
    wmax = -p("-zvel").asDouble(0.0);
    tmax = p("-T").asDouble(1.0);
  }
  _init();
}

void Pipe::_init(void) {
  if (sim.verbose)
    printf("Created Pipe with radius %f and halflength %f\n", radius,
           halflength);

  // D-cyl can float around the domain, but does not support rotation. TODO
  bBlockRotation[0] = true;
  bBlockRotation[1] = true;
  bBlockRotation[2] = true;
}

void Pipe::create() {
  const Real h = sim.hmin;
  const PipeObstacle::FillBlocks kernel(radius, halflength, h, position);
  create_base<PipeObstacle::FillBlocks>(kernel);
}

void Pipe::computeVelocities() {
  if (accel) {
    if (sim.time < tmax)
      transVel_imposed[0] = umax * sim.time / tmax;
    else {
      transVel_imposed[0] = umax;
      transVel_imposed[1] = vmax;
      transVel_imposed[2] = wmax;
    }
  }

  Obstacle::computeVelocities();
}

void Pipe::finalize() {
  // this method allows any computation that requires the char function
  // to be computed. E.g. compute the effective center of mass or removing
  // momenta from udef
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

// static constexpr Real EPSILON = std::numeric_limits<Real>::epsilon();

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

////////////////////////////////////////////////////////////
// PLATE FILL BLOCKS
////////////////////////////////////////////////////////////
namespace {
struct PlateFillBlocks : FillBlocksBase<PlateFillBlocks> {
  const Real cx, cy, cz;     // Center.
  const Real nx, ny, nz;     // Normal. NORMALIZED.
  const Real ax, ay, az;     // A-side vector. NORMALIZED.
  const Real bx, by, bz;     // A-side vector. NORMALIZED.
  const Real half_a;         // Half-size in A direction.
  const Real half_b;         // Half-size in B direction.
  const Real half_thickness; // Half-thickess. Edges are rounded.

  Real aabb[3][2]; // Axis-aligned bounding box.

  PlateFillBlocks(Real cx, Real cy, Real cz, Real nx, Real ny, Real nz, Real ax,
                  Real ay, Real az, Real bx, Real by, Real bz, Real half_a,
                  Real half_b, Real half_thickness, Real h);

  // Required by FillBlocksBase.
  bool isTouching(const BlockInfo &, const ScalarBlock &b) const;
  Real signedDistance(Real x, Real y, Real z) const;
};
} // Anonymous namespace.

PlateFillBlocks::PlateFillBlocks(const Real _cx, const Real _cy, const Real _cz,
                                 const Real _nx, const Real _ny, const Real _nz,
                                 const Real _ax, const Real _ay, const Real _az,
                                 const Real _bx, const Real _by, const Real _bz,
                                 const Real _half_a, const Real _half_b,
                                 const Real _half_thickness, const Real h)
    : cx(_cx), cy(_cy), cz(_cz), nx(_nx), ny(_ny), nz(_nz), ax(_ax), ay(_ay),
      az(_az), bx(_bx), by(_by), bz(_bz), half_a(_half_a), half_b(_half_b),
      half_thickness(_half_thickness) {
  using std::fabs;

  // Assert normalized.
  assert(fabs(nx * nx + ny * ny + nz * nz - 1) < (Real)1e-9);
  assert(fabs(ax * ax + ay * ay + az * az - 1) < (Real)1e-9);
  assert(fabs(bx * bx + by * by + bz * bz - 1) < (Real)1e-9);

  // Assert n, a and b are mutually orthogonal.
  assert(fabs(nx * ax + ny * ay + nz * az) < (Real)1e-9);
  assert(fabs(nx * bx + ny * by + nz * bz) < (Real)1e-9);
  assert(fabs(ax * bx + ay * by + az * bz) < (Real)1e-9);

  const Real skin = (2 + SURFDH) * h;
  const Real tx =
      skin + fabs(ax * half_a) + fabs(bx * half_b) + fabs(nx * half_thickness);
  const Real ty =
      skin + fabs(ay * half_a) + fabs(by * half_b) + fabs(ny * half_thickness);
  const Real tz =
      skin + fabs(az * half_a) + fabs(bz * half_b) + fabs(nz * half_thickness);

  aabb[0][0] = cx - tx;
  aabb[0][1] = cx + tx;
  aabb[1][0] = cy - ty;
  aabb[1][1] = cy + ty;
  aabb[2][0] = cz - tz;
  aabb[2][1] = cz + tz;
}

bool PlateFillBlocks::isTouching(const BlockInfo &info,
                                 const ScalarBlock &b) const {
  Real MINP[3], MAXP[3];
  info.pos(MINP, 0, 0, 0);
  info.pos(MAXP, ScalarBlock::sizeX - 1, ScalarBlock::sizeY - 1,
           ScalarBlock::sizeZ - 1);
  return aabb[0][0] <= MAXP[0] && aabb[0][1] >= MINP[0] &&
         aabb[1][0] <= MAXP[1] && aabb[1][1] >= MINP[1] &&
         aabb[2][0] <= MAXP[2] && aabb[2][1] >= MINP[2];
}

Real PlateFillBlocks::signedDistance(const Real x, const Real y,
                                     const Real z) const {
  // Move plane to the center.
  const Real dx = x - cx;
  const Real dy = y - cy;
  const Real dz = z - cz;
  const Real dotn = dx * nx + dy * ny + dz * nz;

  // Project (x, y, z) to the centered plane.
  const Real px = dx - dotn * nx;
  const Real py = dy - dotn * ny;
  const Real pz = dz - dotn * nz;

  // Project into directions a and b.
  const Real dota = px * ax + py * ay + pz * az;
  const Real dotb = px * bx + py * by + pz * bz;

  // Distance to the rectangle edges in the plane coordinate system.
  const Real a = std::fabs(dota) - half_a;
  const Real b = std::fabs(dotb) - half_b;
  const Real n = std::fabs(dotn) - half_thickness;

  if (a <= 0 && b <= 0 && n <= 0) {
    // Inside, return a positive number.
    return -std::min(n, std::min(a, b));
  } else {
    // Outside, return a negative number.
    const Real a0 = std::max((Real)0, a);
    const Real b0 = std::max((Real)0, b);
    const Real n0 = std::max((Real)0, n);
    return -std::sqrt(a0 * a0 + b0 * b0 + n0 * n0);
  }

  // ROUNDED EDGES.
  // return half_thickness - std::sqrt(dotn * dotn + a0 * a0 + b0 * b0);
}

////////////////////////////////////////////////////////////
// PLATE OBSTACLE OPERATOR
////////////////////////////////////////////////////////////

Plate::Plate(SimulationData &s, ArgumentParser &p) : Obstacle(s, p) {
  p.set_strict_mode();
  half_a = (Real)0.5 * p("-a").asDouble();
  half_b = (Real)0.5 * p("-b").asDouble();
  half_thickness = (Real)0.5 * p("-thickness").asDouble();
  p.unset_strict_mode();

  bool has_alpha = p.check("-alpha");
  if (has_alpha) {
    _from_alpha(M_PI / 180.0 * p("-alpha").asDouble());
  } else {
    p.set_strict_mode();
    nx = p("-nx").asDouble();
    ny = p("-ny").asDouble();
    nz = p("-nz").asDouble();
    ax = p("-ax").asDouble();
    ay = p("-ay").asDouble();
    az = p("-az").asDouble();
    p.unset_strict_mode();
  }

  _init();
}

void Plate::_from_alpha(const Real alpha) {
  nx = std::cos(alpha);
  ny = std::sin(alpha);
  nz = 0;
  ax = -std::sin(alpha);
  ay = std::cos(alpha);
  az = 0;
}

void Plate::_init(void) {
  _normalize(&nx, &ny, &nz);
  _normalized_cross(nx, ny, nz, ax, ay, az, &bx, &by, &bz);
  _normalized_cross(bx, by, bz, nx, ny, nz, &ax, &ay, &az);

  // Plateq can float around the domain, but does not support rotation. TODO
  bBlockRotation[0] = true;
  bBlockRotation[1] = true;
  bBlockRotation[2] = true;
}

void Plate::create() {
  const Real h = sim.hmin;
  const PlateFillBlocks K(position[0], position[1], position[2], nx, ny, nz, ax,
                          ay, az, bx, by, bz, half_a, half_b, half_thickness,
                          h);

  create_base<PlateFillBlocks>(K);
}

void Plate::finalize() {
  // this method allows any computation that requires the char function
  // to be computed. E.g. compute the effective center of mass or removing
  // momenta from udef
}

CubismUP_3D_NAMESPACE_END
    //
    //  CubismUP_3D
    //  Copyright (c) 2023 CSE-Lab, ETH Zurich, Switzerland.
    //  Distributed under the terms of the MIT license.
    //
    //  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
    //

    namespace cubismup3d {

  void PoissonSolverAMR::solve() {
    // Algorithm 11 from the paper:
    //"The communication-hiding pipelined BiCGstab method for the parallel
    // solution of large unsymmetric linear systems" by S. Cools, W. Vanroose
    // This
    // is a BiCGstab with less global communication (reductions) that are
    // overlapped with computation.

    // Warning: 'input'  initially contains the RHS of the system!
    // Warning: 'output' initially contains the initial solution guess x0!
    const auto &AxInfo =
        sim.lhsInfo(); // input ->getBlocksInfo(); //will store the LHS result
    const auto &zInfo = sim.presInfo(); // output->getBlocksInfo(); //will store
                                        // the input 'x' when LHS is computed
    const size_t Nblocks = zInfo.size(); // total blocks of this rank
    const int BSX = VectorBlock::sizeX;  // block size in x direction
    const int BSY = VectorBlock::sizeY;  // block size in y direction
    const int BSZ = VectorBlock::sizeZ;  // block size in z direction
    const size_t N =
        BSX * BSY * BSZ * Nblocks; // total number of variables of this rank
    const Real eps = 1e-100; // used in denominators, to not divide by zero
    const Real max_error =
        sim.PoissonErrorTol; // error tolerance for Linf norm of residual
    const Real max_rel_error =
        sim.PoissonErrorTolRel; // relative error tolerance for Linf(r)/Linf(r0)
    const int max_restarts = 100;
    bool serious_breakdown =
        false;            // shows if the solver will restart in this iteration
    bool useXopt = false; //(is almost always true) use the solution that had
                          // the smallest residual
    int restarts = 0;     // count how many restarts have been made
    Real min_norm =
        1e50;          // residual norm (for best solution, see also 'useXopt')
    Real norm_1 = 0.0; // used to decide if the solver will restart
    Real norm_2 = 0.0; // used to decide if the solver will restart
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
    b.resize(N);     // RHS of the system will be stored here
    x_opt.resize(N); // solution with minimum residual

// initialize b,r,x
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      ScalarBlock &__restrict__ rhs = *(ScalarBlock *)AxInfo[i].ptrBlock;
      const ScalarBlock &__restrict__ zz = *(ScalarBlock *)zInfo[i].ptrBlock;

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

    // In what follows, we indicate by (*n*) the n-th step of the algorithm

    //(*2*) r0 = b - A*x0, r0hat = M^{-1}*r0, w0=A*r0hat, w0hat=M^{-1}w0
    _lhs(x, r0);
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      r0[i] = r[i] - r0[i];
      r[i] = r0[i];
    }
    _preconditioner(r0, rhat);
    _lhs(rhat, w);
    _preconditioner(w, what);

    //(*3*) t0=A*w0hat, alpha0 = (r0,r0) / (r0,w0), beta=0
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
      if (verbose)
        std::cout << "[Poisson solver]: initial error norm:" << norm << "\n";
    }
    const Real init_norm = norm;

    //(*4*) for k=0,1,...
    int k;
    for (k = 0; k < 1000; k++) {
      Real qy = 0.0;
      Real yy = 0.0;

      //(*5*),(*6*),...,(*11*)
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
// every 50 iterations we use the residual replacement strategy, to prevent loss
// of accuracy and compute stuff with the exact (not pipelined) versions
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

      //(*12*) begin reduction (q,y),(y,y)
      MPI_Request request;
      Real quantities[7];
      quantities[0] = qy;
      quantities[1] = yy;
      MPI_Iallreduce(MPI_IN_PLACE, &quantities, 2, MPI_Real, MPI_SUM, m_comm,
                     &request);

      //(*13*) computation zhat = M^{-1}*z
      _preconditioner(z, zhat);

      //(*14*) computation v = A*zhat
      _lhs(zhat, v);

      //(*15*) end reduction
      MPI_Waitall(1, &request, MPI_STATUSES_IGNORE);
      qy = quantities[0];
      yy = quantities[1];

      //(*16*) omega = (q,y)/(y,y)
      omega = qy / (yy + eps);

      //(*17*),(*18*),(*19*),(*20*)
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
// every 50 iterations we use the residual replacement strategy, to prevent loss
// of accuracy and compute stuff with the exact (not pipelined) versions
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

      //(*21*) begin reductions
      MPI_Iallreduce(MPI_IN_PLACE, &quantities, 7, MPI_Real, MPI_SUM, m_comm,
                     &request);

      //(*22*) computation what = M^{-1}*w
      _preconditioner(w, what);

      //(*23*) computation t = A*what
      _lhs(what, t);

      //(*24*) end reductions
      MPI_Waitall(1, &request, MPI_STATUSES_IGNORE);
      r0r = quantities[0];
      r0w = quantities[1];
      r0s = quantities[2];
      r0z = quantities[3];
      norm_1 = quantities[4];
      norm_2 = quantities[5];
      norm = std::sqrt(quantities[6]);

      //(*25*)
      beta = alpha / (omega + eps) * r0r / (r0r_prev + eps);

      //(*26*)
      alpha = r0r / (r0w + beta * r0s - beta * omega * r0z);
      Real alphat = 1.0 / (omega + eps) + r0w / (r0r + eps) -
                    beta * omega * r0z / (r0r + eps);
      alphat = 1.0 / (alphat + eps);
      if (std::fabs(alphat) < 10 * std::fabs(alpha))
        alpha = alphat;

      r0r_prev = r0r;
      // Check if restart should be made. If so, current solution estimate is
      // used as an initial guess and solver starts again.
      serious_breakdown = r0r * r0r < 1e-16 * norm_1 * norm_2;
      if (serious_breakdown && restarts < max_restarts) {
        restarts++;
        if (verbose)
          std::cout << "  [Poisson solver]: Restart at iteration: " << k
                    << " norm: " << norm << std::endl;

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
        if (verbose)
          std::cout << "  [Poisson solver]: Converged after " << k
                    << " iterations.\n";
        break;
      }
    }

    if (verbose) {
      std::cout << " Error norm (relative) = " << min_norm << "/" << max_error
                << std::endl;
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
} // namespace cubismup3d

/*
Optimization comments:
  - The innermost loop has to be very simple in order for the compiler to
    optimize it. Temporary accumulators and storage arrays have to be used to
    enable vectorization.

  - In order to vectorize the stencil, the shifted west and east pointers to p
    have to be provided separately, without the compiler knowing that they are
    related to the same buffer.

  - The same would be true for south, north, back and front shifts, but we pad
    the p block not with +/-1 padding but with +/-4, and put a proper offset
    (depending on sizeof(Real)) to have everything nicely aligned with respect
    to the 32B boundary. This was tested only on AVX-256, but should work for
    AVX-512 as well.

  - For correctness, the p pointers must not have __restrict__, since p,
    pW and pE do overlap. (Not important here though, since we removed the
    convergence for loop, see below). All other arrays do have __restrict__, so
    this does not affect vectorization anyway.

  - The outer for loop that repeats the kernel until convergence breaks the
    vectorization of the stencil in gcc, hence it was removed from this file.

  - Putting this loop into another function does not help, since the compiler
    merges the two functions and breaks the vectorization. This can be fixed by
    adding `static __attribute__((noinline))` to the kernel function, but it is
    a bit risky, and doesn't seem to improve the generated code. The cost of
    the bare function call here is about 3ns.

  - Not tested here, but unaligned access can be up to 2x slower than aligned,
    so it is important to ensure alignment.
    https://www.agner.org/optimize/blog/read.php?i=423


Compilation hints:
  - If gcc is used, the Ax--p stencil won't be vectorized unless version 11 or
    later is used.

  - -ffast-math might affect ILP and reductions. Not verified.

  - Changing the order of operations may cause the compiler to produce
    different operation order and hence cause the number of convergence
    iterations to change.

  - To show the assembly, use e.g.
      objdump -dS -Mintel --no-show-raw-insn PoissonSolverAMRKernels.cpp.o >
PoissonSolverAMRKErnels.cpp.lst

  - With gcc 11, it might be necessary to use "-g -gdwarf-4" instead of "-g"
    for objdump to work. For more information look here:
    https://gcc.gnu.org/gcc-11/changes.html


Benchmarks info for Broadwell CPU with AVX2 (256-bit):
  - Computational limit is 2x 256-bit SIMD FMAs per cycle == 16 FLOPs/cycle.

  - Memory limit for L1 cache is 2x256-bit reads and 1x256-bit write per cycle.
    See "Haswell and Broadwell pipeline", section "Read and write bandwidth":
    https://www.agner.org/optimize/microarchitecture.pdf

    These amount to 64B reads and 32B writes per cycle, however we get about
    80% of that, consistent with benchmarks here:
    https://www.agner.org/optimize/blog/read.php?i=423

  - The kernels below are memory bound.
*/

namespace cubismup3d {
namespace poisson_kernels {

// Note: kDivEpsilon is too small for single precision!
static constexpr Real kDivEpsilon = 1e-55;
static constexpr Real kNormRelCriterion = 1e-7;
static constexpr Real kNormAbsCriterion = 1e-16;
static constexpr Real kSqrNormRelCriterion =
    kNormRelCriterion * kNormRelCriterion;
static constexpr Real kSqrNormAbsCriterion =
    kNormAbsCriterion * kNormAbsCriterion;

/*
// Reference non-vectorized implementation of the kernel.
Real kernelPoissonGetZInnerReference(
    PaddedBlock & __restrict__ p,
    Block & __restrict__ Ax,
    Block & __restrict__ r,
    Block & __restrict__ block,
    const Real sqrNorm0,
    const Real rr)
{
  Real a2 = 0;
  for (int iz = 0; iz < NZ; ++iz)
  for (int iy = 0; iy < NY; ++iy)
  for (int ix = 0; ix < NX; ++ix) {
    Ax[iz][iy][ix] = p[iz + 1][iy + 1][ix + xPad - 1]
                   + p[iz + 1][iy + 1][ix + xPad + 1]
                   + p[iz + 1][iy + 0][ix + xPad]
                   + p[iz + 1][iy + 2][ix + xPad]
                   + p[iz + 0][iy + 1][ix + xPad]
                   + p[iz + 2][iy + 1][ix + xPad]
                   - 6 * p[iz + 1][iy + 1][ix + xPad];
    a2 += p[iz + 1][iy + 1][ix + xPad] * Ax[iz][iy][ix];
  }

  const Real a = rr / (a2 + kDivEpsilon);
  Real sqrNorm = 0;
  for (int iz = 0; iz < NZ; ++iz)
  for (int iy = 0; iy < NY; ++iy)
  for (int ix = 0; ix < NX; ++ix) {
    block[iz][iy][ix] += a * p[iz + 1][iy + 1][ix + xPad];
    r[iz][iy][ix] -= a * Ax[iz][iy][ix];
    sqrNorm += r[iz][iy][ix] * r[iz][iy][ix];
  }

  const Real beta = sqrNorm / (rr + kDivEpsilon);
  const Real rrNew = sqrNorm;
  const Real norm = std::sqrt(sqrNorm) / N;

  if (norm / std::sqrt(sqrNorm0) < kNormRelCriterion || norm <
kNormAbsCriterion) return 0;

  for (int iz = 0; iz < NZ; ++iz)
  for (int iy = 0; iy < NY; ++iy)
  for (int ix = 0; ix < NX; ++ix) {
    p[iz + 1][iy + 1][ix + xPad] =
        r[iz][iy][ix] + beta * p[iz + 1][iy + 1][ix + xPad];
  }

  return rrNew;
}
*/

/// Update `r -= a * Ax` and return `sum(r^2)`.
static inline Real subAndSumSqr(Block &__restrict__ r_,
                                const Block &__restrict__ Ax_, Real a) {
  // The block structure is not important here, we can treat it as a contiguous
  // array. However, we group into groups of length 16, to help with ILP and
  // vectorization.
  constexpr int MX = 16;
  constexpr int MY = NX * NY * NZ / MX;
  using SquashedBlock = Real[MY][MX];
  static_assert(NX * NY % MX == 0 && sizeof(Block) == sizeof(SquashedBlock));
  SquashedBlock &__restrict__ r = (SquashedBlock &)r_;
  SquashedBlock &__restrict__ Ax = (SquashedBlock &)Ax_;

  // This kernel reaches neither the compute nor the memory bound.
  // The problem could be high latency of FMA instructions.
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
    // if ((uintptr_t)ptr % align != offset)
    //   throw std::runtime_error("wrong alignment");
    assert((uintptr_t)ptr % align == offset);

    // Works with gcc, clang and icc.
    return (T *)__builtin_assume_aligned(ptr, align, offset);
  } else {
    return ptr; // No alignment assumptions for long double.
  }
}

Real kernelPoissonGetZInner(PaddedBlock &p_, const Real *pW_, const Real *pE_,
                            Block &__restrict__ Ax_, Block &__restrict__ r_,
                            Block &__restrict__ block_, const Real sqrNorm0,
                            const Real rr) {
  PaddedBlock &p = *assumeAligned(&p_, 64, 64 - xPad * sizeof(Real));
  const PaddedBlock &pW =
      *(PaddedBlock *)pW_; // Aligned to 64B + 24 (for doubles).
  const PaddedBlock &pE =
      *(PaddedBlock *)pE_; // Aligned to 64B + 40 (for doubles).
  Block &__restrict__ Ax = *assumeAligned(&Ax_, 64);
  Block &__restrict__ r = *assumeAligned(&r_, 64);
  Block &__restrict__ block = *assumeAligned(&block_, kBlockAlignment);

  // Broadwell: 6.0-6.6 FLOP/cycle, depending probably on array alignments.
  Real a2Partial[NX] = {};
  for (int iz = 0; iz < NZ; ++iz)
    for (int iy = 0; iy < NY; ++iy) {
      // On Broadwell and earlier it might be beneficial to turn some of these
      // a+b additions into FMAs of form 1*a+b, because those CPUs can do 2
      // FMAs/cycle and only 1 ADD/cycle. However, it wouldn't be simple to
      // convience the compiler to do so, and it wouldn't matter from Skylake
      // on. https://www.agner.org/optimize/blog/read.php?i=415

      Real tmpAx[NX];
      for (int ix = 0; ix < NX; ++ix) {
        tmpAx[ix] = pW[iz + 1][iy + 1][ix + xPad] +
                    pE[iz + 1][iy + 1][ix + xPad] -
                    6 * p[iz + 1][iy + 1][ix + xPad];
      }

      // This kernel is memory bound. The compiler should figure out that some
      // loads can be reused between consecutive iy.

      // Merging the following two loops (i.e. to ensure symmetry preservation
      // when there is no -ffast-math) kills vectorization in gcc 11.
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

  // Interleaving this kernel with the next one seems to improve the
  // maximum performance by 5-10% (after fine-tuning MX in the subAndSumSqr
  // part), but it increases the variance a lot so it is not clear whether it
  // is faster on average. For now, keeping it separate.
  for (int iz = 0; iz < NZ; ++iz)
    for (int iy = 0; iy < NY; ++iy)
      for (int ix = 0; ix < NX; ++ix)
        block[iz][iy][ix] += a * p[iz + 1][iy + 1][ix + xPad];

  // Kernel: 2 reads + 1 write + 4 FLOPs/cycle -> should be memory bound.
  // Broadwell: 9.2 FLOP/cycle, 37+18.5 B/cycle -> latency bound?
  // r -= a * Ax, sqrSum = sum(r^2)
  const Real sqrSum = subAndSumSqr(r, Ax, a);

  const Real beta = sqrSum / (rr + kDivEpsilon);
  const Real sqrNorm = (Real)1 / (N * N) * sqrSum;

  if (sqrNorm < kSqrNormRelCriterion * sqrNorm0 ||
      sqrNorm < kSqrNormAbsCriterion)
    return -1.0;

  // Kernel: 2 reads + 1 write + 2 FLOPs per cell -> limit is L1 cache.
  // Broadwell: 6.5 FLOP/cycle, 52+26 B/cycle
  for (int iz = 0; iz < NZ; ++iz)
    for (int iy = 0; iy < NY; ++iy)
      for (int ix = 0; ix < NX; ++ix) {
        p[iz + 1][iy + 1][ix + xPad] =
            r[iz][iy][ix] + beta * p[iz + 1][iy + 1][ix + xPad];
      }

  const Real rrNew = sqrSum;
  return rrNew;
}

void getZImplParallel(const std::vector<cubism::BlockInfo> &vInfo) {
  const size_t Nblocks = vInfo.size();

  // We could enable this, we don't really care about denormals.
  // _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

  // A struct to enforce relative alignment between matrices. The relative
  // alignment of Ax and r MUST NOT be a multiple of 4KB due to cache bank
  // conflicts. See "Haswell and Broadwell pipeline", section
  // "Cache and memory access" here:
  // https://www.agner.org/optimize/microarchitecture.pdf
  struct Tmp {
    // It seems like some offsets with respect to the page boundary of 4KB are
    // faster than the others. (This is accomplished by adding an offset here
    // and using alignas(4096) below). However, this is likely CPU-dependent,
    // so we don't hardcode such fine-tunings here.
    // char offset[0xec0];
    Block r;
    // Ensure p[0+1][0+1][0+xPad] is 64B-aligned for AVX-512 to work.
    char padding1[64 - xPad * sizeof(Real)];
    PaddedBlock p;
    char padding2[xPad * sizeof(Real)];
    Block Ax;
  };
  alignas(64) Tmp tmp{}; // See the kernels cpp file for required alignments.
  Block &r = tmp.r;
  Block &Ax = tmp.Ax;
  PaddedBlock &p = tmp.p;

#pragma omp for
  for (size_t i = 0; i < Nblocks; ++i) {
    static_assert(sizeof(ScalarBlock) == sizeof(Block));
    assert((uintptr_t)vInfo[i].ptrBlock % kBlockAlignment == 0);
    Block &block =
        *(Block *)__builtin_assume_aligned(vInfo[i].ptrBlock, kBlockAlignment);

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
      // rr = kernelPoissonGetZInnerReference(p,Ax, r, block, sqrNorm0, rr);
      rr = kernelPoissonGetZInner(p, pW, pE, Ax, r, block, sqrNorm0, rr);
      if (rr <= 0)
        break;
    }
  }
}

} // namespace poisson_kernels
} // namespace cubismup3d

namespace cubismup3d {

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
} // namespace cubismup3d
//
//  CubismUP_3D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

CubismUP_3D_NAMESPACE_BEGIN

    using CHIMAT = Real[CUP_BLOCK_SIZEZ][CUP_BLOCK_SIZEY][CUP_BLOCK_SIZEX];
using UDEFMAT = Real[CUP_BLOCK_SIZEZ][CUP_BLOCK_SIZEY][CUP_BLOCK_SIZEX][3];

struct KernelDivPressure {
  const SimulationData &sim;
  const StencilInfo stencil = StencilInfo(-1, -1, -1, 2, 2, 2, false, {0});
  const std::vector<BlockInfo> &tmpVInfo = sim.tmpVInfo();
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;

  KernelDivPressure(const SimulationData &s) : sim(s) {}

  void operator()(const ScalarLab &lab, const BlockInfo &info) const {
    VectorBlock &__restrict__ b = (*sim.tmpV)(info.blockID);
    const Real fac = info.h;
    for (int z = 0; z < Nz; ++z)
      for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x)
#ifdef PRESERVE_SYMMETRY
          b(x, y, z).u[0] =
              fac * (ConsistentSum(lab(x + 1, y, z).s + lab(x - 1, y, z).s,
                                   lab(x, y + 1, z).s + lab(x, y - 1, z).s,
                                   lab(x, y, z + 1).s + lab(x, y, z - 1).s) -
                     6.0 * lab(x, y, z).s);
#else
          b(x, y, z).u[0] = fac * (lab(x + 1, y, z).s + lab(x - 1, y, z).s +
                                   lab(x, y + 1, z).s + lab(x, y - 1, z).s +
                                   lab(x, y, z + 1).s + lab(x, y, z - 1).s -
                                   6.0 * lab(x, y, z).s);
#endif

    BlockCase<VectorBlock> *tempCase =
        (BlockCase<VectorBlock> *)(tmpVInfo[info.blockID].auxiliary);

    if (tempCase == nullptr)
      return; // no flux corrections needed for this block

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
  const std::vector<BlockInfo> &lhsInfo = sim.lhsInfo();
  const std::vector<BlockInfo> &chiInfo = sim.chiInfo();
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;

  KernelPressureRHS(SimulationData &s, const Real a_dt) : sim(s), dt(a_dt) {}

  void operator()(const VectorLab &lab, const VectorLab &uDefLab,
                  const BlockInfo &info, const BlockInfo &info2) const {
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
#ifdef PRESERVE_SYMMETRY
            p(x, y, z).s =
                fac * ConsistentSum(LE.u[0] - LW.u[0], LN.u[1] - LS.u[1],
                                    LB.u[2] - LF.u[2]);
#else
            p(x, y, z).s = fac * (LE.u[0] - LW.u[0] + LN.u[1] - LS.u[1] +
                                  LB.u[2] - LF.u[2]);
#endif
          }
          {
            const VectorElement &LW = uDefLab(x - 1, y, z),
                                &LE = uDefLab(x + 1, y, z);
            const VectorElement &LS = uDefLab(x, y - 1, z),
                                &LN = uDefLab(x, y + 1, z);
            const VectorElement &LF = uDefLab(x, y, z - 1),
                                &LB = uDefLab(x, y, z + 1);
#ifdef PRESERVE_SYMMETRY
            const Real divUs = ConsistentSum(
                LE.u[0] - LW.u[0], LN.u[1] - LS.u[1], LB.u[2] - LF.u[2]);
#else
            const Real divUs =
                LE.u[0] - LW.u[0] + LN.u[1] - LS.u[1] + LB.u[2] - LF.u[2];
#endif
            p(x, y, z).s += -c(x, y, z).s * fac * divUs;
          }
        }

    BlockCase<ScalarBlock> *tempCase =
        (BlockCase<ScalarBlock> *)(lhsInfo[info2.blockID].auxiliary);

    if (tempCase == nullptr)
      return; // no flux corrections needed for this block

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

/// Add obstacle's udef to tmpV
static void kernelUpdateTmpV(SimulationData &sim) {
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;
  const std::vector<BlockInfo> &chiInfo = sim.chiInfo();
#pragma omp parallel
  {
    for (const auto &obstacle : sim.obstacle_vector->getObstacleVector()) {
      const auto &obstblocks = obstacle->getObstacleBlocks();
#pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < chiInfo.size(); ++i) {
        const BlockInfo &info = chiInfo[i];
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
              // What if multiple obstacles share a block? Do not write udef
              // onto grid if CHI stored on the grid is greater than obst's CHI.
              if (c(x, y, z).s > CHI[z][y][x])
                continue;
              // What if two obstacles overlap? Let's plus equal. After all here
              // we are computing divUs, maybe one obstacle has divUs 0. We will
              // need a repulsion term of the velocity at some point in the
              // code.
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
  const std::vector<BlockInfo> &tmpVInfo = sim.tmpVInfo();
  const Real dt;
  const int Nx = VectorBlock::sizeX;
  const int Ny = VectorBlock::sizeY;
  const int Nz = VectorBlock::sizeZ;

  KernelGradP(SimulationData &s, const Real a_dt) : sim(s), dt(a_dt) {}

  ~KernelGradP() {}

  void operator()(const ScalarLab &lab, const BlockInfo &info) const {
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
      return; // no flux corrections needed for this block

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
  const std::vector<BlockInfo> &presInfo = sim.presInfo();

  pOld.resize(Nx * Ny * Nz * presInfo.size());

  // 1. Compute pRHS
  {
// pOld -> store p
// p -> store RHS
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

    // place Udef on tmpV
    if (sim.obstacle_vector->nObstacles() > 0)
      kernelUpdateTmpV(sim);

    KernelPressureRHS K(sim, dt);
    compute<KernelPressureRHS, VectorGrid, VectorLab, VectorGrid, VectorLab,
            ScalarGrid>(K, *sim.vel, *sim.tmpV, true, sim.lhs);
  }

  ////2. Add div(p_old) to rhs and set initial guess phi = 0, i.e. p^{n+1}=p^{n}
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
  } else // just set initial guess phi = 0, i.e. p^{n+1}=p^{n}
  {
#pragma omp parallel for
    for (size_t i = 0; i < presInfo.size(); i++) {
      ScalarBlock &p = (*sim.pres)(i);
      p.clear();
    }
  }

  // Solve the Poisson equation and use pressure to perform
  // pressure projection of the velocity field.

  // The rhs of the linear system is contained in sim.lhs
  // The initial guess is contained in sim.pres
  // Here we solve for phi := p^{n+1}-p^{n} if step < step_2nd_start
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

  const std::vector<BlockInfo> &velInfo = sim.velInfo();

  if (sim.step > sim.step_2nd_start) // recover p^{n+1} = phi + p^{n}
  {
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      ScalarBlock &p = (*sim.pres)(i);
      for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
          for (int x = 0; x < Nx; ++x)
            p(x, y, z).s += pOld[i * Nx * Ny * Nz + z * Ny * Nx + y * Nx + x];
    }
  }

  // Compute grad(P) and put it to the vector tmpV
  compute<ScalarLab>(KernelGradP(sim, dt), sim.pres, sim.tmpV);

// Perform the projection and set u^{n+1} = u - grad(P)*dt
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

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

std::shared_ptr<Simulation>
createSimulation(const MPI_Comm comm, const std::vector<std::string> &argv) {
  std::vector<char *> cargv(argv.size() + 1);
  char cmd[] = "prg";
  cargv[0] = cmd;
  for (size_t i = 0; i < argv.size(); ++i)
    cargv[i + 1] = const_cast<char *>(argv[i].data());
  return std::make_shared<Simulation>((int)cargv.size(), cargv.data(), comm);
}

Simulation::Simulation(int argc, char **argv, MPI_Comm comm)
    : parser(argc, argv), sim(comm, parser) {
  if (sim.verbose) {
#pragma omp parallel
    {
      int numThreads = omp_get_num_threads();
      int size;
      MPI_Comm_size(comm, &size);
#pragma omp master
      std::cout << "[CUP3D] Running with " << size << " rank(s) and "
                << numThreads << " thread(s)." << std::endl;
    }
  }
}

void Simulation::init() {
  // Make sure given arguments are valid
  if (sim.verbose)
    std::cout << "[CUP3D] Parsing Arguments.. " << std::endl;
  sim._preprocessArguments();

  // Setup and Initialize Grid
  if (sim.verbose)
    std::cout << "[CUP3D] Allocating Grid.. " << std::endl;
  setupGrid();

  // Setup Computational Pipeline
  if (sim.verbose)
    std::cout << "[CUP3D] Creating Computational Pipeline.. " << std::endl;
  setupOperators();

  // Initalize Obstacles
  if (sim.verbose)
    std::cout << "[CUP3D] Initializing Obstacles.. " << std::endl;
  sim.obstacle_vector = new ObstacleVector(sim);
  ObstacleFactory(sim).addObstacles(parser);

  // CreateObstacles
  if (sim.verbose)
    std::cout << "[CUP3D] Creating Obstacles.. " << std::endl;
  (*sim.pipeline[0])(0);

  // Initialize Flow Field
  if (sim.verbose)
    std::cout << "[CUP3D] Initializing Flow Field.. " << std::endl;

  FILE *fField = fopen("field.restart", "r");
  if (fField == NULL) {
    if (sim.verbose)
      std::cout << "[CUP3D] Performing Initial Refinement of Grid.. "
                << std::endl;
    initialGridRefinement();
  } else {
    fclose(fField);
    deserialize();
  }
}

void Simulation::initialGridRefinement() {
  // CreateObstacles and set initial conditions
  (*sim.pipeline[0])(0);
  _ic();

  const int lmax = sim.StaticObstacles ? sim.levelMax : 3 * sim.levelMax;
  for (int l = 0; l < lmax; l++) {
    if (sim.verbose)
      std::cout << "[CUP3D] - refinement " << l << "/" << lmax - 1 << std::endl;

    // Refinement or compression of Grid
    adaptMesh();

    // set initial conditions again. If this is not done, we start with the
    // refined (interpolated) version of the ic, which is less accurate
    (*sim.pipeline[0])(0);
    _ic();
  }
}

void Simulation::adaptMesh() {
  sim.startProfiler("Mesh refinement");

  computeVorticity();
  compute<ScalarLab>(GradChiOnTmp(sim), sim.chi);

  sim.tmpV_amr->Tag();
  sim.lhs_amr->TagLike(sim.tmpVInfo());
  sim.vel_amr->TagLike(sim.tmpVInfo());
  sim.chi_amr->TagLike(sim.tmpVInfo());
  sim.pres_amr->TagLike(sim.tmpVInfo());
  sim.chi_amr->Adapt(sim.time, sim.verbose, true);
  sim.lhs_amr->Adapt(sim.time, false, true);
  sim.tmpV_amr->Adapt(sim.time, false, true);
  sim.pres_amr->Adapt(sim.time, false, false);
  sim.vel_amr->Adapt(sim.time, false, false);

  sim.MeshChanged = sim.pres->UpdateFluxCorrection;

  sim.stopProfiler();
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
  // Refine/compress only according to chi field for now
  sim.chi_amr = new ScalarAMR(*(sim.chi), sim.Rtol, sim.Ctol);
  sim.lhs_amr = new ScalarAMR(*(sim.lhs), sim.Rtol, sim.Ctol);
  sim.pres_amr = new ScalarAMR(*(sim.pres), sim.Rtol, sim.Ctol);
  sim.vel_amr = new VectorAMR(*(sim.vel), sim.Rtol, sim.Ctol);
  sim.tmpV_amr = new VectorAMR(*(sim.tmpV), sim.Rtol, sim.Ctol);
}

void Simulation::setupOperators() {
  // Creates the char function, sdf, and def vel for all obstacles at the curr
  // timestep. At this point we do NOT know the translation and rot vel of the
  // obstacles. We need to solve implicit system when the pre-penalization vel
  // is finalized on the grid.
  // Here we also compute obstacles' centres of mass which are computed from
  // the char func on the grid. This is different from "position" which is
  // the quantity that is advected and used to construct shape.
  sim.pipeline.push_back(std::make_shared<CreateObstacles>(sim));

  // Performs:
  // \tilde{u} = u_t + \delta t (\nu \nabla^2 u_t - (u_t \cdot \nabla) u_t )
  if (sim.implicitDiffusion)
    sim.pipeline.push_back(std::make_shared<AdvectionDiffusionImplicit>(sim));
  else
    sim.pipeline.push_back(std::make_shared<AdvectionDiffusion>(sim));

  // Apply pressure gradient to drive flow
  if (sim.uMax_forced > 0) {
    if (sim.bFixMassFlux) // Fix mass flux
      sim.pipeline.push_back(std::make_shared<FixMassFlux>(sim));
    else // apply uniform gradient
      sim.pipeline.push_back(std::make_shared<ExternalForcing>(sim));
  }

  // Update obstacle velocities and penalize velocity
  sim.pipeline.push_back(std::make_shared<UpdateObstacles>(sim));
  sim.pipeline.push_back(std::make_shared<Penalization>(sim));

  // Places Udef on the grid and computes the RHS of the Poisson Eq
  // overwrites tmpU, tmpV, tmpW and pressure solver's RHS. Then,
  // Solves the Poisson Eq to get the pressure and finalizes the velocity
  sim.pipeline.push_back(std::make_shared<PressureProjection>(sim));

  // With finalized velocity and pressure, compute forces and dissipation
  sim.pipeline.push_back(std::make_shared<ComputeForces>(sim));
  sim.pipeline.push_back(std::make_shared<ComputeDissipation>(sim));

  // sim.pipeline.push_back(std::make_shared<ComputeDivergence>(sim));
  if (sim.rank == 0) {
    printf("[CUP3D] Operator ordering:\n");
    for (size_t c = 0; c < sim.pipeline.size(); c++)
      printf("\t - %s\n", sim.pipeline[c]->getName().c_str());
  }
}

void Simulation::serialize(const std::string append) {
  sim.startProfiler("DumpHDF5_MPI");

  {
    logger.flush();
    std::stringstream name;
    name << "restart_" << std::setfill('0') << std::setw(9) << sim.step;
    DumpHDF5_MPI<StreamerScalar, Real>(*sim.pres, sim.time,
                                       "pres_" + name.str(),
                                       sim.path4serialization, false);
    DumpHDF5_MPI<StreamerVector, Real>(*sim.vel, sim.time, "vel_" + name.str(),
                                       sim.path4serialization, false);
    sim.writeRestartFiles();
  }

  std::stringstream name;
  if (append == "")
    name << "_";
  else
    name << append;
  name << std::setfill('0') << std::setw(9) << sim.step;

  if (sim.dumpOmega || sim.dumpOmegaX || sim.dumpOmegaY || sim.dumpOmegaZ)
    computeVorticity();

  // dump multi-block datasets with scalar quantities or magnitude of vector
  // quantities
  if (sim.dumpP)
    DumpHDF5_MPI2<cubism::StreamerScalar, Real, ScalarGrid>(
        *sim.pres, sim.time, "pres" + name.str(), sim.path4serialization);
  if (sim.dumpChi)
    DumpHDF5_MPI2<cubism::StreamerScalar, Real, ScalarGrid>(
        *sim.chi, sim.time, "chi" + name.str(), sim.path4serialization);
  if (sim.dumpOmega)
    DumpHDF5_MPI2<cubism::StreamerVector, Real, VectorGrid>(
        *sim.tmpV, sim.time, "tmp" + name.str(), sim.path4serialization);
  if (sim.dumpVelocity)
    DumpHDF5_MPI2<cubism::StreamerVector, Real, VectorGrid>(
        *sim.vel, sim.time, "vel" + name.str(), sim.path4serialization);

  // dump components of vectors
  if (sim.dumpOmegaX)
    DumpHDF5_MPI2<StreamerVectorX, Real, VectorGrid>(
        *sim.tmpV, sim.time, "tmpX" + name.str(), sim.path4serialization);
  if (sim.dumpOmegaY)
    DumpHDF5_MPI2<StreamerVectorY, Real, VectorGrid>(
        *sim.tmpV, sim.time, "tmpY" + name.str(), sim.path4serialization);
  if (sim.dumpOmegaZ)
    DumpHDF5_MPI2<StreamerVectorZ, Real, VectorGrid>(
        *sim.tmpV, sim.time, "tmpZ" + name.str(), sim.path4serialization);
  if (sim.dumpVelocityX)
    DumpHDF5_MPI2<StreamerVectorX, Real, VectorGrid>(
        *sim.vel, sim.time, "velX" + name.str(), sim.path4serialization);
  if (sim.dumpVelocityY)
    DumpHDF5_MPI2<StreamerVectorY, Real, VectorGrid>(
        *sim.vel, sim.time, "velY" + name.str(), sim.path4serialization);
  if (sim.dumpVelocityZ)
    DumpHDF5_MPI2<StreamerVectorZ, Real, VectorGrid>(
        *sim.vel, sim.time, "velZ" + name.str(), sim.path4serialization);

  sim.stopProfiler();
}

void Simulation::deserialize() {
  sim.readRestartFiles();
  std::stringstream ss;
  ss << "restart_" << std::setfill('0') << std::setw(9) << sim.step;

  const std::vector<BlockInfo> &chiInfo = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo> &velInfo = sim.vel->getBlocksInfo();
  const std::vector<BlockInfo> &tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo> &lhsInfo = sim.lhs->getBlocksInfo();

  // The only field that is needed for restarting is velocity. Chi is derived
  // from the files we read for obstacles. Here we also read pres so that the
  // Poisson solver has the same initial guess, which in turn leads to restarted
  // simulations having the exact same result as non-restarted ones (we also
  // read pres because we need to read at least one ScalarGrid, see hack below).
  ReadHDF5_MPI<StreamerVector, Real>(*(sim.vel), "vel_" + ss.str(),
                                     sim.path4serialization);
  ReadHDF5_MPI<StreamerScalar, Real>(*(sim.pres), "pres_" + ss.str(),
                                     sim.path4serialization);

  // hack: need to "read" the other grids too, so that the mesh is the same for
  // every grid. So we read VectorGrids from "vel" and ScalarGrids from "pres".
  // We don't care about the grid point values (those are set to zero below), we
  // only care about the grid structure, i.e. refinement levels etc.
  ReadHDF5_MPI<StreamerScalar, Real>(*(sim.chi), "pres_" + ss.str(),
                                     sim.path4serialization);
  ReadHDF5_MPI<StreamerScalar, Real>(*(sim.lhs), "pres_" + ss.str(),
                                     sim.path4serialization);
  ReadHDF5_MPI<StreamerVector, Real>(*(sim.tmpV), "vel_" + ss.str(),
                                     sim.path4serialization);
#pragma omp parallel for
  for (size_t i = 0; i < velInfo.size(); i++) {
    ScalarBlock &CHI = *(ScalarBlock *)chiInfo[i].ptrBlock;
    CHI.clear();
    ScalarBlock &LHS = *(ScalarBlock *)lhsInfo[i].ptrBlock;
    LHS.clear();
    VectorBlock &TMPV = *(VectorBlock *)tmpVInfo[i].ptrBlock;
    TMPV.clear();
  }
  (*sim.pipeline[0])(0);
  sim.readRestartFiles();
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
    serialize(); // save last timestep before aborting
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
    fflush(0);
    MPI_Abort(sim.comm, 1);
  }

  // if DLM>0, adapt lambda such that penal term is independent of time step
  if (sim.DLM > 0)
    sim.lambda = sim.DLM / sim.dt;

  if (sim.rank == 0) {
    printf("==================================================================="
           "====\n");
    printf("[CUP3D] step: %d, time: %f, dt: %.2e, uinf: {%f %f %f}, maxU:%f, "
           "minH:%f, CFL:%.2e, lambda:%.2e, collision?:%d, blocks:%zu\n",
           sim.step, sim.time, sim.dt, sim.uinf[0], sim.uinf[1], sim.uinf[2],
           sim.uMax_measured, hMin, CFL, sim.lambda, sim.bCollision,
           sim.velInfo().size());
  }

  if (sim.step > sim.step_2nd_start) {
    const Real a = dt_old;
    const Real b = sim.dt;
    const Real c1 = -(a + b) / (a * b);
    const Real c2 = b / (a + b) / a;
    sim.coefU[0] = -b * (c1 + c2);
    sim.coefU[1] = b * c1;
    sim.coefU[2] = b * c2;
    // sim.coefU[0] = 1.5;
    // sim.coefU[1] = -2.0;
    // sim.coefU[2] = 0.5;
  }
  return sim.dt;
}

bool Simulation::advance(const Real dt) {
  const bool bDumpFreq =
      (sim.saveFreq > 0 && (sim.step + 1) % sim.saveFreq == 0);
  const bool bDumpTime =
      (sim.dumpTime > 0 && (sim.time + dt) > sim.nextSaveTime);
  if (bDumpTime)
    sim.nextSaveTime += sim.dumpTime;
  sim.bDump = (bDumpFreq || bDumpTime);

  // The mesh be adapted before objects are placed on grid
  if (sim.step % 20 == 0 || sim.step < 10)
    adaptMesh();

  for (size_t c = 0; c < sim.pipeline.size(); c++) {
    sim.startProfiler(sim.pipeline[c]->getName());
    (*sim.pipeline[c])(dt);
    sim.stopProfiler();
  }
  sim.step++;
  sim.time += dt;

  if (sim.bDump)
    serialize();

  if (sim.rank == 0 && sim.freqProfiler > 0 && sim.step % sim.freqProfiler == 0)
    sim.printResetProfiler();

  if ((sim.endTime > 0 && sim.time > sim.endTime) ||
      (sim.nsteps != 0 && sim.step >= sim.nsteps)) {
    if (sim.verbose) {
      sim.printResetProfiler();
      std::cout << "Finished at time " << sim.time << " in " << sim.step
                << " steps.\n";
    }
    return true; // Finished.
  }

  return false; // Not yet finished.
}

void Simulation::computeVorticity() {
  ComputeVorticity findOmega(sim);
  findOmega(0);
}

void Simulation::insertOperator(std::shared_ptr<Operator> op) {
  sim.pipeline.push_back(std::move(op));
}

CubismUP_3D_NAMESPACE_END
    //
    //  CubismUP_3D
    //  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
    //  Distributed under the terms of the MIT license.
    //

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

BCflag cubismBCX;
BCflag cubismBCY;
BCflag cubismBCZ;
SimulationData::SimulationData(MPI_Comm mpicomm, ArgumentParser &parser)
    : comm(mpicomm) {
  // Initialize MPI related variables
  MPI_Comm_rank(comm, &rank);

  // Print parser content
  if (rank == 0)
    parser.print_args();

  // ========== PARSE ARGUMENTS ==========

  // BLOCKS PER DIMENSION
  bpdx = parser("-bpdx").asInt();
  bpdy = parser("-bpdy").asInt();
  bpdz = parser("-bpdz").asInt();

  // AMR SETTINGS
  levelMax = parser("-levelMax").asInt();
  levelStart = parser("-levelStart").asInt(levelMax - 1);
  Rtol = parser("-Rtol").asDouble();
  Ctol = parser("-Ctol").asDouble();
  levelMaxVorticity = parser("-levelMaxVorticity").asInt(levelMax);
  StaticObstacles = parser("-StaticObstacles").asBool(false);

  // SIMULATION DOMAIN
  extents[0] = parser("extentx").asDouble(0);
  extents[1] = parser("extenty").asDouble(0);
  extents[2] = parser("extentz").asDouble(0);
  if (extents[0] + extents[1] + extents[2] < 1e-21)
    extents[0] = parser("extent").asDouble(1);

  // SPEED OF FRAME OF REFERENCE
  uinf[0] = parser("-uinfx").asDouble(0.0);
  uinf[1] = parser("-uinfy").asDouble(0.0);
  uinf[2] = parser("-uinfz").asDouble(0.0);

  // TIMESTEPPING
  CFL = parser("-CFL").asDouble(.1);
  dt = parser("-dt").asDouble(0);
  rampup = parser("-rampup").asInt(100); // number of dt ramp-up steps
  nsteps = parser("-nsteps").asInt(0);   // 0 to disable this stopping critera.
  endTime = parser("-tend").asDouble(0); // 0 to disable this stopping critera.
  step_2nd_start = 2;

  // FLOW
  nu = parser("-nu").asDouble();

  // IC
  initCond = parser("-initCond").asString("zero");

  // SPEED FOR CHANNEL FLOW
  uMax_forced = parser("-uMax_forced").asDouble(0.0);
  bFixMassFlux = parser("-bFixMassFlux").asBool(false);

  // PENALIZATION
  bImplicitPenalization = parser("-implicitPenalization").asBool(true);
  lambda = parser("-lambda").asDouble(1e6);
  DLM = parser("-use-dlm").asDouble(0);

  // DISSIPATION DIAGNOSTIC
  freqDiagnostics = parser("-freqDiagnostics").asInt(100);

  // PROFILER
  freqProfiler = parser("-freqProfiler").asInt(0);

  // POISSON SOLVER
  PoissonErrorTol = parser("-poissonTol").asDouble(1e-6); // absolute error
  PoissonErrorTolRel =
      parser("-poissonTolRel").asDouble(1e-4);           // relative error
  bMeanConstraint = parser("-bMeanConstraint").asInt(1); // zero mean constraint
  poissonSolver = parser("-poissonSolver").asString("iterative");

  // IMPLICIT DIFFUSION SOLVER
  implicitDiffusion = parser("-implicitDiffusion").asBool(false);
  DiffusionErrorTol = parser("-diffusionTol").asDouble(1e-6); // absolute error
  DiffusionErrorTolRel =
      parser("diffusionTolRel").asDouble(1e-4); // relative error

  uMax_allowed = parser("-umax").asDouble(10.0);

  // BOUNDARY CONDITIONS
  // accepted periodic, freespace or wall
  std::string BC_x = parser("-BC_x").asString("freespace");
  std::string BC_y = parser("-BC_y").asString("freespace");
  std::string BC_z = parser("-BC_z").asString("freespace");
  BCx_flag = string2BCflag(BC_x);
  BCy_flag = string2BCflag(BC_y);
  BCz_flag = string2BCflag(BC_z);

  cubismBCX = BCx_flag;
  cubismBCY = BCy_flag;
  cubismBCZ = BCz_flag;

  // OUTPUT
  muteAll = parser("-muteAll").asInt(0);
  verbose = muteAll ? false : parser("-verbose").asInt(1) && rank == 0;
  int dumpFreq = parser("-fdump").asDouble(
      0); // dumpFreq==0 means dump freq (in #steps) is not active
  dumpTime = parser("-tdump").asDouble(
      0.0); // dumpTime==0 means dump freq (in time)   is not active
  saveFreq = parser("-fsave").asInt(
      0); // dumpFreq==0 means dump freq (in #steps) is not active

  // TEMP: Removed distinction saving-dumping. Backward compatibility:
  if (saveFreq <= 0 && dumpFreq > 0)
    saveFreq = dumpFreq;
  path4serialization = parser("-serialization").asString("./");

  // Dumping
  dumpChi = parser("-dumpChi").asBool(true);
  dumpOmega = parser("-dumpOmega").asBool(true);
  dumpP = parser("-dumpP").asBool(false);
  dumpOmegaX = parser("-dumpOmegaX").asBool(false);
  dumpOmegaY = parser("-dumpOmegaY").asBool(false);
  dumpOmegaZ = parser("-dumpOmegaZ").asBool(false);
  dumpVelocity = parser("-dumpVelocity").asBool(false);
  dumpVelocityX = parser("-dumpVelocityX").asBool(false);
  dumpVelocityY = parser("-dumpVelocityY").asBool(false);
  dumpVelocityZ = parser("-dumpVelocityZ").asBool(false);
}

void SimulationData::_preprocessArguments() {
  assert(profiler == nullptr); // This should not be possible at all.
  profiler = new cubism::Profiler();
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
  delete profiler;
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

void SimulationData::startProfiler(std::string name) const {
  profiler->push_start(name);
}
void SimulationData::stopProfiler() const { profiler->pop_stop(); }
void SimulationData::printResetProfiler() {
  profiler->printSummary();
  profiler->reset();
}

void SimulationData::writeRestartFiles() {
  // write restart file for field
  if (rank == 0) {
    std::stringstream ssR;
    ssR << path4serialization + "/field.restart";
    FILE *fField = fopen(ssR.str().c_str(), "w");
    if (fField == NULL) {
      printf("Could not write %s. Aborting...\n", "field.restart");
      fflush(0);
      abort();
    }
    assert(fField != NULL);
    fprintf(fField, "time: %20.20e\n", (double)time);
    fprintf(fField, "stepid: %d\n", step);
    fprintf(fField, "uinfx: %20.20e\n", (double)uinf[0]);
    fprintf(fField, "uinfy: %20.20e\n", (double)uinf[1]);
    fprintf(fField, "uinfz: %20.20e\n", (double)uinf[2]);
    fprintf(fField, "dt: %20.20e\n", (double)dt);
    fclose(fField);
  }

  // write restart file for shapes
  int size;
  MPI_Comm_size(comm, &size);
  const size_t tasks = obstacle_vector->nObstacles();
  size_t my_share = tasks / size;
  if (tasks % size != 0 && rank == size - 1) // last rank gets what's left
  {
    my_share += tasks % size;
  }
  const size_t my_start = rank * (tasks / size);
  const size_t my_end = my_start + my_share;

#pragma omp parallel for schedule(static, 1)
  for (size_t j = my_start; j < my_end; j++) {
    auto &shape = obstacle_vector->getObstacleVector()[j];
    std::stringstream ssR;
    ssR << path4serialization + "/shape_" << shape->obstacleID << ".restart";
    FILE *fShape = fopen(ssR.str().c_str(), "w");
    if (fShape == NULL) {
      printf("Could not write %s. Aborting...\n", ssR.str().c_str());
      fflush(0);
      abort();
    }
    shape->saveRestart(fShape);
    fclose(fShape);
  }
}

void SimulationData::readRestartFiles() {
  // read restart file for field
  FILE *fField = fopen("field.restart", "r");
  if (fField == NULL) {
    printf("Could not read %s. Aborting...\n", "field.restart");
    fflush(0);
    abort();
  }
  assert(fField != NULL);
  if (rank == 0 && verbose)
    printf("Reading %s...\n", "field.restart");
  bool ret = true;
  double in_time, in_uinfx, in_uinfy, in_uinfz, in_dt;
  ret = ret && 1 == fscanf(fField, "time: %le\n", &in_time);
  ret = ret && 1 == fscanf(fField, "stepid: %d\n", &step);
  ret = ret && 1 == fscanf(fField, "uinfx: %le\n", &in_uinfx);
  ret = ret && 1 == fscanf(fField, "uinfy: %le\n", &in_uinfy);
  ret = ret && 1 == fscanf(fField, "uinfz: %le\n", &in_uinfz);
  ret = ret && 1 == fscanf(fField, "dt: %le\n", &in_dt);
  time = (Real)in_time;
  uinf[0] = (Real)in_uinfx;
  uinf[1] = (Real)in_uinfy;
  uinf[2] = (Real)in_uinfz;
  dt = (Real)in_dt;
  fclose(fField);
  if ((not ret) || step < 0 || time < 0) {
    printf("Error reading restart file. Aborting...\n");
    fflush(0);
    abort();
  }
  if (rank == 0 && verbose)
    printf("Restarting flow.. time: %le, stepid: %d, uinfx: %le, uinfy: %le, "
           "uinfz: %le\n",
           (double)time, step, (double)uinf[0], (double)uinf[1],
           (double)uinf[2]);
  nextSaveTime = time + dumpTime;

  // read restart file for shapes
  for (auto &shape : obstacle_vector->getObstacleVector()) {
    std::stringstream ssR;
    ssR << "shape_" << shape->obstacleID << ".restart";
    FILE *fShape = fopen(ssR.str().c_str(), "r");
    if (fShape == NULL) {
      printf("Could not read %s. Aborting...\n", ssR.str().c_str());
      fflush(0);
      abort();
    }
    if (rank == 0 && verbose)
      printf("Reading %s...\n", ssR.str().c_str());
    shape->loadRestart(fShape);
    fclose(fShape);
  }
}

CubismUP_3D_NAMESPACE_END
    //
    //  CubismUP_3D
    //  Copyright (c) 2023 CSE-Lab, ETH Zurich, Switzerland.
    //  Distributed under the terms of the MIT license.
    //

    using namespace cubism;

CubismUP_3D_NAMESPACE_BEGIN

    struct GradScalarOnTmpV {
  GradScalarOnTmpV(const SimulationData &s) : sim(s) {}
  const SimulationData &sim;
  const StencilInfo stencil{-1, -1, -1, 2, 2, 2, false, {0}};
  const std::vector<cubism::BlockInfo> &tmpVInfo = sim.tmpV->getBlocksInfo();
  void operator()(ScalarLab &lab, const BlockInfo &info) const {
    auto &__restrict__ TMPV = *(VectorBlock *)tmpVInfo[info.blockID].ptrBlock;
    const Real ih = 0.5 / info.h;
    for (int z = 0; z < ScalarBlock::sizeZ; ++z)
      for (int y = 0; y < ScalarBlock::sizeY; ++y)
        for (int x = 0; x < ScalarBlock::sizeX; ++x) {
          TMPV(x, y, z).u[0] = ih * (lab(x + 1, y, z).s - lab(x - 1, y, z).s);
          TMPV(x, y, z).u[1] = ih * (lab(x, y + 1, z).s - lab(x, y - 1, z).s);
          TMPV(x, y, z).u[2] = ih * (lab(x, y, z + 1).s - lab(x, y, z - 1).s);
        }
  }
};

SmartNaca::SmartNaca(SimulationData &s, ArgumentParser &p)
    : Naca(s, p), Nactuators(p("-Nactuators").asInt(2)),
      actuator_ds(p("-actuatords").asDouble(0.05)),
      thickness(p("-tRatio").asDouble(0.12)) {
  actuators.resize(Nactuators, 0.);
  actuatorSchedulers.resize(Nactuators);
  actuators_prev_value.resize(Nactuators);
  actuators_next_value.resize(Nactuators);
}

void SmartNaca::finalize() {
  const Real *const rS = myFish->rS;
  Real *const rX = myFish->rX;
  Real *const rY = myFish->rY;
  Real *const rZ = myFish->rZ;
  Real *const norX = myFish->norX;
  Real *const norY = myFish->norY;
  Real *const norZ = myFish->norZ;
  Real *const binX = myFish->binX;
  Real *const binY = myFish->binY;
  Real *const binZ = myFish->binZ;
  const Real *const width = myFish->width;
  PutFishOnBlocks putfish(myFish, position, quaternion);
#pragma omp parallel for
  for (int ss = 0; ss < myFish->Nm; ss++) {
    Real x[3] = {rX[ss], rY[ss], rZ[ss]};
    Real n[3] = {norX[ss], norY[ss], norZ[ss]};
    Real b[3] = {binX[ss], binY[ss], binZ[ss]};
    putfish.changeToComputationalFrame(x);
    putfish.changeVelocityToComputationalFrame(n);
    putfish.changeVelocityToComputationalFrame(b);
    rX[ss] = x[0];
    rY[ss] = x[1];
    rZ[ss] = x[2];
    norX[ss] = n[0];
    norY[ss] = n[1];
    norZ[ss] = n[2];
    binX[ss] = b[0];
    binY[ss] = b[1];
    binZ[ss] = b[2];
  }

#if 0
  //dummy actuator values for testing
  static bool visited = false;
  if (sim.time > 2.0 && visited == false)
  {
    visited = true;
    std::vector<Real> q(actuators.size());
    for (int i = 0 ; i < (int)actuators.size(); i ++) q[i] = 0.25*(2*(i+1)%2-1);
    q[0] = 0.5;
    q[1] = -0.25;
    act(q,0);
  }
#endif

  const Real transition_duration = 1.0;
  Real tot = 0.0;
  for (size_t idx = 0; idx < actuators.size(); idx++) {
    Real dummy;
    actuatorSchedulers[idx].transition(
        sim.time, t_change, t_change + transition_duration,
        actuators_prev_value[idx], actuators_next_value[idx]);
    actuatorSchedulers[idx].gimmeValues(sim.time, actuators[idx], dummy);
    tot += std::fabs(actuators[idx]);
  }
  const double cd =
      force[0] / (0.5 * transVel[0] * transVel[0] * thickness * thickness);
  fx_integral += -std::fabs(cd) * sim.dt;
  if (tot < 1e-21)
    return;

  // Compute gradient of chi and of signed-distance-function here.
  // Used later for the actuators
  const std::vector<cubism::BlockInfo> &tmpInfo = sim.lhs->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &tmpVInfo = sim.tmpV->getBlocksInfo();
  const size_t Nblocks = tmpVInfo.size();
  const int Nz = ScalarBlock::sizeZ;
  const int Ny = ScalarBlock::sizeY;
  const int Nx = ScalarBlock::sizeX;

  // store grad(chi) in a vector and grad(SDF) in tmpV
  cubism::compute<ScalarLab>(GradScalarOnTmpV(sim), sim.chi);
  std::vector<double> gradChi(Nx * Ny * Nz * Nblocks * 3);
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    auto &__restrict__ TMP = *(ScalarBlock *)tmpInfo[i].ptrBlock;
    auto &__restrict__ TMPV = *(VectorBlock *)tmpVInfo[i].ptrBlock;
    for (int iz = 0; iz < Nz; iz++)
      for (int iy = 0; iy < Ny; iy++)
        for (int ix = 0; ix < Nx; ix++) {
          const size_t idx = i * Nz * Ny * Nx + iz * Ny * Nx + iy * Nx + ix;
          gradChi[3 * idx + 0] = TMPV(ix, iy, iz).u[0];
          gradChi[3 * idx + 1] = TMPV(ix, iy, iz).u[1];
          gradChi[3 * idx + 2] = TMPV(ix, iy, iz).u[2];
          TMP(ix, iy, iz).s = 0;
        }
    if (obstacleBlocks[i] == nullptr)
      continue; // obst not in block
    ObstacleBlock &o = *obstacleBlocks[i];
    const auto &__restrict__ SDF = o.sdfLab;
    for (int iz = 0; iz < Nz; iz++)
      for (int iy = 0; iy < Ny; iy++)
        for (int ix = 0; ix < Nx; ix++) {
          TMP(ix, iy, iz).s = SDF[iz + 1][iy + 1][ix + 1];
        }
  }
  cubism::compute<ScalarLab>(GradScalarOnTmpV(sim), sim.lhs);

  std::vector<int> idx_store;
  std::vector<int> ix_store;
  std::vector<int> iy_store;
  std::vector<int> iz_store;
  std::vector<long long> id_store;
  std::vector<Real> nx_store;
  std::vector<Real> ny_store;
  std::vector<Real> nz_store;
  std::vector<Real> cc_store;
  Real surface = 0.0;
  Real surface_c = 0.0;
  Real mass_flux = 0.0;

#pragma omp parallel for reduction(+ : surface, surface_c, mass_flux)
  for (const auto &info : sim.vel->getBlocksInfo()) {
    if (obstacleBlocks[info.blockID] == nullptr)
      continue; // obst not in block
    ObstacleBlock &o = *obstacleBlocks[info.blockID];
    auto &__restrict__ UDEF = o.udef;
    const auto &__restrict__ SDF = o.sdfLab;
    const auto &__restrict__ TMPV =
        *(VectorBlock *)tmpVInfo[info.blockID].ptrBlock;

    for (int iz = 0; iz < ScalarBlock::sizeZ; iz++)
      for (int iy = 0; iy < ScalarBlock::sizeY; iy++)
        for (int ix = 0; ix < ScalarBlock::sizeX; ix++) {
          if (SDF[iz + 1][iy + 1][ix + 1] > info.h ||
              SDF[iz + 1][iy + 1][ix + 1] < -info.h)
            continue;
          UDEF[iz][iy][ix][0] = 0.0;
          UDEF[iz][iy][ix][1] = 0.0;
          UDEF[iz][iy][ix][2] = 0.0;
          Real p[3];
          info.pos(p, ix, iy, iz);

          if (std::fabs(p[2] - position[2]) > 0.40 * thickness)
            continue;

          // find closest surface point to analytical expression
          int ss_min = 0;
          int sign_min = 0;
          Real dist_min = 1e10;

          for (int ss = 0; ss < myFish->Nm; ss++) {
            Real Pp[3] = {rX[ss] + width[ss] * norX[ss],
                          rY[ss] + width[ss] * norY[ss], 0};
            Real Pm[3] = {rX[ss] - width[ss] * norX[ss],
                          rY[ss] - width[ss] * norY[ss], 0};
            const Real dp = pow(Pp[0] - p[0], 2) + pow(Pp[1] - p[1], 2);
            const Real dm = pow(Pm[0] - p[0], 2) + pow(Pm[1] - p[1], 2);
            if (dp < dist_min) {
              sign_min = 1;
              dist_min = dp;
              ss_min = ss;
            }
            if (dm < dist_min) {
              sign_min = -1;
              dist_min = dm;
              ss_min = ss;
            }
          }

          const Real smax = rS[myFish->Nm - 1] - rS[0];
          const Real ds = 2 * smax / Nactuators;
          const Real current_s = rS[ss_min];
          if (current_s < 0.01 * length || current_s > 0.99 * length)
            continue;
          int idx = (current_s / ds); // this is the closest actuator
          const Real s0 = 0.5 * ds + idx * ds;
          if (sign_min == -1)
            idx += Nactuators / 2;
          const Real h3 = info.h * info.h * info.h;

          if (std::fabs(current_s - s0) < 0.5 * actuator_ds * length) {
            const size_t index =
                info.blockID * Nz * Ny * Nx + iz * Ny * Nx + iy * Nx + ix;
            const Real dchidx = gradChi[3 * index];
            const Real dchidy = gradChi[3 * index + 1];
            const Real dchidz = gradChi[3 * index + 2];
            Real nx = TMPV(ix, iy, iz).u[0];
            Real ny = TMPV(ix, iy, iz).u[1];
            Real nz = TMPV(ix, iy, iz).u[2];
            const Real nn = pow(nx * nx + ny * ny + nz * nz + 1e-21, -0.5);
            nx *= nn;
            ny *= nn;
            nz *= nn;
            const double c0 =
                std::fabs(current_s - s0) / (0.5 * actuator_ds * length);
            const double c = 1.0 - c0 * c0;
            UDEF[iz][iy][ix][0] = c * actuators[idx] * nx;
            UDEF[iz][iy][ix][1] = c * actuators[idx] * ny;
            UDEF[iz][iy][ix][2] = c * actuators[idx] * nz;
#pragma omp critical
            {
              ix_store.push_back(ix);
              iy_store.push_back(iy);
              iz_store.push_back(iz);
              id_store.push_back(info.blockID);
              nx_store.push_back(nx);
              ny_store.push_back(ny);
              nz_store.push_back(nz);
              cc_store.push_back(c);
              idx_store.push_back(idx);
            }
            const Real fac = (dchidx * nx + dchidy * ny + dchidz * nz) * h3;
            mass_flux +=
                fac * (UDEF[iz][iy][ix][0] * nx + UDEF[iz][iy][ix][1] * ny +
                       UDEF[iz][iy][ix][2] * nz);
            surface += fac;
            surface_c += fac * c;
          }
        }
  }

  Real Qtot[3] = {mass_flux, surface, surface_c};
  MPI_Allreduce(MPI_IN_PLACE, Qtot, 3, MPI_Real, MPI_SUM, sim.comm);
  // const Real uMean = Qtot[0]/Qtot[1];
  const Real qqqqq = Qtot[0] / Qtot[2];

// Substract total mass flux (divided by surface) from actuator velocities
#pragma omp parallel for
  for (size_t idx = 0; idx < id_store.size(); idx++) {
    const long long blockID = id_store[idx];
    const int ix = ix_store[idx];
    const int iy = iy_store[idx];
    const int iz = iz_store[idx];
    const Real nx = nx_store[idx];
    const Real ny = ny_store[idx];
    const Real nz = nz_store[idx];
    const int idx_st = idx_store[idx];
    const Real c = cc_store[idx];
    ObstacleBlock &o = *obstacleBlocks[blockID];
    auto &__restrict__ UDEF = o.udef;
    UDEF[iz][iy][ix][0] = c * (actuators[idx_st] - qqqqq) * nx;
    UDEF[iz][iy][ix][1] = c * (actuators[idx_st] - qqqqq) * ny;
    UDEF[iz][iy][ix][2] = c * (actuators[idx_st] - qqqqq) * nz;
  }
}

void SmartNaca::act(std::vector<Real> action, const int agentID) {
  t_change = sim.time;
  if (action.size() != actuators.size()) {
    std::cerr << "action size needs to be equal to actuators\n";
    fflush(0);
    abort();
  }
  for (size_t i = 0; i < action.size(); i++) {
    actuators_prev_value[i] = actuators[i];
    actuators_next_value[i] = action[i];
  }
}

Real SmartNaca::reward(const int agentID) {
  Real retval = fx_integral / 0.1; // 0.1 is the action times
  fx_integral = 0;
  Real regularizer = 0.0;
  for (size_t idx = 0; idx < actuators.size(); idx++) {
    regularizer += actuators[idx] * actuators[idx];
  }
  regularizer = pow(regularizer, 0.5) / actuators.size();
  return retval - 0.1 * regularizer;
}

std::vector<Real> SmartNaca::state(const int agentID) {
  std::vector<Real> S;
#if 0
  const int bins = 64;

  const Real dtheta = 2.*M_PI / bins;
  std::vector<int>   n_s   (bins,0.0);
  std::vector<Real>  p_s   (bins,0.0);
  std::vector<Real> fX_s   (bins,0.0);
  std::vector<Real> fY_s   (bins,0.0);
  for(auto & block : obstacleBlocks) if(block not_eq nullptr)
  {
    for(size_t i=0; i<block->n_surfPoints; i++)
    {
      const Real x     = block->x_s[i] - origC[0];
      const Real y     = block->y_s[i] - origC[1];
      const Real ang   = atan2(y,x);
      const Real theta = ang < 0 ? ang + 2*M_PI : ang;
      const Real p     = block->p_s[i];
      const Real fx    = block->fX_s[i];
      const Real fy    = block->fY_s[i];
      const int idx = theta / dtheta;
      n_s [idx] ++;
      p_s [idx] += p;
      fX_s[idx] += fx;
      fY_s[idx] += fy;
    }
  }

  MPI_Allreduce(MPI_IN_PLACE,n_s.data(),n_s.size(),MPI_INT ,MPI_SUM,sim.comm);
  for (int idx = 0 ; idx < bins; idx++)
  {
    p_s [idx] /= n_s[idx];
    fX_s[idx] /= n_s[idx];
    fY_s[idx] /= n_s[idx];
  }

  for (int idx = 0 ; idx < bins; idx++) S.push_back( p_s[idx]);
  for (int idx = 0 ; idx < bins; idx++) S.push_back(fX_s[idx]);
  for (int idx = 0 ; idx < bins; idx++) S.push_back(fY_s[idx]);
  MPI_Allreduce(MPI_IN_PLACE,  S.data(),  S.size(),MPI_Real,MPI_SUM,sim.comm);
  S.push_back(forcex);
  S.push_back(forcey);
  S.push_back(torque);

  if (sim.rank ==0 )
    for (size_t i = 0 ; i < S.size() ; i++) std::cout << S[i] << " ";

#endif
  return S;
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

namespace SphereObstacle {
struct FillBlocks : FillBlocksBase<FillBlocks> {
  const Real radius, h, safety = (2 + SURFDH) * h;
  const Real position[3];
  const Real box[3][2] = {{(Real)position[0] - (radius + safety),
                           (Real)position[0] + (radius + safety)},
                          {(Real)position[1] - (radius + safety),
                           (Real)position[1] + (radius + safety)},
                          {(Real)position[2] - (radius + safety),
                           (Real)position[2] + (radius + safety)}};

  FillBlocks(const Real _radius, const Real max_dx, const Real pos[3])
      : radius(_radius), h(max_dx), position{pos[0], pos[1], pos[2]} {}

  inline bool isTouching(const BlockInfo &info, const ScalarBlock &b) const {
    Real MINP[3], MAXP[3];
    info.pos(MINP, 0, 0, 0);
    info.pos(MAXP, ScalarBlock::sizeX - 1, ScalarBlock::sizeY - 1,
             ScalarBlock::sizeZ - 1);
    const Real intersect[3][2] = {
        {std::max(MINP[0], box[0][0]), std::min(MAXP[0], box[0][1])},
        {std::max(MINP[1], box[1][0]), std::min(MAXP[1], box[1][1])},
        {std::max(MINP[2], box[2][0]), std::min(MAXP[2], box[2][1])}};
    return intersect[0][1] - intersect[0][0] > 0 &&
           intersect[1][1] - intersect[1][0] > 0 &&
           intersect[2][1] - intersect[2][0] > 0;
  }

  inline Real signedDistance(const Real x, const Real y, const Real z) const {
    const Real dx = x - position[0], dy = y - position[1], dz = z - position[2];
    return radius -
           std::sqrt(dx * dx + dy * dy + dz * dz); // pos inside, neg outside
  }
};
} // namespace SphereObstacle

namespace HemiSphereObstacle {
struct FillBlocks : FillBlocksBase<FillBlocks> {
  const Real radius, h, safety = (2 + SURFDH) * h;
  const Real position[3];
  const Real box[3][2] = {
      {(Real)position[0] - radius - safety, (Real)position[0] + safety},
      {(Real)position[1] - radius - safety,
       (Real)position[1] + radius + safety},
      {(Real)position[2] - radius - safety,
       (Real)position[2] + radius + safety}};

  FillBlocks(const Real _radius, const Real max_dx, const Real pos[3])
      : radius(_radius), h(max_dx), position{pos[0], pos[1], pos[2]} {}

  inline bool isTouching(const BlockInfo &info, const ScalarBlock &b) const {
    Real MINP[3], MAXP[3];
    info.pos(MINP, 0, 0, 0);
    info.pos(MAXP, ScalarBlock::sizeX - 1, ScalarBlock::sizeY - 1,
             ScalarBlock::sizeZ - 1);
    const Real intersect[3][2] = {
        {std::max(MINP[0], box[0][0]), std::min(MAXP[0], box[0][1])},
        {std::max(MINP[1], box[1][0]), std::min(MAXP[1], box[1][1])},
        {std::max(MINP[2], box[2][0]), std::min(MAXP[2], box[2][1])}};
    return intersect[0][1] - intersect[0][0] > 0 &&
           intersect[1][1] - intersect[1][0] > 0 &&
           intersect[2][1] - intersect[2][0] > 0;
  }

  inline Real signedDistance(const Real x, const Real y,
                             const Real z) const { // pos inside, neg outside
    const Real dx = x - position[0], dy = y - position[1], dz = z - position[2];
    return std::min(-dx, radius - std::sqrt(dx * dx + dy * dy + dz * dz));
  }
};
} // namespace HemiSphereObstacle

Sphere::Sphere(SimulationData &s, ArgumentParser &p)
    : Obstacle(s, p), radius(0.5 * length) {
  accel_decel = p("-accel").asBool(false);
  bHemi = p("-hemisphere").asBool(false);
  if (accel_decel) {
    if (not bForcedInSimFrame[0]) {
      printf("Warning: sphere was not set to be forced in x-dir, yet the "
             "accel_decel pattern is active.\n");
    }
    umax = p("-xvel").asDouble(0.0);
    tmax = p("-T").asDouble(1.);
  }
}

void Sphere::create() {
  const Real h = sim.hmin;
  if (bHemi) {
    const HemiSphereObstacle::FillBlocks K(radius, h, position);
    create_base<HemiSphereObstacle::FillBlocks>(K);
  } else {
    const SphereObstacle::FillBlocks K(radius, h, position);
    create_base<SphereObstacle::FillBlocks>(K);
  }
}

void Sphere::finalize() {
  // this method allows any computation that requires the char function
  // to be computed. E.g. compute the effective center of mass or removing
  // momenta from udef
}

void Sphere::computeVelocities() {
  if (accel_decel) {
    if (sim.time < tmax)
      transVel_imposed[0] = umax * sim.time / tmax;
    else if (sim.time < 2 * tmax)
      transVel_imposed[0] = umax * (2 * tmax - sim.time) / tmax;
    else
      transVel_imposed[0] = 0;
  }

  Obstacle::computeVelocities();
}

CubismUP_3D_NAMESPACE_END

    CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

void CurvatureDefinedFishData::execute(const Real time, const Real l_tnext,
                                       const std::vector<Real> &input) {
  if (input.size() == 1) {
    // 1.midline curvature
    rlBendingScheduler.Turn(input[0], l_tnext);
  } else if (input.size() == 3) {
    assert(control_torsion == false);

    // 1.midline curvature
    rlBendingScheduler.Turn(input[0], l_tnext);

    // 2.swimming period
    if (TperiodPID)
      std::cout << "Warning: PID controller should not be used with RL."
                << std::endl;
    current_period = periodPIDval;
    next_period = Tperiod * (1 + input[1]);
    transition_start = l_tnext;
  } else if (input.size() == 5) {
    assert(control_torsion == true);

    // 1.midline curvature
    rlBendingScheduler.Turn(input[0], l_tnext);

    // 2.swimming period
    if (TperiodPID)
      std::cout << "Warning: PID controller should not be used with RL."
                << std::endl;
    current_period = periodPIDval;
    next_period = Tperiod * (1 + input[1]);
    transition_start = l_tnext;

    // 3.midline torsion
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
  if (transition_start < t &&
      t < transition_start + transition_duration) // timeshift also rampedup
  {
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

#if 1 // ramp-up over Tperiod
  const std::array<Real, 6> curvatureZeros = std::array<Real, 6>();
  curvatureScheduler.transition(0, 0, Tperiod, curvatureZeros, curvatureValues);
#else // no rampup for debug
  curvatureScheduler.transition(t, 0, Tperiod, curvatureValues,
                                curvatureValues);
#endif

  // query the schedulers for current values
  curvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rC, vC);
  rlBendingScheduler.gimmeValues(t, periodPIDval, length, bendPoints, Nm, rS,
                                 rB, vB);

  // next term takes into account the derivative of periodPIDval in darg:
  const Real diffT =
      TperiodPID ? 1 - (t - time0) * periodPIDdif / periodPIDval : 1;
  // time derivative of arg:
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
// compute normal and binormal vectors for a given midline
#pragma omp parallel for
  for (int i = 1; i < Nm - 1; i++) {
    // 2nd order finite difference for non-uniform grid
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

    // Project the normal vector computed by the Frenet equations onto
    // (-ty,tx,tz) which is a vector on the plane that is perpendicular to the
    // tangent vector t. This projection defines the new normal vector.
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

    // Compute the bi-normal vector as t x n
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

  // take care of first and last point
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

void StefanFish::saveRestart(FILE *f) {
  assert(f != NULL);
  Fish::saveRestart(f);
  CurvatureDefinedFishData *const cFish =
      dynamic_cast<CurvatureDefinedFishData *>(myFish);
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(7) << "_" << obstacleID << "_";
  std::string filename = "Schedulers" + ss.str() + ".restart";
  {
    std::ofstream savestream;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<Real>::digits10 + 1);
    savestream.open(filename);
    {
      const auto &c = cFish->curvatureScheduler;
      savestream << c.t0 << "\t" << c.t1 << std::endl;
      for (int i = 0; i < c.npoints; ++i)
        savestream << c.parameters_t0[i] << "\t" << c.parameters_t1[i] << "\t"
                   << c.dparameters_t0[i] << std::endl;
    }
    {
      const auto &c = cFish->periodScheduler;
      savestream << c.t0 << "\t" << c.t1 << std::endl;
      for (int i = 0; i < c.npoints; ++i)
        savestream << c.parameters_t0[i] << "\t" << c.parameters_t1[i] << "\t"
                   << c.dparameters_t0[i] << std::endl;
    }
    {
      const auto &c = cFish->rlBendingScheduler;
      savestream << c.t0 << "\t" << c.t1 << std::endl;
      for (int i = 0; i < c.npoints; ++i)
        savestream << c.parameters_t0[i] << "\t" << c.parameters_t1[i] << "\t"
                   << c.dparameters_t0[i] << std::endl;
    }
    {
      const auto &c = cFish->torsionScheduler;
      savestream << c.t0 << "\t" << c.t1 << std::endl;
      for (int i = 0; i < c.npoints; ++i)
        savestream << c.parameters_t0[i] << "\t" << c.parameters_t1[i] << "\t"
                   << c.dparameters_t0[i] << std::endl;
    }
    {
      savestream << r_axis.size() << std::endl;
      for (size_t i = 0; i < r_axis.size(); i++) {
        const auto &r = r_axis[i];
        savestream << r[0] << "\t" << r[1] << "\t" << r[2] << "\t" << r[3]
                   << std::endl;
      }
    }

    savestream.close();
  }

  // Save these numbers for PID controller and other stuff. Maybe not all of
  // them are needed but we don't care, it's only a few numbers.
  fprintf(f, "origC_x: %20.20e\n", (double)origC[0]);
  fprintf(f, "origC_y: %20.20e\n", (double)origC[1]);
  fprintf(f, "origC_z: %20.20e\n", (double)origC[2]);
  fprintf(f, "lastTact                 : %20.20e\n", (double)cFish->lastTact);
  fprintf(f, "lastCurv                 : %20.20e\n", (double)cFish->lastCurv);
  fprintf(f, "oldrCurv                 : %20.20e\n", (double)cFish->oldrCurv);
  fprintf(f, "periodPIDval             : %20.20e\n",
          (double)cFish->periodPIDval);
  fprintf(f, "periodPIDdif             : %20.20e\n",
          (double)cFish->periodPIDdif);
  fprintf(f, "time0                    : %20.20e\n", (double)cFish->time0);
  fprintf(f, "timeshift                : %20.20e\n", (double)cFish->timeshift);
  fprintf(f, "Ttorsion_start           : %20.20e\n",
          (double)cFish->Ttorsion_start);
  fprintf(f, "current_period           : %20.20e\n",
          (double)cFish->current_period);
  fprintf(f, "next_period              : %20.20e\n",
          (double)cFish->next_period);
  fprintf(f, "transition_start         : %20.20e\n",
          (double)cFish->transition_start);
  fprintf(f, "transition_duration      : %20.20e\n",
          (double)cFish->transition_duration);
  fprintf(f, "torsionValues[0]         : %20.20e\n",
          (double)cFish->torsionValues[0]);
  fprintf(f, "torsionValues[1]         : %20.20e\n",
          (double)cFish->torsionValues[1]);
  fprintf(f, "torsionValues[2]         : %20.20e\n",
          (double)cFish->torsionValues[2]);
  fprintf(f, "torsionValues_previous[0]: %20.20e\n",
          (double)cFish->torsionValues_previous[0]);
  fprintf(f, "torsionValues_previous[1]: %20.20e\n",
          (double)cFish->torsionValues_previous[1]);
  fprintf(f, "torsionValues_previous[2]: %20.20e\n",
          (double)cFish->torsionValues_previous[2]);
  fprintf(f, "TperiodPID               : %d\n", (int)cFish->TperiodPID);
  fprintf(f, "control_torsion          : %d\n", (int)cFish->control_torsion);
  fprintf(f, "alpha                    : %20.20e\n", (double)cFish->alpha);
  fprintf(f, "dalpha                   : %20.20e\n", (double)cFish->dalpha);
  fprintf(f, "beta                     : %20.20e\n", (double)cFish->beta);
  fprintf(f, "dbeta                    : %20.20e\n", (double)cFish->dbeta);
  fprintf(f, "gamma                    : %20.20e\n", (double)cFish->gamma);
  fprintf(f, "dgamma                   : %20.20e\n", (double)cFish->dgamma);
}

void StefanFish::loadRestart(FILE *f) {
  assert(f != NULL);
  Fish::loadRestart(f);
  CurvatureDefinedFishData *const cFish =
      dynamic_cast<CurvatureDefinedFishData *>(myFish);
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(7) << "_" << obstacleID << "_";
  std::ifstream restartstream;
  std::string filename = "Schedulers" + ss.str() + ".restart";
  restartstream.open(filename);
  {
    auto &c = cFish->curvatureScheduler;
    restartstream >> c.t0 >> c.t1;
    for (int i = 0; i < c.npoints; ++i)
      restartstream >> c.parameters_t0[i] >> c.parameters_t1[i] >>
          c.dparameters_t0[i];
  }
  {
    auto &c = cFish->periodScheduler;
    restartstream >> c.t0 >> c.t1;
    for (int i = 0; i < c.npoints; ++i)
      restartstream >> c.parameters_t0[i] >> c.parameters_t1[i] >>
          c.dparameters_t0[i];
  }
  {
    auto &c = cFish->rlBendingScheduler;
    restartstream >> c.t0 >> c.t1;
    for (int i = 0; i < c.npoints; ++i)
      restartstream >> c.parameters_t0[i] >> c.parameters_t1[i] >>
          c.dparameters_t0[i];
  }
  {
    auto &c = cFish->torsionScheduler;
    restartstream >> c.t0 >> c.t1;
    for (int i = 0; i < c.npoints; ++i)
      restartstream >> c.parameters_t0[i] >> c.parameters_t1[i] >>
          c.dparameters_t0[i];
  }
  {
    size_t nr = 0;
    restartstream >> nr;
    for (size_t i = 0; i < nr; i++) {
      std::array<Real, 4> r;
      restartstream >> r[0] >> r[1] >> r[2] >> r[3];
      r_axis.push_back(r);
    }
  }
  restartstream.close();

  bool ret = true;
  double temp;
  int temp1;
  ret = ret && 1 == fscanf(f, "origC_x: %le\n", &temp);
  origC[0] = temp;
  ret = ret && 1 == fscanf(f, "origC_y: %le\n", &temp);
  origC[1] = temp;
  ret = ret && 1 == fscanf(f, "origC_z: %le\n", &temp);
  origC[2] = temp;
  ret = ret && 1 == fscanf(f, "lastTact                 : %le\n", &temp);
  cFish->lastTact = temp;
  ret = ret && 1 == fscanf(f, "lastCurv                 : %le\n", &temp);
  cFish->lastCurv = temp;
  ret = ret && 1 == fscanf(f, "oldrCurv                 : %le\n", &temp);
  cFish->oldrCurv = temp;
  ret = ret && 1 == fscanf(f, "periodPIDval             : %le\n", &temp);
  cFish->periodPIDval = temp;
  ret = ret && 1 == fscanf(f, "periodPIDdif             : %le\n", &temp);
  cFish->periodPIDdif = temp;
  ret = ret && 1 == fscanf(f, "time0                    : %le\n", &temp);
  cFish->time0 = temp;
  ret = ret && 1 == fscanf(f, "timeshift                : %le\n", &temp);
  cFish->timeshift = temp;
  ret = ret && 1 == fscanf(f, "Ttorsion_start           : %le\n", &temp);
  cFish->Ttorsion_start = temp;
  ret = ret && 1 == fscanf(f, "current_period           : %le\n", &temp);
  cFish->current_period = temp;
  ret = ret && 1 == fscanf(f, "next_period              : %le\n", &temp);
  cFish->next_period = temp;
  ret = ret && 1 == fscanf(f, "transition_start         : %le\n", &temp);
  cFish->transition_start = temp;
  ret = ret && 1 == fscanf(f, "transition_duration      : %le\n", &temp);
  cFish->transition_duration = temp;
  ret = ret && 1 == fscanf(f, "torsionValues[0]         : %le\n", &temp);
  cFish->torsionValues[0] = temp;
  ret = ret && 1 == fscanf(f, "torsionValues[1]         : %le\n", &temp);
  cFish->torsionValues[1] = temp;
  ret = ret && 1 == fscanf(f, "torsionValues[2]         : %le\n", &temp);
  cFish->torsionValues[2] = temp;
  ret = ret && 1 == fscanf(f, "torsionValues_previous[0]: %le\n", &temp);
  cFish->torsionValues_previous[0] = temp;
  ret = ret && 1 == fscanf(f, "torsionValues_previous[1]: %le\n", &temp);
  cFish->torsionValues_previous[1] = temp;
  ret = ret && 1 == fscanf(f, "torsionValues_previous[2]: %le\n", &temp);
  cFish->torsionValues_previous[2] = temp;
  ret = ret && 1 == fscanf(f, "TperiodPID               : %d\n", &temp1);
  cFish->TperiodPID = temp1;
  ret = ret && 1 == fscanf(f, "control_torsion          : %d\n", &temp1);
  cFish->control_torsion = temp1;
  ret = ret && 1 == fscanf(f, "alpha                    : %le\n", &temp);
  cFish->alpha = temp;
  ret = ret && 1 == fscanf(f, "dalpha                   : %le\n", &temp);
  cFish->dalpha = temp;
  ret = ret && 1 == fscanf(f, "beta                     : %le\n", &temp);
  cFish->beta = temp;
  ret = ret && 1 == fscanf(f, "dbeta                    : %le\n", &temp);
  cFish->dbeta = temp;
  ret = ret && 1 == fscanf(f, "gamma                    : %le\n", &temp);
  cFish->gamma = temp;
  ret = ret && 1 == fscanf(f, "dgamma                   : %le\n", &temp);
  cFish->dgamma = temp;

  if ((not ret)) {
    printf("Error reading restart file. Aborting...\n");
    fflush(0);
    abort();
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

  // compute pitch, roll, yaw
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
  // const Real pitch = asin (2.0 * (q[2] * q[0] - q[3] * q[1]));

  const bool roll_is_small = std::fabs(roll) < M_PI / 9.; // 20 degrees
  const bool yaw_is_small = std::fabs(yaw) < M_PI / 9.;   // 20 degrees

  if (bCorrectPosition) {
    // 1.control position in x
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

    // 2.control position in y and yaw angle
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
    // 3. control pitch
    const Real pitch_tgt = 0;
    const Real dphi = pitch - pitch_tgt;
    // const Real g         = (roll_is_small && yaw_is_small) ? -wzp * dphi :
    // 0.0;

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

  // if (sim.rank == 0)
  //{
  //  char buf[500];
  //  sprintf(buf, "gamma%d.txt",obstacleID);
  //  FILE * f = fopen(buf, "a");
  //  fprintf(f, "%g %g %g %g %g %g %g
  //  \n",sim.time,cFish->alpha,cFish->dalpha,cFish->beta,cFish->dbeta,cFish->gamma,cFish->dgamma);
  //  fclose(f);
  //}

  Fish::create();
}

void StefanFish::computeVelocities() {
  Obstacle::computeVelocities();

  // Compute angular velocity component on the rolling axis of the fish and set
  // it to 0. Then, impose rolling angular velocity that will make the rolling
  // angle go to zero after 0.5Tperiod time has passed. Important: this assumes
  // an initial orientation (q = (1,0,0,0)) along the x-axis for the fish
  //           where the head is facing the (-1,0,0) direction
  //           (this is the default initial orientation for fish).
  if (bCorrectRoll) {
    auto *const cFish = dynamic_cast<CurvatureDefinedFishData *>(myFish);
    const Real *const q = quaternion;
    Real *const o = angVel;

    // roll angle and droll/dt
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

    // 1st column of rotation matrix, roll axis
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

//////////////////////////////////
// Reinforcement Learning functions
//////////////////////////////////

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
    actions[1] = 0; // no pitching
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

  // sensor locations
  const std::array<Real, 3> locFront = {cFish->sensorLocation[0 * 3 + 0],
                                        cFish->sensorLocation[0 * 3 + 1],
                                        cFish->sensorLocation[0 * 3 + 2]};
  const std::array<Real, 3> locUpper = {cFish->sensorLocation[1 * 3 + 0],
                                        cFish->sensorLocation[1 * 3 + 1],
                                        cFish->sensorLocation[1 * 3 + 2]};
  const std::array<Real, 3> locLower = {cFish->sensorLocation[2 * 3 + 0],
                                        cFish->sensorLocation[2 * 3 + 1],
                                        cFish->sensorLocation[2 * 3 + 2]};
  // compute shear stress force (x,y,z) components
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
  const std::vector<cubism::BlockInfo> &velInfo = sim.velInfo();
  for (size_t i = 0; i < velInfo.size(); ++i) {
    // compute lower left and top right corners of block (+- 0.5 h because pos
    // returns cell centers)
    std::array<Real, 3> MIN = velInfo[i].pos<Real>(0, 0, 0);
    std::array<Real, 3> MAX = velInfo[i].pos<Real>(
        ScalarBlock::sizeX - 1, ScalarBlock::sizeY - 1, ScalarBlock::sizeZ - 1);
    MIN[0] -= 0.5 * velInfo[i].h;
    MIN[1] -= 0.5 * velInfo[i].h;
    MIN[2] -= 0.5 * velInfo[i].h;
    MAX[0] += 0.5 * velInfo[i].h;
    MAX[1] += 0.5 * velInfo[i].h;
    MAX[2] += 0.5 * velInfo[i].h;

    // check whether point is inside block
    if (pos[0] >= MIN[0] && pos[1] >= MIN[1] && pos[2] >= MIN[2] &&
        pos[0] <= MAX[0] && pos[1] <= MAX[1] && pos[2] <= MAX[2]) {
      return i;
    }
  }
  return -1; // rank does not contain point
};

// returns shear at given surface location
std::array<Real, 3>
StefanFish::getShear(const std::array<Real, 3> pSurf) const {
  const std::vector<cubism::BlockInfo> &velInfo = sim.velInfo();

  Real myF[3] = {0, 0, 0};

  // Get blockId of block that contains point pSurf.
  ssize_t blockIdSurf = holdingBlockID(pSurf);
  if (blockIdSurf >= 0) {
    const auto &skinBinfo = velInfo[blockIdSurf];

    // check whether obstacle block exists
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

  return std::array<Real, 3>{{myF[0], myF[1], myF[2]}}; // return shear
};

CubismUP_3D_NAMESPACE_END int main(int argc, char **argv) {
  int provided;
  const auto SECURITY = MPI_THREAD_FUNNELED;
  MPI_Init_thread(&argc, &argv, SECURITY, &provided);
  if (provided < SECURITY) {
    printf("ERROR: MPI implementation does not have required thread support\n");
    fflush(0);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::cout << "============================================================="
                 "==========\n";
    std::cout << "Cubism UP 3D (velocity-pressure 3D incompressible "
                 "Navier-Stokes solver)\n";
    std::cout << "============================================================="
                 "==========\n";
#ifdef NDEBUG
    std::cout << "Running in RELEASE mode!\n";
#else
    std::cout << "Running in DEBUG mode!\n";
#endif
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = MPI_Wtime();
  cubismup3d::Simulation *sim =
      new cubismup3d::Simulation(argc, argv, MPI_COMM_WORLD);
  sim->init();
  sim->simulate();
  delete sim;
  MPI_Barrier(MPI_COMM_WORLD);
  double t2 = MPI_Wtime();
  if (rank == 0)
    std::cout << "Total time = " << t2 - t1 << std::endl;

  MPI_Finalize();
  return 0;
}
