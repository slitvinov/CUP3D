//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_Obstacle_h
#define CubismUP_3D_Obstacle_h

#include "ObstacleBlock.h"
#include "SimulationData.h"

#include <array>
#include <fstream>

// forward declaration of derived class for visitor

namespace cubism { class ArgumentParser; }

CubismUP_3D_NAMESPACE_BEGIN

class Obstacle;
class ObstacleVector;

/*
 * Structure containing all externally configurable parameters of a base obstacle.
 */
struct ObstacleArguments
{
  double length = 0.0;
  std::array<double, 3> position = {{0.0, 0.0, 0.0}};
  std::array<double, 4> quaternion = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 3> enforcedVelocity = {{0.0, 0.0, 0.0}};  // Only if bForcedInSimFrame.
  std::array<bool, 3> bForcedInSimFrame = {{false, false, false}};
  std::array<bool, 3> bFixFrameOfRef = {{false, false, false}};
  bool bFixToPlanar = false;
  bool bComputeForces = true;

  ObstacleArguments() = default;

  /* Convert human-readable format into internal representation of parameters. */
  ObstacleArguments(const SimulationData & sim, cubism::ArgumentParser &parser);
};


struct ObstacleVisitor
{
  virtual ~ObstacleVisitor() {}

  virtual void visit(Obstacle* const obstacle) = 0;
  //virtual void visit(IF3D_ObstacleVector  * const obstacle) {}
};

class Obstacle
{
protected:
  const SimulationData & sim;
  FluidGridMPI * const grid = sim.grid;
  std::vector<ObstacleBlock*> obstacleBlocks;
  bool printedHeaderVels = false;
  bool isSelfPropelled = false;
public:
  int obstacleID=0;
  bool bInteractive=0, bHasSkin=0, bForces=0;
  double quaternion[4] = {1,0,0,0}, _2Dangle = 0, phaseShift=0; //orientation
  double position[3] = {0,0,0}, absPos[3] = {0,0,0}, transVel[3] = {0,0,0};
  double angVel[3] = {0,0,0}, J[6] = {0,0,0,0,0,0}; //mom of inertia
  // computed from chi on the grid:
  double centerOfMass[3] = {0,0,0};
  //from penalization:
  double mass=0, force[3] = {0,0,0}, torque[3] = {0,0,0};
  //from compute forces: perimeter, circulation and forces
  double totChi=0, gamma[3]={0,0,0}, surfForce[3]={0,0,0};
  //pressure and viscous contribution from compute forces:
  double presForce[3]={0,0,0}, viscForce[3]={0,0,0}, surfTorque[3]={0,0,0};
  double drag=0, thrust=0, Pout=0, PoutBnd=0, pLocom=0;
  double defPower=0, defPowerBnd=0, Pthrust=0, Pdrag=0, EffPDef=0, EffPDefBnd=0;
  double transVel_correction[3]={0,0,0}, angVel_correction[3]={0,0,0}, length;
  //forced obstacles:
  double transVel_computed[3]= {0,0,0}, angVel_computed[3]= {0,0,0};
  double transVel_imposed[3]= {0,0,0};

  // stuff dealing with frame of reference:
  std::array<bool, 3> bFixFrameOfRef = {{false, false, false}};
  std::array<bool, 3> bForcedInSimFrame = {{false, false, false}};
  std::array<bool, 3> bBlockRotation = {{false, false, false}};

protected:
  virtual void _writeComputedVelToFile();
  virtual void _writeDiagForcesToFile();
  virtual void _writeSurfForcesToFile();
  //void _finalizeAngVel(Real AV[3], const Real J[6], const Real& gam0, const Real& gam1, const Real& gam2);

public:
  Obstacle(SimulationData& s, const ObstacleArguments &args);
  Obstacle(SimulationData& s, cubism::ArgumentParser &parser);

  Obstacle(SimulationData& s) : sim(s) {  }


  virtual void Accept(ObstacleVisitor * visitor);
  virtual Real getD() const {return length;}

  virtual void computeVelocities();
  virtual void computeForces();
  virtual void update();
  virtual void save(std::string filename = std::string());
  virtual void restart(std::string filename = std::string());

  virtual void create();
  virtual void finalize();

  //methods that work for all obstacles
  std::vector<ObstacleBlock*> getObstacleBlocks() const
  {
      return obstacleBlocks;
  }
  std::vector<ObstacleBlock*>* getObstacleBlocksPtr()
  {
      return &obstacleBlocks;
  }
  void getObstacleBlocks(std::vector<ObstacleBlock*>*& obstblock_ptr)
  {
      obstblock_ptr = &obstacleBlocks;
  }

  virtual ~Obstacle()
  {
    for(auto & entry : obstacleBlocks) {
      if(entry != nullptr) {
          delete entry;
          entry = nullptr;
      }
    }
    obstacleBlocks.clear();
  }

  virtual std::array<double,3> getTranslationVelocity() const;
  virtual std::array<double,3> getAngularVelocity() const;
  virtual std::array<double,3> getCenterOfMass() const;

  // driver to execute finite difference kernels either on all points relevant
  // to the mass of the obstacle (where we have char func) or only on surface

  template<typename T>
  void create_base(const T& kernel)
  {
    for(auto & entry : obstacleBlocks) {
      if(entry == nullptr) continue;
      delete entry;
      entry = nullptr;
    }
    const std::vector<cubism::BlockInfo>& vInfo = sim.vInfo();
    obstacleBlocks.resize(vInfo.size(), nullptr);

    #pragma omp parallel for schedule(dynamic, 1)
    for(size_t i=0; i<vInfo.size(); i++) {
      const cubism::BlockInfo& info = vInfo[i];
      const FluidBlock &b = *(FluidBlock *)info.ptrBlock;
      if(kernel.isTouching(b)) {
        assert(obstacleBlocks[info.blockID] == nullptr);
        obstacleBlocks[info.blockID] = new ObstacleBlock();
        obstacleBlocks[info.blockID]->clear(); //memset 0
        kernel(info, obstacleBlocks[info.blockID]);
      }
    }
  }
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_Obstacle_h