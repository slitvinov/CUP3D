//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ivica Kicic (kicici@ethz.ch) in May 2018.
//

#include "obstacles/ExternalObstacle.h"
#include "obstacles/extra/ObstacleLibrary.h"

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

namespace {

// TODO: The position shift should be done here, not in the external code.
struct FillBlocksExternal : FillBlocksBase<FillBlocksExternal>
{
  const ExternalObstacleSettings &S;

  FillBlocksExternal(const ExternalObstacleSettings &_S) : S(_S) { }

  inline bool isTouching(const FluidBlock&b) const
  {
    // Ask the external code if it the block is overlapping the box.
    return S.is_touching_fn(b.min_pos, b.max_pos);
  }

  Real signedDistance(const Real x, const Real y, const Real z) const
  {
    // Ask the external code what's the signed distance.
    return S.signed_distance_fn({x, y, z});
  }

  /*
   * Fill out `ObstacleBlock::udef` by calling the external velocity function.
   */
  void setVelocity(const BlockInfo &info, ObstacleBlock * const o) const {
    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
      const std::array<Real, 3> p = info.pos<Real>(ix, iy, iz);
      const std::array<Real, 3> udef = S.velocity_fn(p);

      o->udef[iz][iy][ix][0] = udef[0];
      o->udef[iz][iy][ix][1] = udef[1];
      o->udef[iz][iy][ix][2] = udef[2];
    }
  }
};

}  // namespace (empty)

ExternalObstacle::ExternalObstacle(SimulationData&s, const ObstacleArguments &args)
    : Obstacle(s, args)
{
  bForcedInSimFrame = {true, true, true};
  bFixFrameOfRef = {true, true, true};
  bBlockRotation = {true, true, true};
}

void ExternalObstacle::computeVelocities()
{
  Obstacle::computeVelocities();

  if (settings.com_velocity_fn != nullptr) {
    const std::array<Real, 3> v = settings.com_velocity_fn();
    transVel[0] = transVel_imposed[0] = v[0];
    transVel[1] = transVel_imposed[1] = v[1];
    transVel[2] = transVel_imposed[2] = v[2];
  }
}

/*
 * Finds out which blocks are affected (overlap with the object).
 */
void ExternalObstacle::create()
{
  if (!settings.is_touching_fn
        || !settings.signed_distance_fn
        || !settings.velocity_fn
        || !settings.com_velocity_fn) {
    // `Simulation` automatically invokes `create` during initalization.
    // Instead of changing the code there, we ignore that `create` request here.
    return;
  }

  const FillBlocksExternal kernel(settings);
  create_base<FillBlocksExternal>(kernel);

  const std::vector<cubism::BlockInfo>& vInfo = sim.vInfo();
  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < (int)vInfo.size(); ++i) {
    const BlockInfo &info = vInfo[i];
    if (obstacleBlocks[info.blockID] == nullptr) continue;
    kernel.setVelocity(info, obstacleBlocks[info.blockID]);
  }
}

void ExternalObstacle::finalize()
{
  // this method allows any computation that requires the char function
  // to be computed. E.g. compute the effective center of mass or removing
  // momenta from udef
}

CubismUP_3D_NAMESPACE_END
