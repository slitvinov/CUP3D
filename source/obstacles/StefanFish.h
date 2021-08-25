//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Wim van Rees.
//

#ifndef CubismUP_3D_StefanFish_h
#define CubismUP_3D_StefanFish_h

#include "Fish.h"

CubismUP_3D_NAMESPACE_BEGIN

class StefanFish: public Fish
{
protected:
  Real origC[2] = {(Real)0, (Real)0};
  Real origAng = 0;
public:
  StefanFish(SimulationData&s, cubism::ArgumentParser&p);
  void save(std::string filename = std::string()) override;
  void restart(std::string filename) override;
  void create() override;

  // member function for action in RL
  void act(const Real lTact, const std::vector<double>& a) const;
  double getLearnTPeriod() const;

  // member functions for state in RL
  std::vector<double> state() const;

  // Helpers for state function
  size_t holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo) const;

  std::array<int, 2> safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh ) const;

  std::array<Real, 2> skinVel(const std::array<Real,2> pSkin, const std::vector<cubism::BlockInfo>& velInfo) const;

  std::array<Real, 2> sensVel(const std::array<Real,2> pSens, const std::vector<cubism::BlockInfo>& velInfo) const;
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_StefanFish_h
