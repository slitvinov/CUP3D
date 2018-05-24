//
//  CubismUP_3D
//
//  Written by Guido Novati ( novatig@ethz.ch ).
//  This file started as an extension of code written by Wim van Rees
//  Copyright (c) 2017 ETHZ. All rights reserved.
//

#ifndef __IncompressibleFluids3D__IF3D_NacaOperator__
#define __IncompressibleFluids3D__IF3D_NacaOperator__

#include <cmath>
#include <array>
#include "IF3D_FishOperator.h"

class IF3D_NacaOperator: public IF3D_FishOperator
{
  double Apitch, Fpitch, Ppitch, Mpitch, Fheave, Aheave, tOld = 0;
  bool bCreated;
 public:
  IF3D_NacaOperator(FluidGridMPI*g, ArgumentParser&p, const Real*const u);
  void update(const int stepID, const double t, const double dt, const Real* Uinf) override;
  void computeVelocities(const Real* Uinf) override;
  void writeSDFOnBlocks(const mapBlock2Segs& segmentsPerBlock) override;
};


#endif /* defined(__IncompressibleFluids3D__IF3D_CarlingFish__) */
