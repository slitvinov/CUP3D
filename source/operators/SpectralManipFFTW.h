//
//  CubismUP_3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Hugues de Laroussilhe.
//

#ifndef CubismUP_3D_SpectralManipFFTW_h
#define CubismUP_3D_SpectralManipFFTW_h

#include "SpectralManip.h"

CubismUP_3D_NAMESPACE_BEGIN

class SpectralManipPeriodic : public SpectralManip
{
  ptrdiff_t alloc_local=0;
  ptrdiff_t local_n0=0, local_0_start=0;
  ptrdiff_t local_n1=0, local_1_start=0;
  void * fwd_u, * fwd_v, * fwd_w, * fwd_cs2;
  void * bwd_u, * bwd_v, * bwd_w;

public:

  SpectralManipPeriodic(SimulationData & s);
  ~SpectralManipPeriodic();

  void prepareFwd() override;
  void prepareBwd() override;

  void runFwd() const override;
  void runBwd() const override;

  void _compute_largeModesForcing() override;
  void _compute_analysis() override;
  void _compute_IC(const std::vector<Real> &K,
                   const std::vector<Real> &E) override;
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_SpectralManipFFTW_h
