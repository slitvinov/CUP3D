#pragma once

#include <memory>
#include "Operator.h"
#include "PoissonSolverBase.h"
#include "ObstacleVector.h"
#include "PoissonSolverAMR.h"

CubismUP_3D_NAMESPACE_BEGIN

    class PoissonSolverBase;

class PressureProjection : public Operator {
protected:
  // Alias of sim.pressureSolver.
  std::shared_ptr<PoissonSolverBase> pressureSolver;
  std::vector<Real> pOld;

public:
  PressureProjection(SimulationData &s);
  ~PressureProjection() = default;

  void operator()(Real dt) override;

  std::string getName() { return "PressureProjection"; }
};

CubismUP_3D_NAMESPACE_END
