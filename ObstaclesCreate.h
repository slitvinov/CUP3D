#pragma once

CubismUP_3D_NAMESPACE_BEGIN

    class CreateObstacles : public Operator {
public:
  CreateObstacles(SimulationData &s) : Operator(s) {}

  void operator()(const Real dt);

  std::string getName() { return "CreateObstacles"; }
};

CubismUP_3D_NAMESPACE_END
