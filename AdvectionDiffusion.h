#pragma once
CubismUP_3D_NAMESPACE_BEGIN

    class AdvectionDiffusion : public Operator {
  std::vector<Real> vOld;

public:
  AdvectionDiffusion(SimulationData &s) : Operator(s) {}

  ~AdvectionDiffusion() {}

  void operator()(const Real dt);

  std::string getName() { return "AdvectionDiffusion"; }
};

CubismUP_3D_NAMESPACE_END
