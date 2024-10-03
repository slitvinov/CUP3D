#pragma once
CubismUP_3D_NAMESPACE_BEGIN

    class ExternalForcing : public Operator {
public:
  ExternalForcing(SimulationData &s) : Operator(s) {}

  void operator()(const double dt);

  std::string getName() { return "ExternalForcing"; }
};

CubismUP_3D_NAMESPACE_END
