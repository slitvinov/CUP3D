#pragma once
CubismUP_3D_NAMESPACE_BEGIN

    class Operator {
public:
  SimulationData &sim;
  Operator(SimulationData &s) noexcept : sim(s) {}
  virtual ~Operator() = default;
  virtual void operator()(Real dt) = 0;
  virtual std::string getName() = 0;
};
CubismUP_3D_NAMESPACE_END
