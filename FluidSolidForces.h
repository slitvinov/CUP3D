
CubismUP_3D_NAMESPACE_BEGIN

    class ComputeForces : public Operator {
public:
  ComputeForces(SimulationData &s) : Operator(s) {}

  void operator()(const Real dt);

  std::string getName() { return "ComputeForces"; }
};

CubismUP_3D_NAMESPACE_END
