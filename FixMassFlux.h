
CubismUP_3D_NAMESPACE_BEGIN

    class FixMassFlux : public Operator {
public:
  FixMassFlux(SimulationData &s);

  void operator()(const double dt);

  std::string getName() { return "FixMassFlux"; }
};

CubismUP_3D_NAMESPACE_END
