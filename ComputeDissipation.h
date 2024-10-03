#ifndef CubismUP_3D_ComputeDissipation_h
#define CubismUP_3D_ComputeDissipation_h

CubismUP_3D_NAMESPACE_BEGIN

    class ComputeDissipation : public Operator {
public:
  ComputeDissipation(SimulationData &s) : Operator(s) {}
  void operator()(const Real dt);
  std::string getName() { return "Dissipation"; }
};

CubismUP_3D_NAMESPACE_END
#endif
