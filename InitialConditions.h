#ifndef CubismUP_3D_InitialConditions_h
#define CubismUP_3D_InitialConditions_h
CubismUP_3D_NAMESPACE_BEGIN

    class InitialConditions : public Operator {
public:
  InitialConditions(SimulationData &s) : Operator(s) {}

  template <typename K> inline void run(const K kernel) {
    std::vector<cubism::BlockInfo> &vInfo = sim.velInfo();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < vInfo.size(); i++)
      kernel(vInfo[i], *(VectorBlock *)vInfo[i].ptrBlock);
  }

  void operator()(const Real dt);

  std::string getName() { return "IC"; }
};

CubismUP_3D_NAMESPACE_END
#endif
