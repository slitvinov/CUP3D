#ifndef CubismUP_3D_Penalization_h
#define CubismUP_3D_Penalization_h

#include "Operator.h"

CubismUP_3D_NAMESPACE_BEGIN

    class Penalization : public Operator {
public:
  Penalization(SimulationData &s);

  void operator()(const Real dt);

  void preventCollidingObstacles() const;

  std::string getName() { return "Penalization"; }
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_Penalization_h
