#ifndef CubismUP_3D_Pipe_h
#define CubismUP_3D_Pipe_h
CubismUP_3D_NAMESPACE_BEGIN

    class Pipe : public Obstacle {
  const Real radius;
  const Real halflength;
  std::string section = "circular"; // or whatever
  Real umax = 0;
  Real vmax = 0;
  Real wmax = 0;
  Real tmax = 1;
  // special case: startup with unif accel to umax in tmax, and then decel to 0
  bool accel = false;

public:
  Pipe(SimulationData &s, cubism::ArgumentParser &p);
  void _init(void);
  void create() override;
  void finalize() override;
  void computeVelocities() override;
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_Pipe_h
