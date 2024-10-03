#ifndef CubismUP_3D_Timer_h
#define CubismUP_3D_Timer_h
CubismUP_3D_NAMESPACE_BEGIN

    class Timer {
  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;

public:
  void start() { t_start = std::chrono::high_resolution_clock::now(); }

  Real stop() {
    t_end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<Real>(t_end - t_start).count();
  }
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_Timer_h
