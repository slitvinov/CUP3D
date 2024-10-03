#ifndef CubismUP_3D_ObstacleFactory_h
#define CubismUP_3D_ObstacleFactory_h
namespace cubism {
class ArgumentParser;
}

CubismUP_3D_NAMESPACE_BEGIN

    class ObstacleFactory {
  SimulationData &sim;

public:
  ObstacleFactory(SimulationData &s) : sim(s) {}

  /* Add obstacles defined with `-factory` and `-factory-content` arguments. */
  void addObstacles(cubism::ArgumentParser &parser);

  /* Add obstacles specified with a given string. */
  void addObstacles(const std::string &factoryContent);
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_ObstacleFactory_h