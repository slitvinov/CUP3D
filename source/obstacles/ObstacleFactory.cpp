//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Wim van Rees.
//

#include "ObstacleFactory.h"
#include "ObstacleVector.h"
#include "extra/FactoryFileLineParser.h"

#include "CarlingFish.h"
#include "Cylinder.h"
#include "Ellipsoid.h"
#include "Naca.h"
#include "Plate.h"
#include "Sphere.h"
#include "StefanFish.h"

#include <iostream>
#include <fstream>

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;
using VectorType = ObstacleVector::VectorType;

/*
 * Create an obstacle instance given its name and arguments.
 */
static std::shared_ptr<Obstacle>
_createObstacle(SimulationData &sim,
                const std::string &objectName,
                FactoryFileLineParser &lineParser)
{
  if (objectName == "Sphere")
    return std::make_shared<Sphere>(sim, lineParser);
  if (objectName == "StefanFish")
    return std::make_shared<StefanFish>(sim, lineParser);
  if (objectName == "CarlingFish")
    return std::make_shared<CarlingFish>(sim, lineParser);
  if (objectName == "Naca")
    return std::make_shared<Naca>(sim, lineParser);
  if (objectName == "Cylinder")
    return std::make_shared<Cylinder>(sim, lineParser);
  if (objectName == "Plate")
    return std::make_shared<Plate>(sim, lineParser);
  if (objectName == "Ellipsoid")
    return std::make_shared<Ellipsoid>(sim, lineParser);

  if (sim.rank == 0) {
    std::cout << "Case " << objectName << " is not defined: aborting\n" << std::flush;
    abort();
  }

  return {};
}

/*
 * Add one obstacle per non-empty non-comment line of the given stream.
 */
static void _addObstacles(SimulationData &sim, std::stringstream &stream)
{
  if (sim.rank == 0)
    printf("Factory content:\n%s\n\n", stream.str().c_str());
  // here we store the data per object
  std::vector<std::pair<std::string, FactoryFileLineParser>> factoryLines;
  std::string line;

  while (std::getline(stream, line)) {
      std::istringstream line_stream(line);
      std::string ID;
      line_stream >> ID;
      if (ID.empty() || ID[0] == '#') continue;  // Comments and empty lines ignored.
      factoryLines.emplace_back(ID, FactoryFileLineParser(line_stream));
  }
  if (factoryLines.empty()) {
    if (sim.rank == 0)
      std::cout << "OBSTACLE FACTORY did not create any obstacles.\n";
    return;
  }
  if (sim.rank == 0) {
    std::cout << "-------------   OBSTACLE FACTORY : START ("
              << factoryLines.size() << " objects)   ------------\n";
  }

  for (auto & l : factoryLines)
    sim.obstacle_vector->addObstacle(_createObstacle(sim, l.first, l.second));

  if (sim.rank == 0)
    std::cout << "-------------   OBSTACLE FACTORY : END   ------------" << std::endl;
}

void ObstacleFactory::addObstacles(cubism::ArgumentParser &parser)
{
  // Read parser information
  parser.unset_strict_mode();
  const std::string factory_filename = parser("-factory").asString("factory");
  const std::string factory_content = parser("-factory-content").asString("");

  std::stringstream stream(factory_content);
  if (!factory_filename.empty()) {
    // https://stackoverflow.com/questions/132358/how-to-read-file-content-into-istringstream
    // Good enough solution.
    std::ifstream file(factory_filename);
    if (file.is_open()) {
      stream << '\n';
      stream << file.rdbuf();
    }
  }

  _addObstacles(sim, stream);
}

void ObstacleFactory::addObstacles(const std::string &factoryContent)
{
  std::stringstream stream(factoryContent);
  _addObstacles(sim, stream);
}

CubismUP_3D_NAMESPACE_END
