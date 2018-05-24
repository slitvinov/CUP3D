//
//  CubismUP_3D
//
//  Written by Guido Novati ( novatig@ethz.ch ).
//  This file started as an extension of code written by Wim van Rees
//  Copyright (c) 2017 ETHZ. All rights reserved.
//

#include <iostream>
#include <fstream>
#include "IF3D_ObstacleFactory.h"
#include "IF2D_FactoryFileLineParser.h"

//#include "IF3D_CarlingFishOperator.h"
//#include "IF3D_CylinderObstacleOperator.h"
#include "IF3D_DCylinderObstacleOperator.h"
//#include "IF3D_DeadFishOperator.h"
#include "IF3D_ExternalObstacleOperator.h"
#include "IF3D_NacaOperator.h"
#include "IF3D_PlateObstacleOperator.h"
#include "IF3D_SphereObstacleOperator.h"
#include "IF3D_StefanFishOperator.h"
#include "IF3D_VortexOperator.h"

/*
#include "IF3D_EllipseObstacleOperator.h"
#include "IF3D_CylinderPairOperator.h"
 */

std::vector<IF3D_ObstacleOperator*> IF3D_ObstacleFactory::create(ArgumentParser & parser)
{
  std::vector<IF3D_ObstacleOperator*> retval;

  // Read parser information
  parser.unset_strict_mode();
  const std::string factory_filename = parser("-factory").asString("factory");
  const std::string factory_content = parser("-factory-content").asString("");

  assert(factory_filename != "" || factory_content != "");

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
  if (rank == 0)
      printf("Factory (file) + factory (cmdline argument):\n%s\n\n", stream.str().c_str());
  // here we store the data per object
  std::vector<std::pair<std::string, IF2D_FactoryFileLineParser>> factoryLines;
  std::string line;

  while (std::getline(stream, line)) {
      std::istringstream line_stream(line);
      std::string ID;
      line_stream >> ID;
      if(ID.empty() or ID[0]=='#') continue; // Comments and empty lines ignored
      IF2D_FactoryFileLineParser ffparser(line_stream);
      factoryLines.push_back(make_pair(ID,ffparser));
  }
  if(rank==0)
  std::cout << "-------------   OBSTACLE FACTORY : START (" << factoryLines.size() <<" objects)   ------------" << std::endl;

  int k(0);
  for(auto & object : factoryLines)
  {
    std::string objectName = object.first;

      if( objectName == "IF3D_Sphere" )
    {
      retval.push_back(new IF3D_SphereObstacleOperator(grid,object.second,Uinf));
    }
    //else if( objectName == "IF3D_DeadFish" )
    //{
    //  retval.push_back(new IF3D_DeadFishOperator(grid,object.second,Uinf));
    //}
    else if( objectName == "IF3D_StefanFish" )
    {
      retval.push_back(new IF3D_StefanFishOperator(grid,object.second,Uinf));
    }
    //else if( objectName == "IF3D_CarlingFish" )
    //{
    //  retval.push_back(new IF3D_CarlingFishOperator(grid,object.second,Uinf));
    //}
    else if( objectName == "IF3D_NacaOperator" )
    {
      retval.push_back(new IF3D_NacaOperator(grid,object.second,Uinf));
    }
    else if( objectName == "IF3D_VortexOperator" )
    {
      retval.push_back(new IF3D_VortexOperator(grid,object.second,Uinf));
    }
    else if( objectName == "IF3D_DCylinder" )
    {
      retval.push_back(new IF3D_DCylinderObstacleOperator(grid,object.second,Uinf));
    }
    else if( objectName == "IF3D_PlateObstacle" )
    {
      retval.push_back(new IF3D_PlateObstacleOperator(grid,object.second,Uinf));
    }
    else if( objectName == "IF3D_ExternalObstacleOperator" )
    {
      retval.push_back(new IF3D_ExternalObstacleOperator(grid,object.second,Uinf));
    }
    /*
    else if( objectName == "IF3D_Ellipse" )
    {
        retval.push_back(new IF3D_EllipseObstacleOperator(grid,object.second,max_bpd));
    }
    else if( objectName == "IF3D_CylinderPair" )
    {
        retval.push_back(new IF3D_CylinderPairOperator(grid,object.second,max_bpd));
    }
    else if( objectName == "IF3D_DeadFish" )
    {
        retval.push_back(new IF3D_DeadFishOperator(grid,object.second,max_bpd));
    }
    else if( objectName == "IF3D_StefanTurn" )
    {
        retval.push_back(new IF3D_StefanTurnOperator(grid,object.second,max_bpd));
    }
    else if( objectName == "IF3D_StefanLearnTurn" )
    {
        retval.push_back(new IF3D_StefanLearnTurnOperator(grid,object.second,max_bpd));
    }
    else if( objectName == "IF3D_Cstart" )
    {
        retval.push_back(new IF3D_CstartOperator(grid,object.second,max_bpd));
    }
    else if( objectName == "IF3D_VortexGenerator" )
    {
        retval.push_back(new IF3D_GenerateVortexOperator(grid,object.second,max_bpd));
    }
     */
    else
    {
        if (rank == 0) {
            std::cout << "Case " << objectName << " is not defined: aborting" << std::endl;
            abort();
        }
    }

    retval.back()->obstacleID = k++;
  }

  if(rank==0)
  std::cout << "-------------   OBSTACLE FACTORY : END   ------------" << std::endl;

  return retval;
}
