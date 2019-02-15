//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ivica Kicic (kicici@ethz.ch) in May 2018.
//

/*
 * This obstacle can be used to insert obstacles whose shape and velocity is
 * defined by an external code. Intended to be used when CubismUP_3D used as a
 * library.
 */
#include "../CubismUP_3D.h"
#include "IF3D_ObstacleOperator.h"

#ifndef CUBISMUP3D_EXTERNAL_OBSTACLE_OPERATOR_H
#define CUBISMUP3D_EXTERNAL_OBSTACLE_OPERATOR_H

class IF3D_ExternalObstacleOperator : public IF3D_ObstacleOperator
{
public:
    cubismup3d::ExternalObstacleSettings settings;

    IF3D_ExternalObstacleOperator(
            SimulationData&s,
            const ObstacleArguments &args);
    IF3D_ExternalObstacleOperator(
            SimulationData&s, ArgumentParser &p)
        : IF3D_ExternalObstacleOperator(s, ObstacleArguments(s, p)) {}

    void computeVelocities(const Real *Uinf) override;
    void create(int step_id, double time, double dt, const Real *Uinf) override;
};

#endif