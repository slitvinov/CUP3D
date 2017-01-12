//
//  TestTravelingWave.h
//  CubismUP_3D
//
//  Created by Christian Conti on 4/28/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef __CubismUP_3D__TestTravelingWave__
#define __CubismUP_3D__TestTravelingWave__

#include "Test.h"
#include "GenericCoordinator.h"

class TestTravelingWave : public Test
{
protected:
	double nu;
	double dtCFL, dtLCFL, dtFourier;
	double time, endTime;
	int step;
	int rank, nprocs;
	
	string path2file;
	
	vector<GenericCoordinator *> pipeline;
	
	void _ic();
	void _analytical(Real x, Real y, Real z, double t, Real &u, Real &v, Real &w, Real &p);
	
public:
	TestTravelingWave(const int argc, const char ** argv, const int bpd);
	~TestTravelingWave();
	
	void run();
	void check();
};

#endif /* defined(__CubismUP_3D__TestTravelingWave__) */
