//
//  TestRotation.cpp
//  CubismUP_3D
//
//  Created by Christian Conti on 3/16/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "TestRotation.h"
#include "ProcessOperatorsOMP.h"
#include "CoordinatorComputeShape.h"
#include "CoordinatorBodyVelocities.h"
#include <sstream>
#include <cmath>

void TestRotation::_ic()
{
	const Real center[3] = {.5,.5,.5};
	const Real rhoS = 2;
	const Real moll = 2;
	const int gridsize = 1024;
	const Real scale = .15;
	const Real tx = .425222;
	const Real ty = .455217;
	const Real tz = .377632;
	//Geometry::Quaternion q(1, 0, 0, 0);
	Geometry::Quaternion q(cos(.125*M_PI), 0, sin(.125*M_PI), 0);
	const string filename = "/cluster/home/infk/cconti/CubismUP_3D/launch/geometries/Samara_v3.obj";
	shape = new GeometryMesh(filename, gridsize, .0, center, rhoS, moll, moll, scale, tx, ty, tz, q);
	Real com[3];
	shape->getCenterOfMass(com);
	
	//const Real radius = .2;
	//const Real rhoS = 15./(8.*M_PI * radius*radius*radius*radius*radius);
	//shape = new Sphere(center, radius, rhoS, moll, moll);
	
	cout << "\tGeometricCoM:\t" << com[0] << " " << com[1] << " " << com[2] << endl;
	
	vector<BlockInfo> vInfo = grid->getBlocksInfo();
	
	const double dh = vInfo[0].h_gridpoint;
	
	Real cx = 0;
	Real cy = 0;
	Real cz = 0;
	Real vol = 0;
	
#pragma omp parallel for reduction(+:cx) reduction(+:cy) reduction(+:cz) reduction(+:vol)
	for(int i=0; i<(int)vInfo.size(); i++)
	{
		BlockInfo info = vInfo[i];
		FluidBlock& b = *(FluidBlock*)info.ptrBlock;
		
		for(int iz=0; iz<FluidBlock::sizeZ; iz++)
			for(int iy=0; iy<FluidBlock::sizeY; iy++)
				for(int ix=0; ix<FluidBlock::sizeX; ix++)
				{
					Real p[3];
					info.pos(p, ix, iy, iz);
					
					b(ix,iy,iz).chi = shape->chi(p, info.h_gridpoint);
					b(ix,iy,iz).rho = shape->rho(p, info.h_gridpoint, b(ix,iy,iz).chi);
					
					// this is for testing purposes only! do it the clean way!!
					b(ix,iy,iz).p = 0;
					b(ix,iy,iz).divU = 0;
					b(ix,iy,iz).pOld = 0;
					
					const Real rhochi = b(ix,iy,iz).rho * b(ix,iy,iz).chi;
					cx += p[0] * rhochi * dh*dh*dh;
					cy += p[1] * rhochi * dh*dh*dh;
					cz += p[2] * rhochi * dh*dh*dh;
					vol += rhochi * dh*dh*dh;
				}
	}
	
	cx /= vol;
	cy /= vol;
	cz /= vol;
	
	cout << "\tGridCM:\t" << cx << " " << cy << " " << cz << endl;
	com[0] = cx;
	com[1] = cy;
	com[2] = cz;
	shape->setCenterOfMass(com);
	shape->setCentroid(com);
	
#pragma omp parallel for
	for(int i=0; i<(int)vInfo.size(); i++)
	{
		BlockInfo info = vInfo[i];
		FluidBlock& b = *(FluidBlock*)info.ptrBlock;
		
		for(int iz=0; iz<FluidBlock::sizeZ; iz++)
			for(int iy=0; iy<FluidBlock::sizeY; iy++)
				for(int ix=0; ix<FluidBlock::sizeX; ix++)
				{
					Real p[3];
					info.pos(p, ix, iy, iz);
					
					const Real dist[3] = {p[0]-com[0],p[1]-com[1],p[2]-com[2]};
					
					// this is for testCase==1, no effect on testCase==0
					//b(ix,iy,iz).u = -dist[1] * 2 * M_PI;
					//b(ix,iy,iz).v =  dist[0] * 2 * M_PI;
					//b(ix,iy,iz).w = 0;
					
					b(ix,iy,iz).u = -dist[2] * 2 * M_PI;
					b(ix,iy,iz).v = 0;
					b(ix,iy,iz).w =  dist[0] * 2 * M_PI;
					
					//b(ix,iy,iz).u = 0;
					//b(ix,iy,iz).v =  dist[2] * 2 * M_PI;
					//b(ix,iy,iz).w = -dist[1] * 2 * M_PI;
				}
	}
	
	stringstream ss;
	ss << path2file << "-" << bpd << "-IC.vti";
	
	dumper.Write(*grid, ss.str());
}

TestRotation::TestRotation(const int argc, const char ** argv, const int testCase, const int bpd, const double dt) : Test(argc,argv), testCase(testCase), bpd(bpd), dt(1./100.), nsteps(100)
{
	grid = new FluidGrid(bpd,bpd,bpd);
	
	path2file = parser("-file").asString("../data/testRotation");
	_ic();
	
}

TestRotation::~TestRotation()
{
	delete grid;
}

void TestRotation::run()
{
	const int sizeX = bpd * FluidBlock::sizeX;
	const int sizeY = bpd * FluidBlock::sizeY;
	const int sizeZ = bpd * FluidBlock::sizeZ;
	
	Real u[3] = {0,0,0};
	Real lambda = 1;
	Real maxU = 0;
	CoordinatorComputeShape coordComputeShape(shape, grid);
	CoordinatorBodyVelocities coordBodyVelocities(&u[0], &u[1], &u[2], &lambda, shape, &maxU, grid);
	
	Real rotation[3][3];
	shape->getOrientation(rotation);
	cout << "Orientation: " << setprecision(4) << rotation[0][0] << " " << rotation[0][1] << " " << rotation[0][2] << " " << rotation[1][0] << " " << rotation[1][1] << " " << rotation[1][2] << " " << rotation[2][0] << " " << rotation[2][1] << " " << rotation[2][2] << " " << endl;
	
	for (int step=0; step<nsteps; step++)
	{
		vector<BlockInfo> vInfo = grid->getBlocksInfo();
		
		if (testCase==0)
		{
			u[0] = 0;
			u[1] = 0;
			u[2] = 0;
			const Real mass = 1;
			const Real dthetadt[3] = { 0, -2*M_PI, 0 };
			const Real J[6] = { 1,1,1,0,0,0 };
			shape->updatePosition(u, dthetadt, J, mass, dt);
		}
		else if (testCase==1)
			coordBodyVelocities(dt);
		else
			abort();
		
		coordComputeShape(dt);
		//*
		if (step%5==0)
		{
			shape->getOrientation(rotation);
			cout << "Orientation (step " << step << "): " << rotation[0][0] << " " << rotation[0][1] << " " << rotation[0][2] << " " << rotation[1][0] << " " << rotation[1][1] << " " << rotation[1][2] << " " << rotation[2][0] << " " << rotation[2][1] << " " << rotation[2][2] << " " << endl;
			
			stringstream ss;
			ss << path2file << "-" << bpd << "-" << step << ".vti";
			
			dumper.Write(*grid, ss.str());
		}
		//*/
	}
}

void TestRotation::check()
{
	// the only thing to check here is the orientation
	Real rotation[3][3];
	shape->getOrientation(rotation);
	//cout << "Orientation error: " << shape->getOrientation() - M_PI/4. << endl;
	cout << "Orientation error: " << 1-rotation[0][0] << " " << 0-rotation[0][1] << " " << 0-rotation[0][2] << " " << 0-rotation[1][0] << " " << 1-rotation[1][1] << " " << 0-rotation[1][2] << " " << 0-rotation[2][0] << " " << 0-rotation[2][1] << " " << 1-rotation[2][2] << " " << endl;
	//cout << "Orientation error: " << rotation[0][0] << " " << rotation[0][1] << " " << rotation[0][2] << " " << rotation[1][0] << " " << rotation[1][1] << " " << rotation[1][2] << " " << rotation[2][0] << " " << rotation[2][1] << " " << rotation[2][2] << " " << endl;
}