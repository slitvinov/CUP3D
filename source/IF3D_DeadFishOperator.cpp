//
//  IF3D_CarlingFishOperator.cpp
//  IncompressibleFluids3D
//
//  Created by Wim van Rees on 4/15/13.
//
//

#include "IF3D_CarlingFishOperator.h"
#include "IF3D_FishLibrary.h"
#include "GenericOperator.h"

IF3D_CarlingFishOperator::IF3D_CarlingFishOperator(FluidGridMPI * grid, ArgumentParser & parser)
: IF3D_FishOperator(grid, parser), ext_pos{0,0,0}
{
	_parseArguments(parser);
	const Real target_Nm = TGTPPB*length/vInfo[0].h_gridpoint;
	const Real dx_extension = 0.25*vInfo[0].h_gridpoint;
	const int Nm = NPPSEG*(int)std::ceil(target_Nm/NPPSEG)+1;
	printf("%d %f %f %f %f\n",Nm,length,Tperiod,phaseShift,dx_extension);
	fflush(0);
	// multiple of NPPSEG: TODO why?
	myFish = new CarlingFishMidlineData(Nm, length, Tperiod, phaseShift, dx_extension, 0.);
}

void IF3D_DeadFishOperator::_parseArguments(ArgumentParser & parser)
{
	IF3D_FishOperator::_parseArguments(parser);
	parser.set_strict_mode();
	ID = parser("-DID").asInt();

	parser.unset_strict_mode();
	P0 = parser("-xpos").asDouble(0.0);
	VelX = parser("-VelX").asDouble(0.0);
	Ltow = parser("-Ltow").asDouble(2.5);
	Ttow = parser("-Ttow").asDouble(10.0);
	Atow = parser("-Atow").asDouble(-1.);

	if (Atow>0) {
		transVel[0] = 2*Ltow*length/(Ttow*Tperiod);
		if (ID==0) position[1] = 0.5 - Atow*length;
		if (ID==1) position[1] = 0.5 + Atow*length;
		if (ID==1) transVel[0] -= 4*Ltow*length/(Ttow*Tperiod);
	}
	printf("created IF2D_DeadFish: xpos=%f ypos=%f L=%f T=%\n",position[0],position[1],length,Tperiod);
	printf("P0=%3.3f VelX=%3.3f Ltow=%3.3f Ttow=%3.3f Atow=%3.3f ID=%d \n",P0,VelX,Ltow,Ttow,Atow,ID);
}

void IF3D_DeadFishOperator::update(const int step_id, const Real t, const Real dt, const Real *Uinf)
{
	if (Atow>0) {
		//constant acceleration
		double accel = 4*Ltow*length/(Ttow*Tperiod)/(Ttow*Tperiod);
		//alternates between negative and positive
		if (fmod(t/Tperiod,2.*Ttow)>Ttow) accel=-accel;
		double s_c = 1.0;
		//2 obstacles in antiphase
		if (ID==0) {
			accel=-accel;
			s_c=-s_c;
		}
		//integrate in time constant accel:
		position[0] += dt*transVel[0] + 0.5*accel*dt*dt;
		transVel[0] += accel*dt;
		//why here?
		ext_pos[0] += dt*(transVel[0]-2*Ltow*length/(Ttow*Tperiod)) + 0.5*accel*dt*dt;
		const Real arg = .5*M_PI*(P0-ext_pos[0])/Ltow/length;
		position[1] = 0.5 + s_c*Atow*length*std::cos(arg);
		const double fac1  = .5*s_c*Atow*M_PI/Ltow*sin(arg);
		transVel[1] = fac1*transVel[0];
		ext_pos[1] += dt*(transVel[1]-Uinf[1]);
		angle = atan(fac1);
		const double fac2  = -0.25*Atow*s_c*M_PI*M_PI/Ltow*cos(arg)*transVel[0]/Ltow/length;
		angVel[2] = fac2/(1+fac1*fac1);

		const Real dqdt[4] = {
				0.5*(-angVel[2]*quaternion[3]),
				0.5*(-angVel[2]*quaternion[2]),
				0.5*(+angVel[2]*quaternion[1]),
				0.5*(+angVel[2]*quaternion[0])
		};
		const Real deltaq[4] = {dqdt[0]*dt, dqdt[1]*dt, dqdt[2]*dt, dqdt[3]*dt};
		const Real deltaq_length = std::sqrt(deltaq[0]*deltaq[0]+deltaq[1]*deltaq[1]+deltaq[2]*deltaq[2]+deltaq[3]*deltaq[3]);
		if(deltaq_length>std::numeric_limits<Real>::epsilon()) {
			const Real tanfac = std::tan(deltaq_length)/deltaq_length;
			const Real num[4] = {
					quaternion[0]+tanfac*deltaq[0],
					quaternion[1]+tanfac*deltaq[1],
					quaternion[2]+tanfac*deltaq[2],
					quaternion[3]+tanfac*deltaq[3],
			};

			const Real invDenum = 1./(std::sqrt(num[0]*num[0]+num[1]*num[1]+num[2]*num[2]+num[3]*num[3]));
			quaternion[0] = num[0]*invDenum;
			quaternion[1] = num[1]*invDenum;
			quaternion[2] = num[2]*invDenum;
			quaternion[3] = num[3]*invDenum;
		}

		printf("accel %f, posx %f, posy %f. velx %f, vely %f, angvel %f\n",
				accel,position[0],position[1],transVel[0],transVel[1],angVel[2]);
		//position[1] = 0.5 + Yamplit*sin(2*M_PI*(position[0]-P0)/Yperiod + M_PI*Yphase);
		//angle = (bTilt) ? angle + atan(2*M_PI*Yamplit*cos(2*M_PI*(position[0]-P0)/Yperiod + M_PI*Yphase)/Yperiod) : angle;
		//cout << Atow << length << P0-position[0]<<endl;

		_writeComputedVelToFile(step_id, t, Uinf);
		if (position[0]<0.1) abort();
	}
}

void IF3D_DeadFishOperator::computeVelocities(const Real* Uinf)
{
    Real CM[3];
    this->getCenterOfMass(CM);
    const int N = vInfo.size();

    //local variables
    Real V(0.0), J0(0.0), J1(0.0), J2(0.0), J3(0.0), J4(0.0), J5(0.0);
    Real lm0(0.0), lm1(0.0), lm2(0.0); //linear momenta
    Real am0(0.0), am1(0.0), am2(0.0); //angular momenta

#pragma omp parallel for schedule(static) reduction(+:V,lm0,lm1,lm2,J0,J1,J2,J3,J4,J5,am0,am1,am2)
    for(int i=0; i<vInfo.size(); i++) {
        BlockInfo info = vInfo[i];
        FluidBlock & b = *(FluidBlock*)info.ptrBlock;
        const auto pos = obstacleBlocks.find(info.blockID);
        if(pos == obstacleBlocks.end()) continue;

		for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
            const Real Xs = pos->second->chi[iz][iy][ix];
            if (Xs == 0) continue;
            Real p[3];
            info.pos(p, ix, iy, iz);
            p[0]-=CM[0];
            p[1]-=CM[1];
            p[2]-=CM[2];
            V     += Xs;
            lm0   += Xs * b(ix,iy,iz).u;
            lm1   += Xs * b(ix,iy,iz).v;
            lm2   += Xs * b(ix,iy,iz).w;
            am0 += Xs*(p[1]*b(ix,iy,iz).w - p[2]*b(ix,iy,iz).v);
            am1 += Xs*(p[2]*b(ix,iy,iz).u - p[0]*b(ix,iy,iz).w);
            am2 += Xs*(p[0]*b(ix,iy,iz).v - p[1]*b(ix,iy,iz).u);
            J0+=Xs*(p[1]*p[1]+p[2]*p[2]);
            J1+=Xs*(p[0]*p[0]+p[2]*p[2]);
            J2+=Xs*(p[0]*p[0]+p[1]*p[1]);
            J3-=Xs*p[0]*p[1];
            J4-=Xs*p[0]*p[2];
            J5-=Xs*p[1]*p[2];
        }
    }

    double globals[13];
    double locals[13] = {lm0,lm1,lm2,am0,am1,am2,J0,J1,J2,J3,J4,J5,V};
	MPI::COMM_WORLD.Allreduce(locals, globals, 13, MPI::DOUBLE, MPI::SUM);

    assert(globals[12] > std::numeric_limits<double>::epsilon());
    const Real dv = std::pow(vInfo[0].h_gridpoint,3);
    transVel_comp[0] = globals[0]/globals[12] + Uinf[0];
    transVel_comp[1] = globals[1]/globals[12] + Uinf[1];
    transVel_comp[2] = globals[2]/globals[12] + Uinf[2];
    Real tmp_AV[3] = {
    		globals[3]* dv,
			globals[4]* dv,
			globals[5]* dv
    };
    J[0] = globals[6] * dv;
    J[1] = globals[7] * dv;
    J[2] = globals[8] * dv;
    J[3] = globals[9] * dv;
    J[4] = globals[10]* dv;
    J[5] = globals[11]* dv;
    volume = globals[12] * dv;

    _finalizeAngVel(angVel_comp, J, tmp_AV[0], tmp_AV[1], tmp_AV[1]);
}

void IF3D_DeadFishOperator::save(const int stepID, const Real t, string filename)
{
    //assert(std::abs(t-sim_time)<std::numeric_limits<Real>::epsilon());
    std::ofstream savestream;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<Real>::digits10 + 1);
    savestream.open(filename + ".txt");

    savestream<<t<<"\t"<<sim_dt<<std::endl;
    savestream<<position[0]<<"\t"<<position[1]<<"\t"<<position[2]<<std::endl;
    savestream<<ext_pos[0]<<"\t"<<ext_pos[1]<<"\t"<<ext_pos[2]<<std::endl;
    savestream<<quaternion[0]<<"\t"<<quaternion[1]<<"\t"<<quaternion[2]<<"\t"<<quaternion[3]<<std::endl;
    savestream<<transVel[0]<<"\t"<<transVel[1]<<"\t"<<transVel[2]<<std::endl;
    savestream<<angVel[0]<<"\t"<<angVel[1]<<"\t"<<angVel[2]<<std::endl;
    savestream<<theta_internal<<"\t"<<angvel_internal<<"\t"<<adjTh<<std::endl;
    savestream.close();

}

void IF3D_DeadFishOperator::restart(const Real t, string filename)
{
    std::ifstream restartstream;
    restartstream.open(filename+".txt");
    restartstream >> sim_time >> sim_dt;
    assert(std::abs(sim_time-t) < std::numeric_limits<Real>::epsilon());
    restartstream >> position[0] >> position[1] >> position[2];
    restartstream >> ext_pos[0] >> ext_pos[1] >> ext_pos[2];
    restartstream >> quaternion[0] >> quaternion[1] >> quaternion[2] >> quaternion[3];
    restartstream >> transVel[0] >> transVel[1] >> transVel[2];
    restartstream >> angVel[0] >> angVel[1] >> angVel[2];
    restartstream >> theta_internal >> angvel_internal >> adjTh;
    restartstream.close();

	std::cout<<"RESTARTED FISH: "<<std::endl;
	std::cout<<"TIME, DT: "<<sim_time<<" "<<sim_dt<<std::endl;
	std::cout<<"POS: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<std::endl;
	std::cout<<"ANGLE: "<<quaternion[0]<<" "<<quaternion[1]<<" "<<quaternion[2]<<" "<<quaternion[3]<<std::endl;
	std::cout<<"TVEL: "<<transVel[0]<<" "<<transVel[1]<<" "<<transVel[2]<<std::endl;
	std::cout<<"AVEL: "<<angVel[0]<<" "<<angVel[1]<<" "<<angVel[2]<<std::endl;
	std::cout<<"INTERN: "<<theta_internal<<" "<<angvel_internal<<std::endl;
}