//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and
//  Hugues de Laroussilhe (huguesdelaroussilhe@gmail.com).
//

#include "smarties.h"

#include "Simulation.h"
#include "operators/SGS_RL.h"
#include "spectralOperators/SpectralManip.h"
#include "spectralOperators/HITtargetData.h"

#include <Cubism/ArgumentParser.h>
#include <Cubism/Profiler.h>

#include <sys/unistd.h> // hostname
#include <sys/stat.h>  // mkdir options
#include <unistd.h>   // chdir
#include <sstream>

using Real = cubismup3d::Real;

inline bool isTerminal(cubismup3d::SimulationData& sim)
{
  std::atomic<bool> bSimValid { true };

  const auto& vInfo = sim.vInfo();
  const auto isNotValid = [](const Real val) {
    return std::fabs(val) > 1e3;
  };

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < vInfo.size(); ++i) {
    const cubismup3d::FluidBlock&b= *(cubismup3d::FluidBlock*)vInfo[i].ptrBlock;
    for(int iz=0; iz<cubismup3d::FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<cubismup3d::FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<cubismup3d::FluidBlock::sizeX; ++ix)
      if ( isNotValid(b(ix,iy,iz).u) || isNotValid(b(ix,iy,iz).v) ||
           isNotValid(b(ix,iy,iz).w) )
        bSimValid = false;
  }
  int isSimValid = bSimValid.load() == true? 1 : 0; // just for clarity
  MPI_Allreduce(MPI_IN_PLACE, &isSimValid, 1, MPI_INT, MPI_PROD, sim.grid->getCartComm());
  if (isSimValid == 0) printf("field exploded\n");
  return isSimValid == 0;
}

inline void app_main(
  smarties::Communicator*const comm, // communicator with smarties
  MPI_Comm mpicom,                  // mpi_comm that mpi-based apps can use
  int argc, char**argv             // args read from app's runtime settings file
)
{
  // print received arguments:
  for(int i=0; i<argc; i++) {printf("arg: %s\n",argv[i]); fflush(0);}

  #ifdef CUP_ASYNC_DUMP
    const auto SECURITY = MPI_THREAD_MULTIPLE;
  #else
    const auto SECURITY = MPI_THREAD_FUNNELED;
  #endif
  int provided; MPI_Query_thread(&provided);
  if (provided < SECURITY ) {
    printf("ERROR: MPI implementation does not have required thread support\n");
    fflush(0); MPI_Abort(mpicom, 1);
  }
  int rank; MPI_Comm_rank(mpicom, &rank);
  //int wrank; MPI_Comm_rank(MPI_COMM_WORLD, &wrank);

  if (rank==0) {
    char hostname[1024];
    hostname[1023] = '\0';
    gethostname(hostname, 1023);
    std::cout <<
    "=======================================================================\n";
    std::cout <<
    "Cubism UP 3D (velocity-pressure 3D incompressible Navier-Stokes solver)\n";
    std::cout <<
    "=======================================================================\n";
    #ifdef NDEBUG
        std::cout<<"Running on "<<hostname<<"in RELEASE mode!\n";
    #else
        std::cout<<"Running on "<<hostname<<"in DEBUG mode!\n";
    #endif
  }

  cubism::ArgumentParser parser(argc, argv);
  cubismup3d::Simulation sim(mpicom, parser);
  const int maxGridN = sim.sim.local_bpdx * CUP_BLOCK_SIZE;
  cubismup3d::HITtargetData target(maxGridN, parser("-initCondFileTokens").asString());
  const Real LES_RL_FREQ_A = parser("-RL_freqActions").asDouble( 4.0);
  const Real fac = 2.5 / std::sqrt(LES_RL_FREQ_A);
  const Real LES_RL_N_TSIM = parser("-RL_nIntTperSim").asDouble(20.0) * fac;
  const bool bGridAgents = parser("-RL_gridPointAgents").asInt(0);

  target.smartiesFolderStructure = true;
  cubism::Profiler profiler;

  const int nActions = 1, nStates = cubismup3d::SGS_RL::nStateComponents();
  // BIG TROUBLE WITH NAGENTS!
  // If every grid point is an agent: probably will allocate too much memory
  // and crash because smarties allocates a trajectory for each point
  // If only one agent: sequences will be garbled together and cannot
  // send clean Sequences.
  // Also, rememebr that separate agents are thread safe!
  // let's say that each fluid block has one agent
  const int nAgentPerBlock = 1;
  const int nBlock=sim.sim.local_bpdx * sim.sim.local_bpdy * sim.sim.local_bpdz;
  const int nAgents = nBlock * nAgentPerBlock; // actual learning agents
  const int nThreadSafetyAgents = omp_get_max_threads();
  comm->setStateActionDims(nStates, nActions);
  comm->setNumAgents(nAgents + nThreadSafetyAgents);

  const std::vector<double> lower_act_bound{0.04}, upper_act_bound{0.08};
  comm->setActionScales(upper_act_bound, lower_act_bound, false);
  comm->disableDataTrackingForAgents(nAgents, nAgents + nThreadSafetyAgents);
  comm->agentsShareExplorationNoise();

  comm->finalizeProblemDescription(); // required for thread safety

  if( comm->isTraining() ) { // disable all dumping. //  && wrank != 1
    sim.sim.b3Ddump = false; sim.sim.muteAll  = true;
    sim.sim.b2Ddump = false; sim.sim.saveFreq = 0;
    sim.sim.verbose = false; sim.sim.saveTime = 0;
  }

  char dirname[1024]; dirname[1023] = '\0';
  unsigned sim_id = 0, tot_steps = 0;

  // Terminate loop if reached max number of time steps. Never terminate if 0
  while(true) // train loop
  {
    // avoid too many unneeded folders:
    if(sim_id == 0 || not comm->isTraining()) { //  || wrank == 1
      sprintf(dirname, "run_%08u/", sim_id);
      printf("Starting a new sim in directory %s\n", dirname);
      mkdir(dirname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      chdir(dirname);
    }

    assert(sim.sim.spectralManip not_eq nullptr);
    const cubismup3d::HITstatistics & stats = sim.sim.spectralManip->stats;

    target.sampleParameters(comm->getPRNG());
    assert(target.holdsTargetData == true);
    sim.sim.enInjectionRate = target.eps;
    sim.sim.nu = target.nu;
    sim.sim.spectralIC = "fromFile";
    sim.sim.initCondModes = target.mode;
    sim.sim.initCondSpectrum = target.E_mean;
    const Real tau_integral = target.tInteg;
    const Real tau_eta = stats.getKolmogorovT(target.epsVis, target.nu);
    const Real timeUpdateLES = tau_eta / LES_RL_FREQ_A;
    const Real timeSimulationMax = LES_RL_N_TSIM * tau_integral;
    const int maxNumUpdatesPerSim = timeSimulationMax / timeUpdateLES;
    printf("Reset simulation up to time=0 with SGS for eps:%f nu:%f Re:%f. "
           "Max %d action turns per simulation.\n", target.eps,
           target.nu, target.Re_lambda(), maxNumUpdatesPerSim);

    profiler.push_start("init");
    while(true) { // initialization loop
      sim.reset();
      bool ICsuccess = true;
      for (int prelim_step = 0; prelim_step < 2; ++prelim_step) {
        sim.sim.sgs = "SSM";
        sim.sim.nextAnalysisTime = 0;
        sim.timestep( sim.calcMaxTimestep() );
        if ( isTerminal( sim.sim ) ) { ICsuccess = false; break; }
      }
      if( ICsuccess ) break;
      printf("failed, try new IC\n");
    }
    profiler.pop_stop();

    int step = 0;
    double avgReward = 0;
    bool policyFailed = false;
    sim.sim.sgs = "RLSM";

    cubismup3d::SGS_RL updateLES(sim.sim, comm, nAgentPerBlock);

    while (true) //simulation loop
    {
      const bool timeOut = step >= maxNumUpdatesPerSim;
      // even if timeOut call updateLES to send all the states of last step
      profiler.push_start("rl");
      // Sum of rewards should not have to change when i change action freq
      // or num of integral time steps for sim. 40 is the reference value:
      const double r_t = std::exp(avgReward) / maxNumUpdatesPerSim;

      //printf("S:%e %e %e %e %e\n", stats.tke, stats.dissip_visc,
      //  stats.dissip_tot, stats.lambda, stats.l_integral); fflush(0);
      updateLES.run(sim.sim.dt, step==0, timeOut, stats, target, r_t, bGridAgents);
      profiler.pop_stop();

      if(timeOut) { profiler.printSummary(); profiler.reset(); break; }
      // old ver: seldom analyze was wrong because of exp average later
      // sim.sim.nextAnalysisTime = (step+1) * timeUpdateLES;
      profiler.push_start("sim");
      while ( sim.sim.time < (step+1) * timeUpdateLES )
      {
        // new ver: always analyze for updateReward to make sense
        sim.sim.nextAnalysisTime = sim.sim.time;
        const double dt = sim.calcMaxTimestep();
        if ( sim.timestep( dt ) ) { // if true sim has ended
          printf("Set -tend 0. This file decides the length of train sim.\n");
          assert(false); fflush(0); MPI_Abort(mpicom, 1);
        }
        // Average reward over integral time:
        target.updateReward(stats, dt / timeUpdateLES, avgReward);
        //printf("r:%Le %Le\n", target.computeLogP(stats),
        //  target.logPdenom - target.computeLogP(stats)); fflush(0);
        if ( isTerminal( sim.sim ) ) {
           policyFailed = true; break;
        }
      }
      profiler.pop_stop();
      step++;
      tot_steps++;

      if ( policyFailed ) {
        // Agent gets penalized if the simulations blows up. For KDE reward,
        // penal is -0.5 max_reward * (n of missing steps to finish the episode)
        // WARNING: not consistent with L2 norm reward
        const std::vector<double> S_T(nStates, 0); // values in S_T dont matter
        const double Nmax = maxNumUpdatesPerSim, R_T = 10 * (step - Nmax)/Nmax;
        printf("policy failed with rew %f after %d steps\n", R_T, step);
        for(int i=0; i<nAgents; ++i) comm->sendTermState(S_T, R_T, i);
        //printf("comm->sendTermState"); fflush(0);
        profiler.printSummary();
        profiler.reset();
        break;
      }
    } // simulation is done

    if(not comm->isTraining()) { //  || wrank == 1
      chdir("../"); // matches previous if
    }

    sim_id++;
  }
}

int main(int argc, char**argv)
{
  smarties::Engine e(argc, argv);
  if( e.parse() ) return 1;
  // print only one agent per simulation:
  e.setIsLoggingAllData(2);
  e.run( app_main );
  return 0;
}
