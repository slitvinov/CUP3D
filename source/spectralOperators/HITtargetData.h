//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and
//  Hugues de Laroussilhe (huguesdelaroussilhe@gmail.com).
//

#ifndef CubismUP_3D_TargetData_h
#define CubismUP_3D_TargetData_h

#include <vector>
#include <sstream>

CubismUP_3D_NAMESPACE_BEGIN

struct HITtargetData
{
  const int myGridN;
  const int nyquist = myGridN/2, nBin = nyquist-1;
  std::vector<std::string> targetFiles_tokens;
  std::string active_token; // specifies eps/nu combo, taken from the list above

  // If we read the target data in order to compute the RL reward mind that
  // smarties will copy target data to a base run directory, there it will mkdir
  // a folder for each environment simulation and therein a subfolder for each
  // simulation run. Therefore, the path to the target data begins with '../../'
  bool smartiesFolderStructure = true;
  bool holdsTargetData = false;
  size_t nModes;
  std::vector<double> logE_mean, mode, E_mean;
  std::vector<std::vector<double>> logE_invCov;

  double eps, nu, tKinEn, epsVis, epsTot, lInteg, tInteg, avg_Du, std_Du;
  double logPdenom;

  double Re_lambda() const
  {
     const double uprime = std::sqrt(2.0/3.0 * tKinEn);
     const double lambda = std::sqrt(15 * nu / epsTot) * uprime;
     return uprime * lambda / nu;
  }

  void sampleParameters(std::mt19937& gen)
  {
    std::uniform_int_distribution<size_t> dist(0, targetFiles_tokens.size()-1);
    active_token = targetFiles_tokens[dist(gen)];
    readAll(active_token);
  }

  void readAll(const std::string paramspec)
  {
    holdsTargetData = true;
    readScalars(paramspec); // for a (eps,nu) read TKE, lambda, epsVisc...
    readMeanSpectrum(paramspec); // read also mean energy spectrum
    readInvCovSpectrum(paramspec); // read also covariance matrix of logE
  }

  void readScalars(const std::string paramspec)
  {
    std::string line; char arg[32]; double stdev, _dt;
    std::string fpath = smartiesFolderStructure? "../../" : "./";
    std::ifstream file(fpath + "scalars_" + paramspec);
    if (!file.is_open()) {
      printf("scalars FILE NOT FOUND\n");
      holdsTargetData = false;
      return;
    }

    std::getline(file, line);
    sscanf(line.c_str(), "%s %le", arg, &eps);
    assert(strcmp(arg, "eps") == 0);

    std::getline(file, line);
    sscanf(line.c_str(), "%s %le", arg, &nu);
    assert(strcmp(arg, "nu") == 0);

    std::getline(file, line);
    sscanf(line.c_str(), "%s %le %le", arg, &_dt, &stdev);
    assert(strcmp(arg, "dt") == 0);

    std::getline(file, line);
    sscanf(line.c_str(), "%s %le %le", arg, &tKinEn, &stdev);
    assert(strcmp(arg, "tKinEn") == 0);

    std::getline(file, line);
    sscanf(line.c_str(), "%s %le %le", arg, &epsVis, &stdev);
    assert(strcmp(arg, "epsVis") == 0);

    std::getline(file, line);
    sscanf(line.c_str(), "%s %le %le", arg, &epsTot, &stdev);
    assert(strcmp(arg, "epsTot") == 0);

    std::getline(file, line);
    sscanf(line.c_str(), "%s %le %le", arg, &lInteg, &stdev);
    assert(strcmp(arg, "lInteg") == 0);

    std::getline(file, line);
    sscanf(line.c_str(), "%s %le %le", arg, &tInteg, &stdev);
    assert(strcmp(arg, "tInteg") == 0);

    std::getline(file, line);
    sscanf(line.c_str(), "%s %le %le", arg, &avg_Du, &stdev);
    assert(strcmp(arg, "avg_Du") == 0);

    std::getline(file, line);
    sscanf(line.c_str(), "%s %le %le", arg, &std_Du, &stdev);
    assert(strcmp(arg, "std_Du") == 0);

    std::getline(file, line);
    sscanf(line.c_str(), "%s %le", arg, &stdev);
    assert(strcmp(arg, "ReLamd") == 0);

    std::getline(file, line);
    sscanf(line.c_str(), "%s %le", arg, &logPdenom);
    assert(strcmp(arg, "logPdenom") == 0);

    printf("Params eps:%e nu:%e with mean quantities: tKinEn=%e epsVis=%e "
           "epsTot=%e tInteg=%e lInteg=%e avg_Du=%e std_Du=%e\n", eps, nu,
            tKinEn, epsVis, epsTot, tInteg, lInteg, avg_Du, std_Du);
    fflush(0);
  }

  void readMeanSpectrum(const std::string paramspec)
  {
    std::string line;
    logE_mean.clear(); mode.clear();
    std::string fpath = smartiesFolderStructure? "../../" : "./";
    std::ifstream file(fpath + "spectrumLogE_" + paramspec);
    if (!file.is_open()) {
      printf("spectrumLogE FILE NOT FOUND\n");
      holdsTargetData = false;
      return;
    }

    while (std::getline(file, line)) {
        mode.push_back(0); logE_mean.push_back(0);
        sscanf(line.c_str(), "%le, %le", & mode.back(), & logE_mean.back());
    }
    nModes = mode.size();
    assert(myGridN <= (int) nModes);
    //for (size_t i=0; i<nModes; ++i) printf("%f %f\n", mode[i], logE_mean[i]);
    //fflush(0);
    E_mean = std::vector<double>(nModes);
    for(size_t i = 0; i<nModes; ++i) E_mean[i] = std::exp(logE_mean[i]);
  }

  void readInvCovSpectrum(const std::string paramspec)
  {
    std::string line;
    logE_invCov = std::vector<std::vector<double>>(
        nModes, std::vector<double>(nModes,0) );
    std::string fpath = smartiesFolderStructure? "../../" : "./";
    std::ifstream file(fpath + "invCovLogE_" + paramspec);
    if (!file.is_open()) {
      printf("invCovLogE FILE NOT FOUND\n");
      holdsTargetData = false;
      return;
    }

    int j = 0;
    while (std::getline(file, line)) {
        int i = 0;
        std::istringstream linestream(line);
        while (std::getline(linestream, line, ','))
          logE_invCov[j][i++] = std::stof(line);
        assert(i >= nBin);
        j++;
    }
    assert(j >= nBin);
  }

  HITtargetData(const int maxGridN, std::string params): myGridN(maxGridN),
    targetFiles_tokens(readTargetSpecs(params))
  {
  }

  static std::vector<std::string> readTargetSpecs(std::string paramsList)
  {
    std::stringstream ss(paramsList);
    std::vector<std::string> tokens;
    std::string item;
    while (getline(ss, item, ',')) tokens.push_back(item);
    return tokens;
  }

  long double computeLogArg(const HITstatistics& stats)
  {
    std::vector<double> logE(stats.nBin);
    for (int i=0; i<stats.nBin; ++i) logE[i] = std::log(stats.E_msr[i]);
    const long double fac = 0.5 / stats.nBin;
    long double dev = 0;
    for (int j=0; j<stats.nBin; ++j)
      for (int i=0; i<stats.nBin; ++i) {
        const long double dLogEi = logE[i] - logE_mean[i];
        const long double dLogEj = logE[j] - logE_mean[j];
        dev += fac * dLogEj * logE_invCov[j][i] * dLogEi;
      }
    //printf("got dE Cov dE = %Le\n", dev);
    // normalize with expectation of L2 squared norm of N(0,I) distrib vector:
    // E[X^2] = sum E[x^2] = sum Var[x] = trace I = nBin
    assert(dev >= 0);
    return dev;
  }

  long double computeLogP(const HITstatistics& stats)
  {
     return logPdenom - computeLogArg(stats);
  }

  void updateReward(const HITstatistics& stats, const Real alpha, Real& reward)
  {
    auto logarg = computeLogArg(stats);
    if (logarg > 10.0) logarg = 10.0 * std::sqrt(logarg / 10.0);
    //const long double arg = 1 - computeLogP(stats);
    //const long double newRew = arg > 1 ? 1 / arg : std::exp(1-arg);
    const long double newRew = logPdenom - logarg; // computeLogP(stats);
    //printf("Rt : %e, %e - %Le\n", newRew, logPdenom, dev);
    reward = (1-alpha) * reward + alpha * newRew;
  }
};

CubismUP_3D_NAMESPACE_END
#endif

#if 0
inline void updateGradScaling(const cubismup3d::SimulationData& sim,
                              const cubismup3d::HITstatistics& stats,
                              const Real timeUpdateLES,
                                    Real & scaling_factor)
{
  // Scale gradients with tau_eta corrected with eps_forcing and nu_sgs
  // tau_eta = (nu/eps)^0.5 but nu_tot  = nu + nu_sgs
  // and eps_tot = eps_nu + eps_numerics + eps_sgs = eps_forcing
  // tau_eta_corrected = tau_eta +
  //                   D(tau_eta)/Dnu * delta_nu + D(tau_eta)/Deps * delta_eps
  //                   = tau_eta * 1/2 * tau_eta *  delta_nu  / nu
  //                             - 1/2 * tau_eta * (delta_eps / eps) )
  //                   = tau_eta * (1 + nu_sgs/nu/2 - (eps_forcing-eps)/eps/2)
  // Average gradient scaling over time between LES updates
  const Real beta = sim.dt / timeUpdateLES;
  //const Real turbEnergy = stats.tke;
  const Real viscDissip = stats.dissip_visc, totalDissip = stats.dissip_tot;
  const Real avgSGS_nu = sim.nu_sgs, nu = sim.nu;
  const Real tau_eta_sim = std::sqrt(nu / viscDissip);
  const Real correction = 1.0 + 0.5 * (totalDissip - viscDissip) / viscDissip
                              + 0.5 * avgSGS_nu / nu;
                            //+ 0.5 * avgSGS_nu / std::sqrt(nu * viscDissip);
  const Real newScaling = tau_eta_sim * correction;
  if(scaling_factor < 0) scaling_factor = newScaling; // initialization
  else scaling_factor = (1-beta) * scaling_factor + beta * newScaling;
}
#endif