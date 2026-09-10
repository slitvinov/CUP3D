#!/bin/sh

OMP_NUM_THREADS=4 exec mpiexec -n 2 ./main \
     -bMeanConstraint 2 \
     -bpdx 1 \
     -bpdy 1 \
     -bpdz 1 \
     -CFL 0.4 \
     -Ctol 0.1 \
     -dt 0 \
     -extent 1 \
     -factory-content \
     'L=0.4 T=1.0 phi=0 amplitudeFactor=1 xpos=0.35 ypos=0.5 zpos=0.5 planarAngle=180 heightProfile=danio widthProfile=stefan bFixFrameOfRef_x=1 bFixFrameOfRef_y=1 bFixFrameOfRef_z=1 bForcedInSimFrame_x=0 bForcedInSimFrame_y=0 bForcedInSimFrame_z=0 xvel=0 yvel=0 zvel=0 bFixToPlanar=0 CorrectPosition=0 CorrectPositionZ=0 CorrectRoll=0 wyp=1 wzp=1
      L=0.4 T=1.0 phi=0 amplitudeFactor=1 xpos=0.6 ypos=0.5 zpos=0.5 planarAngle=0 heightProfile=danio widthProfile=stefan bFixFrameOfRef_x=0 bFixFrameOfRef_y=0 bFixFrameOfRef_z=0 bForcedInSimFrame_x=0 bForcedInSimFrame_y=0 bForcedInSimFrame_z=0 xvel=0 yvel=0 zvel=0 bFixToPlanar=0 CorrectPosition=0 CorrectPositionZ=0 CorrectRoll=0 wyp=1 wzp=1' \
     -lambda 1e6 \
     -levelMax 5 \
     -levelStart 3 \
     -nsteps 0 \
     -nu 0.001 \
     -poissonTol 1e-6 \
     -poissonTolRel 1e-4 \
     -rampup 100 \
     -Rtol 5 \
     -StaticObstacles 0 \
     -tdump 0.05 \
     -tend 0.2 \
     -uinfx 0 \
     -uinfy 0 \
     -uinfz 0 \
     -umax 10 \
     -use-dlm 0
