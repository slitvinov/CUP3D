OMP_NUM_THREADS=1 exec mpiexec -n 2 ./main \
     -bMeanConstraint 2 \
     -bpdx 1 \
     -bpdy 1 \
     -bpdz 1 \
     -CFL 0.4 \
     -Ctol 0.1 \
     -extentx 1 \
     -factory-content \
     'StefanFish L=0.4 T=1.0 xpos=0.2 ypos=0.5 zpos=0.5 planarAngle=180 heightProfile=danio widthProfile=stefan bFixFrameOfRef=1
      StefanFish L=0.4 T=1.0 xpos=0.7 ypos=0.5 zpos=0.5 heightProfile=danio widthProfile=stefan' \
     -levelMax 4 \
     -levelStart 3 \
     -nu 0.001 \
     -poissonSolver iterative \
     -Rtol 5 \
     -tdump 0.02 \
     -tend 0.2 \
