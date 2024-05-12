exec mpiexec ./main \
     -bMeanConstraint 2 \
     -bpdx 4 \
     -bpdy 2 \
     -bpdz 2 \
     -CFL 0.4 \
     -Ctol 0.1 \
     -dumpOmegaX 1 \
     -dumpOmegaY 1 \
     -dumpOmegaZ 1 \
     -extentx 4 \
     -factory-content \
     'StefanFish L=0.2 T=1.0 xpos=1 ypos=0.5 zpos=1 CorrectPosition=true CorrectZ=true CorrectRoll=true heightProfile=danio widthProfile=stefan bFixFrameOfRef=1
      StefanFish L=0.2 T=1.0 xpos=1 ypos=1.5 zpos=1 CorrectPosition=true CorrectZ=true CorrectRoll=true heightProfile=danio widthProfile=stefan' \
     -levelMax 6 \
     -levelStart 4 \
     -nu 0.00001 \
     -poissonSolver iterative \
     -Rtol 5 \
     -tdump 0.1 \
     -tend 1.0 \
