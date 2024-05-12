exec mpiexec ./main \
     -bMeanConstraint 2 \
     -bpdx 8 \
     -bpdy 4 \
     -bpdz 4 \
     -CFL 0.4 \
     -Ctol 0.1 \
     -dumpOmegaX 1 \
     -dumpOmegaY 1 \
     -dumpOmegaZ 1 \
     -dumpVelocityX 1 \
     -dumpVelocityY 1 \
     -dumpVelocityZ 1 \
     -extentx 8 \
     -factory-content \
     'StefanFish L=0.2 T=1.0 xpos=3.4 ypos=1.5 zpos=1.8 CorrectPosition=true CorrectZ=true CorrectRoll=true heightProfile=danio widthProfile=stefan bFixFrameOfRef=1
      StefanFish L=0.2 T=1.0 xpos=3.9 ypos=1.7 zpos=1.5 CorrectPosition=true CorrectZ=true CorrectRoll=true heightProfile=danio widthProfile=stefan
      StefanFish L=0.2 T=1.0 xpos=3.7 ypos=2.0 zpos=1.6 CorrectPosition=true CorrectZ=true CorrectRoll=true heightProfile=danio widthProfile=stefan' \
     -levelMax 5 \
     -levelStart 4 \
     -nu 0.00001 \
     -poissonSolver iterative \
     -Rtol 5 \
     -tdump 0.1 \
     -tend 1.0 \
