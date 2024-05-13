OMP_NUM_THREADS=1 exec mpiexec ${exe=./main} \
     -bMeanConstraint 2 \
     -bpdx 1 \
     -bpdy 1 \
     -bpdz 1 \
     -CFL 0.4 \
     -Ctol 0.1 \
     -extentx 1 \
     -factory-content \
     'StefanFish L=0.2 T=1.0 xpos=0.4 ypos=0.5 zpos=0.25 CorrectPosition=true CorrectZ=true CorrectRoll=true heightProfile=danio widthProfile=stefan bFixFrameOfRef=1
      StefanFish L=0.2 T=1.0 xpos=0.4 ypos=0.5 zpos=0.50 CorrectPosition=true CorrectZ=true CorrectRoll=true heightProfile=danio widthProfile=stefan
      StefanFish L=0.2 T=1.0 xpos=0.4 ypos=0.5 zpos=0.75 CorrectPosition=true CorrectZ=true CorrectRoll=true heightProfile=danio widthProfile=stefan' \
     -levelMax 6 \
     -levelStart 4 \
     -nu 0.00001 \
     -poissonSolver iterative \
     -Rtol 5 \
     -tdump 1.0 \
     -tend 10.0 \
