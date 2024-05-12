python fish.py --fish 3
. ./fish.sh
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
     -factory-content $FACTORY \
     -levelMax 7 \
     -levelStart 4 \
     -nu 0.00001 \
     -poissonSolver iterative \
     -Rtol 5 \
     -tdump 0.1 \
     -tend 1.0 \
