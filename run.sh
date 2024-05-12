python fish.py --fish 3
. ./fish.sh
exec mpiexec ./main -extentx 8 -levelMax 7 -levelStart 4 -Rtol 5 -Ctol 0.1 -bMeanConstraint 2 -bpdx 8 -bpdy 4 -bpdz 4 -nu 0.00001 -CFL 0.4 -poissonSolver iterative \
     -tdump 0.1 -tend 1.0 -dumpOmegaX 1 -dumpOmegaY 1 dumpOmegaZ 1 -dumpVelocityX 1 -dumpVelocityY 1 -dumpVelocityZ 1 \
     -factory-content $FACTORY
