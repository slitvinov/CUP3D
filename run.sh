python launch/BallofFish.py --fish 2
. ./settingsEllipsoidSwarm1.sh
OPTIONS+=" -tdump 0.1 -tend 1.0"
OPTIONS+=" -dumpOmegaX 1 -dumpOmegaY 1 dumpOmegaZ 1"
OPTIONS+=" -dumpVelocityX 1 -dumpVelocityY 1 -dumpVelocityZ 1"
mpiexec ./main $OPTIONS -factory-content $FACTORY
