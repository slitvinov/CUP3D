#!/bin/bash

BASENAME=ForcedSphereStartup_07
NNODE=8
FFACTORY=factoryForcedSphereStartup

OPTIONS=
OPTIONS+=" -bpdx 64 -bpdy 32 -bpdz 32"
OPTIONS+=" -2Ddump 0"
OPTIONS+=" -tdump 0.025"
OPTIONS+=" -nprocsx ${NNODE}"
OPTIONS+=" -CFL 0.1"
OPTIONS+=" -uinfx 0.0"
OPTIONS+=" -length 0.1"
OPTIONS+=" -lambda 1e5"
OPTIONS+=" -nu 5e-5"
OPTIONS+=" -tend 8"