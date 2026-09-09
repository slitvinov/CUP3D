CUP3D: 3D incompressible Navier-Stokes solver with adaptive mesh
refinement for flows around swimming fish, written in C with MPI.

macOS
-----

Dependencies

    brew install open-mpi gcc make

Compile

    OMPI_CC=gcc-15 gmake

Run

    ./run.sh

Postprocessing

    ./tool/post.py vel.*.xdmf2

Coverage

    brew install gcovr
    OMPI_CC=gcc-15 GCOV=gcov-15 sh cover.sh
