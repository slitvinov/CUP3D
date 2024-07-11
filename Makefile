.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp
.SUFFIXES: .o

MPICXX = h5c++.mpich
GSL_CFLAGS = `pkg-config --cflags gsl`
GSL_LDFLAGS = `pkg-config --libs gsl`
CUBISMFLAGS = \
-DCUP_ALIGNMENT=64 \
-DCUP_BLOCK_SIZEX=8 \
-DCUP_BLOCK_SIZEY=8 \
-DCUP_BLOCK_SIZEZ=8 \
-DCUP_NO_MACROS_HEADER \
-DDIMENSION=3 \
-D_DOUBLE_PRECISION_ \
-DNDEBUG \
-I. \
-O3 \
-std=c++17 \

S = \
source/ArgumentParser.cpp \
source/main.cpp \
source/Obstacles/CarlingFish.cpp \
source/Obstacles/Cylinder.cpp \
source/Obstacles/CylinderNozzle.cpp \
source/Obstacles/Ellipsoid.cpp \
source/Obstacles/ExternalObstacle.cpp \
source/Obstacles/Fish.cpp \
source/Obstacles/FishLibrary.cpp \
source/Obstacles/FishShapes.cpp \
source/Obstacles/Naca.cpp \
source/Obstacles/Obstacle.cpp \
source/Obstacles/ObstacleFactory.cpp \
source/Obstacles/Pipe.cpp \
source/Obstacles/Plate.cpp \
source/Obstacles/SmartNaca.cpp \
source/Obstacles/Sphere.cpp \
source/Obstacles/StefanFish.cpp \
source/operators/AdvectionDiffusion.cpp \
source/operators/AdvectionDiffusionImplicit.cpp \
source/operators/ComputeDissipation.cpp \
source/operators/ExternalForcing.cpp \
source/operators/FixMassFlux.cpp \
source/operators/FluidSolidForces.cpp \
source/operators/InitialConditions.cpp \
source/operators/ObstaclesCreate.cpp \
source/operators/ObstaclesUpdate.cpp \
source/operators/Penalization.cpp \
source/operators/PressureProjection.cpp \
source/poisson/DiffusionSolverAMRKernels.cpp \
source/poisson/PoissonSolverAMR.cpp \
source/poisson/PoissonSolverAMRKernels.cpp \
source/poisson/PoissonSolverBase.cpp \
source/Simulation.cpp \
source/SimulationData.cpp \
source/Utils/BufferedLogger.cpp \

main: $(S:.cpp=.o)
	$(MPICXX) -o main $(S:.cpp=.o) $(GSL_LDFLAGS) $(LDFLAGS) -fopenmp
.cpp.o:
	$(MPICXX) -o $@ -c $< $(CUBISMFLAGS) $(CXXFLAGS) $(GSL_CFLAGS)
clean:
	-rm main $(S:.cpp=.o)
