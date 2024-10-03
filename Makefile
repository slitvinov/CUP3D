.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp
.SUFFIXES: .o

MPICXX = h5c++.mpich
GSL_CFLAGS != pkg-config --cflags gsl
GSL_LDFLAGS != pkg-config --libs gsl
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
AdvectionDiffusion.cpp \
AdvectionDiffusionImplicit.cpp \
ArgumentParser.cpp \
BufferedLogger.cpp \
CarlingFish.cpp \
ComputeDissipation.cpp \
Cylinder.cpp \
CylinderNozzle.cpp \
DiffusionSolverAMRKernels.cpp \
Ellipsoid.cpp \
ExternalForcing.cpp \
ExternalObstacle.cpp \
Fish.cpp \
FishLibrary.cpp \
FishShapes.cpp \
FixMassFlux.cpp \
FluidSolidForces.cpp \
InitialConditions.cpp \
main.cpp \
Naca.cpp \
Obstacle.cpp \
ObstacleFactory.cpp \
ObstaclesCreate.cpp \
ObstaclesUpdate.cpp \
Penalization.cpp \
Pipe.cpp \
Plate.cpp \
PoissonSolverAMR.cpp \
PoissonSolverAMRKernels.cpp \
PoissonSolverBase.cpp \
PressureProjection.cpp \
Simulation.cpp \
SimulationData.cpp \
SmartNaca.cpp \
Sphere.cpp \
StefanFish.cpp \

main: $(S:.cpp=.o)
	$(MPICXX) -o main $(S:.cpp=.o) $(GSL_LDFLAGS) $(LDFLAGS) -fopenmp
.cpp.o:
	$(MPICXX) -o $@ -c $< $(CUBISMFLAGS) $(CXXFLAGS) $(GSL_CFLAGS)
clean:
	-rm main $(S:.cpp=.o)
