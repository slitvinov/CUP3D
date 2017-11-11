#!/bin/bash
SETTINGSNAME=$1

MYNAME=`whoami`
BASEPATH="${SCRATCH}/CubismUP3D/"
#lfs setstripe -c 1 ${BASEPATH}${RUNFOLDER}

if [ ! -f $SETTINGSNAME ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME

NPROCESSORS=$((${NNODE}*12))
FOLDER=${BASEPATH}${BASENAME}
mkdir -p ${FOLDER}

cp $SETTINGSNAME ${FOLDER}/settings.sh
cp ${FFACTORY} ${FOLDER}/factory
cp ../makefiles/simulation ${FOLDER}
cp launchDaint.sh ${FOLDER}
cp runDaint.sh ${FOLDER}/run.sh
#cp hingedParams.txt ${FOLDER}
cp -r ../source ${FOLDER}

cd ${FOLDER}

cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=s658
#SBATCH --job-name="${BASENAME}"
#SBATCH --output=${BASENAME}_out_%j.txt
#SBATCH --error=${BASENAME}_err_%j.txt
#SBATCH --time=${WCLOCK}
#SBATCH --nodes=${NNODE}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=2
#SBATCH --constraint=gpu
#SBATCH --mail-user=${MYNAME}@ethz.ch
#SBATCH --mail-type=ALL


export LD_LIBRARY_PATH=/users/novatig/accfft/build_dbg/:$LD_LIBRARY_PATH
module load daint-gpu GSL cray-hdf5-parallel
module load cudatoolkit fftw

export OMP_NUM_THREADS=24
export MYROUNDS=10000
export USEMAXTHREADS=1
srun --ntasks ${NNODE} --threads-per-core=2 --ntasks-per-node=1 --cpus-per-task=12 time ./simulation ${OPTIONS}
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch
