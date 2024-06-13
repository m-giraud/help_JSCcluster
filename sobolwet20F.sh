#!/bin/bash
#SBATCH --job-name=wetO9F
#SBATCH -A esmtst
#SBATCH --nodes=116
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1 
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.giraud@fz-juelich.de

module --force purge
module load Stages/2020
module load GCC/10.3.0
module load ParaStationMPI/5.4.10-1
module load GCCcore/.10.3.0
module load Python/3.8.5
module load OpenCV
module load CMake
module load SciPy-Stack
module load mpi4py


echo ${SLURM_CPUS_PER_TASK}
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"
cd /p/project/cesmtst/giraud1/CPlantBox/applications/sobol/results/sobolSEB
rm -rf 9wet20functional
mkdir 9wet20functional
cd /p/project/cesmtst/giraud1/CPlantBox/applications/sobol
srun python3 testmpi2.py 9 wet 20 functional