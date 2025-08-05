#!/bin/bash
export LD_LIBRARY_PATH=/home/s/steinman/$scinet_user/../ahaleyyy/.conda/envs/$solver_env_name/lib:/scinet/niagara/software/2020a/opt/intel-2020u1-intelmpi-2020u1/boost/1.69.0/lib:$LD_LIBRARY_PATH
mkdir hpclog
mkdir logs

sbatch --time=$estimated_required_time --nodes=1 --ntasks-per-node=$num_cores  << EOST
#!/bin/bash
#SBATCH --output=hpclog/art_%x_stdout_%j.txt
module purge
#provided the environment was set as per the readme, the following just loads the correct modules and activates the conda environment
module load NiaEnv/.2020a intel/2020u1 intelmpi/2020u1 intelpython3/2020u1 cmake/3.16.3 boost/1.69.0 eigen/3.3.7 hdf5/1.8.21 netcdf/4.6.3 gmp/6.2.0 mpfr/4.0.2 swig/4.0.1 petsc/3.10.5 trilinos/12.12.1 fenics/2019.1.0
source activate $HOME/../ahaleyyy/.conda/envs/$solver_env_name

mpirun -n 40 oasis NSfracStep problem=Artery uOrder=$uOrder timesteps=$timesteps_per_cycle period=$period cycles=$cycles save_frequency=$save_frequency mesh_name=$casename period=$period viscosity=$viscosity

EOST