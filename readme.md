README for the new BSL Solver

# NOTES:
=========================================================================
1) Anywhere you see [YOUR USERNAME HERE] just put your Niagara username with no brackets there.

# GETTING STARTED:
=========================================================================
1) You will need an environment for the BSLSolver that you add to the input file. 

The environment must be created as follows on the Niagara login node:

cd ~

module load NiaEnv/.2020a intel/2020u1 intelmpi/2020u1 intelpython3/2020u1 cmake/3.16.3 boost/1.69.0 eigen/3.3.7 hdf5/1.8.21 netcdf/4.6.3 gmp/6.2.0 mpfr/4.0.2 swig/4.0.1 petsc/3.10.5 trilinos/12.12.1 fenics/2019.1.0

conda create -n oasis python=3.7 -y

source activate oasis

conda install mpi4py matplotlib gxx_linux-64 -y

NOTE: Dolfin is NOT compiled with mpi4py!!!

Fixes: ImportError: /lib64/libz.so.1: version `ZLIB_1.2.9' not found

export LD_LIBRARY_PATH=/home/s/scinet/[YOUR USERNAME HERE]/.conda/envs/oasis/lib:$LD_LIBRARY_PATH

# Install Oasis
git clone https://github.com/mikaem/Oasis.git

cd Oasis

This will install Oasis in development mode:

python -m pip install -e . 

# Install BSLSolver
git clone https://github.com/Biomedical-Simulation-Lab/BSLSolver.git

cd BSLSolver

python -m pip install -e .

conda install scipy -y

2) You will need to have a local dijitso cache in your scratch directory, because you can't write to your dijitso cache in your home directory (as you could locally). To do this, you will need to copy dijitso:

cp -r $HOME/.cache/dijitso $SCRATCH/.cache/

In your ~/.bashrc, add the following lines:

export DIJITSO_CACHE_DIR=/scratch/s/steinman/[YOUR USERNAME HERE]/.cache/dijitso
export INSTANT_CACHE_DIR=/scratch/s/steinman/[YOUR USERNAME HERE]/.local/instant-cache/
export INSTANT_ERROR_DIR=/scratch/s/steinman/[YOUR USERNAME HERE]/.local/instant-error/

3) Put the directory BSLSolver into your $HOME directory

# TO RUN: look in the example directory for an aneurysm case example
=========================================================================
1) You will need three files in the directory you wish to run in:

Artery.py

in_default.sh (change the name of this file. in the example folder it is called c0003_ICA_T.sh)

solver-v2.sh

2) You will need a directory named "data" which contains two files:

[casename].xml.gz - this is the mesh

[casename].info - this gives flowrate information about the inlets and outlets of the mesh and which waveform to use


