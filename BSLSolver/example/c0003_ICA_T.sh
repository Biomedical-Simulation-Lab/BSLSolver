#!/bin/bash
#Anything listed between set -a and set +a can be used as variables in any subsequent commands (this is a workaround for sourcing or exporting every variable)
set -a 
#CFD job script input file
#you need to give your username or this won't work at all
scinet_user=ahaleyyy
#solver environment name (specific to you)
solver_env_name=oasis
#Whether or not you are using debug node
debug=on
#What your case will be called in the output files & on Niagara
casename="c0003_ICA_T"
#Number of cycles to run
cycles=1 #default is 1
#Number of timesteps for each cycle 
timesteps_per_cycle=20000 #default is 2000
#Velocity order
uOrder=2 #default is 1
#How many timesteps before saving
save_frequency=8 #default 5
#Amount of time Niagara will need to run the case (max 24 hours)
estimated_required_time="00:30:00"
#Amount of time needed to post-process the case (this is run on a single proc)
post_processing_time_minutes=180
#Number of cores to use per node (everything run on a singe node) 
num_cores=80 #(40 hyperthreaded to 80)
#Whether or not to save ftle field. Default is False
save_ftle=True

set +a
#OPTIONAL ARGUMENTS
#Place these within set -+a and add them to argument list of mpirun command in solver.sh
#period=951 #ms
#viscosity=0.0035 #default 0.0035
#zero_pressure_outlets= False #default True
#include_gravitational_effects=False #default False
#flat_profile_at_inlet_bc=False #default False
#maxwtime=(23*60*60) #default (23*60*60)

###IF YOU WANT THE NORMAL OASIS OUTPUT, IT IS NORMALLY SUPPRESSED!!###
#You can get the normal output if you add the following to your variables list
#save_step=1000 #default is 100000
#checkpoint=500 #Mehdi checkpoints at every tstep

#Run the submission script to Niagara
#"$@" just includes any additional keyword arguments passed when you run this script (eg. the command "./in_default.sh hello" would pass the variable hello to this script, and it would be then passed to the solver.sh file subsequently, and could be accessed with $1)
./solver-v2.sh "$@"
