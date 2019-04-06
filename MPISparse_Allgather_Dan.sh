#!/bin/bash

################################################
######## Sample SLURM Submission Script ########
################################################
## Please note that for uncommented options, one pound
## symbol must be present; for commented options, two 
## pound symbols are present. Ex, #SBATCH [option]
## is considered UNCOMMENTED and ##SBATCH [option] is
## considered COMMENTED. This only applies for #SBATCH.
## Any other commented lines can be done with one or two
## pound symbols.

## SBATCH --job-name [jobname]
## Job Name; Can be anything. If not set, will default
## to job id.
#SBATCH --job-name MPISparse_Allgather_Dan

## When working with jobs, it is important to note that
## SLURM (scheduler) will take care of most things for you.
## If you have no requirements for running code, (as in,
## your program isn't constrained by resources like network
## speed, memory on the same bus as the processor, etc.)
## you can get away with simply specifying the number of 
## tasks (--ntasks) and you should be good to go. Otherwise,
## look over the next options to see if anything applies to
## you.


## SBATCH --nodes [numnodes]
## Number of nodes to distribute your job to. Optional, only
## useful if your job needs multiple nodes (ex, to maximize
## bandwidth to the storage node). Note that SLURM will
## distribute your job in the following way: given n number of
## nodes and m number of tasks, for every n node, floor(m/n)
## will be assigned to node n. That is, if you have 4 tasks
## and 2 nodes, 2 tasks will be assigned per node. 
#SBATCH --nodes 1

## SBATCH --ntasks [numtasks]
## Number of tasks your job will use. If running an MPI job,
## this argument functions the same way mpiexec -n does. 
## For example, if you're running an MPI job that needs 40 
## processes, you'll specify '#SBATCH --ntasks 40' and omit
## the -n option from mpirun. If your job has any form of
## threading (ex, OpenMP), you may want to also uncomment
## --cpu_per_task
#SBATCH --ntasks 1

## SBATCH --mem [memory]
## Amount of memory your job needs per node, regardless of the
## number of tasks allocated per node. --mem-per-cpu is preferred
## over this option unless you have specific reason to avoid it.
## by default, the memory limit is set to 100MB per node.
##SBATCH --mem 100MB

## SBATCH --mem-per-cpu [memory]
## Amount of memory your job needs per cpu (task in most cases).
## Preferred over --mem, but optional if your job won't use more than
## 100MB per node. If your job allocates more than one cpu per task,
## take note that this allocates per CPU, NOT TASK. That being said,
## if your job allocates two CPU's per task and sets this to 100MB,
## theoretically your job will have 200MB of memory leeway (provided your
## task ended up forking, etc.)
##SBATCH --mem-per-cpu 10M

## SBATCH --partition [partname]
## Partition to run your job in. For normal blueshark users, defaults to
## short, which has a max run time of 15 minutes. For MTH 4082 users,
## the partition defaults to class, which has a max run time of 10 minutes.
#SBATCH --partition class

## SBATCH --time [time]
## Amount of time your job is expected to run. Job will terminate
## if it exceeds this time. This time also cannot be larger than
## the max execution time of the partition your job was submitted to. 
## The time is written as D-HH:MM:SS. ex, for 7 days and 2 hours, write as
## 7-02:00:00. When the time is less than one day, D- can be omitted.
#SBATCH --time 00:05:00

## SBATCH --error|output [filename]
## File(s) to save your job's output to. Note that if the file already exists,
## output will be truncated, NOT appended. These options support substitution     
## characters. (ex, %J will be replaced	with the job id, prevents overwriting
## of output)
#SBATCH --error=MPISparse_Allgather_Dan.err
#SBATCH	--output=MPISparse_Allgather_Dan.out


######################################
######## END OF SLURM OPTIONS ########
######################################

## From this point on, simply specify what you want
## your job to do.
## Common practice is to load required modules here
## before executing the job:
# module load mpich
# module load gcc

## Also note that mpiexec/mpirun no longer need
## the -n option: the number of tasks is taken from
## SLURM's --ntasks option.

mpiexec ~/Assignments/Homework/Final\ Project/MPISparse_Allgather_Dan
