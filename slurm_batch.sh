#!/bin/bash
# Author(s): James Owers (james.f.owers@gmail.com)
#
# example usage:
# ```
# EXPT_FILE=experiments.txt  # <- this has a command to run on each line
# NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
# MAX_PARALLEL_JOBS=12
# sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} slurm_arrayjob.sh $EXPT_FILE
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b slurm_arrayjob.sh -e experiments.txt -m 12
# ```


# ====================
# Options for sbatch
# ====================
# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Megabytes of RAM required. Check `cluster-status` for node configurations
# #SBATCH --mem=32000

# Number of CPUs to use. Check `cluster-status` for node configurations
# #SBATCH --cpus-per-task=8

# Maximum time for the job to run, format: days-hours:minutes:seconds
# #SBATCH --time=1-00:00:00

# #SBATCH --gres=gpu:2
# =====================
# Logging information
# =====================
# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================
echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
# SCRATCH_DISK=/disk/scratch
# SCRATCH_HOME=${SCRATCH_DISK}/${USER}
# mkdir -p ${SCRATCH_HOME}

# Activate your conda environment
#CONDA_ENV_NAME=stable-baselines
#echo "Activating conda environment: ${CONDA_ENV_NAME}"
#conda activate ${CONDA_ENV_NAME}


# =================================
# Move input data to scratch disk
# =================================
echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS
# proj_home=/home/${USER}/git/cluster-scripts  # you may need to change this
# src_path=${proj_home}/experiments/examples/simple/data/input

# input data directory path on the scratch disk of the node
# dest_path=${SCRATCH_HOME}/simple/data/input
# mkdir -p ${dest_path}  # make it if required

# Important notes about rsync:
# * the --compress option is going to compress the data before transfer to send
#   as a stream. THIS IS IMPORTANT - transferring many files is very very slow
# * the final slash at the end of ${src_path}/ is important if you want to send
#   its contents, rather than the directory itself. For example, without a
#   final slash here, we would create an extra directory at the destination:
#       ${SCRATCH_HOME}/project_name/data/input/input
# * for more about the (endless) rsync options, see the docs:
#       https://download.samba.org/pub/rsync/rsync.html

# rsync --archive --update --compress --progress ${src_path}/ ${dest_path}


# ==============================
# Finally, run the experiment!
# ==============================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

# experiment_text_file=$1
# COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"
COMMAND="`bash run_glue_finetune.sh CoLA 10 FineTune gpt2`"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"


# ======================================
# Move output data from scratch to DFS
# ======================================
echo "Moving output data back to DFS"

src_path=${SCRATCH_HOME}/simple/data/output
dest_path=${proj_home}/experiments/examples/simple/data/output
# rsync --archive --update --compress --progress ${src_path}/ ${dest_path}


# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"