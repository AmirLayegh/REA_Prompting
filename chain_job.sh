#!/bin/bash
#SBATCH --gpus 1 -C "thin" -t 48:00:00


echo "Hello cluster computing world!"

echo "JOB: ${SLURM_JOB_ID}"

echo "The following is RAM info."
free -h

echo "The following is GPU info."
nvidia-smi

echo "Launching experiments with apptainer."

echo "PYTHONPATH is: $PYTHONPATH"

echo "Current working directory is: $(pwd)"

#export PYTHONPATH=/proj/layegh/users/x_amila/CoV:$PYTHONPATH


#apptainer exec  --nv  /proj/layegh/users/x_amila/CoV/Project/nebula.sif python3 /proj/layegh/users/x_amila/CoV/Project/main.py

apptainer exec --env OPENAI_API_KEY=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 --nv  /proj/layegh/users/x_amila/CoV/Project/nebula.sif python3  /proj/layegh/users/x_amila/ChainofRefinement/main.py

#apptainer exec --env OPENAI_API_KEY=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 nebula.sif python3 re_zephyr.py