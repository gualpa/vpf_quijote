#!/bin/bash
#SBATCH --job-name=vpf_s8p_pl               # Job name
#SBATCH --output=output_%j.txt                # Output file (%j expands to job ID)
#SBATCH --error=error_%j.txt                  # Error file (%j expands to job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=2                     # Number of CPUs (adjust based on your script)
#SBATCH --partition=batch                     # Partition name
#SBATCH --time=3-00:00:00  

# Load necessary modules (if any)
eval "$(conda shell.bash hook)"
module load gcc
conda activate vpf


#Export the command line arguments as environment variables
export cosm="s8_p"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
# Run the Python script
srun python /home/fdavilakurban/Proyectos/VPF_Quijote/codes/rvpf_new_batch_cosm_parallel.py
