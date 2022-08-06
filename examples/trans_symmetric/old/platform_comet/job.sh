#!/bin/bash
#SBATCH --job-name="test"
#SBATCH --nodes=1
#SBATCH --partition=compute
#SBATCH --ntasks-per-node=24
#SBATCH -t 24:00:00
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user ztwang@utk.edu
#SBATCH --export=ALL
#SBATCH -A riu119

date
module list
export OMP_NUM_THREADS=24
echo "Number of threads = $OMP_NUM_THREADS"

./Fermi_Hubbard_square.x >> output.txt

sacct --format JobID,jobname,CPUTime,MaxRSS

echo "done!"
date
