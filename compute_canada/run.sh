#!/bin/bash
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:0:0
#SBATCH --mail-user=jkbator@student.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --gpus=1

module purge
module load StdEnv/2020 gcc/11.3.0 cuda/11.8.0 python/3.8.2

# Create virtual environment
virtualenv --no-download gdg
source gdg/bin/activate


# Prepare the environment 
./prepare.sh


# Check if a job name argument is provided
if [ $# -lt 1 ]; then
    echo "Error: Missing job name argument. Please provide a valid job name."
    exit 1
fi

job_name=$1

cp -r ../imagenet $SLURM_TMPDIR

# Check the entered job name and run the corresponding script
cd ./jobs
case $job_name in
    test)
        ./test.sh
        ;;
    classifier_eval)
        ./classifier_eval.sh
        ;;
    dp_sample)
        ./dp_sample.sh
        ;;
    *)
        echo "Error: Invalid job name."
        exit 1
        ;;
esac