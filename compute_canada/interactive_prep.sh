module --force purge
module load StdEnv/2020 gcc/11.3.0 cuda/11.8.0 python/3.8.2

# Create virtual environment
virtualenv --no-download gdg
source gdg/bin/activate


# Prepare the environment 
./prepare.sh

mkdir -p /tmp/data
cp -r ../imagenet /tmp/data
