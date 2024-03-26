#!/bin/bash

echo "Please enter the jobname:"
read jobname

# Checking if the ENV_FILE is set and exists
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  source $ENV_FILE
  set +o allexport
else
  echo "Environment file not found."
  exit 1
fi

# Configuration variables
REPO='https://github.com/Aurenore/elects'
BRANCH_NAME='docker_test'
REVISION='HEAD'
USER=$GIT_USER
PASSWORD=$GIT_TOKEN
JOBNAME_PREFIX="elects-breizhcrops-training"
TARGET_DIRECTORY_TO_CLONE="/workspace/cloned_repos"
PROJECTUSER_PATH="/mydata/studentanya/anya"
DATAROOT="$PROJECTUSER_PATH/elects_data"
DATA="breizhcrops"
SNAPSHOTSPATH="$PROJECTUSER_PATH/elects_snapshots/$DATA/$JOBNAME_PREFIX/model.pth"

# Display configuration to the user
echo "Cloning $REPO"
echo "Branch: $BRANCH_NAME"
echo "Revision: $REVISION"
echo "Username: $USER"
echo "Target directory: $TARGET_DIRECTORY_TO_CLONE"
echo "Data root: $DATAROOT"
echo "Snapshot path: $SNAPSHOTSPATH"
echo "home: $HOME"

# Submitting the job
runai submit $jobname \
  --job-name-prefix $JOBNAME_PREFIX \
  --image aurenore/elects \
  --environment WANDB_API_KEY=$SECRET_WANDB_API_KEY \
  --working-dir $TARGET_DIRECTORY_TO_CLONE/elects \
  --backoff-limit 1 \
  --git-sync source=$REPO,branch=$BRANCH_NAME,rev=$REVISION,username=$USER,password=$PASSWORD,target=$TARGET_DIRECTORY_TO_CLONE \
  -- python train.py --dataset $DATA --dataroot $DATAROOT --snapshot $SNAPSHOTSPATH --epochs 100
  