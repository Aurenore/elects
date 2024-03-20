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
TARGET_DIRECTORY_TO_CLONE="/myhome/cloned_repos"
DATAROOT="/mydata/studentanya/anya/elects_data"
SNAPSHOTSPATH="/mydata/studentanya/anya/elects_snapshots/${jobname}/model.pth"

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
  --image aurenore/elects \
  --environment WANDB_API_KEY=$SECRET_WANDB_API_KEY \
  --working-dir $TARGET_DIRECTORY_TO_CLONE/elects \
  --git-sync source=$REPO,branch=$BRANCH_NAME,rev=$REVISION,username=$USER,password=$PASSWORD,target=$TARGET_DIRECTORY_TO_CLONE \
  -- python EDA/train.py --dataset 'breizhcrops' --dataroot $DATAROOT --snapshot $SNAPSHOTSPATH
  
#   bash -c "echo in_container && \
#                         export WANDB_API_KEY=$SECRET_WANDB_API_KEY && \
#                         echo \$TARGET_DIRECTORY_TO_CLONE && \
#                         cd \$TARGET_DIRECTORY_TO_CLONE/elects && \
#                         python3 EDA/train.py --dataset 'breizhcrops' --dataroot \$DATAROOT --snapshot \$SNAPSHOTSPATH"
