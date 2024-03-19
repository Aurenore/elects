echo "Please enter the jobname"
read jobname

if [ -f "$ENV_FILE" ]; then
  set -o allexport
  . $ENV_FILE
  set +o allexport
fi

REPO='https://github.com/Aurenore/elects' # The URL of the repository
BRANCH_NAME='docker_test' # The branch name
REVISION='HEAD' # The revision to checkout
USER=$GIT_USER # The username of the repository
PASSWORD=$GIT_TOKEN # the secret token 
TARGET_DIRECTORY_TO_CLONE="/workspace" # The directory to clone the repository into

DATAROOT="/mydata/studentanya/anya/elects_data"
SNAPSHOTSPATH=“/mydata/studentanya/anya/elects_snapshots/$jobname/model.pth“

echo "Cloning $REPO"
echo "Branch: $BRANCH_NAME"
echo "Revision: $REVISION"
echo "Username: $USER"
echo "Target directory: $TARGET_DIRECTORY_TO_CLONE"
echo "Data root: $DATAROOT"
echo "Snapshot path: $SNAPSHOTSPATH"

runai submit $jobname \
--image aurenore/elects \
--git-sync source=$REPO,branch=$BRANCH_NAME,rev=$REVISION,username=$USER,password=$PASSWORD,target=$TARGET_DIRECTORY_TO_CLONE \
--interactive 
#-- export WANDB_API_KEY=$SECRET_WANDB_API_KEY # cd $TARGET_DIRECTORY_TO_CLONE/elects && git config --global --add safe.directory /workspace/rev-43300cb1b18f5ceb6517aed1a85f4b1161319de4 #&& python3 train.py --dataset "breizhcrops" --dataroot $DATAROOT --snapshot $SNAPSHOTSPATH
# to add after git sync: --gpu 0.2 \
