echo "Please enter the jobname"
read jobname

if [ -f "$ENV_FILE" ]; then
  set -o allexport
  . $ENV_FILE
  set +o allexport
fi

REPO='https://github.com/Aurenore/elects.git' # The URL of the repository
BRANCH_NAME='docker_test' # The branch name
REVISION='HEAD' # The revision to checkout
USER=$GIT_USER # The username of the repository
PASSWORD=$GIT_TOKEN # the token 
TARGET_DIRECTORY_TO_CLONE=$HOME # The directory to clone the repository into

echo "Cloning $REPO"
echo "Branch: $BRANCH_NAME"
echo "Revision: $REVISION"
echo "Username: $USER"
echo "Target directory: $TARGET_DIRECTORY_TO_CLONE"

runai submit $jobname \
--image aurenore/runai-job \
--gpu 0.2 \
--git-sync source=$REPO,branch=$BRANCH_NAME,rev=$REVISION,username=$USER,password=$PASSWORD,target=$TARGET_DIRECTORY_TO_CLONE \
-- cd $TARGET_DIRECTORY_TO_CLONE/elects.git && python3 EDA/train.py