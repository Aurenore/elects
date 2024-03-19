# firts, check that the envionrment variable are set: 
echo $DATAROOT
echo $SNAPSHOTSPATH
echo $TARGET_DIRECTORY_TO_CLONE
echo $SECRET_WANDB_API_KEY

# then, run the first part of the command
export WANDB_API_KEY=$SECRET_WANDB_API_KEY
cd $TARGET_DIRECTORY_TO_CLONE/elects
git config --global --add safe.directory /workspace/rev-43300cb1b18f5ceb6517aed1a85f4b1161319de4
python3 train.py --dataset "breizhcrops" --dataroot $DATAROOT --snapshot $SNAPSHOTSPATH