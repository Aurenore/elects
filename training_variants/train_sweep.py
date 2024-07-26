import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import wandb
from utils.train.main import main_train

if __name__ == "__main__":
    wandb.init(
        tags=["D-ELECTS"],
    )
    config = wandb.config
    main_train(config)
