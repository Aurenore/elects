import sys
import os 
#os.environ['MPLCONFIGDIR'] = "$HOME"
#os.environ["WANDB_DIR"] = os.path.join(os.path.dirname(__file__), "..", "wandb")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import wandb
from utils.train.main import main_train

if __name__ == '__main__':
    wandb.init(
        notes="ELECTS with new cost function",
        tags=["ELECTS", "earlyrnn", "trials", "sweep", "kp", "alphas", "with bias init", "with weight in earliness reward", "early wrong prediction penalty"],
    )
    config = wandb.config
    main_train(config)
