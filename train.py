import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import wandb
from utils.train.main import main_train
from utils.helpers_config import load_personal_config
from utils.train.helpers_training import get_run_config
    
if __name__ == '__main__':
    # example usage: 
    # python train.py --configpath config/best_model_config.json
    run_config = get_run_config()
    personal_config = load_personal_config(os.path.join("config", "personal_config.yaml"))
    
    wandb.init(
        dir=personal_config["wandb_dir"],
        project=personal_config["project"],
        tags=["D-ELECTS"],
        config=run_config
    )
    config = wandb.config
    print(config)
    main_train(config)
