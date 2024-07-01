import sys
import os 
#os.environ['MPLCONFIGDIR'] = "$HOME"
#os.environ["WANDB_DIR"] = os.path.join(os.path.dirname(__file__), "..", "wandb")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import wandb
from utils.train.helpers_training import  parse_args
from utils.train.main import main_train

if __name__ == '__main__':
    # use example: 
    # python train.py --loss daily_reward_piecewise_lin_regr --decision-head day --loss-weight balanced --corrected False --daily-timestamps False --day-head-init-bias 1 --alpha-decay 1.,0.4 --start-decision-head-training 10 --p-thresh 0.7 --hidden-dims 128 --batchsize 128 --factor v1 --percentage-other-alphas 0.45074209570884705,0.14149823784828186,0.4077596366405487
    args = parse_args()
    wandb.init(
        dir="/mydata/studentanya/anya/wandb/",
        project="MasterThesis",
        notes="compare 2 models, train one here.",
        tags=["ELECTS", args.dataset, args.backbonemodel],
        config={
        "backbonemodel": args.backbonemodel,
        "dataset": args.dataset,
        "epsilon": args.epsilon,
        "learning_rate": args.learning_rate, 
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "device": args.device,
        "epochs": args.epochs,
        "sequencelength": args.sequencelength,
        "hidden_dims": args.hidden_dims,
        "batchsize": args.batchsize,
        "dataroot": args.dataroot,
        "snapshot": args.snapshot,
        "sequencelength": args.sequencelength,
        "loss": args.loss,
        "decision_head": args.decision_head,
        "loss_weight": args.loss_weight,
        "resume": args.resume,
        "validation_set": args.validation_set,
        "corrected": args.corrected,
        "daily_timestamps": args.daily_timestamps,
        "original_time_serie_lengths": args.original_time_serie_lengths,
        "alpha": args.alpha,
        "day_head_init_bias": args.day_head_init_bias,
        "alpha_decay": args.alpha_decay,
        "start_decision_head_training": args.start_decision_head_training,
        "percentage_earliness_reward": args.percentage_earliness_reward,
        "p_thresh": args.p_thresh,
        "architecture": "EarlyRNN",
        "optimizer": "AdamW",
        "criterion": args.loss,
        "factor": args.factor,
        "percentage_other_alphas": args.percentage_other_alphas,
        }
    )
    config = wandb.config
    print(config)
    main_train(config)
