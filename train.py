import sys
import os 
#os.environ['MPLCONFIGDIR'] = "$HOME"
#os.environ["WANDB_DIR"] = os.path.join(os.path.dirname(__file__), "..", "wandb")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from tqdm import tqdm
import pandas as pd
import wandb
from utils.helpers_training import train_epoch, set_up_config, set_up_model, set_up_optimizer, set_up_class_weights, \
    set_up_criterion, set_up_resume, mus_should_be_updated, update_mus_during_training, get_all_metrics, \
    update_patience, log_description, plots_during_training, load_dataset, plot_label_distribution_in_training, parse_args
from utils.helpers_testing import test_epoch

def main(config):
    # ----------------------------- CONFIGURATION -----------------------------
    config, extra_padding_list = set_up_config(config)
    
    # ----------------------------- LOAD DATASET -----------------------------
    traindataloader, testdataloader, train_ds, test_ds, nclasses, class_names, input_dim, length_sorted_doy_dict_test = load_dataset(config)
    
    # ----------------------------- VISUALIZATION: label distribution -----------------------------
    plot_label_distribution_in_training(train_ds, test_ds, class_names)
        
    # ----------------------------- SET UP MODEL -----------------------------
    model = set_up_model(config, nclasses, input_dim)
    optimizer = set_up_optimizer(config, model)
    class_weights = set_up_class_weights(config, train_ds)
    criterion, mus = set_up_criterion(config, class_weights, nclasses)
    train_stats, start_epoch = set_up_resume(config, model, optimizer)
    not_improved = 0
    
    # ----------------------------- TRAINING -----------------------------
    print("starting training...")
    with tqdm(range(start_epoch, config.epochs + 1)) as pbar:
        for epoch in pbar:
            if mus_should_be_updated(config, epoch):
                mus = update_mus_during_training(config, criterion, stats, dict_results_epoch, epoch, mus)
            
            # train and test epoch
            dict_args = {"epoch": epoch}
            trainloss = train_epoch(model, traindataloader, optimizer, criterion, device=config.device, extra_padding_list=extra_padding_list, **dict_args)
            testloss, stats = test_epoch(model, testdataloader, criterion, config.device, extra_padding_list=extra_padding_list, return_id=test_ds.return_id, daily_timestamps=config.daily_timestamps, **dict_args)

            # get metrics
            dict_results_epoch, train_stats = get_all_metrics(stats, config, epoch, train_stats, trainloss, testloss, criterion, class_names)
            
            # plot metrics
            dict_results_epoch = plots_during_training(epoch, stats, config, dict_results_epoch, class_names, length_sorted_doy_dict_test, mus, nclasses)
            wandb.log(dict_results_epoch)
            
            # save model if testloss improved
            df = pd.DataFrame(train_stats).set_index("epoch")
            savemsg, not_improved = update_patience(df, testloss, config, model, optimizer, not_improved)
            
            # log description
            log_description(pbar, epoch, trainloss, testloss, dict_results_epoch, savemsg)
            
            # stop training if testloss did not improve
            if config.patience is not None:
                if not_improved > config.patience:
                    print(f"stopping training. testloss {testloss:.2f} did not improve in {config.patience} epochs.")
                    break 
    wandb.finish()


if __name__ == '__main__':
    # use example: 
    # python train.py --backbonemodel TempCNN --backbonemodel TempCNN --dataset breizhcrops --epochs 100 --hidden-dims 16 --sequencelength 70 --extra-padding-list 35 0
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
        "extra_padding_list": args.extra_padding_list,  
        "hidden_dims": args.hidden_dims,
        "batchsize": args.batchsize,
        "dataroot": args.dataroot,
        "snapshot": args.snapshot,
        "left_padding": args.left_padding,
        "sequencelength": args.sequencelength,
        "loss": args.loss,
        "decision_head": args.decision_head,
        "loss_weight": args.loss_weight,
        "resume": args.resume,
        "validation_set": args.validation_set,
        "corrected": args.corrected,
        "daily_timestamps": args.daily_timestamps,
        "original_time_serie_lenghs": args.original_time_serie_lenghs,
        "alpha": args.alpha,
        "day_head_init_bias": args.day_head_init_bias,
        "alpha_decay": args.alpha_decay,
        "start_decision_head_training": args.start_decision_head_training,
        "percentage_earliness_reward": args.percentage_earliness_reward,
        "mu": args.mu,
        "p_thresh": args.p_thresh,
        "architecture": "EarlyRNN",
        "optimizer": "AdamW",
        "criterion": args.loss,
        }
    )
    main(args)
