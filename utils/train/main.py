import sys
import os 
#os.environ['MPLCONFIGDIR'] = "$HOME"
#os.environ["WANDB_DIR"] = os.path.join(os.path.dirname(__file__), "..", "wandb")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from tqdm import tqdm
import pandas as pd
import wandb
from utils.helpers_mu import mus_should_be_updated, update_mus_during_training
from utils.train.helpers_training import train_epoch, set_up_model, set_up_optimizer, set_up_class_weights, \
    set_up_criterion, set_up_resume, get_all_metrics, update_patience, log_description, \
    plots_during_training, load_dataset, plot_label_distribution_in_training 
from utils.helpers_config import set_up_config
from utils.test.helpers_testing import test_epoch

def main_train(config):
    # ----------------------------- CONFIGURATION -----------------------------
    config = set_up_config(config)
    
    # ----------------------------- LOAD DATASET -----------------------------
    traindataloader, testdataloader, train_ds, test_ds, nclasses, class_names, input_dim, length_sorted_doy_dict_test = load_dataset(config)
    
    # ----------------------------- VISUALIZATION: label distribution -----------------------------
    plot_label_distribution_in_training(train_ds, test_ds, class_names)
        
    # ----------------------------- SET UP MODEL -----------------------------
    model = set_up_model(config, nclasses, input_dim)
    optimizer = set_up_optimizer(config, model)
    class_weights = set_up_class_weights(config, train_ds)
    criterion, mus, mu = set_up_criterion(config, class_weights, nclasses)
    train_stats, start_epoch = set_up_resume(config, model, optimizer)
    not_improved = 0
    
    # ----------------------------- TRAINING -----------------------------
    print("starting training...")
    with tqdm(range(start_epoch, config.epochs + 1)) as pbar:
        for epoch in pbar:
            if mus_should_be_updated(config, epoch):
                mus = update_mus_during_training(config, criterion, stats, epoch, mus, mu)
            
            # train and test epoch
            dict_args = {"epoch": epoch}
            trainloss = train_epoch(model, traindataloader, optimizer, criterion, device=config.device, **dict_args)
            testloss, stats = test_epoch(model, testdataloader, criterion, config, return_id=test_ds.return_id, **dict_args)

            # get metrics
            dict_results_epoch, train_stats = get_all_metrics(stats, config, epoch, train_stats, trainloss, testloss, criterion, class_names, mus)
            
            # plot metrics
            dict_results_epoch = plots_during_training(epoch, stats, config, dict_results_epoch, class_names, length_sorted_doy_dict_test, mus, nclasses, config.sequencelength)
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
    print("training finished.")
    config.update({"mus": mus}, allow_val_change=True)
    wandb.finish()