""" train on the train and validation sets"""
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import wandb
from tqdm import tqdm
import pandas as pd
from utils.train.helpers_training import get_run_config, train_epoch, set_up_model, set_up_optimizer, set_up_class_weights, \
    set_up_criterion, set_up_resume, get_all_metrics, log_description, plots_during_training, load_dataset, \
    save_model_artifact, plot_label_distribution_in_training 
from utils.helpers_config import set_up_config, load_personal_config
from utils.test.helpers_testing import test_epoch

def main(config):
    partition="final_train"
    # ----------------------------- CONFIGURATION -----------------------------
    config = set_up_config(config, final_train=True)
    assert config.validation_set == "eval", "validation_set should be 'eval' for final training"
    
    # ----------------------------- LOAD DATASET -----------------------------
    traindataloader, testdataloader, train_ds, test_ds, nclasses, class_names, input_dim, length_sorted_doy_dict_test = load_dataset(config, partition=partition)
        
    # ----------------------------- VISUALIZATION: label distribution -----------------------------
    plot_label_distribution_in_training(train_ds, test_ds, class_names)
        
    # ----------------------------- SET UP MODEL -----------------------------
    model = set_up_model(config, nclasses, input_dim)
    optimizer = set_up_optimizer(config, model)
    class_weights = set_up_class_weights(config, train_ds)
    criterion, mus, mu = set_up_criterion(config, class_weights, nclasses, mus=config.mus)
    train_stats, start_epoch = set_up_resume(config, model, optimizer)
    not_improved = 0
    
    # ----------------------------- TRAINING -----------------------------
    print("starting training...")
    with tqdm(range(start_epoch, config.epochs + 1)) as pbar:
        for epoch in pbar:            
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
            
            # log description
            savemsg = save_model_artifact(config, model, optimizer, df)
            log_description(pbar, epoch, trainloss, testloss, dict_results_epoch, savemsg)

    print("training finished.")
    wandb.finish()
    
if __name__ == '__main__':
    # usage example 
    # python training_variants/final_train.py --configpath config/best_model_config.json
    run_config = get_run_config()
    personal_config = load_personal_config(os.path.join("config", "personal_config.yaml"))
    wandb.init(
        dir=personal_config["wandb_dir"],
        project=personal_config["project"],
        tags=["D-ELECTS", "final train on both train and validation sets"],
        config=run_config
    )
    config = wandb.config
    print(config)
    main(config)
