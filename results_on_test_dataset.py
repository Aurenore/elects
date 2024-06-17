#!/usr/bin/env python
# coding: utf-8
# # test model on test data 
# Test the model on the test data and save the results.

import os 
os.environ['MPLCONFIGDIR'] = '/myhome'
from utils.plots import plot_label_distribution_datasets
from utils.helpers_config import set_up_config, save_config
from utils.helpers_testing import get_test_stats_from_model, load_test_dataset, save_test_stats
from utils.plots_test import plots_all_figs_at_test
import matplotlib.pyplot as plt
from utils.test.load_model import get_all_runs, get_loaded_model_and_criterion, get_model_and_model_path
from utils.helpers_mu import get_mus_from_config
import argparse

def main(run_name):
    print(f"Test the model from run '{run_name}' on the test dataset")
    local_dataroot = os.path.join(os.environ.get("HOME", os.environ.get("USERPROFILE")),"elects_data")
    print("Local dataroot: ", local_dataroot)

    # ## Download the model from wandb 
    entity, project = "aurenore", "MasterThesis"
    runs_df, runs = get_all_runs(entity, project)

    # get the run with name:
    run_idx = runs_df[runs_df.name == run_name].index[0]
    run = runs[run_idx]
    run_config = argparse.Namespace(**run.config)
    model_artifact, model_path = get_model_and_model_path(run)

    # get and save the config
    config_path = save_config(model_path, run)
    print("config:")
    for key, value in run.config.items():
        print(f"{key}: {value}")

    args, _ = set_up_config(run_config)
    args.dataroot = local_dataroot

    # ----------------------------- LOAD DATASET -----------------------------
    # Set the sequence length to 150 like in the original paper.
    sequencelength_test = 150 # by elects paper
    test_ds, nclasses, class_names, input_dim = load_test_dataset(args, sequencelength_test)

    # ----------------------------- VISUALIZATION: label distribution -----------------------------
    datasets = [test_ds]
    sets_labels = ["Test"]
    fig, ax = plt.subplots(figsize=(15, 7))
    fig, ax = plot_label_distribution_datasets(datasets, sets_labels, fig, ax, title='Label distribution', labels_names=class_names)

    # ## Load the models and the criterions
    mus = get_mus_from_config(run_config)
    model, criterion = get_loaded_model_and_criterion(run, nclasses, input_dim, mus=mus)

    # ## Test the model on the test dataset
    test_stats, stats = get_test_stats_from_model(model, test_ds, criterion, run_config)
    print("test_stats:\n", test_stats)
    test_stats_path = save_test_stats(model_path, test_stats)

    # ----------------------------- VISUALIZATION: stopping times and timestamps left-----------------------------
    plots_all_figs_at_test(args, stats, model_path, run_config, class_names, nclasses, mus, sequencelength_test)


if __name__ == "__main__":
    # run with: python results_on_test_dataset.py --run-name "lemon-donkey-4636"
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, help="name of the run to test")
    args = parser.parse_args()
    run_name = args.run_name
    main(run_name)
    print("Done.")