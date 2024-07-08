#!/usr/bin/env python
# coding: utf-8
# # test model on test data 
# Test the model on the test data and save the results.

import os 
os.environ['MPLCONFIGDIR'] = '/myhome'
from utils.plots import plot_label_distribution_datasets
from utils.helpers_config import set_up_config, save_config, print_config
from utils.test.helpers_testing import get_test_stats_from_model, load_test_dataset, save_test_stats
from utils.plots_test import plots_all_figs_at_test
import matplotlib.pyplot as plt
from utils.test.load_model import get_all_runs, get_loaded_model_and_criterion, get_model_and_model_path
from utils.helpers_mu import get_mus_from_config
from utils.results_analysis.extract_video import download_images, add_files_to_images, save_video
import argparse

def main(run_name, sequencelength_test, plot_label_distribution=False, local_dataroot=os.path.join(os.environ.get("HOME", os.environ.get("USERPROFILE")),"elects_data")):
    print(f"Test the model from run '{run_name}' on the test dataset")

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
    print_config(run)
    args  = set_up_config(run_config)
    if local_dataroot == 'config':
        local_dataroot = args.dataroot
    print("Local dataroot: ", local_dataroot)
    args.dataroot = local_dataroot

    # ----------------------------- LOAD DATASET -----------------------------
    # Set the sequence length to 150 like in the original paper.
    if sequencelength_test is None: 
        sequencelength_test = run_config.sequencelength
    else: 
        args.sequencelength = sequencelength_test
    test_ds, nclasses, class_names, input_dim = load_test_dataset(args)

    # ----------------------------- VISUALIZATION: label distribution -----------------------------
    if plot_label_distribution:
        datasets = [test_ds]
        sets_labels = ["Test"]
        fig, ax = plt.subplots(figsize=(15, 7))
        fig, ax = plot_label_distribution_datasets(datasets, sets_labels, fig, ax, title='Label distribution', labels_names=class_names)

    # ## Load the models and the criterions
    mus = get_mus_from_config(args)
    model, criterion = get_loaded_model_and_criterion(run, nclasses, input_dim, mus=mus)

    # ## Test the model on the test dataset
    test_stats, stats = get_test_stats_from_model(model, test_ds, criterion, args)
    print("test_stats:\n", test_stats)
    test_stats_path = save_test_stats(model_path, test_stats)

    # ----------------------------- VISUALIZATION: stopping times and timestamps left-----------------------------
    plots_all_figs_at_test(args, stats, model_path, args, class_names, nclasses, mus)
    
    # ----------------------------- VISUALIZATION: videos of the performance during training -----------------------------
    videos_names = ["class_probabilities_wrt_time", "boxplot", "timestamps_left_plot"]
    for name_image in videos_names:
        download_images(name_image, run, model_path)
        images, images_directory = add_files_to_images(model_path, name_image)
        video_path = save_video(images_directory, images, name_image+"_video.mp4")


if __name__ == "__main__":
    # run with: python test.py --run-name lemon-donkey-4636
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, help="name of the run to test")
    parser.add_argument("--sequencelength-test", type=int, help="sequence length of the test dataset", default=None)
    parser.add_argument("--plot-label-distribution", type=bool, help="plot the label distribution", default=False)
    parser.add_argument("--dataroot", type=str, help="local dataroot", default='default')
    args = parser.parse_args()
    run_name = args.run_name
    sequencelength_test = args.sequencelength_test
    plot_label_distribution = args.plot_label_distribution
    if args.dataroot == 'default':
        local_dataroot = os.path.join(os.environ.get("HOME", os.environ.get("USERPROFILE")),"elects_data")
    elif args.dataroot == 'config':
        local_dataroot = 'config'
    else:
        local_dataroot = args.dataroot
    main(run_name, sequencelength_test, plot_label_distribution, local_dataroot=local_dataroot)
    print("Done.")