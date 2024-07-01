import os 
os.environ['MPLCONFIGDIR'] = '/myhome'
from utils.helpers_config import set_up_config, save_config, print_config
from utils.test.helpers_testing import get_test_stats_from_model, load_test_dataset, save_test_stats, load_test_stats
from utils.test.load_model import get_loaded_model_and_criterion, get_model_and_model_path
from utils.helpers_mu import get_mus_from_config
import argparse
import pandas as pd 

def create_results_table(run_names, runs_df, runs, local_dataroot, metrics_to_keep = ["Wrong pred. penalty", "accuracy", "elects_earliness", "harmonic_mean", "std_score"]):
    results_table = dict()
    is_data_loaded = False
    for run_name in run_names:
        print("run: ", run_name)
        run_idx = runs_df[runs_df.name == run_name].index[0]
        run = runs[run_idx]
        run_config = argparse.Namespace(**run.config)
        model_artifact, model_path = get_model_and_model_path(run)
        
        # check if test_stats already exist in model_path 
        test_stats_path = os.path.join(model_path, "test_stats.json")
        if os.path.exists(test_stats_path):
            print(f"Test stats already exist in {test_stats_path}")
            test_stats = load_test_stats(model_path)
        else:
            print(f"Test stats do not exist in {test_stats_path}, will compute them")

            # get and save the config
            config_path = save_config(model_path, run)
            print_config(run)
            args  = set_up_config(run_config)
            args.dataroot = local_dataroot 
            
            if not is_data_loaded:
                # Load the data
                sequencelength_test = run_config.sequencelength
                args.preload_ram = True
                test_ds, nclasses, class_names, input_dim = load_test_dataset(args, sequencelength_test)
                is_data_loaded = True
                
            # ## Load the models and the criterions
            mus = get_mus_from_config(run_config)
            model, criterion = get_loaded_model_and_criterion(run, nclasses, input_dim, mus=mus)

            # ## Test the model on the test dataset
            test_stats, stats = get_test_stats_from_model(model, test_ds, criterion, run_config)
            _ = save_test_stats(model_path, test_stats)

        # add the test_stats to the results table
        results_table[run_name] = test_stats
        results_table[run_name]["Wrong pred. penalty"] = run_config.factor
        print("*"*100)
    
    # only keep columns accuracy, earliness, harmonic_mean, std_score
    results_table = {run_name: {key: value for key, value in test_stats.items() if key in metrics_to_keep} for run_name, test_stats in results_table.items()}
    return results_table


def create_results_table_end_of_training(run_names, runs_df, runs, metrics_to_keep = ["accuracy", "elects_earliness", "harmonic_mean", "std_score"]):
    results_table = dict()
    for run_name in run_names:
        run_idx = runs_df[runs_df.name == run_name].index[0]
        run = runs[run_idx]
        results_run = dict()
        for metric in metrics_to_keep:
            results_run[metric] = run.summary[metric]
        factor = run.config["factor"]
        results_run["Wrong pred. penalty"] = factor
        results_table[run_name] = results_run
    return results_table


def get_latex_table(results_table, new_columns = ["Wrong pred. penalty", "Accuracy", "Earliness", "Harmonic Mean", "STD score"]):
    df = pd.DataFrame(results_table).T
    # move the column "Wrong pred. penalty" as the first column 
    df = df[["Wrong pred. penalty"] + [col for col in df.columns if col != "Wrong pred. penalty"]]
    # change the name of the columns to be more readable
    df = df.rename(columns=dict(zip(df.columns, new_columns)))
    # change content of the table to 2 decimal places if it is a flaot 
    df = df.map(lambda x: str.format("{:0_.2f}", x) if isinstance(x, float) else x)
    # sort the table alphabetically with the "Wrong pred. penalty" column
    df = df.sort_values(by="Wrong pred. penalty")
    latex_table = df.to_latex(index=False)
    return latex_table