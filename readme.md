# D-ELECTS - In Season Crop Classification using Satellite Imagery
This repository contains the code of my master thesis project.  The report is available [here](Master_Thesis__In_Season_Crop_Classification_using_Satellite_Imagery%20.pdf).

<img width="100%" src="png/input-output.drawio.png">

The Readme is structured as follows:
- [D-ELECTS - In Season Crop Classification using Satellite Imagery](#d-elects---in-season-crop-classification-using-satellite-imagery)
  - [1. Abstract](#1-abstract)
  - [2. Dependencies](#2-dependencies)
  - [3. Train the Model](#3-train-the-model)
    - [Monitor training](#monitor-training)
      - [2. Final Train](#2-final-train)
  - [4. Test the Model](#4-test-the-model)
  - [5. Notebooks](#5-notebooks)
  - [6. References](#6-references)

## 1. Abstract
[TO COMPLETE]

## 2. Dependencies 
[TO COMPLETE]

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## 3. Train the Model

### Monitor training 
Create an account on [wandb](https://wandb.ai) and login to your account in the terminal
```bash
wandb login
```
Create a project on wandb. 

Then, fill the [`personal_config.yaml`](config/personal_config.yaml) file with your wandb data 

```yaml 
entity: <username>
project: <projectname>
```	

### Start training loop
[TO COMPLETE]
To start the training loop run
```
❯ python train.py --
Setting up a new session...
epoch 100: trainloss 1.70, testloss 1.97, accuracy 0.87, earliness 0.48. classification loss 7.43, earliness reward 3.48: 100%|███| 100/100 [06:34<00:00,  3.95s/it]
```
The BavarianCrops dataset is automatically downloaded.
Additional options (e.g., `--alpha`, `--epsilon`, `--batchsize`) are available with `python train.py --help`.


### Other training variants (optional)
Two training variants are available in the [`training_variants`](training_variants) folder. They aim to optimize the hyperparameters of the model and to train the model on the training and validation sets.

#### 1. Sweep Train
For hyperparameter optimization, you can use the [`train_sweep.py`](training_variants/train_sweep.py) script. To do so, first initialize a sweep with
```
wandb sweep --project <projectname> <configpath>
```
where `<configpath>` is the path to the sweep configuration file. An example of sweep configuration file is given in [`config/sweep_config.yaml`](config/sweep_config.yaml).
Make sure to change `dataroot` and `snapshot` values to the correct paths. This command will return a `<sweep_id>`.

Then, launch wandb agent with 
```bash
wandb agent <username>/<projectname>/<sweep_id>
```
where `<sweep_id>` is the id of the sweep you initialized. You can follow the training process on the wandb dashboard.

You can find the best model via the wandb dashboard, and download the configuration file from there. You can also find the best model by downloading the results locally and selecting the model according to your preferences. See the beginning of the notebook [`prediction_one_parcel.ipynb`](notebooks/prediction_one_parcel.ipynb)  for an example of how to do this.

Save the configuration file of the selected model in json format.

#### 2. Final Train
Once the best configuration of hyperparameters is found, you can train the model on both the training and the validation sets with the [`final_train.py`](training_variants/final_train.py) script. To do so, run
```bash 
python training_variants/final_train.py --configpath <configpath>
```
where `<configpath>` is the path to the selected model configuration file. An example is given in [`config/best_model_config.json`](config/best_model_config.json).

## 4. Test the Model
To test the model on the test set run
```
python test.py --run-name <run-name>
```
where `<run-name>` is the name of the wandb run you want to test. The test set is automatically downloaded.
Several options are available with `python test.py --help`.

## 5. Notebooks
In the `notebooks` folder, you can find several notebooks to reproduce the results of the paper.
1. [`dataset_plot.ipynb`](notebooks/dataset_plot.ipynb) - explore the BreizhCrops and the Reduced BreizhCrops datasets;
2. [`plot_maps.ipynb`](notebooks/plot_maps.ipynb) - load a wandb run and plot the classification maps;
3. [`prediction_one_parcel.ipynb`](notebooks/prediction_one_parcel.ipynb) - view the runs of the wandb project, select one run, and predict the crop type of a single parcel. Make sure to add the sweep name in the [`personal_config.yaml`](config/personal_config.yaml) file under the `sweep` key.
4. [`results_table.ipynb`](notebooks/results_table.ipynb) - load the results of the wandb project and create a table with the results of some selected runs.


## 6. References
> Marc Rußwurm, Nicolas Courty, Remi Emonet, Sebastien Lefévre, Devis Tuia, and Romain Tavenard (2023). End-to-End Learned Early Classification of Time Series for In-Season Crop Type Mapping. ISPRS Journal of Photogrammetry and Remote Sensing. 196. 445-456. https://doi.org/10.1016/j.isprsjprs.2022.12.016