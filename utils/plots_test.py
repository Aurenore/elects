import datetime
import seaborn 
import seaborn as sns
import matplotlib.pyplot as plt
from data import LABELS_NAMES

PALETTE=sns.color_palette("colorblind")

import numpy as np 


def add_doys_lines(ax, dates_of_interest:list=["2017-05-01", "2017-06-01","2017-07-01","2017-08-01","2017-09-01"], ymax=1.1, ymin=0.):
    doys_of_interest = [datetime.datetime.strptime(d,"%Y-%m-%d").timetuple().tm_yday for d in dates_of_interest]
    for d, date in zip(doys_of_interest, dates_of_interest):
        ax.axvline(d, color="gray", ymin=ymin, ymax=ymax, linestyle="--")
        ax.text(d, ymax, date.replace("2017-0",""), ha="center", fontsize=8)


def plot(doys, class_probabilities, stop_probabilities):
    fig, axs = plt.subplots(2,1, figsize=(6,4), sharex=True)
    seaborn.despine()
    ax = axs[0]
    ax.plot(doys, class_probabilities)
    #ax.set_xlabel("day of year")
    ax.set_ylabel("class score")
    add_doys_lines(ax)

    ax = axs[1]
    ax.plot(doys, stop_probabilities)
    ax.set_xlabel("day of year")
    ax.set_ylabel("stopping probability")
    add_doys_lines(ax)
    

def plot_id(id, stats, doys_dict, model, device, test_ds):
    idx = list(stats["ids"][:,0]).index(id)

    X,y, id = test_ds[idx]
    doys = doys_dict[id]

    t_stop = stats["t_stop"][idx,0]
    doy_stop = doys[t_stop]
    date_stop = datetime.datetime(2017,1,1) + datetime.timedelta(days=int(doy_stop-1))

    msk = X[:,0]>0
    X = X[msk]
    X = X.unsqueeze(0).to(device)
    log_class_probabilities, probability_stopping = model(X)

    class_probabilities = log_class_probabilities[0].cpu().detach().exp()
    stop_probabilities = probability_stopping[0].cpu().detach()

    plot(doys, class_probabilities, stop_probabilities)
    print(id)
    y = stats["targets"][idx,0]
    print(f"class {y}")

    pred = stats["predictions_at_t_stop"][idx,0]
    print(f"pred {pred}")

    print(t_stop, doy_stop, date_stop)


def plot_doy_prob(id, stats, doys_dict, model, device, test_ds, fig, ax, color=None, alpha=1.0):
    idx = list(stats["ids"][:,0]).index(id)

    X,y, id = test_ds[idx]
    doys = doys_dict[id]

    msk = X[:,0]>0
    X = X[msk]
    X = X.unsqueeze(0).to(device)
    log_class_probabilities, probability_stopping = model(X)

    class_probabilities = log_class_probabilities[0].cpu().detach().exp()
    stop_probabilities = probability_stopping[0].cpu().detach()

    if doys.shape[0] != stop_probabilities.shape[0]:
        if doys.shape[0] > stop_probabilities.shape[0]:
            doys = doys[:stop_probabilities.shape[0]]
        else:
            stop_probabilities = stop_probabilities[:doys.shape[0]]
    
    
    ax.plot(doys, stop_probabilities, color=color, alpha=alpha)


    return fig, ax


def plot_all_doy_probs(stats, doys_dict, palette=PALETTE, nclasses=9, class_names=LABELS_NAMES, alpha=0.2, nsamples=500):
    fig, axes = plt.subplots(nclasses, figsize=(12,nclasses*3))
    seaborn.despine()
    axes[-1].set_xlabel("day of year")
    for y in range(nclasses):
        ax = axes[y]
        ax.set_title(class_names[y])
        ax.set_ylabel("stopping probability")
        #get indices of class y 
        index_list = [i for i in range(stats["targets"].shape[0]) if stats["targets"][i,0] == y]
        # choose nsamples random samples in index_list
        if nsamples < len(index_list):
            index_list = np.random.choice(index_list, nsamples, replace=False)
        # doys = [doys_dict[stats["ids"][i,0]] for i in index_list]
        # stop_probs = [stats["probability_stopping"][i] for i in index_list]
        color = palette[y]
        print("for class", y, "we have", len(index_list), "samples")
        # print("type of doys: ", type(doys))
        # print("type of stop_probs: ", type(stop_probs))
        for index in index_list: 
            doy = doys_dict[stats["ids"][index,0]]
            seq_length = stats["seqlengths"][index]
            if seq_length > doy.shape[0]:
                seq_length = doy.shape[0]
            else:
                doy = doy[:seq_length]
            stop_prob = stats["probability_stopping"][index][:seq_length]
            ax.plot(doy, stop_prob, color=color, alpha=alpha)
    fig.suptitle("Stopping probabilities for each class")
    fig.tight_layout()
    return fig, axes