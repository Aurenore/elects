import os
import matplotlib.pyplot as plt
from data.breizhcrops import SELECTED_BANDS

def plot_one_parcel_prediction(class_names, log_class_probabilities, stopping_criteria, predictions_at_t_stop, t_stop, y_true, model_path):
    figsize = (6, 3)
    # plot the classification probabilities with respect to time 
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.plot(log_class_probabilities[0].exp().detach().numpy(), label=class_names)
    ax.set_title("Class probabilities")
    ax.set_ylim(0,1)
    ax.set_ylabel("Class probability")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.axvline(x=t_stop, color='black', linestyle='--')
    
    # save the figure 
    fig_filename = os.path.join(model_path, "one_parcel_prediction_class_prob.png")
    fig.savefig(fig_filename, bbox_inches='tight')
    print(f"Figure saved at: {fig_filename}")
    
    # plot the timestamps left 
    fig, ax = plt.subplots(1, figsize=figsize)
    color = "#4F8FCC"
    ax.plot(stopping_criteria[0].detach().numpy(), color=color)
    ax.set_title("Timestamps left until stopping")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Timestamp left")

    # set a vertical line at the stopping time t_stop 
    ax.axvline(x=t_stop, color='black', linestyle='--')
    ymin, ymax = ax.get_ylim()
    ax.text(t_stop+3., ymax-10., f"stop: {t_stop.item()}", verticalalignment='center')
    fig.tight_layout()
    # save the figure 
    fig_filename = os.path.join(model_path, "one_parcel_prediction_timestamps_left.png")
    fig.savefig(fig_filename, bbox_inches='tight')
    print(f"Figure saved at: {fig_filename}")

    print("Predicted label: ", class_names[predictions_at_t_stop.item()])
    print("True label: ", class_names[y_true[0,0]])
    
    
def plot_spectral_bands(X, fig, ax, model_path):
    # get the indices, a list of the indices of the non zero elements
    idx = [i for i, x in enumerate(X[:,0]!=0) if x]
    # only plot the non zero elements
    for i in range(13):
        ax.plot(idx, X[X[:,i]!=0,i], label=f"{SELECTED_BANDS['L1C'][i]}", linewidth=0.5)

    ax.set_title("Parcel data", fontsize=24)
    ax.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Day of year", fontsize=18)
    ax.grid()
    # save the figure 
    fig_filename = os.path.join(model_path, "one_parcel_data.png")
    fig.savefig(fig_filename, bbox_inches='tight')
    print(f"Figure saved at: {fig_filename}")