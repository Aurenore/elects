import torch

def extract_mu_thresh(class_prob, y_true, p_tresh, mu_default):
    """ Extract mus for each class from the class probability.
        Mus are the first day when the mean probability of the class is above p_tresh. 
        If the mean probability is never above p_tresh, mu is set to mu_default.
    
    Args:
        class_prob (np.array): shape (n_samples, n_days, nb_classes)
        y_true (np.array): shape (n_samples)
        p_tresh (float): threshold for the mean probability
        mu_default (int): default value for mu when mean probability is never above p_tresh
        
    Returns:
        mus (list): list of mus for each class
    """
    mus = []
    nb_classes = class_prob.shape[2]
    for label in range(nb_classes):
        mean_prob = class_prob[y_true == label].mean(axis=0) # shape (n_days, nb_classes)
        mean_prob_i = mean_prob[:, label] # shape (n_days)
        # set mu when mean_prob > p_tresh, for the first time 
        seq = mean_prob_i > p_tresh
        mu = seq.argmax()
        if mu==0 and mean_prob_i[0]<p_tresh: 
            mu = mu_default
        mus.append(mu)
    return mus

def mus_should_be_updated(config, epoch, freq_update=5):
    """ checks if the mus should be updated, depending on the loss function and the epoch. 
        mus should be updated for the daily_reward_lin_regr_loss, starting from the start_decision_head_training epoch, and every 5 epochs.
    
    Args:
        config (wandb.config): configuration of the run
        epoch (int): current epoch
    
    Returns:
        bool: True if the mus should be updated, False otherwise    
    """
    if "lin_regr" in config.loss and epoch>=config.start_decision_head_training and epoch%freq_update==0:
        return True
    else:
        return False

def update_mus_during_training(config, criterion, stats, epoch, mus, mu_default):
    """ updates the mus during training, depending on the loss function and the epoch.
    
    Args:
        config (wandb.config): configuration of the run
        criterion: the criterion
        stats: the statistics
        epoch (int): current epoch
        mus: mus for the daily_reward_lin_regr_loss
        mu_default: default mu for the daily_reward_lin_regr_loss
    Returns:
        mus: updated mus
    """
    # compute the new mus from the classification probabilities
    mus = extract_mu_thresh(stats["class_probabilities"], stats["targets"][:, 0], config.p_thresh, mu_default)
    criterion.update_mus(torch.tensor(mus))
    print(f"At epoch {epoch}, updated parameter mus: \n{mus}")
    return mus


def get_mus_from_config(run_config):
    return torch.tensor(run_config.mus)
    