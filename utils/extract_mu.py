def extract_mu_thresh(class_prob, y_true, p_tresh, mu_default):
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