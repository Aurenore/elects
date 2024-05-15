def harmonic_mean_score(accuracy: float, classification_earliness:float)->float:
    return 2.*(1.-classification_earliness)*accuracy/(1.-classification_earliness+accuracy)