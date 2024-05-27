def harmonic_mean_score(accuracy: float, earliness:float)->float:
    return 2.*earliness*accuracy/(earliness+accuracy)