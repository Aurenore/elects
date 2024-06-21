import sys 
import os 
import numpy as np
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)
from utils.metrics import get_std_score


class TestStdScore():    
    def test_get_std_score_zero_std(self):
        stats =  {"targets": np.array([[1, 1], [1, 1], [0, 0], [0, 0]]), "t_stop": np.array([0, 0, 1, 1])}
        nclasses = 2
        assert np.isclose(get_std_score(stats, nclasses), 0.), "Zero std test failed"
        
    def test_get_std_score_one_class(self):
        stats = {"targets": np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), "t_stop": np.array([0, 1, 0, 1])}
        std = np.std(stats["t_stop"])
        nclasses = 1
        assert np.isclose(get_std_score(stats, nclasses), std), "One class test failed"
        
        
        

        