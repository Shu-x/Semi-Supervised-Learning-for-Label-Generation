import numpy as np

class NDCG_at_k:
    '''
    Normalised Discounted Cumulative Gain (NDCG) is a ranking quality metric.
    It varies from 0.0 to 1.0, with 1.0 representing the ideal ranking
    of the entities where all relevant items are at the top of the list.

    Author:
       Lee Shuxian
    '''
    def __init__(self, y_true, y_pred, k=1):
        '''Initialize
        Attributes:
            y_true::array, shape = [n_samples]
                Ground truth labels represended as integers (1 or 0).
            y_pred::array, shape = [n_samples]
                Predicted probablities of a binary class label
            k::int
                number of items in the list to focus on
        '''
        self.y_true = y_true
        self.y_pred = y_pred
        self.k = k

    def dcg(self, y_true, y_score, k):
        order = np.argsort(y_score)[::-1]   # Sort indices by predicted scores in descending order
        y_true = np.take(y_true, order[:k]) # Take top k true labels based on sorted order  
        
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(1, k + 1) + 1)
        dcg = np.sum(gains / discounts)
    
        return dcg
    
    def ndcg(self):
        ideal_dcg = self.dcg(self.y_true, self.y_true, self.k)
        actual_dcg = self.dcg(self.y_true, self.y_pred, self.k)
        if ideal_dcg > 0:
            ndcg = actual_dcg / ideal_dcg
        else:
            ndcg = 0
        
        return ndcg