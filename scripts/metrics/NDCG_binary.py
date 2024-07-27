import numpy as np

class NDCG:
    '''
    Normalised Discounted Cumulative Gain (NDCG) is a ranking quality metric.
    It varies from 0.0 to 1.0, with 1.0 representing the ideal ranking
    of the entities where all relevant items are at the top of the list.

    Attributes:
    ----------
    y_true: array, shape = [n_samples]
        Ground truth labels represented as integers (1 or 0).
    y_pred: array, shape = [n_samples]
        Predicted probablities of a binary class label, represented as float.
    k: int
        Number of items to focus on. Default k=1.

    Methods:
    -------
    dcg_at_k:
        Returns the Discounted Cumulative Gain for top k items.
    ndcg_at_k:
        Returns the Normalized Discounted Cumulative Gain (DCG/Ideal_DCG) for top k items.
    """

    Author:
    -------
       Lee Shuxian
    '''
    def __init__(self, y_true, y_pred, k=1):
        '''Initialize
        Parameters:
        ----------
        y_true: array, shape = [n_samples]
            Ground truth labels represented as integers (1 or 0).
        y_pred: array, shape = [n_samples]
            Predicted probablities of a binary class label, represented as float.
        k: int
            Number of items to focus on. Default k=1.
        '''
        self.y_true = y_true
        self.y_pred = y_pred
        self.k = k

        self.name = f'NDCG@{self.k}'

    def dcg_at_k(self) -> float:
        '''
        Returns the Discounted Cumulative Gain (DCG) for top k items.

        Returns:
        ----------
        DCG score: float
            DCG score for top k items.
        '''

        order = np.argsort(self.y_pred)[::-1]   # Sort indices by predicted scores in descending order
        y_true = np.take(y_true, order[:self.k]) # Take top k true labels based on sorted order  
        
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(1, self.k + 1) + 1)
        dcg = np.sum(gains / discounts)
    
        return dcg
    
    def ndcg_at_k(self) -> float:
        '''
        Returns the Normalized Discounted Cumulative Gain (DCG/Ideal_DCG) for top k items.

        Returns:
        ----------
        NDCG score for top k items: float
        '''

        ideal_dcg = self.dcg_at_k(self.y_true, self.y_true, self.k)
        actual_dcg = self.dcg_at_k(self.y_true, self.y_pred, self.k)
        if ideal_dcg > 0:
            ndcg = actual_dcg / ideal_dcg
        else:
            ndcg = 0.0
        
        return ndcg