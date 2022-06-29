from pyexpat import model
import numpy as np
import math
from copy import deepcopy
from sklearn.base import clone

class ScikitAttributionModel():

    def __init__(self, model, baseline=None, retrain=False):

        self.model = deepcopy(model)
        self.retrain = retrain

        self.baseline = baseline
        if self.baseline is None and not self.retrain:
            self.baseline = 0

    def retrain_model(self, X, y):
        model = clone(self.model)
        model.fit(X, y)
        return model

    def predict_function(self, X, y, model=None):
        if model is not None:
            return model.score(X, y)
        return self.model.score(X, y)

    def initial_step(self, X, y):
        return self.predict_function(X, y)

    def fit(self, X, y):
        '''
        X: ndarray(N, D)
        '''
        assert len(X.shape) > 1, "X requires more than 1 features"
        scores = self._fit(X, y)
        return scores


class FESPForScikit(ScikitAttributionModel):

    def __init__(self, model, baseline=None, retrain=False):

        super().__init__(model, baseline, retrain)

    def _fit(self, X, y):
        
        v_n = self.initial_step(X, y)
        n, d = X.shape
        masks = np.split(np.eye(d).flatten(), d, axis=-1)

        inf_values, sup_values = 0, 0

        for mask in masks:
            inf_value, sup_value = self.shapley_loop(X, y, mask)
            sup_value = (0 - sup_value)
            inf_values += inf_value
            sup_values += sup_value
            
        inf_sum = inf_values.sum()
        sup_sum = sup_values.sum()

        w = (v_n - sup_sum) / (inf_sum - sup_sum)
        scores = w * inf_values + (1 - w) * sup_values
        return scores

    def shapley_loop(self, X, y=None, mask=None):

        if self.retrain and self.baseline is not None:
            X_0 = X[:, mask.astype(bool)]
            X_1 = X[:, (1 - mask).astype(bool)] 
        else:
            mask = mask.reshape(1, -1)
            X_0 = X * mask + (1 - mask) * self.baseline
            X_1 = X * (1 - mask) + mask * self.baseline
            mask = mask.flatten()

        model = self.retrain_model(X_0, y) if self.retrain else None
        inf_value = self.predict_function(X_0, y, model)

        model = self.retrain_model(X_1, y) if self.retrain else None
        sup_value = self.predict_function(X_1, y, model)
        
        return inf_value*mask, sup_value*mask


class ESForScikit(ScikitAttributionModel):

    def __init__(self, model, baseline=None, retrain=False):

        super().__init__(model, baseline, retrain)

    def _fit(self, X, y):
        
        v_n = self.initial_step(X, y)
        n, d = X.shape
        masks = np.split(np.eye(d).flatten(), d, axis=-1)

        inf_values = 0

        for mask in masks:
            inf_value = self.shapley_loop(X, y, mask)
            inf_values += inf_value
            
        inf_sum = inf_values.sum()
        scores = inf_values + (v_n - inf_sum) / d

        return scores

    def shapley_loop(self, X, y=None, mask=None):

        if self.retrain:
            X_0 = X[:, mask.astype(bool)] 
        else:
            mask = mask.reshape(1, -1)
            X_0 = X * mask + (1 - mask) * self.baseline
            mask = mask.flatten()

        model = self.retrain_model(X_0, y) if self.retrain else None
        inf_value = self.predict_function(X_0, y, model)

        return inf_value*mask


class OcclusionForScikit(ScikitAttributionModel):

    def __init__(self, model, baseline=None, retrain=False):

        super().__init__(model, baseline, retrain)

    def _fit(self, X, y):
        
        v_n = self.initial_step(X, y)
        n, d = X.shape
        masks = np.split(np.eye(d).flatten(), d, axis=-1)

        sup_values = 0

        for mask in masks:
            sup_value = self.shapley_loop(X, y, mask)
            sup_values += sup_value

        scores = v_n - sup_values

        return scores

    def shapley_loop(self, X, y=None, mask=None):

        if self.retrain:
            X_1 = X[:, (1 - mask).astype(bool)] 
        else:
            mask = mask.reshape(1, -1)
            X_1 = X * (1 - mask) + mask * self.baseline
            mask = mask.flatten()

        model = self.retrain_model(X_1, y) if self.retrain else None
        sup_value = self.predict_function(X_1, y, model)

        return sup_value*mask
