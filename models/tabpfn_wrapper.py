import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import logging
try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("Warning: TabPFN not available. Install with: pip install tabpfn")
class TabPFNClassifierWrapper:
    def __init__(self, logger=None, device='auto', n_estimators=4, 
                 random_state=0, model_path='auto', ignore_pretraining_limits=False,
                 softmax_temperature=0.9, balance_probabilities=False,
                 average_before_softmax=False, fit_mode='fit_preprocessors',
                 memory_saving_mode='auto', n_jobs=-1):
        self.model = None
        self.logger = logger or logging.getLogger(__name__)
        self._estimator_type = "classifier"
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN is not installed. Install with: pip install tabpfn")
        self.device = device
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_path = model_path
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.fit_mode = fit_mode
        self.memory_saving_mode = memory_saving_mode
        self.n_jobs = n_jobs
    def fit(self, X_train, y_train, params=None):
        if params is None:
            params = {
                'device': self.device,
                'n_estimators': self.n_estimators,
                'random_state': self.random_state,
                'model_path': self.model_path,
                'ignore_pretraining_limits': self.ignore_pretraining_limits,
                'softmax_temperature': self.softmax_temperature,
                'balance_probabilities': self.balance_probabilities,
                'average_before_softmax': self.average_before_softmax,
                'fit_mode': self.fit_mode,
                'memory_saving_mode': self.memory_saving_mode,
                'n_jobs': self.n_jobs
            }
        try:
            self.model = TabPFNClassifier(
                device=params['device'],
                n_estimators=params['n_estimators'],
                random_state=params['random_state'],
                model_path=params['model_path'],
                ignore_pretraining_limits=params['ignore_pretraining_limits'],
                softmax_temperature=params['softmax_temperature'],
                balance_probabilities=params['balance_probabilities'],
                average_before_softmax=params['average_before_softmax'],
                fit_mode=params['fit_mode'],
                memory_saving_mode=params['memory_saving_mode'],
                n_jobs=params['n_jobs']
            )
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_train)
            y_pred_proba = self.model.predict_proba(X_train)
            train_auc = roc_auc_score(y_train, y_pred_proba[:, 1])
            train_acc = accuracy_score(y_train, y_pred)
            self.logger.info(f"TabPFN Classifier training complete (params: {params}) - Train AUC: {train_auc:.4f}, Train accuracy: {train_acc:.4f}")
            return True
        except Exception as e:
            self.logger.error(f"TabPFN Classifier training failed: {str(e)}")
            return False
    def predict(self, X_test):
        if self.model is None:
            return np.zeros(len(X_test))
        try:
            return self.model.predict(X_test)
        except Exception as e:
            self.logger.error(f"TabPFN Classifier prediction failed: {str(e)}")
            return np.zeros(len(X_test))
    def predict_proba(self, X_test):
        if self.model is None:
            return np.zeros((len(X_test), 2))
        try:
            return self.model.predict_proba(X_test)
        except Exception as e:
            self.logger.error(f"TabPFN Classifier probability prediction failed: {str(e)}")
            return np.zeros((len(X_test), 2))
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        metrics = {}
        try:
            if len(np.unique(y_test)) == 2:
                metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
        except:
            metrics = {key: 0.0 for key in ['auc', 'accuracy', 'f1', 'precision', 'recall']}
        return metrics, y_pred, y_pred_proba
    def get_params(self, deep=True):
        return {
            'device': self.device,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'model_path': self.model_path,
            'ignore_pretraining_limits': self.ignore_pretraining_limits,
            'softmax_temperature': self.softmax_temperature,
            'average_before_softmax': self.average_before_softmax,
            'fit_mode': self.fit_mode,
            'memory_saving_mode': self.memory_saving_mode,
            'n_jobs': self.n_jobs
        }
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    def score(self, X, y):
        y_pred = self.predict(X)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, y_pred) 