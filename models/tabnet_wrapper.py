import torch
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import logging
import warnings
class TabNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, logger=None,
                 n_d=8,
                 n_a=8,
                 n_steps=3,
                 gamma=1.0,
                 n_shared=2,
                 cat_idxs=[],
                 cat_dims=[],
                 cat_emb_dim=1,
                 lambda_sparse=1e-3,
                 momentum=0.3,
                 clip_value=2,
                 optimizer_fn=torch.optim.Adam,
                 optimizer_params=dict(lr=1e-3),
                 scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
                 scheduler_params=dict(T_max=100),
                 mask_type='entmax',
                 max_epochs=100,
                 patience=10,
                 batch_size=256,
                 virtual_batch_size=128,
                 num_workers=0,
                 drop_last=False,
                 seed=42,
                 verbose=False):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_shared = n_shared
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dim = cat_emb_dim
        self.lambda_sparse = lambda_sparse
        self.momentum = momentum
        self.clip_value = clip_value
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params
        self.mask_type = mask_type
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.seed = seed
        self.verbose = verbose
        self.model_ = None
        self.classes_ = None
        self._feature_columns = None
        self._is_fitted = False
        self.logger = logger or logging.getLogger(__name__)
    def fit(self, X, y, **fit_params):
        if isinstance(X, pd.DataFrame): 
            X_df = X.copy()
            self._feature_columns = list(X_df.columns)
        elif isinstance(X, np.ndarray):
            if self._feature_columns is None or len(self._feature_columns) != X.shape[1]: 
                self._feature_columns = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=self._feature_columns)
        else: 
            raise TypeError(f"Input X must be DataFrame or NumPy array, got {type(X)}")
        if isinstance(y, pd.Series): 
            y_values = y.values
        elif isinstance(y, pd.DataFrame): 
            y_values = y.iloc[:, 0].values
        elif isinstance(y, (np.ndarray, list)): 
            y_values = np.asarray(y)
        else: 
            raise TypeError(f"Input y must be Series, DataFrame, NumPy array or list, got {type(y)}")
        X_df.reset_index(drop=True, inplace=True)
        y_values = y_values.flatten()
        if X_df.shape[0] != y_values.shape[0]: 
            raise ValueError(f"Length mismatch: X({X_df.shape[0]}) != y({y_values.shape[0]})")
        self.classes_ = np.unique(y_values)
        if len(self.classes_) == 1:
            warnings.warn("Only one class found in y.")
        try:
            model_params = {
                'optimizer_fn': self.optimizer_fn,
                'optimizer_params': self.optimizer_params,
                'scheduler_fn': self.scheduler_fn,
                'scheduler_params': self.scheduler_params,
                'mask_type': self.mask_type,
                'n_d': self.n_d,
                'n_a': self.n_a,
                'n_steps': self.n_steps,
                'gamma': self.gamma,
                'n_shared': self.n_shared,
                'cat_idxs': self.cat_idxs,
                'cat_dims': self.cat_dims,
                'cat_emb_dim': self.cat_emb_dim,
                'lambda_sparse': self.lambda_sparse,
                'momentum': self.momentum,
                'clip_value': self.clip_value,
                'verbose': self.verbose
            }
            try:
                model_params['n_ind'] = self.n_ind
            except:
                pass
            self.model_ = TabNetClassifier(**model_params)
            self.logger.info(f"TabNet model initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error during TabNet initialization: {e}")
            self.model_ = None
            raise e
        try:
            self.model_.fit(
                X_train=X_df.values,
                y_train=y_values,
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=self.batch_size,
                virtual_batch_size=self.virtual_batch_size,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                **fit_params
            )
            self._is_fitted = True
            try:
                y_pred = self.predict(X_df)
                y_pred_proba = self.predict_proba(X_df)
                train_auc = roc_auc_score(y_values, y_pred_proba[:, 1])
                train_acc = accuracy_score(y_values, y_pred)
                self.logger.info(f"TabNet model fitted successfully. - Train AUC: {train_auc:.4f}, Train accuracy: {train_acc:.4f}")
            except Exception as e:
                self.logger.info(f"TabNet model fitted successfully. - Performance evaluation failed: {str(e)}")
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error during TabNet training: {error_msg}")
            if "pop from empty list" in error_msg:
                self.logger.warning(f"'pop from empty list' error during TabNet training (ignoring and continuing): {error_msg}")
                self._is_fitted = True
            else:
                self._is_fitted = False
        return self
    def _prepare_predict_input(self, X):
        if not self._is_fitted: 
            raise RuntimeError("This TabNetWrapper instance is not fitted yet. Please call fit() before making predictions.")
        if self.model_ is None: 
            raise RuntimeError("Internal TabNet model is not available. Model may not have been initialized properly.")
        if isinstance(X, pd.DataFrame): 
            X_df = X.copy()
        elif isinstance(X, np.ndarray):
            if self._feature_columns is None or len(self._feature_columns) != X.shape[1]: 
                raise ValueError("NumPy predict input: feature names/order unknown or mismatch.")
            X_df = pd.DataFrame(X, columns=self._feature_columns)
        else: 
            raise TypeError(f"Input X must be DataFrame or NumPy array, got {type(X)}")
        if self._feature_columns:
            expected_cols = set(self._feature_columns)
            current_cols = set(X_df.columns)
            missing_cols = expected_cols - current_cols
            if missing_cols: 
                raise ValueError(f"Predict input missing columns: {missing_cols}")
            extra_cols = current_cols - expected_cols
            if extra_cols: 
                warnings.warn(f"Predict input has extra columns: {extra_cols}. They will be ignored.")
            cols_to_select = self._feature_columns[:]
            if list(X_df.columns) != cols_to_select:
                try: 
                    X_df = X_df[cols_to_select]
                except KeyError as e: 
                    raise ValueError(f"Column mismatch during reordering. Missing: {e}") from e
        if self._feature_columns and list(X_df.columns) != self._feature_columns:
            X_df = X_df[self._feature_columns]
        return X_df.values
    def predict_proba(self, X):
        X_values = self._prepare_predict_input(X)
        if self.classes_ is None: 
            raise RuntimeError("Classes not stored. Model might not be fitted correctly.")
        try:
            probabilities = self.model_.predict_proba(X_values)
            return probabilities
        except Exception as e:
            self.logger.error(f"Error during TabNet prediction: {e}")
            if len(self.classes_) == 2:
                return np.zeros((len(X_values), 2))
            else:
                return np.zeros((len(X_values), len(self.classes_)))
    def predict(self, X):
        probabilities = self.predict_proba(X)
        if np.isnan(probabilities).any():
            warnings.warn("NaN values found in predicted probabilities. Replacing NaN with 0 before argmax.")
            probabilities = np.nan_to_num(probabilities, nan=0.0)
        predicted_indices = np.argmax(probabilities, axis=1)
        if self.classes_ is None: 
            raise RuntimeError("Classes not loaded, cannot map indices to labels.")
        predicted_labels = self.classes_[predicted_indices]
        return predicted_labels
    def set_params(self, **params):
        if not params:
            return self
        valid_params = {}
        for param, value in params.items():
            if hasattr(self, param):
                if param in ['optimizer_params', 'scheduler_params'] and isinstance(value, dict):
                    current_value = getattr(self, param, {})
                    current_value.update(value)
                    valid_params[param] = current_value
                else:
                    valid_params[param] = value
            else:
                self.logger.warning(f"Parameter '{param}' not found in TabNetWrapper. Ignoring.")
        for param, value in valid_params.items():
            setattr(self, param, value)
        self.model_ = None
        self._is_fitted = False
        return self
    def get_params(self, deep=True):
        return super().get_params(deep=deep)
    @property
    def _estimator_type(self):
        return "classifier"
    def __getstate__(self):
        state = self.__dict__.copy()
        state['model_'] = None
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model_ = None 