import numpy as np
import pandas as pd
import joblib
import os
import logging
from datetime import datetime
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, BaseCrossValidator
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('..')
from data_loader.kinematics_loader import load_kinematics_data
logger = logging.getLogger(__name__)
class PatientBasedCV(BaseCrossValidator):
    def __init__(self, random_state=42, n_splits=None):
        self.random_state = random_state
        self.n_splits = n_splits
    def split(self, X, y=None, groups=None):
        if not hasattr(X, 'columns') or 'patient_id' not in X.columns:
            raise ValueError("X does not have 'patient_id' column.")
        patient_ids = X['patient_id']
        patients = []
        controls = []
        for pid in patient_ids.unique():
            if 'patient' in str(pid).lower():
                patients.append(pid)
            elif 'control' in str(pid).lower():
                controls.append(pid)
        if len(patients) == 0 or len(controls) == 0:
            raise ValueError("No patient or control data.")
        max_possible_splits = min(len(patients), len(controls))
        if self.n_splits is None:
            n_splits = max_possible_splits
        else:
            n_splits = min(self.n_splits, max_possible_splits)
        np.random.seed(self.random_state)
        patients_shuffled = np.random.permutation(patients)
        controls_shuffled = np.random.permutation(controls)
        for i in range(n_splits):
            val_patients = []
            val_patients.append(patients_shuffled[i])
            val_patients.append(controls_shuffled[i])
            train_mask = ~patient_ids.isin(val_patients)
            val_mask = patient_ids.isin(val_patients)
            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]
            if len(train_indices) > 0 and len(val_indices) > 0:
                yield train_indices, val_indices
    def get_n_splits(self, X=None, y=None, groups=None):
        if X is None or not hasattr(X, 'columns') or 'patient_id' not in X.columns:
            return 3
        patient_ids = X['patient_id']
        patients = []
        controls = []
        for pid in patient_ids.unique():
            if 'patient' in str(pid).lower():
                patients.append(pid)
            elif 'control' in str(pid).lower():
                controls.append(pid)
        max_possible_splits = min(len(patients), len(controls))
        if self.n_splits is None:
            return max_possible_splits
        else:
            return min(self.n_splits, max_possible_splits)
class PatientBasedGridSearchCV:
    def __init__(self, estimator, param_grid, cv=None, scoring='f1', random_state=42):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv if cv is not None else PatientBasedCV(random_state=random_state)
        self.scoring = scoring
        self.random_state = random_state
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = {}
    def fit(self, X, y):
        if not hasattr(X, 'columns') or 'patient_id' not in X.columns:
            raise ValueError("X does not have 'patient_id' column.")
        feature_cols = [col for col in X.columns if col != 'patient_id']
        X_features = X[feature_cols]
        param_combinations = self._generate_param_combinations()
        cv_scores = []
        param_results = []
        for params in param_combinations:
            print(f"Testing parameter combination: {params}")
            try:
                model = self.estimator.__class__()
                base_params = self.estimator.get_params()
                model.set_params(**base_params)
                model.set_params(**params)
            except Exception as e:
                print(f"Error copying model: {e}")
                if hasattr(self.estimator, '__class__'):
                    model = self.estimator.__class__(**params)
                else:
                    continue
            fold_scores = []
            fold_count = 0
            for train_idx, val_idx in self.cv.split(X, y):
                fold_count += 1
                X_train_fold = X_features.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X_features.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                if self.scoring == 'f1':
                    score = f1_score(y_val_fold, y_pred)
                elif self.scoring == 'accuracy':
                    score = accuracy_score(y_val_fold, y_pred)
                elif self.scoring == 'auc':
                    y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                    score = roc_auc_score(y_val_fold, y_pred_proba)
                else:
                    score = f1_score(y_val_fold, y_pred)
                fold_scores.append(score)
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            cv_scores.append(mean_score)
            param_results.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'fold_scores': fold_scores
            })
        best_idx = np.argmax(cv_scores)
        self.best_score_ = cv_scores[best_idx]
        self.best_params_ = param_results[best_idx]['params']
        try:
            self.best_estimator_ = self.estimator.__class__()
            base_params = self.estimator.get_params()
            self.best_estimator_.set_params(**base_params)
            self.best_estimator_.set_params(**self.best_params_)
        except Exception as e:
            print(f"Error creating best model: {e}")
            if hasattr(self.estimator, '__class__'):
                self.best_estimator_ = self.estimator.__class__(**self.best_params_)
            else:
                raise ValueError("Cannot create best model.")
        self.best_estimator_.fit(X_features, y)
        self.cv_results_ = {
            'param_results': param_results,
            'best_params': self.best_params_,
            'best_score': self.best_score_
        }
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info(f"Best performance: {self.best_score_:.4f}")
        return self
    def _generate_param_combinations(self):
        import itertools
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = []
        for values in itertools.product(*param_values):
            param_dict = dict(zip(param_names, values))
            combinations.append(param_dict)
        return combinations
    def predict(self, X):
        if self.best_estimator_ is None:
            raise ValueError("Model not trained. Call fit() first.")
        if hasattr(X, 'columns') and 'patient_id' in X.columns:
            X = X.drop('patient_id', axis=1)
        return self.best_estimator_.predict(X)
    def predict_proba(self, X):
        if self.best_estimator_ is None:
            raise ValueError("Model not trained. Call fit() first.")
        if hasattr(X, 'columns') and 'patient_id' in X.columns:
            X = X.drop('patient_id', axis=1)
        return self.best_estimator_.predict_proba(X)
class NestedCVExperiment:
    def __init__(self):
        self.results = {}
        self.best_models = {}
        self.predictions = {}
        self.logger = logging.getLogger(__name__)
    def bootstrap_confidence_interval(self, y_true, y_pred_proba, metric_func, n_bootstrap=1000, confidence_level=0.95):
        n_samples = len(y_true)
        bootstrap_scores = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sample_y_true = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
            sample_y_pred_proba = y_pred_proba[indices]
            try:
                if metric_func == roc_auc_score:
                    score = metric_func(sample_y_true, sample_y_pred_proba)
                else:
                    sample_y_pred = (sample_y_pred_proba > 0.5).astype(int)
                    score = metric_func(sample_y_true, sample_y_pred)
                bootstrap_scores.append(score)
            except:
                bootstrap_scores.append(0.0)
        bootstrap_scores = np.array(bootstrap_scores)
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        lower_ci = np.percentile(bootstrap_scores, lower_percentile)
        upper_ci = np.percentile(bootstrap_scores, upper_percentile)
        actual_score = metric_func(y_true, y_pred_proba) if metric_func == roc_auc_score else metric_func(y_true, (y_pred_proba > 0.5).astype(int))
        return actual_score, lower_ci, upper_ci
    def calculate_comprehensive_metrics(self, y_true, y_pred_proba, y_pred):
        metrics = {}
        metrics['auc'], metrics['auc_lower_ci'], metrics['auc_upper_ci'] = self.bootstrap_confidence_interval(
            y_true, y_pred_proba, roc_auc_score
        )
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        return metrics
    def define_models(self):
        self.models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'param_grid': {
                    'n_estimators': [30, 50],
                    'max_depth': [1, 3],
                    'min_samples_split': [10, 20],
                    'min_samples_leaf': [4, 8]
                }
            },
            'Bagging': {
                'model': BaggingClassifier(random_state=42),
                'param_grid': {
                    'n_estimators': [30, 50],
                    'max_samples': [0.8, 1],
                    'max_features': [0.8, 1]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss', device='cpu'),
                'param_grid': {
                    'n_estimators': [30, 50],
                    'max_depth': [1, 3],
                    'learning_rate': [0.01, 0.1],
                    'min_child_weight': [10, 20]
                }
            },
            'MLP': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'param_grid': {
                    'hidden_layer_sizes': [(30,), (50,)],
                    'alpha': [0.01, 0.1],
                    'learning_rate_init': [0.001, 0.0001]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'param_grid': {
                    'C': [0.001, 0.01, 0.1],
                    'penalty': ['l2'],
                    'solver': ['lbfgs']
                }
            },
            'SVC': {
                'model': SVC(random_state=42, probability=True),
                'param_grid': {
                    'C': [0.001, 0.01, 0.1],
                    'gamma': ['scale'],
                    'kernel': ['rbf']
                }
            }
        }
        return self.models
    def run_fixed_validation_nested_cv(self, X_train, y_train, X_val, y_val, X_test, y_test, model_name, model, param_grid, cv_outer=5):
        self.logger.info(f"Starting: {model_name} model fixed Validation Set Nested CV")
        outer_scores = []
        outer_predictions = []
        outer_models = []
        for outer_fold in range(cv_outer):
            self.logger.info(f"Outer Loop Fold {outer_fold + 1}/{cv_outer}")
            cv_inner = PatientBasedCV(random_state=42 + outer_fold)
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv_inner,
                scoring='f1',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            X_val_features = X_val.drop('patient_id', axis=1)
            y_val_pred_proba = best_model.predict_proba(X_val_features)[:, 1]
            y_val_pred = best_model.predict(X_val_features)
            val_metrics = self.calculate_comprehensive_metrics(y_val, y_val_pred_proba, y_val_pred)
            outer_scores.append(val_metrics)
            outer_predictions.append({
                'y_val_pred': y_val_pred,
                'y_val_pred_proba': y_val_pred_proba
            })
            outer_models.append(best_model)
            self.logger.info(f"  Fold {outer_fold + 1} - F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        avg_val_metrics = {}
        for metric in ['f1', 'auc', 'accuracy', 'precision', 'recall']:
            values = [score[metric] for score in outer_scores]
            avg_val_metrics[f'avg_{metric}'] = np.mean(values)
            avg_val_metrics[f'std_{metric}'] = np.std(values)
        best_fold_idx = np.argmax([score['auc'] for score in outer_scores])
        best_model = outer_models[best_fold_idx]
        best_params = grid_search.best_params_
        outer_summary = {
            'fold_count': cv_outer,
            'best_fold_idx': best_fold_idx,
            'best_fold_auc': outer_scores[best_fold_idx]['auc'],
            'all_fold_aucs': [score['auc'] for score in outer_scores],
            'all_fold_f1s': [score['f1'] for score in outer_scores],
            'auc_stability': np.std([score['auc'] for score in outer_scores]),
            'f1_stability': np.std([score['f1'] for score in outer_scores])
        }
        X_test_features = X_test.drop('patient_id', axis=1)
        y_test_pred_proba = best_model.predict_proba(X_test_features)[:, 1]
        y_test_pred = best_model.predict(X_test_features)
        test_metrics = self.calculate_comprehensive_metrics(y_test, y_test_pred_proba, y_test_pred)
        self.logger.info(f"\n{model_name} fixed Validation Set Nested CV results:")
        self.logger.info(f"  Validation performance (mean ± std):")
        self.logger.info(f"    F1: {avg_val_metrics['avg_f1']:.4f} ± {avg_val_metrics['std_f1']:.4f}")
        self.logger.info(f"    AUC: {avg_val_metrics['avg_auc']:.4f} ± {avg_val_metrics['std_auc']:.4f}")
        self.logger.info(f"    Accuracy: {avg_val_metrics['avg_accuracy']:.4f} ± {avg_val_metrics['std_accuracy']:.4f}")
        self.logger.info(f"  Model stability:")
        self.logger.info(f"    AUC stability (std): {outer_summary['auc_stability']:.4f}")
        self.logger.info(f"    F1 stability (std): {outer_summary['f1_stability']:.4f}")
        self.logger.info(f"    Best performing fold: {best_fold_idx + 1} (AUC: {outer_summary['best_fold_auc']:.4f})")
        self.logger.info(f"  Best parameters: {best_params}")
        self.logger.info(f"\n{model_name} Test performance (best model):")
        self.logger.info(f"  F1: {test_metrics['f1']:.4f}")
        self.logger.info(f"  AUC: {test_metrics['auc']:.4f} (95% CI: {test_metrics['auc_lower_ci']:.4f}-{test_metrics['auc_upper_ci']:.4f})")
        self.logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        return {
            'best_model': best_model,
            'best_params': best_params,
            'val_metrics': avg_val_metrics,
            'test_metrics': test_metrics,
            'y_test_pred': y_test_pred,
            'y_test_pred_proba': y_test_pred_proba,
            'outer_loop_summary': outer_summary,
            'outer_loop_details': {
                'fold_scores': outer_scores,
                'fold_predictions': outer_predictions,
                'fold_models': outer_models
            }
        }
    def run_simple_validation(self, X_train, y_train, X_val, y_val, X_test, y_test, model_name, model, param_grid):
        self.logger.info(f"Starting: {model_name} model simple Validation")
        cv_inner = PatientBasedCV(random_state=42)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv_inner,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        X_val_features = X_val.drop('patient_id', axis=1)
        y_val_pred_proba = best_model.predict_proba(X_val_features)[:, 1]
        y_val_pred = best_model.predict(X_val_features)
        val_metrics = self.calculate_comprehensive_metrics(y_val, y_val_pred_proba, y_val_pred)
        X_test_features = X_test.drop('patient_id', axis=1)
        y_test_pred_proba = best_model.predict_proba(X_test_features)[:, 1]
        y_test_pred = best_model.predict(X_test_features)
        test_metrics = self.calculate_comprehensive_metrics(y_test, y_test_pred_proba, y_test_pred)
        self.logger.info(f"{model_name} Validation performance:")
        self.logger.info(f"  F1: {val_metrics['f1']:.4f}")
        self.logger.info(f"  AUC: {val_metrics['auc']:.4f}")
        self.logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        self.logger.info(f"  Best parameters: {best_params}")
        self.logger.info(f"{model_name} Test performance:")
        self.logger.info(f"  F1: {test_metrics['f1']:.4f}")
        self.logger.info(f"  AUC: {test_metrics['auc']:.4f} (95% CI: {test_metrics['auc_lower_ci']:.4f}-{test_metrics['auc_upper_ci']:.4f})")
        self.logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        return {
            'best_model': best_model,
            'best_params': best_params,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'y_test_pred': y_test_pred,
            'y_test_pred_proba': y_test_pred_proba
        }
    def run_experiment(self, variables=None):
        self.logger.info("=== Starting fixed Validation Set Nested CV experiment ===")
        from data_loader.kinematics_loader import KinematicsLoader
        loader = KinematicsLoader()
        data = loader.load_data()
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
        self.logger.info(f"Variables: {variables if variables else 'all variables'}")
        self.logger.info(f"VIF-based Feature Selection applied (max VIF: 10.0)")
        self.logger.info(f"Data normalization: auto (StandardScaler)")
        self.logger.info(f"Training data: {X_train.shape}")
        self.logger.info(f"Validation data: {X_val.shape}")
        self.logger.info(f"Test data: {X_test.shape}")
        models = self.define_models()
        results = {}
        for model_name, model_info in models.items():
            self.logger.info(f"\n{model_name} model evaluation in progress...")
            try:
                result = self.run_fixed_validation_nested_cv(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    model_name, model_info['model'], model_info['param_grid'], cv_outer=5
                )
                results[model_name] = result
            except Exception as e:
                self.logger.error(f"Error during experiment: {str(e)}")
                self.logger.error(f"Detailed error information:\n{str(e)}")
                continue
        self.analyze_outer_loop_stability(results)
        self.results = results
        self.visualize_outer_loop_results(results, save_dir="results")
        return results
    def analyze_outer_loop_stability(self, results):
        self.logger.info("\n=== Outer Loop stability analysis ===")
        for model_name, result in results.items():
            if 'outer_loop_summary' in result:
                summary = result['outer_loop_summary']
                self.logger.info(f"\n{model_name} model stability:")
                self.logger.info(f"  AUC range: {min(summary['all_fold_aucs']):.4f} - {max(summary['all_fold_aucs']):.4f}")
                self.logger.info(f"  F1 range: {min(summary['all_fold_f1s']):.4f} - {max(summary['all_fold_f1s']):.4f}")
                self.logger.info(f"  AUC stability (std): {summary['auc_stability']:.4f}")
                self.logger.info(f"  F1 stability (std): {summary['f1_stability']:.4f}")
                if summary['auc_stability'] < 0.05:
                    stability_grade = "Very Stable"
                elif summary['auc_stability'] < 0.1:
                    stability_grade = "Stable"
                elif summary['auc_stability'] < 0.15:
                    stability_grade = "Moderate"
                else:
                    stability_grade = "Unstable"
                self.logger.info(f"  Stability grade: {stability_grade}")
    def save_results(self, save_dir="results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(save_dir, f"experiment_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        results_file = os.path.join(experiment_dir, "results.json")
        self.save_all_models(experiment_dir)
        self.save_all_predictions(experiment_dir)
        self.logger.info(f"Experiment results saved to {experiment_dir}.")
        return experiment_dir
    def save_all_models(self, experiment_dir):
        models_dir = os.path.join(experiment_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        pass
    def save_all_predictions(self, experiment_dir):
        predictions_dir = os.path.join(experiment_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        if hasattr(self, 'results') and self.results:
            outer_loop_data = []
            for model_name, result in self.results.items():
                if 'outer_loop_details' in result:
                    details = result['outer_loop_details']
                    for fold_idx, (score, pred) in enumerate(zip(details['fold_scores'], details['fold_predictions'])):
                        outer_loop_data.append({
                            'model_name': model_name,
                            'fold_idx': fold_idx + 1,
                            'f1_score': score['f1'],
                            'auc_score': score['auc'],
                            'accuracy': score['accuracy'],
                            'precision': score['precision'],
                            'recall': score['recall']
                        })
            if outer_loop_data:
                outer_loop_df = pd.DataFrame(outer_loop_data)
                outer_loop_csv_path = os.path.join(experiment_dir, "outer_loop_detailed_results.csv")
                outer_loop_df.to_csv(outer_loop_csv_path, index=False, encoding='utf-8-sig')
                self.logger.info(f"Outer Loop CSV Saved: {outer_loop_csv_path}")
        pass
class NestedCVExperiment:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    def run_experiment(self, variables=None, normalize=True):
        simple_exp = SimpleValidationExperiment()
        return simple_exp.run_experiment(variables, normalize) 