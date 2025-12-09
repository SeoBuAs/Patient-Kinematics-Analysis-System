import sys
import os
sys.path.append('.')
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from experiments.nested_cv import PatientBasedCV, PatientBasedGridSearchCV
from data_loader.kinematics_loader import KinematicsLoader
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class CompleteExperimentNoVIF:
    def __init__(self):
        self.results = {}
        self.best_models = {}
        self.final_models = {}
        self.experiment_dir = None
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
    def define_models(self):
        return {
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
                    'subsample': [0.8, 1],
                    'colsample_bytree': [0.8, 1],
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
        return models
    def run_nested_cv_for_model(self, model_name, model_info, X_train, y_train, X_val, y_val, X_test, y_test):
        logger.info(f"\n=== {model_name} model with fixed validation set (without VIF) ===")
        cv_outer = PatientBasedCV(random_state=42, n_splits=5)
        outer_scores = []
        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_outer.split(X_train, y_train), 1):
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            X_val_fold = X_val
            y_val_fold = y_val
            feature_cols = [col for col in X_train_fold.columns if col != 'patient_id']
            X_train_features = X_train_fold[feature_cols]
            X_val_features = X_val_fold[feature_cols]
            logger.info(f"  Fold {fold_idx}: num features = {len(feature_cols)} (original features without VIF)")
            grid_search = PatientBasedGridSearchCV(
                estimator=model_info['model'],
                param_grid=model_info['param_grid'],
                cv=PatientBasedCV(random_state=42, n_splits=5),
                scoring='f1'
            )
            grid_search.fit(X_train_fold, y_train_fold)
            y_val_pred = grid_search.predict(X_val_features)
            y_val_pred_proba = grid_search.predict_proba(X_val_features)[:, 1]
            val_accuracy = accuracy_score(y_val_fold, y_val_pred)
            val_f1 = f1_score(y_val_fold, y_val_pred)
            val_auc = roc_auc_score(y_val_fold, y_val_pred_proba)
            val_precision = precision_score(y_val_fold, y_val_pred)
            val_recall = recall_score(y_val_fold, y_val_pred)
            outer_scores.append({
                'accuracy': val_accuracy,
                'f1': val_f1,
                'auc': val_auc,
                'precision': val_precision,
                'recall': val_recall,
                'test_proba': y_val_pred_proba,
                'test_true': y_val_fold.values
            })
            fold_results.append({
                'fold': fold_idx,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'val_precision': val_precision,
                'val_recall': val_recall
            })
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        summary = {}
        for metric in metrics:
            values = [score[metric] for score in outer_scores]
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
        auc_values = [score['auc'] for score in outer_scores]
        summary['auc_mean'] = np.mean(auc_values)
        summary['auc_std'] = np.std(auc_values)
        all_test_proba = np.concatenate([score.get('test_proba', []) for score in outer_scores])
        all_test_true = np.concatenate([score.get('test_true', []) for score in outer_scores])
        if len(all_test_proba) > 0 and len(all_test_true) > 0:
            summary['auc_mean'], summary['auc_lower_ci'], summary['auc_upper_ci'] = self.bootstrap_confidence_interval(
                all_test_true, all_test_proba, roc_auc_score
            )
        else:
            summary['auc_lower_ci'] = summary['auc_mean'] - 1.96 * summary['auc_std'] / np.sqrt(len(auc_values))
            summary['auc_upper_ci'] = summary['auc_mean'] + 1.96 * summary['auc_std'] / np.sqrt(len(auc_values))
        logger.info(f"{model_name} Nested CV results (without VIF):")
        logger.info(f"  Accuracy: {summary['accuracy_mean']:.4f} (±{summary['accuracy_std']:.4f})")
        logger.info(f"  F1: {summary['f1_mean']:.4f} (±{summary['f1_std']:.4f})")
        logger.info(f"  AUC: {summary['auc_mean']:.4f} ({summary['auc_lower_ci']:.4f}-{summary['auc_upper_ci']:.4f})")
        feature_cols = [col for col in X_train.columns if col != 'patient_id']
        X_train_features = X_train[feature_cols]
        X_test_features = X_test[feature_cols]
        logger.info(f"Final model training: num features = {len(feature_cols)} (original features without VIF)")
        final_grid_search = PatientBasedGridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['param_grid'],
            cv=PatientBasedCV(random_state=42, n_splits=5),
            scoring='f1'
        )
        final_grid_search.fit(X_train, y_train)
        y_test_pred = final_grid_search.predict(X_test_features)
        y_test_pred_proba = final_grid_search.predict_proba(X_test_features)[:, 1]
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_pred_proba)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_auc, test_auc_lower_ci, test_auc_upper_ci = self.bootstrap_confidence_interval(
            y_test, y_test_pred_proba, roc_auc_score
        )
        logger.info(f"{model_name} final test performance (without VIF):")
        logger.info(f"  Accuracy: {test_accuracy:.4f}")
        logger.info(f"  F1: {test_f1:.4f}")
        logger.info(f"  AUC: {test_auc:.4f} ({test_auc_lower_ci:.4f}-{test_auc_upper_ci:.4f})")
        logger.info(f"  Precision: {test_precision:.4f}")
        logger.info(f"  Recall: {test_recall:.4f}")
        return {
            'fold_results': fold_results,
            'summary': summary,
            'final_model': final_grid_search,
            'test_performance': {
                'accuracy': test_accuracy,
                'f1': test_f1,
                'auc': test_auc,
                'auc_lower_ci': test_auc_lower_ci,
                'auc_upper_ci': test_auc_upper_ci,
                'precision': test_precision,
                'recall': test_recall
            },
            'test_predictions': y_test_pred,
            'test_probabilities': y_test_pred_proba,
            'test_true': y_test.values,
            'best_params': final_grid_search.best_params_
        }
    def run_complete_experiment(self):
        logger.info("=== Starting complete patient-based experiment with fixed validation set (without VIF) ===")
        loader = KinematicsLoader()
        data = loader.load_data(use_vif_selection=False)
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
        logger.info(f"Training data shape: {X_train.shape} (without VIF)")
        logger.info(f"Validation data shape: {X_val.shape} (without VIF)")
        logger.info(f"Test data shape: {X_test.shape} (without VIF)")
        models = self.define_models()
        for model_name, model_info in models.items():
            try:
                result = self.run_nested_cv_for_model(
                    model_name, model_info, X_train, y_train, X_val, y_val, X_test, y_test
                )
                self.results[model_name] = result
                self.final_models[model_name] = result['final_model']
            except Exception as e:
                logger.error(f"{model_name} error during model experiment: {str(e)}")
                continue
        self.save_results()
        return self.results
    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"results/complete_experiment_no_vif_{timestamp}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.save_models()
        self.save_csv_results()
        self.save_predictions()
        logger.info(f"Experiment results (without VIF) saved to {self.experiment_dir}.")
    def save_models(self):
        models_dir = os.path.join(self.experiment_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        for model_name, model in self.final_models.items():
            model_path = os.path.join(models_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            logger.info(f"{model_name} model saved: {model_path}")
    def save_csv_results(self):
        summary_data = []
        for model_name, result in self.results.items():
            summary = result['summary']
            test_perf = result['test_performance']
            best_params = result['best_params']
            summary_data.append({
                'Model': model_name,
                'Test_Accuracy': f"{test_perf['accuracy']:.3f}",
                'Test_F1': f"{test_perf['f1']:.3f}",
                'Test_AUC': f"{test_perf['auc']:.3f} ({test_perf['auc_lower_ci']:.3f}-{test_perf['auc_upper_ci']:.3f})",
                'Test_Precision': f"{test_perf['precision']:.3f}",
                'Test_Recall': f"{test_perf['recall']:.3f}",
                'Best_Params': str(best_params)
            })
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.experiment_dir, "model_summary_no_vif.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Model summary saved (without VIF): {summary_path}")
        for model_name, result in self.results.items():
            fold_data = []
            for fold_result in result['fold_results']:
                fold_data.append({
                    'Model': model_name,
                    'Fold': fold_result['fold'],
                    'Best_Params': str(fold_result['best_params']),
                    'Best_Score': fold_result['best_score'],
                    'Val_Accuracy': fold_result['val_accuracy'],
                    'Val_F1': fold_result['val_f1'],
                    'Val_AUC': fold_result['val_auc'],
                    'Val_Precision': fold_result['val_precision'],
                    'Val_Recall': fold_result['val_recall']
                })
            fold_df = pd.DataFrame(fold_data)
            fold_path = os.path.join(self.experiment_dir, f"{model_name}_fold_details_no_vif.csv")
            fold_df.to_csv(fold_path, index=False)
            logger.info(f"{model_name} fold detailed results saved (without VIF): {fold_path}")
    def save_predictions(self):
        predictions_dir = os.path.join(self.experiment_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        for model_name, result in self.results.items():
            pred_data = {
                'Model': model_name,
                'True_Labels': result['test_true'],
                'Probabilities': result['test_probabilities']
            }
            pred_df = pd.DataFrame(pred_data)
            pred_path = os.path.join(predictions_dir, f"{model_name}_test_predictions_no_vif.csv")
            pred_df.to_csv(pred_path, index=False)
            logger.info(f"{model_name} prediction results saved (without VIF): {pred_path}")
if __name__ == "__main__":
    experiment = CompleteExperimentNoVIF()
    results = experiment.run_complete_experiment()
    logger.info("Complete experiment finished (without VIF)!") 