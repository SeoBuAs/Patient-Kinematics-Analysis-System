import numpy as np
import pandas as pd
import joblib
import os
import logging
import json
from datetime import datetime
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('..')
from data_loader.kinematics_loader import load_kinematics_data
from models.ft_transformer_wrapper import FTTransformerWrapper
from models.tabpfn_wrapper import TabPFNClassifierWrapper
from models.saint_wrapper import SAINTClassifierWrapper
from models.tabnet_wrapper import TabNetWrapper as TabNetClassifierWrapper
class DeepModelExperiment:
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
        models = {}
        models['FTTransformer'] = {
            'model': FTTransformerWrapper(),
            'param_grid': {
                'dim': [16],
                'depth': [1],
                'heads': [2],
                'attn_dropout': [0.3],
                'ff_dropout': [0.3],
                'lr': [1e-3],
                'batch_size': [32],
                'epochs': [20],
                'early_stopping_patience': [5]
            }
        }
        models['TabPFN'] = {
            'model': TabPFNClassifierWrapper(),
            'param_grid': {
                'n_estimators': [4],
                'random_state': [42],
                'softmax_temperature': [1.0],
                'balance_probabilities': [True]
            }
        }
        models['SAINT'] = {
            'model': SAINTClassifierWrapper(),
            'param_grid': {
                'hidden_dim': [32],
                'num_layers': [1],
                'learning_rate': [1e-3],
                'batch_size': [32],
                'num_epochs': [20],
                'dropout': [0.3]
            }
        }
        models['TabNet'] = {
            'model': TabNetClassifierWrapper(),
            'param_grid': {
                'n_d': [4],
                'n_a': [4],
                'n_steps': [2],
                'gamma': [1.0],
                'lambda_sparse': [1e-3],
                'momentum': [0.3],
                'clip_value': [1],
                'max_epochs': [20],
                'patience': [5]
            }
        }
        return models
    def run_nested_cv(self, X, y, model_name, model, param_grid, cv_outer=5, cv_inner=5):
        self.logger.info(f"\n{model_name} model Nested CV starting...")
        outer_cv = StratifiedKFold(n_splits=cv_outer, shuffle=True, random_state=42)
        outer_scores = []
        all_predictions = []
        all_true_labels = []
        outer_loop_details = []
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            self.logger.info(f"  Outer fold {fold + 1}/{cv_outer}")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            inner_cv = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=42)
            try:
                grid_search = GridSearchCV(
                    model,
                    param_grid,
                    cv=inner_cv,
                    scoring='f1',
                    n_jobs=1,
                    verbose=0
                )
                try:
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                except Exception as fit_error:
                    self.logger.error(f"    Model training failed: {str(fit_error)}")
                    try:
                        default_params = {k: v[0] for k, v in param_grid.items()}
                        model.set_params(**default_params)
                        model.fit(X_train, y_train)
                        best_model = model
                        best_params = default_params
                    except Exception as default_error:
                        self.logger.error(f"    Failed even with default parameters: {str(default_error)}")
                        outer_scores.append(0.0)
                        continue
                try:
                    y_pred = best_model.predict(X_test)
                    y_pred_proba = best_model.predict_proba(X_test)
                    if y_pred_proba.shape[1] == 2:
                        y_pred_proba = y_pred_proba[:, 1]
                    metrics = self.calculate_comprehensive_metrics(y_test, y_pred_proba, y_pred)
                    outer_scores.append(metrics['f1'])
                    all_predictions.extend(y_pred_proba)
                    all_true_labels.extend(y_test)
                    outer_loop_details.append({
                        'Model': model_name,
                        'Fold': fold + 1,
                        'F1_Score': round(metrics['f1'], 3),
                        'AUC_Score': round(metrics['auc'], 3),
                        'Accuracy_Score': round(metrics['accuracy'], 3),
                        'Precision_Score': round(metrics['precision'], 3),
                        'Recall_Score': round(metrics['recall'], 3),
                        'Best_Params': str(best_params)
                    })
                    self.logger.info(f"    F1: {metrics['f1']:.4f}")
                    self.logger.info(f"    AUC: {metrics['auc']:.4f} (CI: {metrics['auc_lower_ci']:.4f}-{metrics['auc_upper_ci']:.4f})")
                    self.logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
                    self.logger.info(f"    Precision: {metrics['precision']:.4f}")
                    self.logger.info(f"    Recall: {metrics['recall']:.4f}")
                except Exception as pred_error:
                    self.logger.error(f"    Prediction failed: {str(pred_error)}")
                    outer_scores.append(0.0)
            except Exception as e:
                self.logger.error(f"    Error in fold {fold + 1}: {str(e)}")
                outer_scores.append(0.0)
        if all_predictions:
            try:
                overall_metrics = self.calculate_comprehensive_metrics(
                    np.array(all_true_labels), 
                    np.array(all_predictions), 
                    (np.array(all_predictions) > 0.5).astype(int)
                )
            except Exception as overall_error:
                self.logger.error(f"    Overall results calculation failed: {str(overall_error)}")
                overall_metrics = {key: 0.0 for key in ['auc', 'accuracy', 'f1', 'precision', 'recall']}
        else:
            overall_metrics = {key: 0.0 for key in ['auc', 'accuracy', 'f1', 'precision', 'recall']}
        best_params = {}
        best_model = None
        if 'best_params' in locals():
            best_params = best_params
        elif 'grid_search' in locals() and hasattr(grid_search, 'best_params_'):
            best_params = grid_search.best_params_
        if 'best_model' in locals():
            best_model = best_model
        results = {
            'model_name': model_name,
            'cv_scores': outer_scores,
            'mean_cv_score': np.mean(outer_scores) if outer_scores else 0.0,
            'std_cv_score': np.std(outer_scores) if outer_scores else 0.0,
            'overall_metrics': overall_metrics,
            'best_params': best_params,
            'best_model': best_model,
            'predictions': np.array(all_predictions),
            'true_labels': np.array(all_true_labels),
            'outer_loop_details': outer_loop_details
        }
        self.logger.info(f"  {model_name} complete - Mean F1: {results['mean_cv_score']:.4f} ± {results['std_cv_score']:.4f}")
        return results
    def train_best_model(self, X_train, y_train, X_test, y_test, model_name, model, param_grid):
        self.logger.info(f"Starting training best model: {model_name}")
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        try:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv_inner,
                scoring='f1',
                n_jobs=1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        except Exception as e:
            self.logger.error(f"GridSearch failed, trying with default parameters: {str(e)}")
            try:
                default_params = {k: v[0] for k, v in param_grid.items()}
                model.set_params(**default_params)
                model.fit(X_train, y_train)
                best_model = model
                best_params = default_params
            except Exception as default_error:
                self.logger.error(f"Failed even with default parameters: {str(default_error)}")
                return {
                    'best_model': None,
                    'best_params': {},
                    'y_pred': np.zeros(len(y_test)),
                    'y_pred_proba': np.zeros(len(y_test)),
                    'test_auc': 0.0,
                    'metrics': {key: 0.0 for key in ['auc', 'accuracy', 'f1', 'precision', 'recall']}
                }
        try:
            y_pred_proba = best_model.predict_proba(X_test)
            if y_pred_proba.shape[1] == 2:
                y_pred_proba = y_pred_proba[:, 1]
            y_pred = best_model.predict(X_test)
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred_proba, y_pred)
            self.logger.info(f"{model_name} Test performance:")
            self.logger.info(f"  AUC: {metrics['auc']:.4f} (95% CI: {metrics['auc_lower_ci']:.4f}-{metrics['auc_upper_ci']:.4f})")
            self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"  F1-Score: {metrics['f1']:.4f}")
            self.logger.info(f"  Precision: {metrics['precision']:.4f}")
            self.logger.info(f"  Recall: {metrics['recall']:.4f}")
            return {
                'best_model': best_model,
                'best_params': best_params,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'test_auc': metrics['auc'],
                'metrics': metrics
            }
        except Exception as pred_error:
            self.logger.error(f"Prediction failed: {str(pred_error)}")
            return {
                'best_model': best_model,
                'best_params': best_params,
                'y_pred': np.zeros(len(y_test)),
                'y_pred_proba': np.zeros(len(y_test)),
                'test_auc': 0.0,
                'metrics': {key: 0.0 for key in ['auc', 'accuracy', 'f1', 'precision', 'recall']}
            }
    def run_experiment(self, variables=None, normalize=True):
        self.logger.info("Starting deep learning model experiment...")
        self.logger.info("Loading data...")
        data_dict = load_kinematics_data()
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        X = pd.concat([X_train, X_test], ignore_index=True)
        y = pd.concat([y_train, y_test], ignore_index=True)
        self.logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        self.logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        if data_dict['normalized']:
            self.logger.info("Data already normalized.")
        else:
            self.logger.info("Data not normalized.")
        models = self.define_models()
        for model_name, model_info in models.items():
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"{model_name} model experiment starting")
            self.logger.info(f"{'='*50}")
            try:
                results = self.run_nested_cv(
                    X, y, 
                    model_name, 
                    model_info['model'], 
                    model_info['param_grid']
                )
                self.results[model_name] = results
                try:
                    final_result = self.train_best_model(
                        X, y, X_test, y_test,
                        model_name,
                        model_info['model'],
                        model_info['param_grid']
                    )
                    self.best_models[model_name] = final_result['best_model']
                    self.predictions[model_name] = {
                        'y_pred': final_result['y_pred'],
                        'y_pred_proba': final_result['y_pred_proba'],
                        'y_true': y_test.values,
                        'metrics': final_result['metrics']
                    }
                    self.results[model_name].update({
                        'test_auc': final_result['test_auc'],
                        'best_params': final_result['best_params']
                    })
                except Exception as final_error:
                    self.logger.error(f"{model_name} error during final model training: {str(final_error)}")
            except Exception as e:
                self.logger.error(f"{model_name} error during model experiment: {str(e)}")
                self.results[model_name] = {
                    'model_name': model_name,
                    'error': str(e),
                    'cv_scores': [],
                    'mean_cv_score': 0.0,
                    'std_cv_score': 0.0,
                    'overall_metrics': {key: 0.0 for key in ['auc', 'accuracy', 'f1', 'precision', 'recall']}
                }
        return self.results
    def save_results(self, save_dir="results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = f"{save_dir}/deep_models_{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)
        self.logger.info(f"Results save path: {experiment_dir}")
        summary_data = []
        for model_name, result in self.results.items():
            if 'error' in result:
                summary_data.append({
                    'Model': model_name,
                    'Mean_F1': 0.0,
                    'Std_F1': 0.0,
                    'Overall_AUC': 0.0,
                    'Overall_Accuracy': 0.0,
                    'Overall_F1': 0.0,
                    'Overall_Precision': 0.0,
                    'Overall_Recall': 0.0,
                    'Status': 'Error',
                    'Error_Message': result['error']
                })
            else:
                overall_metrics = result['overall_metrics']
                summary_data.append({
                    'Model': model_name,
                    'Mean_F1': round(result['mean_cv_score'], 3),
                    'Std_F1': round(result['std_cv_score'], 3),
                    'Overall_AUC': round(overall_metrics['auc'], 3),
                    'Overall_Accuracy': round(overall_metrics['accuracy'], 3),
                    'Overall_F1': round(overall_metrics['f1'], 3),
                    'Overall_Precision': round(overall_metrics['precision'], 3),
                    'Overall_Recall': round(overall_metrics['recall'], 3),
                    'Status': 'Success',
                    'Error_Message': ''
                })
        results_df = pd.DataFrame(summary_data)
        results_df = results_df.sort_values('Overall_F1', ascending=False)
        all_outer_loop_details = []
        for model_name, result in self.results.items():
            if 'outer_loop_details' in result:
                all_outer_loop_details.extend(result['outer_loop_details'])
        if all_outer_loop_details:
            outer_loop_df = pd.DataFrame(all_outer_loop_details)
            outer_loop_csv_path = f"{experiment_dir}/deep_models_outer_loop_details.csv"
            outer_loop_df.to_csv(outer_loop_csv_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"Outer Loop detailed results CSV saved: {outer_loop_csv_path}")
        csv_path = f"{experiment_dir}/deep_models_results.csv"
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"Results CSV saved: {csv_path}")
        import json
        json_path = f"{experiment_dir}/deep_models_detailed_results.json"
        json_results = {
            'experiment_info': {
                'timestamp': timestamp,
                'total_models': len(self.results),
                'successful_models': len([r for r in self.results.values() if 'error' not in r]),
                'failed_models': len([r for r in self.results.values() if 'error' in r])
            },
            'model_results': {}
        }
        for model_name, result in self.results.items():
            if 'error' in result:
                json_results['model_results'][model_name] = {
                    'status': 'failed',
                    'error_message': result['error'],
                    'metrics': {
                        'auc': 0.0,
                        'accuracy': 0.0,
                        'f1': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                    }
                }
            else:
                auc_ci = f"{result['overall_metrics']['auc']:.3f}({result['overall_metrics']['auc_lower_ci']:.3f}-{result['overall_metrics']['auc_upper_ci']:.3f})"
                json_results['model_results'][model_name] = {
                    'status': 'success',
                    'cross_validation': {
                        'mean_auc': float(result['mean_cv_score']),
                        'std_auc': float(result['std_cv_score']),
                        'cv_scores': [float(score) for score in result['cv_scores']]
                    },
                    'test_performance': {
                        'auc': {
                            'value': float(result['overall_metrics']['auc']),
                            'confidence_interval': f"{result['overall_metrics']['auc_lower_ci']:.3f}-{result['overall_metrics']['auc_upper_ci']:.3f}",
                            'formatted': auc_ci
                        },
                        'accuracy': float(result['overall_metrics']['accuracy']),
                        'f1_score': float(result['overall_metrics']['f1']),
                        'precision': float(result['overall_metrics']['precision']),
                        'recall': float(result['overall_metrics']['recall']),
                    },
                    'best_hyperparameters': result.get('best_params', {}),
                    'model_info': {
                        'model_name': result['model_name'],
                        'training_time': result.get('training_time', 0.0),
                        'prediction_time': result.get('prediction_time', 0.0)
                    }
                }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Detailed results JSON saved: {json_path}")
        self.save_all_models(experiment_dir, results_df)
        self.save_all_predictions(experiment_dir, results_df)
        self.save_best_model(experiment_dir, results_df)
        self.save_best_predictions(experiment_dir, results_df)
        self.logger.info("\n" + "="*80)
        self.logger.info("Deep learning model experiment results summary")
        self.logger.info("="*80)
        for _, row in results_df.iterrows():
            if row['Status'] == 'Success':
                self.logger.info(f"{row['Model']:15s} | AUC: {row['Mean_AUC']:.4f} ± {row['Std_AUC']:.4f} | Overall AUC: {row['Overall_AUC']:.4f}")
            else:
                self.logger.info(f"{row['Model']:15s} | Error: {row['Error_Message']}")
        return results_df
    def save_best_model(self, experiment_dir, results_df):
        success_models = results_df[results_df['Status'] == 'Success']
        if len(success_models) == 0:
            self.logger.warning("No successful models to save.")
            return
        best_model_name = success_models.iloc[0]['Model']
        best_model_result = self.results[best_model_name]
        self.logger.info(f"Best Model saved: {best_model_name} (AUC: {best_model_result['mean_cv_score']:.4f})")
        best_model_info = {
            'model_name': best_model_name,
            'mean_auc': float(best_model_result['mean_cv_score']),
            'std_auc': float(best_model_result['std_cv_score']),
            'best_params': best_model_result.get('best_params', {}),
            'overall_metrics': best_model_result['overall_metrics'],
            'cv_scores': [float(score) for score in best_model_result['cv_scores']]
        }
        best_model_info_path = f"{experiment_dir}/best_model_info.json"
        with open(best_model_info_path, 'w', encoding='utf-8') as f:
            json.dump(best_model_info, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Best Model info saved: {best_model_info_path}")
        if best_model_result.get('best_model') is not None:
            model = best_model_result['best_model']
            best_model_path = f"{experiment_dir}/best_model"
            try:
                if best_model_name == 'FTTransformer':
                    torch.save(model.state_dict(), f"{best_model_path}.pth")
                    self.logger.info(f"Best Model saved: {best_model_path}.pth")
                elif best_model_name == 'TabPFN':
                    import pickle
                    with open(f"{best_model_path}.pkl", 'wb') as f:
                        pickle.dump(model, f)
                    self.logger.info(f"Best Model saved: {best_model_path}.pkl")
                elif best_model_name == 'TabR':
                    torch.save(model.state_dict(), f"{best_model_path}.pth")
                    self.logger.info(f"Best Model saved: {best_model_path}.pth")
                elif best_model_name == 'TabNet':
                    if hasattr(model, 'save_model'):
                        model.save_model(best_model_path)
                        self.logger.info(f"Best Model saved: {best_model_path}")
                    else:
                        torch.save(model.state_dict(), f"{best_model_path}.pth")
                        self.logger.info(f"Best Model saved: {best_model_path}.pth")
            except Exception as e:
                self.logger.error(f"Best Model save failed: {str(e)}")
    def save_best_predictions(self, experiment_dir, results_df):
        success_models = results_df[results_df['Status'] == 'Success']
        if len(success_models) == 0:
            self.logger.warning("No successful models to save.")
            return
        best_model_name = success_models.iloc[0]['Model']
        best_model_result = self.results[best_model_name]
        self.logger.info(f"Best Model prediction results saved: {best_model_name}")
        if 'predictions' in best_model_result:
            predictions = best_model_result['predictions']
            best_pred_path = f"{experiment_dir}/best_model_predictions.npz"
            np.savez(best_pred_path, **predictions)
            self.logger.info(f"Best Model prediction results saved: {best_pred_path}")
    def save_all_predictions(self, experiment_dir, results_df):
        self.logger.info("Saving all model prediction results...")
        for model_name, result in self.results.items():
            if 'predictions' in result and result.get('best_model') is not None:
                predictions = result['predictions']
                pred_path = f"{experiment_dir}/{model_name}_predictions.npz"
                np.savez(pred_path, **predictions)
                self.logger.info(f"{model_name} prediction results saved: {pred_path}")
    def save_all_models(self, experiment_dir, results_df):
        self.logger.info("Saving all models...")
        for model_name, result in self.results.items():
            if result.get('best_model') is not None:
                model = result['best_model']
                model_path = f"{experiment_dir}/{model_name}_model"
                try:
                    if model_name == 'FTTransformer':
                        torch.save(model.state_dict(), f"{model_path}.pth")
                        self.logger.info(f"{model_name} model saved: {model_path}.pth")
                    elif model_name == 'TabPFN':
                        import pickle
                        with open(f"{model_path}.pkl", 'wb') as f:
                            pickle.dump(model, f)
                        self.logger.info(f"{model_name} model saved: {model_path}.pkl")
                    elif model_name == 'TabR':
                        torch.save(model.state_dict(), f"{model_path}.pth")
                        self.logger.info(f"{model_name} model saved: {model_path}.pth")
                    elif model_name == 'TabNet':
                        if hasattr(model, 'save_model'):
                            model.save_model(f"{model_path}")
                            self.logger.info(f"{model_name} model saved: {model_path}")
                        else:
                            torch.save(model.state_dict(), f"{model_path}.pth")
                            self.logger.info(f"{model_name} model saved: {model_path}.pth")
                except Exception as e:
                    self.logger.error(f"{model_name} model save failed: {str(e)}") 