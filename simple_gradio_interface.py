import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
import os
import joblib
import tempfile
import uuid
from data_loader.kinematics_loader import KinematicsLoader
from typing import Optional, List, Tuple
import warnings
import io
import base64
warnings.filterwarnings('ignore')

DEFAULT_MODEL_PATH = \
    "./MLP.joblib"
DEFAULT_DATA_PATH = \
    "./content"
DEFAULT_MAX_VIF = 10.0
DEFAULT_SEED = 40


def align_features_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        common = [c for c in model_features if c in X.columns]
        if common:
            return X[common]
    return X


def build_kernel_explainer(model, X_background: pd.DataFrame):
    predict_fn = lambda X: model.predict_proba(pd.DataFrame(X, columns=X_background.columns))[:, 1]
    explainer = shap.KernelExplainer(predict_fn, X_background, link='logit')
    return explainer


def load_model_and_data(model_path: str = DEFAULT_MODEL_PATH,
                        data_path: str = DEFAULT_DATA_PATH,
                        max_vif: float = DEFAULT_MAX_VIF,
                        seed: int = DEFAULT_SEED):
    """Load MLP model and data (VIF applied, with scaler)"""
    print("Loading model and data...")

    loaded_model = joblib.load(model_path)
    model = loaded_model.best_estimator_ if hasattr(loaded_model, 'best_estimator_') else loaded_model
    print(f"Model loaded: {model_path}")

    loader = KinematicsLoader(data_path=data_path)
    data = loader.load_data(use_vif_selection=True, max_vif=max_vif, random_seed=seed)
    
    scaler = data['scaler']
    print(f"Scaler loaded")

    X_train = data['X_train'].copy()
    X_test = data['X_test'].copy()
    y_test = data['y_test'].copy()

    if 'patient_id' in X_train.columns:
        patient_id_train = X_train['patient_id'].copy()
        X_train = X_train.drop(columns=['patient_id'])
    else:
        patient_id_train = pd.Series(index=X_train.index, dtype=str)

    if 'patient_id' in X_test.columns:
        patient_id_test = X_test['patient_id'].copy()
        X_test = X_test.drop(columns=['patient_id'])
    else:
        patient_id_test = pd.Series(index=X_test.index, dtype=str)

    X_train = align_features_to_model(X_train, model)
    X_test = align_features_to_model(X_test, model)

    bg_n = min(len(X_train), 50)
    X_bg = X_train.sample(n=bg_n, random_state=0) if len(X_train) > 0 else X_train
    shap_explainer = build_kernel_explainer(model, X_bg)

    lime_explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Normal', 'Patient'],
        mode='classification',
        random_state=42,
    )

    if patient_id_test.isna().all() or (patient_id_test == '').all():
        patient_labels = [f"sample_{i}" for i in range(len(X_test))]
    else:
        df_test_with_id = X_test.copy()
        df_test_with_id['patient_id'] = patient_id_test.values
        rep_by_patient = df_test_with_id.groupby('patient_id').mean(numeric_only=True)
        patient_labels = rep_by_patient.index.tolist()

    selected_vars = X_train.columns.tolist()

    print("Model and data loading complete!")
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'patient_id_test': patient_id_test,
        'patient_labels': patient_labels,
        'selected_vars': selected_vars,
        'scaler': scaler,
        'shap_explainer': shap_explainer,
        'lime_explainer': lime_explainer,
    }


def get_patient_list(patient_labels):
    return patient_labels


def create_shap_waterfall(shap_values, feature_names):
    """Create SHAP Waterfall Plot (single sample)"""
    if shap_values is None:
        return None

    try:
        plt.figure(figsize=(12, 8))
        shap_values = np.array(shap_values).reshape(-1)
        explanation = shap.Explanation(values=shap_values, feature_names=feature_names, base_values=0.0)
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.title("SHAP Waterfall Plot", fontsize=14, fontweight='bold')
        plt.tight_layout()
        temp_path = "temp_shap_waterfall.png"
        plt.savefig(temp_path, dpi=300, bbox_inches='tight')
        plt.close()
        return temp_path
    except Exception as e:
        print(f"SHAP waterfall plot error: {e}")
        raise e


def create_lime_plot(lime_exp, label=1):
    if lime_exp is None:
        return None
    if label is None:
        if getattr(lime_exp, 'top_labels', None):
            label = lime_exp.top_labels[0]
        else:
            try:
                avail = lime_exp.available_labels()
                if len(avail):
                    label = avail[0]
            except Exception:
                label = None
    plt.figure(figsize=(12, 8))
    if label is not None:
        lime_exp.as_pyplot_figure(label=label)
    else:
        lime_exp.as_pyplot_figure()
    title_suffix = " (class=Patient)" if label == 1 else " (class=Normal)"
    plt.title("LIME Local Explanation" + title_suffix, fontsize=14, fontweight='bold')
    plt.tight_layout()
    temp_path = "temp_lime_plot.png"
    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close()
    return temp_path


def process_uploaded_file(file_path: str, file_type: str = "csv"):
    """Process uploaded file and convert to DataFrame"""
    try:
        if file_type.lower() == "csv":
            df = pd.read_csv(file_path)
        elif file_type.lower() == "mot":
            # Use KinematicsLoader's read_mot_file method
            loader = KinematicsLoader()
            df = loader.read_mot_file(file_path)
        else:
            return None, "Unsupported file format."
        
        # Data type cleanup
        # Convert string columns to numeric (convert to NaN on error)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if df.empty:
            return None, "No valid numeric data found."
        
        return df, "File loaded successfully."
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def create_time_series_plot(predictions, probabilities, timestamps=None):
    """Generate time series prediction probability plot"""
    try:
        # matplotlib backend reset
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if timestamps is None:
            timestamps = range(len(probabilities))
        
        ax.plot(timestamps, probabilities, 'b-', linewidth=2, label='Patient Probability')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Threshold')
        
        # Find and highlight maximum probability
        max_prob_idx = np.argmax(probabilities)
        max_prob_value = probabilities[max_prob_idx]
        
        # Color coding based on prediction results
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if i == max_prob_idx:
                # Highlight maximum probability point
                ax.scatter(timestamps[i], prob, c='gold', s=100, alpha=0.9, 
                          edgecolors='black', linewidth=2, zorder=5)
                # Add annotation for max probability
                ax.annotate(f'Max: {max_prob_value:.3f}', 
                           xy=(timestamps[i], prob), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            else:
                color = 'red' if pred == 1 else 'green'
                ax.scatter(timestamps[i], prob, c=color, s=50, alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Patient Probability')
        ax.set_title(f'Model Prediction Probability Over Time (Max: {max_prob_value:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        temp_path = "temp_timeseries.png"
        fig.savefig(temp_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return temp_path
    except Exception as e:
        print(f"Time series plot generation error: {e}")
        return None

def analyze_uploaded_data(df, model, shap_explainer, lime_explainer, selected_vars, scaler):
    """Perform comprehensive analysis on uploaded data"""
    if df is None or df.empty:
        return None, None, None, None, None
    
    # Remove patient_id column if exists
    if 'patient_id' in df.columns:
        df_analysis = df.drop(columns=['patient_id'])
    else:
        df_analysis = df.copy()
    
    # Align features to model
    df_analysis = align_features_to_model(df_analysis, model)
    
    # Select only numeric columns and convert strings to numbers
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns
    df_analysis = df_analysis[numeric_cols]
    
    # Convert string columns to numeric (convert to NaN on error)
    for col in df_analysis.columns:
        if df_analysis[col].dtype == 'object':
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
    
    # Remove rows with NaN values
    df_analysis = df_analysis.dropna()
    
    if df_analysis.empty:
        return None, None, None, None, None
    
    # Apply scaling to uploaded data
    df_analysis_scaled = pd.DataFrame(
        scaler.transform(df_analysis),
        columns=df_analysis.columns,
        index=df_analysis.index
    )
    print(f"Uploaded data scaled using fitted scaler")
    
    # Perform prediction for each row using scaled data
    predictions = []
    probabilities = []
    
    for idx, row in df_analysis_scaled.iterrows():
        x_vec = row.to_frame().T
        
        # Prediction
        proba = model.predict_proba(x_vec)[0]
        pred_label = int(proba[1] >= proba[0])
        
        predictions.append(pred_label)
        probabilities.append(float(proba[1]))
    
    # Calculate average probability
    avg_probability = np.mean(probabilities)
    avg_prediction = "Patient" if avg_probability > 0.5 else "Normal"
    
    # Generate time series plot
    timeseries_plot = create_time_series_plot(predictions, probabilities)
    
    # Find highest probability sample
    max_prob_idx = np.argmax(probabilities)
    max_prob_value = probabilities[max_prob_idx]
    max_prob_row = df_analysis_scaled.iloc[max_prob_idx:max_prob_idx+1]  # Use scaled data
    
    # SHAP/LIME analysis only for highest probability sample
    max_shap_plot = None
    max_lime_plot = None
    
    try:
        # SHAP analysis (exactly same as Individual Patient Analysis)
        x_vec = max_prob_row.copy()
        shap_values = shap_explainer.shap_values(x_vec, nsamples=100)
        max_shap_plot = create_shap_waterfall(shap_values, selected_vars)
    except Exception as e:
        print(f"SHAP analysis error: {e}")
        max_shap_plot = None
    
    try:
        # LIME analysis
        lime_exp = lime_explainer.explain_instance(
            max_prob_row.values[0],
            model.predict_proba,
            num_features=min(10, len(selected_vars)),
            num_samples=5000,
            top_labels=2,
        )
        max_lime_plot = create_lime_plot(lime_exp, label=1)
        

        if max_lime_plot and os.path.exists(max_lime_plot):
            print(f"LIME plot ready: {max_lime_plot}")
        else:
            print("LIME plot not ready")
            max_lime_plot = None
    except Exception as e:
        print(f"LIME analysis error: {e}")
        max_lime_plot = None
    
    results = {
        'avg_prediction': avg_prediction,
        'avg_probability': avg_probability,
        'max_probability': max_prob_value,
        'max_prob_idx': max_prob_idx,
        'total_samples': len(df),
        'patient_samples': sum(predictions),
        'normal_samples': len(predictions) - sum(predictions)
    }
    
    return results, timeseries_plot, max_shap_plot, max_lime_plot, max_prob_idx

def create_upload_analysis_display(results):
    """Display average analysis results for uploaded data"""
    if results is None:
        return "Please upload data and run analysis."
    
    if results['avg_prediction'] == 'Patient':
        prediction_text = "Patient"
        prediction_color = "#ff6b6b"
    else:
        prediction_text = "Normal"
        prediction_color = "#51cf66"
    
    # Determine max probability color
    max_prob_color = "#ffd700" if results['max_probability'] > 0.7 else "#ffa500" if results['max_probability'] > 0.5 else "#87ceeb"
    
    html_content = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; font-family: 'Arial', sans-serif;">
        <h2 style="text-align: center; margin-bottom: 20px; font-size: 20px;">🏥 Uploaded Data Analysis Results</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 20px;">
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0 0 8px 0; font-size: 14px;">Average Prediction</h3>
                <p style="font-size: 18px; font-weight: bold; margin: 0; color: {prediction_color};">{prediction_text}</p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0 0 8px 0; font-size: 14px;">Average Probability</h3>
                <p style="font-size: 18px; font-weight: bold; margin: 0; color: {prediction_color};">{results['avg_probability']*100:.1f}%</p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0 0 8px 0; font-size: 14px;">Max Probability</h3>
                <p style="font-size: 18px; font-weight: bold; margin: 0; color: {max_prob_color};">{results['max_probability']*100:.1f}%</p>
                <p style="font-size: 12px; margin: 5px 0 0 0; opacity: 0.8;">Sample #{results['max_prob_idx'] + 1}</p>
            </div>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px;">
            <h3 style="text-align: center; margin: 0 0 15px 0; font-size: 16px;">📊 Sample Statistics</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                <div style="text-align: center;">
                    <h4 style="margin: 0 0 8px 0; font-size: 12px;">Total Samples</h4>
                    <p style="font-size: 16px; font-weight: bold; margin: 0;">{results['total_samples']}</p>
                </div>
                <div style="text-align: center;">
                    <h4 style="margin: 0 0 8px 0; font-size: 12px;">Patient Samples</h4>
                    <p style="font-size: 16px; font-weight: bold; margin: 0; color: #ff6b6b;">{results['patient_samples']}</p>
                </div>
                <div style="text-align: center;">
                    <h4 style="margin: 0 0 8px 0; font-size: 12px;">Normal Samples</h4>
                    <p style="font-size: 16px; font-weight: bold; margin: 0; color: #51cf66;">{results['normal_samples']}</p>
                </div>
            </div>
        </div>
    </div>
    """
    return html_content

def create_prediction_display(results):
    if results is None or isinstance(results, str):
        return "Please select a patient."
    if results['prediction'] == '환자':
        prediction_text = "Patient"
        prediction_color = "#ff6b6b"
        confidence_color = "#ff8e8e"
    else:
        prediction_text = "Normal"
        prediction_color = "#51cf66"
        confidence_color = "#69db7c"
    confidence_level = ""
    if results['confidence'] >= 0.9:
        confidence_level = "Very High"
    elif results['confidence'] >= 0.8:
        confidence_level = "High"
    elif results['confidence'] >= 0.7:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    html_content = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; font-family: 'Arial', sans-serif;">
        <h2 style="text-align: center; margin-bottom: 20px; font-size: 20px;">🏥 Patient Kinematics Analysis Results</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0 0 8px 0; font-size: 14px;">Patient</h3>
                <p style="font-size: 18px; font-weight: bold; margin: 0;">{results['patient_key']}</p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0 0 8px 0; font-size: 14px;">Prediction</h3>
                <p style="font-size: 18px; font-weight: bold; margin: 0; color: {prediction_color};">{prediction_text}</p>
            </div>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px;">
            <h3 style="text-align: center; margin: 0 0 15px 0; font-size: 16px;">📊 Prediction Analysis</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div style="text-align: center;">
                    <h4 style="margin: 0 0 8px 0; font-size: 12px;">Probability (Patient)</h4>
                    <p style="font-size: 20px; font-weight: bold; margin: 0; color: {prediction_color};">{results['probability']*100:.1f}%</p>
                </div>
                <div style="text-align: center;">
                    <h4 style="margin: 0 0 8px 0; font-size: 12px;">Confidence Level</h4>
                    <p style="font-size: 16px; font-weight: bold; margin: 0; color: {confidence_color};">{confidence_level}</p>
                </div>
            </div>
        </div>
    </div>
    """
    return html_content


def analyze_patient(patient_choice, state):
    if not patient_choice:
        return "Please select a patient.", None, None

    model = state['model']
    X_train = state['X_train']
    X_test = state['X_test']
    y_test = state['y_test']
    patient_id_test = state['patient_id_test']
    shap_explainer = state['shap_explainer']
    lime_explainer = state['lime_explainer']
    selected_vars = state['selected_vars']

    if patient_id_test.isna().all() or (patient_id_test == '').all():
        try:
            idx = int(patient_choice.split('_')[-1])
        except Exception:
            idx = 0
        x_vec = X_test.iloc[idx:idx+1]
        patient_key = patient_choice
    else:
        mask = (patient_id_test == patient_choice)
        if not mask.any():
            return "Invalid patient id.", None, None
        x_vec = X_test.loc[mask].mean(numeric_only=True).to_frame().T
        patient_key = patient_choice

    proba = model.predict_proba(x_vec)[0]
    pred_label = int(proba[1] >= proba[0])

    shap_values = shap_explainer.shap_values(x_vec, nsamples=100)
    shap_plot = create_shap_waterfall(shap_values, selected_vars)

    # LIME
    lime_exp = lime_explainer.explain_instance(
        x_vec.values[0],
        model.predict_proba,
        num_features=min(10, len(selected_vars)),
        num_samples=5000,
        top_labels=2,
    )

    lime_plot = create_lime_plot(lime_exp, label=1)

    results = {
        'patient_key': patient_key,
        'prediction': '환자' if pred_label == 1 else '정상',
        'probability': float(proba[1]),
        'confidence': float(max(proba)),
    }

    prediction_display = create_prediction_display(results)
    return prediction_display, shap_plot, lime_plot


def analyze_selected_patient(patient_choice):
    return analyze_patient(patient_choice, APP_STATE)

def handle_file_upload(file, file_type):
    """Handle file upload"""
    if file is None:
        return None, "Please select a file.", None
    
    df, message = process_uploaded_file(file.name, file_type)
    if df is not None:
        # Show first 10 rows for preview
        preview_df = df.head(10)
        return df, message, preview_df
    else:
        return None, message, None

def run_upload_analysis(uploaded_data):
    """Run analysis on uploaded data"""
    if uploaded_data is None:
        return "Please upload data first.", None, None, None, None
    
    results, timeseries_plot, _, _, max_idx = analyze_uploaded_data(
        uploaded_data, 
        APP_STATE['model'], 
        APP_STATE['shap_explainer'], 
        APP_STATE['lime_explainer'], 
        APP_STATE['selected_vars'],
        APP_STATE['scaler']
    )
    
    analysis_display = create_upload_analysis_display(results)
    return analysis_display, timeseries_plot, None, None, f"Highest probability sample: {max_idx + 1}"

def run_shap_lime_analysis(uploaded_data):
    """Run SHAP/LIME analysis on uploaded data"""
    if uploaded_data is None:
        return None, None, "Please upload data first."
    
    try:
        # Remove patient_id column if exists
        if 'patient_id' in uploaded_data.columns:
            df_analysis = uploaded_data.drop(columns=['patient_id'])
        else:
            df_analysis = uploaded_data.copy()
        
        # Align features to model
        df_analysis = align_features_to_model(df_analysis, APP_STATE['model'])
        
        # Select only numeric columns and convert strings to numbers
        numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns
        df_analysis = df_analysis[numeric_cols]
        
        # Convert string columns to numeric (convert to NaN on error)
        for col in df_analysis.columns:
            if df_analysis[col].dtype == 'object':
                df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        
        # Remove rows with NaN values
        df_analysis = df_analysis.dropna()
        
        if df_analysis.empty:
            return None, None, "No valid data found for analysis."
        
        # Apply scaling to uploaded data
        df_analysis_scaled = pd.DataFrame(
            APP_STATE['scaler'].transform(df_analysis),
            columns=df_analysis.columns,
            index=df_analysis.index
        )
        
        # Perform prediction to find highest probability sample
        probabilities = []
        for idx, row in df_analysis_scaled.iterrows():
            x_vec = row.to_frame().T
            proba = APP_STATE['model'].predict_proba(x_vec)[0]
            probabilities.append(float(proba[1]))
        
        # Find highest probability sample
        max_prob_idx = np.argmax(probabilities)
        max_prob_row = df_analysis_scaled.iloc[max_prob_idx:max_prob_idx+1]
        
        # SHAP/LIME analysis
        max_shap_plot = None
        max_lime_plot = None
        
        try:
            # SHAP analysis
            x_vec = max_prob_row.copy()
            shap_values = APP_STATE['shap_explainer'].shap_values(x_vec, nsamples=100)
            max_shap_plot = create_shap_waterfall(shap_values, APP_STATE['selected_vars'])
            
            if max_shap_plot and os.path.exists(max_shap_plot):
                print(f"SHAP plot ready: {max_shap_plot}")
            else:
                print("SHAP plot not ready")
                max_shap_plot = None
        except Exception as e:
            print(f"SHAP analysis error: {e}")
            max_shap_plot = None
        
        try:
            # LIME analysis
            lime_exp = APP_STATE['lime_explainer'].explain_instance(
                max_prob_row.values[0],
                APP_STATE['model'].predict_proba,
                num_features=min(10, len(APP_STATE['selected_vars'])),
                num_samples=5000,
                top_labels=2,
            )
            max_lime_plot = create_lime_plot(lime_exp, label=1)
            
            if max_lime_plot and os.path.exists(max_lime_plot):
                print(f"LIME plot ready: {max_lime_plot}")
            else:
                print("LIME plot not ready")
                max_lime_plot = None
        except Exception as e:
            print(f"LIME analysis error: {e}")
            max_lime_plot = None
        
        status = f"Analysis completed. Sample {max_prob_idx + 1} (highest probability: {probabilities[max_prob_idx]:.3f})"
        return max_shap_plot, max_lime_plot, status
        
    except Exception as e:
        return None, None, f"Analysis error: {str(e)}"

APP_STATE = load_model_and_data(
    model_path=DEFAULT_MODEL_PATH,
    data_path=DEFAULT_DATA_PATH,
    max_vif=DEFAULT_MAX_VIF,
    seed=DEFAULT_SEED,
)
PATIENT_LABELS = APP_STATE['patient_labels']


# Interface configuration
with gr.Blocks(title="Patient Kinematics Analysis System (MLP)") as interface:
    gr.Markdown("# 🏥 Patient Kinematics Analysis System (MLP)")
    gr.Markdown("### MLP Model for Patient Diagnosis and Interpretation")
    
    with gr.Tabs():
        # Tab 1: How to Use
        with gr.Tab("📋 How to Use"):
            gr.Markdown("""
            ## 📋 How to Prepare Your Data
            
            This system analyzes movement data from CSV or MoT files. Here's how to get your data ready:
            
            ### 📱 Step 1: Record Movement (OpenCap)
            - Go to **[OpenCap.ai](https://www.opencap.ai/)**
            - Use your smartphone to record walking/movement videos
            - Follow their simple setup guide (2 phones at 30-45° angles)
            - Let OpenCap process your video into 3D movement data
            
            ### 🔧 Step 2: Process Data (OpenSim) 
            - Download **[OpenSim](https://simtk.org/frs/?group_id=91)**
            - Import your OpenCap data into OpenSim
            - Run "Inverse Kinematics" analysis
            - Export the results as CSV or MoT files
            
            ### 📤 Step 3: Upload Here
            - Upload your processed CSV/MoT files below
            - Get AI-powered movement analysis and diagnosis
            
            💡 **Tip**: Both OpenCap and OpenSim are free, open-source tools with detailed tutorials!
            
            ---
            
            ## ⚠️ Important Medical Notice
            
            **This system provides AI-powered movement analysis for informational purposes.**
            
            - The analysis results are **NOT** a substitute for professional medical diagnosis
            - If you have concerns about the model's diagnosis results, please **consult with a qualified healthcare professional**
            - This tool can be used by patients, but medical decisions should always be made with proper medical consultation
            - Always seek professional medical advice for any health-related decisions
            """)
        
        # Tab 2: File Upload
        with gr.Tab("📁 File Upload"):
            gr.Markdown("### Upload CSV or MoT files for analysis")
            
            with gr.Row():
                with gr.Column(scale=1):
                    file_type = gr.Radio(
                        choices=["csv", "mot"],
                        value="csv",
                        label="File Type"
                    )
                    file_upload = gr.File(
                        label="Upload File",
                        file_types=[".csv", ".txt", ".mot"]
                    )
                    upload_btn = gr.Button("📤 Upload File", variant="primary")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)
                
                with gr.Column(scale=2):
                    gr.Markdown("### File Preview")
                    file_preview = gr.Dataframe(label="Data Preview", interactive=False)
            
            uploaded_data = gr.State(value=None)
            
            upload_btn.click(
                fn=handle_file_upload,
                inputs=[file_upload, file_type],
                outputs=[uploaded_data, upload_status, file_preview]
            )
            
            file_upload.change(
                fn=lambda x: (None, "File selected. Please click Upload File button.", None) if x else (None, "", None),
                inputs=[file_upload],
                outputs=[uploaded_data, upload_status, file_preview]
            )

        # Tab 3: Average Analysis and Time Series
        with gr.Tab("📊 Analysis & Time Series"):
            gr.Markdown("### Average Analysis and Time Series Prediction")
            
            with gr.Row():
                with gr.Column(scale=1):
                    analyze_upload_btn = gr.Button("🔍 Run Analysis", variant="primary")
                    analysis_info = gr.Textbox(label="Analysis Info", interactive=False)
                
                with gr.Column(scale=2):
                    gr.Markdown("### Average Analysis Results")
                    avg_analysis_display = gr.HTML("Upload data and run analysis to see results.")
            
            with gr.Row():
                gr.Markdown("### Time Series Prediction Flow")
                timeseries_plot = gr.Image(label="Time Series Analysis")
            
            analyze_upload_btn.click(
                fn=run_upload_analysis,
                inputs=[uploaded_data],
                outputs=[avg_analysis_display, timeseries_plot, gr.State(), gr.State(), analysis_info]
            )

        # Tab 4: SHAP/LIME Analysis
        with gr.Tab("🔍 SHAP & LIME Analysis"):
            gr.Markdown("### SHAP and LIME Analysis for Highest Probability Time Frame")
            gr.Markdown("**Note**: Please run analysis in 'Analysis & Time Series' tab first, then click the button below.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    shap_lime_btn = gr.Button("🔍 Generate SHAP & LIME Analysis", variant="primary")
                    shap_lime_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=2):
                    gr.Markdown("### Analysis Results")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🌊 SHAP Waterfall Plot")
                    max_shap_plot = gr.Image(label="SHAP Analysis (Highest Probability)")
                with gr.Column():
                    gr.Markdown("### 🍋 LIME Local Explanation")
                    max_lime_plot = gr.Image(label="LIME Analysis (Highest Probability)")
            
            # SHAP/LIME analysis button
            shap_lime_btn.click(
                fn=run_shap_lime_analysis,
                inputs=[uploaded_data],
                outputs=[max_shap_plot, max_lime_plot, shap_lime_status]
            )

        # Tab 5: Individual Patient Analysis Tab (Reference)
        with gr.Tab("👤 Individual Patient Analysis"):
            gr.Markdown("### Individual Patient Analysis (Reference)")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 Patient Selection")
                    patient_dropdown = gr.Dropdown(
                        label="Select patient_id",
                        choices=PATIENT_LABELS,
                        value=PATIENT_LABELS[0] if len(PATIENT_LABELS) else None,
                    )
                    analyze_btn = gr.Button("🔍 Run Analysis", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Analysis Results")
                    result_text = gr.HTML("Please select a patient and run analysis.")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🌊 SHAP Waterfall Plot")
                    shap_plot = gr.Image(label="SHAP Analysis Results")
                with gr.Column():
                    gr.Markdown("### 🍋 LIME Local Explanation")
                    lime_plot = gr.Image(label="LIME Analysis Results")

            analyze_btn.click(
                fn=analyze_selected_patient,
                inputs=[patient_dropdown],
                outputs=[result_text, shap_plot, lime_plot]
            )

if __name__ == "__main__":
    interface.launch(share=True, server_name="0.0.0.0", server_port=7860)
