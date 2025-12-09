# Patient-Kinematics-Analysis-System
Deep learning-based movement analysis system for patient diagnosis using biomechanical data.

## 📌 Overview

AI-powered system that analyzes human movement patterns from biomechanical data to assist in patient diagnosis. Uses state-of-the-art deep learning models (FT-Transformer, SAINT, TabNet, TabPFN) and traditional ML algorithms with SHAP/LIME explainability.

## 🗂️ Project Structure

```
rehabilitation-analysis/
├── simple_gradio_interface.py       # Web interface (main entry point)
├── run_complete_experiment.py       # Train all ML models
├── run_complete_experiment_no_vif.py # Train all ML models (no VIF)
├── run_deep_learning_experiment.py  # Train DL models only
├── run_deep_learning_experiment_no_vif.py # Train DL models (no VIF)
├── content/                         # Data directory (.mot files)
│   └── XXX_gait_*.mot              # Kinematics data files
├── data_loader/
│   └── kinematics_loader.py        # Data loading & preprocessing
├── experiments/
│   ├── nested_cv.py                # Patient-based nested CV
│   └── deep_models.py              # Deep learning training
└── models/
    ├── ft_transformer_wrapper.py   # FT-Transformer wrapper
    ├── saint_wrapper.py            # SAINT wrapper
    ├── tabnet_wrapper.py           # TabNet wrapper
    └── tabpfn_wrapper.py           # TabPFN wrapper
```

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

### 📤 Step 3: Upload & Analyze
- Place your processed files in the `content/` directory
- Or use the Gradio web interface to upload files
- Get AI-powered movement analysis and diagnosis

💡 **Tip**: Both OpenCap and OpenSim are free, open-source tools with detailed tutorials!

## 📊 Features

The system analyzes **30+ biomechanical features** including:

- **Pelvis**: tilt, list, rotation, translation (tx, ty, tz)
- **Hip**: flexion, adduction, rotation (left & right)
- **Knee**: angle (left & right)
- **Ankle**: angle, subtalar angle, mtp angle (left & right)
- **Lumbar**: extension, bending, rotation
- **Arm**: flexion, adduction, rotation (left & right)
- **Elbow**: flexion, pro_sup (left & right)

## 🔬 Methodology

### Patient-Based Cross-Validation
```
1. Split data by patient (not by samples)
   ↓
2. Nested CV (Outer: 5-fold, Inner: 3-fold)
   ↓
3. VIF-based feature selection (optional)
   ↓
4. Hyperparameter tuning (GridSearchCV)
   ↓
5. Train on full training set
   ↓
6. Evaluate on held-out test set
   ↓
7. SHAP & LIME explainability
```

## 🚀 Quick Start

### 1. Train Models

#### Train All Models (ML + DL)
```bash
# With VIF feature selection
python run_complete_experiment.py

# Without VIF
python run_complete_experiment_no_vif.py
```

#### Train Deep Learning Models Only
```bash
# With VIF feature selection
python run_deep_learning_experiment.py

# Without VIF
python run_deep_learning_experiment_no_vif.py
```

### 2. Launch Web Interface After Training

```bash
python simple_gradio_interface.py
```

Then open your browser at `http://localhost:7860`

