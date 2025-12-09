import pandas as pd
import random
import numpy as np
from typing import List, Optional, Dict, Any
import os
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')
class KinematicsLoader:
    def __init__(self, data_path: str = "content"):
        self.data_path = data_path
        self.patient_files = [
            'XXX_gait_1.mot', 'XXX_gait_2.mot', 'XXX_gait_3.mot', 'XXX_gait_4.mot',
            'XXX_gait_5.mot', 'XXX_gait_6.mot', 'XXX_gait_7.mot', 'XXX_gait_8.mot',
            'XXX_gait_9.mot', 'XXX_gait_10.mot'
        ]
        self.control_files = [
            'XXX_gait_1.mot', 'XXX_gait_2.mot', 'XXX_gait_3.mot', 'XXX_gait_4.mot',
            'XXX_gait_5.mot', 'XXX_gait_6.mot', 'XXX_gait_7.mot', 'XXX_gait_8.mot',
            'XXX_gait_9.mot', 'XXX_gait_10.mot'
        ]
        self.default_variables = [
            'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 
            'pelvis_ty', 'pelvis_tz', 'hip_flexion_r', 'hip_adduction_r', 
            'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 
            'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 
            'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l', 
            'lumbar_extension', 'lumbar_bending', 'lumbar_rotation', 'arm_flex_r', 
            'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 'pro_sup_r', 'arm_flex_l', 'arm_add_l', 
            'arm_rot_l', 'elbow_flex_l', 'pro_sup_l'
        ]
    def read_mot_file(self, file_path: str) -> pd.DataFrame:
        data = []
        header = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 10:
                    header = line.strip().split()
                else:
                    row = line.strip().split()
                    if len(row) == len(header):
                        data.append(row)
        df = pd.DataFrame(data, columns=header)
        return df.dropna(how='all')
    def calculate_vif(self, X: pd.DataFrame) -> Dict[str, float]:
        vif_data = {}
        feature_cols = [col for col in X.columns if col != 'patient_id']
        X_features = X[feature_cols]
        for i, column in enumerate(feature_cols):
            vif = variance_inflation_factor(X_features.values, i)
            vif_data[column] = vif
        return vif_data
    def select_variables_by_vif(self, X: pd.DataFrame, max_vif: float = 10.0) -> List[str]:
        print(f"Starting VIF-based variable selection (max VIF: {max_vif})")
        feature_cols = [col for col in X.columns if col != 'patient_id']
        print(f"Initial number of variables: {len(feature_cols)}")
        selected_vars = list(feature_cols)
        iteration = 1
        while True:
            X_current = X[selected_vars + ['patient_id']]
            vif_scores = self.calculate_vif(X_current)
            print(f"\nIteration {iteration} - VIF scores:")
            for var, vif in sorted(vif_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {var}: {vif:.3f}")
            max_vif_var = max(vif_scores.items(), key=lambda x: x[1])
            if max_vif_var[1] <= max_vif:
                print(f"\nAll variables have VIF <= {max_vif}.")
                print(f"Final number of selected variables: {len(selected_vars)}")
                break
            var_to_remove = max_vif_var[0]
            selected_vars.remove(var_to_remove)
            print(f"\nRemoving variable '{var_to_remove}' with highest VIF (VIF: {max_vif_var[1]:.3f})")
            print(f"Remaining variables: {len(selected_vars)}")
            iteration += 1
            if len(selected_vars) < 2:
                print("Warning: Too few variables for VIF calculation.")
                break
        return selected_vars
    def load_data(self, variables: Optional[List[str]] = None, random_seed: int = 40, use_vif_selection: bool = True, max_vif: float = 10.0) -> Dict[str, Any]:
        random.seed(random_seed)
        np.random.seed(random_seed)
        patient_dfs = []
        control_dfs = []
        for file_name in self.patient_files:
            file_path = os.path.join(self.data_path, file_name)
            if os.path.exists(file_path):
                df = self.read_mot_file(file_path)
                patient_dfs.append(df)
        for file_name in self.control_files:
            file_path = os.path.join(self.data_path, file_name)
            if os.path.exists(file_path):
                df = self.read_mot_file(file_path)
                control_dfs.append(df)
        random.shuffle(patient_dfs)
        random.shuffle(control_dfs)
        p_train = patient_dfs[:6]
        p_val = patient_dfs[6:8]
        p_test = patient_dfs[8:]
        c_train = control_dfs[:6]
        c_val = control_dfs[6:8]
        c_test = control_dfs[8:]
        for i, df in enumerate(p_train + p_val + p_test):
            df['patient'] = 1
            df['patient_id'] = f"patient_{i}"
        for i, df in enumerate(c_train + c_val + c_test):
            df['patient'] = 0
            df['patient_id'] = f"control_{i}"
        train_df = pd.concat(p_train + c_train, ignore_index=True)
        val_df = pd.concat(p_val + c_val, ignore_index=True)
        test_df = pd.concat(p_test + c_test, ignore_index=True)
        if variables is None:
            variables = self.default_variables
        available_vars = [var for var in variables if var in train_df.columns]
        X_train = train_df[available_vars].astype(float)
        y_train = train_df['patient'].astype(float)
        X_val = val_df[available_vars].astype(float)
        y_val = val_df['patient'].astype(float)
        X_test = test_df[available_vars].astype(float)
        y_test = test_df['patient'].astype(float)
        X_train['patient_id'] = train_df['patient_id']
        X_val['patient_id'] = val_df['patient_id']
        X_test['patient_id'] = test_df['patient_id']
        print(f"Initial number of variables: {len(available_vars)}")
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        if use_vif_selection:
            print("=" * 60)
            print("Starting VIF-based variable selection (based on train data)")
            print("=" * 60)
            selected_vars = self.select_variables_by_vif(X_train, max_vif)
            X_train = X_train[selected_vars + ['patient_id']]
            X_val = X_val[selected_vars + ['patient_id']]
            X_test = X_test[selected_vars + ['patient_id']]
            available_vars = selected_vars
            print(f"\nFinal selected variables:")
            for i, var in enumerate(selected_vars, 1):
                print(f"  {i:2d}. {var}")
            print("=" * 60)
        patient_id_train = X_train['patient_id']
        patient_id_val = X_val['patient_id']
        patient_id_test = X_test['patient_id']
        X_train_features = X_train.drop('patient_id', axis=1)
        X_val_features = X_val.drop('patient_id', axis=1)
        X_test_features = X_test.drop('patient_id', axis=1)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        X_val_scaled = scaler.transform(X_val_features)
        X_test_scaled = scaler.transform(X_test_features)
        X_train = pd.DataFrame(X_train_scaled, columns=available_vars, index=X_train.index)
        X_val = pd.DataFrame(X_val_scaled, columns=available_vars, index=X_val.index)
        X_test = pd.DataFrame(X_test_scaled, columns=available_vars, index=X_test.index)
        X_train['patient_id'] = patient_id_train
        X_val['patient_id'] = patient_id_val
        X_test['patient_id'] = patient_id_test
        print(f"Data normalization complete (using StandardScaler)")
        print(f"Final training data shape: {X_train.shape}")
        print(f"Final validation data shape: {X_val.shape}")
        print(f"Final test data shape: {X_test.shape}")
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'variables': available_vars,
            'vif_selected': use_vif_selection,
            'max_vif': max_vif if use_vif_selection else None,
            'scaler': scaler
        }
def load_kinematics_data(variables: Optional[List[str]] = None, use_vif_selection: bool = True, max_vif: float = 10.0, **kwargs) -> Dict[str, Any]:
    loader = KinematicsLoader()
    return loader.load_data(variables=variables, use_vif_selection=use_vif_selection, max_vif=max_vif, **kwargs)