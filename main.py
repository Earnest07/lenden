# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import shap # Import SHAP for explainability
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb


# --- Custom TargetEncoder Transformer (MUST BE IDENTICAL TO TRAINING SCRIPTS) ---
# This class needs to be defined here so joblib can correctly load the pipelines
# which might have instances of this custom transformer.
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, smoothing=1.0):
        self.cols = cols
        self.smoothing = smoothing
        self.mapping = {}
        self.global_mean = None
        self.target_map = {}

    def fit(self, X, y):
        if y is None:
            raise ValueError("TargetEncoder requires 'y' during fit for target mean calculation.")
        
        if y.dtype == 'object' or y.dtype == 'string':
            unique_labels = sorted(y.unique())
            self.target_map = {label: i for i, label in enumerate(unique_labels)}
            y_numeric = y.map(self.target_map)
        else:
            y_numeric = y

        self.global_mean = y_numeric.mean()

        if not isinstance(X, pd.DataFrame):
            if self.cols and len(self.cols) == X.shape[1]:
                X_df = pd.DataFrame(X, columns=self.cols)
            else:
                X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        for col in self.cols:
            if col not in X_df.columns:
                print(f"Warning: Column '{col}' not found in X during TargetEncoder fit. Skipping.")
                continue

            means = y_numeric.groupby(X_df[col]).mean()
            counts = y_numeric.groupby(X_df[col]).count()
            smoothed_means = (means * counts + self.global_mean * self.smoothing) / (counts + self.smoothing)
            self.mapping[col] = smoothed_means.to_dict()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            if self.cols and len(self.cols) == X.shape[1]:
                X_transformed = pd.DataFrame(X, columns=self.cols)
            else:
                X_transformed = pd.DataFrame(X)
        else:
            X_transformed = X.copy()

        for col in self.cols:
            if col in self.mapping:
                X_transformed[col] = X_transformed[col].map(self.mapping[col]).fillna(self.global_mean)
            else:
                X_transformed[col] = np.full(len(X_transformed), self.global_mean)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return self.cols


# --- Define Creditworthiness Thresholds (MUST BE IDENTICAL TO TRAINING SCRIPT) ---
def assign_creditworthiness(score):
    if score >= 0.5:
        return 'Good'
    elif -0.5 <= score < 0.5:
        return 'Average'
    else:
        return 'Poor'

# --- Custom Feature Engineering Function for Demographic Model (for internal CS model) ---
def create_demographic_engineered_features_internal(X):
    X_engineered = X.copy()
    if 'age' in X_engineered.columns:
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
        X_engineered['age_bin'] = pd.cut(
            X_engineered['age'],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True
        ).astype(object)
    X_engineered.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X_engineered

# --- Custom Feature Engineering Function for Location Model (for internal CS model) ---
def create_location_engineered_features_internal(X):
    X_engineered = X.copy()
    balance_cols = [f'var_{i}' for i in [0, 1, 4, 8, 18, 19, 21, 30, 34, 35, 38, 59, 68]]
    credit_limit_cols = [f'var_{i}' for i in [2, 3, 5, 10, 11, 12, 22, 23, 26, 27, 28, 29, 33, 44, 47]]
    
    # Filter for columns that actually exist in X_engineered to avoid KeyErrors
    existing_balance_cols = [col for col in balance_cols if col in X_engineered.columns and pd.api.types.is_numeric_dtype(X_engineered[col])]
    existing_credit_limit_cols = [col for col in credit_limit_cols if col in X_engineered.columns and pd.api.types.is_numeric_dtype(X_engineered[col])]

    if existing_balance_cols:
        X_engineered['total_balance'] = X_engineered[existing_balance_cols].sum(axis=1, skipna=True)
    if existing_credit_limit_cols:
        X_engineered['total_credit_limit'] = X_engineered[existing_credit_limit_cols].sum(axis=1, skipna=True)

    if 'total_balance' in X_engineered.columns and 'total_credit_limit' in X_engineered.columns:
        epsilon = 1e-6 
        X_engineered['credit_utilization_ratio'] = X_engineered['total_balance'] / (X_engineered['total_credit_limit'] + epsilon)
        X_engineered['credit_utilization_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    X_engineered.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X_engineered

# --- Define Selected `var_XX` Features for Simplified Input (Subset of 0-75) ---
# These are the var_XX features directly used in engineered features or explicitly in slave models
# and will be included in the Pydantic model and frontend form.
selected_var_cols = [
    'var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6', 'var_7', 'var_8',
    'var_9', 'var_10', 'var_11', 'var_12', 'var_13', 'var_14', 'var_15', 'var_16',
    'var_17', 'var_18', 'var_19', 'var_20', 'var_21', 'var_22', 'var_23', 'var_24', 'var_25',
    'var_26', 'var_27', 'var_28', 'var_29', 'var_30', 'var_31', 'var_33', 'var_34', 'var_35',
    'var_36', 'var_37', 'var_38', 'var_39', 'var_40', 'var_41', 'var_42', 'var_43', 'var_44',
    'var_45', 'var_46', 'var_47', 'var_48', 'var_49', 'var_50', 'var_51', 'var_52', 'var_53',
    'var_54', 'var_55', 'var_56', 'var_57', 'var_58', 'var_59', 'var_60', 'var_61', 'var_62',
    'var_63', 'var_64', 'var_65', 'var_66', 'var_67', 'var_68', 'var_69', 'var_70', 'var_71',
    'var_72', 'var_73', 'var_74', 'var_75' # var_74, var_75 are categorical
]


# All original categorical columns used across models (for type conversion)
categorical_cols_to_convert_all = [
    'gender', 'marital_status', 'city', 'state', 'residence_ownership',
    'pin', 'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_74', 'var_75'
]

# --- Master Model Specific Feature Engineering Function ---
# This function must be exactly as it was when the master model was trained,
# to ensure consistent feature engineering.
def create_master_meta_features(X):
    X_engineered = X.copy()
    slave_income_cols = [
        'behavioral_income', 'demographic_income', 'location_income', 'device_income'
    ]
    existing_slave_income_cols = [col for col in slave_income_cols if col in X_engineered.columns]

    if existing_slave_income_cols:
        X_engineered['avg_slave_income'] = X_engineered[existing_slave_income_cols].mean(axis=1)
        if len(existing_slave_income_cols) > 1:
            X_engineered['std_slave_income'] = X_engineered[existing_slave_income_cols].std(axis=1).fillna(0)
        else:
            X_engineered['std_slave_income'] = 0 # If only one or zero, std dev is 0 or undefined
    return X_engineered


# --- Define API App ---
app = FastAPI()

# Global variables to store loaded models and SHAP explainer
models: Dict[str, Any] = {}
credit_score_model: Optional[Pipeline] = None
master_explainer: Optional[shap.TreeExplainer] = None
master_transformed_feature_names: List[str] = []

@app.on_event("startup")
async def load_and_train_models():
    global models, credit_score_model, master_explainer, master_transformed_feature_names
    try:
        # Load the raw dataset for internal credit score model training
        try:
            df = pd.read_csv('processed_dataset.csv')
            print("Dataset 'processed_dataset.csv' loaded for internal credit score model training.")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="processed_dataset.csv not found. Required for internal credit score model training.")

        # Ensure consistent type conversion for all categorical columns in the dataset
        for col in categorical_cols_to_convert_all:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', np.nan)
        
        # --- Internal Credit Score Prediction Model Training ---
        # Features for the credit score model: all features *except* var_32 (the target)
        credit_score_features_candidate = [col for col in df.columns if col not in ['id', 'target_income', 'var_32']]
        credit_score_features = [col for col in credit_score_features_candidate if col in df.columns]

        X_cs = df[credit_score_features].copy()
        y_cs = df['var_32'].copy() # var_32 is the credit score target

        # Drop rows where var_32 is NaN for credit score model training
        valid_cs_indices = y_cs.dropna().index
        X_cs = X_cs.loc[valid_cs_indices]
        y_cs = y_cs.loc[valid_cs_indices]

        # Identify numerical and categorical features for credit score model
        # Re-apply transformations to a sample to ensure feature names are correct after `FunctionTransformer`
        X_cs_demog_engineered = create_demographic_engineered_features_internal(X_cs.head())
        X_cs_loc_engineered = create_location_engineered_features_internal(X_cs.head())
        
        # Merge results from both
        X_cs_combined_engineered = X_cs_demog_engineered.merge(X_cs_loc_engineered.drop(columns=X_cs_demog_engineered.columns.intersection(X_cs_loc_engineered.columns)), left_index=True, right_index=True, how='left')
        
        # Correctly identify numerical and categorical features for credit score model preprocessing
        credit_score_numerical_features = [col for col in X_cs_combined_engineered.columns if pd.api.types.is_numeric_dtype(X_cs_combined_engineered[col]) and col != 'var_32']
        credit_score_categorical_features = [col for col in X_cs_combined_engineered.columns if (pd.api.types.is_object_dtype(X_cs_combined_engineered[col]) or pd.api.types.is_string_dtype(X_cs_combined_engineered[col])) and col != 'var_32']

        # Credit Score Model Preprocessor
        cs_numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cs_categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        credit_score_preprocessor = ColumnTransformer(
            transformers=[
                ('demog_fe', FunctionTransformer(create_demographic_engineered_features_internal, validate=False), [col for col in X_cs.columns if col in ['age'] + [c for c in categorical_cols_to_convert_all if c in X_cs.columns]]),
                ('loc_fe', FunctionTransformer(create_location_engineered_features_internal, validate=False), [col for col in X_cs.columns if col in ['var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_8', 'var_10', 'var_11', 'var_12', 'var_18', 'var_19', 'var_21', 'var_22', 'var_23', 'var_26', 'var_27', 'var_28', 'var_29', 'var_30', 'var_33', 'var_34', 'var_35', 'var_38', 'var_44', 'var_47', 'var_59', 'var_68', 'city', 'state', 'pin']]), # Pass relevant cols
                ('num', cs_numerical_pipeline, credit_score_numerical_features),
                ('cat', cs_categorical_pipeline, credit_score_categorical_features)
            ],
            remainder='passthrough', # Keep other columns in the pipeline for feature inspection
            verbose_feature_names_out=False
        )

        credit_score_model = Pipeline([
            ('preprocessor', credit_score_preprocessor),
            ('regressor', lgb.LGBMRegressor(random_state=42)) # Simple LGBM for credit score
        ])
        print("Training internal Credit Score model...")
        credit_score_model.fit(X_cs, y_cs)
        print("Internal Credit Score model trained successfully!")

        # --- Load other models ---
        models['slave_models'] = joblib.load('slave_models_dict.pkl')
        models['master_model'] = joblib.load('master_model_pipeline.pkl')
        models['creditworthiness_model'] = joblib.load('creditworthiness_model_pipeline.pkl')
        print("Hierarchical and Creditworthiness models loaded successfully!")

        # Initialize SHAP Explainer for Master Model
        master_regressor_model = models['master_model'].named_steps['regressor']
        master_preprocessor_pipeline_step = models['master_model'].named_steps['preprocessor']
        master_feature_engineer_step = models['master_model'].named_steps['feature_engineer_master']

        master_explainer = shap.TreeExplainer(master_regressor_model)

        # To get feature names after preprocessing for SHAP, need a representative sample.
        # This dummy data needs to be comprehensive enough for all pipeline steps.
        dummy_master_input = pd.DataFrame([{
            # Slave income predictions (numerical)
            'behavioral_income': 0.0, 'demographic_income': 0.0,
            'location_income': 0.0, 'device_income': 0.0,
            # Original numerical features
            'age': 30.0, 'var_32': 700.0, # Credit score
            'financial_health_score': 0.0, 'total_balance': 0.0,
            'avg_credit_util': 0.0, 'loan_to_income_1': 0.0, 'loan_to_income_2': 0.0,
            # Other var_XX used in slave models or engineering, now reduced set
            **{f: 0.0 for f in selected_var_cols if f.startswith('var_') and not f in ['var_74', 'var_75', 'var_32']},
            # Categorical features
            'gender': 'Male', 'marital_status': 'Single', 'residence_ownership': 'Rented',
            'city': 'Mumbai', 'state': 'Maharashtra', 'pin': '400001',
            'device_model': 'iPhone', 'device_category': 'Smartphone',
            'platform': 'iOS', 'device_manufacturer': 'Apple',
            'var_74': 'No negative comments', 'var_75': 'Type A'
        }])
        
        # Ensure categorical types are consistent
        for col in categorical_cols_to_convert_all:
            if col in dummy_master_input.columns:
                dummy_master_input[col] = dummy_master_input[col].astype(str).replace('nan', np.nan)

        # Apply the master feature engineering step first (meta-features)
        dummy_master_input_engineered = master_feature_engineer_step.transform(dummy_master_input)
        
        # Now apply the preprocessor to get the final feature names
        dummy_transformed_output = master_preprocessor_pipeline_step.transform(dummy_master_input_engineered)
        master_transformed_feature_names = master_preprocessor_pipeline_step.get_feature_names_out(dummy_master_input_engineered.columns)

        print("SHAP Explainer for Master Model initialized.")

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Required model or data file not found: {e}. Please ensure all .pkl files and processed_dataset.csv are in the same directory.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model loading/training/SHAP initialization: {e}")

# --- Pydantic Input Data Model for FastAPI (Simplified `var_XX` inputs) ---
class UserInput(BaseModel):
    age: float
    gender: str
    marital_status: str
    city: str
    state: str
    pin: str
    residence_ownership: str
    device_model: str
    device_category: str
    platform: str
    device_manufacturer: str
    var_32: Optional[float] = None # Credit Score is now optional

    # Only include selected var_XX features
    var_0: Optional[float] = None
    var_1: Optional[float] = None
    var_2: Optional[float] = None
    var_3: Optional[float] = None
    var_4: Optional[float] = None
    var_5: Optional[float] = None
    var_6: Optional[float] = None
    var_7: Optional[float] = None
    var_8: Optional[float] = None
    var_9: Optional[float] = None
    var_10: Optional[float] = None
    var_11: Optional[float] = None
    var_12: Optional[float] = None
    var_13: Optional[float] = None
    var_14: Optional[float] = None
    var_15: Optional[float] = None
    var_16: Optional[float] = None
    var_17: Optional[float] = None
    var_18: Optional[float] = None
    var_19: Optional[float] = None
    var_20: Optional[float] = None
    var_21: Optional[float] = None
    var_22: Optional[float] = None
    var_23: Optional[float] = None
    var_24: Optional[float] = None
    var_25: Optional[float] = None
    var_26: Optional[float] = None
    var_27: Optional[float] = None
    var_28: Optional[float] = None
    var_29: Optional[float] = None
    var_30: Optional[float] = None
    var_31: Optional[float] = None
    var_33: Optional[float] = None
    var_34: Optional[float] = None
    var_35: Optional[float] = None
    var_36: Optional[float] = None
    var_37: Optional[float] = None
    var_38: Optional[float] = None
    var_39: Optional[float] = None
    var_40: Optional[float] = None
    var_41: Optional[float] = None
    var_42: Optional[float] = None
    var_43: Optional[float] = None
    var_44: Optional[float] = None
    var_45: Optional[float] = None
    var_46: Optional[float] = None
    var_47: Optional[float] = None
    var_48: Optional[float] = None
    var_49: Optional[float] = None
    var_50: Optional[float] = None
    var_51: Optional[float] = None
    var_52: Optional[float] = None
    var_53: Optional[float] = None
    var_54: Optional[float] = None
    var_55: Optional[float] = None
    var_56: Optional[float] = None
    var_57: Optional[float] = None
    var_58: Optional[float] = None
    var_59: Optional[float] = None
    var_60: Optional[float] = None
    var_61: Optional[float] = None
    var_62: Optional[float] = None
    var_63: Optional[float] = None
    var_64: Optional[float] = None
    var_65: Optional[float] = None
    var_66: Optional[float] = None
    var_67: Optional[float] = None
    var_68: Optional[float] = None
    var_69: Optional[float] = None
    var_70: Optional[float] = None
    var_71: Optional[float] = None
    var_72: Optional[float] = None
    var_73: Optional[float] = None
    var_74: Optional[str] = None # Categorical
    var_75: Optional[str] = None # Categorical


# --- Prediction Endpoint ---
@app.post("/predict_lending_metrics")
async def predict_lending_metrics(user_input: UserInput):
    try:
        # Convert input Pydantic model to Pandas DataFrame
        # Missing optional fields will be NaN in the DataFrame
        input_dict = user_input.dict()
        input_df = pd.DataFrame([input_dict])

        # --- Handle Credit Score (var_32) ---
        predicted_cs_value = None
        if user_input.var_32 is None or user_input.var_32 == 0: # If credit score is not provided
            if credit_score_model is None:
                 raise HTTPException(status_code=500, detail="Credit score model not initialized. Cannot predict missing credit score.")
            
            # Prepare input for credit score model
            credit_score_input_df = input_df.drop(columns=['var_32', 'target_income'], errors='ignore').copy()
            # Ensure categorical types are consistent before passing to CS model
            for col in categorical_cols_to_convert_all:
                if col in credit_score_input_df.columns:
                    credit_score_input_df[col] = credit_score_input_df[col].astype(str).replace('nan', np.nan)
            
            predicted_cs_value = credit_score_model.predict(credit_score_input_df)[0]
            # Ensure predicted credit score is within a reasonable range (e.g., 300-900)
            predicted_cs_value = max(300, min(900, predicted_cs_value))
            input_df['var_32'] = predicted_cs_value # Use predicted value for subsequent models
            print(f"Credit score (var_32) was missing, predicted: {predicted_cs_value:.2f}")
        else:
            predicted_cs_value = user_input.var_32
            input_df['var_32'] = predicted_cs_value # Ensure it's in the DataFrame for consistency
            print(f"Credit score (var_32) provided: {predicted_cs_value:.2f}")
        
        # --- Feature Engineering (MUST BE IDENTICAL TO TRAINING SCRIPTS) ---
        # Fill missing selected_var_cols with 0.0 or a default for calculations
        # Assuming that pipelines handle NaNs from not-provided fields via imputers.
        for col in selected_var_cols:
            if col not in input_df.columns:
                input_df[col] = np.nan # Or 0.0, depending on expected behavior for missing inputs
        
        # Financial Health Score
        input_df['financial_health_score'] = input_df['var_0'].fillna(0) - input_df['var_1'].fillna(0)
        
        # Total Balance
        balance_cols_for_calc = [f'var_{i}' for i in [0, 1, 4, 8, 18, 19, 21, 30, 34, 35, 38, 59, 68]]
        existing_balance_cols = [col for col in balance_cols_for_calc if col in input_df.columns]
        input_df['total_balance'] = input_df[existing_balance_cols].fillna(0).sum(axis=1)

        # Avg Credit Utilization
        credit_limit_cols_for_calc = [f'var_{i}' for i in [2, 3, 5, 10, 11, 12, 22, 23, 26, 27, 28, 29, 33, 44, 47]]
        existing_credit_limit_cols = [col for col in credit_limit_cols_for_calc if col in input_df.columns]
        total_credit_limit_sum = input_df[existing_credit_limit_cols].fillna(0).sum(axis=1)
        input_df['avg_credit_util'] = np.where(
            total_credit_limit_sum != 0,
            input_df['total_balance'] / total_credit_limit_sum,
            0.0
        )

        # Loan to Income Ratios
        input_df['loan_to_income_1'] = np.where(input_df['var_0'].fillna(0) != 0, input_df['var_6'].fillna(0) / input_df['var_0'].fillna(0), 0.0)
        input_df['loan_to_income_2'] = np.where(input_df['var_1'].fillna(0) != 0, input_df['var_7'].fillna(0) / input_df['var_1'].fillna(0), 0.0)

        input_df = input_df.replace([np.inf, -np.inf], np.nan) # Replace infinities

        # Ensure categorical columns have correct dtype as string for consistency
        for col in categorical_cols_to_convert_all:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str).replace('nan', np.nan)


        # --- Step 2: Predict with Slave Models ---
        slave_predictions = {}
        # Features for each slave model (MUST BE IDENTICAL to slave model training scripts)
        # These now rely on the 'selected_var_cols' subset for var_XX
        SLAVE_MODEL_FEATURES_UPDATED = {
            # Behavioral model receives all selected_var_cols + engineered + var_32
            'behavioral': [col for col in selected_var_cols + ['var_32'] + ['financial_health_score', 'total_balance', 'avg_credit_util', 'loan_to_income_1', 'loan_to_income_2']],
            'demographic': ['age', 'gender', 'marital_status', 'residence_ownership', 'var_32', 'financial_health_score'],
            'location': ['city', 'state', 'pin', 'var_32', 'financial_health_score', 'total_balance', 'avg_credit_util', 'loan_to_income_1', 'loan_to_income_2'],
            'device': ['device_model', 'device_category', 'platform', 'device_manufacturer', 'var_32', 'financial_health_score', 'total_balance', 'avg_credit_util', 'loan_to_income_1', 'loan_to_income_2'],
        }

        for model_name, pipeline in models['slave_models'].items():
            features_for_this_slave = SLAVE_MODEL_FEATURES_UPDATED.get(model_name, [])
            
            # Filter input_df to only include features relevant for this slave model
            # For robustness, add missing features as NaN before passing to pipeline
            slave_input_df = pd.DataFrame(columns=features_for_this_slave) # Create df with all expected columns
            for col in features_for_this_slave:
                if col in input_df.columns:
                    slave_input_df[col] = input_df[col]
                else:
                    slave_input_df[col] = np.nan # Fill missing with NaN, pipeline's imputer will handle

            try:
                # Ensure correct dtypes for categorical columns in slave_input_df
                for col in categorical_cols_to_convert_all:
                    if col in slave_input_df.columns:
                        slave_input_df[col] = slave_input_df[col].astype(str).replace('nan', np.nan)

                predicted_income = pipeline.predict(slave_input_df)[0]
                slave_predictions[f'{model_name}_income'] = predicted_income if predicted_income >= 0 else 0
            except Exception as e:
                print(f"Error predicting with {model_name} model: {e}. Using NaN for its prediction.")
                slave_predictions[f'{model_name}_income'] = np.nan

        for key, value in slave_predictions.items():
            input_df[key] = value

        # --- Step 3: Prepare Input for Master Model (including its own feature engineering) ---
        MASTER_MODEL_INPUT_FEATURES_UPDATED = [
            'behavioral_income', 'demographic_income', 'location_income', 'device_income',
            'age', 'var_32', 'financial_health_score', 'total_balance',
            'avg_credit_util', 'loan_to_income_1', 'loan_to_income_2',
            'avg_slave_income', 'std_slave_income',
        ]
        
        # X_master_predict will only contain features required by the master model's pipeline
        X_master_predict = input_df[
            [col for col in MASTER_MODEL_INPUT_FEATURES_UPDATED if col in input_df.columns]
        ].copy()

        # Ensure all expected features for master model's first step are present, fill missing with NaN
        for col in MASTER_MODEL_INPUT_FEATURES_UPDATED:
            if col not in X_master_predict.columns:
                X_master_predict[col] = np.nan


        # --- Step 4: Predict Annual Income with Master Model ---
        final_predicted_income = models['master_model'].predict(X_master_predict)[0]
        final_predicted_income = max(0, final_predicted_income)

        income_range_low = final_predicted_income * 0.90
        income_range_high = final_predicted_income * 1.10

        # --- Step 5: Predict Creditworthiness with Creditworthiness Model ---
        CREDITWORTHINESS_MODEL_INPUT_FEATURES_UPDATED = [
            'final_predicted_income',
            'financial_health_score', 'total_balance', 'avg_credit_util',
            'loan_to_income_1', 'loan_to_income_2', 'age',
            'gender', 'marital_status', 'residence_ownership',
            'city', 'state', 'pin',
            'device_model', 'device_category', 'platform', 'device_manufacturer',
            'var_74', 'var_75',
            'var_32' # Now var_32 (credit score) is always available in input_df
        ]

        creditworthiness_input_df = input_df[
            [col for col in CREDITWORTHINESS_MODEL_INPUT_FEATURES_UPDATED if col in input_df.columns]
        ].copy()
        
        for col in CREDITWORTHINESS_MODEL_INPUT_FEATURES_UPDATED:
            if col not in creditworthiness_input_df.columns:
                creditworthiness_input_df[col] = np.nan

        predicted_creditworthiness_label = models['creditworthiness_model'].predict(creditworthiness_input_df)[0]

        # --- Step 6: Calculate Influence Scores (SHAP Values for Master Model) ---
        influence_scores_output = {}
        top_master_model_feature_influences = {}

        if master_explainer and master_transformed_feature_names:
            master_preprocessor_step = models['master_model'].named_steps['preprocessor']
            master_feature_engineer_step = models['master_model'].named_steps['feature_engineer_master']
            
            X_master_for_shap_engineered = master_feature_engineer_step.transform(X_master_predict)
            X_single_sample_transformed = master_preprocessor_step.transform(X_master_for_shap_engineered)
            
            shap_values = master_explainer.shap_values(X_single_sample_transformed)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_dict = dict(zip(master_transformed_feature_names, shap_values))

            # --- Aggregate SHAP values by conceptual influence domains ---
            demographic_shap = (
                shap_dict.get('demographic_income', 0.0) +
                shap_dict.get('num__age', 0.0) +
                shap_dict.get('onehot_cat__gender_Male', 0.0) + shap_dict.get('onehot_cat__gender_Female', 0.0) + shap_dict.get('onehot_cat__gender_Other', 0.0) + shap_dict.get('onehot_cat__gender_Prefer not to say', 0.0) +
                shap_dict.get('onehot_cat__marital_status_Married', 0.0) + shap_dict.get('onehot_cat__marital_status_Single', 0.0) + shap_dict.get('onehot_cat__marital_status_Divorced', 0.0) + shap_dict.get('onehot_cat__marital_status_Widowed', 0.0) + shap_dict.get('onehot_cat__marital_status_Other', 0.0) +
                shap_dict.get('onehot_cat__residence_ownership_Owned', 0.0) + shap_dict.get('onehot_cat__residence_ownership_Rented', 0.0) + shap_dict.get('onehot_cat__residence_ownership_Family', 0.0) + shap_dict.get('onehot_cat__residence_ownership_Other', 0.0)
            )
            
            location_shap = (
                shap_dict.get('location_income', 0.0) +
                shap_dict.get('city', 0.0) +
                shap_dict.get('state', 0.0) +
                shap_dict.get('pin', 0.0)
            )

            behavioral_shap = (
                shap_dict.get('behavioral_income', 0.0) +
                shap_dict.get('num__var_32', 0.0) +
                shap_dict.get('num__financial_health_score', 0.0) +
                shap_dict.get('num__total_balance', 0.0) +
                shap_dict.get('num__avg_credit_util', 0.0) +
                shap_dict.get('num__loan_to_income_1', 0.0) +
                shap_dict.get('num__loan_to_income_2', 0.0) +
                shap_dict.get('var_74', 0.0) +
                shap_dict.get('var_75', 0.0)
            )

            digital_shap = (
                shap_dict.get('device_income', 0.0) +
                shap_dict.get('device_model', 0.0) +
                shap_dict.get('device_category', 0.0) +
                shap_dict.get('platform', 0.0) +
                shap_dict.get('device_manufacturer', 0.0)
            )

            avg_slave_income_shap = shap_dict.get('num__avg_slave_income', 0.0)
            std_slave_income_shap = shap_dict.get('num__std_slave_income', 0.0)

            influence_scores_output = {
                "Demographic Influence (on Income)": demographic_shap,
                "Location Influence (on Income)": location_shap,
                "Behavioral Influence (on Income)": behavioral_shap,
                "Digital Footprint Influence (on Income)": digital_shap,
                "Overall Slave Average Income Influence": avg_slave_income_shap,
                "Overall Slave Std Dev Income Influence": std_slave_income_shap,
            }

            sorted_shap_features = sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)
            top_master_model_feature_influences = {name: value for name, value in sorted_shap_features[:5]}

        else:
            influence_scores_output = {"Error": "SHAP explainer not initialized. Cannot provide influence details."}
            top_master_model_feature_influences = {"Error": "SHAP not available."}


        # --- Step 7: Return Results ---
        return {
            "predicted_annual_income": round(final_predicted_income, 2),
            "annual_income_range": {
                "low": round(income_range_low, 2),
                "high": round(income_range_high, 2)
            },
            "predicted_creditworthiness": predicted_creditworthiness_label,
            "predicted_credit_score": round(predicted_cs_value, 2), # New: return the credit score used
            "income_influence_details": influence_scores_output,
            "top_master_model_feature_influences": top_master_model_feature_influences,
            "debug_info_slave_predictions": {k: round(v, 2) if isinstance(v, (int, float)) else str(v) for k, v in slave_predictions.items()}
        }

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

