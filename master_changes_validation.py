import pandas as pd
import numpy as np
import joblib # Import joblib for loading models
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error # Import additional evaluation metric
from sklearn.base import BaseEstimator, TransformerMixin # For FunctionTransformer


# --- Custom Feature Extraction Function for Master Model (MUST BE IDENTICAL TO TRAINING SCRIPT) ---
def create_master_meta_features(X):
    X_engineered = X.copy()

    slave_income_cols = [
        'behavioral_income',
        'demographic_income',
        'location_income',
        'device_income'
    ]
    existing_slave_income_cols = [col for col in slave_income_cols if col in X_engineered.columns]

    if existing_slave_income_cols:
        X_engineered['avg_slave_income'] = X_engineered[existing_slave_income_cols].mean(axis=1)
        if len(existing_slave_income_cols) > 1:
            X_engineered['std_slave_income'] = X_engineered[existing_slave_income_cols].std(axis=1).fillna(0)
        else:
            X_engineered['std_slave_income'] = 0 # If only one or zero, std dev is 0 or undefined

    return X_engineered


# --- Define Consistent Feature and Target Lists (MUST BE IDENTICAL TO TRAINING SCRIPT) ---
TARGET = 'target_income'
EXCLUDE_COLS = ['id', TARGET]

categorical_cols_to_convert = [
    'gender', 'marital_status', 'city', 'state', 'residence_ownership',
    'pin',
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_74', 'var_75'
]

# --- EXPANDED FEATURES FOR MASTER MODEL - MUST MATCH master_changes.py ---
master_features_candidate = [
    'behavioral_income',
    'demographic_income',
    'location_income',
    'device_income',
    'age',
    'var_32',
    'financial_health_score',
    'total_balance',
    'avg_credit_util',
    'loan_to_income_1',
    'loan_to_income_2',
    'avg_slave_income',   # New meta-feature
    'std_slave_income',   # New meta-feature
]


# --- 1. Load the trained Master model ---
try:
    master_model_pipeline = joblib.load('master_model_pipeline.pkl')
    print("Master model loaded successfully from 'master_model_pipeline.pkl'.")
except FileNotFoundError:
    print("Error: 'master_model_pipeline.pkl' not found.")
    print("Please ensure you have run 'master_changes.py' first to train and save the model.")
    exit()

# --- 2. Load the base validation dataset and slave model predictions for validation ---
try:
    df_validation_base = pd.read_csv('processed_dataset_400.csv')
    print("\nBase validation dataset 'processed_dataset_400.csv' loaded successfully.")

    df_val_behavioral = pd.read_csv('processed_dataset_400_with_behavioral_income.csv')
    df_val_demographic = pd.read_csv('processed_dataset_400_with_demographic_income.csv')
    df_val_location = pd.read_csv('processed_dataset_400_with_location_income.csv')
    df_val_device = pd.read_csv('processed_dataset_400_with_device_income.csv')
    print("Slave model predictions for validation set loaded successfully.")

    for col in categorical_cols_to_convert:
        if col in df_validation_base.columns:
            df_validation_base[col] = df_validation_base[col].astype(str).replace('nan', np.nan)
    
except FileNotFoundError as e:
    print(f"Error: Required validation file not found - {e}")
    print("Please ensure all 'processed_dataset_400_with_*.csv' files are present (run slave model validations first).")
    exit()

# --- Validate and clean IDs before merging to prevent row duplication ---
if df_validation_base['id'].duplicated().any():
    print("Warning: Duplicate 'id' found in 'processed_dataset_400.csv'. This may affect merging integrity.")

def clean_slave_df(slave_df, name):
    if slave_df['id'].duplicated().any():
        print(f"Warning: Duplicate 'id' found in {name}. Keeping first occurrence for each ID during merge.")
        return slave_df.drop_duplicates(subset=['id'], keep='first').copy()
    return slave_df.copy()

df_val_behavioral = clean_slave_df(df_val_behavioral, 'processed_dataset_400_with_behavioral_income.csv')
df_val_demographic = clean_slave_df(df_val_demographic, 'processed_dataset_400_with_demographic_income.csv')
df_val_location = clean_slave_df(df_val_location, 'processed_dataset_400_with_location_income.csv')
df_val_device = clean_slave_df(df_val_device, 'processed_dataset_400_with_device_income.csv')


# --- 3. Merge slave model predictions into the validation DataFrame ---
df_validation_master = df_validation_base.copy()

df_validation_master = pd.merge(df_validation_master, df_val_behavioral[['id', 'behavioral_income']], on='id', how='left')
df_validation_master = pd.merge(df_validation_master, df_val_demographic[['id', 'demographic_income']], on='id', how='left')
df_validation_master = pd.merge(df_validation_master, df_val_location[['id', 'location_income']], on='id', how='left')
df_validation_master = pd.merge(df_validation_master, df_val_device[['id', 'device_income']], on='id', how='left')

print("\nValidation DataFrame after merging slave predictions (head):")
print(df_validation_master[['id', 'target_income', 'behavioral_income', 'demographic_income', 'location_income', 'device_income']].head())
print(f"Total rows in validation master DataFrame (after merge attempt): {len(df_validation_master)}")


# --- 4. Prepare features and target for prediction and evaluation ---
# This part is crucial: apply the same meta-feature engineering as in the training script
df_validation_master_engineered = create_master_meta_features(df_validation_master.copy()) # Apply meta-feature engineering here

master_features_for_prediction = [col for col in master_features_candidate if col in df_validation_master_engineered.columns and col not in EXCLUDE_COLS]

# Drop rows where TARGET or any of the master features are NaN from the validation set
df_clean_validation_master = df_validation_master_engineered.dropna(subset=[TARGET] + master_features_for_prediction).copy()

print(f"Total rows in clean validation master DataFrame (after NaN drop): {len(df_clean_validation_master)}")


X_validation_master = df_clean_validation_master[master_features_for_prediction].copy()
y_validation_true_master = df_clean_validation_master[TARGET].copy()


# --- 5. Generate 'final_predicted_income' predictions for the validation dataset ---
print("\nGenerating 'final_predicted_income' predictions for the validation dataset...")
y_validation_pred_master = master_model_pipeline.predict(X_validation_master)

y_validation_pred_master[y_validation_pred_master < 0] = 0
df_clean_validation_master['final_predicted_income'] = y_validation_pred_master


# --- 6. Evaluate the Master Model on the validation set ---
print("\nMaster Model Evaluation on Validation Set:")
validation_mae_master = mean_absolute_error(y_validation_true_master, y_validation_pred_master)
validation_r2_master = r2_score(y_validation_true_master, y_validation_pred_master)
validation_mse_master = mean_squared_error(y_validation_true_master, y_validation_pred_master)
validation_rmse_master = np.sqrt(validation_mse_master)

print(f"Mean Absolute Error (MAE) on Validation Set: ${validation_mae_master:,.2f}")
print(f"R-squared (R2) Score on Validation Set: {validation_r2_master:.4f}")
print(f"Mean Squared Error (MSE) on Validation Set: ${validation_mse_master:,.2f}")
print(f"Root Mean Squared Error (RMSE) on Validation Set: ${validation_rmse_master:,.2f}")


# --- 7. Save the updated validation dataset ---
print("\nValidation DataFrame with 'final_predicted_income' column:")
print(df_clean_validation_master[['id', 'target_income', 'behavioral_income', 'demographic_income', 'location_income', 'device_income', 'final_predicted_income']].head())
print(f"Number of rows with 'final_predicted_income' predictions: {df_clean_validation_master['final_predicted_income'].count()}")

output_validation_filename_master = 'processed_dataset_400_with_all_final_predictions.csv'
df_clean_validation_master.to_csv(output_validation_filename_master, index=False)
print(f"\nUpdated validation dataset saved to '{output_validation_filename_master}'")

print("\nMaster Model validation process complete.")
