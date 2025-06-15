import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# --- Custom Feature Extraction Function for Master Model ---
def create_master_meta_features(X):
    # Ensure X is a DataFrame for easy column access and to avoid SettingWithCopyWarning
    X_engineered = X.copy()

    # Columns expected to be present from slave model predictions
    slave_income_cols = [
        'behavioral_income',
        'demographic_income',
        'location_income',
        'device_income'
    ]

    # Filter for only those slave income columns that are actually in X
    existing_slave_income_cols = [col for col in slave_income_cols if col in X_engineered.columns]

    if existing_slave_income_cols:
        # Calculate average of slave income predictions
        X_engineered['avg_slave_income'] = X_engineered[existing_slave_income_cols].mean(axis=1)
        
        # Calculate standard deviation of slave income predictions (as a measure of discrepancy)
        # Handle cases where there might be only one or zero slave income columns,
        # or if all values for a row are NaN, which would result in NaN for std.
        # Fill these with 0 or a sensible default if it makes sense (e.g., if one prediction is enough for std to be 0).
        if len(existing_slave_income_cols) > 1:
            X_engineered['std_slave_income'] = X_engineered[existing_slave_income_cols].std(axis=1).fillna(0)
        else:
            X_engineered['std_slave_income'] = 0 # If only one or zero, std dev is 0 or undefined

    return X_engineered


# --- 1. Load the original dataset and all slave model predictions ---
try:
    df_original = pd.read_csv('processed_dataset.csv')
    print("Original dataset 'processed_dataset.csv' loaded successfully.")
    print("Original DataFrame head:")
    print(df_original.head())
    print("\nOriginal DataFrame Info:")
    df_original.info()

    # --- Load slave model predictions ---
    df_behavioral = pd.read_csv('behavioral_income.csv')
    df_demographic = pd.read_csv('demographic_income.csv')
    df_location = pd.read_csv('location_income.csv')
    df_device = pd.read_csv('device_income.csv')
    print("\nSlave model predictions loaded successfully.")

    # --- Explicitly convert categorical columns in original df to string type ---
    categorical_cols_to_convert = [
        'gender', 'marital_status', 'city', 'state', 'residence_ownership',
        'pin',
        'device_model', 'device_category', 'platform', 'device_manufacturer',
        'var_74', 'var_75'
    ]
    for col in categorical_cols_to_convert:
        if col in df_original.columns:
            # BUG FIX: Changed 'df[col]' to 'df_original[col]'
            df_original[col] = df_original[col].astype(str).replace('nan', np.nan)
    print("\nOriginal DataFrame Info (after explicit type conversion for categorical columns):")
    df_original.info()

except FileNotFoundError as e:
    print(f"Error: Required file not found - {e}")
    print("Please ensure 'processed_dataset.csv', 'behavioral_income.csv', 'demographic_income.csv', 'location_income.csv', and 'device_income.csv' are in the same directory.")
    exit()


# --- 2. Merge all slave model predictions into the original DataFrame ---
df_master = df_original.copy()

df_master = pd.merge(df_master, df_behavioral[['id', 'behavioral_income']], on='id', how='left')
df_master = pd.merge(df_master, df_demographic[['id', 'demographic_income']], on='id', how='left')
df_master = pd.merge(df_master, df_location[['id', 'location_income']], on='id', how='left')
df_master = pd.merge(df_master, df_device[['id', 'device_income']], on='id', how='left')

print("\nDataFrame after merging slave predictions (head):")
print(df_master[['id', 'target_income', 'behavioral_income', 'demographic_income', 'location_income', 'device_income']].head())
print(f"Total rows in master DataFrame: {len(df_master)}")


# --- 3. Define Features for the Master Model and Target ---
TARGET = 'target_income'
EXCLUDE_COLS = ['id', TARGET]

# Master model features: predictions from slave models + original key features + NEW meta-features
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

# Filter master features to ensure they exist in the DataFrame
master_features = [col for col in master_features_candidate if col in df_master.columns and col not in EXCLUDE_COLS]


# --- 4. Split Data into Training and Testing Sets for Cross-Validation ---
df_clean_master = df_master.dropna(subset=[TARGET] + master_features).copy() # Use new master_features list

X_master = df_clean_master[master_features].copy()
y_master = df_clean_master[TARGET]

print(f"\nMaster Data shape for CV: {X_master.shape}")

# --- NEW: Create a dummy DataFrame to correctly identify numerical/categorical features after meta-feature engineering ---
# This step is crucial because master_numerical_features and master_categorical_features
# need to include 'avg_slave_income' and 'std_slave_income' before the ColumnTransformer is defined.
# The FunctionTransformer is the first step in the pipeline, so the preprocessor will see these.
# Using .head(5) for efficiency, assuming the first few rows are representative of column types.
X_master_temp_engineered = create_master_meta_features(X_master.head(5).copy())

master_numerical_features = [col for col in master_features_candidate if col in X_master_temp_engineered.columns and pd.api.types.is_numeric_dtype(X_master_temp_engineered[col])]
master_categorical_features = [col for col in master_features_candidate if col in X_master_temp_engineered.columns and (pd.api.types.is_object_dtype(X_master_temp_engineered[col]) or pd.api.types.is_string_dtype(X_master_temp_engineered[col]))]


print(f"\nMaster Numerical Features (after meta-feature engineering): {master_numerical_features}")
print(f"Master Categorical Features (after meta-feature engineering): {master_categorical_features}")


# --- 5. Preprocessing Pipeline for Master Model ---

# Numerical pipeline: Impute with median, then scale
master_numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: Impute with mode, then One-Hot Encode (if any categorical features remain in master_features)
master_categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine pipelines using ColumnTransformer for master features
# The FunctionTransformer is added here to create meta-features BEFORE preprocessing
master_preprocessor = ColumnTransformer(
    transformers=[
        ('num', master_numerical_pipeline, master_numerical_features), # Use the updated list
        ('cat', master_categorical_pipeline, master_categorical_features) # Use the updated list
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

# --- 6. Build and Train the Master Model Pipeline with Aggressive Hyperparameter Tuning & Cross-Validation ---

lgbm_regressor = lgb.LGBMRegressor(
    random_state=42,
    n_estimators=5000,
    learning_rate=0.003,
    num_leaves=32,
    max_depth=-1,
    min_child_samples=80,
    reg_alpha=0.3,
    reg_lambda=0.3,
    colsample_bytree=0.6,
    subsample=0.6,
    subsample_freq=1,
)

master_model_pipeline = Pipeline([
    ('feature_engineer_master', FunctionTransformer(create_master_meta_features, validate=False)), # NEW: Meta-feature creation
    ('preprocessor', master_preprocessor),
    ('regressor', lgbm_regressor)
])

print("\nTraining the Master Model with Cross-Validation...")

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_maes = []
fold_r2s = []
oof_preds = np.zeros(len(X_master))

for fold, (train_index, test_index) in enumerate(kf.split(X_master, y_master)):
    print(f"\n--- Fold {fold+1}/{n_splits} ---")
    X_train_fold, X_test_fold = X_master.iloc[train_index], X_master.iloc[test_index]
    y_train_fold, y_test_fold = y_master.iloc[train_index], y_master.iloc[test_index]

    master_model_pipeline.fit(X_train_fold, y_train_fold)
    y_pred_fold = master_model_pipeline.predict(X_test_fold)

    y_pred_fold[y_pred_fold < 0] = 0

    mae_fold = mean_absolute_error(y_test_fold, y_pred_fold)
    r2_fold = r2_score(y_test_fold, y_pred_fold)

    fold_maes.append(mae_fold)
    fold_r2s.append(r2_fold)
    oof_preds[test_index] = y_pred_fold

    print(f"Fold {fold+1} MAE: ${mae_fold:,.2f}")
    print(f"Fold {fold+1} R2: {r2_fold:.4f}")

print("\n--- Cross-Validation Complete ---")
print(f"Average MAE across {n_splits} folds: ${np.mean(fold_maes):,.2f} +/- ${np.std(fold_maes):,.2f}")
print(f"Average R2 across {n_splits} folds: {np.mean(fold_r2s):.4f} +/- {np.std(fold_r2s):.4f}")

print("\nRetraining Master Model on full dataset for final use...")
master_model_pipeline.fit(X_master, y_master)
print("Master Model retraining complete on full dataset.")

joblib.dump(master_model_pipeline, 'master_model_pipeline.pkl')
print("\nMaster model saved to 'master_model_pipeline.pkl'")


# --- 7. Make Predictions with the Re-trained Master Model (on full dataset) ---
y_pred_full_dataset = master_model_pipeline.predict(df_master[master_features].copy())

y_pred_full_dataset[y_pred_full_dataset < 0] = 0

# --- 8. Evaluation on the FULL training dataset (for reference, CV is primary metric) ---
mae_full = mean_absolute_error(df_master.dropna(subset=[TARGET] + master_features)[TARGET], y_pred_full_dataset)
r2_full = r2_score(df_master.dropna(subset=[TARGET] + master_features)[TARGET], y_pred_full_dataset)

print(f"\nMaster Model Evaluation on Full Training Data (for reference, CV is primary metric):")
print(f"Mean Absolute Error (MAE) for Master Model: ${mae_full:,.2f}")
print(f"R-squared (R2) Score for Master Model: {r2_full:.4f}")

# --- 9. Generate 'final_predicted_income' column for the entire dataset ---
print("\nGenerating 'final_predicted_income' predictions for the entire dataset...")
df_master['final_predicted_income'] = master_model_pipeline.predict(df_master[master_features].copy())

df_master.loc[df_master['final_predicted_income'] < 0, 'final_predicted_income'] = 0

print("\nDataFrame with 'final_predicted_income' column:")
print(df_master[['id', 'target_income', 'behavioral_income', 'demographic_income', 'location_income', 'device_income', 'final_predicted_income']].head())
print(f"Number of rows with 'final_predicted_income' predictions: {df_master['final_predicted_income'].count()}")


# --- 10. Save the final DataFrame ---
output_final_filename = 'processed_dataset_with_all_income_predictions.csv'
df_master.to_csv(output_final_filename, index=False)
print(f"\nFinal dataset saved to '{output_final_filename}'")
