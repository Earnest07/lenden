import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import joblib # For saving the model

# --- Custom Feature Engineering Function for Location Model ---
def create_location_engineered_features(X):
    # It's crucial to work on a copy to avoid SettingWithCopyWarning
    X_engineered = X.copy()

    # --- Aggregated Financial Metrics (Relevant to Location Context) ---
    # These capture overall financial magnitudes which can vary significantly by location
    # Note: These column lists should be consistent with the features passed to this transformer
    # Ensure they are numeric before summing.
    
    balance_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and 'balance' in col and pd.api.types.is_numeric_dtype(X_engineered[col])]
    credit_limit_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and 'credit_limit' in col and pd.api.types.is_numeric_dtype(X_engineered[col])]
    loan_amt_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and 'loan_amt' in col and pd.api.types.is_numeric_dtype(X_engineered[col])]
    emi_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and 'total_emi' in col and pd.api.types.is_numeric_dtype(X_engineered[col])]
    repayment_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and 'repayment' in col and pd.api.types.is_numeric_dtype(X_engineered[col])]
    inquiry_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and ('inquires' in col or 'inquiries' in col) and pd.api.types.is_numeric_dtype(X_engineered[col])]
    total_loans_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and 'total_loans' in col and pd.api.types.is_numeric_dtype(X_engineered[col])]
    
    if balance_cols_in_X:
        X_engineered['total_balance'] = X_engineered[balance_cols_in_X].sum(axis=1, skipna=True)
    if credit_limit_cols_in_X:
        X_engineered['total_credit_limit'] = X_engineered[credit_limit_cols_in_X].sum(axis=1, skipna=True)
    if loan_amt_cols_in_X:
        X_engineered['total_loan_amount'] = X_engineered[loan_amt_cols_in_X].sum(axis=1, skipna=True)
    if emi_cols_in_X:
        X_engineered['total_emi_sum'] = X_engineered[emi_cols_in_X].sum(axis=1, skipna=True)
    if repayment_cols_in_X:
        X_engineered['total_repayment_sum'] = X_engineered[repayment_cols_in_X].sum(axis=1, skipna=True)
    if inquiry_cols_in_X:
        X_engineered['total_inquiries_count'] = X_engineered[inquiry_cols_in_X].sum(axis=1, skipna=True)
    if total_loans_cols_in_X:
        X_engineered['total_loans_count'] = X_engineered[total_loans_cols_in_X].sum(axis=1, skipna=True)
    
    # Example Ratio: Credit Utilization (handle division by zero)
    if 'total_balance' in X_engineered.columns and 'total_credit_limit' in X_engineered.columns:
        # Add a small epsilon to avoid division by zero if total_credit_limit can be zero
        epsilon = 1e-6 
        X_engineered['credit_utilization_ratio'] = X_engineered['total_balance'] / (X_engineered['total_credit_limit'] + epsilon)
        # Replace inf with NaN after division
        X_engineered['credit_utilization_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)


    # Handle any potential infinite values that might arise from divisions or other operations
    X_engineered.replace([np.inf, -np.inf], np.nan, inplace=True)

    return X_engineered


# --- 1. Load the actual dataset ---
# This script strictly relies on 'processed_dataset.csv' being present.
# If not found, a FileNotFoundError will be raised.
df = pd.read_csv('processed_dataset.csv')
print("Dataset 'processed_dataset.csv' loaded successfully.")
print("Original DataFrame head:")
print(df.head())
print("\nDataFrame Info (before type explicit conversion):")
df.info()

# --- Explicitly convert known categorical columns to string type ---
# This is crucial to ensure they are treated as categorical strings consistently.
categorical_cols_to_convert = [
    'gender', 'marital_status', 'city', 'state', 'residence_ownership',
    'pin', # Explicitly treating pin as categorical for location-based features
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_74', 'var_75'
]
for col in categorical_cols_to_convert:
    if col in df.columns:
        # Convert to string, then replace string 'nan' (if any) with actual np.nan
        df[col] = df[col].astype(str).replace('nan', np.nan)
print("\nDataFrame Info (after explicit type conversion for categorical columns):")
df.info()


# --- 2. Define Features for the Location Slave Model and Target ---
TARGET = 'target_income'
EXCLUDE_COLS = ['id', TARGET] # Exclude 'id' and the 'TARGET' itself from features

# Define a comprehensive list of features for the Location Slave Model.
# This includes core location, demographic, device, and all relevant financial 'var_' columns
# that are *not* explicitly classified as pure behavioral. Many financial aspects
# can be seen through a 'location' lens (e.g., average income/debt in a city).
location_features_candidate = [
    'age', # Demographic factor often related to location/life stage
    'gender', 'marital_status', 'residence_ownership', # Core demographic/housing
    'city', 'state', 'pin', # Core geographical
    'device_model', 'device_category', 'platform', 'device_manufacturer', # Device usage often has regional patterns

    'var_32',  # credit_score - Crucial financial indicator, can vary by location

    # All balance related 'var_' columns (financial health linked to location)
    'var_0', 'var_1', 'var_4', 'var_8', 'var_18', 'var_19', 'var_21', 'var_30',
    'var_34', 'var_35', 'var_38', 'var_59', 'var_68',

    # All credit limit related 'var_' columns (access to credit can be regional)
    'var_2', 'var_3', 'var_5', 'var_10', 'var_11', 'var_12', 'var_22', 'var_23',
    'var_26', 'var_27', 'var_28', 'var_29', 'var_33', 'var_44', 'var_47',

    # All loan amount related 'var_' columns (cost of living, loan availability by region)
    'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31', 'var_36',
    'var_39', 'var_42', 'var_65', 'var_72',

    # All EMI related 'var_' columns (debt burden, cost of living)
    'var_9', 'var_17', 'var_41', 'var_43', 'var_46', 'var_51', 'var_56',

    # All repayment related 'var_' columns (financial behavior linked to location)
    'var_37', 'var_48', 'var_49', 'var_50', 'var_52', 'var_55', 'var_67',
    'var_69', 'var_70', 'var_73',

    # All inquiry related 'var_' columns (financial activity, regional lending patterns)
    'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71',

    # All total_loans and related activity 'var_' columns (overall financial engagement)
    'var_40',  # closed_loan
    'var_53', 'var_54', 'var_57', 'var_60', 'var_62', 'var_63', 'var_64', 'var_66',
]

# Filter features to ensure they exist in the DataFrame and are not excluded
location_features = [col for col in location_features_candidate if col in df.columns and col not in EXCLUDE_COLS]

# --- 3. Split Data into Training and Testing Sets ---
# Drop rows where TARGET is NaN if any exist (critical for training a regression model)
df_clean = df.dropna(subset=[TARGET]).copy()

# X will include all the features that will be passed to the custom transformer first
X_location = df_clean[location_features].copy()
y = df_clean[TARGET]

X_train_location, X_test_location, y_train, y_test = train_test_split(
    X_location, y, test_size=0.2, random_state=42
)

# --- 4. Identify Numerical and Categorical Features AFTER custom transformations ---
# Apply custom feature engineering on a small sample to get the actual column names
# and their types after feature engineering. This ensures ColumnTransformer is accurate.
sample_df_for_type_detection = create_location_engineered_features(X_train_location.head())

location_numerical_features = [col for col in sample_df_for_type_detection.columns
                                  if pd.api.types.is_numeric_dtype(sample_df_for_type_detection[col])]
location_categorical_features = [col for col in sample_df_for_type_detection.columns
                                    if pd.api.types.is_object_dtype(sample_df_for_type_detection[col]) or pd.api.types.is_string_dtype(sample_df_for_type_detection[col])]

print(f"\nLocation Numerical Features (after engineering): {location_numerical_features}")
print(f"Location Categorical Features (after engineering): {location_categorical_features}")


# --- 5. Preprocessing Pipeline for Location Model ---

# Numerical pipeline: Impute with median, then scale
location_numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # Median for numerical NaNs
    ('scaler', StandardScaler())
])

# Categorical pipeline: Impute with mode, then One-Hot Encode
location_categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), # Most frequent for categorical NaNs
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # handle_unknown='ignore' for unseen categories
])

# Combine pipelines using ColumnTransformer for location features
location_preprocessor = ColumnTransformer(
    transformers=[
        ('num', location_numerical_pipeline, location_numerical_features),
        ('cat', location_categorical_pipeline, location_categorical_features)
    ],
    remainder='drop', # Drop any columns not explicitly handled
    verbose_feature_names_out=False # Suppress warning about feature names in LightGBM
)

# --- 6. Build and Train the Location Slave Model Pipeline ---
location_model_pipeline = Pipeline([
    ('feature_engineer', FunctionTransformer(create_location_engineered_features, validate=False)), # Custom FE step
    ('preprocessor', location_preprocessor),
    ('regressor', lgb.LGBMRegressor(random_state=42, n_estimators=1500, learning_rate=0.03)) # Increased estimators, reduced LR
])

print("\nTraining the Location Slave Model...")
location_model_pipeline.fit(X_train_location, y_train)
print("Location Slave Model training complete.")

# --- ADDED LINE: Save the trained model ---
joblib.dump(location_model_pipeline, 'location_model_pipeline.pkl')
print("\nLocation model saved to 'location_model_pipeline.pkl'")


# --- 7. Make Predictions with the Location Slave Model ---
y_pred_location = location_model_pipeline.predict(X_test_location)

# Ensure predictions are non-negative if income cannot be negative
y_pred_location[y_pred_location < 0] = 0

# --- 8. Evaluate the Location Slave Model ---
mae_location = mean_absolute_error(y_test, y_pred_location)
r2_location = r2_score(y_test, y_pred_location)

print(f"\nLocation Slave Model Evaluation on Test Set:")
print(f"Mean Absolute Error (MAE) for Location Model: ${mae_location:,.2f}")
print(f"R-squared (R2) Score for Location Model: {r2_location:.4f}")

# --- 9. Generate 'location_income' column for the original DataFrame ---
print("\nGenerating 'location_income' predictions for the entire dataset...")
# Predict location income for the entire original dataset
# Pass a copy of the features from the original df to avoid issues with alignment
df['location_income'] = location_model_pipeline.predict(df[location_features].copy())

# Ensure location_income is non-negative
df.loc[df['location_income'] < 0, 'location_income'] = 0

print("\nDataFrame with 'location_income' column:")
print(df[['id', 'target_income', 'location_income']].head())
print(f"Number of rows with 'location_income' predictions: {df['location_income'].count()}")


# --- 10. Save the updated DataFrame and the location_income.csv ---

# Save the entire DataFrame with the new 'location_income' column
output_full_filename = 'processed_dataset_with_location_income.csv'
df.to_csv(output_full_filename, index=False)
print(f"\nUpdated dataset saved to '{output_full_filename}'")

# Create a new DataFrame with only 'id' and 'location_income' and save it
location_income_df = df[['id', 'location_income']].copy()
output_location_filename = 'location_income.csv'
location_income_df.to_csv(output_location_filename, index=False)
print(f"Location income predictions saved to '{output_location_filename}'")

# --- 11. Example Prediction for a New Data Point using Location Model ---
# This part is just for demonstration of how to use the trained model for new data.
# The new data point needs to include ALL features that the model expects,
# including all the 'var_' columns that are part of 'location_features_candidate'.
new_data_point_location = pd.DataFrame([{
    'age': 35, 'gender': 'Male', 'marital_status': 'Married', 'residence_ownership': 'Owned',
    'city': 'Mumbai', 'state': 'MH', 'pin': '400001', # Pin as string
    'device_model': 'iPhone 12', 'device_category': 'Smartphone', 'platform': 'iOS', 'device_manufacturer': 'Apple',
    'var_32': 750,

    'var_0': 1000, 'var_1': 500, 'var_4': 2000, 'var_8': 1000, 'var_18': 300, 'var_19': 600, 'var_21': 900,
    'var_30': 1500, 'var_34': 1800, 'var_35': 2200, 'var_38': 2500, 'var_59': 1000, 'var_68': 3000,

    'var_2': 10000, 'var_3': 15000, 'var_5': 20000, 'var_10': 9000, 'var_11': 8000, 'var_12': 12000,
    'var_22': 50000, 'var_23': 60000, 'var_26': 70000, 'var_27': 80000, 'var_28': 90000, 'var_29': 100000,
    'var_33': 110000, 'var_44': 120000, 'var_47': 130000,

    'var_6': 5000, 'var_7': 7000, 'var_13': 20000, 'var_14': 15000, 'var_20': 10000, 'var_24': 8000,
    'var_31': 12000, 'var_36': 15000, 'var_39': 18000, 'var_42': 22000, 'var_65': 25000, 'var_72': 28000,

    'var_9': 500, 'var_17': 700, 'var_41': 800, 'var_43': 900, 'var_46': 600, 'var_51': 400, 'var_56': 300,

    'var_37': 400, 'var_48': 500, 'var_49': 600, 'var_50': 700, 'var_52': 800, 'var_55': 900,
    'var_67': 1000, 'var_69': 1100, 'var_70': 1200, 'var_73': 1300,

    'var_15': 1, 'var_16': 0, 'var_25': 0, 'var_45': 2, 'var_58': 1, 'var_61': 0, 'var_71': 0,

    'var_40': 5, 'var_53': 10, 'var_54': 3, 'var_57': 12, 'var_60': 11, 'var_62': 2,
    'var_63': 15, 'var_64': 18, 'var_66': 20
}])

# Explicitly convert categorical columns in the new_data_point_location to string type
for col in categorical_cols_to_convert: # Use the comprehensive list for robustness
    if col in new_data_point_location.columns:
        new_data_point_location[col] = new_data_point_location[col].astype(str)

# Ensure the new data point contains all and only the features used for training this model.
new_data_point_location_for_prediction = new_data_point_location[location_features].copy()

predicted_location_income = location_model_pipeline.predict(new_data_point_location_for_prediction)[0]
print(f"\nPredicted 'location_income' for a new data point: ${predicted_location_income:,.2f}")

# --- End of Location Slave Model Code ---
