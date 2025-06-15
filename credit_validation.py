import pandas as pd
import numpy as np
import joblib # Import joblib for loading models
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.base import BaseEstimator, TransformerMixin # For Custom TargetEncoder

# --- Custom TargetEncoder for categorical features (MUST BE IDENTICAL TO TRAINING SCRIPT) ---
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, smoothing=1.0):
        self.cols = cols
        self.smoothing = smoothing
        self.mapping = {}
        self.global_mean = None

    def fit(self, X, y):
        # This fit method is called during pipeline.fit(X_train, y_train).
        # For validation, this fit method is not directly called; the preprocessor
        # within the loaded pipeline will use its already fitted state.
        # This implementation is included for completeness and consistency with the training script.
        if y is None:
            raise ValueError("TargetEncoder requires 'y' during fit for target mean calculation.")
        
        if y.dtype == 'object' or y.dtype == 'string':
            self.target_map = {
                'Poor': 0,
                'Average': 1,
                'Good': 2
            }
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


# --- Define Consistent Feature and Target Lists (MUST BE IDENTICAL TO TRAINING SCRIPT - creditworthiness.py) ---
TARGET = 'creditworthiness_label'
CREDIT_SCORE_COL = 'var_32'

# Define creditworthiness categories (MUST BE IDENTICAL TO TRAINING SCRIPT)
def assign_creditworthiness(score):
    if score >= 0.5:
        return 'Good'
    elif -0.5 <= score < 0.5:
        return 'Average'
    else:
        return 'Poor'

EXCLUDE_COLS = ['id', 'target_income', CREDIT_SCORE_COL, TARGET]

# AGGRESSIVELY REDUCED FEATURES (MUST BE IDENTICAL TO TRAINING SCRIPT)
creditworthiness_features_candidate = [
    'final_predicted_income',
    'financial_health_score',
    'total_balance',
    'avg_credit_util',
    'loan_to_income_1',
    'loan_to_income_2',
    'age',
    'gender',
    'marital_status',
    'residence_ownership',
    'city',
    'state',
    'pin',
    'device_model',
    'device_category',
    'platform',
    'device_manufacturer',
    'var_74',
    'var_75',
]

# Split categorical features for different encoding strategies (MUST BE IDENTICAL TO TRAINING SCRIPT)
high_cardinality_for_target_encoding = [
    col for col in ['city', 'state', 'pin', 'device_model', 'device_category', 'platform', 'device_manufacturer', 'var_74', 'var_75'] if col in creditworthiness_features_candidate
]
low_cardinality_for_onehot_encoding = [
    col for col in ['gender', 'marital_status', 'residence_ownership'] if col in creditworthiness_features_candidate
]


# --- 1. Load the trained Creditworthiness model ---
try:
    creditworthiness_model_pipeline = joblib.load('creditworthiness_model_pipeline.pkl')
    print("Creditworthiness model loaded successfully from 'creditworthiness_model_pipeline.pkl'.")
except FileNotFoundError:
    print("Error: 'creditworthiness_model_pipeline.pkl' not found.")
    print("Please ensure you have run 'creditworthiness.py' first to train and save the model.")
    exit()

# --- 2. Load the validation datasets ---
try:
    df_validation = pd.read_csv('processed_dataset_400.csv')
    print("\nBase validation dataset 'processed_dataset_400.csv' loaded successfully.")

    # Load the predictions from the Master Model for the validation dataset
    df_val_master_predictions = pd.read_csv('processed_dataset_400_with_all_final_predictions.csv')
    print("Master model predictions for validation set loaded successfully.")

    # Merge final_predicted_income into the base validation DataFrame
    df_validation = pd.merge(df_validation, df_val_master_predictions[['id', 'final_predicted_income']], on='id', how='left')
    print("\nValidation DataFrame after merging final income predictions (head):")
    print(df_validation[['id', 'var_32', 'target_income', 'final_predicted_income']].head())

    # --- Explicitly convert categorical columns to string type (MUST BE IDENTICAL TO TRAINING SCRIPT) ---
    categorical_cols_to_convert_all = [ # All original categorical columns that were converted
        'gender', 'marital_status', 'city', 'state', 'residence_ownership',
        'pin',
        'device_model', 'device_category', 'platform', 'device_manufacturer',
        'var_74', 'var_75'
    ]
    for col in categorical_cols_to_convert_all:
        if col in df_validation.columns:
            df_validation[col] = df_validation[col].astype(str).replace('nan', np.nan)
    print("\nValidation DataFrame Info (after explicit type conversion for categorical columns):")
    df_validation.info()

except FileNotFoundError as e:
    print(f"Error: Required validation file not found - {e}")
    print("Please ensure 'processed_dataset_400.csv' and 'processed_dataset_400_with_all_final_predictions.csv' are present.")
    exit()


# --- 3. Prepare the Target Variable for Validation ---
# CRITICAL FIX: Assign creditworthiness_label to the main df_validation DataFrame FIRST
df_validation[TARGET] = df_validation[CREDIT_SCORE_COL].apply(assign_creditworthiness)

# Now, create df_validation_clean for prediction/evaluation by dropping NaNs in relevant columns
df_validation_clean = df_validation.dropna(subset=[CREDIT_SCORE_COL]).copy()


print(f"\nCreditworthiness Distribution on Validation Set:\n{df_validation_clean[TARGET].value_counts()}")

# Check for single class in validation target
if len(df_validation_clean[TARGET].unique()) < 2:
    print("\nERROR: Only one class found in validation target variable after mapping.")
    print("This indicates an issue with the validation data's 'var_32' distribution or the thresholds.")
    exit()


# --- 4. Prepare Features for Prediction ---
# Filter features based on the candidate list used in training
creditworthiness_features = [col for col in creditworthiness_features_candidate if col in df_validation_clean.columns and col not in EXCLUDE_COLS]

X_validation = df_validation_clean[creditworthiness_features].copy()
y_validation_true = df_validation_clean[TARGET].copy() # Get the true target values for evaluation

print(f"\nCreditworthiness Validation data shape: {X_validation.shape}")
print(f"Validation Target Distribution:\n{y_validation_true.value_counts(normalize=True)}")


# --- 5. Make Predictions on the Validation Set ---
print("\nGenerating 'predicted_creditworthiness_label' for the validation dataset...")
y_validation_pred = creditworthiness_model_pipeline.predict(X_validation)


# --- 6. Evaluate the Model on the Validation Set ---
print("\nCreditworthiness Model Evaluation on Validation Set:")
print(f"Accuracy: {accuracy_score(y_validation_true, y_validation_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_validation_true, y_validation_pred))

cm_val = confusion_matrix(y_validation_true, y_validation_pred)
print("\nConfusion Matrix (Validation Set):")
print(cm_val)


# --- 7. Generate 'predicted_creditworthiness_label' column for the entire validation dataset ---
print("\nGenerating 'predicted_creditworthiness_label' for the entire validation dataset (for saving)...")
# Predict on the full validation DataFrame (df_validation) to get predictions for all original IDs
df_validation['predicted_creditworthiness_label'] = creditworthiness_model_pipeline.predict(df_validation[creditworthiness_features].copy())

print("\nValidation DataFrame with 'predicted_creditworthiness_label' column (head):")
print(df_validation[['id', 'var_32', 'creditworthiness_label', 'predicted_creditworthiness_label', 'final_predicted_income']].head())
print(f"Number of rows with 'predicted_creditworthiness_label' predictions: {df_validation['predicted_creditworthiness_label'].count()}")


# --- 8. Save the final Validation DataFrame ---
output_validation_filename = 'processed_dataset_400_with_creditworthiness_predictions.csv'
df_validation.to_csv(output_validation_filename, index=False)
print(f"\nFinal validation dataset saved to '{output_validation_filename}'")

print("\nCreditworthiness validation process complete.")
