# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import lightgbm as lgb # Using LGBMClassifier for classification
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# import joblib # For saving the model
# from sklearn.base import BaseEstimator, TransformerMixin # For Custom TargetEncoder

# # --- Custom TargetEncoder for categorical features ---
# class TargetEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self, cols=None, smoothing=1.0):
#         self.cols = cols
#         self.smoothing = smoothing
#         self.mapping = {}
#         self.global_mean = None

#     def fit(self, X, y):
#         if y is None:
#             raise ValueError("TargetEncoder requires 'y' during fit for target mean calculation.")
        
#         if y.dtype == 'object' or y.dtype == 'string':
#             self.target_map = {
#                 'Poor': 0,
#                 'Average': 1,
#                 'Good': 2
#             }
#             y_numeric = y.map(self.target_map)
#         else:
#             y_numeric = y

#         self.global_mean = y_numeric.mean()

#         if not isinstance(X, pd.DataFrame):
#             if self.cols and len(self.cols) == X.shape[1]:
#                 X_df = pd.DataFrame(X, columns=self.cols)
#             else:
#                 X_df = pd.DataFrame(X)
#         else:
#             X_df = X.copy()

#         for col in self.cols:
#             if col not in X_df.columns:
#                 print(f"Warning: Column '{col}' not found in X during TargetEncoder fit. Skipping.")
#                 continue

#             means = y_numeric.groupby(X_df[col]).mean()
#             counts = y_numeric.groupby(X_df[col]).count()

#             smoothed_means = (means * counts + self.global_mean * self.smoothing) / (counts + self.smoothing)
#             self.mapping[col] = smoothed_means.to_dict()
#         return self

#     def transform(self, X):
#         if not isinstance(X, pd.DataFrame):
#             if self.cols and len(self.cols) == X.shape[1]:
#                 X_transformed = pd.DataFrame(X, columns=self.cols)
#             else:
#                 X_transformed = pd.DataFrame(X)
#         else:
#             X_transformed = X.copy()

#         for col in self.cols:
#             if col in self.mapping:
#                 X_transformed[col] = X_transformed[col].map(self.mapping[col]).fillna(self.global_mean)
#             else:
#                 X_transformed[col] = np.full(len(X_transformed), self.global_mean)
#         return X_transformed

#     def get_feature_names_out(self, input_features=None):
#         return self.cols


# # --- 1. Load the original dataset and Master Model predictions ---
# try:
#     df_original = pd.read_csv('processed_dataset.csv')
#     print("Original dataset 'processed_dataset.csv' loaded successfully.")
    
#     df_master_predictions = pd.read_csv('processed_dataset_with_all_income_predictions.csv')
#     print("Master model predictions loaded successfully.")

#     df = pd.merge(df_original, df_master_predictions[['id', 'final_predicted_income']], on='id', how='left')
#     print("\nDataFrame after merging final income predictions (head):")
#     print(df[['id', 'var_32', 'target_income', 'final_predicted_income']].head())
#     print("\nDataFrame Info (before type explicit conversion):")
#     df.info()

#     categorical_cols_to_convert = [
#         'gender', 'marital_status', 'city', 'state', 'residence_ownership',
#         'pin',
#         'device_model', 'device_category', 'platform', 'device_manufacturer',
#         'var_74', 'var_75'
#     ]
#     for col in categorical_cols_to_convert:
#         if col in df.columns:
#             df[col] = df[col].astype(str).replace('nan', np.nan)
#     print("\nDataFrame Info (after explicit type conversion for categorical columns):")
#     df.info()

# except FileNotFoundError as e:
#     print(f"Error: Required file not found - {e}")
#     print("Please ensure 'processed_dataset.csv' and 'processed_dataset_with_all_income_predictions.csv' are in the same directory.")
#     exit()


# # --- 2. Define the Target Variable: Creditworthiness Labels ---
# TARGET = 'creditworthiness_label'
# CREDIT_SCORE_COL = 'var_32' # The column containing credit scores

# # Drop rows where credit_score is NaN, as we need it to define the target
# df_clean = df.dropna(subset=[CREDIT_SCORE_COL]).copy()

# # --- IMPORTANT DEBUGGING STEP: Analyze actual var_32 distribution ---
# print(f"\nAnalyzing '{CREDIT_SCORE_COL}' (credit_score) distribution:")
# print(df_clean[CREDIT_SCORE_COL].describe())
# print(f"\nValue Counts for '{CREDIT_SCORE_COL}' (binned for overview):")
# print(df_clean[CREDIT_SCORE_COL].value_counts(bins=10, normalize=True).sort_index())


# # Define creditworthiness categories based on the OBSERVED var_32 distribution
# # Adjusted thresholds based on your sample showing var_32 values between 0 and ~2.1
# # *** IMPORTANT: REVIEW THE ABOVE OUTPUT AND ADJUST THESE THRESHOLDS IF NEEDED ***
# # Current assumption: var_32 is scaled, where values like 0.05, 1.31, 2.08 are common.
# def assign_creditworthiness(score):
#     if score >= 1.5: # Example threshold for 'Good' if max is around 2-3
#         return 'Good'
#     elif 0.5 <= score < 1.5: # Example threshold for 'Average'
#         return 'Average'
#     else: # score < 0.5
#         return 'Poor'

# df_clean[TARGET] = df_clean[CREDIT_SCORE_COL].apply(assign_creditworthiness)
# print(f"\nCreditworthiness Distribution (after re-mapping):\n{df_clean[TARGET].value_counts()}")

# # Check if there's only one class after mapping
# if len(df_clean[TARGET].unique()) < 2:
#     print("\nERROR: Only one class found in target variable after mapping.")
#     print("Please review the 'var_32' distribution printed above and adjust 'assign_creditworthiness' thresholds accordingly.")
#     exit()


# # --- 3. Define Features for the Creditworthiness Model ---
# EXCLUDE_COLS = ['id', 'target_income', CREDIT_SCORE_COL, TARGET]

# # SIGNIFICANT CHANGE: AGGRESSIVELY REDUCED FEATURES TO PREVENT MEMORY ERRORS
# # Focusing only on the most impactful and engineered features.
# creditworthiness_features_candidate = [
#     'final_predicted_income', # Crucial input from our Master Income Model
#     # Core Financial & Credit-related (Engineered features) - these are very powerful
#     'financial_health_score',
#     'total_balance',
#     'avg_credit_util',
#     'loan_to_income_1',
#     'loan_to_income_2',
#     # Only the most essential raw numericals
#     'age',
#     # Only core categorical features with manageable cardinality.
#     # High-cardinality ones like 'city', 'pin', 'device_model' will use TargetEncoder.
#     'gender',
#     'marital_status',
#     'residence_ownership',
    
#     # These will be handled by TargetEncoder
#     'city',
#     'state', # Often lower cardinality than city/pin but can still be high. Keeping for target encoding.
#     'pin',
#     'device_model',
#     'device_category', # Can be high cardinality
#     'platform',
#     'device_manufacturer', # Can be high cardinality
#     # Removed var_74, var_75 explicitly as they might have many unique values or be redundant with engineered features.
# ]

# # Filter features to ensure they exist in the DataFrame and are not excluded
# creditworthiness_features = [col for col in creditworthiness_features_candidate if col in df_clean.columns and col not in EXCLUDE_COLS]

# # Split categorical features for different encoding strategies
# # Adjusted lists to reflect aggressive reduction
# high_cardinality_for_target_encoding = [
#     col for col in ['city', 'state', 'pin', 'device_model', 'device_category', 'platform', 'device_manufacturer'] if col in creditworthiness_features
# ]
# low_cardinality_for_onehot_encoding = [
#     col for col in ['gender', 'marital_status', 'residence_ownership'] if col in creditworthiness_features
# ]

# # Ensure no overlap and all desired categorical features are covered
# print(f"\nCategorical Features for Target Encoding: {high_cardinality_for_target_encoding}")
# print(f"Categorical Features for One-Hot Encoding: {low_cardinality_for_onehot_encoding}")

# # Separate numerical features
# creditworthiness_numerical_features = [col for col in creditworthiness_features if pd.api.types.is_numeric_dtype(df_clean[col])]
# print(f"Numerical Features for Preprocessing: {creditworthiness_numerical_features}")


# # --- 4. Split Data into Training and Testing Sets ---
# X = df_clean[creditworthiness_features].copy()
# y = df_clean[TARGET].copy()

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y # Stratify for classification
# )

# print(f"\nCreditworthiness Training data shape: {X_train.shape}")
# print(f"Creditworthiness Testing data shape: {X_test.shape}")
# print(f"Training Target Distribution:\n{y_train.value_counts(normalize=True)}")
# print(f"Testing Target Distribution:\n{y_test.value_counts(normalize=True)}")


# # --- 5. Preprocessing Pipeline for Creditworthiness Model ---

# # Numerical pipeline: Impute with MEAN, then scale
# numerical_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler())
# ])

# # One-Hot Encoding pipeline for low cardinality categorical features
# onehot_categorical_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# # Target Encoding pipeline for high cardinality categorical features
# target_categorical_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('target_encoder', TargetEncoder(cols=high_cardinality_for_target_encoding, smoothing=10.0))
# ])

# # Combine pipelines using ColumnTransformer
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_pipeline, creditworthiness_numerical_features),
#         ('onehot_cat', onehot_categorical_pipeline, low_cardinality_for_onehot_encoding),
#         ('target_cat', target_categorical_pipeline, high_cardinality_for_target_encoding)
#     ],
#     remainder='drop',
#     verbose_feature_names_out=False,
#     n_jobs=1 # Crucial for large datasets to avoid memory errors during parallel processing
# )

# # --- 6. Build and Train the Creditworthiness Classification Model ---
# lgbm_classifier = lgb.LGBMClassifier(
#     random_state=42,
#     objective='multiclass',
#     num_class=len(y.unique()),
#     n_estimators=2000,
#     learning_rate=0.02,
#     num_leaves=40,
#     max_depth=-1,
#     min_child_samples=30,
#     reg_alpha=0.1,
#     reg_lambda=0.1,
#     colsample_bytree=0.7,
#     subsample=0.7,
#     subsample_freq=1,
# )

# creditworthiness_model_pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('classifier', lgbm_classifier)
# ])

# print("\nTraining the Creditworthiness Classification Model...")
# creditworthiness_model_pipeline.fit(X_train, y_train)
# print("Creditworthiness Classification Model training complete.")

# # --- Save the trained model ---
# joblib.dump(creditworthiness_model_pipeline, 'creditworthiness_model_pipeline.pkl')
# print("\nCreditworthiness model saved to 'creditworthiness_model_pipeline.pkl'")


# # --- 7. Make Predictions and Evaluate the Model ---
# y_pred = creditworthiness_model_pipeline.predict(X_test)

# print("\nCreditworthiness Model Evaluation on Test Set:")
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# cm = confusion_matrix(y_test, y_pred)
# print("\nConfusion Matrix:")
# print(cm)


# # --- 8. Generate 'predicted_creditworthiness_label' for the entire dataset ---
# print("\nGenerating 'predicted_creditworthiness_label' for the entire dataset...")
# df['predicted_creditworthiness_label'] = creditworthiness_model_pipeline.predict(df[creditworthiness_features].copy())

# print("\nDataFrame with 'predicted_creditworthiness_label' column (head):")
# print(df[['id', 'var_32', 'creditworthiness_label', 'predicted_creditworthiness_label', 'final_predicted_income']].head())
# print(f"Number of rows with 'predicted_creditworthiness_label' predictions: {df['predicted_creditworthiness_label'].count()}")


# # --- 9. Save the final DataFrame ---
# output_final_filename = 'processed_dataset_with_creditworthiness_predictions.csv'
# df.to_csv(output_final_filename, index=False)
# print(f"\nFinal dataset with creditworthiness predictions saved to '{output_final_filename}'")

# print("\nCreditworthiness classification process complete.")

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb # Using LGBMClassifier for classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib # For saving the model
from sklearn.base import BaseEstimator, TransformerMixin # For Custom TargetEncoder

# Function to get memory usage (for debugging, can be removed later)
def get_memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2) # RSS in MB
    return f"{mem:.2f} MB"

print(f"Initial memory usage: {get_memory_usage()}")


# --- Custom TargetEncoder for categorical features ---
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, smoothing=1.0):
        self.cols = cols
        self.smoothing = smoothing
        self.mapping = {}
        self.global_mean = None

    def fit(self, X, y):
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


# --- 1. Load the original dataset and Master Model predictions ---
try:
    df_original = pd.read_csv('processed_dataset.csv')
    print("Original dataset 'processed_dataset.csv' loaded successfully.")
    print(f"Memory usage after loading original: {get_memory_usage()}")
    
    df_master_predictions = pd.read_csv('processed_dataset_with_all_income_predictions.csv')
    print("Master model predictions loaded successfully.")
    print(f"Memory usage after loading master predictions: {get_memory_usage()}")

    df = pd.merge(df_original, df_master_predictions[['id', 'final_predicted_income']], on='id', how='left')
    print("\nDataFrame after merging final income predictions (head):")
    print(df[['id', 'var_32', 'target_income', 'final_predicted_income']].head())
    print(f"Memory usage after merging: {get_memory_usage()}")

    categorical_cols_to_convert = [
        'gender', 'marital_status', 'city', 'state', 'residence_ownership',
        'pin',
        'device_model', 'device_category', 'platform', 'device_manufacturer',
        'var_74', 'var_75'
    ]
    for col in categorical_cols_to_convert:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', np.nan)
    print("\nDataFrame Info (after explicit type conversion for categorical columns):")
    df.info()
    print(f"Memory usage after categorical conversion: {get_memory_usage()}")

except FileNotFoundError as e:
    print(f"Error: Required file not found - {e}")
    print("Please ensure 'processed_dataset.csv' and 'processed_dataset_with_all_income_predictions.csv' are in the same directory.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during initial loading: {e}")
    sys.exit(1)


# --- 2. Define the Target Variable: Creditworthiness Labels ---
TARGET = 'creditworthiness_label'
CREDIT_SCORE_COL = 'var_32' # The column containing credit scores

# Define creditworthiness categories
def assign_creditworthiness(score):
    # Adjusted thresholds based on your sample showing var_32 values between -2.2 and ~1.9
    if score >= 0.5: # Top range
        return 'Good'
    elif -0.5 <= score < 0.5: # Mid-range
        return 'Average'
    else: # score < -0.5 # Lower range
        return 'Poor'

# CRITICAL FIX: Assign creditworthiness_label to the main df DataFrame
df[TARGET] = df[CREDIT_SCORE_COL].apply(assign_creditworthiness)
print(f"Memory usage after assigning '{TARGET}' to main DataFrame: {get_memory_usage()}")


# Now, create df_clean for training/testing by dropping NaNs in relevant columns
df_clean = df.dropna(subset=[CREDIT_SCORE_COL]).copy()
print(f"Memory usage after creating df_clean (dropping NaN for '{CREDIT_SCORE_COL}'): {get_memory_usage()}")


# --- IMPORTANT DEBUGGING STEP: Analyze actual var_32 distribution ---
print(f"\nAnalyzing '{CREDIT_SCORE_COL}' (credit_score) distribution:")
print(df_clean[CREDIT_SCORE_COL].describe())
print(f"\nValue Counts for '{CREDIT_SCORE_COL}' (binned for overview):")
print(df_clean[CREDIT_SCORE_COL].value_counts(bins=20, normalize=True).sort_index())
print(f"Memory usage after '{CREDIT_SCORE_COL}' analysis: {get_memory_usage()}")


print(f"\nCreditworthiness Distribution (after re-mapping):\n{df_clean[TARGET].value_counts()}")

# Check if there's only one class after mapping
if len(df_clean[TARGET].unique()) < 2:
    print("\nERROR: Only one class found in target variable after mapping.")
    print("Please review the 'var_32' distribution printed above and adjust 'assign_creditworthiness' thresholds accordingly.")
    sys.exit(1)


# --- 3. Define Features for the Creditworthiness Model ---
EXCLUDE_COLS = ['id', 'target_income', CREDIT_SCORE_COL, TARGET]

# AGGRESSIVELY REDUCED FEATURES TO PREVENT MEMORY ERRORS
# Focusing only on the most impactful and engineered features.
creditworthiness_features_candidate = [
    'final_predicted_income', # Crucial input from our Master Income Model
    # Core Financial & Credit-related (Engineered features) - these are very powerful
    'financial_health_score',
    'total_balance',
    'avg_credit_util',
    'loan_to_income_1',
    'loan_to_income_2',
    # Only the most essential raw numericals
    'age',
    # Only core categorical features with manageable cardinality.
    'gender',
    'marital_status',
    'residence_ownership',
    
    # These will be handled by TargetEncoder (re-including var_74, var_75)
    'city',
    'state',
    'pin',
    'device_model',
    'device_category',
    'platform',
    'device_manufacturer',
    'var_74', # Re-included, will be target encoded
    'var_75', # Re-included, will be target encoded
]

# Filter features to ensure they exist in the DataFrame and are not excluded
creditworthiness_features = [col for col in creditworthiness_features_candidate if col in df_clean.columns and col not in EXCLUDE_COLS]

# Split categorical features for different encoding strategies
high_cardinality_for_target_encoding = [
    col for col in ['city', 'state', 'pin', 'device_model', 'device_category', 'platform', 'device_manufacturer', 'var_74', 'var_75'] if col in creditworthiness_features
]
low_cardinality_for_onehot_encoding = [
    col for col in ['gender', 'marital_status', 'residence_ownership'] if col in creditworthiness_features
]

print(f"\nCategorical Features for Target Encoding: {high_cardinality_for_target_encoding}")
print(f"Categorical Features for One-Hot Encoding: {low_cardinality_for_onehot_encoding}")

# Separate numerical features
creditworthiness_numerical_features = [col for col in creditworthiness_features if pd.api.types.is_numeric_dtype(df_clean[col])]
print(f"Numerical Features for Preprocessing: {creditworthiness_numerical_features}")


# --- 4. Split Data into Training and Testing Sets ---
X = df_clean[creditworthiness_features].copy()
y = df_clean[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # Stratify for classification
)

print(f"\nCreditworthiness Training data shape: {X_train.shape}")
print(f"Creditworthiness Testing data shape: {X_test.shape}")
print(f"Training Target Distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Testing Target Distribution:\n{y_test.value_counts(normalize=True)}")


# --- 5. Preprocessing Pipeline for Creditworthiness Model ---

# Numerical pipeline: Impute with MEAN, then scale
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# One-Hot Encoding pipeline for low cardinality categorical features
onehot_categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Target Encoding pipeline for high cardinality categorical features
target_categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('target_encoder', TargetEncoder(cols=high_cardinality_for_target_encoding, smoothing=10.0))
])

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, creditworthiness_numerical_features),
        ('onehot_cat', onehot_categorical_pipeline, low_cardinality_for_onehot_encoding),
        ('target_cat', target_categorical_pipeline, high_cardinality_for_target_encoding)
    ],
    remainder='drop',
    verbose_feature_names_out=False,
    n_jobs=1 # Critical for large datasets to avoid memory errors during parallel processing
)

# --- 6. Build and Train the Creditworthiness Classification Model ---
lgbm_classifier = lgb.LGBMClassifier(
    random_state=42,
    objective='multiclass',
    num_class=len(y.unique()), # This will now be 3 classes
    n_estimators=3000, # Increased
    learning_rate=0.015, # Reduced
    num_leaves=30,     # Reduced for less complexity
    max_depth=-1,
    min_child_samples=50, # Increased for more robust leaves
    reg_alpha=0.2,     # Increased regularization
    reg_lambda=0.2,    # Increased regularization
    colsample_bytree=0.7,
    subsample=0.7,
    subsample_freq=1,
    is_unbalance=True # Added for class imbalance handling
)

creditworthiness_model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgbm_classifier)
])

print("\nTraining the Creditworthiness Classification Model...")
creditworthiness_model_pipeline.fit(X_train, y_train)
print("Creditworthiness Classification Model training complete.")

# --- Save the trained model ---
joblib.dump(creditworthiness_model_pipeline, 'creditworthiness_model_pipeline.pkl')
print("\nCreditworthiness model saved to 'creditworthiness_model_pipeline.pkl'")


# --- 7. Make Predictions and Evaluate the Model ---
y_pred = creditworthiness_model_pipeline.predict(X_test)

print("\nCreditworthiness Model Evaluation on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


# --- 8. Generate 'predicted_creditworthiness_label' for the entire dataset ---
print("\nGenerating 'predicted_creditworthiness_label' for the entire dataset...")
# --- Batch Prediction for Large Datasets ---
chunk_size = 100000 # Process 100,000 rows at a time (adjust based on RAM)
all_predictions = []
total_rows = len(df)

for i in range(0, total_rows, chunk_size):
    # Select a chunk of the DataFrame
    chunk_df = df.iloc[i:i + chunk_size].copy()
    
    # Predict on the current chunk
    chunk_predictions = creditworthiness_model_pipeline.predict(chunk_df[creditworthiness_features].copy())
    all_predictions.extend(chunk_predictions)
    print(f"Processed {min(i + chunk_size, total_rows)}/{total_rows} rows for prediction. Current memory: {get_memory_usage()}")

df['predicted_creditworthiness_label'] = all_predictions

print("\nDataFrame with 'predicted_creditworthiness_label' column (head):")
print(df[['id', 'var_32', 'creditworthiness_label', 'predicted_creditworthiness_label', 'final_predicted_income']].head())
print(f"Number of rows with 'predicted_creditworthiness_label' predictions: {df['predicted_creditworthiness_label'].count()}")


# --- 9. Save the final DataFrame ---
output_final_filename = 'processed_dataset_with_creditworthiness_predictions.csv'
df.to_csv(output_final_filename, index=False)
print(f"\nFinal dataset with creditworthiness predictions saved to '{output_final_filename}'")

print("\nCreditworthiness classification process complete.")
