# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# import numpy as np
# from flask_cors import CORS
# import os # Import os to check for file existence
# from sklearn.base import BaseEstimator, TransformerMixin # Required for custom classes

# app = Flask(__name__)
# CORS(app) # Enable CORS for all routes

# models = {}

# # --- Custom Feature Engineering Function for Location Model (Copied from location.py) ---
# def create_location_engineered_features(X):
#     # It's crucial to work on a copy to avoid SettingWithCopyWarning
#     X_engineered = X.copy()

#     # --- Aggregated Financial Metrics (Relevant to Location Context) ---
#     # These capture overall financial magnitudes which can vary significantly by location
#     # Note: These column lists should be consistent with the features passed to this transformer
#     # Ensure they are numeric before summing.
    
#     # Filter for numeric columns within X_engineered before summing
#     balance_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and any(s in col for s in ['balance', '_0', '_1', '_4', '_8', '_18', '_19', '_21', '_30', '_34', '_35', '_38', '_59', '_68']) and pd.api.types.is_numeric_dtype(X_engineered[col])]
#     credit_limit_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and any(s in col for s in ['credit_limit', '_2', '_3', '_5', '_10', '_11', '_12', '_22', '_23', '_26', '_27', '_28', '_29', '_33', '_44', '_47']) and pd.api.types.is_numeric_dtype(X_engineered[col])]
#     loan_amt_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and any(s in col for s in ['loan_amt', '_6', '_7', '_13', '_14', '_20', '_24', '_31', '_36', '_39', '_42', '_65', '_72']) and pd.api.types.is_numeric_dtype(X_engineered[col])]
#     emi_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and any(s in col for s in ['total_emi', '_9', '_17', '_41', '_43', '_46', '_51', '_56']) and pd.api.types.is_numeric_dtype(X_engineered[col])]
#     repayment_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and any(s in col for s in ['repayment', '_37', '_48', '_49', '_50', '_52', '_55', '_67', '_69', '_70', '_73']) and pd.api.types.is_numeric_dtype(X_engineered[col])]
#     inquiry_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and any(s in col for s in ['inquires', 'inquiries', '_15', '_16', '_25', '_45', '_58', '_61', '_71']) and pd.api.types.is_numeric_dtype(X_engineered[col])]
#     total_loans_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and any(s in col for s in ['total_loans', '_40', '_53', '_54', '_57', '_60', '_62', '_63', '_64', '_66']) and pd.api.types.is_numeric_dtype(X_engineered[col])]
    
#     if balance_cols_in_X:
#         X_engineered['total_balance_fe'] = X_engineered[balance_cols_in_X].sum(axis=1, skipna=True)
#     else:
#         X_engineered['total_balance_fe'] = np.nan # Ensure column exists if no source columns

#     if credit_limit_cols_in_X:
#         X_engineered['total_credit_limit_fe'] = X_engineered[credit_limit_cols_in_X].sum(axis=1, skipna=True)
#     else:
#         X_engineered['total_credit_limit_fe'] = np.nan # Ensure column exists

#     if loan_amt_cols_in_X:
#         X_engineered['total_loan_amount_fe'] = X_engineered[loan_amt_cols_in_X].sum(axis=1, skipna=True)
#     else:
#         X_engineered['total_loan_amount_fe'] = np.nan # Ensure column exists

#     if emi_cols_in_X:
#         X_engineered['total_emi_sum_fe'] = X_engineered[emi_cols_in_X].sum(axis=1, skipna=True)
#     else:
#         X_engineered['total_emi_sum_fe'] = np.nan # Ensure column exists

#     if repayment_cols_in_X:
#         X_engineered['total_repayment_sum_fe'] = X_engineered[repayment_cols_in_X].sum(axis=1, skipna=True)
#     else:
#         X_engineered['total_repayment_sum_fe'] = np.nan # Ensure column exists

#     if inquiry_cols_in_X:
#         X_engineered['total_inquiries_count_fe'] = X_engineered[inquiry_cols_in_X].sum(axis=1, skipna=True)
#     else:
#         X_engineered['total_inquiries_count_fe'] = np.nan # Ensure column exists

#     if total_loans_cols_in_X:
#         X_engineered['total_loans_count_fe'] = X_engineered[total_loans_cols_in_X].sum(axis=1, skipna=True)
#     else:
#         X_engineered['total_loans_count_fe'] = np.nan # Ensure column exists
    
#     # Example Ratio: Credit Utilization (handle division by zero)
#     # Use the newly created _fe features for ratios
#     if 'total_balance_fe' in X_engineered.columns and 'total_credit_limit_fe' in X_engineered.columns:
#         epsilon = 1e-6 
#         X_engineered['credit_utilization_ratio_fe'] = X_engineered['total_balance_fe'] / (X_engineered['total_credit_limit_fe'] + epsilon)
#         X_engineered['credit_utilization_ratio_fe'].replace([np.inf, -np.inf], np.nan, inplace=True)
#     else:
#         X_engineered['credit_utilization_ratio_fe'] = np.nan # Ensure column exists

#     # Handle any potential infinite values that might arise from divisions or other operations
#     X_engineered.replace([np.inf, -np.inf], np.nan, inplace=True)

#     return X_engineered

# # --- Custom Feature Engineering Function for Demographic Model (Copied from demographic.py) ---
# def create_demographic_engineered_features(X):
#     # It's crucial to work on a copy to avoid SettingWithCopyWarning
#     X_engineered = X.copy()

#     # --- Age Binning ---
#     # Convert continuous age into categorical bins.
#     if 'age' in X_engineered.columns and pd.api.types.is_numeric_dtype(X_engineered['age']):
#         bins = [0, 25, 35, 45, 55, 65, 100] # Define age ranges
#         labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+'] # Labels for bins
#         X_engineered['age_bin'] = pd.cut(
#             X_engineered['age'],
#             bins=bins,
#             labels=labels,
#             right=False, # Interval is [min, max)
#             include_lowest=True # Include the first value in the lowest bin
#         ).astype(object) # Ensure the output is an object type for OneHotEncoder
#     else:
#         X_engineered['age_bin'] = np.nan # Ensure column exists if age is missing/invalid

#     # --- Interaction Term: Age multiplied by Credit Score ---
#     # This can capture non-linear relationships or specific segments (e.g., older individuals with high credit scores).
#     if 'age' in X_engineered.columns and 'var_32' in X_engineered.columns and \
#        pd.api.types.is_numeric_dtype(X_engineered['age']) and pd.api.types.is_numeric_dtype(X_engineered['var_32']):
#         X_engineered['age_x_credit_score'] = X_engineered['age'] * X_engineered['var_32']
#     else:
#         X_engineered['age_x_credit_score'] = np.nan # Ensure column exists

#     # Handle any potential infinite values that might arise from divisions (e.g., if you add custom ratios)
#     X_engineered.replace([np.inf, -np.inf], np.nan, inplace=True)

#     return X_engineered

# # --- Custom TargetEncoder for categorical features (Copied from device.py) ---
# class TargetEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self, cols=None, smoothing=1.0):
#         self.cols = cols
#         self.smoothing = smoothing
#         self.mapping = {}
#         self.global_mean = None

#     def fit(self, X, y): # y is now directly passed and used
#         if y is None:
#             raise ValueError("TargetEncoder requires 'y' during fit for target mean calculation.")
        
#         self.global_mean = y.mean()

#         # Ensure 'X' is a DataFrame to handle column names
#         if not isinstance(X, pd.DataFrame):
#             # If X is a numpy array, reconstruct it with expected column names for grouping
#             if self.cols and len(self.cols) == X.shape[1]:
#                 X_df = pd.DataFrame(X, columns=self.cols)
#             else:
#                 # If we can't infer column names, proceed but warn
#                 print("Warning: X is not a DataFrame and column names cannot be inferred. Proceeding without specific column name checks in TargetEncoder.")
#                 X_df = pd.DataFrame(X) # Proceed with generic column names if no self.cols
#         else:
#             X_df = X.copy() # Work on a copy

#         for col in self.cols:
#             if col not in X_df.columns:
#                 print(f"Warning: Column '{col}' not found in X during TargetEncoder fit. Skipping.")
#                 continue

#             # Calculate means for each category by grouping X_df[col] and applying to y
#             means = y.groupby(X_df[col]).mean()
#             counts = y.groupby(X_df[col]).count()

#             # Apply smoothing
#             smoothed_means = (means * counts + self.global_mean * self.smoothing) / (counts + self.smoothing)
#             self.mapping[col] = smoothed_means.to_dict()
#         return self

#     def transform(self, X):
#         # Ensure 'X' is a DataFrame to handle column names
#         if not isinstance(X, pd.DataFrame):
#             if self.cols and len(self.cols) == X.shape[1]:
#                 X_transformed = pd.DataFrame(X, columns=self.cols)
#             else:
#                 print("Warning: X is not a DataFrame and column names cannot be inferred. Proceeding without specific column name checks in TargetEncoder.")
#                 X_transformed = pd.DataFrame(X) # Proceed with generic column names if no self.cols
#         else:
#             X_transformed = X.copy() # Work on a copy

#         for col in self.cols:
#             if col in self.mapping:
#                 # Fill missing values in X_transformed[col] with np.nan before mapping.
#                 # Then map using the learned mapping. For unseen categories, or after mapping
#                 # if there are still NaNs (e.g., original NaNs), fill with global mean.
#                 X_transformed[col] = X_transformed[col].map(self.mapping[col]).fillna(self.global_mean)
#             else:
#                 # If column was not in mapping during fit (e.g., from a new data point),
#                 # fill its values with the global mean.
#                 X_transformed[col] = np.full(len(X_transformed), self.global_mean)
#         return X_transformed

#     def get_feature_names_out(self, input_features=None):
#         # For TargetEncoder, output features are just the input columns, but numerical.
#         return self.cols


# # --- Define the FULL list of features expected by EACH model pipeline. ---
# # These lists MUST precisely match the `*_features` lists in your corresponding training scripts.
# # For simplicity, these are hardcoded here based on the comprehensive lists seen in your training scripts.
# # If your training scripts change feature sets, these lists MUST be updated.

# # Comprehensive list of features for Demographic Model
# DEMOGRAPHIC_MODEL_FEATURES = [
#     'age', 'gender', 'marital_status', 'residence_ownership',
#     'city', 'state', 'pin',
#     'device_model', 'device_category', 'platform', 'device_manufacturer',
#     'var_32', # credit_score
#     'var_0', 'var_1', 'var_4', 'var_8', 'var_18', 'var_19', 'var_21', 'var_30',
#     'var_34', 'var_35', 'var_38', 'var_59', 'var_68', # Balance related
#     'var_2', 'var_3', 'var_5', 'var_10', 'var_11', 'var_12', 'var_22', 'var_23',
#     'var_26', 'var_27', 'var_28', 'var_29', 'var_33', 'var_44', 'var_47', # Credit limit related
#     'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31', 'var_36',
#     'var_39', 'var_42', 'var_65', 'var_72', # Loan amount related
#     'var_9', 'var_17', 'var_41', 'var_43', 'var_46', 'var_51', 'var_56', # EMI related
#     'var_37', 'var_48', 'var_49', 'var_50', 'var_52', 'var_55', 'var_67',
#     'var_69', 'var_70', 'var_73', # Repayment related
#     'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71', # Inquiry related
#     'var_40', # closed_loan
#     'var_53', 'var_54', 'var_57', 'var_60', 'var_62', 'var_63', 'var_64', 'var_66', # Total loans activity
#     'var_74', 'var_75' # Score comments/type
# ]

# # Comprehensive list of features for Behavioral Model
# BEHAVIORAL_MODEL_FEATURES = [
#     'var_0', 'var_1', 'var_4', 'var_8', 'var_18', 'var_19', 'var_21', 'var_30', 'var_34', 'var_35',
#     'var_38', 'var_59', 'var_68', # Balance related
#     'var_2', 'var_3', 'var_5', 'var_10', 'var_11', 'var_12', 'var_22', 'var_23', 'var_26', 'var_27',
#     'var_28', 'var_29', 'var_33', 'var_44', 'var_47', # Credit limit related
#     'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31', 'var_36', 'var_39', 'var_42',
#     'var_65', 'var_72', # Loan amount related
#     'var_9', 'var_17', 'var_41', 'var_43', 'var_46', 'var_51', 'var_56', # EMI related
#     'var_37', 'var_48', 'var_49', 'var_50', 'var_52', 'var_55', 'var_67',
#     'var_69', 'var_70', 'var_73', # Repayment related
#     'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71', # Inquiry related
#     'var_40', # closed_loan
#     'var_53', 'var_54', 'var_57', 'var_60', 'var_62', 'var_63', 'var_64', 'var_66', # Total loans activity
#     'var_32', # credit_score
#     'var_74', 'var_75' # Score comments/type
# ]

# # Comprehensive list of features for Location Model
# LOCATION_MODEL_FEATURES = [
#     'age', 'gender', 'marital_status', 'residence_ownership',
#     'city', 'state', 'pin',
#     'device_model', 'device_category', 'platform', 'device_manufacturer',
#     'var_32', # credit_score
#     'var_0', 'var_1', 'var_4', 'var_8', 'var_18', 'var_19', 'var_21', 'var_30',
#     'var_34', 'var_35', 'var_38', 'var_59', 'var_68', # Balance related
#     'var_2', 'var_3', 'var_5', 'var_10', 'var_11', 'var_12', 'var_22', 'var_23',
#     'var_26', 'var_27', 'var_28', 'var_29', 'var_33', 'var_44', 'var_47', # Credit limit related
#     'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31', 'var_36',
#     'var_39', 'var_42', 'var_65', 'var_72', # Loan amount related
#     'var_9', 'var_17', 'var_41', 'var_43', 'var_46', 'var_51', 'var_56', # EMI related
#     'var_37', 'var_48', 'var_49', 'var_50', 'var_52', 'var_55', 'var_67',
#     'var_69', 'var_70', 'var_73', # Repayment related
#     'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71', # Inquiry related
#     'var_40', # closed_loan
#     'var_53', 'var_54', 'var_57', 'var_60', 'var_62', 'var_63', 'var_64', 'var_66', # Total loans activity
# ]

# # Comprehensive list of features for Device Model
# DEVICE_MODEL_FEATURES = [
#     'device_model', 'device_category', 'platform', 'device_manufacturer',
#     'var_32', # credit_score
#     'var_0', 'var_2', 'var_6', 'var_9', # Example balance, credit_limit, loan_amt, emi
#     'var_15', # Example inquiry count
#     'var_62', # Example total loan recent
#     'var_10', # active_credit_limit_1
#     'var_11', # credit_limit_recent_1
#     'var_24', # loan_amt_recent
#     'var_25', # total_inquiries_recent
# ]

# # Master model features are outputs of slave models + core features
# MASTER_MODEL_FEATURES = [
#     'age', 'var_32', 'financial_health_score', 'total_balance', 'avg_credit_util',
#     'loan_to_income_1', 'loan_to_income_2',
#     'behavioral_income', 'demographic_income', 'location_income', 'device_income'
# ]

# # Creditworthiness model features - UPDATED to include missing categorical features
# CREDITWORTHINESS_MODEL_FEATURES = [
#     'age', 'var_32', 'financial_health_score', 'total_balance', 'avg_credit_util',
#     'loan_to_income_1', 'loan_to_income_2',
#     'behavioral_income', 'demographic_income', 'location_income', 'device_income',
#     'var_74', 'var_75', # If your credit model uses these raw, include them
    
#     # ADDED MISSING CATEGORICAL COLUMNS (from traceback)
#     'city', 'residence_ownership', 'platform', 'device_manufacturer', 'gender',
#     'pin', 'state', 'device_category', 'device_model', 'marital_status'
# ]


# def load_models():
#     """Loads all pre-trained machine learning models from .pkl files."""
#     global models
#     model_paths = {
#         'behavioral': 'behavioral_model_pipeline.pkl',
#         'demographic': 'demographic_model_pipeline.pkl',
#         'location': 'location_model_pipeline.pkl',
#         'device': 'device_model_pipeline.pkl',
#         'master': 'master_model_pipeline.pkl',
#         'creditworthiness': 'creditworthiness_model_pipeline.pkl'
#     }

#     for model_name, path in model_paths.items():
#         if not os.path.exists(path):
#             print(f"Error: Model file not found at {path}. Please ensure all .pkl models are generated and in the correct directory.")
#             # Exit the application if critical models are missing
#             import sys
#             sys.exit(1)
#         try:
#             # For models using custom transformers, load them with the custom objects in scope
#             if model_name == 'demographic':
#                 # Custom objects for demographic model: create_demographic_engineered_features
#                 models[model_name] = joblib.load(path)
#             elif model_name == 'location':
#                 # Custom objects for location model: create_location_engineered_features
#                 models[model_name] = joblib.load(path)
#             elif model_name == 'device':
#                 # Custom objects for device model: TargetEncoder
#                 models[model_name] = joblib.load(path)
#             else:
#                 models[model_name] = joblib.load(path)
            
#             print(f"Model '{model_name}' loaded successfully from '{path}'.")

#         except Exception as e:
#             print(f"Error loading model '{model_name}' from '{path}': {e}")
#             import sys
#             sys.exit(1)

# # Load models when the Flask app starts
# load_models()

# @app.route('/predict', methods=['POST'])
# def predict():
#     """
#     Handles POST requests for loan outcome predictions.
#     Receives input data from the frontend, processes it through slave models,
#     then the master model, and finally the creditworthiness model.
#     Returns JSON response with predicted incomes and creditworthiness.
#     """
#     if not request.json:
#         return jsonify({'success': False, 'error': 'No JSON data received'}), 400

#     data = request.json
#     print(f"Received data: {data}")

#     try:
#         # --- Prepare input and predict for Behavioral Model ---
#         # Ensure all expected features are in the DataFrame, fill missing with np.nan
#         behavioral_input_data = {feature: data.get(feature, np.nan) for feature in BEHAVIORAL_MODEL_FEATURES}
#         behavioral_input = pd.DataFrame([behavioral_input_data])
#         # Convert specific categorical columns to string type if they are expected as such
#         # This list should match what your behavioral.py script does.
#         behavioral_categorical_cols = ['var_74', 'var_75'] # Assuming these are the main categoricals for behavioral
#         for col in behavioral_categorical_cols:
#             if col in behavioral_input.columns:
#                 # Replace string 'None' or 'nan' with actual np.nan before processing
#                 behavioral_input[col] = behavioral_input[col].astype(str).replace(['None', 'nan'], np.nan)
        
#         print(f"Behavioral input DF columns: {behavioral_input.columns.tolist()}")
#         print(f"Behavioral input DF head:\n{behavioral_input.head()}")
#         behavioral_income = models['behavioral'].predict(behavioral_input)[0]
#         if behavioral_income < 0: behavioral_income = 0 # Ensure non-negative income

#         # --- Prepare input and predict for Demographic Model ---
#         demographic_input_data = {feature: data.get(feature, np.nan) for feature in DEMOGRAPHIC_MODEL_FEATURES}
#         demographic_input = pd.DataFrame([demographic_input_data])
#         # Convert specific categorical columns to string type if they are expected as such
#         # This list should match what your demographic.py script does.
#         demographic_categorical_cols = [
#             'gender', 'marital_status', 'city', 'state', 'residence_ownership',
#             'pin', 'device_model', 'device_category', 'platform', 'device_manufacturer',
#             'var_74', 'var_75'
#         ]
#         for col in demographic_categorical_cols:
#             if col in demographic_input.columns:
#                 demographic_input[col] = demographic_input[col].astype(str).replace(['None', 'nan'], np.nan)

#         print(f"Demographic input DF columns: {demographic_input.columns.tolist()}")
#         print(f"Demographic input DF head:\n{demographic_input.head()}")
#         demographic_income = models['demographic'].predict(demographic_input)[0]
#         if demographic_income < 0: demographic_income = 0

#         # --- Prepare input and predict for Location Model ---
#         location_input_data = {feature: data.get(feature, np.nan) for feature in LOCATION_MODEL_FEATURES}
#         location_input = pd.DataFrame([location_input_data])
#         # Convert specific categorical columns to string type if they are expected as such
#         # This list should match what your location.py script does.
#         location_categorical_cols = [
#             'gender', 'marital_status', 'city', 'state', 'residence_ownership',
#             'pin', 'device_model', 'device_category', 'platform', 'device_manufacturer'
#             # Note: var_74, var_75 are not explicitly in location.py's features, but adding if they might implicitly be via other paths
#         ]
#         for col in location_categorical_cols:
#             if col in location_input.columns:
#                 location_input[col] = location_input[col].astype(str).replace(['None', 'nan'], np.nan)

#         print(f"Location input DF columns: {location_input.columns.tolist()}")
#         print(f"Location input DF head:\n{location_input.head()}")
#         location_income = models['location'].predict(location_input)[0]
#         if location_income < 0: location_income = 0

#         # --- Prepare input and predict for Device Model ---
#         device_input_data = {feature: data.get(feature, np.nan) for feature in DEVICE_MODEL_FEATURES}
#         device_input = pd.DataFrame([device_input_data])
#         # Convert specific categorical columns to string type if they are expected as such
#         # This list should match what your device.py script does.
#         device_categorical_cols = [
#             'device_model', 'device_category', 'platform', 'device_manufacturer'
#         ]
#         for col in device_categorical_cols:
#             if col in device_input.columns:
#                 device_input[col] = device_input[col].astype(str).replace(['None', 'nan'], np.nan)

#         print(f"Device input DF columns: {device_input.columns.tolist()}")
#         print(f"Device input DF head:\n{device_input.head()}")
#         device_income = models['device'].predict(device_input)[0]
#         if device_income < 0: device_income = 0

#         # --- Prepare input and predict for Master Model ---
#         master_input_data = {feature: data.get(feature, np.nan) for feature in MASTER_MODEL_FEATURES}
#         # Override with predicted slave model incomes
#         master_input_data['behavioral_income'] = behavioral_income
#         master_input_data['demographic_income'] = demographic_income
#         master_input_data['location_income'] = location_income
#         master_input_data['device_income'] = device_income
#         master_input = pd.DataFrame([master_input_data])

#         print(f"Master input DF columns: {master_input.columns.tolist()}")
#         print(f"Master input DF head:\n{master_input.head()}")
#         final_predicted_income = models['master'].predict(master_input)[0]
#         if final_predicted_income < 0: final_predicted_income = 0

#         # --- Prepare input and predict for Creditworthiness Model ---
#         credit_input_data = {feature: data.get(feature, np.nan) for feature in CREDITWORTHINESS_MODEL_FEATURES}
#         # Add the derived income features and final predicted income
#         credit_input_data['behavioral_income'] = behavioral_income
#         credit_input_data['demographic_income'] = demographic_income
#         credit_input_data['location_income'] = location_income
#         credit_input_data['device_income'] = device_income
#         credit_input_data['final_predicted_income'] = final_predicted_income # Make sure credit model expects this if used

#         credit_input = pd.DataFrame([credit_input_data])
#         # Handle categorical columns for creditworthiness model if it uses them
#         # This list needs to be comprehensive for all categoricals the credit model might expect.
#         credit_categorical_cols = [
#             'gender', 'marital_status', 'city', 'state', 'residence_ownership', 'pin',
#             'device_model', 'device_category', 'platform', 'device_manufacturer',
#             'var_74', 'var_75'
#         ]
#         for col in credit_categorical_cols:
#             if col in credit_input.columns:
#                 credit_input[col] = credit_input[col].astype(str).replace(['None', 'nan'], np.nan)

#         print(f"Creditworthiness input DF columns: {credit_input.columns.tolist()}")
#         print(f"Creditworthiness input DF head:\n{credit_input.head()}")
#         predicted_creditworthiness = models['creditworthiness'].predict(credit_input)[0] # Assuming it's a regression (e.g., probability or score)

#         return jsonify({
#             'success': True,
#             'behavioral_income': behavioral_income,
#             'demographic_income': demographic_income,
#             'location_income': location_income,
#             'device_income': device_income,
#             'final_predicted_income': final_predicted_income,
#             'predicted_creditworthiness': predicted_creditworthiness
#         })

#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         import traceback
#         traceback.print_exc() # Print full traceback to console for detailed error
#         return jsonify({'success': False, 'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

# app.py (Unified Backend for all Models)
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import FunctionTransformer
from category_encoders import TargetEncoder # Correctly imported TargetEncoder

app = Flask(__name__)
CORS(app) # Enable CORS for cross-origin requests from the frontend

# --- Define Paths to Model Files ---
# Ensure these paths are correct relative to where you run this app.py
MODEL_PATHS = {
    'behavioral': 'behavioral_model_pipeline.pkl',
    'demographic': 'demographic_model_pipeline.pkl',
    'creditworthiness': 'creditworthiness_model_pipeline.pkl',
    'location': 'location_model_pipeline.pkl', # Device model path explicitly included
    'device': 'device_model_pipeline.pkl', # Device model path explicitly included
    'master': 'full_income_prediction_pipeline.joblib'
}

# --- Global dictionary to store loaded models ---
loaded_models = {}

# --- Define expected features and categorical columns for each model pipeline ---
# These lists must precisely match the features used during each model's training.
# The order here is CRITICAL and should match your training script's feature order.

# Behavioral Model Features
BEHAVIORAL_FEATURES = [
    'var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6', 'var_7',
    'var_8', 'var_9', 'var_10', 'var_11', 'var_12', 'var_13', 'var_14', 'var_15',
    'var_16', 'var_17', 'var_18', 'var_19', 'var_20', 'var_21', 'var_22', 'var_23',
    'var_24', 'var_25', 'var_26', 'var_27', 'var_28', 'var_29', 'var_30',
    'var_31', 'var_32', 'var_33', 'var_34', 'var_35', 'var_36', 'var_37', 'var_38',
    'var_39', 'var_40', 'var_41', 'var_42', 'var_43', 'var_44', 'var_45', 'var_46',
    'var_47', 'var_48', 'var_49', 'var_50', 'var_51', 'var_52', 'var_53', 'var_54',
    'var_55', 'var_56', 'var_57', 'var_58', 'var_59', 'var_60', 'var_61', 'var_62',
    'var_63', 'var_64', 'var_65', 'var_66', 'var_67', 'var_68', 'var_69', 'var_70',
    'var_71', 'var_72', 'var_73', 'var_74', 'var_75'
]
CATEGORICAL_COLS_BEHAVIORAL = ['var_74', 'var_75']

# Demographic Model Features (ensure order matches training)
DEMOGRAPHIC_FEATURES = [
    'age', 'gender', 'marital_status', 'residence_ownership', 'city', 'state', 'pin',
    'device_model', 'device_category', 'platform', 'device_manufacturer', 'var_32',
    'var_0', 'var_1', 'var_4', 'var_8', 'var_18', 'var_19', 'var_21', 'var_30', 'var_34', 'var_35', 'var_38', 'var_59', 'var_68',
    'var_2', 'var_3', 'var_5', 'var_10', 'var_11', 'var_12', 'var_22', 'var_23', 'var_26', 'var_27', 'var_28', 'var_29', 'var_33', 'var_44', 'var_47',
    'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31', 'var_36', 'var_39', 'var_42', 'var_65', 'var_72',
    'var_9', 'var_17', 'var_41', 'var_43', 'var_46', 'var_51', 'var_56',
    'var_37', 'var_48', 'var_49', 'var_50', 'var_52', 'var_55', 'var_67', 'var_69', 'var_70', 'var_73',
    'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71',
    'var_40', 'var_53', 'var_54', 'var_57', 'var_60', 'var_62', 'var_63', 'var_64', 'var_66',
    'var_74', 'var_75'
]
CATEGORICAL_COLS_DEMOGRAPHIC = [
    'gender', 'marital_status', 'city', 'state', 'residence_ownership', 'pin',
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_74', 'var_75'
]

# Creditworthiness Model Features (ensure order matches training)
CREDITWORTHINESS_FEATURES = [
    'final_predicted_income', 'financial_health_score', 'total_balance',
    'avg_credit_util', 'loan_to_income_1', 'loan_to_income_2',
    'age', 'gender', 'marital_status', 'residence_ownership',
    'city', 'state', 'pin',
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_74', 'var_75'
]
CATEGORICAL_COLS_CREDITWORTHINESS = [
    'gender', 'marital_status', 'city', 'state', 'residence_ownership', 'pin',
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_74', 'var_75'
]

# Location Model Features (ensure order matches training)
LOCATION_FEATURES = [
    'age', 'gender', 'marital_status', 'residence_ownership', 'city', 'state', 'pin',
    'device_model', 'device_category', 'platform', 'device_manufacturer', 'var_32',
    'var_0', 'var_1', 'var_4', 'var_8', 'var_18', 'var_19', 'var_21', 'var_30', 'var_34', 'var_35', 'var_38', 'var_59', 'var_68',
    'var_2', 'var_3', 'var_5', 'var_10', 'var_11', 'var_12', 'var_22', 'var_23', 'var_26', 'var_27', 'var_28', 'var_29', 'var_33', 'var_44', 'var_47',
    'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31', 'var_36', 'var_39', 'var_42', 'var_65', 'var_72',
    'var_9', 'var_17', 'var_41', 'var_43', 'var_46', 'var_51', 'var_56',
    'var_37', 'var_48', 'var_49', 'var_50', 'var_52', 'var_55', 'var_67', 'var_69', 'var_70', 'var_73',
    'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71',
    'var_40', 'var_53', 'var_54', 'var_57', 'var_60', 'var_62', 'var_63', 'var_64', 'var_66',
    'var_74', 'var_75'
]
CATEGORICAL_COLS_LOCATION = [
    'gender', 'marital_status', 'city', 'state', 'residence_ownership', 'pin',
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_74', 'var_75'
]

# Device Model Features (ensure order matches training)
DEVICE_FEATURES = [
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_32', # credit_score
    'var_0', 'var_2', 'var_6', 'var_9', # Example balance, credit_limit, loan_amt, emi
    'var_15', # Example inquiry count
    'var_62', # Example total loan recent
    'var_10', # active_credit_limit_1
    'var_11', # credit_limit_recent_1
    'var_24', # loan_amt_recent
    'var_25', # total_inquiries_recent
]
CATEGORICAL_COLS_DEVICE = [
    'device_model', 'device_category', 'platform', 'device_manufacturer'
]

# Master Model Features (raw inputs before internal feature engineering, ensure order matches training)
MASTER_RAW_FEATURES = [
    'var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'pin', 'var_5', 'var_6', 'var_7',
    'var_8', 'var_9', 'var_10', 'var_11', 'var_12', 'var_13', 'var_14', 'var_15',
    'var_16', 'var_17', 'var_18', 'var_19', 'var_20', 'var_21', 'var_22', 'var_23',
    'var_24', 'var_25', 'var_26', 'var_27', 'age', 'var_28', 'var_29', 'var_30',
    'var_31', 'var_32', 'var_33', 'var_34', 'var_35', 'var_36', 'var_37', 'var_38',
    'var_39', 'var_40', 'var_41', 'var_42', 'var_43', 'var_44', 'var_45', 'var_46',
    'var_47', 'var_48', 'var_49', 'var_50', 'var_51', 'var_52', 'var_53', 'var_54',
    'var_55', 'var_56', 'var_57', 'var_58', 'var_59', 'var_60', 'var_61', 'var_62',
    'var_63', 'var_64', 'var_65', 'var_66', 'var_67', 'var_68', 'var_69', 'var_70',
    'var_71', 'var_72', 'var_73',
    'var_74', 'var_75', 'gender', 'marital_status', 'city', 'state',
    'residence_ownership', 'device_model', 'device_category', 'platform',
    'device_manufacturer'
]
CATEGORICAL_COLS_MASTER = [
    'var_74', 'var_75', 'gender', 'marital_status', 'city', 'state',
    'residence_ownership', 'device_model', 'device_category', 'platform',
    'device_manufacturer', 'pin' # 'pin' is treated as categorical for Master pipeline
]


# Combine all possible features from all models for initial DataFrame creation
ALL_POSSIBLE_FEATURES = sorted(list(set(
    BEHAVIORAL_FEATURES + DEMOGRAPHIC_FEATURES + CREDITWORTHINESS_FEATURES +
    LOCATION_FEATURES + DEVICE_FEATURES + MASTER_RAW_FEATURES
)))

# Combine all categorical column names across all models
ALL_CATEGORICAL_COLS = sorted(list(set(
    CATEGORICAL_COLS_BEHAVIORAL + CATEGORICAL_COLS_DEMOGRAPHIC +
    CATEGORICAL_COLS_CREDITWORTHINESS + CATEGORICAL_COLS_LOCATION +
    CATEGORICAL_COLS_DEVICE + CATEGORICAL_COLS_MASTER
)))


# --- Custom Feature Engineering Functions (MUST BE IDENTICAL TO TRAINING SCRIPTS) ---
# These functions are part of the saved scikit-learn pipelines.
# They must be defined here so joblib.load can properly deserialize the pipelines.

def create_demographic_engineered_features(X):
    X_engineered = X.copy()
    if 'age' in X_engineered.columns and 'var_32' in X_engineered.columns: # var_32 is credit_score
        X_engineered['age_x_credit_score'] = X_engineered['age'] * X_engineered['var_32']
    else:
        X_engineered['age_x_credit_score'] = np.nan

    if 'age' in X_engineered.columns:
        # Simple binning: 0-25 -> Young, 26-45 -> Adult, 46+ -> Senior
        X_engineered['age_bin'] = pd.cut(X_engineered['age'],
                                         bins=[-1, 25, 45, np.inf],
                                         labels=['Young', 'Adult', 'Senior'],
                                         right=True,
                                         include_lowest=True).astype(str)
        X_engineered['age_bin'] = X_engineered['age_bin'].replace('nan', np.nan)
    else:
        X_engineered['age_bin'] = np.nan

    X_engineered.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X_engineered

def create_location_engineered_features(X):
    X_engineered = X.copy()
    epsilon = 1e-6 # Define epsilon once

    # Total Balance
    balance_cols = [f'var_{i}' for i in [0, 1, 4, 8, 18, 19, 21, 30, 34, 35, 38, 59, 68]]
    existing_balance_cols = [col for col in balance_cols if col in X_engineered.columns]
    if existing_balance_cols:
        X_engineered['total_balance'] = X_engineered[existing_balance_cols].sum(axis=1, skipna=True)
    else:
        X_engineered['total_balance'] = np.nan

    # Total Credit Limit
    credit_limit_cols = [f'var_{i}' for i in [2, 3, 5, 10, 11, 12, 22, 23, 26, 27, 28, 29, 33, 44, 47]]
    existing_credit_limit_cols = [col for col in credit_limit_cols if col in X_engineered.columns]
    if existing_credit_limit_cols:
        X_engineered['total_credit_limit'] = X_engineered[existing_credit_limit_cols].sum(axis=1, skipna=True)
    else:
        X_engineered['total_credit_limit'] = np.nan

    # Total Loan Amount
    loan_amount_cols = [f'var_{i}' for i in [6, 7, 13, 14, 20, 24, 31, 36, 39, 42, 65, 72]]
    existing_loan_amount_cols = [col for col in loan_amount_cols if col in X_engineered.columns]
    if existing_loan_amount_cols:
        X_engineered['total_loan_amount'] = X_engineered[existing_loan_amount_cols].sum(axis=1, skipna=True)
    else:
        X_engineered['total_loan_amount'] = np.nan

    # Total EMI Sum
    emi_cols = [f'var_{i}' for i in [9, 17, 41, 43, 46, 51, 56]]
    existing_emi_cols = [col for col in emi_cols if col in X_engineered.columns]
    if existing_emi_cols:
        X_engineered['total_emi_sum'] = X_engineered[existing_emi_cols].sum(axis=1, skipna=True)
    else:
        X_engineered['total_emi_sum'] = np.nan

    # Total Repayment Sum
    repayment_cols = [f'var_{i}' for i in [37, 48, 49, 50, 52, 55, 67, 69, 70, 73]]
    existing_repayment_cols = [col for col in repayment_cols if col in X_engineered.columns]
    if existing_repayment_cols:
        X_engineered['total_repayment_sum'] = X_engineered[existing_repayment_cols].sum(axis=1, skipna=True)
    else:
        X_engineered['total_repayment_sum'] = np.nan

    # Total Inquiries Count
    inquiry_cols = [f'var_{i}' for i in [15, 16, 25, 45, 58, 61, 71]]
    existing_inquiry_cols = [col for col in inquiry_cols if col in X_engineered.columns]
    if existing_inquiry_cols:
        X_engineered['total_inquiries_count'] = X_engineered[existing_inquiry_cols].sum(axis=1, skipna=True)
    else:
        X_engineered['total_inquiries_count'] = np.nan

    # Total Loans Count (Approximate, based on loan activity vars)
    loans_count_cols = [f'var_{i}' for i in [40, 53, 54, 57, 60, 62, 63, 64, 66]]
    existing_loans_count_cols = [col for col in loans_count_cols if col in X_engineered.columns]
    if existing_loans_count_cols:
        X_engineered['total_loans_count'] = X_engineered[existing_loans_count_cols].sum(axis=1, skipna=True) # Summing these could represent activity
    else:
        X_engineered['total_loans_count'] = np.nan

    # Credit Utilization Ratio
    if 'total_balance' in X_engineered.columns and 'total_credit_limit' in X_engineered.columns:
        X_engineered['credit_utilization_ratio'] = X_engineered['total_balance'] / (X_engineered['total_credit_limit'] + epsilon)
        X_engineered['credit_utilization_ratio'] = X_engineered['credit_utilization_ratio'].replace([np.inf, -np.inf], np.nan) # Fix inplace warning
    else:
        X_engineered['credit_utilization_ratio'] = np.nan

    X_engineered.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X_engineered

def create_master_engineered_features(X):
    # It's crucial to work on a copy to avoid SettingWithCopyWarning
    X_engineered = X.copy()
    epsilon = 1e-6 # Define epsilon once

    # Example 1: Ratio of balance to credit limit
    balance_credit_map = {
        'balance_credit_ratio_1': ('var_0', 'var_2'),
        'balance_credit_ratio_2': ('var_1', 'var_3'),
        'balance_credit_ratio_3': ('var_4', 'var_5'),
    }

    for new_col, (balance_col, credit_col) in balance_credit_map.items():
        if balance_col in X_engineered.columns and credit_col in X_engineered.columns:
            temp_balance = X_engineered[balance_col].fillna(0)
            temp_credit = X_engineered[credit_col].fillna(0)
            X_engineered[new_col] = temp_balance / (temp_credit + epsilon)
            X_engineered[new_col] = X_engineered[new_col].replace([np.inf, -np.inf], np.nan) # Fix inplace warning
        else:
            X_engineered[new_col] = np.nan

    # Example 2: Total loan amount
    actual_loan_amt_cols = [
        'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31',
        'var_36', 'var_39', 'var_42', 'var_65', 'var_72'
    ]
    existing_loan_cols = [col for col in actual_loan_amt_cols if col in X_engineered.columns]
    if existing_loan_cols:
        X_engineered['total_loan_amount'] = X_engineered[existing_loan_cols].sum(axis=1, skipna=True)
    else:
        X_engineered['total_loan_amount'] = np.nan

    # Example 3: Total inquiries
    actual_inquiries_cols = [
        'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71'
    ]
    existing_inquiries_cols = [col for col in actual_inquiries_cols if col in X_engineered.columns]
    if existing_inquiries_cols:
        X_engineered['total_inquiries_all'] = X_engineered[existing_inquiries_cols].sum(axis=1, skipna=True)
    else:
        X_engineered['total_inquiries_all'] = np.nan

    # Example 4: Age-Credit Score Interaction
    if 'age' in X_engineered.columns and 'var_32' in X_engineered.columns: # var_32 is credit_score
        X_engineered['age_credit_score_interaction'] = X_engineered['age'] * X_engineered['var_32']
    else:
        X_engineered['age_credit_score_interaction'] = np.nan

    # Financial Health Score (Dummy example, replace with actual calculation if available)
    if 'var_0' in X_engineered.columns and 'var_6' in X_engineered.columns:
        X_engineered['financial_health_score'] = X_engineered['var_0'] / (X_engineered['var_6'] + epsilon)
        X_engineered['financial_health_score'] = X_engineered['financial_health_score'].replace([np.inf, -np.inf], np.nan) # Fix inplace warning
    else:
        X_engineered['financial_health_score'] = np.nan
    
    # Total Balance (Dummy, if not already in X_engineered from previous steps or direct input)
    if 'var_0' in X_engineered.columns and 'var_1' in X_engineered.columns:
        X_engineered['total_balance'] = X_engineered['var_0'] + X_engineered['var_1']
    else:
        X_engineered['total_balance'] = np.nan

    # Average Credit Utilization (Dummy, if not directly available)
    if 'var_2' in X_engineered.columns and 'var_0' in X_engineered.columns:
        X_engineered['avg_credit_util'] = X_engineered['var_0'] / (X_engineered['var_2'] + epsilon)
        X_engineered['avg_credit_util'] = X_engineered['avg_credit_util'].replace([np.inf, -np.inf], np.nan) # Fix inplace warning
    else:
        X_engineered['avg_credit_util'] = np.nan

    # Loan to Income Ratios (Dummy, assuming target_income might be used, or a proxy)
    if 'total_loan_amount' in X_engineered.columns and 'var_32' in X_engineered.columns:
         X_engineered['loan_to_income_1'] = X_engineered['total_loan_amount'] / (X_engineered['var_32'] + epsilon)
         X_engineered['loan_to_income_1'] = X_engineered['loan_to_income_1'].replace([np.inf, -np.inf], np.nan) # Fix inplace warning
    else:
         X_engineered['loan_to_income_1'] = np.nan

    if 'total_loan_amount' in X_engineered.columns and 'age' in X_engineered.columns:
        X_engineered['loan_to_income_2'] = X_engineered['total_loan_amount'] / (X_engineered['age'] * 1000 + epsilon)
        X_engineered['loan_to_income_2'] = X_engineered['loan_to_income_2'].replace([np.inf, -np.inf], np.nan) # Fix inplace warning
    else:
        X_engineered['loan_to_income_2'] = np.nan

    X_engineered.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X_engineered


# --- Load all trained models ---
for model_name, path in MODEL_PATHS.items():
    try:
        if model_name == 'master':
            # Joblib needs to know about the custom function for FunctionTransformer
            loaded_models[model_name] = joblib.load(path)
        else:
            loaded_models[model_name] = joblib.load(path)
        print(f"Model '{model_name}' loaded successfully from '{path}'.")
    except FileNotFoundError:
        print(f"ERROR: Model file for '{model_name}' not found at '{path}'. Skipping this model.")
        loaded_models[model_name] = None
    except Exception as e:
        print(f"ERROR: Could not load model '{model_name}' from '{path}': {e}. Skipping this model.")
        loaded_models[model_name] = None

@app.route('/predict_all', methods=['POST'])
def predict_all():
    data = request.get_json(force=True)

    if not data:
        return jsonify({'error': 'No input data provided.'}), 400

    results = {}
    overall_status_code = 200

    # Create a single DataFrame from the incoming JSON data with ALL_POSSIBLE_FEATURES
    # This ensures all columns the models *might* expect are present, even if NaN.
    input_data_dict = {}
    for col in ALL_POSSIBLE_FEATURES:
        val = data.get(col)
        # Convert numerical inputs to floats, or NaN if empty/None
        if col in ['age', 'var_32', 'var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6', 'var_7', 'var_8', 'var_9', 'var_10', 'var_11', 'var_12', 'var_13', 'var_14', 'var_15', 'var_16', 'var_17', 'var_18', 'var_19', 'var_20', 'var_21', 'var_22', 'var_23', 'var_24', 'var_25', 'var_26', 'var_27', 'var_28', 'var_29', 'var_30', 'var_31', 'var_33', 'var_34', 'var_35', 'var_36', 'var_37', 'var_38', 'var_39', 'var_40', 'var_41', 'var_42', 'var_43', 'var_44', 'var_45', 'var_46', 'var_47', 'var_48', 'var_49', 'var_50', 'var_51', 'var_52', 'var_53', 'var_54', 'var_55', 'var_56', 'var_57', 'var_58', 'var_59', 'var_60', 'var_61', 'var_62', 'var_63', 'var_64', 'var_65', 'var_66', 'var_67', 'var_68', 'var_69', 'var_70', 'var_71', 'var_72', 'var_73',
                    'final_predicted_income', 'financial_health_score', 'total_balance', 'avg_credit_util', 'loan_to_income_1', 'loan_to_income_2']:
            try:
                input_data_dict[col] = float(val) if val is not None and val != '' else np.nan
            except ValueError:
                input_data_dict[col] = np.nan # If conversion fails, set to NaN
        else: # Treat all other as strings, convert empty string to NaN
            input_data_dict[col] = str(val) if val is not None and val != '' else np.nan

    # Create the DataFrame ensuring all ALL_POSSIBLE_FEATURES are columns
    input_df_full = pd.DataFrame([input_data_dict], columns=ALL_POSSIBLE_FEATURES)

    # Convert known categorical columns to object/string type
    for col in ALL_CATEGORICAL_COLS:
        if col in input_df_full.columns:
            input_df_full[col] = input_df_full[col].astype(str)
            input_df_full.loc[input_df_full[col] == 'nan', col] = np.nan # Convert 'nan' strings to actual np.nan

    # --- Predict with Behavioral Model ---
    if loaded_models.get('behavioral'):
        try:
            # Ensure behavioral_input_df has exact features in correct order
            behavioral_input_df = input_df_full.reindex(columns=BEHAVIORAL_FEATURES).copy()
            predicted_income = loaded_models['behavioral'].predict(behavioral_input_df)[0]
            results['behavioral_income_prediction'] = round(float(max(0, predicted_income)), 2)
        except Exception as e:
            results['behavioral_income_prediction_error'] = f"Behavioral model prediction failed: {str(e)}"
            overall_status_code = 500
    else:
        results['behavioral_income_prediction_status'] = "Model not loaded."

    # --- Predict with Demographic Model ---
    if loaded_models.get('demographic'):
        try:
            demographic_input_df = input_df_full.reindex(columns=DEMOGRAPHIC_FEATURES).copy()
            predicted_income = loaded_models['demographic'].predict(demographic_input_df)[0]
            results['demographic_income_prediction'] = round(float(max(0, predicted_income)), 2)
        except Exception as e:
            results['demographic_income_prediction_error'] = f"Demographic model prediction failed: {str(e)}"
            overall_status_code = 500
    else:
        results['demographic_income_prediction_status'] = "Model not loaded."

    # --- Predict with Location Model ---
    if loaded_models.get('location'):
        try:
            location_input_df = input_df_full.reindex(columns=LOCATION_FEATURES).copy()
            predicted_income = loaded_models['location'].predict(location_input_df)[0]
            results['location_income_prediction'] = round(float(max(0, predicted_income)), 2)
        except Exception as e:
            results['location_income_prediction_error'] = f"Location model prediction failed: {str(e)}"
            overall_status_code = 500
    else:
        results['location_income_prediction_status'] = "Model not loaded."

    # --- Predict with Device Model ---
    if loaded_models.get('device'):
        try:
            device_input_df = input_df_full.reindex(columns=DEVICE_FEATURES).copy()
            predicted_income = loaded_models['device'].predict(device_input_df)[0]
            results['device_income_prediction'] = round(float(max(0, predicted_income)), 2)
        except Exception as e:
            results['device_income_prediction_error'] = f"Device model prediction failed: {str(e)}"
            overall_status_code = 500
    else:
        results['device_income_prediction_status'] = "Model not loaded."

    # --- Predict with Master Model ---
    if loaded_models.get('master'):
        try:
            master_input_df = input_df_full.reindex(columns=MASTER_RAW_FEATURES).copy()
            predicted_income = loaded_models['master'].predict(master_input_df)[0]
            results['master_income_prediction'] = round(float(max(0, predicted_income)), 2)
            # Propagate master prediction to input_df_full for creditworthiness model
            input_df_full['final_predicted_income'] = predicted_income
        except Exception as e:
            results['master_income_prediction_error'] = f"Master model prediction failed: {str(e)}"
            overall_status_code = 500
    else:
        results['master_income_prediction_status'] = "Model not loaded."

    # --- Predict with Creditworthiness Model ---
    # This model depends on 'final_predicted_income', which should now be in input_df_full
    if loaded_models.get('creditworthiness'):
        try:
            # Ensure final_predicted_income is set in input_df_full if master model ran
            if 'final_predicted_income' not in input_df_full.columns or pd.isna(input_df_full['final_predicted_income'].iloc[0]):
                # If master prediction failed or wasn't run, use value from frontend input if available, else NaN
                input_df_full['final_predicted_income'] = input_df_full['final_predicted_income'].fillna(data.get('final_predicted_income', np.nan))

            creditworthiness_input_df = input_df_full.reindex(columns=CREDITWORTHINESS_FEATURES).copy()
            
            if hasattr(loaded_models['creditworthiness'], 'predict_proba'):
                probabilities = loaded_models['creditworthiness'].predict_proba(creditworthiness_input_df)[0]
                classes = loaded_models['creditworthiness'].classes_
                prob_dict = {str(classes[i]): round(float(probabilities[i]), 4) for i in range(len(classes))}
                results['creditworthiness_prediction_probabilities'] = prob_dict
            
            predicted_label = loaded_models['creditworthiness'].predict(creditworthiness_input_df)[0]
            results['creditworthiness_prediction_label'] = str(predicted_label)

        except Exception as e:
            results['creditworthiness_prediction_error'] = f"Creditworthiness model prediction failed: {str(e)}"
            overall_status_code = 500
    else:
        results['creditworthiness_prediction_status'] = "Model not loaded."

    return jsonify(results), overall_status_code

# To run the Flask app: python app.py
if __name__ == '__main__':
    # Load models at startup
    app.run(debug=True, port=5000) # Running on a single port for all services
