import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import pickle
from decision_tree import DecisionTree, RandomForest
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
from tqdm import tqdm

# Create model directory
os.makedirs("models", exist_ok=True)

# Determine available CPU cores
NUM_CORES = multiprocessing.cpu_count() - 1 or 1  # Reserve one core for the system
print(f"Will use {NUM_CORES} CPU cores for training")

class RochesterWeatherPredictor:
    """Rochester weather prediction model training and evaluation"""
    
    def __init__(self, data_path="data/rochester_with_targets.csv"):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = {}
        self.y_test = {}
        self.models = {}
        self.feature_importance = {}
        self.feature_names = {}
        self.target_columns = [
            'temp_higher_than_yesterday',
            'temp_higher_than_average',
            'has_more_than_trace_precip'
        ]
        
        # Define important feature groups
        self.feature_groups = {
            'temperature': [
                'max_temp', 'min_temp', 'avg_temp', 'departure', 'hdd', 'cdd', # Add hdd/cdd
                'max_temp_day1', 'min_temp_day1', 'avg_temp_day1',
                'max_temp_day2', 'min_temp_day2', 'avg_temp_day2',
                'max_temp_day3', 'min_temp_day3', 'avg_temp_day3',
                'max_temp_day4', 'min_temp_day4', 'avg_temp_day4',
                'max_temp_day5', 'min_temp_day5', 'avg_temp_day5'
            ],
            'precipitation': [
                'precipitation', 'precipitation_day1', 'precipitation_day2',
                'precipitation_day3', 'precipitation_day4', 'precipitation_day5',
                'days_since_rain', 'days_since_snow'
            ],
            'snow': [
                'snow', 'snow_depth', 
                'snow_day1', 'snow_depth_day1', 
                'snow_day2', 'snow_depth_day2',
                'snow_day3', 'snow_depth_day3', 
                'snow_day4', 'snow_depth_day4',
                'snow_day5', 'snow_depth_day5'
            ],
            'wind': [
                'avg_wind_speed', 'avg_wind_speed_day1', 'avg_wind_speed_day2',
                'avg_wind_speed_day3', 'avg_wind_speed_day4', 'avg_wind_speed_day5',
                'peak_wind_speed', 'peak_wind_speed_day1', 'peak_wind_speed_day2',
                'peak_wind_speed_day3', 'peak_wind_speed_day4', 'peak_wind_speed_day5'
            ],
            'statistics': [
                'temp_std_3d', 'temp_std_5d', 'temp_mean_3d', 'temp_mean_5d',
                'precip_sum_3d', 'precip_sum_5d', 'precip_mean_3d', 'precip_mean_5d',
                'wind_mean_3d', 'wind_mean_5d', 'wind_std_3d', 'wind_std_5d'
            ],
            # Add potential boolean flag feature group
            'flags': [
                'has_precipitation', 'has_rain', 'has_snow', 'has_fog', 
                'has_thunder', 'has_smoke', 'has_hail', 'has_dust'
            ]
        }
        
        # Define features to exclude related to precipitation/snow for the current day (stricter)
        self.leakage_features_precip = [
            'precipitation', 'snow', 'snow_depth',
            'has_precipitation', 'has_rain', 'has_snow',
            # Add potential leakage features
            'days_since_rain', 'days_since_snow',
            'precip_sum_3d', 'precip_sum_5d', 
            'precip_mean_3d', 'precip_mean_5d'
        ]
        
        # Define features to exclude related to average temperature for the current day
        self.leakage_features_temp_avg = [
            'departure', 'hdd', 'cdd' # hdd/cdd are also related to average temperature and baseline
        ]
    
    def load_data(self):
        """Load data"""
        print(f"Loading data: {self.data_path}")
        try:
            self.data = pd.read_csv(self.data_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            print(f"Data loaded successfully, shape: {self.data.shape}")
            
            # Ensure all potential feature columns exist, even if they're not in the CSV
            all_potential_features = set()
            for group in self.feature_groups.values():
                all_potential_features.update(group)
                
            for feature_col in all_potential_features:
                 if feature_col not in self.data.columns:
                     # For flags, fill with 0; for others, let clean_data handle it
                     if feature_col in self.feature_groups.get('flags', []):
                         self.data[feature_col] = 0
                         print(f"Warning: Flag feature '{feature_col}' not in data, added and filled with 0.")
                     # else: # Let clean_data handle other missing columns
                     #     print(f"Info: Feature column '{feature_col}' not in original data.")
                         
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def clean_data(self):
        """Clean data, handle non-numeric and missing values"""
        print("Cleaning data...")
        
        # Copy data to avoid modifying original data
        clean_data = self.data.copy()
        
        # Handle 'M' values and other non-numeric data
        for col in clean_data.columns:
            if col not in ['date', 'city', 'city_code'] + self.target_columns:
                # Replace 'M' and 'T' (trace) with NaN, especially for numeric columns
                if clean_data[col].dtype == 'object':
                    clean_data[col] = clean_data[col].replace(['M', 'T'], np.nan)
                    # Try to convert to numeric type
                    try:
                        clean_data[col] = pd.to_numeric(clean_data[col], errors='coerce')
                    except:
                        print(f"Warning: Cannot convert column '{col}' to numeric type, even after replacing M/T.")
                # If already numeric, also replace T (might be mixed in)
                elif pd.api.types.is_numeric_dtype(clean_data[col].dtype):
                     # Numeric columns shouldn't have 'T', but just in case
                     pass # Usually not needed, but keep the logic
        
        # Fill missing values with median
        filled_cols = []
        for col in clean_data.columns:
            if col not in ['date', 'city', 'city_code'] + self.target_columns:
                if clean_data[col].isna().any():
                    # Check if column is completely NaN
                    if clean_data[col].isna().all():
                         print(f"Warning: Column '{col}' is all missing values, will fill with 0.")
                         clean_data[col] = clean_data[col].fillna(0)
                         filled_cols.append(f"{col}(all NaN->0)")
                    else:
                        # Ensure column is numeric type before calculating median
                        if pd.api.types.is_numeric_dtype(clean_data[col].dtype):
                             median_val = clean_data[col].median()
                             clean_data[col] = clean_data[col].fillna(median_val)
                             filled_cols.append(f"{col}(median {median_val:.2f})")
                        else:
                             # For non-numeric columns (theoretically shouldn't exist after replacement), fill with a special value or mode
                             print(f"Warning: Non-numeric column '{col}' still has missing values, will fill with 'Unknown'.")
                             clean_data[col] = clean_data[col].fillna('Unknown') 
                             filled_cols.append(f"{col}(non-numeric->'Unknown')")
                             
        if filled_cols:
             print("Filled missing values in the following columns:", ", ".join(filled_cols))
        else:
             print("No missing values found in the data that need filling.")

        self.data = clean_data
        print("Data cleaning complete")
    
    def analyze_feature_importance(self):
        """Analyze feature importance"""
        print("\nAnalyzing feature importance...")
        
        # For each target variable
        for target in self.target_columns:
            print(f"\nAnalyzing feature importance for {target}:")
            
            # For each feature group
            for group_name, features in self.feature_groups.items():
                # Calculate correlation of features in this group with the target variable
                correlations = []
                valid_features_in_group = [f for f in features if f in self.data.columns]
                
                if not valid_features_in_group:
                    # print(f"\nNo valid features in {group_name} group.")
                    continue
                    
                for feature in valid_features_in_group:
                    # Ensure both target and feature columns are numeric type for correlation calculation
                    if pd.api.types.is_numeric_dtype(self.data[feature]) and pd.api.types.is_numeric_dtype(self.data[target]):
                        try:
                            # Avoid calculating perfect correlation with itself
                            if feature == target:
                                continue
                            corr = abs(self.data[feature].corr(self.data[target]))
                            # Check if corr is NaN (if a column has zero variance)
                            if not np.isnan(corr):
                                correlations.append((feature, corr))
                            # else:
                                # print(f"Warning: Got NaN when calculating correlation between '{feature}' and '{target}' (possibly zero variance)")
                        except Exception as e:
                            print(f"Warning: Error calculating correlation between '{feature}' and '{target}': {str(e)}")
                    # else:
                         # print(f"Info: Skipping correlation calculation for non-numeric feature '{feature}' or target '{target}'.")
                
                # Sort and display
                if correlations:
                    correlations.sort(key=lambda x: x[1], reverse=True)
                    print(f"\n{group_name} group feature importance (Top 5):")
                    for feature, corr in correlations[:5]: 
                        print(f"  {feature}: {corr:.4f}")
                # else:
                    # print(f"\nNo valid correlations calculated for {group_name} group.")

    
    def select_features(self, target):
        """Select most important features for a specific target variable, prevent data leakage"""
        print(f"\nSelecting features for {target}...")
        
        potential_features = []
        excluded_count = 0
        for col in self.data.columns:
            # Exclude non-feature columns
            if col in ['date', 'city', 'city_code'] + self.target_columns:
                continue
            
            # Exclude specific leakage features based on target
            excluded = False
            if target == 'has_more_than_trace_precip' and col in self.leakage_features_precip:
                excluded = True
            elif target == 'temp_higher_than_average' and col in self.leakage_features_temp_avg:
                 excluded = True
                 
            if excluded:
                 # print(f"  Excluding potential leakage feature: {col}") # Can uncomment to see details
                 excluded_count += 1
                 continue
                
            potential_features.append(col)
        
        if excluded_count > 0:
            print(f"  Excluded {excluded_count} potential leakage features for target '{target}'.")
            
        # Select features based on correlation
        correlations = []
        for feature in potential_features:
             # Ensure both target and feature columns are numeric type for correlation calculation
            if pd.api.types.is_numeric_dtype(self.data[feature]) and pd.api.types.is_numeric_dtype(self.data[target]):
                try:
                    corr = abs(self.data[feature].corr(self.data[target]))
                    if not np.isnan(corr):
                         correlations.append((feature, corr))
                    # else:
                         # pass # Silently handle features with zero variance
                except Exception as e:
                    print(f"Warning: Error calculating correlation between '{feature}' and '{target}': {str(e)}")

        
        # Sort by correlation
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Select features with highest correlation, requiring correlation greater than 0.05 (slightly relaxed, as strong features were excluded)
        corr_threshold = 0.05
        selected_features = [f[0] for f in correlations if f[1] > corr_threshold][:15]  # Maximum 15 features
        
        if not selected_features:
             print(f"Warning: No features found with correlation > {corr_threshold} for {target}. Will use top 10 features by correlation.")
             selected_features = [f[0] for f in correlations][:10]
             
        print(f"\nSelected features for {target} ({len(selected_features)}):")
        for feature in selected_features:
            print(f"  {feature}")
            
        self.feature_names[target] = selected_features # Save feature names for each target
        return selected_features
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Data preprocessing and splitting"""
        print("\nStarting data preprocessing...")
        
        # Ensure target variables are numeric
        for target in self.target_columns:
            if target in self.data.columns:
                self.data[target] = self.data[target].astype(int)
            else:
                print(f"Error: Target column '{target}' lost after data loading or cleaning.")
                return False # Preprocessing failed
        
        self.X_train, self.X_test, self.y_train, self.y_test = {}, {}, {}, {}
        processed_targets = []
        
        # For each target variable
        for target in self.target_columns:
            print("--"*10)
            # Select features
            selected_features = self.select_features(target)
            
            # If no features were selected, skip this target
            if not selected_features:
                print(f"Warning: Cannot select any features for target '{target}', skipping training for this target.")
                continue
                
            # Prepare feature data
            feature_data = self.data[selected_features].copy()
            
            # Double-check for non-numeric columns in feature data (theoretically shouldn't happen)
            non_numeric_cols = feature_data.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                 print(f"Warning: Found non-numeric columns in features selected for '{target}': {list(non_numeric_cols)}. Attempting conversion...")
                 for col in non_numeric_cols:
                     try:
                         feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
                         # If still has NaN after conversion, fill with 0
                         if feature_data[col].isna().any():
                              print(f"  Column '{col}' still has NaN after conversion, filling with 0.")
                              feature_data[col] = feature_data[col].fillna(0)
                     except:
                          print(f"Error: Cannot force convert feature column '{col}' to numeric type. Skipping target '{target}'.")
                          # Skip subsequent processing for current target
                          selected_features = [] # Clear feature list to trigger skip logic
                          break 
                 if not selected_features: # If we broke out of inner loop
                      continue
                          
            # Data normalization
            scaler = StandardScaler()
            try:
                X_scaled = scaler.fit_transform(feature_data)
            except ValueError as e:
                print(f"Error: Error normalizing data for target '{target}': {e}")
                print(f"  Feature data preview:\n{feature_data.head()}")
                continue # Skip this target
                
            # Split data
            y = self.data[target].values
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
                )
            except ValueError as e:
                 print(f"Error: Error splitting data for target '{target}' (possibly due to class imbalance or too few samples): {e}")
                 continue # Skip this target
                 
            # Save data (separately for each target)
            self.X_train[target] = X_train
            self.X_test[target] = X_test
            self.y_train[target] = y_train
            self.y_test[target] = y_test
            processed_targets.append(target)
            
            # Save preprocessor
            with open(f'models/scaler_{target}.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save feature names (already updated self.feature_names[target] in select_features)
            with open(f'models/feature_names_{target}.json', 'w') as f:
                json.dump(selected_features, f)
            
            print(f"\n{target} preprocessing complete. Number of features: {len(selected_features)}")
            print(f"Training set size: {X_train.shape}")
            print(f"Test set size: {X_test.shape}")
                
        # Check if at least one target was successfully processed
        if not processed_targets:
            print("Error: Failed to preprocess data for any target.")
            return False
            
        print(f"\nData preprocessing complete, successfully processed targets: {processed_targets}")
        return True

    
    def train_single_tree(self, target):
        """Train a single decision tree model"""
        # Check if data for this target exists
        if target not in self.X_train:
             print(f"Info: Skipping decision tree training because target '{target}' data not preprocessed.")
             return None, 0.0
             
        print(f"\nTraining decision tree model for prediction: {target}")
        
        # Adjust parameter grid (increase min_samples_split)
        param_grid = {
            'max_depth': [5, 7, 10],
            'min_samples_split': [20, 30, 40],
            'min_information_gain': [1e-5, 1e-4]
        }
        
        # Create parameter combinations list
        param_combinations = []
        for max_depth in param_grid['max_depth']:
            for min_samples_split in param_grid['min_samples_split']:
                for min_info_gain in param_grid['min_information_gain']:
                    param_combinations.append((max_depth, min_samples_split, min_info_gain))
        
        best_params = None
        best_score = -1 # Initialize to negative, so any valid score is better
        
        # Get data for current target
        X_train_target = self.X_train[target]
        y_train_target = self.y_train[target]
        X_test_target = self.X_test[target]
        y_test_target = self.y_test[target]
        feature_names_target = self.feature_names[target]
        
        # Evaluate each parameter combination
        results = []
        with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
             futures = []
             for params in param_combinations:
                 # Submit task to process pool
                 future = executor.submit(self._train_eval_dt_worker, params, X_train_target, y_train_target, X_test_target, y_test_target, feature_names_target)
                 futures.append(future)
             
             # Use tqdm to show progress and collect results
             for future in tqdm(futures, desc=f"{target} DT parameter search", total=len(param_combinations)):
                 try:
                     result = future.result() # Get result (containing score and parameters)
                     results.append(result)
                 except Exception as e:
                     print(f"Error: Decision tree training subprocess failed: {e}")
                     results.append({'score': -1, 'params': None}) # Add a failure marker
                     
        # Find best score and parameters from results
        if results:
            best_result = max(results, key=lambda x: x['score'])
            best_score = best_result['score']
            best_params = best_result['params']
        
        # If no best parameters found (e.g., all scores are -1 or training failed), use default values
        if best_params is None or best_score < 0:
             print(f"Warning: No valid best decision tree parameters found ({target}), will use default parameters.")
             best_params = {'max_depth': 5, 'min_samples_split': 30, 'min_information_gain': 1e-5}
             best_score = 0 # Reset score
             
        print(f"\nBest parameters ({target} DT): {best_params}")
        print(f"Test set best accuracy (parameter search phase): {best_score:.4f}")
        
        # Train final model using best parameters
        print("\nTraining final model with best parameters...")
        final_model = DecisionTree(
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_information_gain=best_params['min_information_gain']
        )
        try:
             final_model.fit(X_train_target, y_train_target, feature_names_target)
        except Exception as e:
             print(f"Error: Failed to train final decision tree model ({target}): {e}")
             return None, 0.0 # Training failed
             
        # Evaluate on test set
        try:
             y_pred = final_model.predict(X_test_target)
             test_acc = accuracy_score(y_test_target, y_pred)
             print(f"Final test set accuracy: {test_acc:.4f}")
             
             # Print classification report
             print("\nClassification report:")
             print(classification_report(y_test_target, y_pred, zero_division=0))
             
             # Print confusion matrix
             print("\nConfusion matrix:")
             cm = confusion_matrix(y_test_target, y_pred)
             print(cm)
             
             # Save model
             model_path = f"models/tree_{target}.pkl"
             final_model.save_model(model_path)
             
             # Export decision rules
             rules_path = f"models/tree_{target}_rules.json"
             final_model.export_rules(rules_path)
             
             self.models[f"tree_{target}"] = final_model
             return final_model, test_acc
             
        except Exception as e:
             print(f"Error: Failed to evaluate or save final decision tree model ({target}): {e}")
             return final_model, 0.0 # Model may be trained, but evaluation failed

    # Helper method for parallel decision tree training
    def _train_eval_dt_worker(self, params, X_train, y_train, X_test, y_test, feature_names):
         max_depth, min_samples_split, min_info_gain = params
         try:
             model = DecisionTree(
                 max_depth=max_depth,
                 min_samples_split=min_samples_split,
                 min_information_gain=min_info_gain
             )
             model.fit(X_train, y_train, feature_names)
             y_pred = model.predict(X_test)
             score = accuracy_score(y_test, y_pred)
             return {'score': score, 'params': {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_information_gain': min_info_gain}}
         except Exception as e:
             # print(f"DT Worker Error: {e} for params {params}") # Can uncomment for debugging
             return {'score': -1, 'params': params} # Return failure marker

    
    def train_random_forest(self, target):
        """Train random forest model"""
        # Check if data for this target exists
        if target not in self.X_train:
             print(f"Info: Skipping random forest training because target '{target}' data not preprocessed.")
             return None, 0.0
             
        print(f"\nTraining random forest model for prediction: {target}")
        
        # Adjust parameter grid
        param_grid = {
            'n_trees': [50, 70, 100], # Increase number of trees
            'max_depth': [7, 10, 15],
            'max_features': [7, 'sqrt', 'log2'] # Add log2 option
        }
        
        # Create parameter combinations list
        param_combinations = []
        for n_trees in param_grid['n_trees']:
            for max_depth in param_grid['max_depth']:
                for max_features_param in param_grid['max_features']: 
                    param_combinations.append((n_trees, max_depth, max_features_param))
        
        best_params = None
        best_score = -1
        
        # Get data for current target
        X_train_target = self.X_train[target]
        y_train_target = self.y_train[target]
        X_test_target = self.X_test[target]
        y_test_target = self.y_test[target]
        feature_names_target = self.feature_names[target]
        
        # Evaluate each parameter combination (in parallel)
        results = []
        with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
            futures = []
            for params in param_combinations:
                 future = executor.submit(self._train_eval_rf_worker, params, X_train_target, y_train_target, X_test_target, y_test_target, feature_names_target)
                 futures.append(future)
                 
            for future in tqdm(futures, desc=f"{target} RF parameter search", total=len(param_combinations)):
                 try:
                      result = future.result()
                      results.append(result)
                 except Exception as e:
                      print(f"Error: Random forest training subprocess failed: {e}")
                      results.append({'score': -1, 'params': None})
                      
        # Find best result
        if results:
            best_result = max(results, key=lambda x: x['score'])
            best_score = best_result['score']
            best_params = best_result['params']
                
        # If no best parameters found, use default values
        if best_params is None or best_score < 0:
             print(f"Warning: No valid best random forest parameters found ({target}), will use default parameters.")
             best_params = {'n_trees': 50, 'max_depth': 10, 'max_features': 'sqrt'}
             best_score = 0
        
        print(f"\nBest parameters ({target} RF): {best_params}")
        print(f"Test set best accuracy (parameter search phase): {best_score:.4f}")
        
        # Train final model with best parameters
        print("\nTraining final model with best parameters...")
        final_model = RandomForest(
            n_trees=best_params['n_trees'],
            max_depth=best_params['max_depth'],
            max_features=best_params['max_features']
        )
        try:
            final_model.fit(X_train_target, y_train_target, feature_names_target)
        except Exception as e:
            print(f"Error: Failed to train final random forest model ({target}): {e}")
            return None, 0.0
            
        # Evaluate on test set
        try:
            y_pred = final_model.predict(X_test_target)
            test_acc = accuracy_score(y_test_target, y_pred)
            print(f"Final test set accuracy: {test_acc:.4f}")
            
            # Print classification report
            print("\nClassification report:")
            print(classification_report(y_test_target, y_pred, zero_division=0))
            
            # Print confusion matrix
            print("\nConfusion matrix:")
            cm = confusion_matrix(y_test_target, y_pred)
            print(cm)
            
            # Save model
            model_path = f"models/forest_{target}.pkl"
            final_model.save_model(model_path)
            
            self.models[f"forest_{target}"] = final_model
            return final_model, test_acc
            
        except Exception as e:
             print(f"Error: Failed to evaluate or save final random forest model ({target}): {e}")
             return final_model, 0.0

    # Helper method for parallel random forest training
    def _train_eval_rf_worker(self, params, X_train, y_train, X_test, y_test, feature_names):
        n_trees, max_depth, max_features = params
        try:
            model = RandomForest(
                n_trees=n_trees,
                max_depth=max_depth,
                max_features=max_features
            )
            model.fit(X_train, y_train, feature_names)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            return {'score': score, 'params': {'n_trees': n_trees, 'max_depth': max_depth, 'max_features': max_features}}
        except Exception as e:
            # print(f"RF Worker Error: {e} for params {params}") # Can uncomment for debugging
            return {'score': -1, 'params': params} # Return failure marker

    
    def train_all_models(self):
        """Train models for all target variables"""
        self.results = {}
        
        start_time = time.time()
        
        # Preprocess data (includes feature selection)
        if not self.preprocess_data():
            print("Error: Data preprocessing failed, cannot continue training.")
            return None
            
        successful_targets = list(self.X_train.keys()) # Get list of successfully preprocessed targets
        print(f"\nWill train models for the following targets: {successful_targets}")
        
        for i, target in enumerate(successful_targets):
            target_start_time = time.time()
            print(f"\n{'='*50}")
            print(f"Starting training for {target} models ({i+1}/{len(successful_targets)})")
            print(f"{'='*50}")
            
            # Train decision tree
            tree_model, tree_acc = self.train_single_tree(target)
            
            # Train random forest
            forest_model, forest_acc = self.train_random_forest(target)
            
            # Only record results when model training was successful (model is not None)
            if tree_model is not None or forest_model is not None:
                self.results[target] = {
                    'tree': {
                        'model': tree_model,
                        'accuracy': tree_acc if tree_model else 0.0
                    },
                    'forest': {
                        'model': forest_model,
                        'accuracy': forest_acc if forest_model else 0.0
                    }
                }
                target_time = time.time() - target_start_time
                print(f"\n{target} training completed, time taken: {target_time:.2f} seconds")
            else:
                 print(f"Info: Both models failed for target '{target}', not recording results.")

        
        total_time = time.time() - start_time
        print(f"\nAll available models training completed, total time: {total_time:.2f} seconds")
        
        # Print summary
        print("\n\nResults Summary:")
        print(f"{'='*50}")
        if not self.results:
             print("No successfully trained models to summarize.")
             return None
             
        for target in self.results: # Only summarize successfully trained targets
            print(f"\nTarget: {target}")
            tree_accuracy = self.results[target]['tree']['accuracy']
            forest_accuracy = self.results[target]['forest']['accuracy']
            print(f"Decision tree accuracy: {tree_accuracy:.4f}")
            print(f"Random forest accuracy: {forest_accuracy:.4f}")
            
            # Determine best model
            if tree_accuracy == 0.0 and forest_accuracy == 0.0:
                 print("Best model: (both failed or accuracy is 0)")
            elif tree_accuracy >= forest_accuracy:
                 print(f"Best model: tree")
            else:
                 print(f"Best model: forest")
        
        return self.results
    
    def save_best_models(self):
        """Save best model configurations"""
        if not hasattr(self, 'results') or not self.results:
             print("Error: No training results to save best model configurations.")
             return None
             
        best_models = {}
        
        for target in self.results:
            tree_acc = self.results[target]['tree']['accuracy']
            forest_acc = self.results[target]['forest']['accuracy']
            
            # If both models failed, don't save best model for this target
            if tree_acc == 0.0 and forest_acc == 0.0:
                print(f"Info: Skipping best model save for '{target}' as both models failed or accuracy is 0.")
                continue
                
            # Choose model with higher accuracy
            best_type = 'tree' if tree_acc >= forest_acc else 'forest'
            best_acc = max(tree_acc, forest_acc)
            
            best_models[target] = {
                'type': best_type,
                'accuracy': best_acc,
                'features': self.feature_names.get(target, []) # Add list of features used
            }
        
        # Save best model configurations
        if best_models: # Ensure there are models to save
            with open('models/best_models.json', 'w') as f:
                json.dump(best_models, f, indent=2)
            print("\nBest model configurations saved to models/best_models.json")
        else:
            print("\nNo successful model configurations to save.")
            
        return best_models

def main():
    # Record overall start time
    total_start_time = time.time()
    
    # Create weather predictor
    predictor = RochesterWeatherPredictor()
    
    # Load data
    if not predictor.load_data():
        print("Cannot load data, please run data processing script first")
        return
    
    # Clean data
    predictor.clean_data()
    
    # Analyze feature importance
    predictor.analyze_feature_importance()
    
    # Train all models (preprocessing included)
    results = predictor.train_all_models()
    
    # Save best models
    if results:
        predictor.save_best_models()
    else:
        print("Due to training failure or no results, best models not saved.")

    
    # Print total time taken
    total_time = time.time() - total_start_time
    print(f"\nEntire process completed, total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()