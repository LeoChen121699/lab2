import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from decision_tree import DecisionTree, RandomForest

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def parse_cf6_row(row_data):
    """
    Parse simplified CF6 data row, handling special values like T (trace) and M (missing)
    
    Parameters:
        row_data (str): Simplified CF6 data row with space-separated values
    
    Returns:
        dict: Dictionary containing parsed data
    """
    fields = row_data.split()
    if len(fields) < 13:
        logger.error(f"Data row has insufficient fields: {row_data}")
        return None
    
    try:
        result = {
            'day': float(fields[0]),
            'max_temp': _parse_value(fields[1]),
            'min_temp': _parse_value(fields[2]),
            'avg_temp': _parse_value(fields[3]),
            'departure': _parse_value(fields[4]),
            'hdd': _parse_value(fields[5]),
            'cdd': _parse_value(fields[6]),
            'precipitation': _parse_value(fields[7]),
            'snow': _parse_value(fields[8]),
            'snow_depth': _parse_value(fields[9]),
            'avg_wind_speed': _parse_value(fields[10]),
            'max_wind_speed': _parse_value(fields[11]),
            'max_wind_dir': _parse_value(fields[12])
        }
        
        # Add weather codes (if present)
        if len(fields) >= 17:
            result['weather_codes'] = fields[16]
        else:
            result['weather_codes'] = ""
        
        return result
    except Exception as e:
        logger.error(f"Error parsing row: {row_data}")
        logger.error(f"Error: {str(e)}")
        return None

def _parse_value(value_str):
    """
    Parse value, handling special values like T (trace) and M (missing)
    
    Parameters:
        value_str (str): Value to parse
    
    Returns:
        float: Parsed float value
    """
    if value_str == 'T':
        return 0.01  # Set trace amount to 0.01
    elif value_str == 'M':
        return np.nan  # Set missing value to NaN
    else:
        try:
            return float(value_str)
        except ValueError:
            return np.nan

def load_model(model_path):
    """
    Load model
    
    Parameters:
        model_path (str): Model file path
    
    Returns:
        object: Loaded model object
    """
    logger.info(f"Attempting to load model: {model_path}")
    try:
        # Determine if it's a decision tree or random forest model based on filename
        if 'tree_' in model_path.lower():
            model = DecisionTree.load_model(model_path)
        elif 'forest_' in model_path.lower():
            model = RandomForest.load_model(model_path)
        else:
            logger.error(f"Unable to determine model type: {model_path}")
            return None
        
        logger.info(f"Successfully loaded model: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        return None

def extract_features(day_data_list):
    """
    Extract features from multiple days of data
    
    Parameters:
        day_data_list (list): List containing parsed data for multiple days
    
    Returns:
        dict: Dictionary containing extracted features
    """
    features = {}
    
    # Ensure there's enough data
    if len(day_data_list) < 1:
        return None
    
    # Add basic features
    for i, day_data in enumerate(day_data_list, 1):
        day_key = f'day{i}'
        for field in ['max_temp', 'min_temp', 'avg_temp', 'precipitation', 
                     'snow', 'avg_wind_speed', 'max_wind_speed', 'max_wind_dir']:
            if field in day_data:
                features[f'{field}_{day_key}'] = day_data[field]
    
    # Add derived features
    
    # Temperature change
    if 'avg_temp_day1' in features and 'avg_temp_day2' in features:
        features['temp_change'] = features['avg_temp_day1'] - features['avg_temp_day2']
    
    # Determine if there's precipitation
    if 'precipitation_day1' in features:
        features['has_precipitation'] = features['precipitation_day1'] > 0.01
    
    return features

def prepare_prediction_data(features):
    """
    Prepare feature data for prediction
    
    Parameters:
        features (dict): Extracted features
    
    Returns:
        DataFrame: Prepared prediction data
    """
    # Create DataFrame with all features
    df = pd.DataFrame([features])
    
    # Fill missing values
    df = df.fillna(0)
    
    # Convert boolean values to integers
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    
    return df

def predict(modeltype, day5, day4, day3, day2, day1):
    """
    Predict weather
    
    Parameters:
        modeltype: 'besttree' or 'bestforest'
        day5 - day1: CF6 data for the past 5 days, each day is a dictionary with city code as key and CF6 record as value
    
    Returns:
        List containing three boolean values representing predictions for three questions
    """
    try:
        logger.info(f"Using model {modeltype} for prediction")
        
        # Ensure models directory exists
        if not os.path.exists("models"):
            os.makedirs("models", exist_ok=True)
            logger.warning("Warning: models directory doesn't exist, created it")
        
        # Build model paths
        model_prefix = "tree" if modeltype == "besttree" else "forest"
        model_paths = {
            'temp_higher_than_yesterday': f"models/{model_prefix}_temp_higher_than_yesterday.pkl",
            'temp_higher_than_average': f"models/{model_prefix}_temp_higher_than_average.pkl",
            'has_more_than_trace_precip': f"models/{model_prefix}_has_more_than_trace_precip.pkl"
        }
        
        # Process input data
        city_data = {}
        
        # Only focus on ROC city data
        city_code = 'ROC'
        if city_code in day1:
            city_day_data = []
            
            # Collect 5 days of data
            for day_data in [day5, day4, day3, day2, day1]:
                if city_code in day_data:
                    parsed_data = parse_cf6_row(day_data[city_code])
                    if parsed_data:
                        city_day_data.append(parsed_data)
            
            if city_day_data:
                # Extract features
                features = extract_features(city_day_data)
                
                if features:
                    # Prepare prediction data
                    prediction_data = prepare_prediction_data(features)
                    
                    # Try to load models and make predictions
                    predictions = []
                    
                    for target, model_path in model_paths.items():
                        # Try to load model
                        model = load_model(model_path)
                        
                        if model is not None:
                            # Use model to make prediction
                            try:
                                # Ensure data structure meets model requirements
                                X = prediction_data.values
                                prediction = model.predict(X)[0]
                                predictions.append(bool(prediction))
                                logger.info(f"Model prediction for {target}: {bool(prediction)}")
                            except Exception as e:
                                logger.error(f"Model prediction failed: {e}")
                                # If prediction fails, use rule-based prediction
                                prediction = _rule_based_prediction(features, target)
                                predictions.append(prediction)
                        else:
                            logger.error(f"Unable to load model: {model_path}")
                            # If model is unavailable, use rule-based prediction
                            prediction = _rule_based_prediction(features, target)
                            predictions.append(prediction)
                    
                    return predictions
        
        # If valid data can't be extracted or target city not found, return default values
        return [False, False, False]
    
    except Exception as e:
        logger.error(f"Error during prediction process: {str(e)}")
        return [False, False, False]  # Default return values

def _rule_based_prediction(features, target):
    """
    Use rule-based prediction when model is unavailable
    
    Parameters:
        features (dict): Extracted features
        target (str): Target variable name
    
    Returns:
        bool: Prediction result
    """
    logger.info(f"Using rule-based prediction for {target}")
    if target == 'temp_higher_than_yesterday':
        return features.get('avg_temp_day1', 0) > features.get('avg_temp_day2', 0)
    elif target == 'temp_higher_than_average':
        # Calculate 5-day average temperature
        temps = []
        for i in range(1, 6):
            temp = features.get(f'avg_temp_day{i}')
            if temp is not None and not np.isnan(temp):
                temps.append(temp)
        avg_temp = sum(temps) / len(temps) if temps else 0
        return features.get('avg_temp_day1', 0) > avg_temp
    elif target == 'has_more_than_trace_precip':
        return features.get('precipitation_day1', 0) > 0.01
    else:
        return False

# Example usage
if __name__ == "__main__":
    # This is a simplified example
    sample_data = {
        "ROC": "14 25 19 22 -5 43 0 T 0.1 3 14.0 26 260 M M 8 34 270"
    }
    
    # Test parsing functionality
    print("Parse result:", parse_cf6_row(sample_data["ROC"]))
    
    # Test prediction functionality
    predictions = predict("besttree", sample_data, sample_data, sample_data, sample_data, sample_data)
    print("Prediction results:", predictions) 