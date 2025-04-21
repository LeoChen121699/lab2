import os
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import glob

class CF6Parser:
    """Parse CF6 report data and extract useful features"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        
    def parse_cf6_file(self, file_path):
        """Parse a single CF6 file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract year and month from file name
            file_name = os.path.basename(file_path)
            file_match = re.search(r'(\d{4})_(\d{2})\.txt$', file_name)
            
            # If the file name matches the expected format, use year and month from file name
            file_year = None
            file_month = None
            if file_match:
                file_year = int(file_match.group(1))
                file_month = int(file_match.group(2))
                print(f"Year-month extracted from file name: {file_year}-{file_month}")
            
            # Extract month, year, and city information - improved regex to avoid capturing extra text
            month_match = re.search(r'MONTH:\s+(\w+)', content)
            year_match = re.search(r'YEAR:\s+(\d+)', content)
            station_match = re.search(r'STATION:\s+([\w\s]+?)(?:\n|$)', content)
            
            if not (month_match and station_match):
                print(f"Unable to extract basic information from file: {file_path}")
                return None
            
            month_name = month_match.group(1)
            
            # Prioritize year from file name, as year in file content might be incorrect
            if file_year is not None:
                year = file_year
            elif year_match:
                year = int(year_match.group(1))
                # Fix obviously incorrect future years
                current_year = datetime.now().year
                if year > current_year:
                    print(f"Warning: Year {year} in file content is a future year, trying to use file name or correct...")
                    if file_year is not None:
                        year = file_year
                    else:
                        print(f"Correcting future year {year} to current year {current_year}")
                        year = current_year
            else:
                print(f"Warning: Unable to get year information, skipping file: {file_path}")
                return None
            
            # Clean station name, remove extra spaces
            station = station_match.group(1).strip()
            
            # Convert month name to number
            month_map = {
                'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4,
                'MAY': 5, 'JUNE': 6, 'JULY': 7, 'AUGUST': 8,
                'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12
            }
            
            # Prioritize month from file name
            if file_month is not None:
                month = file_month
            else:
                month = month_map.get(month_name, None)
            
            if month is None:
                print(f"Unable to recognize month name: {month_name}")
                return None
            
            print(f"Using year-month: {year}-{month} for file: {file_path}")
            
            # More accurately extract data rows
            data_section = re.search(r'={80,}\n.*?\n={80,}\n(.*?)(?:={80,}|SM\s)', content, re.DOTALL)
            if not data_section:
                print(f"Unable to extract data section from file: {file_path}")
                return None
                
            data_text = data_section.group(1)
            data_lines = [line.strip() for line in data_text.split('\n') if line.strip() and not line.strip().startswith('SM')]
            
            # Parse data rows
            daily_data = []
            
            for line in data_lines:
                # Ensure line has at least one number
                if not re.search(r'\d', line):
                    continue
                    
                # Use more reliable method to split fields
                fields = line.split()
                if len(fields) < 13:  # Ensure at least basic fields exist
                    continue
                
                try:
                    # Ensure first field is an integer (date)
                    try:
                        day = int(fields[0])
                    except ValueError:
                        continue
                        
                    # Restrict day to reasonable range
                    if day < 1 or day > 31:
                        continue
                        
                    max_temp = self._parse_value(fields[1])
                    min_temp = self._parse_value(fields[2])
                    avg_temp = self._parse_value(fields[3])
                    departure = self._parse_value(fields[4])
                    
                    # Ensure temperature fields are in reasonable range
                    if (not pd.isna(max_temp) and (max_temp < -100 or max_temp > 150)) or \
                       (not pd.isna(min_temp) and (min_temp < -100 or min_temp > 150)) or \
                       (not pd.isna(avg_temp) and (avg_temp < -100 or avg_temp > 150)):
                        continue
                    
                    hdd = self._parse_value(fields[5])
                    cdd = self._parse_value(fields[6])
                    precipitation = self._parse_value(fields[7])
                    snow = self._parse_value(fields[8])
                    snow_depth = self._parse_value(fields[9])
                    avg_wind_speed = self._parse_value(fields[10])
                    max_wind_speed = self._parse_value(fields[11])
                    max_wind_dir = self._parse_value(fields[12])
                    
                    # Parse weather description field (WX)
                    wx_idx = 16 if len(fields) >= 17 else None
                    weather_codes = fields[wx_idx] if wx_idx and wx_idx < len(fields) else ""
                    
                    # Create date
                    try:
                        date = datetime(year, month, day)
                    except ValueError:
                        print(f"Invalid date: {year}-{month}-{day} in file: {file_path}")
                        continue
                    
                    daily_data.append({
                        'date': date,
                        'city': station,
                        'max_temp': max_temp,
                        'min_temp': min_temp,
                        'avg_temp': avg_temp,
                        'departure': departure,
                        'hdd': hdd,
                        'cdd': cdd,
                        'precipitation': precipitation,
                        'snow': snow,
                        'snow_depth': snow_depth,
                        'avg_wind_speed': avg_wind_speed,
                        'max_wind_speed': max_wind_speed,
                        'max_wind_dir': max_wind_dir,
                        'weather_codes': weather_codes
                    })
                except Exception as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {str(e)}")
            
            return daily_data
        
        except Exception as e:
            print(f"Error processing file: {file_path}")
            print(f"Error: {str(e)}")
            return None
    
    def _parse_value(self, value_str):
        """Parse value, handling special values like T (trace) and M (missing)"""
        if value_str == 'T':
            return 0.01  # Set trace amount to 0.01
        elif value_str == 'M':
            return np.nan  # Set missing value to NaN
        else:
            try:
                return float(value_str)
            except ValueError:
                return np.nan
    
    def parse_city_data(self, city_code):
        """Parse all data files for a specific city"""
        city_dir = os.path.join(self.data_dir, city_code)
        if not os.path.exists(city_dir):
            print(f"City directory does not exist: {city_dir}")
            return None
        
        all_data = []
        files = glob.glob(os.path.join(city_dir, "*.txt"))
        
        for file_path in sorted(files):
            print(f"Parsing file: {file_path}")
            file_data = self.parse_cf6_file(file_path)
            if file_data:
                all_data.extend(file_data)
        
        if all_data:
            df = pd.DataFrame(all_data)
            df = df.sort_values('date')
            return df
        
        return None
    
    def parse_all_cities(self):
        """Parse data for all cities"""
        city_dirs = [d for d in os.listdir(self.data_dir) 
                   if os.path.isdir(os.path.join(self.data_dir, d))]
        
        all_city_data = {}
        
        for city_code in city_dirs:
            print(f"Processing city: {city_code}")
            city_df = self.parse_city_data(city_code)
            if city_df is not None:
                all_city_data[city_code] = city_df
        
        return all_city_data

class FeatureExtractor:
    """Extract features from parsed CF6 data"""
    
    def __init__(self, city_data_dict):
        self.city_data = city_data_dict
        self.combined_data = None
        
    def combine_city_data(self):
        """Combine data from all cities"""
        combined_dfs = []
        
        for city_code, df in self.city_data.items():
            if df is None or df.empty:
                continue
                
            df = df.copy()
            # Ensure city name is cleaned
            df['city'] = df['city'].str.strip()
            # Add city code
            df['city_code'] = city_code
            # Remove potential duplicate rows
            df = df.drop_duplicates(subset=['date', 'city_code'])
            combined_dfs.append(df)
        
        if combined_dfs:
            self.combined_data = pd.concat(combined_dfs)
            # Ensure sorted by date and city code and remove potential duplicate data
            self.combined_data = self.combined_data.sort_values(['date', 'city_code']).reset_index(drop=True)
            return self.combined_data
        
        return None
    
    def add_basic_features(self, df):
        """Add basic features"""
        df = df.copy()
        
        # Add date-related features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['season'] = df['month'].apply(self._get_season)
        
        # Precipitation features
        df['has_precipitation'] = df['precipitation'] > 0.01
        df['has_snow'] = df['snow'] > 0.1
        
        # Temperature related features
        df['temp_range'] = df['max_temp'] - df['min_temp']
        
        # Wind direction categorization
        df['wind_direction_category'] = df['max_wind_dir'].apply(self._categorize_wind_direction)
        
        # Weather code features
        df['has_rain'] = df['weather_codes'].apply(lambda x: '1' in str(x))
        df['has_fog'] = df['weather_codes'].apply(lambda x: '2' in str(x))
        df['has_thunder'] = df['weather_codes'].apply(lambda x: '3' in str(x))
        df['has_ice'] = df['weather_codes'].apply(lambda x: '4' in str(x))
        df['has_haze'] = df['weather_codes'].apply(lambda x: '5' in str(x))
        df['has_dust'] = df['weather_codes'].apply(lambda x: '6' in str(x))
        df['has_smoke'] = df['weather_codes'].apply(lambda x: '7' in str(x))
        df['has_blowing'] = df['weather_codes'].apply(lambda x: '8' in str(x))
        df['has_tornado'] = df['weather_codes'].apply(lambda x: '9' in str(x))
        
        return df
    
    def _get_season(self, month):
        """Determine season based on month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _categorize_wind_direction(self, direction):
        """Categorize wind direction angle"""
        if pd.isna(direction):
            return 'unknown'
        
        try:
            direction = float(direction)
            if 337.5 <= direction or direction < 22.5:
                return 'north'
            elif 22.5 <= direction < 67.5:
                return 'northeast'
            elif 67.5 <= direction < 112.5:
                return 'east'
            elif 112.5 <= direction < 157.5:
                return 'southeast'
            elif 157.5 <= direction < 202.5:
                return 'south'
            elif 202.5 <= direction < 247.5:
                return 'southwest'
            elif 247.5 <= direction < 292.5:
                return 'west'
            elif 292.5 <= direction < 337.5:
                return 'northwest'
            else:
                return 'unknown'
        except:
            return 'unknown'
    
    def add_historical_features(self, df, days=5):
        """Add historical features (data from past few days)"""
        df = df.copy()
        
        # Group by city, then add historical features
        grouped = df.groupby('city_code')
        dfs = []
        
        for city, city_data in grouped:
            city_df = city_data.copy().sort_values('date')
            
            # Add historical data for each feature of interest
            features_to_shift = [
                'max_temp', 'min_temp', 'avg_temp', 'precipitation', 
                'snow', 'avg_wind_speed', 'max_wind_speed'
            ]
            
            # Create a new dictionary to store all historical features
            history_features = {}
            
            # Step 1: Calculate historical values for each feature (shift operation)
            for feature in features_to_shift:
                for day in range(1, days + 1):
                    history_features[f'{feature}_day{day}'] = city_df[feature].shift(day)
            
            # Step 2: Calculate statistical features (using already calculated historical values)
            for feature in features_to_shift:
                # Average, max, and min for past days
                for day in range(2, days + 1):
                    # Create temporary DataFrame with columns needed for statistics
                    temp_data = {}
                    for i in range(1, day + 1):
                        temp_data[f'day{i}'] = history_features[f'{feature}_day{i}']
                    
                    # Convert temporary data to DataFrame
                    temp_df = pd.DataFrame(temp_data, index=city_df.index)
                    
                    # Calculate statistics
                    history_features[f'{feature}_last{day}d_avg'] = temp_df.mean(axis=1)
                    history_features[f'{feature}_last{day}d_max'] = temp_df.max(axis=1)
                    history_features[f'{feature}_last{day}d_min'] = temp_df.min(axis=1)
            
            # Add previous day's temperature change
            if 'avg_temp_day1' in history_features and 'avg_temp_day2' in history_features:
                history_features['temp_change_day1'] = history_features['avg_temp_day1'] - history_features['avg_temp_day2']
            else:
                history_features['temp_change_day1'] = np.nan
            
            # Add historical precipitation indicators
            history_features['days_since_rain'] = self._compute_days_since(city_df, 'precipitation', threshold=0.01)
            history_features['days_since_snow'] = self._compute_days_since(city_df, 'snow', threshold=0.1)
            
            # Add all historical features to DataFrame at once
            history_df = pd.DataFrame(history_features, index=city_df.index)
            city_df = pd.concat([city_df, history_df], axis=1)
            
            dfs.append(city_df)
        
        if dfs:
            result = pd.concat(dfs)
            return result.sort_values(['date', 'city_code'])
        
        return df
    
    def _compute_days_since(self, df, column, threshold=0):
        """Calculate days since an event (like rainfall)"""
        result = []
        count = 0
        
        for i, row in df.iterrows():
            if pd.isna(row[column]) or row[column] <= threshold:
                count += 1
            else:
                count = 0
            result.append(count)
        
        return pd.Series(result, index=df.index)
    
    def prepare_data(self):
        """Prepare complete feature dataset"""
        if self.combined_data is None:
            self.combine_city_data()
            
        if self.combined_data is None or self.combined_data.empty:
            return None
        
        # Data quality checks
        # Check if dates are valid
        self.combined_data = self.combined_data[pd.notnull(self.combined_data['date'])]
        
        # Check if temperature values are in reasonable range
        temp_mask = (
            (pd.isna(self.combined_data['max_temp']) | ((self.combined_data['max_temp'] >= -100) & (self.combined_data['max_temp'] <= 150))) &
            (pd.isna(self.combined_data['min_temp']) | ((self.combined_data['min_temp'] >= -100) & (self.combined_data['min_temp'] <= 150))) &
            (pd.isna(self.combined_data['avg_temp']) | ((self.combined_data['avg_temp'] >= -100) & (self.combined_data['avg_temp'] <= 150)))
        )
        self.combined_data = self.combined_data[temp_mask]
        
        # Ensure max_temp >= min_temp
        valid_temp_mask = pd.isna(self.combined_data['max_temp']) | pd.isna(self.combined_data['min_temp']) | (self.combined_data['max_temp'] >= self.combined_data['min_temp'])
        self.combined_data = self.combined_data[valid_temp_mask]
        
        # Add basic features
        print("Adding basic features...")
        featured_data = self.add_basic_features(self.combined_data)
        
        # Add historical features
        print("Adding historical features...")
        featured_data = self.add_historical_features(featured_data)
        
        # Add data quality metric
        featured_data['data_quality'] = featured_data.count(axis=1) / featured_data.shape[1]
        
        # Remove rows with too many missing values
        min_data_quality = 0.8  # At least 80% of data must be valid
        featured_data = featured_data[featured_data['data_quality'] >= min_data_quality]
        featured_data = featured_data.drop('data_quality', axis=1)
        
        print(f"Processed data shape: {featured_data.shape}")
        
        return featured_data
    
    def prepare_target_variables(self, df, target_city='ROC'):
        """Prepare target variables: temperature higher than yesterday, temperature higher than average, has precipitation"""
        df = df.copy()
        
        # Only select data for target city
        target_df = df[df['city_code'] == target_city].copy()
        
        if target_df.empty:
            print(f"No data found for target city {target_city}")
            return None
        
        # Ensure all fields required for target variable calculation exist
        required_fields = ['avg_temp', 'avg_temp_day1', 'departure', 'precipitation']
        missing_fields = [field for field in required_fields if field not in target_df.columns]
        
        if missing_fields:
            print(f"Warning: Missing fields required for target variable calculation: {missing_fields}")
            print("Attempting to continue with available data...")
        
        # 1. Is today's temperature higher than yesterday
        if 'avg_temp' in target_df.columns and 'avg_temp_day1' in target_df.columns:
            target_df['temp_higher_than_yesterday'] = target_df['avg_temp'] > target_df['avg_temp_day1']
        else:
            print("Warning: Cannot calculate 'temp higher than yesterday' target variable, missing required fields")
            target_df['temp_higher_than_yesterday'] = np.nan
        
        # 2. Is today's average temperature higher than the normal level
        if 'avg_temp' in target_df.columns and 'departure' in target_df.columns:
            # Use departure (deviation from normal) to determine if higher than average
            target_df['temp_higher_than_average'] = target_df['departure'] > 0
        else:
            print("Warning: Cannot calculate 'temp higher than average' target variable, missing required fields")
            target_df['temp_higher_than_average'] = np.nan
        
        # 3. Is there more than trace precipitation today
        if 'precipitation' in target_df.columns:
            target_df['has_more_than_trace_precip'] = target_df['precipitation'] > 0.01
        else:
            print("Warning: Cannot calculate 'has more than trace precipitation' target variable, missing required fields")
            target_df['has_more_than_trace_precip'] = np.nan
        
        # Remove rows with missing target variables
        valid_target_rows = (
            target_df['temp_higher_than_yesterday'].notna() | 
            target_df['temp_higher_than_average'].notna() | 
            target_df['has_more_than_trace_precip'].notna()
        )
        
        target_df = target_df[valid_target_rows]
        
        if target_df.empty:
            print("Warning: All data empty after target variable calculation")
            return None
            
        return target_df
    
    def save_processed_data(self, output_file='data/processed_data.csv'):
        """Save processed data"""
        processed_data = self.prepare_data()
        
        if processed_data is not None and not processed_data.empty:
            # Ensure date format is correct
            processed_data['date'] = pd.to_datetime(processed_data['date']).dt.strftime('%Y-%m-%d')
            
            # Check and handle text columns that might contain commas or line breaks
            text_columns = processed_data.select_dtypes(include=['object']).columns
            for col in text_columns:
                processed_data[col] = processed_data[col].astype(str).str.replace(',', ';').str.replace('\n', ' ')
            
            # Save CSV file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            processed_data.to_csv(output_file, index=False, quoting=1)  # quoting=1 uses quotes around non-numeric fields
            print(f"Processed data saved to {output_file}")
            
            # Additionally save statistics
            stats_file = output_file.replace('.csv', '_stats.txt')
            with open(stats_file, 'w') as f:
                f.write(f"Data shape: {processed_data.shape}\n")
                f.write(f"Total records: {len(processed_data)}\n")
                f.write(f"Number of cities: {processed_data['city_code'].nunique()}\n")
                f.write(f"Date range: {processed_data['date'].min()} to {processed_data['date'].max()}\n")
                f.write("\nData sample:\n")
                f.write(str(processed_data.head(3)))
                f.write("\n\nColumn statistics:\n")
                for col in processed_data.columns:
                    missing = processed_data[col].isna().sum()
                    pct_missing = missing / len(processed_data) * 100
                    f.write(f"{col}: {missing} missing values ({pct_missing:.1f}%)\n")
            
            return output_file
        
        print("Processed data is empty, no file saved")
        return None

def main():
    # Parse CF6 data
    parser = CF6Parser()
    city_data = parser.parse_all_cities()
    
    if not city_data:
        print("No data found, please run data_collector.py to collect data first")
        return
    
    # Extract features
    extractor = FeatureExtractor(city_data)
    processed_file = extractor.save_processed_data()
    
    if processed_file:
        try:
            # Prepare target variables
            processed_data = pd.read_csv(processed_file)
            processed_data['date'] = pd.to_datetime(processed_data['date'])
            
            target_data = extractor.prepare_target_variables(processed_data)
            
            if target_data is not None and not target_data.empty:
                # Save Rochester data with target variables
                target_file = 'data/rochester_with_targets.csv'
                
                # Ensure date format is correct
                target_data['date'] = target_data['date'].dt.strftime('%Y-%m-%d')
                
                # Check and handle text columns that might contain commas or line breaks
                text_columns = target_data.select_dtypes(include=['object']).columns
                for col in text_columns:
                    target_data[col] = target_data[col].astype(str).str.replace(',', ';').str.replace('\n', ' ')
                
                # Save CSV file
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                target_data.to_csv(target_file, index=False, quoting=1)
                print(f"Rochester data with target variables saved to {target_file}")
                print(f"Shape: {target_data.shape}")
                
                # Create a simple data validation report
                print("\nData Validation Report:")
                print(f"Total records: {len(target_data)}")
                print(f"Total predictions - Temperature higher than yesterday: {target_data['temp_higher_than_yesterday'].sum()} ({target_data['temp_higher_than_yesterday'].mean()*100:.1f}%)")
                print(f"Total predictions - Temperature higher than average: {target_data['temp_higher_than_average'].sum()} ({target_data['temp_higher_than_average'].mean()*100:.1f}%)")
                print(f"Total predictions - Has precipitation: {target_data['has_more_than_trace_precip'].sum()} ({target_data['has_more_than_trace_precip'].mean()*100:.1f}%)")
            else:
                print("Unable to generate target variable data")
        except Exception as e:
            print(f"Error processing target variables: {str(e)}")

if __name__ == "__main__":
    main()