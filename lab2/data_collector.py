import os
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import time
import random
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Create data directory
os.makedirs("data", exist_ok=True)

# Target city list (including Rochester and cities to its west)
# The following are sample city codes, which can be added or modified as needed
CITIES = {
    "ROC": "Rochester",  # Rochester
    "BUF": "Buffalo",    # Buffalo
    "CLE": "Cleveland",  # Cleveland
    "DTW": "Detroit",    # Detroit
    # "MSP": "Minneapolis",# Minneapolis
    # "MKE": "Milwaukee",  # Milwaukee
    # "ALB": "Albany",     # Albany
    # "BOS": "Boston",     # Boston
    # "BTV": "Burlington", # Burlington
    # "BGM": "Binghamton", # Binghamton
    # "ERI": "Erie",       # Erie
    # "PIT": "Pittsburgh", # Pittsburgh
    # "SYR": "Syracuse",   # Syracuse
    # "IAD": "Washington", # Washington
}

# Configuration parameters
MAX_WORKERS = 8  # Number of parallel threads
MAX_RETRIES = 3  # Maximum retry attempts
RETRY_BACKOFF_FACTOR = 0.5  # Retry interval factor
RETRY_STATUS_FORCELIST = [500, 502, 503, 504]  # HTTP status codes that need retry
MAX_VERSIONS = 50  # Maximum number of versions to retrieve for each city

def create_session():
    """Create a session with retry mechanism"""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_FORCELIST,
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_cf6_url(city_code, version):
    """Generate CF6 report URL using version number instead of year/month"""
    # For ROC (Rochester), use BUF as the site parameter
    site = "BUF" if city_code == "ROC" else "NWS"
    return f"https://forecast.weather.gov/product.php?site={site}&issuedby={city_code}&product=CF6&format=txt&version={version}&glossary=0"

def extract_cf6_data(html_content, city_code):
    """Extract CF6 data from HTML page"""
    soup = BeautifulSoup(html_content, 'html.parser')
    pre_tags = soup.find_all('pre')
    
    if not pre_tags:
        return None
    
    # CF6 data is typically in the first <pre> tag
    cf6_text = pre_tags[0].get_text()
    
    # Extract valid CF6 section
    if "PRELIMINARY LOCAL CLIMATOLOGICAL DATA" in cf6_text and city_code in cf6_text:
        return cf6_text
    
    return None

def extract_year_month(cf6_data):
    """Extract year and month information from CF6 data"""
    if cf6_data is None:
        return None, None
    
    month_match = re.search(r'MONTH:\s+(\w+)', cf6_data)
    year_match = re.search(r'YEAR:\s+(\d+)', cf6_data)
    
    if month_match and year_match:
        month_name = month_match.group(1)
        year = int(year_match.group(1))
        
        # Convert month name to number
        month_map = {
            'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4,
            'MAY': 5, 'JUNE': 6, 'JULY': 7, 'AUGUST': 8,
            'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12
        }
        
        month = month_map.get(month_name.upper())
        if month:
            return year, month
    
    return None, None

def save_cf6_data(city_code, year, month, cf6_data):
    """Save CF6 data to file"""
    if year is None or month is None:
        print(f"  Unable to determine year or month, skipping save")
        return None
    
    directory = f"data/{city_code}"
    os.makedirs(directory, exist_ok=True)
    
    filename = f"{directory}/{year}_{month:02d}.txt"
    with open(filename, 'w') as f:
        f.write(cf6_data)
    
    return filename

def fetch_city_version_data(city_code, version, session):
    """Fetch data for specific city and version"""
    try:
        city_name = CITIES.get(city_code, city_code)
        print(f"Collecting data for {city_name} ({city_code}): Version {version}")
        
        url = get_cf6_url(city_code, version)
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            cf6_data = extract_cf6_data(response.text, city_code)
            if cf6_data:
                year, month = extract_year_month(cf6_data)
                if year and month:
                    filename = save_cf6_data(city_code, year, month, cf6_data)
                    if filename:
                        print(f"  Saved to {filename} (Year: {year}, Month: {month})")
                        return True
                    else:
                        print(f"  Failed to save file")
                else:
                    print(f"  Unable to extract year and month from content")
            else:
                print(f"  Failed to extract CF6 data")
        else:
            print(f"  Request failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"  {city_code} Version {version} - Error occurred: {str(e)}")
    
    return False

def collect_data():
    """Collect CF6 data for all cities"""
    # Generate all (city, version) combinations to fetch
    tasks = []
    for city_code in CITIES:
        for version in range(1, MAX_VERSIONS + 1):
            tasks.append((city_code, version))
    
    # Create session
    session = create_session()
    
    # Use thread pool to fetch data in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(fetch_city_version_data, city_code, version, session): (city_code, version)
            for city_code, version in tasks
        }
        
        # Process results
        completed_count = 0
        success_count = 0
        for future in concurrent.futures.as_completed(future_to_task):
            city_code, version = future_to_task[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
            except Exception as e:
                print(f"Task exception: {city_code} Version {version} - {str(e)}")
            
            completed_count += 1
            # Simple progress report
            print(f"Progress: {completed_count}/{len(tasks)} completed ({completed_count/len(tasks)*100:.1f}%), Successful: {success_count}")

def main():
    print(f"Starting CF6 data collection using version number method")
    print(f"Using {MAX_WORKERS} parallel threads, maximum retry attempts: {MAX_RETRIES}")
    print(f"City list: {', '.join(CITIES.values())}")
    print(f"Maximum versions to fetch per city: {MAX_VERSIONS}")
    
    collect_data()
    print("Data collection complete!")

if __name__ == "__main__":
    main() 