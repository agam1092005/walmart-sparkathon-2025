from flask import Blueprint, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import os

data_breach_bp = Blueprint('data_breach_bp', __name__)

def scrape_xposedornot(domain):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get("https://xposedornot.com/breaches")
        time.sleep(2)  # Wait for page to load

        search_input = driver.find_element(By.CSS_SELECTOR, "input[type='search']")
        search_input.clear()
        search_input.send_keys(domain)
        time.sleep(2)  # Wait for table to update

        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        results = []
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if not cols or len(cols) < 7:
                continue
            result = {
                "breach_id": cols[0].text,
                "breach_date": cols[1].text,
                "domain": cols[2].text,
                "exposed_data": cols[3].text,
                "exposed_records": cols[4].text,
                "description": cols[5].text,
                "industry": cols[6].text,
            }
            results.append(result)
        return results
    finally:
        driver.quit()

def search_breach_csv(company_name):
    """Search for company breaches in the local breach.csv file"""
    try:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'datasets', 'breach.csv')
        
        df = pd.read_csv(csv_path)
        
        company_lower = company_name.lower()
        
        matches = df[df['Breach ID'].str.lower().str.contains(company_lower, na=False)]
        
        if matches.empty:
            return []
        
        results = []
        for _, row in matches.iterrows():
            result = {
                "breach_id": row['Breach ID'],
                "breach_date": row['Breach Date'],
                "domain": row['Domain'],
                "exposed_data": row['Exposed Data'],
                "exposed_records": row['Exposed Records'],
                "description": row['Description'],
                "industry": row['Industry'],
                "password_risk": row['Password Risk'],
                "searchable": row['Searchable'],
                "sensitive": row['Sensitive'],
                "verified": row['Verified']
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error reading breach.csv: {e}")
        return []

@data_breach_bp.route('/search', methods=['POST'])
def search_data_breach():
    data = request.get_json()
    domain = data.get('domain')
    
    if not domain:
        return jsonify({"error": "Missing 'domain' in request body"}), 400

    try:
        results = scrape_xposedornot(domain)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@data_breach_bp.route('/search_local', methods=['POST'])
def search_local_breach():
    """Search for company breaches in the local breach.csv file"""
    data = request.get_json()
    company_name = data.get('company_name')
    
    if not company_name:
        return jsonify({"error": "Missing 'company_name' in request body"}), 400

    try:
        results = search_breach_csv(company_name)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500