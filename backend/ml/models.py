import requests
from werkzeug.utils import secure_filename
import subprocess
import sys
import os
import time
import tempfile
import pandas as pd

POCKETBASE_URL = 'http://127.0.0.1:8090'
DATA_COLLECTION = 'data'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_to_pocketbase(company, file, token=None):
    if not token:
        return False, {'error': 'Authentication required'}, 401
    if not company:
        return False, {'error': 'Company name is required'}, 400
    if file.filename == '':
        return False, {'error': 'No file selected'}, 400
    if not allowed_file(file.filename):
        return False, {'error': 'Only CSV files are allowed'}, 400

    # Preprocess CSV: drop NA in label, convert label to int
    try:
        df = pd.read_csv(file)
        if 'label_is_fraud' not in df.columns:
            return False, {'error': 'Missing label column: label_is_fraud'}, 400
        df = df.dropna(subset=['label_is_fraud'])
        df['label_is_fraud'] = df['label_is_fraud'].astype(int)
    except Exception as e:
        return False, {'error': f'Failed to read/clean CSV: {str(e)}'}, 400
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', newline='') as tmp_txt:
        df.to_csv(tmp_txt, index=False)
        txt_path = tmp_txt.name
    txt_filename = secure_filename(os.path.splitext(file.filename)[0] + '.txt')

    data = {
        'company': company,
        'submitted': 'true',
        'hasTrained': 'false',
        'encrypted': 'false'
    }
    files = {
        'dataset': (txt_filename, open(txt_path, 'rb'), 'text/plain')
    }
    headers = {'Authorization': f'Bearer {token}'}
    resp = requests.post(
        f"{POCKETBASE_URL}/api/collections/{DATA_COLLECTION}/records",
        data=data,
        files=files,
        headers=headers
    )
    os.remove(txt_path)
    if resp.status_code == 400 and 'already exists' in resp.text:
        return False, {'error': 'Company already exists'}, 409
    if not resp.ok:
        return False, {'error': 'PocketBase error', 'details': resp.text}, 500
    return True, {'success': True, 'record': resp.json()}, 200

def start_background_training(company, pb_token):
    script_path = os.path.join(os.path.dirname(__file__), 'client_runner.py')
    subprocess.Popen([sys.executable, script_path, company, pb_token])

def get_company_status(company):
    url = f"{POCKETBASE_URL}/api/collections/{DATA_COLLECTION}/records?filter=company='{company}'"
    resp = requests.get(url)
    if not resp.ok:
        return None, {'status': 'error', 'message': 'PocketBase error'}, 500
    items = resp.json().get('items', [])
    if not items:
        return None, {'status': 'error', 'message': 'Company not found.'}, 404
    record = items[0]
    result = {
        'company': record.get('company'),
        'submitted': record.get('submitted'),
        'hasTrained': record.get('hasTrained'),
        'encrypted': record.get('encrypted'),
    }
    if 'modelPath' in record:
        result['modelPath'] = record['modelPath']
    # If dataset file is present, add its URL
    if 'dataset' in record and record['dataset']:
        # PocketBase file URL: /api/files/<collectionId>/<recordId>/<filename>
        collection_id = record.get('collectionId', DATA_COLLECTION)
        record_id = record.get('id')
        filename = record['dataset']
        result['datasetUrl'] = f"{POCKETBASE_URL}/api/files/{collection_id}/{record_id}/{filename}"
    return result, None, 200

def get_user_org_name_from_token(token):
    headers = {'Authorization': f'Bearer {token}'}
    try:
        resp = requests.post(f'{POCKETBASE_URL}/api/collections/users/auth-refresh', headers=headers)
        if resp.ok:
            return resp.json().get('record', {}).get('org_name')
    except Exception:
        pass
    return None
