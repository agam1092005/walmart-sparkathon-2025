from flask import Blueprint, request, jsonify
from .models import upload_to_pocketbase, start_background_training, get_company_status, get_user_org_name_from_token
import requests

ml_bp = Blueprint('ml_bp', __name__)
POCKETBASE_URL = "http://127.0.0.1:8090"

def get_num_clients_from_pocketbase():
    """Get the number of unique clients from the global collection in PocketBase"""

    res = requests.get(f"{POCKETBASE_URL}/api/collections/global/records")
    res.raise_for_status()
    items = res.json().get("items", [])
    if not items:
        return 0
    return items[0].get("clients")


@ml_bp.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "message": "ML service is running"}), 200

@ml_bp.route('/upload', methods=['POST'])
def upload_csv():
    token = request.cookies.get('pb_token')
    if not token:
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ', 1)[1]
    if not token:
        return jsonify({'error': 'Authentication required'}), 401

    org_name = get_user_org_name_from_token(token)
    if not org_name:
        return jsonify({'error': 'Could not determine company/org_name from user info'}), 400

    if 'dataset' not in request.files:
        return jsonify({'error': 'Missing dataset field'}), 400
    file = request.files['dataset']
    success, result, status = upload_to_pocketbase(org_name, file, token)
    if success:
        start_background_training(org_name, token)
        print(f"[INFO] Started training subprocess for {org_name}")
    return jsonify(result), status

@ml_bp.route('/api/train/<company>', methods=['POST'])
def train_company(company):
    token = request.cookies.get('pb_token')
    if not token:
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ', 1)[1]
    if not token:
        return jsonify({'error': 'Authentication required'}), 401
    start_background_training(company, token)
    return jsonify({
        "status": "started",
        "message": f"Training for {company} is running in background."
    })

@ml_bp.route('/api/status/<company>', methods=['GET'])
def status_company(company):
    result, error, status = get_company_status(company)
    if error:
        return jsonify(error), status
    return jsonify(result), 200

@ml_bp.route('/status', methods=['GET'])
def get_user_status():
    """Get status for the authenticated user's company"""
    token = request.cookies.get('pb_token')
    if not token:
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ', 1)[1]
    if not token:
        return jsonify({"msg": "No token found"}), 401
    
    org_name = get_user_org_name_from_token(token)
    if not org_name:
        return jsonify({"msg": "Invalid token"}), 401
    
    print(f"[DEBUG] Fetching status for org: {org_name}")
    status_result, error, status_code = get_company_status(org_name)
    
    if error:
        print(f"[DEBUG] Error fetching status: {error}, status_code: {status_code}")
        if status_code == 404:
            status_data = {
                "org_name": org_name,
                "submitted": False,
                "hasTrained": False,
                "encrypted": False,
                "detected": 0,
            }
            print(f"[DEBUG] Returning default status for new company: {status_data}")
            return jsonify(status_data), 200
        else:
            return jsonify({"msg": "Error fetching status", "error": error}), status_code
    
    def to_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == 'true'
        return False
    
    status_data = {
        "org_name": org_name,
        "submitted": to_bool(status_result.get('submitted', False)),
        "hasTrained": to_bool(status_result.get('hasTrained', False)),
        "encrypted": to_bool(status_result.get('encrypted', False)),
        "detected": int(status_result.get('detected')),
    }
    
    print(f"[DEBUG] Returning status data: {status_data}")
    return jsonify(status_data), 200

@ml_bp.route('/clients_count', methods=['GET'])
def get_clients_count():
    """Get the number of clients contributing to the global model"""
    try:
        num_clients = get_num_clients_from_pocketbase()
        return jsonify({
            "num_clients": num_clients,
            "status": "success"
        }), 200
    except Exception as e:
        return jsonify({
            "error": "Failed to fetch client count",
            "details": str(e)
        }), 500