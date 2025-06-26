from flask import Blueprint, request, jsonify
from .models import upload_to_pocketbase, start_background_training, get_company_status, get_user_org_name_from_token

ml_bp = Blueprint('ml_bp', __name__)

@ml_bp.route('/upload', methods=['POST'])
def upload_csv():
    # Try to get token from cookie or Authorization header
    token = request.cookies.get('pb_token')
    if not token:
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ', 1)[1]
    if not token:
        return jsonify({'error': 'Authentication required'}), 401

    # Get org_name from user info
    org_name = get_user_org_name_from_token(token)
    if not org_name:
        return jsonify({'error': 'Could not determine company/org_name from user info'}), 400

    if 'dataset' not in request.files:
        return jsonify({'error': 'Missing dataset field'}), 400
    file = request.files['dataset']
    success, result, status = upload_to_pocketbase(org_name, file, token)
    if success:
        start_background_training(org_name)
        print(f"[INFO] Started training subprocess for {org_name}")
    return jsonify(result), status

@ml_bp.route('/api/train/<company>', methods=['POST'])
def train_company(company):
    start_background_training(company)
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

