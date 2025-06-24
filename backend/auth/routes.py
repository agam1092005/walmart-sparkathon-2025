from flask import Blueprint, request, jsonify
from .models import signup_user, login_user

auth_bp = Blueprint('auth_bp', __name__)

@auth_bp.route('/signup', methods=['POST'])
def signup():
    org_name = request.json.get('org_name')
    email = request.json.get('email')
    password = request.json.get('password')

    if not email or not password:
        return jsonify({"msg": "Email and password are required"}), 400

    result = signup_user(org_name, email, password)
    if not result:
        return jsonify({"msg": "Failed to create user"}), 500

    return jsonify({"msg": "User created successfully", "user": result}), 201

@auth_bp.route('/signin', methods=['POST'])
def signin():
    email = request.json.get('email')
    password = request.json.get('password')

    if not email or not password:
        return jsonify({"msg": "Email and password are required"}), 400

    auth_result = login_user(email, password)
    if not auth_result:
        return jsonify({"msg": "Invalid email or password"}), 401

    return jsonify({
        "msg": "Login successful",
        "token": auth_result.get('token'),
        "user": auth_result.get('record')
    }), 200
