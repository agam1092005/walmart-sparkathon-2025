from flask import Blueprint, request, jsonify, make_response
from .models import signup_user, login_user, get_company_data_by_token
from flask_jwt_extended import create_access_token

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

    token = auth_result.get('token')
    user = auth_result.get('record')
    resp = make_response(jsonify({
        "msg": "Login successful",
        "token": token,
        "user": user
    }), 200)
    # Set the token as a cookie (HttpOnly, Secure)
    resp.set_cookie('pb_token', token, httponly=False, samesite='Lax')
    return resp

@auth_bp.route('/logout', methods=['POST'])
def logout():
    resp = make_response(jsonify({"msg": "Logged out"}))
    resp.set_cookie('pb_token', '', expires=0)
    return resp


