from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token, jwt_required, unset_jwt_cookies, \
    unset_access_cookies
from .models import add_user, get_user, update_password

auth_bp = Blueprint('auth_bp', __name__)

@auth_bp.route('/signup', methods=['POST'])
def signup():
    org_name = request.json.get('org_name')
    email = request.json.get('email')
    password = request.json.get('password')

    if not email or not password:
        return jsonify({"msg": "Email and password are required"}), 400

    if get_user(email):
        return jsonify({"msg": "User already exists"}), 400

    hashed_password = generate_password_hash(password)
    add_user(org_name, email, hashed_password)

    access_token = create_access_token(identity=email)
    response = jsonify({"msg": "User created successfully"})
    response.set_cookie('jwt', access_token)

    return response, 201


@auth_bp.route('/signin', methods=['POST'])
def signin():
    email = request.json.get('email')
    password = request.json.get('password')

    if not email or not password:
        return jsonify({"msg": "Email and password are required"}), 400

    user = get_user(email)

    if user['provider'] != "email":
        return jsonify({
            "msg": "Looks like email you have entered already has an account. To proceed, please use another login method."}), 401

    if not user or not check_password_hash(user['password'], password):
        return jsonify({"msg": "Opps, that isn't right, please try again."}), 401

    access_token = create_access_token(identity=email)
    response = jsonify({"msg": "Login successful"})
    response.set_cookie('jwt', access_token)

    return response, 200


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    response = jsonify({"msg": "Logout successful"})
    unset_jwt_cookies(response)
    unset_access_cookies(response)
    return response, 200


@auth_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    email = request.json.get('email')
    new_password = request.json.get('new_password')

    if not email:
        return jsonify({"msg": "Email is required"}), 400
    if not new_password:
        return jsonify({"msg": "New password is required"}), 400

    hashed_password = generate_password_hash(new_password)

    if get_user(email):
        update_password(email, hashed_password)
        return jsonify({"msg": "Password updated successfully"}), 200
    else:
        return jsonify({"msg": "User not found"}), 404