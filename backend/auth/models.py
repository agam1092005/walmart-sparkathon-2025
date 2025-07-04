import requests

POCKETBASE_URL = 'http://127.0.0.1:8090'
USERS_COLLECTION = 'users'

def signup_user(org_name, email, password):
    data = {
        'org_name': org_name,
        'email': email,
        'password': password,
        'passwordConfirm': password
    }
    resp = requests.post(f"{POCKETBASE_URL}/api/collections/{USERS_COLLECTION}/records", json=data)
    print("PocketBase response:", resp.status_code, resp.text) 
    return resp.json() if resp.status_code == 200 else None

def login_user(email, password):
    data = {
        'identity': email,
        'password': password
    }
    resp = requests.post(f"{POCKETBASE_URL}/api/collections/{USERS_COLLECTION}/auth-with-password", json=data)
    return resp.json() if resp.status_code == 200 else None

def get_company_data_by_token(token):
    """Get company data using the provided token"""
    headers = {
        'Authorization': f'Bearer {token}'
    }
    # First validate the token by getting user info
    resp = requests.get(f"{POCKETBASE_URL}/api/users/profile", headers=headers)
    if resp.status_code != 200:
        return None
    
    user_data = resp.json()
    return {
        'org_name': user_data.get('org_name'),
        'email': user_data.get('email'),
        'id': user_data.get('id')
    }
