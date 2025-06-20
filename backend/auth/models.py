from db import mongo

def add_user(org_name, email, hashed_password):
    mongo.db.users.insert_one({'org_name': org_name, 'email': email, 'password': hashed_password})


def update_password(email, new_hashed_password):
    user = mongo.db.users.find_one({'email': email})
    if user:
        user['password'] = new_hashed_password
        mongo.db.users.save(user)


def get_user(email):
    return mongo.db.users.find_one({'email': email})
