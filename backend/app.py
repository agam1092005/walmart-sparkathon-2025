from flask import Flask, jsonify
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from db import mongo
from config import Config
from auth import create_auth_blueprint

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, supports_credentials=True)
mongo.init_app(app)
jwt_manager = JWTManager(app)

app.register_blueprint(create_auth_blueprint(), url_prefix='/v1/auth')

@app.route('/v1/health', methods=['GET'])
def versionInfo():
    return jsonify({"version": "1.0.0", "maintenance": True})


if __name__ == '__main__':
    app.run(port=5555)