from flask import Flask, jsonify
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from config import Config
from auth import create_auth_blueprint
from data_breach import create_data_breach_blueprint
from ml import create_ml_blueprint

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, 
     supports_credentials=True,
     origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"])
jwt_manager = JWTManager(app)

app.register_blueprint(create_auth_blueprint(), url_prefix='/v1/auth')
app.register_blueprint(create_data_breach_blueprint(), url_prefix='/v1/data_breach')
app.register_blueprint(create_ml_blueprint(), url_prefix='/v1/ml')

@app.route('/v1/health', methods=['GET'])
def versionInfo():
    return jsonify({"version": "1.0.0", "maintenance": True})

if __name__ == '__main__':
    app.run(port=5555)