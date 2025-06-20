from flask import Flask, jsonify
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from db import mongo
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, supports_credentials=True)
mongo.init_app(app)
jwt_manager = JWTManager(app)

@app.route('/v1/health', methods=['GET'])
def versionInfo():
    return jsonify({"version": "1.0.0", "maintenance": True})


if __name__ == '__main__':
    app.run(port=5555)