import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')