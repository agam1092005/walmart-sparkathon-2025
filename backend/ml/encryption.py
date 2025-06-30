from cryptography.fernet import Fernet
import base64


def generate_symmetric_key() -> bytes:
    """Generate a 32-byte Fernet key (AES-GCM-based)"""
    return Fernet.generate_key()


def encrypt_bytes(data: bytes, key: bytes) -> str:
    """Encrypts raw bytes with the given Fernet key and returns base64 string"""
    fernet_key = base64.urlsafe_b64encode(key[:32])
    cipher = Fernet(fernet_key)
    return cipher.encrypt(data).decode()

def decrypt_bytes(ciphertext: str, key: bytes) -> bytes:
    """Decrypts a base64-encoded string with the given Fernet key"""
    fernet_key = base64.urlsafe_b64encode(key[:32])
    cipher = Fernet(fernet_key)
    return cipher.decrypt(ciphertext.encode())


def encrypt_file(in_path: str, out_path: str, key: bytes = None):
    """Encrypts a binary file with a Fernet key and saves it"""
    key = key or generate_symmetric_key()
    with open(in_path, "rb") as f:
        data = f.read()
    encrypted = encrypt_bytes(data, key)
    with open(out_path, "wb") as f:
        f.write(encrypted.encode())
