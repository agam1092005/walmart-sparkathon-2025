from cryptography.fernet import Fernet
import base64


def generate_symmetric_key() -> bytes:
    """Generate a 32-byte Fernet key (AES-GCM-based)"""
    return Fernet.generate_key()


def get_shared_key() -> bytes:
    # WARNING: For demo only! Replace with secure key management in production.
    # This must be a 32-byte url-safe base64-encoded key (Fernet requirement)
    return b'QWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXo0NTY3ODkwMTIzNA=='


def encrypt_bytes(data: bytes, key: bytes = None) -> str:
    """Encrypts raw bytes with the shared Fernet key and returns base64 string"""
    if key is None:
        key = get_shared_key()
    fernet_key = base64.urlsafe_b64encode(key[:32])
    cipher = Fernet(fernet_key)
    return cipher.encrypt(data).decode()


def decrypt_bytes(ciphertext: str, key: bytes = None) -> bytes:
    """Decrypts a base64-encoded string with the shared Fernet key"""
    if key is None:
        key = get_shared_key()
    fernet_key = base64.urlsafe_b64encode(key[:32])
    cipher = Fernet(fernet_key)
    if isinstance(ciphertext, str):
        ciphertext = ciphertext.encode()
    return cipher.decrypt(ciphertext)


def encrypt_file(in_path: str, out_path: str, key: bytes = None):
    """Encrypts a binary file with a Fernet key and saves it"""
    key = key or generate_symmetric_key()
    with open(in_path, "rb") as f:
        data = f.read()
    encrypted = encrypt_bytes(data, key)
    with open(out_path, "wb") as f:
        f.write(encrypted.encode())
