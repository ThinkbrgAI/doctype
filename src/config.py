import os
import json
from pathlib import Path
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        self.config_dir = Path.home() / '.doctype'
        self.config_dir.mkdir(exist_ok=True)
        self.key_file = self.config_dir / 'key.bin'
        self.config_file = self.config_dir / 'config.enc'
        self._init_encryption()

    def _init_encryption(self):
        """Initialize or load encryption key"""
        if not self.key_file.exists():
            key = Fernet.generate_key()
            self.key_file.write_bytes(key)
        self.fernet = Fernet(self.key_file.read_bytes())

    def save_api_key(self, api_key):
        """Save encrypted API key"""
        config = {'api_key': api_key}
        encrypted_data = self.fernet.encrypt(json.dumps(config).encode())
        self.config_file.write_bytes(encrypted_data)

    def get_api_key(self):
        """Get decrypted API key"""
        if not self.config_file.exists():
            return None
        try:
            encrypted_data = self.config_file.read_bytes()
            decrypted_data = self.fernet.decrypt(encrypted_data)
            config = json.loads(decrypted_data)
            return config.get('api_key')
        except Exception:
            return None

def setup_api_key(api_key=None):
    """Set up API key either from input or environment"""
    config = SecureConfig()
    
    # If API key is provided, save it
    if api_key:
        config.save_api_key(api_key)
        return api_key
        
    # Otherwise try to get from environment or user input
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ").strip()
        
    if api_key:
        config.save_api_key(api_key)
        
    return api_key
