import os
from pathlib import Path
import shutil

def create_project_files(base_dir: Path):
    """Create all necessary project files"""
    
    # Ensure base directory exists
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary of file contents
    files = {
        # Main project files
        base_dir / 'config.py': '''import os
from pathlib import Path
from cryptography.fernet import Fernet
import base64
import json
from getpass import getpass

class SecureConfig:
    def __init__(self):
        self.config_dir = Path.home() / '.doctype'
        self.config_file = self.config_dir / 'config.enc'
        self.key_file = self.config_dir / '.key'
        self._ensure_config_dir()
        
    def _ensure_config_dir(self):
        """Ensure configuration directory exists"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.chmod(0o700)  # Restrict directory permissions
        
    def _generate_key(self):
        """Generate a new encryption key"""
        return Fernet.generate_key()
    
    def _save_key(self, key):
        """Save encryption key securely"""
        self.key_file.write_bytes(key)
        self.key_file.chmod(0o600)  # Restrict file permissions
        
    def _load_key(self):
        """Load the encryption key"""
        if not self.key_file.exists():
            key = self._generate_key()
            self._save_key(key)
        return self.key_file.read_bytes()
    
    def save_api_key(self, api_key):
        """Encrypt and save the API key"""
        try:
            key = self._load_key()
            f = Fernet(key)
            config = {
                'api_key': api_key,
                'timestamp': str(Path.ctime(Path(self.config_file))) if self.config_file.exists() else None
            }
            encrypted_data = f.encrypt(json.dumps(config).encode())
            self.config_file.write_bytes(encrypted_data)
            self.config_file.chmod(0o600)
            return True
        except Exception as e:
            print(f"Error saving API key: {str(e)}")
            return False
    
    def get_api_key(self):
        """Retrieve the API key"""
        try:
            if not self.config_file.exists():
                return None
            key = self._load_key()
            f = Fernet(key)
            encrypted_data = self.config_file.read_bytes()
            config = json.loads(f.decrypt(encrypted_data))
            return config.get('api_key')
        except Exception as e:
            print(f"Error reading API key: {str(e)}")
            return None
    
    def clear_config(self):
        """Remove all configuration files"""
        try:
            if self.config_file.exists():
                self.config_file.unlink()
            if self.key_file.exists():
                self.key_file.unlink()
            return True
        except Exception as e:
            print(f"Error clearing configuration: {str(e)}")
            return False

def setup_api_key():
    """Interactive function to set up the API key"""
    config = SecureConfig()
    existing_key = config.get_api_key()
    if existing_key:
        print("\\nAn API key is already configured.")
        choice = input("Do you want to replace it? (y/N): ").lower()
        if choice != 'y':
            return existing_key
    
    print("\\nPlease enter your OpenAI API key.")
    print("The key will be stored securely in your home directory.")
    api_key = getpass("API Key: ")
    
    if config.save_api_key(api_key):
        print("API key saved successfully!")
        return api_key
    else:
        print("Failed to save API key.")
        return None
''',

        base_dir / 'requirements.txt': '''openai>=1.0.0
pdf2image>=1.16.3
python-magic>=0.4.27
pandas>=1.5.3
Pillow>=9.5.0
openpyxl>=3.1.2
cryptography>=41.0.0
''',

        base_dir / '.gitignore': '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Logs and databases
*.log
*.sqlite

# Environment variables
.env

# Project specific
.doctype/
output/
temp/
logs/
''',

        base_dir / 'README.md': '''# Document Classification System

A vision-based document classification system using GPT-4 Vision API.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run setup:
   ```bash
   python setup_environment.py
   ```

3. Place PDF files in `input/pdfs/` directory

4. Run classifier:
   ```bash
   python run_classifier.py
   ```

## Directory Structure

- `input/pdfs/`: Place PDF files here for classification
- `output/reports/`: Classification results
- `output/logs/`: System logs
- `config/`: Configuration files
- `samples/`: Sample document categories
- `temp/`: Temporary processing files

## Security

API keys are stored securely in the user's home directory:
- Encrypted configuration: `~/.doctype/config.enc`
- Encryption key: `~/.doctype/.key`
'''
    }
    
    # Create each file
    for file_path, content in files.items():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        print(f"Created: {file_path}")

def main():
    base_dir = Path("A:/doctype")
    create_project_files(base_dir)
    print("\nInitial files created successfully!")
    print("Next steps:")
    print("1. Run: pip install -r requirements.txt")
    print("2. Run: python setup_environment.py")

if __name__ == "__main__":
    main() 