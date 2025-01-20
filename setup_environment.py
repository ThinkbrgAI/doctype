import os
import sys
import subprocess
import platform
from pathlib import Path
from config import SecureConfig, setup_api_key
from create_folder_structure import create_folder_structure

def setup_environment():
    """Set up the required environment for the document classifier"""
    
    # Install pip requirements
    print("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Handle libmagic installation on Windows
    if platform.system() == 'Windows':
        print("Installing libmagic for Windows...")
        try:
            # Try to install python-magic-bin which includes libmagic for Windows
            subprocess.check_call([sys.executable, "-m", "pip", "install", "python-magic-bin"])
        except subprocess.CalledProcessError:
            print("""
Error: Could not install libmagic automatically.
Please follow these manual steps:
1. Download the libmagic DLL from: https://github.com/nscaife/file-windows/releases
2. Extract the ZIP file
3. Copy the magic1.dll file to C:\\Windows\\System32
""")
            sys.exit(1)
    
    print("Environment setup completed successfully!")

if __name__ == "__main__":
    setup_environment() 