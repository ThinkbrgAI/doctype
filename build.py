import PyInstaller.__main__
import sys
import os

# Get the directory containing build.py
base_path = os.path.dirname(os.path.abspath(__file__))

PyInstaller.__main__.run([
    'main.py',
    '--name=DocType Classifier',
    '--onefile',
    '--windowed',
    '--icon=assets/icon.ico',  # We'll create this
    '--add-data=assets/icon.ico;assets',
    '--clean',
    f'--workpath={os.path.join(base_path, "build")}',
    f'--distpath={os.path.join(base_path, "dist")}',
    '--add-binary=poppler;poppler',  # For PDF processing
    '--hidden-import=PIL._tkinter_finder'
]) 