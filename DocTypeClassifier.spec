# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all necessary data files and submodules
hidden_imports = [
    'PIL._tkinter_finder',
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'pdf2image',
    'magic',
    'pytesseract',
    'openai',
    'numpy',
    'pandas',
    'cv2',
    'cryptography'
] + collect_submodules('pdf2image')

# Collect data files
datas = [
    ('assets/icon.ico', 'assets'),
    ('src/*.py', 'src')
]

# Add poppler binaries
poppler_path = os.path.join(SPECPATH, 'venv', 'Lib', 'site-packages', 'poppler')
if os.path.exists(poppler_path):
    datas.append((poppler_path, 'poppler'))

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DocType Classifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico'
) 