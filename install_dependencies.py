# Install script for Enhanced RL Agent
# Run this first: python install_dependencies.py

import subprocess
import sys

print("=" * 70)
print(" Installing Dependencies for Enhanced RL Farming Agent")
print("=" * 70)

packages = [
    'torch',
    'torchvision', 
    'opencv-python',
    'numpy',
    'mss',
    'pyautogui',
    'pillow',
    'easyocr',
    'pytesseract'
]

print("\nPackages to install:")
for pkg in packages:
    print(f"  • {pkg}")

print("\nStarting installation...\n")

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    print("\n" + "=" * 70)
    print(" ✓ All dependencies installed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. python configure_evil_lands.py  (setup regions)")
    print("  2. python enhanced_rl_agent.py     (start training)")
    
except subprocess.CalledProcessError as e:
    print(f"\n✗ Installation failed: {e}")
    print("\nTry manual installation:")
    print(f"  pip install {' '.join(packages)}")
    sys.exit(1)
