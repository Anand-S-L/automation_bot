# State-of-the-Art RL Farming Agent - Setup Script
# Run this in PowerShell as Administrator

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  State-of-the-Art RL Farming Agent - Setup" -ForegroundColor Cyan
Write-Host "  Phase 2: Perception System Complete" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  $pythonVersion" -ForegroundColor Green

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Python not found!" -ForegroundColor Red
    Write-Host "  Please install Python 3.8+ from python.org" -ForegroundColor Red
    exit 1
}

# Check if already installed
Write-Host "`nChecking existing installations..." -ForegroundColor Yellow

$torchInstalled = python -c "import torch; print('OK')" 2>&1
$cv2Installed = python -c "import cv2; print('OK')" 2>&1
$mssInstalled = python -c "import mss; print('OK')" 2>&1

# Install dependencies
Write-Host "`n=== Installing Dependencies ===" -ForegroundColor Cyan

if ($torchInstalled -ne "OK") {
    Write-Host "`n[1/4] Installing PyTorch..." -ForegroundColor Yellow
    Write-Host "  Detecting GPU..." -ForegroundColor Gray
    
    $hasNvidia = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    
    if ($hasNvidia) {
        Write-Host "  NVIDIA GPU detected! Installing CUDA version..." -ForegroundColor Green
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    } else {
        Write-Host "  No NVIDIA GPU. Installing CPU version..." -ForegroundColor Yellow
        python -m pip install torch torchvision
    }
} else {
    Write-Host "`n[1/4] PyTorch already installed ✓" -ForegroundColor Green
}

if ($cv2Installed -ne "OK") {
    Write-Host "`n[2/4] Installing OpenCV..." -ForegroundColor Yellow
    python -m pip install opencv-python numpy
} else {
    Write-Host "`n[2/4] OpenCV already installed ✓" -ForegroundColor Green
}

if ($mssInstalled -ne "OK") {
    Write-Host "`n[3/4] Installing screen capture libraries..." -ForegroundColor Yellow
    python -m pip install mss pyautogui pillow
} else {
    Write-Host "`n[3/4] Screen capture libraries already installed ✓" -ForegroundColor Green
}

Write-Host "`n[4/4] Installing OCR and monitoring tools..." -ForegroundColor Yellow
python -m pip install easyocr pytesseract tensorboard wandb tqdm matplotlib

Write-Host "`n=== Installation Complete! ===" -ForegroundColor Green

# Verify installation
Write-Host "`n=== Verifying Installation ===" -ForegroundColor Cyan

Write-Host "  PyTorch: " -NoNewline
python -c "import torch; print(f'v{torch.__version__} (CUDA: {torch.cuda.is_available()})')"

Write-Host "  OpenCV: " -NoNewline
python -c "import cv2; print(f'v{cv2.__version__}')"

Write-Host "  NumPy: " -NoNewline
python -c "import numpy; print(f'v{numpy.__version__}')"

Write-Host "  MSS: " -NoNewline
python -c "import mss; print('OK')"

Write-Host "  PyAutoGUI: " -NoNewline
python -c "import pyautogui; print('OK')"

$easyocrOK = python -c "import easyocr; print('OK')" 2>&1
if ($easyocrOK -eq "OK") {
    Write-Host "  EasyOCR: OK" -ForegroundColor Green
} else {
    Write-Host "  EasyOCR: FAILED (will use Tesseract)" -ForegroundColor Yellow
}

Write-Host "  TensorBoard: " -NoNewline
python -c "from torch.utils.tensorboard import SummaryWriter; print('OK')"

# Check Tesseract
Write-Host "`n=== Checking Tesseract OCR ===" -ForegroundColor Cyan
$tesseractPath = "C:\Program Files\Tesseract-OCR\tesseract.exe"

if (Test-Path $tesseractPath) {
    Write-Host "  Tesseract found! ✓" -ForegroundColor Green
} else {
    Write-Host "  Tesseract not installed." -ForegroundColor Yellow
    Write-Host "  Download from: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
    Write-Host "  (Optional - EasyOCR is preferred)" -ForegroundColor Gray
}

# GPU Info
Write-Host "`n=== Hardware Info ===" -ForegroundColor Cyan

$hasCuda = python -c "import torch; print(torch.cuda.is_available())" 2>&1
if ($hasCuda -eq "True") {
    Write-Host "  GPU: " -NoNewline
    python -c "import torch; print(torch.cuda.get_device_name(0))"
    Write-Host "  Training speed: FAST (GPU-accelerated) 🚀" -ForegroundColor Green
} else {
    Write-Host "  GPU: None (CPU training)" -ForegroundColor Yellow
    Write-Host "  Training speed: SLOW (5-10x slower than GPU)" -ForegroundColor Yellow
}

# Next steps
Write-Host "`n=== Next Steps ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Test perception modules:" -ForegroundColor White
Write-Host "   python perception/health_detection.py" -ForegroundColor Gray
Write-Host "   python perception/enemy_detection.py" -ForegroundColor Gray
Write-Host "   python perception/reward_detection.py" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Create config file:" -ForegroundColor White
Write-Host "   Copy from IMPLEMENTATION_GUIDE.md" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Start training:" -ForegroundColor White
Write-Host "   python rl_farming_agent.py" -ForegroundColor Gray
Write-Host "   (Choose option 1)" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Monitor training:" -ForegroundColor White
Write-Host "   tensorboard --logdir=runs" -ForegroundColor Gray
Write-Host ""

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete! Ready to train AI. 🤖" -ForegroundColor Green
Write-Host "  See IMPLEMENTATION_GUIDE.md for full guide" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
