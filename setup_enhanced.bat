@echo off
echo ============================================================
echo  Enhanced RL Agent - Quick Setup
echo ============================================================
echo.

echo Step 1: Installing dependencies...
python install_dependencies.py
if errorlevel 1 (
    echo Installation failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Start Evil Lands (windowed mode)
echo   2. Run: python configure_evil_lands.py
echo   3. Run: python enhanced_rl_agent.py
echo.
echo Or just run: python enhanced_rl_agent.py (uses defaults)
echo.
pause
