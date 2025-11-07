# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### Issue: Index Error When Clicking on Minimap

**Error Message:**
```
IndexError: index 439 is out of bounds for axis 0 with size 181
```

**Cause:** 
Clicking outside the captured minimap area (especially on a circular minimap where corners are outside the circle).

**Solution:**
✅ **Use the improved configuration method:**

The tool now uses **center + edge** method instead of corner-based:

1. Run `python configure_minimap.py`
2. Click the **CENTER** of the minimap circle (easy!)
3. Click any point on the **EDGE** (top, bottom, left, or right)
4. The tool calculates the bounding box automatically
5. Only the circular area is analyzed (no corner issues!)

**Why this works:**
- Circular minimap doesn't have accessible corners
- Center + radius method is more accurate
- Bounds checking prevents index errors
- Analysis focuses only on valid circular area

---

### Issue: "No Paths Detected" or "All Obstacles"

**Symptoms:**
- Bot doesn't move or moves randomly
- Debug window shows all red (obstacles) or all green (paths)
- Console shows "No clear path detected"

**Solution:**

1. **Check captured images:**
   ```
   minimap_original.png - Should show clear minimap
   minimap_bright_areas.png - Should show paths in white
   minimap_dark_areas.png - Should show obstacles in white
   ```

2. **Adjust color thresholds in `config_minimap.json`:**

   **If paths not detected (too dark):**
   ```json
   {
     "path_color_lower": [80, 80, 80],  // Lower = more sensitive
     "path_color_upper": [255, 255, 255]
   }
   ```

   **If too many false paths:**
   ```json
   {
     "path_color_lower": [150, 150, 150],  // Higher = less sensitive
     "path_color_upper": [255, 255, 255]
   }
   ```

   **If obstacles not detected:**
   ```json
   {
     "obstacle_color_lower": [0, 0, 0],
     "obstacle_color_upper": [120, 120, 120]  // Higher = more obstacles
   }
   ```

3. **Re-run configuration:**
   ```powershell
   python configure_minimap.py
   ```

---

### Issue: Bot Doesn't Move

**Symptoms:**
- Bot starts but character doesn't move
- No error messages
- Debug window shows but no movement

**Possible Causes & Solutions:**

#### 1. Game Window Not in Focus
**Solution:**
- Make sure BlueStacks/game window is active
- Click on game window after bot starts
- Bot gives you 3 seconds to switch windows

#### 2. Wrong Key Bindings
**Solution:**
- Check game controls: Settings → Controls
- Movement should be **Arrow Keys**
- Camera should be **Numpad 4/6**
- Test keys manually in game first

#### 3. NumLock is OFF
**Solution:**
- Press **NumLock** to turn it ON
- Camera controls (numpad) won't work without it

#### 4. BlueStacks Controls Not Mapped
**Solution:**
- Open BlueStacks Control Editor
- Map in-game movement to arrow keys
- Map camera to numpad keys
- Save control scheme

---

### Issue: Wrong Area Captured

**Symptoms:**
- Debug window shows UI elements, not minimap
- Captures wrong part of screen
- Shows action bars or other UI

**Solution:**

1. **Verify minimap location:**
   - Evil Lands minimap is in **top-right corner**
   - Look for **largest circle** on top-right
   - Slightly down from very top edge

2. **Re-run configuration:**
   ```powershell
   python configure_minimap.py
   ```
   - Be precise with center click
   - Make sure to click actual minimap edge

3. **Manual adjustment:**
   Edit `config_minimap.json`:
   ```json
   {
     "minimap_region": [1670, 50, 200, 200]  // Adjust these values
   }
   ```

4. **Check screen resolution:**
   - BlueStacks display settings
   - Resolution affects coordinates
   - Common: 1920x1080, 1280x720

---

### Issue: Camera Spinning Constantly

**Symptoms:**
- Camera keeps rotating left/right
- Can't maintain direction
- Character moving erratically

**Solution:**

1. **Increase turn threshold:**
   Edit `config_minimap.json`:
   ```json
   {
     "turn_threshold": 25.0  // Higher = less turning (was 15.0)
   }
   ```

2. **Check camera controls:**
   - Verify numpad 4/6 work in game
   - Test manually before running bot
   - Make sure no key conflicts

3. **Disable mouse camera:**
   - In-game settings
   - Disable mouse camera control
   - Use only keyboard camera

---

### Issue: Bot Gets Stuck Frequently

**Symptoms:**
- Character runs into walls
- Keeps hitting same obstacle
- Rotation doesn't help

**Solution:**

1. **Adjust look-ahead distance:**
   ```json
   {
     "look_ahead_distance": 40  // Look further (was 30)
   }
   ```

2. **Lower stuck threshold:**
   ```json
   {
     "stuck_threshold": 3  // React faster (was 5)
   }
   ```

3. **Improve path detection:**
   - Recalibrate colors
   - Make sure minimap is clear
   - Increase game graphics if minimap is blurry

---

### Issue: Lag or Stuttering

**Symptoms:**
- Bot moves slowly
- Long delays between actions
- High CPU usage

**Solution:**

1. **Increase scan interval:**
   ```json
   {
     "scan_interval": 0.6  // Slower scans (was 0.4)
   }
   ```

2. **Disable debug mode:**
   Edit `minimap_navigator.py` line ~520:
   ```python
   navigator.start(debug=False)  // Was debug=True
   ```

3. **Close other applications:**
   - Free up CPU/RAM
   - Close browser tabs
   - Stop background processes

4. **BlueStacks performance:**
   - Settings → Performance
   - Increase CPU cores (4 recommended)
   - Increase RAM (4GB recommended)
   - Enable "High Performance" mode

---

### Issue: Import Errors / Module Not Found

**Error Messages:**
```
ModuleNotFoundError: No module named 'cv2'
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'mss'
```

**Solution:**

```powershell
# Reinstall all dependencies
pip install -r requirements.txt

# Or install individually:
pip install opencv-python
pip install numpy
pip install mss
pip install pyautogui
pip install pillow
```

**If still failing:**
```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Then install with --user flag
pip install --user -r requirements.txt
```

---

### Issue: Permission Errors

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**

1. **Run PowerShell as Administrator:**
   - Right-click PowerShell
   - "Run as Administrator"
   - Navigate to project folder
   - Run bot again

2. **Check file permissions:**
   - Make sure you can write to project folder
   - Close any programs using the files
   - Close Excel/text editors with config files open

---

### Issue: Keys Get Stuck

**Symptoms:**
- Character keeps moving after stopping bot
- Keys held down after Ctrl+C
- Need to manually press keys to unstick

**Solution:**

1. **Proper shutdown:**
   - Always use **Ctrl+C** to stop (not close window)
   - Bot releases keys on proper shutdown

2. **Manual key release:**
   If keys are stuck:
   - Press and release each arrow key: ↑ ↓ ← →
   - Press and release numpad: 4, 6, 8, 5
   - Or restart game

3. **Emergency stop:**
   ```python
   # Create emergency_stop.py
   import pyautogui
   
   for key in ['up', 'down', 'left', 'right', 'num4', 'num6', 'num8', 'num5']:
       pyautogui.keyUp(key)
   print("All keys released!")
   ```
   
   Run: `python emergency_stop.py`

---

### Issue: BlueStacks Window Moves

**Symptoms:**
- Bot stops working after moving window
- Captures wrong area suddenly
- Coordinates invalid after window move

**Solution:**

1. **Keep window stationary:**
   - Don't move BlueStacks window during operation
   - Position it before starting bot
   - Consider "Always on Top" window setting

2. **Reconfigure if moved:**
   ```powershell
   python configure_minimap.py
   ```

3. **Use windowed mode:**
   - Not fullscreen
   - Fixed position
   - Consistent coordinates

---

### Issue: Python Version Problems

**Error:**
```
SyntaxError: invalid syntax
```

**Cause:** Python version too old

**Solution:**

```powershell
# Check Python version
python --version

# Should be 3.8 or higher
# If lower, install Python 3.8+ from python.org
```

Download: https://www.python.org/downloads/

✅ Check "Add Python to PATH" during installation

---

## Quick Diagnostic Steps

### Step 1: Verify Setup
```powershell
# Check Python
python --version  # Should be 3.8+

# Check pip
pip --version

# Check packages
pip list | findstr "opencv numpy mss pyautogui"
```

### Step 2: Test Components

```powershell
# Test screen capture
python configure_screen.py

# Test minimap capture
python configure_minimap.py

# Check generated images
# minimap_original.png - should be clear
# minimap_bright_areas.png - paths
# minimap_dark_areas.png - obstacles
```

### Step 3: Test Controls Manually

In Evil Lands:
- Press ↑ → Character moves forward?
- Press ← → Character moves left?
- Press Numpad 4 → Camera rotates left?
- Press Numpad 6 → Camera rotates right?

If any fail, fix game controls first!

### Step 4: Short Test Run

```powershell
python minimap_navigator.py
```

Run for 30 seconds:
- Does character move?
- Does it avoid obstacles?
- Does camera adjust?

---

## Getting Help

### Information to Provide:

1. **Your setup:**
   - Windows version
   - Python version (`python --version`)
   - BlueStacks version
   - Screen resolution

2. **Error message:**
   - Full error text from PowerShell
   - Line numbers
   - What you were doing

3. **Configuration:**
   - Contents of `config_minimap.json`
   - Generated images (minimap_*.png)

4. **What you tried:**
   - Steps already attempted
   - What changed
   - When it stopped working

---

## Prevention Tips

✅ **Before starting:**
- Install all dependencies
- Configure minimap correctly
- Test controls manually
- Position window properly
- Enable NumLock

✅ **During operation:**
- Don't move BlueStacks window
- Don't minimize game
- Keep game in focus
- Don't change resolution

✅ **After issues:**
- Use Ctrl+C to stop (not X)
- Check all keys released
- Review console output
- Check generated images

---

**Most issues are fixed by:**
1. Reconfiguring minimap (`configure_minimap.py`)
2. Adjusting color thresholds in config
3. Testing controls manually first
4. Running PowerShell as Administrator

Good luck! 🚀
