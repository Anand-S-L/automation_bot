# 🎮 Running the Bot with BlueStacks Emulator

Complete guide for setti3. **Interface Settings:**
   - Make sure **Minimap is Visible** ✅
   - Minimap should be in **top-right corner** (default for Evil Lands)
   - Look for the **largest circle** on the top-right
   - It's positioned **slightly down** from the very top edgeup and running the Evil Lands navigation bot with BlueStacks Android emulator.

---

## 📋 Prerequisites

### 1. Software Requirements
- ✅ **Python 3.8+** installed
- ✅ **BlueStacks 5** (or newer) installed
- ✅ **Evil Lands** installed in BlueStacks
- ✅ **Windows** (7/10/11)

### 2. Check Python Installation
```powershell
python --version
# Should show: Python 3.8 or higher
```

If Python is not installed:
- Download from: https://www.python.org/downloads/
- ✅ Check "Add Python to PATH" during installation

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

Open PowerShell in the project folder and run:

```powershell
# Install required Python packages
pip install -r requirements.txt
```

**Packages installed:**
- `opencv-python` - Computer vision
- `numpy` - Image processing
- `mss` - Fast screen capture
- `pyautogui` - Input simulation
- `pillow` - Image handling

---

### Step 2: Configure BlueStacks

#### A. BlueStacks Window Setup

1. **Open BlueStacks**
2. **Launch Evil Lands**
3. **Set Window Mode:**
   - Click the ⚙️ (Settings) icon in BlueStacks
   - Go to **Display** settings
   - Choose **Windowed mode** or **Fullscreen windowed**
   - Resolution: **1920x1080** (recommended) or **1280x720**
   - DPI: **240** (recommended)
   - Click **Save** and restart BlueStacks

#### B. Game Settings (Evil Lands)

1. **Open Evil Lands Settings** (⚙️ in game)
2. **Graphics Settings:**
   - Set to **Medium** or **High** (for clear minimap)
   - Enable **Show Minimap** ✅
   
3. **Control Settings:**
   - Enable **Keyboard Controls** ✅
   - Map controls as:
     - **Movement:** Arrow Keys (↑↓←→)
     - **Camera Look:** Numpad 4 (left), 6 (right), 8 (up), 5 (down)
   
4. **Interface Settings:**
   - Make sure **Minimap is Visible** ✅
   - Minimap should be in **top-left corner** (default)

#### C. BlueStacks Control Mapping

BlueStacks has built-in keyboard mapping. Configure it:

1. Click the **Keyboard icon** on the right side panel
2. Click **"Advanced Editor"** or **"Control Editor"**
3. Set up controls:

**Movement (Arrow Keys):**
- Place WASD control on screen → Map to Arrow Keys
  - W → Up Arrow
  - A → Left Arrow
  - S → Down Arrow
  - D → Right Arrow

**Camera (Numpad):**
- Place camera controls → Map to Numpad
  - Look Left → Numpad 4
  - Look Right → Numpad 6
  - Look Up → Numpad 8
  - Look Down → Numpad 5

4. Click **Save** (floppy disk icon)

---

### Step 3: Find Minimap Coordinates

Run the configuration tool:

```powershell
python configure_minimap.py
```

**What this does:**
1. Asks you to click **center** of the minimap circle
2. Asks you to click any point on the **edge** of the circle
3. Automatically calculates the bounding box
4. Captures the minimap image
5. Auto-detects path colors (only within circular area)
6. Creates `config_minimap.json`
7. Shows preview to verify detection

**Important Tips for BlueStacks:**
- The minimap in Evil Lands is in the **top-right corner**
- It's the **largest circle** on the top-right
- It's positioned **slightly down** from the very top edge
- For 1920x1080 resolution, typical coordinates:
  - Top-left: `(1670, 50)`
  - Bottom-right: `(1870, 250)`
  - Size: `200x200` pixels

**Example for BlueStacks (1920x1080):**
```json
{
  "minimap_region": [1670, 50, 200, 200]
}
```

**Example for BlueStacks (1280x720):**
```json
{
  "minimap_region": [1080, 35, 150, 150]
}
```

**Example for BlueStacks (1600x900):**
```json
{
  "minimap_region": [1350, 40, 175, 175]
}
```

---

### Step 4: Run the Navigator

Start the minimap-based bot:

```powershell
python minimap_navigator.py
```

**What happens:**
1. Shows countdown: **3 seconds**
2. **Switch to BlueStacks window immediately**
3. Bot starts analyzing minimap
4. Character moves automatically
5. Debug window shows detection (optional)

**To stop:** Press `Ctrl+C` in PowerShell

---

## 🔧 Configuration Files

### `config_minimap.json` - Main Configuration

```json
{
  "minimap_region": [1670, 50, 200, 200],
  "path_color_lower": [120, 120, 120],
  "path_color_upper": [255, 255, 255],
  "obstacle_color_lower": [0, 0, 0],
  "obstacle_color_upper": [80, 80, 80],
  "movement_duration": 0.4,
  "scan_interval": 0.4,
  "look_ahead_distance": 30,
  "turn_threshold": 15.0,
  "stuck_threshold": 5
}
```

### Key Settings Explained:

| Setting | Description | Default | BlueStacks Tip |
|---------|-------------|---------|----------------|
| `minimap_region` | `[left, top, width, height]` of minimap | `[1670, 50, 200, 200]` | Use `configure_minimap.py` to find |
| `path_color_lower` | BGR color for paths (bright areas) | `[120, 120, 120]` | Auto-detected |
| `path_color_upper` | Upper bound for path color | `[255, 255, 255]` | Usually white |
| `obstacle_color_lower` | BGR color for obstacles (dark) | `[0, 0, 0]` | Usually black |
| `obstacle_color_upper` | Upper bound for obstacles | `[80, 80, 80]` | Adjust if needed |
| `movement_duration` | How long to hold arrow keys (seconds) | `0.4` | Try 0.3-0.5 |
| `scan_interval` | Time between minimap scans | `0.4` | Try 0.3-0.6 |
| `look_ahead_distance` | Pixels to look ahead on minimap | `30` | 20-40 range |
| `turn_threshold` | Min angle before camera turns | `15.0` | 10-20 range |
| `stuck_threshold` | Frames before "stuck" recovery | `5` | 3-7 range |

---

## 🎯 BlueStacks-Specific Tips

### 1. Performance Settings

For smooth bot operation:

**BlueStacks Settings → Performance:**
- CPU: **4 cores** (if available)
- RAM: **4 GB** (recommended)
- Performance Mode: **High Performance**
- Frame Rate: **60 FPS**

### 2. Window Position

- Keep BlueStacks window **stationary**
- Don't minimize while bot is running
- Don't move/resize window during operation
- Bot captures specific screen coordinates

### 3. Screen Resolution

**Best resolutions for Evil Lands:**
- **1920x1080** - Best quality, larger minimap
- **1280x720** - Faster, lower resource usage
- **1600x900** - Good balance

After changing resolution:
1. Restart BlueStacks
2. Run `configure_minimap.py` again
3. Update coordinates

### 4. Multi-Instance Manager

If running multiple BlueStacks instances:
- Bot works with **one instance at a time**
- Make sure correct instance is in focus
- Configure for each instance separately

### 5. Virtualization

Enable virtualization for best performance:
- **Intel:** Enable VT-x in BIOS
- **AMD:** Enable SVM in BIOS
- Improves emulator speed significantly

---

## 📊 Testing & Verification

### Test 1: Screen Capture
```powershell
python configure_screen.py
```
Verifies bot can capture BlueStacks screen.

### Test 2: Minimap Detection
```powershell
python configure_minimap.py
```
Verifies minimap location and color detection.

### Test 3: Control Test

Manually test in Evil Lands:
- Press **Arrow Keys** → Character moves
- Press **Numpad 4/6** → Camera rotates
- Check **NumLock is ON** ✅

### Test 4: Short Run
```powershell
python minimap_navigator.py
```
Let it run for 30 seconds, verify:
- ✅ Character moves toward paths
- ✅ Camera adjusts when needed
- ✅ Avoids obstacles
- ✅ Follows narrow trails

---

## ❗ Common Issues & Solutions

### Issue 1: Bot doesn't capture screen
**Cause:** Permission or window focus issue

**Solution:**
```powershell
# Run PowerShell as Administrator
# Right-click PowerShell → "Run as Administrator"
python minimap_navigator.py
```

### Issue 2: Wrong area captured
**Cause:** Minimap coordinates incorrect

**Solution:**
```powershell
# Reconfigure minimap location
python configure_minimap.py
```

### Issue 3: Bot doesn't move character
**Cause:** Controls not mapped correctly

**Solution:**
1. Open BlueStacks Control Editor
2. Verify arrow keys mapped to movement
3. Test manually in game first
4. Save control scheme

### Issue 4: Bot sees no paths (all obstacles)
**Cause:** Color detection thresholds wrong

**Solution:**
Edit `config_minimap.json`:
```json
{
  "path_color_lower": [100, 100, 100],  // Lower = more sensitive
  "obstacle_color_upper": [100, 100, 100]  // Higher = less sensitive
}
```

### Issue 5: Character stuck or spinning
**Cause:** Camera controls not working

**Solution:**
1. Check NumLock is **ON**
2. Verify numpad keys work in game
3. Map camera to numpad in BlueStacks
4. Test numpad 4/6 manually

### Issue 6: Lag or slow response
**Cause:** BlueStacks performance

**Solution:**
1. Increase BlueStacks CPU/RAM allocation
2. Close other applications
3. Increase `scan_interval` in config:
```json
{
  "scan_interval": 0.6  // Slower but more stable
}
```

### Issue 7: Debug window shows wrong colors
**Cause:** Game graphics settings

**Solution:**
1. Check saved images:
   - `minimap_original.png`
   - `minimap_bright_areas.png`
   - `minimap_dark_areas.png`
2. If colors look wrong, adjust in game:
   - Increase brightness
   - Change graphics preset
3. Re-run `configure_minimap.py`

---

## 🎮 Different BlueStacks Versions

### BlueStacks 5 (Recommended)
- Best performance
- Direct keyboard input
- Follow guide as-is

### BlueStacks 4
- Slightly older
- May need compatibility mode
- Same configuration process

### BlueStacks X (Cloud)
- Cloud gaming version
- May have input lag
- Not recommended for bot

---

## 📁 Project Structure

```
farm/
├── autonomous_navigator.py    # 3D vision bot
├── minimap_navigator.py       # Minimap bot (recommended)
├── configure_minimap.py       # Setup tool
├── configure_screen.py        # Screen capture test
├── config_minimap.json        # Minimap config (auto-generated)
├── config.json                # 3D vision config
├── requirements.txt           # Python dependencies
├── README.md                  # General documentation
├── MINIMAP_GUIDE.md          # Minimap navigation guide
├── CONTROLS.md               # Control scheme reference
└── BLUESTACKS_SETUP.md       # This file
```

---

## 🚦 Step-by-Step First Run

### Complete Workflow:

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start BlueStacks and Evil Lands
# - Set windowed mode
# - Make sure minimap is visible
# - Configure keyboard controls

# 3. Configure minimap
python configure_minimap.py
# - Click top-left of minimap
# - Click bottom-right of minimap
# - Wait for analysis
# - Check generated images

# 4. Verify configuration
# - Open config_minimap.json
# - Check minimap_region values
# - Adjust if needed

# 5. Run the bot
python minimap_navigator.py
# - Wait for 3 second countdown
# - Switch to BlueStacks
# - Watch it navigate!

# 6. Stop anytime
# Press Ctrl+C in PowerShell
```

---

## 💡 Pro Tips for BlueStacks

1. **Save Control Scheme:**
   - After configuring controls, export the scheme
   - BlueStacks → Settings → Controls → Export
   - Reload if needed

2. **Use Same Window Position:**
   - Keep BlueStacks in same screen position
   - Coordinates stay consistent
   - No need to reconfigure

3. **Graphics Balance:**
   - High enough for clear minimap
   - Not too high to cause lag
   - Medium-High usually perfect

4. **Test Path Finding:**
   - Start in open area first
   - Let bot run in safe zone
   - Verify path detection works
   - Then try complex areas

5. **Monitor CPU Usage:**
   - BlueStacks + Bot + Python ~40-60% CPU
   - Close other apps if laggy
   - Adjust scan_interval if needed

6. **Multiple Configs:**
   - Create configs for different resolutions
   - `config_minimap_1080p.json`
   - `config_minimap_720p.json`
   - Switch as needed

---

## 🎯 Expected Results

**After proper setup, the bot should:**
- ✅ Capture minimap correctly
- ✅ Detect paths (bright areas)
- ✅ Detect obstacles (dark areas)
- ✅ Move character along paths
- ✅ Avoid walls and obstacles
- ✅ Follow narrow trails
- ✅ Rotate camera when stuck
- ✅ Navigate autonomously

**Console output example:**
```
✓ SAFE | NARROW PATH | Target: 45.2° | Frame: 10
✓ SAFE | NARROW PATH | Target: 43.8° | Frame: 11
✓ SAFE | OPEN AREA | Target: 90.5° | Frame: 12
⚠ CAUTION | NARROW PATH | Target: 25.1° | Frame: 13
```

---

## 📞 Need Help?

If you encounter issues:

1. Check generated images:
   - `minimap_original.png` - Should show your minimap clearly
   - `minimap_bright_areas.png` - Should show paths in white
   - `minimap_dark_areas.png` - Should show obstacles in white

2. Run with debug mode (default) to see real-time detection

3. Test each component separately:
   - Screen capture
   - Minimap detection
   - Control inputs
   - Path finding

4. Review logs in PowerShell for error messages

---

## ⚠️ Important Reminders

- ✅ **NumLock must be ON** for numpad camera controls
- ✅ **BlueStacks window must be in focus** during operation
- ✅ **Don't minimize or move window** while running
- ✅ **Use for educational purposes only**
- ✅ **May violate game ToS** - use responsibly
- ✅ **Test in safe areas first**

---

## 🎓 Learning Resources

- **OpenCV Tutorial:** https://docs.opencv.org/
- **PyAutoGUI Docs:** https://pyautogui.readthedocs.io/
- **BlueStacks Guide:** https://support.bluestacks.com/

---

**Ready to automate! 🚀 Start with Step 1 and follow through. Good luck!**
