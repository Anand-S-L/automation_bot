# 🎯 Quick Reference Card

## Installation (One-Time Setup)

```powershell
# Install Python packages
pip install -r requirements.txt
```

## First-Time Configuration

```powershell
# Find and configure minimap ## Links to Full Documentation

- 📘 **[BLUESTACKS_SETUP.md](BLUESTACKS_SETUP.md)** - Step-by-step BlueStacks setup
- 📍 **[MINIMAP_LOCATION.md](MINIMAP_LOCATION.md)** - Visual guide: Where is the minimap?
- 🗺️ **[MINIMAP_GUIDE.md](MINIMAP_GUIDE.md)** - How minimap navigation works
- 🎮 **[CONTROLS.md](CONTROLS.md)** - Complete control scheme guide
- 🔧 **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Fix common issues
- 📖 **[README.md](README.md)** - General project overviewn
python configure_minimap.py
```

Follow prompts to:
1. Click center of the minimap circle
2. Click any point on the edge of the circle
3. Wait for auto-calibration
4. Verify generated images

## Running the Bot

```powershell
# Start minimap-based navigation (RECOMMENDED)
python minimap_navigator.py
```

**After starting:**
- You have **3 seconds** to switch to game window
- Bot will start navigating automatically
- Press **Ctrl+C** to stop

## Alternative: 3D Vision-Based Navigation

```powershell
# Configure screen region first
python configure_screen.py

# Run 3D vision bot
python autonomous_navigator.py
```

## BlueStacks Requirements

### Before Running:
- ✅ BlueStacks in **windowed mode**
- ✅ Evil Lands running
- ✅ Minimap **visible** (top-right corner, largest circle)
- ✅ Keyboard controls enabled
- ✅ NumLock **ON**

### Controls Setup:
- **Arrow Keys** = Movement (↑↓←→)
- **Numpad 4/6** = Camera (left/right)
- **Numpad 8/5** = Look up/down (optional)

## Configuration Files

### `config_minimap.json` (Auto-generated)
Main settings for minimap navigation:

```json
{
  "minimap_region": [1670, 50, 200, 200],
  "movement_duration": 0.4,
  "scan_interval": 0.4,
  "turn_threshold": 15.0
}
```

### Common Adjustments:

**Faster movement:**
```json
"movement_duration": 0.3,
"scan_interval": 0.3
```

**More precise turning:**
```json
"turn_threshold": 10.0
```

**If paths not detected:**
```json
"path_color_lower": [100, 100, 100]
```

## Console Output

```
✓ SAFE | NARROW PATH | Target: 45.2° | Frame: 10
```

- `✓ SAFE` / `⚠ CAUTION` - Safety status
- `NARROW PATH` / `OPEN AREA` - Path type detected
- `Target: XX°` - Direction (0°=North, 90°=East)
- `Frame: XX` - Frame counter

## Stopping the Bot

Press **Ctrl+C** in PowerShell terminal

All keys automatically released safely.

## Debug Mode

Bot runs with debug visualization by default.

**Shows 3 windows:**
1. **Minimap** - Raw capture with direction arrow
2. **Paths** - Detected walkable areas (green)
3. **Obstacles** - Detected obstacles (red)

**To disable debug:**
Edit `minimap_navigator.py` line ~520:
```python
navigator.start(debug=False)
```

## Common Issues

| Problem | Solution |
|---------|----------|
| Bot doesn't move | Check game has focus, arrow keys work |
| Camera doesn't rotate | Check NumLock ON, numpad mapped in BlueStacks |
| Wrong area captured | Re-run `configure_minimap.py` |
| No paths detected | Lower `path_color_lower` in config |
| Bot gets stuck | Increase `stuck_threshold` in config |
| Lag/Stuttering | Increase `scan_interval` to 0.6 |

## Testing Steps

### 1. Test Screen Capture
```powershell
python configure_screen.py
```

### 2. Test Minimap Detection
```powershell
python configure_minimap.py
```
Check generated images:
- `minimap_original.png`
- `minimap_bright_areas.png`
- `minimap_dark_areas.png`

### 3. Test Controls Manually
In Evil Lands:
- Press arrow keys → Character moves?
- Press numpad 4/6 → Camera rotates?

### 4. Run Bot (Short Test)
```powershell
python minimap_navigator.py
```
Let run for 30 seconds, observe behavior.

## File Structure

```
farm/
├── minimap_navigator.py       ← Main bot (use this!)
├── configure_minimap.py       ← Setup tool
├── config_minimap.json        ← Your settings (auto-created)
├── requirements.txt           ← Dependencies
├── BLUESTACKS_SETUP.md       ← Full BlueStacks guide
├── MINIMAP_GUIDE.md          ← How it works
├── CONTROLS.md               ← Control reference
└── README.md                 ← Overview
```

## Typical BlueStacks Resolution Settings

### 1920x1080 (Full HD)
```json
"minimap_region": [1670, 50, 200, 200]
```
**Note:** Minimap is on top-right, the largest circle, slightly down from top edge

### 1280x720 (HD)
```json
"minimap_region": [1080, 35, 150, 150]
```

### 1600x900 (HD+)
```json
"minimap_region": [1350, 40, 175, 175]
```

**Use `configure_minimap.py` to find exact values!**

## Performance Tips

1. **BlueStacks Performance Mode**: High
2. **CPU Cores**: 4 (if available)
3. **RAM**: 4GB recommended
4. **Frame Rate**: 60 FPS
5. **Graphics**: Medium-High (clear minimap)
6. **Close other apps** while running

## Safety Features

- ✅ Automatic key release on Ctrl+C
- ✅ Safe shutdown on errors
- ✅ Obstacle avoidance
- ✅ Stuck detection and recovery
- ✅ Camera adjustment

## Next Steps

1. **New user?** → Read [BLUESTACKS_SETUP.md](BLUESTACKS_SETUP.md)
2. **Want details?** → Read [MINIMAP_GUIDE.md](MINIMAP_GUIDE.md)
3. **Control issues?** → Read [CONTROLS.md](CONTROLS.md)
4. **Ready to run?** → Follow Quick Start above!

---

**Need help? Check the full guides or review generated images for debugging!**

## Links to Full Documentation

- 📘 **[BLUESTACKS_SETUP.md](BLUESTACKS_SETUP.md)** - Step-by-step BlueStacks setup
- � **[MINIMAP_LOCATION.md](MINIMAP_LOCATION.md)** - Visual guide: Where is the minimap?
- �🗺️ **[MINIMAP_GUIDE.md](MINIMAP_GUIDE.md)** - How minimap navigation works
- 🎮 **[CONTROLS.md](CONTROLS.md)** - Complete control scheme guide
- 📖 **[README.md](README.md)** - General project overview

---

**Happy Automating! 🤖✨**
